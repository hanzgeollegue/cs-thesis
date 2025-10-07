"""
Strict, deterministic ranking engine for resumes.
Processes numeric features and metadata to generate standardized rankings.
"""

import json
import math
import statistics
from typing import Dict, List, Any, Optional


class DeterministicRanker:
    """
    Strict, deterministic ranking engine that processes only numeric features
    and limited metadata to generate standardized rankings.
    """
    
    def rank_candidates(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the input payload and return ranked candidates according to the strict schema.
        
        Args:
            payload: Input data containing job_profile, weights, penalties, hard_rules, 
                    batch_stats, and candidates
        
        Returns:
            Dict containing job_title, ranked_candidates, and batch_notes
        """
        # Extract input data
        job_profile = payload["job_profile"]
        weights = payload["weights"]
        penalties = payload["penalties"]
        hard_rules = payload["hard_rules"]
        batch_stats = payload["batch_stats"]
        candidates = payload["candidates"]
        
        # Step 1: Normalize raw scores to [0,1] using batch min/max
        normalized_candidates = self._normalize_scores(candidates, batch_stats)
        
        # Step 2: Calculate composite scores
        scored_candidates = self._calculate_composite_scores(
            normalized_candidates, weights, penalties, hard_rules
        )
        
        # Step 3: Sort candidates with deterministic tie-breaking
        ranked_candidates = self._sort_candidates_deterministic(scored_candidates, candidates)
        
        # Step 4: Scale to 0-100 and round to 2 decimals
        final_candidates = self._scale_and_round_scores(ranked_candidates)
        
        # Step 5: Generate rationales
        final_candidates = self._generate_rationales(final_candidates)
        
        # Assemble output
        return {
            "job_title": job_profile["title"],
            "ranked_candidates": final_candidates,
            "batch_notes": {
                "normalization_used": {
                    "section_tfidf_min": batch_stats["section_tfidf_min"],
                    "section_tfidf_max": batch_stats["section_tfidf_max"],
                    "skill_tfidf_min": batch_stats["skill_tfidf_min"],
                    "skill_tfidf_max": batch_stats["skill_tfidf_max"],
                    "sbert_min": batch_stats["sbert_min"],
                    "sbert_max": batch_stats["sbert_max"]
                },
                "weights_used": weights,
                "hard_rules": hard_rules
            }
        }
    
    def _normalize_scores(self, candidates: List[Dict], batch_stats: Dict) -> List[Dict]:
        """Normalize raw scores to [0,1] using batch min/max."""
        normalized_candidates = []
        
        for candidate in candidates:
            raw_scores = candidate["raw_scores"]
            
            # Normalize section_tfidf_overall
            section_min = batch_stats["section_tfidf_min"]
            section_max = batch_stats["section_tfidf_max"]
            if section_max == section_min:
                section_norm = 0.5
            else:
                section_norm = (raw_scores["section_tfidf_overall"] - section_min) / (section_max - section_min)
                section_norm = max(0.0, min(1.0, section_norm))
            
            # Normalize skill_tfidf_overall
            skill_min = batch_stats["skill_tfidf_min"]
            skill_max = batch_stats["skill_tfidf_max"]
            if skill_max == skill_min:
                skill_norm = 0.5
            else:
                skill_norm = (raw_scores["skill_tfidf_overall"] - skill_min) / (skill_max - skill_min)
                skill_norm = max(0.0, min(1.0, skill_norm))
            
            # Normalize sbert_overall
            sbert_min = batch_stats["sbert_min"]
            sbert_max = batch_stats["sbert_max"]
            if sbert_max == sbert_min:
                sbert_norm = 0.5
            else:
                sbert_norm = (raw_scores["sbert_overall"] - sbert_min) / (sbert_max - sbert_min)
                sbert_norm = max(0.0, min(1.0, sbert_norm))
            
            normalized_candidate = candidate.copy()
            normalized_candidate["normalized_scores"] = {
                "section_tfidf": section_norm,
                "skill_tfidf": skill_norm,
                "sbert": sbert_norm
            }
            
            normalized_candidates.append(normalized_candidate)
        
        return normalized_candidates
    
    def _calculate_composite_scores(self, candidates: List[Dict], weights: Dict, 
                                   penalties: Dict, hard_rules: Dict) -> List[Dict]:
        """Calculate composite scores with bonuses and penalties."""
        scored_candidates = []
        
        for candidate in candidates:
            norm_scores = candidate["normalized_scores"]
            meta = candidate["meta"]
            
            S = norm_scores["section_tfidf"]
            K = norm_scores["skill_tfidf"]
            B = norm_scores["sbert"]
            
            # Consistency bonus
            scores_list = [S, K, B]
            if len(set(scores_list)) == 1:  # All identical
                consistency = 1.0
            else:
                consistency = 1.0 - statistics.stdev(scores_list)
            consistency = max(0.0, min(1.0, consistency))
            bonus = consistency * weights["consistency_bonus"]
            
            # Base composite score
            base = (weights["section_tfidf"] * S) + (weights["skill_tfidf"] * K) + (weights["sbert"] * B) + bonus
            
            # Apply penalties
            missing_count = meta["missing_must_have_count"]
            low_evidence = meta["low_evidence"]
            
            penalty = (missing_count * penalties["missing_must_have"]) + (penalties["low_evidence"] if low_evidence else 0)
            composite = base - penalty
            composite = max(0.0, min(1.0, composite))
            
            # Hard rule: disqualification
            disqualified = False
            if hard_rules["disqualify_if_missing_must_have"] and missing_count > 0:
                disqualified = True
                composite = 0.0
            
            scored_candidate = candidate.copy()
            scored_candidate["composite_score"] = composite
            scored_candidate["consistency_bonus"] = bonus
            scored_candidate["penalty_points"] = penalty
            scored_candidate["disqualified"] = disqualified
            
            scored_candidates.append(scored_candidate)
        
        return scored_candidates
    
    def _sort_candidates_deterministic(self, candidates: List[Dict], original_candidates: List[Dict]) -> List[Dict]:
        """Sort candidates with deterministic tie-breaking."""
        # Create mapping for original raw scores
        original_map = {c["candidate_id"]: c for c in original_candidates}
        
        def sort_key(candidate):
            cid = candidate["candidate_id"]
            original = original_map[cid]
            
            # Primary: composite score (descending)
            composite = candidate["composite_score"]
            
            # Tie-breaker 1: Higher B (SBERT normalized, descending)
            sbert_norm = candidate["normalized_scores"]["sbert"]
            
            # Tie-breaker 2: Higher experience section TF-IDF (raw, descending)
            experience_raw = original["raw_scores"]["section_tfidf_by_section"].get("experience", 0)
            
            # Tie-breaker 3: Higher skill TF-IDF overall (raw, descending)
            skill_raw = original["raw_scores"]["skill_tfidf_overall"]
            
            # Tie-breaker 4: Lower missing must-have count (ascending)
            missing_count = original["meta"]["missing_must_have_count"]
            
            # Tie-breaker 5: Alphabetical by candidate_id (ascending)
            candidate_id = cid
            
            return (-composite, -sbert_norm, -experience_raw, -skill_raw, missing_count, candidate_id)
        
        return sorted(candidates, key=sort_key)
    
    def _scale_and_round_scores(self, candidates: List[Dict]) -> List[Dict]:
        """Scale scores to 0-100 and round to 2 decimals."""
        scaled_candidates = []
        
        for candidate in candidates:
            scaled_candidate = candidate.copy()
            
            # Scale composite score
            scaled_candidate["composite_score"] = round(candidate["composite_score"] * 100, 2)
            
            # Scale normalized scores
            norm_scores = candidate["normalized_scores"]
            scaled_candidate["normalized_scores"] = {
                "section_tfidf": round(norm_scores["section_tfidf"] * 100, 2),
                "skill_tfidf": round(norm_scores["skill_tfidf"] * 100, 2),
                "sbert": round(norm_scores["sbert"] * 100, 2)
            }
            
            # Scale penalty points
            scaled_candidate["penalty_points"] = round(candidate["penalty_points"] * 100, 2)
            
            scaled_candidates.append(scaled_candidate)
        
        return scaled_candidates
    
    def _generate_rationales(self, candidates: List[Dict]) -> List[Dict]:
        """Generate concise rationales citing only numeric data."""
        final_candidates = []
        
        for candidate in candidates:
            # Extract numeric values for rationale
            S = candidate["normalized_scores"]["section_tfidf"] / 100
            K = candidate["normalized_scores"]["skill_tfidf"] / 100
            B = candidate["normalized_scores"]["sbert"] / 100
            missing = candidate.get("meta", {}).get("missing_must_have_count", 0)
            
            # Generate rationale (≤ 60 words, cite numbers only)
            if candidate["disqualified"]:
                rationale = f"Disqualified: {missing} missing must-have skills. S={S:.2f}, K={K:.2f}, B={B:.2f}."
            else:
                rationale = f"S={S:.2f}, K={K:.2f}, B={B:.2f}, {missing} missing must-have. Composite: {candidate['composite_score']:.2f}/100."
            
            # Ensure rationale is ≤ 60 words
            words = rationale.split()
            if len(words) > 60:
                rationale = " ".join(words[:60]) + "..."
            
            # Build final candidate object according to schema
            final_candidate = {
                "candidate_id": candidate["candidate_id"],
                "composite_score": candidate["composite_score"],
                "normalized": {
                    "section_tfidf": candidate["normalized_scores"]["section_tfidf"],
                    "skill_tfidf": candidate["normalized_scores"]["skill_tfidf"],
                    "sbert": candidate["normalized_scores"]["sbert"]
                },
                "penalties_applied": {
                    "missing_must_have_count": candidate.get("meta", {}).get("missing_must_have_count", 0),
                    "low_evidence": candidate.get("meta", {}).get("low_evidence", False),
                    "penalty_points": candidate["penalty_points"]
                },
                "disqualified": candidate["disqualified"],
                "rationale": rationale
            }
            
            final_candidates.append(final_candidate)
        
        return final_candidates


def process_ranking_payload(payload_json: str) -> str:
    """
    Process a ranking payload and return JSON results.
    
    Args:
        payload_json: JSON string containing the input payload
        
    Returns:
        JSON string containing ranked results
    """
    try:
        payload = json.loads(payload_json)
        ranker = DeterministicRanker()
        result = ranker.rank_candidates(payload)
        return json.dumps(result, separators=(',', ':'))
    except Exception as e:
        # Return error in schema-compliant format
        error_result = {
            "job_title": "Error",
            "ranked_candidates": [],
            "batch_notes": {
                "normalization_used": {
                    "section_tfidf_min": 0.0,
                    "section_tfidf_max": 0.0,
                    "skill_tfidf_min": 0.0,
                    "skill_tfidf_max": 0.0,
                    "sbert_min": 0.0,
                    "sbert_max": 0.0
                },
                "weights_used": {
                    "section_tfidf": 0.0,
                    "skill_tfidf": 0.0,
                    "sbert": 0.0,
                    "consistency_bonus": 0.0
                },
                "hard_rules": {
                    "disqualify_if_missing_must_have": False
                }
            }
        }
        return json.dumps(error_result, separators=(',', ':'))
