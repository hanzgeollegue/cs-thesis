"""
Enhanced NLG Generator with Evidence-Based Analysis

This module provides an enhanced natural language generation system that creates
evidence-based candidate analyses with proper confidence hedging, skills hygiene,
and concrete examples.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from .nlg_config import get_domain_config, DomainConfig
from .nlg_templates import TemplateEngine
from .profile_analyzer import ProfileAnalyzer
from .nlg_polisher import GrammarPolisher
from .nlg_metadata import ProvenanceTracker, MetadataEmbedder
from .nlg_summary import BulletSummaryGenerator
from .evidence_collector import EvidenceCollector

logger = logging.getLogger(__name__)


class EnhancedFactExtractor:
    """Enhanced fact extractor with batch-relative and profile insights."""
    
    def __init__(self, config: Optional[DomainConfig] = None):
        """Initialize with domain configuration."""
        self.config = config or get_domain_config()
        self.profile_analyzer = ProfileAnalyzer(config)
        self.evidence_collector = EvidenceCollector()
    
    def extract_facts(self, candidate_data: Dict[str, Any], jd_criteria: Dict[str, Any], 
                     batch_stats: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract comprehensive facts including batch-relative insights.
        
        Args:
            candidate_data: Candidate resume data
            jd_criteria: Job description criteria
            batch_stats: Batch statistics for comparative analysis
            
        Returns:
            Enhanced facts dictionary
        """
        try:
            scores = candidate_data.get('scores', {})
            parsed = candidate_data.get('parsed', {})
            meta = candidate_data.get('meta', {})
            
            # Basic facts (from original FactExtractor)
            facts = {
                'scores': {
                    'final_score': float(scores.get('final_score', 0.0)),
                    'final_score_percentage': round(float(scores.get('final_score', 0.0)) * 100, 1),
                    'coverage': float(scores.get('coverage', 0.0)),
                    'has_match_skills': bool(scores.get('has_match_skills', False)),
                    'has_match_experience': bool(scores.get('has_match_experience', False))
                },
                'skills': self._clean_skills_data(scores, jd_criteria),
                'experience': self._extract_experience_facts(parsed),
                'education': self._extract_education_facts(parsed),
                'projects': self._extract_projects_facts(parsed),
                'metadata': {
                    'candidate_name': self._extract_candidate_name(candidate_data),
                    'source_file': meta.get('source_file', 'Unknown'),
                    'parsing_quality': self._assess_parsing_quality(parsed)
                }
            }
            
            # Enhanced profile analysis
            profile_context = self.profile_analyzer.analyze_profile_context(candidate_data)
            facts.update(profile_context)
            
            # Collect evidence from multiple sources
            evidence_pool = self.evidence_collector.collect_evidence(candidate_data, jd_criteria)
            facts['evidence'] = evidence_pool
            
            # Assess data quality
            data_quality = self.evidence_collector.assess_data_quality(candidate_data, evidence_pool)
            facts['data_quality'] = data_quality
            
            # Batch-relative insights
            if batch_stats:
                batch_insights = self._extract_batch_insights(
                    facts['scores']['final_score_percentage'], batch_stats
                )
                facts['batch_position'] = batch_insights
            
            return facts
            
        except Exception as e:
            logger.error(f"Error extracting enhanced facts: {e}")
            return {'error': str(e)}
    
    def _clean_skills_data(self, scores: Dict[str, Any], jd_criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and deduplicate skills data to ensure disjoint sets."""
        try:
            matched_required = scores.get('matched_required_skills', [])
            matched_nice = scores.get('matched_nice_skills', [])
            missing_required = scores.get('missing_skills', [])
            
            # Convert to sets for deduplication
            matched_required_set = set(matched_required)
            matched_nice_set = set(matched_nice)
            missing_required_set = set(missing_required)
            
            # Check for conflicts and log them
            conflicts = []
            if matched_required_set & matched_nice_set:
                conflicts.append(f"Required and nice-to-have overlap: {matched_required_set & matched_nice_set}")
            if matched_required_set & missing_required_set:
                conflicts.append(f"Required and missing overlap: {matched_required_set & missing_required_set}")
            if matched_nice_set & missing_required_set:
                conflicts.append(f"Nice-to-have and missing overlap: {matched_nice_set & missing_required_set}")
            
            if conflicts:
                logger.warning(f"Skills hygiene conflicts detected: {'; '.join(conflicts)}")
            
            # Apply priority: matched_required > matched_nice > missing
            # Remove overlaps by priority
            matched_nice_clean = matched_nice_set - matched_required_set
            missing_clean = missing_required_set - matched_required_set - matched_nice_clean
            
            # Include verified vs skills-only breakdown
            verified_skills = scores.get('verified_skills', [])
            skills_only_skills = scores.get('skills_only_skills', [])
            
            return {
                'matched_required': sorted(list(matched_required_set)),
                'verified_skills': verified_skills,  # Skills demonstrated in experience (full credit)
                'skills_only_skills': skills_only_skills,  # Skills only listed in skills section (50% credit)
                'matched_nice': sorted(list(matched_nice_clean)),
                'missing_required': sorted(list(missing_clean)),
                'total_required': len(jd_criteria.get('must_have_skills', [])),
                'total_nice': len(jd_criteria.get('nice_to_have_skills', [])),
                'conflicts_detected': conflicts
            }
            
        except Exception as e:
            logger.error(f"Error cleaning skills data: {e}")
            return {
                'matched_required': scores.get('matched_required_skills', []),
                'verified_skills': scores.get('verified_skills', []),
                'skills_only_skills': scores.get('skills_only_skills', []),
                'matched_nice': scores.get('matched_nice_skills', []),
                'missing_required': scores.get('missing_skills', []),
                'total_required': len(jd_criteria.get('must_have_skills', [])),
                'total_nice': len(jd_criteria.get('nice_to_have_skills', [])),
                'conflicts_detected': [f"Error: {str(e)}"]
            }
    
    def _extract_experience_facts(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Extract experience-related facts."""
        experience_items = parsed.get('experience', [])
        return {
            'count': len(experience_items),
            'has_relevant': len(experience_items) > 0,
            'top_roles': [exp.get('title', 'Unknown') for exp in experience_items[:3]]
        }
    
    def _extract_education_facts(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Extract education-related facts."""
        education_items = parsed.get('education', [])
        return {
            'count': len(education_items),
            'degrees': [edu.get('degree', 'Unknown') for edu in education_items]
        }
    
    def _extract_projects_facts(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Extract project-related facts."""
        project_items = parsed.get('projects', [])
        return {
            'count': len(project_items),
            'technologies': list(set([
                tech for project in project_items 
                for tech in project.get('technologies', [])
            ]))
        }
    
    def _extract_candidate_name(self, candidate_data: Dict[str, Any]) -> str:
        """Extract candidate name from various possible locations."""
        # Try different possible locations for candidate name
        parsed = candidate_data.get('parsed', {})
        
        # Check personal_info section
        personal_info = parsed.get('personal_info', {})
        if personal_info.get('name'):
            return personal_info['name']
        
        # Check metadata
        meta = candidate_data.get('meta', {})
        if meta.get('candidate_name'):
            return meta['candidate_name']
        
        # Fallback to source file name
        source_file = meta.get('source_file', 'Unknown')
        if source_file != 'Unknown':
            return source_file.replace('.pdf', '').replace('_', ' ')
        
        return 'Unknown Candidate'
    
    def _assess_parsing_quality(self, parsed: Dict[str, Any]) -> str:
        """Assess the quality of resume parsing."""
        has_experience = bool(parsed.get('experience'))
        has_skills = bool(parsed.get('skills'))
        has_education = bool(parsed.get('education'))
        
        quality_score = sum([has_experience, has_skills, has_education])
        
        if quality_score >= 3:
            return 'high'
        elif quality_score >= 2:
            return 'medium'
        else:
            return 'low'
    
    def _extract_batch_insights(self, score_pct: float, batch_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Extract batch-relative insights."""
        try:
            all_scores = batch_stats.get('all_final_scores', [])
            if not all_scores:
                return {
                    'percentile': 50,
                    'gap_from_top': 0,
                    'above_median': True,
                    'rank_estimate': 1,
                    'batch_size': 1
                }
            
            # Handle N=1 case
            if len(all_scores) == 1:
                return {
                    'percentile': 100,
                    'gap_from_top': 0,
                    'above_median': True,
                    'rank_estimate': 1,
                    'batch_size': 1,
                    'total_candidates': 1
                }
            
            # Calculate percentile using numpy for accuracy
            import numpy as np
            sorted_scores = sorted(all_scores, reverse=True)
            
            # Find rank (1-indexed) by counting how many scores are strictly greater
            # Use tolerance for floating point comparison to avoid precision issues
            # (e.g., 0.37916 vs 0.379 should be considered equal)
            candidate_score = score_pct / 100.0
            TOLERANCE = 0.0005  # 0.05% tolerance
            rank = 1
            for score in sorted_scores:
                if score > candidate_score + TOLERANCE:
                    rank += 1
                else:
                    break  # Since sorted descending, we can stop here
            
            # Calculate percentile
            percentile = ((len(sorted_scores) - rank + 1) / len(sorted_scores)) * 100
            
            # Calculate gap from top
            top_score = max(all_scores) * 100
            gap_from_top = max(0, top_score - score_pct)
            
            # Only include non-leader gaps for statistics
            if gap_from_top < 1e-6:  # Avoid floating point comparison
                gap_from_top = 0.0
            
            # Check if above median
            median_score = np.percentile(all_scores, 50) * 100
            above_median = score_pct > median_score
            
            candidate_count = batch_stats.get('candidate_count', len(all_scores))
            
            return {
                'percentile': percentile,
                'gap_from_top': gap_from_top,
                'above_median': above_median,
                'rank_estimate': rank,
                'batch_size': candidate_count,
                'total_candidates': candidate_count
            }
            
        except Exception as e:
            logger.warning(f"Error extracting batch insights: {e}")
            return {
                'percentile': 50,
                'gap_from_top': 0,
                'above_median': True,
                'rank_estimate': 1,
                'batch_size': 1
            }


class EnhancedCandidateAnalyzer:
    """Enhanced candidate analyzer with modular architecture."""
    
    def __init__(self, config: Optional[DomainConfig] = None):
        """Initialize with domain configuration."""
        self.config = config or get_domain_config()
        self.template_engine = TemplateEngine(config)
        self.fact_extractor = EnhancedFactExtractor(config)
        self.grammar_polisher = GrammarPolisher(config)
        self.bullet_generator = BulletSummaryGenerator(config)
        self.provenance_tracker = ProvenanceTracker(config)
        self.evidence_collector = EvidenceCollector()
    
    def generate_analysis(self, candidate_data: Dict[str, Any], jd_criteria: Dict[str, Any], 
                         batch_stats: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate enhanced analysis with metadata and bullets.
        
        Args:
            candidate_data: Candidate resume data
            jd_criteria: Job description criteria
            batch_stats: Batch statistics for comparative insights
            
        Returns:
            Complete analysis structure with text, bullets, facts, and metadata
        """
        start_time = time.time()
        
        try:
            # Reset provenance tracker
            self.provenance_tracker.reset()
            
            # Extract enhanced facts
            facts = self.fact_extractor.extract_facts(candidate_data, jd_criteria, batch_stats)
            
            # Set context in tracker
            self.provenance_tracker.set_context(
                candidate_id=candidate_data.get('id', 'unknown'),
                job_description_id=jd_criteria.get('id', 'unknown'),
                batch_context=facts.get('batch_position', {}) if batch_stats else None,
                profile_context=facts.get('profile', {})
            )
            
            # Generate analysis text
            analysis_text = self._generate_analysis_text(candidate_data, facts, jd_criteria)
            
            # Generate bullet points
            bullets = self.bullet_generator.generate_bullets(candidate_data, facts, batch_stats)
            
            # Generate detailed score breakdown
            score_breakdown = self._generate_detailed_score_breakdown(candidate_data, facts)
            
            # Generate metadata
            processing_time_ms = int((time.time() - start_time) * 1000)
            self.provenance_tracker.set_processing_time(processing_time_ms)
            metadata = self.provenance_tracker.get_metadata_dict()
            
            return {
                'text': analysis_text,
                'bullets': bullets,
                'facts': facts,
                'metadata': metadata,
                'score_breakdown': score_breakdown
            }
            
        except Exception as e:
            logger.error(f"Error generating enhanced analysis: {e}")
            return {
                'text': f"Analysis generation failed: {str(e)}",
                'bullets': [f"Error: {str(e)}"],
                'facts': {'error': str(e)},
                'metadata': {'error': str(e)}
            }
    
    def _generate_analysis_text(self, candidate_data: Dict[str, Any], 
                               facts: Dict[str, Any], jd_criteria: Dict[str, Any]) -> str:
        """Generate evidence-based analysis text with proper structure."""
        try:
            scores = facts['scores']
            skills = facts['skills']
            evidence_pool = facts.get('evidence', [])
            data_quality = facts.get('data_quality', {})
            batch_position = facts.get('batch_position', {})
            candidate_id = candidate_data.get('id', 'unknown')
            
            # Determine tone based on data health
            needs_caveat = data_quality.get('needs_caveat', False)
            ce_pairs_count = data_quality.get('evidence_counts', {}).get('ce_pairs', 0)
            quality_level = data_quality.get('quality_level', 'medium')
            
            # Get health flags from candidate data
            parsing_ok = candidate_data.get('parsing_ok', True)
            ce_channel_healthy = candidate_data.get('scores', {}).get('ce_channel_healthy', True)
            
            # Tone gating by data health
            if not parsing_ok:
                tone = 'minimal'  # Limited data available; basic assessment only
            elif ce_channel_healthy and ce_pairs_count >= 20 and quality_level == 'high':
                tone = 'confident'  # Strong evidence available
            elif ce_channel_healthy and ce_pairs_count >= 10:
                tone = 'moderate'  # Some evidence available
            else:
                tone = 'cautious'  # Limited evidence or unhealthy channels
            
            # Adjust tone for low evidence - use hedged language
            use_hedged_tone = tone in ['minimal', 'cautious'] or ce_pairs_count == 0
            
            analysis_parts = []
            
            # 1. Caveat (if data quality is low/medium or parsing failed)
            # Only show caveat if parsing actually failed
            if not parsing_ok:
                caveat = self._generate_caveat(data_quality, candidate_id, parsing_ok)
                if caveat:
                    analysis_parts.append(caveat)
                    self.provenance_tracker.track_sentence(
                        'caveat', 0, ['data_quality'], ['data_quality_assessor']
                    )
            
            # 2. Score Context with Batch Position (adaptive detail)
            score_context = self._generate_score_context(scores, batch_position, candidate_id)
            if score_context:
                analysis_parts.append(score_context)
                self.provenance_tracker.track_sentence(
                    'score_context', 1, ['scores.final_score_percentage', 'batch_position'], 
                    ['score_analyzer', 'batch_analyzer']
                )

            # 2b. Signal breakdown (concise, numeric)
            score_breakdown = self._generate_signal_breakdown(candidate_data)
            if score_breakdown:
                analysis_parts.append(score_breakdown)
                self.provenance_tracker.track_sentence(
                    'score_breakdown', 2, ['scores.tfidf_norm', 'scores.semantic_norm', 'scores.ce_norm', 'scores.coverage', 'scores.gate_reason'],
                    ['score_analyzer']
                )
            
            # 3. Strengths with Evidence (2-3 items max)
            strengths = self._generate_strengths_with_evidence(evidence_pool, skills, jd_criteria, candidate_id, use_hedged_tone)
            for i, strength in enumerate(strengths):
                analysis_parts.append(strength)
                self.provenance_tracker.track_sentence(
                    f'strength_{i}', 2+i, ['evidence', 'skills.matched_required'], ['evidence_analyzer']
                )
            
            # 4. Gaps with Context (1-2 items)
            gaps = self._generate_gaps_with_context(skills, jd_criteria, candidate_id, use_hedged_tone)
            for i, gap in enumerate(gaps):
                analysis_parts.append(gap)
                self.provenance_tracker.track_sentence(
                    f'gap_{i}', 5+i, ['skills.missing_required'], ['gap_analyzer']
                )
            
            # 5. Add validation step suggestion when evidence is thin
            if ce_pairs_count == 0 or quality_level == 'low' or not parsing_ok:
                if ce_pairs_count == 0:
                    validation_suggestion = "Consider requesting a code sample or technical assessment to validate skills."
                elif not parsing_ok:
                    validation_suggestion = "Resume parsing issues detected. Consider requesting a formatted resume or technical assessment."
                else:
                    validation_suggestion = "Limited evidence available. Consider requesting a code sample or technical assessment to validate skills."
                
                analysis_parts.append(validation_suggestion)
                self.provenance_tracker.track_sentence(
                    'validation_suggestion', 2+len(strengths)+len(gaps), ['data_quality'], ['validation_advisor']
                )
            
            # 5. One Concrete Example
            concrete_example = self._generate_concrete_example(evidence_pool, jd_criteria, candidate_id)
            if concrete_example:
                analysis_parts.append(concrete_example)
                self.provenance_tracker.track_sentence(
                    'concrete_example', 7, ['evidence'], ['example_selector']
                )

            # 6. Confidence & Recommendation
            conf_line = self._generate_confidence_line(data_quality)
            if conf_line:
                analysis_parts.append(conf_line)
                self.provenance_tracker.track_sentence(
                    'confidence_line', 8, ['data_quality.average_confidence', 'data_quality.evidence_counts.ce_pairs'], ['confidence_estimator']
                )

            recommendation = self._generate_recommendation(candidate_data, jd_criteria)
            if recommendation:
                analysis_parts.append(recommendation)
                self.provenance_tracker.track_sentence(
                    'recommendation', 9, ['scores.final_score', 'scores.coverage'], ['policy_recommender']
                )

            # 7. Interview probes for gaps/uncertainties
            probes = self._generate_interview_probes(skills, evidence_pool)
            if probes:
                analysis_parts.append(probes)
                self.provenance_tracker.track_sentence(
                    'interview_probes', 10, ['skills.missing_required'], ['gap_analyzer']
                )
            
            # 6. Batch-relative line (if batch stats available)
            if batch_position and tone != 'minimal':
                batch_relative = self._generate_batch_relative_line(batch_position, candidate_id)
                if batch_relative:
                    analysis_parts.append(batch_relative)
                    self.provenance_tracker.track_sentence(
                        'batch_relative', 8, ['batch_position'], ['batch_analyzer']
                    )
            
            # 7. Specialization gating (only if confidence >= 0.5 and signals >= 2)
            if tone in ['confident', 'moderate']:
                specialization = self._generate_specialization_insight(facts, candidate_id)
                if specialization:
                    analysis_parts.append(specialization)
                    self.provenance_tracker.track_sentence(
                        'specialization', 9, ['profile.specializations'], ['specialization_analyzer']
                    )
            
            # Join parts and clean up
            draft_analysis = '. '.join(analysis_parts)
            if not draft_analysis.endswith('.'):
                draft_analysis += '.'
            
            # Apply grammar polish
            polished_analysis, polish_log = self.grammar_polisher.apply_polish(draft_analysis)
            
            # Track polish operations
            for operation in polish_log:
                self.provenance_tracker.track_polish_operation(operation)
            
            return polished_analysis
            
        except Exception as e:
            logger.error(f"Error generating evidence-based analysis text: {e}")
            return f"Analysis unavailable due to processing error: {str(e)}"
    
    def _generate_batch_relative_line(self, batch_position: Dict[str, Any], candidate_id: str) -> Optional[str]:
        """Generate batch-relative line with rank and percentile."""
        try:
            rank = batch_position.get('rank_estimate', 1)  # Use rank_estimate, default to 1
            percentile = batch_position.get('percentile', 0)
            gap_to_top = batch_position.get('gap_to_top', 0)
            total_candidates = batch_position.get('total_candidates', 1)
            
            # Never show "Only candidate" for batch_size > 1
            if total_candidates == 1:
                return "Single candidate evaluation."
            elif rank <= 3:
                return f"Ranks #{rank} of {total_candidates} ({percentile:.0f}th percentile)."
            else:
                return f"Ranks #{rank} of {total_candidates} ({percentile:.0f}th percentile, {gap_to_top:.1f} points behind top)."
                
        except Exception as e:
            logger.warning(f"Error generating batch-relative line: {e}")
            return None
    
    def _generate_specialization_insight(self, facts: Dict[str, Any], candidate_id: str) -> Optional[str]:
        """Generate specialization insight with confidence gating."""
        try:
            profile = facts.get('profile', {})
            specializations = profile.get('specializations', {})
            confidence = profile.get('confidence', 0.0)
            signals = profile.get('signals', 0)
            
            # Specialization gating: only mention if confidence >= 0.5 and signals >= 2
            if confidence < 0.5 or signals < 2:
                return None
            
            primary_spec = specializations.get('primary', '')
            if primary_spec and primary_spec != 'general':
                return f"Shows specialization in {primary_spec} with {signals} supporting signals."
            
            return None
            
        except Exception as e:
            logger.warning(f"Error generating specialization insight: {e}")
            return None
    
    # Helper methods for evidence-based analysis generation
    
    def _generate_caveat(self, data_quality: Dict[str, Any], candidate_id: str, parsing_ok: bool = True) -> str:
        """Generate caveat text based on data quality assessment."""
        try:
            from .nlg_templates import VariantSelector
            
            selector = VariantSelector(candidate_id)
            variant_index = selector.select_variant(3, "caveat")
            
            # Determine caveat type based on data quality issues
            ce_data_availability = data_quality.get('ce_data_availability', {})
            has_ce_data = ce_data_availability.get('has_ce_data', False)
            
            # Check parsing_ok first - this is the most important flag
            if not parsing_ok:
                caveat_type = 'low_parsing'
            elif not data_quality.get('parsing_completeness', {}).get('has_experience', True):
                caveat_type = 'low_parsing'
            elif not has_ce_data:
                caveat_type = 'low_ce_evidence'
            elif data_quality.get('evidence_counts', {}).get('ce_pairs', 0) < 1:
                caveat_type = 'low_ce_evidence'
            else:
                caveat_type = 'low_confidence'
            
            template = self.template_engine.template_lib.get_caveat_template(caveat_type, variant_index)
            return template
            
        except Exception as e:
            logger.warning(f"Error generating caveat: {e}")
            return "Assessment based on available data."

    def _generate_score_context(self, scores: Dict[str, Any], batch_position: Dict[str, Any], candidate_id: str) -> str:
        """Generate score context with adaptive batch detail."""
        try:
            score_pct = scores.get('final_score_percentage', 0)
            percentile = batch_position.get('percentile', 50)
            rank = batch_position.get('rank_estimate', 1)
            batch_size = batch_position.get('batch_size', 1)
            gap_from_top = batch_position.get('gap_from_top', 0)
            
            # Special handling for single-candidate batches
            if batch_size == 1:
                return f"Only candidate in batch ({score_pct:.1f}% match)"
            
            # Determine if this is a shortlist candidate (top 25% or rank <= 3)
            is_shortlist = rank <= 3 or percentile >= 75
            
            if is_shortlist and batch_size > 1:
                if gap_from_top < 1e-6:  # Tied for leader
                    return f"Ranked #{rank} of {batch_size} candidates ({score_pct:.1f}% match, tied for leader)"
                else:
                    return f"Ranked #{rank} of {batch_size} candidates ({score_pct:.1f}% match, {gap_from_top:.1f} points behind leader)"
            else:
                # Simple tier description
                if percentile >= 75:
                    tier_label = "top 25%"
                elif percentile >= 50:
                    tier_label = "above median"
                else:
                    tier_label = "below median"
                return f"{tier_label} match at {score_pct:.1f}%"
                
        except Exception as e:
            logger.warning(f"Error generating score context: {e}")
            return f"{scores.get('final_score_percentage', 0)}% match"

    def _generate_signal_breakdown(self, candidate_data: Dict[str, Any]) -> Optional[str]:
        """Summarize per-signal contributions and gates with detailed explanations."""
        try:
            sc = candidate_data.get('scores', {})
            tf = float(sc.get('tfidf_norm', 0.0))
            sb = float(sc.get('semantic_norm', 0.0))
            ce = float(sc.get('ce_norm', 0.0))
            cov = float(sc.get('coverage', sc.get('coverage', 0.0))) if isinstance(sc.get('coverage', 0.0), (int, float)) else 0.0
            gate_reason = sc.get('gate_reason', '')
            
            # Calculate base composite
            base_composite = 0.5 * sb + 0.3 * ce + 0.2 * tf
            
            # Build detailed explanation
            parts = []
            
            # Signal contributions with explanations
            tf_contrib = 0.2 * tf
            sb_contrib = 0.5 * sb
            ce_contrib = 0.3 * ce
            
            parts.append(f"Score breakdown: Cross-Encoder contributes {ce_contrib:.3f} (30% weight × {ce:.2f}), SBERT contributes {sb_contrib:.3f} (50% weight × {sb:.2f}), TF-IDF contributes {tf_contrib:.3f} (20% weight × {tf:.2f})")
            parts.append(f"Base composite: {base_composite:.3f}")
            parts.append(f"Skills coverage: {cov:.1%}")
            
            if gate_reason:
                parts.append(f"Penalties applied: {gate_reason}")
            
            return '. '.join(parts)
        except Exception as e:
            logger.warning(f"Error generating signal breakdown: {e}")
            return None

    def _generate_detailed_score_breakdown(self, candidate_data: Dict[str, Any], 
                                          facts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive score breakdown with structured data for template rendering.
        
        Returns a dictionary with:
        - signal_scores: raw and normalized scores for each signal
        - base_composite: calculated base composite score
        - coverage_analysis: skills coverage details
        - gates_applied: any penalties or gates
        - final_score: how gates affected the base composite
        - explanation: natural language explanation
        """
        try:
            sc = candidate_data.get('scores', {})
            skills = facts.get('skills', {})
            
            # Extract normalized scores
            tf_norm = float(sc.get('tfidf_norm', 0.0))
            sb_norm = float(sc.get('semantic_norm', 0.0))
            ce_norm = float(sc.get('ce_norm', 0.0))
            
            # Extract raw scores if available
            tf_raw = float(sc.get('combined_tfidf', sc.get('section_tfidf', 0.0)))
            sb_raw = float(sc.get('sbert_score', 0.0))
            ce_raw = float(sc.get('ce_score', 0.0))
            
            # Calculate contributions
            tf_contrib = 0.2 * tf_norm
            sb_contrib = 0.5 * sb_norm
            ce_contrib = 0.3 * ce_norm
            base_composite = tf_contrib + sb_contrib + ce_contrib
            
            # Coverage analysis
            coverage = float(sc.get('coverage', 0.0))
            matched_required = skills.get('matched_required', [])
            missing_required = skills.get('missing_required', [])
            total_required = skills.get('total_required', 0)
            
            # Gate information
            gate_reason = sc.get('gate_reason', '')
            final_score = float(sc.get('final_score', base_composite))
            
            # Determine if gates were applied
            gates_applied = []
            if gate_reason:
                if 'cov' in gate_reason.lower() or 'coverage' in gate_reason.lower():
                    threshold = float(sc.get('gate_threshold', 0.0))
                    gates_applied.append({
                        'type': 'coverage',
                        'reason': f"Skills coverage ({coverage:.1%}) below threshold ({threshold:.1%})",
                        'penalty': f"Score reduced by {(1 - coverage**2) * 100:.1f}%"
                    })
                if 'exp' in gate_reason.lower() or 'year' in gate_reason.lower():
                    gates_applied.append({
                        'type': 'experience',
                        'reason': "Minimum years of experience not met",
                        'penalty': "Score reduced by 50%"
                    })
            
            # Build natural language explanation
            explanation_parts = []
            
            # Signal quality descriptions
            tf_quality = "strong" if tf_norm >= 0.7 else "moderate" if tf_norm >= 0.4 else "weak"
            sb_quality = "strong" if sb_norm >= 0.7 else "moderate" if sb_norm >= 0.4 else "weak"
            ce_quality = "strong" if ce_norm >= 0.7 else "moderate" if ce_norm >= 0.4 else "weak"
            
            explanation_parts.append(
                f"The score combines three signals: Cross-Encoder ({ce_quality}, {ce_norm:.2f}), "
                f"SBERT semantic similarity ({sb_quality}, {sb_norm:.2f}), and TF-IDF keyword matching ({tf_quality}, {tf_norm:.2f})."
            )
            
            explanation_parts.append(
                f"Base composite calculation: (0.5 × {sb_norm:.2f}) + (0.3 × {ce_norm:.2f}) + (0.2 × {tf_norm:.2f}) = {base_composite:.3f}"
            )
            
            if coverage > 0:
                explanation_parts.append(
                    f"Skills coverage: {len(matched_required)}/{total_required} required skills matched ({coverage:.1%})"
                )
                if missing_required:
                    explanation_parts.append(
                        f"Missing skills: {', '.join(missing_required[:5])}{'...' if len(missing_required) > 5 else ''}"
                    )
            
            if gates_applied:
                for gate in gates_applied:
                    explanation_parts.append(f"Gate applied: {gate['reason']}. {gate['penalty']}.")
            
            if abs(final_score - base_composite) > 0.001:
                explanation_parts.append(
                    f"After applying gates, final score: {final_score:.3f} ({final_score * 100:.1f}%)"
                )
            else:
                explanation_parts.append(
                    f"Final score: {final_score:.3f} ({final_score * 100:.1f}%)"
                )
            
            return {
                'signal_scores': {
                    'tfidf': {
                        'raw': tf_raw,
                        'normalized': tf_norm,
                        'contribution': tf_contrib,
                        'weight': 0.2,
                        'description': 'Keyword matching between resume and job description'
                    },
                    'sbert': {
                        'raw': sb_raw,
                        'normalized': sb_norm,
                        'contribution': sb_contrib,
                        'weight': 0.5,
                        'description': 'Semantic similarity using sentence embeddings'
                    },
                    'cross_encoder': {
                        'raw': ce_raw,
                        'normalized': ce_norm,
                        'contribution': ce_contrib,
                        'weight': 0.3,
                        'description': 'Contextual relevance using cross-encoder model'
                    }
                },
                'base_composite': base_composite,
                'coverage_analysis': {
                    'coverage': coverage,
                    'matched_count': len(matched_required),
                    'total_required': total_required,
                    'missing_skills': missing_required
                },
                'gates_applied': gates_applied,
                'final_score': final_score,
                'explanation': '. '.join(explanation_parts)
            }
            
        except Exception as e:
            logger.error(f"Error generating detailed score breakdown: {e}")
            return {
                'error': str(e),
                'signal_scores': {},
                'base_composite': 0.0,
                'coverage_analysis': {},
                'gates_applied': [],
                'final_score': 0.0,
                'explanation': 'Score breakdown unavailable due to processing error.'
            }

    def _generate_confidence_line(self, data_quality: Dict[str, Any]) -> Optional[str]:
        """Produce a short confidence line based on evidence stats."""
        try:
            avg = float(data_quality.get('average_confidence', 0.0))
            ce_pairs = int(data_quality.get('evidence_counts', {}).get('ce_pairs', 0))
            # Simple banding
            if avg >= 0.75 and ce_pairs >= 10:
                band = 'High'
            elif avg >= 0.60 and ce_pairs >= 5:
                band = 'Medium'
            elif avg > 0 or ce_pairs > 0:
                band = 'Low'
            else:
                band = ''
            return f"Confidence: {band} (avg evidence {avg:.2f}; CE pairs {ce_pairs})" if band else None
        except Exception:
            return None

    def _generate_recommendation(self, candidate_data: Dict[str, Any], jd_criteria: Dict[str, Any]) -> Optional[str]:
        """Generate a recommendation line from final score and coverage."""
        try:
            sc = candidate_data.get('scores', {})
            final_score = float(sc.get('final_score', 0.0))
            coverage = float(sc.get('coverage', 0.0))
            # Thresholds can be tuned per domain; use simple defaults
            if final_score >= 0.80 and coverage >= 0.70:
                rec = 'Recommend interview'
            elif final_score >= 0.55 and coverage >= 0.50:
                rec = 'Recommend technical screen'
            else:
                rec = 'Hold for now'
            return f"Recommendation: {rec}"
        except Exception:
            return None

    def _generate_interview_probes(self, skills: Dict[str, Any], evidence_pool: List[Any]) -> Optional[str]:
        """Suggest interview probes based on missing/weak evidence skills."""
        try:
            missing = skills.get('missing_required', []) or []
            if not missing:
                return None
            # Take top 2-3 missing for concise prompts
            probes = ', '.join(missing[:3])
            return f"Interview probes: validate exposure to {probes}"
        except Exception:
            return None
    
    def _generate_strengths_with_evidence(self, evidence_pool: List[Any], skills: Dict[str, Any], 
                                        jd_criteria: Dict[str, Any], candidate_id: str, use_hedged_tone: bool = False) -> List[str]:
        """Generate 2-3 strength statements with evidence."""
        try:
            from .nlg_templates import VariantSelector
            
            strengths = []
            matched_required = skills.get('matched_required', [])
            
            if not matched_required or not evidence_pool:
                return strengths
            
            selector = VariantSelector(candidate_id)
            
            # Get top 2-3 matched skills with best evidence
            for i, skill in enumerate(matched_required[:3]):
                best_evidence = self.evidence_collector.get_best_evidence_for_skill(skill, evidence_pool)
                if best_evidence:
                    variant_index = selector.select_variant(4, f"strength_{i}")
                    
                    # Determine if we should include a micro-quote
                    include_quote = best_evidence.confidence > 0.85 and best_evidence.micro_quote
                    
                    if include_quote:
                        template = self.template_engine.template_lib.get_evidence_template('strength_with_quote', variant_index)
                        strength_text = template.format(
                            skill=skill,
                            context=best_evidence.text,
                            quote=best_evidence.micro_quote
                        )
                    else:
                        template = self.template_engine.template_lib.get_evidence_template('strength_with_context', variant_index)
                        strength_text = template.format(
                            skill=skill,
                            context=best_evidence.text
                        )
                    
                    # Apply hedged language if needed
                    if use_hedged_tone:
                        # Replace strong adjectives with hedged ones
                        strength_text = strength_text.replace('Strong ', 'Appears to have ')
                        strength_text = strength_text.replace('Demonstrates ', 'Shows potential for ')
                        strength_text = strength_text.replace('Shows ', 'Indicates ')
                        strength_text = strength_text.replace('Exhibits ', 'Suggests ')
                    
                    strengths.append(strength_text)
            
            return strengths
            
        except Exception as e:
            logger.warning(f"Error generating strengths: {e}")
            return []
    
    def _generate_gaps_with_context(self, skills: Dict[str, Any], jd_criteria: Dict[str, Any], 
                                   candidate_id: str, use_hedged_tone: bool = False) -> List[str]:
        """Generate 1-2 gap statements with context."""
        try:
            from .nlg_templates import VariantSelector
            
            gaps = []
            missing_required = skills.get('missing_required', [])
            
            if not missing_required:
                return gaps
            
            selector = VariantSelector(candidate_id)
            
            # Get top 1-2 critical missing skills
            for i, skill in enumerate(missing_required[:2]):
                variant_index = selector.select_variant(4, f"gap_{i}")
                template = self.template_engine.template_lib.get_gap_context_template('critical_missing', variant_index)
                
                # Determine role context from JD criteria
                role_context = jd_criteria.get('position_title', 'this role')
                if not role_context:
                    role_context = 'this position'
                
                gap_text = template.format(
                    skill=skill,
                    role_context=role_context
                )
                
                # Apply hedged language if needed
                if use_hedged_tone:
                    # Replace definitive statements with hedged ones
                    gap_text = gap_text.replace('Missing ', 'May be missing ')
                    gap_text = gap_text.replace('No ', 'Limited ')
                    gap_text = gap_text.replace('Lacks ', 'Appears to lack ')
                    gap_text = gap_text.replace('(required for', '(likely needed for')
                    gap_text = gap_text.replace('(needed for', '(potentially needed for')
                    gap_text = gap_text.replace('(critical for', '(important for')
                    gap_text = gap_text.replace('(essential for', '(valuable for')
                
                gaps.append(gap_text)
            
            return gaps
            
        except Exception as e:
            logger.warning(f"Error generating gaps: {e}")
            return []
    
    def _generate_concrete_example(self, evidence_pool: List[Any], jd_criteria: Dict[str, Any], 
                                 candidate_id: str) -> str:
        """Generate one concrete example with evidence."""
        try:
            from .nlg_templates import VariantSelector
            
            best_example = self.evidence_collector.select_best_concrete_example(evidence_pool, jd_criteria)
            if not best_example:
                return ""
            
            selector = VariantSelector(candidate_id)
            variant_index = selector.select_variant(4, "concrete_example")
            
            # Determine if we should include a micro-quote
            include_quote = best_example.confidence > 0.85 and best_example.micro_quote
            
            if include_quote:
                template = self.template_engine.template_lib.get_concrete_example_template('with_quote', variant_index)
                example_text = template.format(
                    skill=best_example.skill,
                    context=best_example.text,
                    company=best_example.company or 'previous role',
                    quote=best_example.micro_quote
                )
            else:
                template = self.template_engine.template_lib.get_concrete_example_template('with_context', variant_index)
                example_text = template.format(
                    skill=best_example.skill,
                    context=best_example.text,
                    company=best_example.company or 'previous role'
                )
            
            return example_text
            
        except Exception as e:
            logger.warning(f"Error generating concrete example: {e}")
            return ""


# Enhanced convenience functions
def generate_candidate_analysis_enhanced(candidate_data: Dict[str, Any], 
                                       jd_criteria: Dict[str, Any], 
                                       batch_stats: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Generate enhanced analysis for a single candidate."""
    analyzer = EnhancedCandidateAnalyzer()
    return analyzer.generate_analysis(candidate_data, jd_criteria, batch_stats)


def generate_candidate_facts_enhanced(candidate_data: Dict[str, Any], 
                                    jd_criteria: Dict[str, Any], 
                                    batch_stats: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Extract enhanced facts for a single candidate."""
    extractor = EnhancedFactExtractor()
    return extractor.extract_facts(candidate_data, jd_criteria, batch_stats)


# Backward compatibility wrapper
def generate_candidate_analysis(candidate_data: Dict[str, Any], 
                              jd_criteria: Dict[str, Any], 
                              batch_stats: Optional[Dict[str, Any]] = None) -> str:
    """
    Backward compatible function that returns just the text.
    Falls back to enhanced version if batch_stats provided.
    """
    if batch_stats is not None:
        # Use enhanced version
        result = generate_candidate_analysis_enhanced(candidate_data, jd_criteria, batch_stats)
        return result.get('text', 'Analysis unavailable')
    else:
        # Use original version for backward compatibility
        from .nlg_generator import CandidateAnalyzer
        analyzer = CandidateAnalyzer()
        return analyzer.generate_analysis(candidate_data, jd_criteria)


def generate_candidate_facts(candidate_data: Dict[str, Any], 
                           jd_criteria: Dict[str, Any], 
                           batch_stats: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Backward compatible function for fact extraction.
    Falls back to enhanced version if batch_stats provided.
    """
    if batch_stats is not None:
        # Use enhanced version
        return generate_candidate_facts_enhanced(candidate_data, jd_criteria, batch_stats)
    else:
        # Use original version for backward compatibility
        from .nlg_generator import FactExtractor
        extractor = FactExtractor()
        return extractor.extract_facts(candidate_data, jd_criteria)
