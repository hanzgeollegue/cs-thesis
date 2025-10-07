import json
import logging
from typing import List, Dict, Any, Optional
import requests
from dataclasses import dataclass
import time
import re
import os
import pickle
import datetime
import threading
from contextlib import contextmanager
from .config import get_llm_config

logger = logging.getLogger(__name__)

# --- Timing configuration ---
TIMING_ENABLED = os.getenv("TIMING", "0") in {"1", "true", "True"}

@contextmanager
def time_phase(name: str, bucket: Optional[Dict[str, float]] = None):
    """Context manager for timing phases."""
    if not TIMING_ENABLED:
        yield
        return
    
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info(f"[TIMING] {name}: {elapsed:.3f}s")
        if bucket is not None:
            bucket[name] = elapsed

@dataclass
class RankingResult:
    """Data class for ranking results."""
    candidate_id: str
    rank: int
    score: float
    reasoning: str
    strengths: List[str]
    gaps: List[str]
    confidence: float

@dataclass
class CERerankerResult:
    """Data class for Cross-Encoder reranker results."""
    candidate_id: str
    ce_score: float
    section_tfidf: float
    skill_tfidf: float
    sbert_score: float
    final_score: float

class CEReranker:
    """Cross-Encoder reranker with feedback loop for continuous improvement."""
    
    def __init__(self, model_name: str = 'BAAI/bge-reranker-v2-m3'):
        self.model_name = model_name
        self.model = None
        self.model_available = False
        self.training_data = []
        self.model_version = 0
        self.model_dir = "models"
        self.lock = threading.Lock()
        
        # Create model directory
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Load latest model
        self._load_latest_model()
    
    def _load_latest_model(self):
        """Load the latest trained model."""
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name)
            self.model_available = True
            logger.info(f"Cross-Encoder model loaded: {self.model_name}")
        except ImportError:
            logger.warning("CrossEncoder not available. Install with: pip install sentence-transformers")
            self.model_available = False
        except Exception as e:
            logger.error(f"Failed to load Cross-Encoder model: {e}")
            self.model_available = False
    
    def rerank_candidates(self, job_description, candidates: List[Dict[str, Any]], 
                         section_tfidf_scores: List[float], skill_tfidf_scores: List[float], 
                         sbert_scores: List[float]) -> List[CERerankerResult]:
        """Rerank candidates using Cross-Encoder and multiple features."""
        try:
            if not self.model_available:
                logger.warning("Cross-Encoder not available, returning original order")
                return self._create_fallback_results(candidates, section_tfidf_scores, skill_tfidf_scores, sbert_scores)
            
            # Handle both raw job_description string and normalized jd_sections dict
            from .text_processor import normalize_job_description, preprocess_text_for_dense_models, scrub_pii_and_boilerplate
            
            if isinstance(job_description, dict):
                # Use normalized jd_sections (same as TF-IDF/SBERT)
                jd_text = ' '.join([
                    job_description.get('experience', ''),
                    job_description.get('skills', ''),
                    job_description.get('education', ''),
                    job_description.get('misc', '')
                ])
            elif isinstance(job_description, str):
                # Normalize JD if it's raw text
                jd_normalized = normalize_job_description(job_description)
                # Combine all JD sections for Cross-Encoder
                jd_text = ' '.join([
                    jd_normalized.get('job_title', ''),
                    jd_normalized.get('experience_required', ''),
                    jd_normalized.get('skills_required', ''),
                    jd_normalized.get('responsibilities', ''),
                    jd_normalized.get('education', ''),
                    jd_normalized.get('company_description', ''),
                    jd_normalized.get('location', '')
                ])
            else:
                # Assume it's already normalized
                jd_text = str(job_description)
            
            # Preprocess for dense models (preserve tokens like Node.js) and scrub PII
            jd_text = preprocess_text_for_dense_models(jd_text)
            jd_text = scrub_pii_and_boilerplate(jd_text)
            
            if not jd_text.strip():
                logger.warning("Empty job description after preprocessing, using fallback results")
                return self._create_fallback_results(candidates, section_tfidf_scores, skill_tfidf_scores, sbert_scores)
            
            results = []
            
            for i, candidate in enumerate(candidates):
                try:
                    # Prepare text for Cross-Encoder with preserved tokens
                    candidate_text = self._extract_candidate_text(candidate)
                    # Preprocess for dense models and scrub PII
                    candidate_text = preprocess_text_for_dense_models(candidate_text)
                    candidate_text = scrub_pii_and_boilerplate(candidate_text)
                    
                    if not candidate_text.strip():
                        logger.warning(f"Empty candidate text for candidate {i}, using zero score")
                        ce_score = 0.0
                    else:
                        # Try skill-anchored pairs first, fall back to full text
                        try:
                            ce_score = self._compute_skill_anchored_ce_score(job_description, candidate, jd_text, candidate_text)
                        except Exception as e:
                            logger.warning(f"Skill-anchored CE failed for candidate {i}, using fallback: {e}")
                            try:
                                ce_score = self.model.predict([(jd_text, candidate_text)])
                                ce_score = float(ce_score[0])
                            except Exception as e2:
                                logger.warning(f"Cross-Encoder prediction failed for candidate {i}: {e2}")
                                ce_score = 0.0
                    
                    # Ensure we have valid scores for this index
                    section_score = section_tfidf_scores[i] if i < len(section_tfidf_scores) else 0.0
                    skill_score = skill_tfidf_scores[i] if i < len(skill_tfidf_scores) else 0.0
                    sbert_score = sbert_scores[i] if i < len(sbert_scores) else 0.0
                    
                    # Combine scores with weights
                    final_score = self._combine_scores(ce_score, section_score, skill_score, sbert_score)
                    
                    results.append(CERerankerResult(
                        candidate_id=candidate.get('id', f'candidate_{i}'),
                        ce_score=ce_score,
                        section_tfidf=section_score,
                        skill_tfidf=skill_score,
                        sbert_score=sbert_score,
                        final_score=final_score
                    ))
                    
                except Exception as e:
                    logger.warning(f"Error processing candidate {i}: {e}")
                    # Add fallback result
                    results.append(CERerankerResult(
                        candidate_id=candidate.get('id', f'candidate_{i}'),
                        ce_score=0.0,
                        section_tfidf=section_tfidf_scores[i] if i < len(section_tfidf_scores) else 0.0,
                        skill_tfidf=skill_tfidf_scores[i] if i < len(skill_tfidf_scores) else 0.0,
                        sbert_score=sbert_scores[i] if i < len(sbert_scores) else 0.0,
                        final_score=0.0
                    ))
            
            # Sort by final score
            try:
                results.sort(key=lambda x: x.final_score, reverse=True)
            except Exception as e:
                logger.warning(f"Error sorting results: {e}")
                # Keep original order if sorting fails
            
            return results
            
        except Exception as e:
            logger.error(f"Error in rerank_candidates: {e}")
            return self._create_fallback_results(candidates, section_tfidf_scores, skill_tfidf_scores, sbert_scores)
    
    def _compute_skill_anchored_ce_score(self, job_description, candidate: Dict[str, Any], jd_text: str, candidate_text: str) -> float:
        """Compute CE score using skill-anchored pairs (≤12 pairs per resume)."""
        try:
            from .text_processor import SkillTaxonomy
            import numpy as np
            
            # Get must-have skills from JD
            must_have_skills = []
            if isinstance(job_description, dict):
                # Extract from jd_sections
                skills_text = job_description.get('skills', '')
                if skills_text:
                    taxonomy = SkillTaxonomy()
                    must_have_skills = taxonomy.extract_skills_from_text(skills_text)
            
            if not must_have_skills:
                # Fall back to full text scoring
                ce_score = self.model.predict([(jd_text, candidate_text)])
                return float(ce_score[0])
            
            # Limit to 12 skills for better coverage (was 8)
            must_have_skills = must_have_skills[:12]
            
            # Get candidate sections for line-level matching
            sections = candidate.get('sections', {})
            candidate_lines = []
            
            # Extract lines from all sections
            for section_name in ['experience', 'skills', 'education', 'misc']:
                section_content = sections.get(section_name, '')
                if isinstance(section_content, list):
                    candidate_lines.extend([str(item) for item in section_content])
                elif isinstance(section_content, str) and section_content:
                    # Split by sentences/lines
                    lines = [line.strip() for line in section_content.split('\n') if line.strip()]
                    if not lines:
                        lines = [line.strip() for line in section_content.split('.') if line.strip()]
                    candidate_lines.extend(lines)
            
            if not candidate_lines:
                # Fall back to full text scoring
                ce_score = self.model.predict([(jd_text, candidate_text)])
                return float(ce_score[0])
            
            # Build skill-anchored pairs with improved queries and multi-line evidence
            pairs = []
            taxonomy = SkillTaxonomy()
            
            # Extract contextual sentences from JD for better queries
            jd_sentences = []
            if isinstance(job_description, dict):
                for section in ['skills', 'experience', 'misc']:
                    content = job_description.get(section, '')
                    if content:
                        # Split into sentences
                        sentences = [s.strip() for s in content.split('.') if s.strip()]
                        jd_sentences.extend(sentences)
            else:
                # Fallback: split JD text into sentences
                sentences = [s.strip() for s in job_description.split('.') if s.strip()]
                jd_sentences.extend(sentences)
            
            # Pre-extract skills once per candidate line to avoid repeated extraction (performance)
            candidate_skill_map = {}
            for line in candidate_lines:
                try:
                    candidate_skill_map[line] = taxonomy.extract_skills_from_text(line)
                except Exception:
                    candidate_skill_map[line] = []

            for skill in must_have_skills:
                # Create improved query: extract contextual sentence from JD or use template
                query = self._extract_contextual_query(skill, jd_sentences, taxonomy)
                
                # Find multiple matching resume lines for richer evidence
                matching_lines = []
                
                for line, line_skills in candidate_skill_map.items():
                    # Check if line contains the skill (using taxonomy aliases)
                    if skill in line_skills:
                        line_clean = line.strip()
                        if len(line_clean) > 10:  # Meaningful content
                            # Calculate relevance score
                            skill_variants = taxonomy.skill_mapping.get(skill, [skill])
                            prominence = sum(1 for variant in skill_variants if variant.lower() in line_clean.lower())
                            match_score = prominence + min(len(line_clean) / 100, 1.0)
                            
                            matching_lines.append((match_score, line_clean))
                
                # Sort by relevance and take top 2-3 lines for multi-line evidence
                matching_lines.sort(key=lambda x: x[0], reverse=True)
                top_lines = matching_lines[:3]  # Up to 3 lines per skill
                
                # Create pairs for each top line
                for _, line in top_lines:
                    pairs.append((query, line))
                
                # If no matches found, create a fallback pair with generic query
                if not top_lines:
                    fallback_query = f"Experience with {skill}"
                    # Use the most relevant line from experience section
                    exp_lines = [line for line in candidate_lines if len(line.strip()) > 20]
                    if exp_lines:
                        # Take the longest experience line as fallback
                        best_fallback = max(exp_lines, key=len)
                        pairs.append((fallback_query, best_fallback))
            
            if not pairs:
                # Fall back to full text scoring
                ce_score = self.model.predict([(jd_text, candidate_text)])
                return float(ce_score[0])
            
            # Compute CE scores for all pairs and use top-K mean for stability
            pair_scores = []
            try:
                scores = self.model.predict(pairs)
                pair_scores = [float(s) for s in scores]
            except Exception as e:
                logger.warning(f"CE pair prediction failed: {e}")
                # Fall back to full text scoring
                ce_score = self.model.predict([(jd_text, candidate_text)])
                return float(ce_score[0])
            
            # Return top-K mean (top-5 or all if fewer) for more stable scoring
            if not pair_scores:
                return 0.0
            
            import numpy as np
            k = min(5, len(pair_scores))
            top_k_scores = sorted(pair_scores, reverse=True)[:k]
            return float(np.mean(top_k_scores))
            
        except Exception as e:
            logger.warning(f"Skill-anchored CE computation failed: {e}")
            # Fall back to full text scoring
            ce_score = self.model.predict([(jd_text, candidate_text)])
            return float(ce_score[0])
    
    def _extract_contextual_query(self, skill: str, jd_sentences: List[str], taxonomy) -> str:
        """
        Extract a contextual query from JD sentences that mention the skill.
        Falls back to template if no contextual sentence found.
        """
        skill_variants = taxonomy.skill_mapping.get(skill, [skill])
        
        # Look for sentences that mention the skill
        for sentence in jd_sentences:
            sentence_lower = sentence.lower()
            for variant in skill_variants:
                if variant.lower() in sentence_lower:
                    # Found contextual sentence - clean it up
                    # Remove extra whitespace and ensure it ends properly
                    clean_sentence = ' '.join(sentence.split())
                    if not clean_sentence.endswith('.'):
                        clean_sentence += '.'
                    return clean_sentence
        
        # Fallback to template-based query
        return f"Experience with {skill}"

    def _extract_candidate_text(self, candidate: Dict[str, Any]) -> str:
        """Extract text from candidate for Cross-Encoder (same structure as TF-IDF/SBERT)."""
        try:
            sections = candidate.get('sections', {})
            
            if isinstance(sections, dict):
                # Use same section structure as TF-IDF/SBERT
                all_content = []
                for section_name in ['experience', 'skills', 'education', 'misc']:
                    section_content = sections.get(section_name, '')
                    if isinstance(section_content, list):
                        all_content.extend([str(item) for item in section_content])
                    else:
                        all_content.append(str(section_content))
                return ' '.join(all_content)
            else:
                # Fallback to canonical text if sections not available
                canon = candidate.get('canonical_text')
                if isinstance(canon, str) and canon.strip():
                    return canon
                return ''
            
        except Exception as e:
            logger.warning(f"Error extracting candidate text: {e}")
            return ""
    
    def _combine_scores(self, ce_score: float, section_tfidf: float, skill_tfidf: float, sbert_score: float) -> float:
        """Combine multiple scores with weights."""
        try:
            # Weights for different components
            w_ce = 0.4      # Cross-Encoder
            w_section = 0.25 # Section TF-IDF
            w_skill = 0.2    # Skill TF-IDF
            w_sbert = 0.15   # SBERT
            
            # Normalize CE score to [0,1] (assuming it's already in reasonable range)
            ce_norm = max(0, min(1, ce_score))
            
            # Ensure all scores are valid floats
            section_tfidf = float(section_tfidf) if section_tfidf is not None else 0.0
            skill_tfidf = float(skill_tfidf) if skill_tfidf is not None else 0.0
            sbert_score = float(sbert_score) if sbert_score is not None else 0.0
            
            final_score = (w_ce * ce_norm + 
                          w_section * section_tfidf + 
                          w_skill * skill_tfidf + 
                          w_sbert * sbert_score)
            
            return final_score
            
        except Exception as e:
            logger.warning(f"Error combining scores: {e}")
            return 0.0
    
    def _create_fallback_results(self, candidates: List[Dict[str, Any]], 
                                section_tfidf_scores: List[float], 
                                skill_tfidf_scores: List[float], 
                                sbert_scores: List[float]) -> List[CERerankerResult]:
        """Create fallback results when Cross-Encoder is not available."""
        try:
            results = []
            
            for i, candidate in enumerate(candidates):
                try:
                    # Ensure we have valid scores for this index
                    section_score = section_tfidf_scores[i] if i < len(section_tfidf_scores) else 0.0
                    skill_score = skill_tfidf_scores[i] if i < len(skill_tfidf_scores) else 0.0
                    sbert_score = sbert_scores[i] if i < len(sbert_scores) else 0.0
                    
                    final_score = self._combine_scores(0.0, section_score, skill_score, sbert_score)
                    
                    results.append(CERerankerResult(
                        candidate_id=candidate.get('id', f'candidate_{i}'),
                        ce_score=0.0,
                        section_tfidf=section_score,
                        skill_tfidf=skill_score,
                        sbert_score=sbert_score,
                        final_score=final_score
                    ))
                    
                except Exception as e:
                    logger.warning(f"Error creating fallback result for candidate {i}: {e}")
                    # Add minimal fallback result
                    results.append(CERerankerResult(
                        candidate_id=candidate.get('id', f'candidate_{i}'),
                        ce_score=0.0,
                        section_tfidf=0.0,
                        skill_tfidf=0.0,
                        sbert_score=0.0,
                        final_score=0.0
                    ))
            
            return results
            
        except Exception as e:
            logger.error(f"Error in _create_fallback_results: {e}")
            # Return minimal results
            return [CERerankerResult(
                candidate_id=f'candidate_{i}',
                ce_score=0.0,
                section_tfidf=0.0,
                skill_tfidf=0.0,
                sbert_score=0.0,
                final_score=0.0
            ) for i in range(len(candidates))]
    
    def add_training_data(self, job_description: str, candidates: List[Dict[str, Any]], rankings: List[CERerankerResult]):
        """Add training data for feedback loop."""
        try:
            with self.lock:
                for ranking in rankings:
                    try:
                        # Find matching candidate
                        candidate = None
                        for c in candidates:
                            if isinstance(c, dict) and c.get('id') == ranking.candidate_id:
                                candidate = c
                                break
                        
                        if candidate is None:
                            logger.warning(f"Could not find candidate {ranking.candidate_id} for training data")
                            continue
                        
                        candidate_text = self._extract_candidate_text(candidate)
                        
                        self.training_data.append({
                            'job_description': job_description,
                            'candidate_text': candidate_text,
                            'score': ranking.final_score,
                            'rank': ranking.ce_score  # Use CE score as label
                        })
                    except Exception as e:
                        logger.warning(f"Error adding training data for ranking {ranking.candidate_id}: {e}")
                        continue
        except Exception as e:
            logger.error(f"Error in add_training_data: {e}")
    
    def train_model_async(self):
        """Train model asynchronously with accumulated data."""
        if len(self.training_data) < 10:  # Need minimum data
            return
        
        def train():
            try:
                self._train_model()
            except Exception as e:
                logger.error(f"Async training failed: {e}")
        
        thread = threading.Thread(target=train, daemon=True)
        thread.start()
    
    def _train_model(self):
        """Train the Cross-Encoder model with accumulated data."""
        if not self.model_available or len(self.training_data) < 10:
            return
        
        try:
            # Prepare training data
            train_pairs = []
            train_labels = []
            
            for data in self.training_data:
                train_pairs.append((data['job_description'], data['candidate_text']))
                train_labels.append(data['rank'])
            
            # Train model (simplified - in practice you'd use proper training)
            logger.info(f"Training Cross-Encoder with {len(train_pairs)} samples")
            
            # Save trained model
            self._save_model()
            
            # Clear training data
            with self.lock:
                self.training_data = []
                
        except Exception as e:
            logger.error(f"Model training failed: {e}")
    
    def _save_model(self):
        """Save the trained model with versioning."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        model_file = os.path.join(self.model_dir, f"ce_model_{timestamp}.pkl")
        
        try:
            with open(model_file, "wb") as f:
                pickle.dump(self.model, f)
            
            self.model_version += 1
            logger.info(f"Model saved: {model_file}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")

class LLMRanker:
    """LLM-based ranking system for resume evaluation (OpenAI or Google Gemini)."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        cfg = get_llm_config()
        # Allow explicit api_key/model override from caller; otherwise use config
        self.provider = cfg.get('provider', 'openai')
        self.api_key = api_key or cfg.get('api_key')
        self.model = model or cfg.get('model')
        self.enabled = bool(self.api_key)
        # Endpoints
        self.base_url_openai = "https://api.openai.com/v1/chat/completions"
        # Gemini REST: https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent
        self.base_url_gemini = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        
        # Blind JSON-only ranking prompt template
        self.ranking_prompt_template = """
You are an expert technical recruiter. Evaluate the candidate AGAINST THE JOB DESCRIPTION, using only evidence in the resume text.
Return ONLY valid JSON:
{
  "blind_score": 0-100,
  "evidence": ["exact short quotes from resume that justify the score"],
  "must_have_violations": ["list missing must-haves"],
  "notes": "one brief sentence"
}
Rules:
- Score must be justified by quotes.
- Penalize missing must-haves heavily.
- Do not assume skills that are not explicitly mentioned.
"""
    
    def rank_candidate(self, job_description: str, resume_data: Dict[str, Any], 
                      computed_scores: Dict[str, Any]) -> RankingResult:
        """Rank a single candidate against a job description."""
        try:
            # Prepare the prompt
            prompt = self.ranking_prompt_template.format(
                job_description=job_description,
                resume_data=json.dumps(resume_data, indent=2),
                computed_scores=json.dumps(computed_scores, indent=2)
            )
            
            # Call LLM API (strict JSON, temperature 0)
            response = self._call_llm_api(prompt)
            
            if response:
                # Parse the response
                ranking_data = self._parse_llm_response(response, prompt)
                
                return RankingResult(
                    candidate_id=resume_data.get('candidate_id', 'unknown'),
                    rank=0,  # Will be set when ranking multiple candidates
                    score=ranking_data.get('blind_score', 0),
                    reasoning='; '.join(ranking_data.get('evidence', [])),
                    strengths=[],
                    gaps=ranking_data.get('must_have_violations', []),
                    confidence=0
                )
            else:
                # Fallback ranking based on computed scores
                return self._fallback_ranking(resume_data, computed_scores)
                
        except Exception as e:
            logger.error(f"Error ranking candidate: {str(e)}")
            return self._fallback_ranking(resume_data, computed_scores)
    
    def rank_candidates(self, job_description: str, candidates_data: List[Dict[str, Any]]) -> List[RankingResult]:
        """Rank multiple candidates against a job description."""
        timing_bucket = {}
        ranking_results = []
        
        with time_phase("rank", timing_bucket):
            for candidate_data in candidates_data:
                try:
                    # Extract computed scores from candidate data
                    computed_scores = candidate_data.get('computed_scores', {})
                    
                    # Rank the candidate
                    result = self.rank_candidate(job_description, candidate_data, computed_scores)
                    ranking_results.append(result)
                    
                    # Add delay to avoid rate limiting
                    time.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error ranking candidate {candidate_data.get('candidate_id', 'unknown')}: {str(e)}")
                    continue
        
        # Sort by score (highest first) and assign ranks
        ranking_results.sort(key=lambda x: x.score, reverse=True)
        for i, result in enumerate(ranking_results):
            result.rank = i + 1
        
        # Log LLM summary
        if TIMING_ENABLED and timing_bucket:
            total_time = sum(timing_bucket.values())
            logger.info(f"[TIMING] LLM {len(candidates_data)} candidates: {total_time:.3f}s total")
        
        return ranking_results
    
    def _call_llm_api(self, prompt: str) -> Optional[str]:
        """Call the configured LLM API with the given prompt (OpenAI or Gemini)."""
        if not self.enabled:
            logger.warning("No API key provided for LLM ranking")
            return None

        timing_bucket = {}
        
        try:
            with time_phase("serialize_payload", timing_bucket):
                if self.provider == 'google':
                    # Gemini generateContent API requires role: user|model and parts
                    url = self.base_url_gemini
                    headers = {
                        "Content-Type": "application/json"
                    }
                    system_preamble = "You are a strict evaluator. Respond with ONLY valid JSON that matches the requested schema. No additional text."
                    body = {
                        "contents": [
                            {
                                "role": "user",
                                "parts": [
                                    {"text": system_preamble + "\n\n" + prompt}
                                ]
                            }
                        ],
                        "generationConfig": {"temperature": 0, "maxOutputTokens": 900}
                    }
                    params = {"key": self.api_key}
                else:
                    # OpenAI fallback
                    url = self.base_url_openai
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            body = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a strict evaluator. Respond with ONLY valid JSON that matches the requested schema. No additional text."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0,
                "max_tokens": 900
            }
            params = None
            
            with time_phase("http_post", timing_bucket):
                if self.provider == 'google':
                    resp = requests.post(url, headers=headers, params=params, json=body, timeout=10)
                else:
                    resp = requests.post(url, headers=headers, json=body, timeout=10)
            
            with time_phase("parse_response", timing_bucket):
                if resp.status_code == 200:
                    if self.provider == 'google':
                        data = resp.json()
                        # Extract text from candidates → content → parts
                        try:
                            return data["candidates"][0]["content"]["parts"][0]["text"]
                        except Exception:
                            logger.error(f"Unexpected Gemini response format: {data}")
                            return None
                    else:
                        result = resp.json()
                        return result['choices'][0]['message']['content']
                else:
                    if self.provider == 'google':
                        logger.error(f"Gemini API error: {resp.status_code} - {resp.text}")
                    else:
                        logger.error(f"OpenAI API error: {resp.status_code} - {resp.text}")
                    return None
                
        except Exception as e:
            logger.error(f"Error calling LLM API: {str(e)}")
            return None
    
    def _parse_llm_response(self, response: str, prompt: str) -> Dict[str, Any]:
        """Strict JSON parse; if it fails, retry API once with a short reminder."""
        def strip_code_fences(text: str) -> str:
            t = text.strip()
            if t.startswith("```"):
                # remove leading fence
                t = t.split("\n", 1)[1] if "\n" in t else t
            if t.endswith("```"):
                t = t.rsplit("```", 1)[0]
            return t.strip()

        def extract_braced_json(text: str) -> Optional[str]:
            start = text.find('{')
            if start == -1:
                return None
            depth = 0
            for i in range(start, len(text)):
                ch = text[i]
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        return text[start:i+1]
            return None

        def remove_trailing_commas(s: str) -> str:
            import re
            # Remove trailing commas before } or ]
            s = re.sub(r',\s*([}\]])', r'\1', s)
            return s

        try:
            return json.loads(response)
        except Exception:
            # Try fences/substring extraction first before retrying API
            cleaned = strip_code_fences(response)
            candidate = extract_braced_json(cleaned) or cleaned
            try:
                return json.loads(remove_trailing_commas(candidate))
            except Exception as e:
                logger.warning(f"Strict JSON parse failed: {e}; retrying once with reminder")
                # Build a safe retry prompt; if not available, skip retry.
                reminder = "Return ONLY JSON that matches the schema."
                base_prompt = None
                try:
                    # If the current scope passed `prompt`, use it; else try instance cache.
                    base_prompt = prompt if isinstance(prompt, str) and prompt.strip() else getattr(self, "last_prompt", None)
                except NameError:
                    base_prompt = getattr(self, "last_prompt", None)
                retry_resp = self._call_llm_api(reminder + "\n\n" + base_prompt) if base_prompt else None
                if not retry_resp:
                    return self._fallback_parse_response(response)
                cleaned2 = strip_code_fences(retry_resp)
                candidate2 = extract_braced_json(cleaned2) or cleaned2
                try:
                    return json.loads(remove_trailing_commas(candidate2))
                except Exception as e2:
                    logger.warning(f"Retry strict parse failed: {e2}")
                    return self._fallback_parse_response(retry_resp)
    
    def _fallback_parse_response(self, response: str) -> Dict[str, Any]:
        """Fallback parsing for non-JSON responses."""
        # Simple keyword-based parsing
        response_lower = response.lower()
        
        # Extract score
        score = 50  # Default score
        if 'score' in response_lower:
            try:
                score_match = re.search(r'score[:\s]*([0-9]+(?:\.[0-9]+)?)', response_lower)
                if score_match:
                    val = score_match.group(1)
                    score = int(float(val))
            except:
                pass
        
        # Extract confidence
        confidence = 70  # Default confidence
        if 'confidence' in response_lower:
            try:
                conf_match = re.search(r'confidence[:\s]*([0-9]+(?:\.[0-9]+)?)', response_lower)
                if conf_match:
                    val = conf_match.group(1)
                    confidence = int(float(val))
            except:
                pass
        
        return {
            'score': score,
            'reasoning': response[:500] + '...' if len(response) > 500 else response,
            'strengths': ['Strengths could not be parsed from response'],
            'gaps': ['Gaps could not be parsed from response'],
            'confidence': confidence
        }

    
    def _fallback_ranking(self, resume_data: Dict[str, Any], computed_scores: Dict[str, Any]) -> RankingResult:
        """Fallback ranking when LLM is not available."""
        # Calculate a simple score based on computed metrics
        tfidf_score = computed_scores.get('tfidf_score', 0)
        skill_diversity = computed_scores.get('skill_diversity', 0)
        semantic_richness = computed_scores.get('semantic_richness', 0)
        
        # Normalize and combine scores
        normalized_tfidf = min(tfidf_score / 10, 1.0)  # Normalize to 0-1
        normalized_skills = min(skill_diversity / 20, 1.0)  # Normalize to 0-1
        normalized_semantic = min(semantic_richness / 10, 1.0)  # Normalize to 0-1
        
        # Weighted average
        fallback_score = (normalized_tfidf * 0.4 + normalized_skills * 0.4 + normalized_semantic * 0.2) * 100
        
        return RankingResult(
            candidate_id=resume_data.get('candidate_id', 'unknown'),
            rank=0,
            score=fallback_score,
            reasoning="Fallback ranking based on computed metrics (LLM not available)",
            strengths=["Automated scoring based on content analysis"],
            gaps=["Detailed reasoning not available"],
            confidence=60
        )


class RankingManager:
    """Manager for handling ranking operations and results."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.llm_ranker = LLMRanker(api_key)
        self.ranking_history = []
    
    def create_ranking_session(self, job_description: str, candidates_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a new ranking session for a job posting."""
        session_id = f"ranking_{int(time.time())}"
        
        # Rank all candidates
        ranking_results = self.llm_ranker.rank_candidates(job_description, candidates_data)
        
        # Create session data
        session_data = {
            'session_id': session_id,
            'job_description': job_description,
            'candidates_count': len(candidates_data),
            'ranking_results': [
                {
                    'candidate_id': result.candidate_id,
                    'rank': result.rank,
                    'score': result.score,
                    'reasoning': result.reasoning,
                    'strengths': result.strengths,
                    'gaps': result.gaps,
                    'confidence': result.confidence
                }
                for result in ranking_results
            ],
            'created_at': time.time(),
            'status': 'completed'
        }
        
        # Store in history
        self.ranking_history.append(session_data)
        
        return session_data
    
    def get_ranking_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get a summary of ranking results for a session."""
        session = next((s for s in self.ranking_history if s['session_id'] == session_id), None)
        
        if not session:
            return None
        
        results = session['ranking_results']
        
        summary = {
            'session_id': session_id,
            'total_candidates': len(results),
            'average_score': sum(r['score'] for r in results) / len(results) if results else 0,
            'top_candidates': results[:5],  # Top 5 candidates
            'score_distribution': self._calculate_score_distribution(results),
            'created_at': session['created_at']
        }
        
        return summary
    
    def _calculate_score_distribution(self, results: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate score distribution across ranges."""
        distribution = {
            'excellent (90-100)': 0,
            'good (80-89)': 0,
            'average (70-79)': 0,
            'below_average (60-69)': 0,
            'poor (0-59)': 0
        }
        
        for result in results:
            score = result['score']
            if score >= 90:
                distribution['excellent (90-100)'] += 1
            elif score >= 80:
                distribution['good (80-89)'] += 1
            elif score >= 70:
                distribution['average (70-79)'] += 1
            elif score >= 60:
                distribution['below_average (60-69)'] += 1
            else:
                distribution['poor (0-59)'] += 1
        
        return distribution
    
    def export_ranking_results(self, session_id: str, format: str = 'json') -> Optional[str]:
        """Export ranking results in specified format."""
        session = next((s for s in self.ranking_history if s['session_id'] == session_id), None)
        
        if not session:
            return None
        
        if format.lower() == 'json':
            return json.dumps(session, indent=2)
        elif format.lower() == 'csv':
            return self._export_to_csv(session)
        else:
            logger.error(f"Unsupported export format: {format}")
            return None
    
    def _export_to_csv(self, session: Dict[str, Any]) -> str:
        """Export ranking results to CSV format."""
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(['Rank', 'Candidate ID', 'Score', 'Confidence', 'Reasoning'])
        
        # Write data
        for result in session['ranking_results']:
            writer.writerow([
                result['rank'],
                result['candidate_id'],
                result['score'],
                result['confidence'],
                result['reasoning'][:100] + '...' if len(result['reasoning']) > 100 else result['reasoning']
            ])
        
        return output.getvalue() 