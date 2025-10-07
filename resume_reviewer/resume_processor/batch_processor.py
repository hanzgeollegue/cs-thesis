import json
import logging
import os
import uuid
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from concurrent.futures import ThreadPoolExecutor
import time
from contextlib import contextmanager

# Version compatibility check
try:
    import sklearn
    sklearn_version = sklearn.__version__
    logger = logging.getLogger(__name__)
    logger.info(f"scikit-learn version: {sklearn_version}")
    
    # Check for known compatibility issues
    if sklearn_version < "0.24.0":
        logger.warning(f"scikit-learn version {sklearn_version} may have compatibility issues. Recommended: >=0.24.0")
        
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.error(f"scikit-learn not available: {e}. TF-IDF scoring will use fallback methods.")

# Use conditional imports to work both in Django and standalone contexts
try:
    # Try relative imports first (Django context)
    from .enhanced_pdf_parser import PDFParser
    from .llm_ranker import LLMRanker, CEReranker
    from .text_processor import SemanticEmbedding
    from .text_processor import clear_tfidf_caches, scrub_pii_and_boilerplate, preprocess_text_for_dense_models, chunk_text_for_sbert
    from .config import (
        get_openai_config,
        get_llm_config,
        validate_config,
        USE_LLM_RANKER,
        PARSE_CONCURRENCY,
        BATCH_TIMEOUT_SEC,
        ENABLE_MATCH_DETECTION,
        REQUIRE_MATCH_FOR_CE,
        ENABLE_META_COMBINER,
        META_RIDGE_L2,
        META_MIN_LABELS,
    )
except ImportError:
    # Fallback to absolute imports (standalone context)
    try:
        from enhanced_pdf_parser import PDFParser
        from llm_ranker import LLMRanker, CEReranker
        from text_processor import SemanticEmbedding
        from text_processor import clear_tfidf_caches, scrub_pii_and_boilerplate, preprocess_text_for_dense_models, chunk_text_for_sbert
        from config import (
            get_openai_config,
            get_llm_config,
            validate_config,
            USE_LLM_RANKER,
            PARSE_CONCURRENCY,
            BATCH_TIMEOUT_SEC,
            ENABLE_MATCH_DETECTION,
            REQUIRE_MATCH_FOR_CE,
            ENABLE_META_COMBINER,
            META_RIDGE_L2,
            META_MIN_LABELS,
        )
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        raise

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
class ResumeScores:
    """Data class for resume scoring results."""
    tfidf_section_score: float
    tfidf_taxonomy_score: float
    semantic_score: float
    cross_encoder: float = 0.0
    final_pre_llm: float = 0.0
    final_pre_llm_display: float = 0.0
    # New separate score fields
    section_tfidf: float = 0.0
    skill_tfidf: float = 0.0
    sbert_score: float = 0.0
    ce_score: float = 0.0
    # Combined TF-IDF and meta-combiner fields
    combined_tfidf: float = 0.0
    tfidf_norm: float = 0.0
    semantic_norm: float = 0.0
    ce_norm: float = 0.0
    has_match_skills: bool = False
    has_match_experience: bool = False
    matched_required_skills: List[str] = field(default_factory=list)
    coverage: float = 0.0
    gate_threshold: float = 0.0
    gate_reason: str = ""
    rationale: str = ""
    final_score: float = 0.0
    final_score_display: float = 0.0

@dataclass
class ParsedResume:
    """Data class for parsed resume data."""
    id: str
    sections: Dict[str, str]
    meta: Dict[str, Any]
    scores: ResumeScores
    matched_skills: List[Dict[str, str]]
    parsed: Dict[str, Any]

class BatchProcessor:
    """Comprehensive batch processor for resume ranking pipeline."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, disable_ocr: bool = False):
        # Initialize PDF parser with custom output directory to avoid Django settings dependency
        current_dir = os.getcwd()
        output_dir = os.path.join(current_dir, 'batch_processing_output')
        self.pdf_parser = PDFParser(output_dir=output_dir, disable_ocr=disable_ocr)
        
        # Get provider-agnostic LLM config and apply overrides
        llm_cfg = get_llm_config()
        self.api_key = api_key or llm_cfg['api_key']
        self.model = model or llm_cfg['model']
        
        # Validate configuration
        config_issues = validate_config()
        if config_issues:
            for issue in config_issues:
                logger.warning(issue)
        
        # Initialize LLM ranker
        if self.api_key:
            self.llm_ranker = LLMRanker(api_key=self.api_key, model=self.model)
            logger.info(f"LLM ranking enabled with model: {self.model}")
        else:
            self.llm_ranker = None
            logger.warning("No API key provided. LLM ranking will use fallback methods.")
        
        # Initialize SBERT for semantic embeddings
        try:
            self.semantic_embedding = SemanticEmbedding()
            if self.semantic_embedding.model_available:
                logger.info("SBERT semantic embedding enabled")
            else:
                logger.warning("SBERT not available. Semantic scoring will use fallback methods.")
        except Exception as e:
            logger.warning(f"Failed to initialize SBERT: {e}")
            self.semantic_embedding = None
        
        # Initialize Cross-Encoder reranker
        try:
            self.ce_reranker = CEReranker()
            if self.ce_reranker.model_available:
                logger.info("Cross-Encoder reranker enabled")
            else:
                logger.warning("Cross-Encoder not available. Reranking will use fallback methods.")
        except Exception as e:
            logger.warning(f"Failed to initialize Cross-Encoder: {e}")
            self.ce_reranker = None
        
        # Section weights as specified
        self.section_weights = {
            'experience': 0.45,
            'skills': 0.35,
            'education': 0.15,
            'misc': 0.05
        }
        
        # Canonical section mapping
        self.section_mapping = {
            'experience': ['experience', 'work experience', 'employment', 'career history', 'work history', 'professional experience'],
            'skills': ['skills', 'technical skills', 'core competencies', 'competencies', 'expertise', 'proficiencies'],
            'education': ['education', 'academic background', 'qualifications', 'academic qualifications'],
            'misc': ['summary', 'profile', 'objective', 'projects', 'certifications', 'awards', 'languages', 'interests', 'volunteer', 'leadership']
        }
        
        # Skill taxonomy (simplified - you can expand this)
        self.skill_taxonomy = {
            'SKILL_001': ['react', 'reactjs', 'react.js'],
            'SKILL_002': ['python', 'python3'],
            'SKILL_003': ['javascript', 'js', 'ecmascript'],
            'SKILL_004': ['java'],
            'SKILL_005': ['sql', 'mysql', 'postgresql', 'database'],
            'SKILL_006': ['aws', 'amazon web services', 'cloud'],
            'SKILL_007': ['docker', 'containerization'],
            'SKILL_008': ['kubernetes', 'k8s'],
            'SKILL_009': ['git', 'version control'],
            'SKILL_010': ['machine learning', 'ml', 'ai', 'artificial intelligence']
        }
        
        # TF-IDF vectorizers for each section
        self.tfidf_vectorizers = {}
        self._initialize_tfidf_vectorizers()
        
        # --- Meta combiner state ---
        # Weights: [bias, tfidf_norm, semantic_norm, ce_norm, has_match_skills, has_match_experience]
        self._meta_w = np.asarray([0.0, 0.30, 0.40, 0.30, 0.05, 0.05], dtype=float)
        self._meta_feedback: List[Dict[str, Any]] = []  # items: {"x": [...], "y": int}
        self._meta_feedback_path = os.path.join(os.getcwd(), 'batch_processing_output', 'meta_feedback.jsonl')
        try:
            os.makedirs(os.path.dirname(self._meta_feedback_path), exist_ok=True)
            if os.path.exists(self._meta_feedback_path):
                with open(self._meta_feedback_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            obj = json.loads(line.strip())
                            if isinstance(obj, dict) and 'x' in obj and 'y' in obj:
                                x = obj['x']
                                y = obj['y']
                                if isinstance(x, list) and len(x) == 6 and int(y) in (0, 1):
                                    self._meta_feedback.append({'x': x, 'y': int(y)})
                        except Exception:
                            continue
        except Exception:
            # Non-blocking
            pass
    
    def _initialize_tfidf_vectorizers(self):
        """Initialize TF-IDF vectorizers for each section."""
        for section in self.section_weights.keys():
            try:
                # Try with basic parameters for maximum compatibility
                self.tfidf_vectorizers[section] = TfidfVectorizer(
                    max_features=1000,
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.95
                )
            except Exception as e:
                logger.warning(f"Error initializing TF-IDF vectorizer for {section}: {e}")
                # Fallback to minimal configuration
                try:
                    self.tfidf_vectorizers[section] = TfidfVectorizer()
                except Exception as e2:
                    logger.error(f"Failed to initialize TF-IDF vectorizer for {section}: {e2}")
                    # Create a dummy vectorizer that won't break the pipeline
                    self.tfidf_vectorizers[section] = None
    
    def process_batch(self, resumes: List[str], job_description: str, jd_criteria: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Main batch processing pipeline."""
        timing_bucket = {}
        batch_start_time = time.perf_counter()
        
        try:
            # Optional: clear TF-IDF caches when testing
            try:
                from .config import DISABLE_TFIDF_CACHE
            except Exception:
                DISABLE_TFIDF_CACHE = False
            if DISABLE_TFIDF_CACHE:
                clear_tfidf_caches()
                logger.info("[PERF] Cleared TF-IDF caches (DISABLE_TFIDF_CACHE=1)")
            # Validate inputs
            if not job_description.strip():
                return {"error": "job_description_required"}
            
            if len(resumes) > 25:
                return {"error": f"Batch limit exceeded. Maximum 25 resumes allowed, got {len(resumes)}"}
            
            # Step 1: Parse PDFs to JSON
            logger.info(f"Starting batch processing of {len(resumes)} resumes")
            with time_phase("parse_all", timing_bucket):
                parsed_resumes = self._parse_pdfs_to_json(resumes)
                
                # Check batch timeout
                if time.perf_counter() - batch_start_time > BATCH_TIMEOUT_SEC:
                    logger.warning(f"Batch processing timed out after {BATCH_TIMEOUT_SEC}s")
                    return {
                        'success': False,
                        'error': f'Batch processing timed out after {BATCH_TIMEOUT_SEC}s',
                        'resumes': parsed_resumes,
                        'final_ranking': [],
                        'batch_summary': {'timeout': True}
                    }
            
            # Step 2: Normalize job description
            logger.info("Normalizing job description")
            with time_phase("jd_normalize", timing_bucket):
                from .text_processor import normalize_job_description
                jd_normalized = normalize_job_description(job_description)
                
                # Convert normalized JD to section format for backward compatibility
                jd_sections = {
                    'experience': jd_normalized.get('experience_required', ''),
                    'skills': jd_normalized.get('skills_required', ''),
                    'education': jd_normalized.get('education', ''),
                    'misc': ' '.join([
                        jd_normalized.get('job_title', ''),
                        jd_normalized.get('responsibilities', ''),
                        jd_normalized.get('company_description', ''),
                        jd_normalized.get('location', '')
                    ])
                }
            
            # Check batch timeout
            if time.perf_counter() - batch_start_time > BATCH_TIMEOUT_SEC:
                logger.warning(f"Batch processing timed out after {BATCH_TIMEOUT_SEC}s")
                return {
                    'success': False,
                    'error': f'Batch processing timed out after {BATCH_TIMEOUT_SEC}s',
                    'resumes': parsed_resumes,
                    'final_ranking': [],
                    'batch_summary': {'timeout': True}
                }
            
            # Step 3: Section-Aware TF-IDF scoring (taxonomy disabled for testing)
            logger.info("Computing Section-Aware TF-IDF scores (taxonomy disabled)")
            with time_phase("tfidf_fit", timing_bucket):
                try:
                    section_tfidf_scores, skill_tfidf_scores = self._compute_section_tfidf_scores(parsed_resumes, jd_sections)
                    # TEMPORARILY DISABLE TAXONOMY FOR TESTING
                    skill_tfidf_scores = [0.0] * len(parsed_resumes)
                    logger.info("Taxonomy TF-IDF scores set to 0.0 for testing")
                except Exception as e:
                    logger.error(f"Error in TF-IDF scoring: {e}")
                    section_tfidf_scores = [0.0] * len(parsed_resumes)
                    skill_tfidf_scores = [0.0] * len(parsed_resumes)
            
            # Step 4: SBERT semantic scoring
            logger.info("Computing SBERT semantic similarity scores")
            with time_phase("sbert_encode", timing_bucket):
                try:
                    # Pass raw job description for normalization within SBERT
                    # Build canonical text per resume and include as a synthetic section to unify model inputs
                    try:
                        from .text_processor import build_canonical_resume_text
                    except Exception:
                        build_canonical_resume_text = None
                    if build_canonical_resume_text:
                        for r in parsed_resumes:
                            try:
                                canon = build_canonical_resume_text(r.sections)
                                # Store for CE path as well
                                r.meta['canonical_text'] = canon
                            except Exception:
                                r.meta['canonical_text'] = ' '
                    sbert_scores = self._compute_sbert_scores(parsed_resumes, job_description)
                except Exception as e:
                    logger.error(f"Error in SBERT scoring: {e}")
                    sbert_scores = [0.0] * len(parsed_resumes)

            # Step 4.5: Combined TF-IDF (raw) and optional match detection
            try:
                # Compute combined TF-IDF and store (taxonomy disabled - use section only)
                for i, r in enumerate(parsed_resumes):
                    sec = section_tfidf_scores[i] if i < len(section_tfidf_scores) else 0.0
                    skl = skill_tfidf_scores[i] if i < len(skill_tfidf_scores) else 0.0
                    # TEMPORARILY DISABLE TAXONOMY: use section score only
                    combined = float(sec)  # was: 0.6 * float(sec) + 0.4 * float(skl)
                    r.scores.combined_tfidf = float(combined)
                
                # Optional raw token match detection
                if ENABLE_MATCH_DETECTION:
                    jd_exp_tokens = self._basic_tokens(jd_sections.get('experience', ''))
                    jd_sk_tokens = self._basic_tokens(jd_sections.get('skills', ''))
                    jd_exp_set = set(jd_exp_tokens)
                    jd_sk_set = set(jd_sk_tokens)
                    # Build matched_skills evidence using taxonomy with skill inference
                    try:
                        from .text_processor import SkillTaxonomy
                        taxonomy = SkillTaxonomy()
                        jd_skill_canon = set([taxonomy.normalize_skill(s) for s in jd_sk_set])
                    except Exception:
                        taxonomy = None
                        jd_skill_canon = set()
                    for r in parsed_resumes:
                        res_exp_tokens = set(self._basic_tokens(r.sections.get('experience', '')))
                        res_sk_tokens = set(self._basic_tokens(r.sections.get('skills', '')))
                        r.scores.has_match_experience = bool(jd_exp_set & res_exp_tokens)
                        r.scores.has_match_skills = bool(jd_sk_set & res_sk_tokens)
                        # Populate matched_skills from taxonomy intersect with skill inference
                        if taxonomy:
                            try:
                                # Extract explicit skills from Skills section
                                skills_text = str(r.sections.get('skills', ''))
                                explicit_skills = set(taxonomy.extract_skills_from_text(skills_text))
                                
                                # Infer skills from Experience and Projects sections
                                exp_text = str(r.sections.get('experience', ''))
                                misc_text = str(r.sections.get('misc', ''))  # often contains projects
                                context_text = exp_text + ' ' + misc_text
                                inferred_skills = set(taxonomy.extract_skills_from_text(context_text))
                                
                                # Combine explicit and inferred skills
                                all_skills = explicit_skills | inferred_skills
                                
                                # Find matches with JD
                                hits = list((all_skills & jd_skill_canon))[:10]  # Increased from 5 to 10
                                
                                # Build matched skills with explicit/inferred tags
                                r.matched_skills = []
                                for h in hits:
                                    skill_entry = {
                                        'skill_id': h,
                                        'surface_forms': [h],
                                        'source': 'explicit' if h in explicit_skills else 'inferred'
                                    }
                                    r.matched_skills.append(skill_entry)
                            except Exception:
                                pass
                else:
                    for r in parsed_resumes:
                        r.scores.has_match_experience = False
                        r.scores.has_match_skills = False
            except Exception as e:
                logger.warning(f"Combined TF-IDF or match detection failed: {e}")
                for r in parsed_resumes:
                    if not hasattr(r.scores, 'combined_tfidf'):
                        r.scores.combined_tfidf = 0.0
                    if not hasattr(r.scores, 'has_match_experience'):
                        r.scores.has_match_experience = False
                    if not hasattr(r.scores, 'has_match_skills'):
                        r.scores.has_match_skills = False
            
            # Step 5: Cross-Encoder reranking with multiple features (with optional short-circuit)
            logger.info("Applying Cross-Encoder reranker with multiple features")
            try:
                # Determine CE eligibility per resume
                ce_allowed_mask = [True] * len(parsed_resumes)
                if REQUIRE_MATCH_FOR_CE and ENABLE_MATCH_DETECTION:
                    ce_allowed_mask = [bool(r.scores.has_match_experience or r.scores.has_match_skills) for r in parsed_resumes]
                
                # Pass jd_sections for consistency with TF-IDF/SBERT
                ce_results = self._apply_cross_encoder_reranking(
                    parsed_resumes,
                    job_description,
                    section_tfidf_scores,
                    skill_tfidf_scores,
                    sbert_scores,
                    ce_allowed_mask,
                    jd_sections
                )
                
                # Store scores on each resume with defensive checks
                for i, result in enumerate(ce_results):
                    try:
                        parsed_resumes[i].scores.final_pre_llm = float(result.final_score)
                        parsed_resumes[i].scores.section_tfidf = float(result.section_tfidf)
                        parsed_resumes[i].scores.skill_tfidf = float(result.skill_tfidf)
                        parsed_resumes[i].scores.sbert_score = float(result.sbert_score)
                        parsed_resumes[i].scores.ce_score = float(result.ce_score)
                        # Store legacy aliases for backward compatibility
                        parsed_resumes[i].scores.tfidf_section_score = parsed_resumes[i].scores.section_tfidf
                        parsed_resumes[i].scores.tfidf_taxonomy_score = parsed_resumes[i].scores.skill_tfidf
                        parsed_resumes[i].scores.semantic_score = parsed_resumes[i].scores.sbert_score
                        parsed_resumes[i].scores.cross_encoder = parsed_resumes[i].scores.ce_score
                    except Exception as e:
                        logger.warning(f"Error storing scores for resume {i}: {e}")
                        # Set fallback scores
                        parsed_resumes[i].scores.final_pre_llm = 0.0
                        parsed_resumes[i].scores.section_tfidf = section_tfidf_scores[i] if i < len(section_tfidf_scores) else 0.0
                        parsed_resumes[i].scores.skill_tfidf = skill_tfidf_scores[i] if i < len(skill_tfidf_scores) else 0.0
                        parsed_resumes[i].scores.sbert_score = sbert_scores[i] if i < len(sbert_scores) else 0.0
                        parsed_resumes[i].scores.ce_score = 0.0
                        # Set legacy aliases
                        parsed_resumes[i].scores.tfidf_section_score = parsed_resumes[i].scores.section_tfidf
                        parsed_resumes[i].scores.tfidf_taxonomy_score = parsed_resumes[i].scores.skill_tfidf
                        parsed_resumes[i].scores.semantic_score = parsed_resumes[i].scores.sbert_score
                        parsed_resumes[i].scores.cross_encoder = parsed_resumes[i].scores.ce_score

                # Step 5.4: Improved SBERT computation with chunking and preserved tokens
                logger.info("Computing improved SBERT scores with chunking")
                try:
                    self._compute_improved_sbert_scores(parsed_resumes, jd_sections)
                except Exception as e:
                    logger.warning(f"Improved SBERT scoring failed: {e}")
                
                # Step 5.5: Normalization with guardrails - NO batch norm for CE!
                tfidf_vals = [float(getattr(r.scores, 'combined_tfidf', 0.0)) for r in parsed_resumes]
                sbert_vals = [float(getattr(r.scores, 'sbert_score', 0.0)) for r in parsed_resumes]
                lo_t, hi_t = (min(tfidf_vals), max(tfidf_vals)) if tfidf_vals else (0.0, 0.0)
                lo_s, hi_s = (min(sbert_vals), max(sbert_vals)) if sbert_vals else (0.0, 0.0)

                # Absolute normalization toggle
                try:
                    from .config import ABSOLUTE_NORMALIZATION
                except Exception:
                    ABSOLUTE_NORMALIZATION = True
                
                for r in parsed_resumes:
                    # Single-candidate guard: if batch size == 1, treat norms as perfect
                    if len(parsed_resumes) == 1:
                        r.scores.tfidf_norm = 1.0
                    else:
                        if ABSOLUTE_NORMALIZATION:
                            # Absolute scaling for TF-IDF combined
                            t_raw = float(getattr(r.scores, 'combined_tfidf', 0.0))
                            # Heuristic thresholds: >=0.40 strong → 1.0; 0.25–0.40 linear; <0.25 scale to 0
                            if t_raw >= 0.40:
                                r.scores.tfidf_norm = 1.0
                            elif t_raw >= 0.25:
                                r.scores.tfidf_norm = (t_raw - 0.25) / 0.15
                            else:
                                r.scores.tfidf_norm = max(0.0, t_raw / 0.25)
                        else:
                            r.scores.tfidf_norm = self._norm01(float(getattr(r.scores, 'combined_tfidf', 0.0)), lo_t, hi_t)
                    
                    # SBERT normalization with guardrails
                    sb_raw = float(getattr(r.scores, 'sbert_score', 0.0))
                    if len(parsed_resumes) == 1:
                        r.scores.semantic_norm = 1.0
                    else:
                        if ABSOLUTE_NORMALIZATION:
                            # Absolute scaling for SBERT
                            if sb_raw >= 0.75:
                                r.scores.semantic_norm = 1.0
                            elif sb_raw >= 0.60:
                                r.scores.semantic_norm = (sb_raw - 0.60) / 0.15
                            else:
                                r.scores.semantic_norm = max(0.0, sb_raw / 0.60)
                        else:
                            if lo_s == hi_s:  # Degenerate case: min==max
                                r.scores.semantic_norm = 0.5
                            else:
                                r.scores.semantic_norm = self._norm01(sb_raw, lo_s, hi_s)
                                # Optional safety valve: floor at 0.25 for high-confidence matches
                                coverage = getattr(r.scores, 'coverage', 0.0)
                                tfidf_n = getattr(r.scores, 'tfidf_norm', 0.0)
                                if coverage >= 0.75 and tfidf_n >= 0.8:
                                    r.scores.semantic_norm = max(0.25, r.scores.semantic_norm)
                    
                    # CE: NO batch normalization - use raw scores (already well-calibrated)
                    # CE scores are cosine similarities from skill-anchored pairs, already in [0,1]
                    ce_raw = float(getattr(r.scores, 'ce_score', 0.0))
                    if ABSOLUTE_NORMALIZATION:
                        if ce_raw >= 0.70:
                            r.scores.ce_norm = 1.0
                        elif ce_raw >= 0.55:
                            r.scores.ce_norm = (ce_raw - 0.55) / 0.15
                        else:
                            r.scores.ce_norm = max(0.0, ce_raw / 0.55)
                    else:
                        r.scores.ce_norm = max(0.0, min(1.0, ce_raw))
                    
                    # Optional: Coverage-aware boost for validated high-quality matches
                    if r.scores.coverage >= 0.75 and ce_raw >= 0.7:
                        r.scores.ce_norm = min(1.0, ce_raw * 1.05)  # 5% boost for validated matches
                    
                    logger.info(f"[NORM] {r.meta.get('source_file', 'unknown')}: "
                               f"CE_raw={ce_raw:.3f} → CE_norm={r.scores.ce_norm:.3f} "
                               f"(no batch norm, coverage={r.scores.coverage:.2f})")
                
                # Optionally refit meta weights if enough labels exist
                self._maybe_refit_meta_weights()
                
                # Compute meta final score per resume with criteria gates/penalties
                # Prepare criteria lookups
                must_have = []
                min_years = 0
                seniority = "mid"  # default
                try:
                    if isinstance(jd_criteria, dict):
                        must_have = [str(s).lower().strip() for s in jd_criteria.get('must_have_skills', []) if s]
                        min_years = int(jd_criteria.get('experience_min_years', 0) or 0)
                        seniority = str(jd_criteria.get('seniority_level', 'mid')).lower()
                except Exception:
                    pass
                
                # Set coverage thresholds by seniority
                coverage_thresholds = {
                    'intern': 0.2,
                    'junior': 0.3,
                    'mid': 0.5,
                    'senior': 0.6,
                    'lead': 0.7,
                    'staff': 0.8
                }
                threshold = coverage_thresholds.get(seniority, 0.5)
                
                for r in parsed_resumes:
                    x = [
                        1.0,
                        float(getattr(r.scores, 'tfidf_norm', 0.5)),
                        float(getattr(r.scores, 'semantic_norm', 0.5)),
                        float(getattr(r.scores, 'ce_norm', 0.5)),
                        1.0 if getattr(r.scores, 'has_match_skills', False) else 0.0,
                        1.0 if getattr(r.scores, 'has_match_experience', False) else 0.0,
                    ]
                    try:
                        # Base composite: 0.5*semantic + 0.3*ce + 0.2*tfidf
                        tf_n = x[1]; sb_n = x[2]; ce_n = x[3]
                        base_composite = 0.5*sb_n + 0.3*ce_n + 0.2*tf_n
                        
                        # Initialize coverage and gate info
                        coverage = 1.0
                        matched_required = []
                        gate_reason = ""
                        
                        # Must-have skills coverage penalty
                        if must_have and taxonomy:
                            try:
                                all_text = ' '.join([str(v) for v in r.sections.values()])
                                matched_required = taxonomy.get_matched_required_skills(all_text, must_have)
                                coverage = len(matched_required) / len(must_have) if must_have else 1.0
                                
                                if coverage < threshold:
                                    # Apply gamma=2 penalty: final = base * (coverage^2)
                                    score = base_composite * (coverage ** 2)
                                    gate_reason = f"skills_coverage_{coverage:.2f}<{threshold:.2f}"
                                else:
                                    score = base_composite
                            except Exception as e:
                                logger.warning(f"Error computing skill coverage: {e}")
                                score = base_composite
                        else:
                            score = base_composite
                        
                        # Experience years gate (heuristic) - keep existing logic
                        if min_years > 0:
                            try:
                                exp_text = r.sections.get('experience', '') or ''
                                yrs = 0
                                for m in re.findall(r"(\d+)\s+year", exp_text.lower()):
                                    try:
                                        yrs = max(yrs, int(m))
                                    except Exception:
                                        pass
                                if yrs < min_years:
                                    score = score * 0.5  # Soft penalty instead of zeroing
                                    if gate_reason:
                                        gate_reason += f",exp_{yrs}<{min_years}"
                                    else:
                                        gate_reason = f"exp_{yrs}<{min_years}"
                            except Exception:
                                pass
                        
                        # Store coverage and gate info
                        r.scores.matched_required_skills = matched_required
                        r.scores.coverage = coverage
                        r.scores.gate_threshold = threshold
                        r.scores.gate_reason = gate_reason
                        
                        # Recompute flags from final matched_required_skills list
                        r.scores.has_match_skills = bool(matched_required)
                        # For experience match, check if experience section has content
                        exp_content = str(r.sections.get('experience', '')).strip()
                        r.scores.has_match_experience = bool(exp_content)
                        
                        # Perfect-fit floor: if full coverage and experience present, enforce floor
                        try:
                            if coverage == 1.0 and r.scores.has_match_experience and getattr(r.scores, 'tfidf_norm', 0.0) >= 0.6:
                                score = max(score, 0.90)
                        except Exception:
                            pass
                        # Clamp to [0,1]
                        score = 0.0 if score < 0.0 else 1.0 if score > 1.0 else score
                    except Exception as e:
                        logger.warning(f"Error computing final score: {e}")
                        score = 0.0
                        r.scores.matched_required_skills = []
                        r.scores.coverage = 0.0
                        r.scores.gate_threshold = threshold
                        r.scores.gate_reason = "error"
                        # Set flags based on final state
                        r.scores.has_match_skills = False
                        exp_content = str(r.sections.get('experience', '')).strip()
                        r.scores.has_match_experience = bool(exp_content)
                    
                    r.scores.final_score = score
                    r.scores.final_score_display = round(100.0 * score, 2)
                    # Keep legacy display synchronized
                    r.scores.final_pre_llm = score
                    r.scores.final_pre_llm_display = r.scores.final_score_display
                    
                    filename = r.meta.get('source_file', 'unknown')
                    missing_skills = set(must_have) - set(matched_required) if must_have and matched_required else []
                    logger.info(f"[META] {filename} coverage={len(matched_required)}/{len(must_have) if must_have else 0} "
                               f"tf={r.scores.tfidf_norm:.3f} sb={r.scores.semantic_norm:.3f} ce={r.scores.ce_norm:.3f} "
                               f"ms={int(bool(r.scores.has_match_skills))} me={int(bool(r.scores.has_match_experience))} "
                               f"missing={list(missing_skills)[:3]}{'...' if len(missing_skills) > 3 else ''} -> final={score:.3f}")
                
                # Select top 50 for LLM processing (apply deterministic tie-breaking)
                try:
                    def sort_key(i):
                        s = getattr(parsed_resumes[i].scores, 'final_score', 0.0)
                        sb = getattr(parsed_resumes[i].scores, 'semantic_norm', 0.0)
                        ce = getattr(parsed_resumes[i].scores, 'ce_norm', 0.0)
                        tf = getattr(parsed_resumes[i].scores, 'tfidf_norm', 0.0)
                        # Tie-breaking: final_score desc, then semantic_norm desc, then ce_norm desc, then tfidf_norm desc, then id
                        return (-s, -sb, -ce, -tf, parsed_resumes[i].id)
                    sorted_indices = sorted(range(len(parsed_resumes)), key=sort_key, reverse=True)
                    top50_idx = sorted_indices[:50]
                    llm_subset = [parsed_resumes[i] for i in top50_idx]
                except Exception as e:
                    logger.warning(f"Error sorting CE results: {e}")
                    llm_subset = parsed_resumes[:50]  # Take first 50 as fallback
                
                # Add training data for feedback loop (async, won't block)
                try:
                    if self.ce_reranker:
                        # Convert ParsedResume objects to dict format for training
                        candidate_dicts = []
                        for resume in parsed_resumes:
                            candidate_dict = {
                                'id': resume.id,
                                'sections': resume.sections
                            }
                            candidate_dicts.append(candidate_dict)
                        
                        self.ce_reranker.add_training_data(job_description, candidate_dicts, ce_results)
                        self.ce_reranker.train_model_async()
                except Exception as e:
                    logger.warning(f"Error in feedback loop: {e}")
                
            except Exception as e:
                logger.warning(f"Cross-Encoder reranker failed, falling back to classical scoring: {e}")
                llm_subset = parsed_resumes
                for i, r in enumerate(parsed_resumes):
                    try:
                        if not hasattr(r.scores, 'final_pre_llm'):
                            r.scores.final_pre_llm = 0.0
                        if not hasattr(r.scores, 'section_tfidf'):
                            r.scores.section_tfidf = section_tfidf_scores[i] if i < len(section_tfidf_scores) else 0.0
                        if not hasattr(r.scores, 'skill_tfidf'):
                            r.scores.skill_tfidf = skill_tfidf_scores[i] if i < len(skill_tfidf_scores) else 0.0
                        if not hasattr(r.scores, 'sbert_score'):
                            r.scores.sbert_score = sbert_scores[i] if i < len(sbert_scores) else 0.0
                        if not hasattr(r.scores, 'ce_score'):
                            r.scores.ce_score = 0.0
                    except Exception as e2:
                        logger.warning(f"Error setting fallback scores for resume {i}: {e2}")

            # Generate plain language rationale for each candidate
            logger.info("Generating candidate rationales")
            for resume in parsed_resumes:
                try:
                    rationale = self._generate_candidate_rationale(resume, jd_criteria)
                    resume.scores.rationale = rationale
                except Exception as e:
                    logger.warning(f"Error generating rationale for resume {getattr(resume, 'id', 'unknown')}: {e}")
                    resume.scores.rationale = "Overall Match: Unable to assess"
            
            # Step 6: LLM ranking on selected subset (if enabled)
            if USE_LLM_RANKER:
                logger.info("Generating LLM-based rankings (top-50 subset)")
                with time_phase("llm_rank", timing_bucket):
                    try:
                        final_ranking = self._generate_llm_rankings(llm_subset, job_description)
                    except Exception as e:
                        logger.error(f"Error in LLM ranking: {e}")
                        final_ranking = self._generate_ce_composite_rankings(parsed_resumes)
            else:
                # Generate plain language rationale for each candidate
                logger.info("Generating candidate rationales")
                for resume in parsed_resumes:
                    try:
                        rationale = self._generate_candidate_rationale(resume, jd_criteria)
                        resume.scores.rationale = rationale
                    except Exception as e:
                        logger.warning(f"Error generating rationale for resume {getattr(resume, 'id', 'unknown')}: {e}")
                        resume.scores.rationale = "Overall Match: Unable to assess"
                
                logger.info("LLM ranking disabled, using Meta combiner composite scoring")
                with time_phase("llm_rank", timing_bucket):
                    try:
                        final_ranking = self._generate_meta_rankings(parsed_resumes)
                    except Exception as e:
                        logger.error(f"Error in CE composite ranking: {e}")
                        # Create minimal fallback ranking
                        final_ranking = [{
                            'id': f'resume_{i}',
                            'rank': i + 1,
                            'scores_snapshot': {
                                'section_tfidf': 0.0,
                                'skill_tfidf': 0.0,
                                'combined_tfidf': 0.0,
                                'sbert_score': 0.0,
                                'ce_score': 0.0,
                                'tfidf_norm': 0.5,
                                'semantic_norm': 0.5,
                                'ce_norm': 0.5,
                                'has_match_skills': 0,
                                'has_match_experience': 0,
                                'final_score': 0.0,
                                'final_pre_llm_display': 0.0
                            },
                            'reasoning': "Error in ranking generation",
                            'feedback': {}
                        } for i in range(len(parsed_resumes))]
            
            # Step 8: Assemble final output
            with time_phase("save_results", timing_bucket):
                try:
                    output = self._assemble_output(parsed_resumes, final_ranking, job_description, jd_criteria=jd_criteria)
                except Exception as e:
                    logger.error(f"Error assembling final output: {e}")
                    # Create minimal fallback output
                    output = {
                        'success': False,
                        'error': f'Error assembling output: {str(e)}',
                        'resumes': parsed_resumes,
                        'final_ranking': final_ranking,
                        'batch_summary': {'error': True}
                    }
            
            logger.info("Batch processing completed successfully")
            
            # Log batch summary
            if TIMING_ENABLED and timing_bucket:
                total_time = sum(timing_bucket.values())
                logger.info(f"[TIMING] SUMMARY batch({len(resumes)} resumes): {total_time:.3f}s total")
            
            return output
            
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Return minimal fallback output
            try:
                fallback_resumes = []
                for i, resume_path in enumerate(resumes):
                    fallback_resumes.append(self._create_fallback_resume(resume_path, str(e)))
                
                return {
                    'success': False,
                    'error': f'Batch processing failed: {str(e)}',
                    'resumes': fallback_resumes,
                    'final_ranking': [],
                    'batch_summary': {'error': True, 'exception': str(e)}
                }
            except Exception as e2:
                logger.error(f"Error creating fallback output: {e2}")
                return {
                    'success': False,
                    'error': f'Critical error: {str(e)}',
                    'resumes': [],
                    'final_ranking': [],
                    'batch_summary': {'critical_error': True}
                }
    
    def _parse_pdfs_to_json(self, resume_paths: List[str]) -> List[ParsedResume]:
        """Parse PDF resumes to structured JSON format."""
        parsed_resumes = []

        if not resume_paths:
            return parsed_resumes

        max_workers = max(1, min(PARSE_CONCURRENCY, len(resume_paths), os.cpu_count() or 2))
        logger.info(f"[PERF] Using {max_workers} workers for PDF parsing (deterministic order)")
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks and collect results in deterministic order
                # This ensures identical inputs produce identical outputs
                futures = [executor.submit(self._parse_single_resume, path) for path in resume_paths]
                
                for i, future in enumerate(futures):
                    path = resume_paths[i]  # Use original order for deterministic results
                    try:
                        parsed_resume = future.result()
                        if parsed_resume:
                            parsed_resumes.append(parsed_resume)
                    except Exception as e:
                        logger.error(f"Error parsing {path}: {str(e)}")
                        # Create a minimal resume entry for failed parses
                        parsed_resumes.append(self._create_fallback_resume(path, str(e)))
        except RuntimeError as e:
            # Common when Django's autoreloader is restarting the interpreter
            logger.warning(f"Thread pool unavailable ({e}); falling back to sequential parsing")
            for path in resume_paths:
                try:
                    parsed_resume = self._parse_single_resume(path)
                    if parsed_resume:
                        parsed_resumes.append(parsed_resume)
                except Exception as ex:
                    logger.error(f"Error parsing {path} sequentially: {str(ex)}")
                    parsed_resumes.append(self._create_fallback_resume(path, str(ex)))

        return parsed_resumes
    
    def _parse_single_resume(self, resume_path: str) -> Optional[ParsedResume]:
        """Parse a single resume PDF."""
        try:
            # Extract structured data using the correct method
            structured_data = self.pdf_parser._extract_structured_data(resume_path)
            
            if not structured_data.get('success', False):
                logger.warning(f"Failed to parse {resume_path}: {structured_data.get('error', 'Unknown error')}")
                return None
            
            # Normalize sections to canonical format
            normalized_sections = self._normalize_sections(structured_data['sections'])
            
            # Create parsed resume object
            resume_id = str(uuid.uuid4())
            filename = os.path.basename(resume_path)
            
            parsed_resume = ParsedResume(
                id=resume_id,
                sections=normalized_sections,
                meta={
                    "source_file": filename,
                    "pages": len(structured_data.get('layout_metadata', {}).get('text_elements', [])),
                    "processing_status": "success"
                },
                scores=ResumeScores(0.0, 0.0, 0.0),
                matched_skills=[],
                parsed=self._extract_parsed_data(structured_data)
            )
            
            return parsed_resume
            
        except Exception as e:
            logger.error(f"Error parsing {resume_path}: {str(e)}")
            return None
    
    def _normalize_sections(self, sections: List[Dict[str, Any]]) -> Dict[str, str]:
        """Normalize section headers to canonical keys."""
        normalized = {
            'experience': '',
            'skills': '',
            'education': '',
            'misc': ''
        }
        
        for section in sections:
            header = section['header'].lower()
            content = ' '.join(section['content'])
            
            # Map to canonical section
            mapped_section = None
            for canonical, aliases in self.section_mapping.items():
                if any(alias in header for alias in aliases):
                    mapped_section = canonical
                    break
            
            if mapped_section:
                normalized[mapped_section] = content
            else:
                # Add to misc if no clear mapping
                if normalized['misc']:
                    normalized['misc'] += ' ' + content
                else:
                    normalized['misc'] = content
        
        return normalized
    
    def _extract_parsed_data(self, structured_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract structured parsed data from resume."""
        parsed = {
            'experience': [],
            'skills': [],
            'education': [],
            'misc': ''
        }
        
        # Extract experience (simplified parsing)
        experience_section = None
        for section in structured_data.get('sections', []):
            if any(word in section['header'].lower() for word in ['experience', 'work', 'employment']):
                experience_section = section
                break
        
        if experience_section:
            # Simple experience parsing (you can enhance this)
            for content in experience_section['content']:
                if content.strip():
                    parsed['experience'].append({
                        'role': 'Software Engineer',  # Placeholder - enhance parsing
                        'company': 'Company Name',   # Placeholder - enhance parsing
                        'dates': '2020-2023',       # Placeholder - enhance parsing
                        'bullets': [content.strip()]
                    })
        
        # Extract skills
        skills_section = None
        for section in structured_data.get('sections', []):
            if any(word in section['header'].lower() for word in ['skills', 'competencies']):
                skills_section = section
                break
        
        if skills_section:
            for content in skills_section['content']:
                if content.strip():
                    parsed['skills'].append(content.strip())
        
        # Extract education
        education_section = None
        for section in structured_data.get('sections', []):
            if any(word in section['header'].lower() for word in ['education', 'academic']):
                education_section = section
                break
        
        if education_section:
            for content in education_section['content']:
                if content.strip():
                    parsed['education'].append({
                        'degree': 'Bachelor\'s',  # Placeholder - enhance parsing
                        'institution': 'University',  # Placeholder - enhance parsing
                        'year': '2020'  # Placeholder - enhance parsing
                    })
        
        # Extract candidate name from summary
        summary = structured_data.get('summary', {})
        candidate_name = summary.get('candidate_name', '')
        if candidate_name:
            parsed['candidate_name'] = candidate_name
        
        return parsed
    
    def _compute_section_tfidf_scores(self, resumes: List[ParsedResume], jd_sections: Dict[str, str]) -> Tuple[List[float], List[float]]:
        """Compute section and skill TF-IDF scores using separate vectorizers."""
        try:
            from .text_processor import SectionAwareTFIDF
            
            if not resumes:
                logger.warning("No resumes provided for TF-IDF scoring")
                return [], []
            
            # Convert resumes to dict format for text processor
            resume_dicts = []
            for resume in resumes:
                try:
                    resume_dict = {
                        'sections': resume.sections,
                        'matched_skills': resume.matched_skills
                    }
                    resume_dicts.append(resume_dict)
                except Exception as e:
                    logger.warning(f"Error processing resume {getattr(resume, 'id', 'unknown')}: {e}")
                    # Add fallback resume dict
                    resume_dicts.append({
                        'sections': {},
                        'matched_skills': []
                    })
            
            # Use SectionAwareTFIDF processor
            tfidf_processor = SectionAwareTFIDF()
            result = tfidf_processor.compute_section_tfidf_scores(resume_dicts, jd_sections)
            
            # Defensive check for None result
            if result is None:
                logger.warning("SectionAwareTFIDF returned None, using fallback scores")
                return [0.0] * len(resumes), [0.0] * len(resumes)
            
            section_scores, skill_scores = result
            
            # Ensure we have the right number of scores
            if len(section_scores) != len(resumes):
                logger.warning(f"Section scores length mismatch: expected {len(resumes)}, got {len(section_scores)}")
                section_scores = [0.0] * len(resumes)
            
            if len(skill_scores) != len(resumes):
                logger.warning(f"Skill scores length mismatch: expected {len(resumes)}, got {len(skill_scores)}")
                skill_scores = [0.0] * len(resumes)
            
            return section_scores, skill_scores
            
        except Exception as e:
            logger.error(f"Error in _compute_section_tfidf_scores: {e}")
            return [0.0] * len(resumes), [0.0] * len(resumes)
    
    def _compute_sbert_scores(self, resumes: List[ParsedResume], job_description: str) -> List[float]:
        """Compute SBERT semantic similarity scores."""
        try:
            if not resumes:
                logger.warning("No resumes provided for SBERT scoring")
                return []
            
            if not self.semantic_embedding or not self.semantic_embedding.model_available:
                logger.warning("SBERT not available, returning zero scores")
                return [0.0] * len(resumes)
            
            # Convert resumes to dict format for text processor
            resume_dicts = []
            for resume in resumes:
                try:
                    resume_dict = {
                        'sections': resume.sections
                    }
                    resume_dicts.append(resume_dict)
                except Exception as e:
                    logger.warning(f"Error processing resume {getattr(resume, 'id', 'unknown')} for SBERT: {e}")
                    # Add fallback resume dict
                    resume_dicts.append({
                        'sections': []
                    })
            
            result = self.semantic_embedding.compute_sbert_scores(resume_dicts, job_description)
            
            # Defensive check for None result
            if result is None:
                logger.warning("SBERT compute_sbert_scores returned None, using fallback scores")
                return [0.0] * len(resumes)
            
            # Ensure we have the right number of scores
            if len(result) != len(resumes):
                logger.warning(f"SBERT scores length mismatch: expected {len(resumes)}, got {len(result)}")
                return [0.0] * len(resumes)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in _compute_sbert_scores: {e}")
            return [0.0] * len(resumes)
    
    def _apply_cross_encoder_reranking(self, resumes: List[ParsedResume], job_description: str,
                                     section_tfidf_scores: List[float], skill_tfidf_scores: List[float], 
                                     sbert_scores: List[float], ce_allowed_mask: Optional[List[bool]] = None, 
                                     jd_sections: Optional[Dict[str, str]] = None) -> List:
        """Apply Cross-Encoder reranking with multiple features."""
        try:
            if not resumes:
                logger.warning("No resumes provided for Cross-Encoder reranking")
                return []
            
            if not self.ce_reranker or not self.ce_reranker.model_available:
                logger.warning("Cross-Encoder not available, using fallback scoring")
                return self._create_fallback_ce_results(resumes, section_tfidf_scores, skill_tfidf_scores, sbert_scores)
            
            # Convert resumes to dict format
            resume_dicts = []
            for idx, resume in enumerate(resumes):
                try:
                    # If CE is not allowed for this resume, pass empty sections to short-circuit CE compute
                    if ce_allowed_mask is not None and idx < len(ce_allowed_mask) and not ce_allowed_mask[idx]:
                        sections_payload = {'experience': '', 'skills': '', 'education': '', 'misc': ''}
                    else:
                        sections_payload = resume.sections
                    resume_dict = {
                        'id': resume.id,
                        'sections': sections_payload
                    }
                    resume_dicts.append(resume_dict)
                except Exception as e:
                    logger.warning(f"Error processing resume {getattr(resume, 'id', 'unknown')} for CE: {e}")
                    # Add fallback resume dict
                    resume_dicts.append({
                        'id': getattr(resume, 'id', f'fallback_{len(resume_dicts)}'),
                        'sections': []
                    })
            
            # Pass jd_sections if available, otherwise use job_description
            jd_input = jd_sections if jd_sections is not None else job_description
            result = self.ce_reranker.rerank_candidates(jd_input, resume_dicts, 
                                                      section_tfidf_scores, skill_tfidf_scores, sbert_scores)
            # Optional short-circuit: force CE score to 0.0 when not allowed
            if ce_allowed_mask is not None and result:
                try:
                    for i in range(min(len(result), len(ce_allowed_mask))):
                        if not ce_allowed_mask[i]:
                            try:
                                result[i].ce_score = 0.0
                            except Exception:
                                pass
                except Exception:
                    pass
            
            # Defensive check for None result
            if result is None:
                logger.warning("Cross-Encoder reranker returned None, using fallback scoring")
                return self._create_fallback_ce_results(resumes, section_tfidf_scores, skill_tfidf_scores, sbert_scores)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in _apply_cross_encoder_reranking: {e}")
            return self._create_fallback_ce_results(resumes, section_tfidf_scores, skill_tfidf_scores, sbert_scores)

    # --- Utility helpers ---
    def _basic_tokens(self, text: str) -> List[str]:
        try:
            return [t for t in re.findall(r"\b[a-zA-Z][a-zA-Z0-9+\-_.]{1,}\b", (text or "").lower())]
        except Exception:
            return []

    def _norm01(self, x: float, lo: Optional[float], hi: Optional[float]) -> float:
        try:
            if hi is None or lo is None or hi <= lo:
                return 0.5
            v = (x - lo) / (hi - lo)
            if v < 0.0:
                return 0.0
            if v > 1.0:
                return 1.0
            return float(v)
        except Exception:
            return 0.5

    def _maybe_refit_meta_weights(self) -> None:
        try:
            if not ENABLE_META_COMBINER:
                return
            if not self._meta_feedback or len(self._meta_feedback) < META_MIN_LABELS:
                return
            # Prepare X, y
            X = np.asarray([row['x'] for row in self._meta_feedback if isinstance(row, dict) and 'x' in row and 'y' in row], dtype=float)
            y = np.asarray([row['y'] for row in self._meta_feedback if isinstance(row, dict) and 'x' in row and 'y' in row], dtype=float)
            if X.ndim != 2 or X.shape[1] != 6 or y.ndim != 1 or X.shape[0] != y.shape[0]:
                return
            # Closed-form ridge: w = (X^T X + λI)^-1 X^T y
            XtX = X.T @ X
            lam = float(META_RIDGE_L2)
            I = np.eye(XtX.shape[0], dtype=float)
            A = XtX + lam * I
            Xty = X.T @ y
            try:
                w = np.linalg.solve(A, Xty)
            except Exception:
                w = np.linalg.pinv(A) @ Xty
            # Clamp weights to reasonable range to avoid exploding contributions
            w = np.asarray(w, dtype=float)
            w = np.clip(w, -2.0, 2.0)
            self._meta_w = w
            logger.info(f"[META] Updated weights: {np.round(self._meta_w, 3).tolist()}")
        except Exception as e:
            logger.warning(f"Meta weight refit failed: {e}")

    def _generate_meta_rankings(self, resumes: List[ParsedResume]) -> List[Dict[str, Any]]:
        try:
            # Sort by meta final score descending
            try:
                resumes_sorted = sorted(resumes, key=lambda r: getattr(r.scores, 'final_score', 0.0), reverse=True)
            except Exception:
                resumes_sorted = resumes
            final_ranking: List[Dict[str, Any]] = []
            for i, r in enumerate(resumes_sorted):
                try:
                    final_ranking.append({
                        'id': getattr(r, 'id', f'resume_{i}'),
                        'rank': i + 1,
                        'scores_snapshot': {
                            'section_tfidf': float(getattr(r.scores, 'section_tfidf', 0.0)),
                            'skill_tfidf': float(getattr(r.scores, 'skill_tfidf', 0.0)),
                            'combined_tfidf': float(getattr(r.scores, 'combined_tfidf', 0.0)),
                            'sbert_score': float(getattr(r.scores, 'sbert_score', 0.0)),
                            'ce_score': float(getattr(r.scores, 'ce_score', 0.0)),
                            'tfidf_norm': float(getattr(r.scores, 'tfidf_norm', 0.5)),
                            'semantic_norm': float(getattr(r.scores, 'semantic_norm', 0.5)),
                            'ce_norm': float(getattr(r.scores, 'ce_norm', 0.5)),
                            'has_match_skills': int(1 if getattr(r.scores, 'has_match_skills', False) else 0),
                            'has_match_experience': int(1 if getattr(r.scores, 'has_match_experience', False) else 0),
                            'matched_required_skills': getattr(r.scores, 'matched_required_skills', []),
                            'coverage': float(getattr(r.scores, 'coverage', 0.0)),
                            'gate_threshold': float(getattr(r.scores, 'gate_threshold', 0.0)),
                            'gate_reason': getattr(r.scores, 'gate_reason', ''),
                            'rationale': getattr(r.scores, 'rationale', 'Overall Match: Unable to assess'),
                            'final_score': float(getattr(r.scores, 'final_score', 0.0)),
                            'final_pre_llm_display': float(getattr(r.scores, 'final_score_display', 0.0)),
                        },
                        'reasoning': (
                            f"Meta composite: tfidf={getattr(r.scores, 'tfidf_norm', 0.5):.3f}, "
                            f"sb={getattr(r.scores, 'semantic_norm', 0.5):.3f}, ce={getattr(r.scores, 'ce_norm', 0.5):.3f}, "
                            f"coverage={getattr(r.scores, 'coverage', 0.0):.2f}, "
                            f"ms={int(1 if getattr(r.scores, 'has_match_skills', False) else 0)}, "
                            f"me={int(1 if getattr(r.scores, 'has_match_experience', False) else 0)}"
                        ),
                        'feedback': {}
                    })
                except Exception as e:
                    logger.warning(f"Error creating meta ranking row: {e}")
                    final_ranking.append({
                        'id': getattr(r, 'id', f'resume_{i}'),
                        'rank': i + 1,
                        'scores_snapshot': {
                            'section_tfidf': 0.0,
                            'skill_tfidf': 0.0,
                            'combined_tfidf': 0.0,
                            'sbert_score': 0.0,
                            'ce_score': 0.0,
                            'tfidf_norm': 0.5,
                            'semantic_norm': 0.5,
                            'ce_norm': 0.5,
                            'has_match_skills': 0,
                            'has_match_experience': 0,
                            'final_score': 0.0,
                            'final_pre_llm_display': 0.0,
                        },
                        'reasoning': "Error in meta ranking",
                        'feedback': {}
                    })
            return final_ranking
        except Exception as e:
            logger.error(f"Error in _generate_meta_rankings: {e}")
            return []
    
    def _create_fallback_ce_results(self, resumes: List[ParsedResume], section_tfidf_scores: List[float], 
                                  skill_tfidf_scores: List[float], sbert_scores: List[float]) -> List:
        """Create fallback results when Cross-Encoder is not available."""
        try:
            from .llm_ranker import CERerankerResult
            
            if not resumes:
                logger.warning("No resumes provided for fallback CE results")
                return []
            
            results = []
            for i, resume in enumerate(resumes):
                try:
                    # Ensure we have valid scores for this index
                    section_score = section_tfidf_scores[i] if i < len(section_tfidf_scores) else 0.0
                    skill_score = skill_tfidf_scores[i] if i < len(skill_tfidf_scores) else 0.0
                    sbert_score = sbert_scores[i] if i < len(sbert_scores) else 0.0
                    
                    # Simple weighted combination
                    final_score = (0.25 * section_score + 
                                  0.2 * skill_score + 
                                  0.15 * sbert_score)
                    
                    results.append(CERerankerResult(
                        candidate_id=getattr(resume, 'id', f'fallback_{i}'),
                        ce_score=0.0,
                        section_tfidf=section_score,
                        skill_tfidf=skill_score,
                        sbert_score=sbert_score,
                        final_score=final_score
                    ))
                except Exception as e:
                    logger.warning(f"Error creating fallback result for resume {i}: {e}")
                    # Add minimal fallback result
                    results.append(CERerankerResult(
                        candidate_id=getattr(resume, 'id', f'fallback_{i}'),
                        ce_score=0.0,
                        section_tfidf=0.0,
                        skill_tfidf=0.0,
                        sbert_score=0.0,
                        final_score=0.0
                    ))
            
            return results
            
        except Exception as e:
            logger.error(f"Error in _create_fallback_ce_results: {e}")
            # Return minimal results
            from .llm_ranker import CERerankerResult
            return [CERerankerResult(
                candidate_id=f'fallback_{i}',
                ce_score=0.0,
                section_tfidf=0.0,
                skill_tfidf=0.0,
                sbert_score=0.0,
                final_score=0.0
            ) for i in range(len(resumes))]
    
    def _generate_ce_composite_rankings(self, resumes: List[ParsedResume]) -> List[Dict[str, Any]]:
        """Generate rankings using Cross-Encoder composite scores."""
        try:
            if not resumes:
                logger.warning("No resumes provided for composite rankings")
                return []
            
            rankings = []
            
            for i, resume in enumerate(resumes):
                try:
                    scores = resume.scores
                    
                    # Get scores with fallbacks
                    section_tfidf = getattr(scores, 'section_tfidf', 0.0)
                    skill_tfidf = getattr(scores, 'skill_tfidf', 0.0)
                    sbert_score = getattr(scores, 'sbert_score', 0.0)
                    ce_score = getattr(scores, 'ce_score', 0.0)
                    final_pre_llm = getattr(scores, 'final_pre_llm', 0.0)
                    
                    ranking = {
                        'id': getattr(resume, 'id', f'resume_{i}'),
                        'rank': i + 1,
                        'scores_snapshot': {
                            'section_tfidf': section_tfidf,
                            'skill_tfidf': skill_tfidf,
                            'sbert_score': sbert_score,
                            'ce_score': ce_score,
                            'final_pre_llm_display': final_pre_llm
                        },
                        'reasoning': f"Composite score: Section TF-IDF={section_tfidf:.3f}, "
                                   f"Skill TF-IDF={skill_tfidf:.3f}, "
                                   f"SBERT={sbert_score:.3f}, "
                                   f"CE={ce_score:.3f}",
                        'feedback': {}  # Empty feedback dict as required
                    }
                    rankings.append(ranking)
                    
                except Exception as e:
                    logger.warning(f"Error creating ranking for resume {i}: {e}")
                    # Add minimal fallback ranking
                    rankings.append({
                        'id': getattr(resume, 'id', f'resume_{i}'),
                        'rank': i + 1,
                        'scores_snapshot': {
                            'section_tfidf': 0.0,
                            'skill_tfidf': 0.0,
                            'sbert_score': 0.0,
                            'ce_score': 0.0,
                            'final_pre_llm_display': 0.0
                        },
                        'reasoning': "Error in score computation",
                        'feedback': {}
                    })
            
            return rankings
            
        except Exception as e:
            logger.error(f"Error in _generate_ce_composite_rankings: {e}")
            # Return minimal fallback rankings
            return [{
                'id': f'resume_{i}',
                'rank': i + 1,
                'scores_snapshot': {
                    'section_tfidf': 0.0,
                    'skill_tfidf': 0.0,
                    'sbert_score': 0.0,
                    'ce_score': 0.0,
                    'final_pre_llm_display': 0.0
                },
                'reasoning': "Error in ranking generation",
                'feedback': {}
            } for i in range(len(resumes))]
    
    def _create_fallback_resume(self, path: str, error: str) -> ParsedResume:
        """Create a fallback resume entry for failed parses."""
        return ParsedResume(
            id=str(uuid.uuid4()),
            sections={'experience': '', 'skills': '', 'education': '', 'misc': ''},
            meta={
                "source_file": os.path.basename(path),
                "pages": 0,
                "processing_status": "failed",
                "error": error
            },
            scores=ResumeScores(0.0, 0.0, 0.0),
            matched_skills=[],
            parsed={'experience': [], 'skills': [], 'education': [], 'misc': ''}
        )
    
    def _extract_jd_sections(self, job_description: str) -> Dict[str, str]:
        """Extract job description sections via improved header heuristics."""
        if not job_description or not job_description.strip():
            return {'experience': '', 'skills': '', 'education': '', 'misc': ''}

        text = job_description.replace('\r', '\n')
        lines = [ln.strip() for ln in text.split('\n') if ln.strip()]

        # Map common JD headers to canonical sections
        header_map = {
            'experience': ['experience', 'responsibilities', 'what you will do', 'role', 'duties', 'job description'],
            'skills': ['requirements', 'qualifications', 'skills', 'nice to have', 'must have', 'preferred', 'technical skills', 'technologies'],
            'education': ['education', 'academic', 'degree', 'requirements', 'qualifications'],
        }

        def to_canonical(h: str) -> Optional[str]:
            h_low = h.lower().strip(':').strip('-').strip()
            for canon, keys in header_map.items():
                if any(k in h_low for k in keys):
                    return canon
            return None

        sections = {'experience': [], 'skills': [], 'education': [], 'misc': []}
        current = 'misc'
        
        for ln in lines:
            # Enhanced header detection: look for common patterns
            is_header = False
            
            # Pattern 1: Short lines with common header words
            if len(ln) <= 80 and re.match(r'^[A-Za-z][A-Za-z\s/&-]{2,}$', ln):
                canon = to_canonical(ln)
                if canon:
                    current = canon
                    is_header = True
            
            # Pattern 2: Lines ending with colon
            elif ln.endswith(':'):
                canon = to_canonical(ln[:-1])
                if canon:
                    current = canon
                    is_header = True
            
            # Pattern 3: Lines starting with bullet points and common words
            elif re.match(r'^[-•*]\s*', ln):
                canon = to_canonical(ln[2:].strip())
                if canon:
                    current = canon
                    is_header = True
            
            if not is_header:
                sections[current].append(ln)

        # If no specific sections found, try to distribute content intelligently
        if not any(sections[k] for k in ['experience', 'skills', 'education']):
            # Look for skill-related keywords in the text
            full_text = ' '.join(lines).lower()
            
            # Enhanced skill keywords
            skill_keywords = [
                'python', 'javascript', 'react', 'sql', 'aws', 'git', 'programming', 'development', 
                'database', 'cloud', 'docker', 'kubernetes', 'node.js', 'angular', 'vue', 'typescript',
                'java', 'c++', 'c#', 'php', 'ruby', 'go', 'rust', 'swift', 'kotlin', 'scala',
                'html', 'css', 'bootstrap', 'jquery', 'mongodb', 'postgresql', 'mysql', 'redis',
                'machine learning', 'ai', 'artificial intelligence', 'data science', 'analytics',
                'devops', 'ci/cd', 'jenkins', 'terraform', 'ansible', 'kubernetes', 'microservices'
            ]
            
            # Enhanced experience keywords
            exp_keywords = [
                'years', 'experience', 'develop', 'build', 'create', 'design', 'implement',
                'architect', 'lead', 'manage', 'mentor', 'collaborate', 'deliver', 'deploy',
                'maintain', 'optimize', 'debug', 'test', 'refactor', 'scale', 'performance'
            ]
            
            # Education keywords
            edu_keywords = [
                'degree', 'bachelor', 'master', 'phd', 'university', 'college', 'certification',
                'diploma', 'course', 'training', 'academic', 'education', 'qualification'
            ]
            
            # Distribute content based on keyword presence
            if any(kw in full_text for kw in skill_keywords):
                sections['skills'] = lines
            if any(kw in full_text for kw in exp_keywords):
                sections['experience'] = lines
            if any(kw in full_text for kw in edu_keywords):
                sections['education'] = lines

        return {k: ' '.join(v).strip() for k, v in sections.items()}
    
    def _compute_section_tfidf_scores(self, resumes: List[ParsedResume], jd_sections: Dict[str, str]) -> Tuple[List[float], List[float]]:
        """Compute Section-Aware TF-IDF scores.
        Always returns a tuple (section_scores, skill_scores).
        """
        # Build corpus for each section
        section_corpus = {section: [] for section in self.section_weights.keys()}
        
        # Add JD sections to corpus
        for section, content in jd_sections.items():
            if section in section_corpus:
                section_corpus[section].append(content)
        
        # Add resume sections to corpus
        for resume in resumes:
            for section, content in resume.sections.items():
                if section in section_corpus:
                    section_corpus[section].append(content)
        
        # Debug logging
        logger.info(f"JD sections: {jd_sections}")
        logger.info(f"Section corpus sizes: {[(k, len(v)) for k, v in section_corpus.items()]}")
        logger.info(f"Number of resumes: {len(resumes)}")
        
        # Reset scores before accumulation
        for resume in resumes:
            resume.scores.tfidf_section_score = 0.0

        # Fit TF-IDF models and compute scores (accumulate weighted per-section similarity)
        for section, corpus in section_corpus.items():
            if len(corpus) > 1:  # Need at least 2 documents for TF-IDF
                try:
                    # Check if vectorizer is available
                    if self.tfidf_vectorizers[section] is None:
                        logger.warning(f"TF-IDF vectorizer for {section} is not available, using fallback scoring")
                        # Use fallback scoring
                        for resume in resumes:
                            sim = self._compute_fallback_score(
                                jd_sections[section], resume.sections.get(section, '')
                            )
                            resume.scores.tfidf_section_score += self.section_weights.get(section, 0.0) * sim
                        continue
                    
                    # Fit the vectorizer
                    vectors = self.tfidf_vectorizers[section].fit_transform(corpus)
                    
                    # Compute similarity between JD and each resume
                    jd_vector = vectors[0:1]  # First document is JD
                    resume_vectors = vectors[1:]  # Rest are resumes
                    
                    similarities = cosine_similarity(jd_vector, resume_vectors).flatten()
                    
                    # Assign scores to resumes
                    for i, resume in enumerate(resumes):
                        if i < len(similarities):
                            score_contribution = self.section_weights.get(section, 0.0) * float(similarities[i])
                            resume.scores.tfidf_section_score += score_contribution
                        else:
                            resume.scores.tfidf_section_score += 0.0
                    
                    # Log section summary
                    if similarities.size > 0:
                        avg_sim = similarities.mean()
                        max_sim = similarities.max()
                        logger.info(f"Section '{section}' - Avg similarity: {avg_sim:.4f}, Max: {max_sim:.4f}")
                            
                except Exception as e:
                    logger.warning(f"Error computing TF-IDF for {section}: {str(e)}")
                    # Use fallback scoring
                    for resume in resumes:
                        sim = self._compute_fallback_score(
                            jd_sections[section], resume.sections.get(section, '')
                        )
                        resume.scores.tfidf_section_score += self.section_weights.get(section, 0.0) * sim
            else:
                logger.warning(f"Section '{section}' has only {len(corpus)} documents, skipping TF-IDF")
        
        # Log final TF-IDF scores summary
        logger.info("Final TF-IDF Section Scores:")
        for i, resume in enumerate(resumes):
            logger.info(f"  Resume {i}: {resume.scores.tfidf_section_score:.4f}")

        # Defensive: ensure tuple return even if called directly
        section_scores = [r.scores.tfidf_section_score for r in resumes]
        skill_scores = [r.scores.tfidf_taxonomy_score if hasattr(r.scores, 'tfidf_taxonomy_score') else 0.0 for r in resumes]
        return section_scores, skill_scores
    
    def _compute_taxonomy_tfidf_scores(self, resumes: List[ParsedResume], jd_sections: Dict[str, str]):
        """Compute Skill-Cluster TF-IDF scores using canonical skill tokens."""
        # Canonicalize skills in resumes and JD
        canon_resumes = []
        for resume in resumes:
            canon_resume = self._canonicalize_skills(resume)
            canon_resumes.append(canon_resume)
        
        canon_jd = self._canonicalize_jd_skills(jd_sections)
        
        # Recompute TF-IDF with canonicalized content
        self._compute_section_tfidf_scores_canonical(canon_resumes, canon_jd)
        
        # Update original resumes with taxonomy scores
        for i, resume in enumerate(resumes):
            if i < len(canon_resumes):
                resume.scores.tfidf_taxonomy_score = canon_resumes[i].scores.tfidf_section_score
                resume.matched_skills = canon_resumes[i].matched_skills
        
        # Log final Taxonomy scores summary
        logger.info("Final TF-IDF Taxonomy Scores:")
        for i, resume in enumerate(resumes):
            logger.info(f"  Resume {i}: {resume.scores.tfidf_taxonomy_score:.4f}")
    
    def _canonicalize_skills(self, resume: ParsedResume) -> ParsedResume:
        """Replace skill phrases with canonical SKILL_ID tokens."""
        canon_resume = ParsedResume(
            id=resume.id,
            sections=resume.sections.copy(),
            meta=resume.meta.copy(),
            scores=resume.scores,
            matched_skills=[],
            parsed=resume.parsed.copy()
        )
        
        # Aggregate unique matched skills and their surface forms
        matched: dict = {}
        
        for section_name, content in resume.sections.items():
            canon_content = content
            for skill_id, surface_forms in self.skill_taxonomy.items():
                for surface_form in surface_forms:
                    if surface_form.lower() in content.lower():
                        # Replace with canonical token
                        canon_content = re.sub(
                            re.escape(surface_form), 
                            skill_id, 
                            canon_content, 
                            flags=re.IGNORECASE
                        )
                        
                        # Track matched skills
                        if skill_id not in matched:
                            matched[skill_id] = set()
                        matched[skill_id].add(surface_form)
            
            canon_resume.sections[section_name] = canon_content
        
        # Convert to list of dicts with deduplicated surface forms
        canon_resume.matched_skills = [
            {'skill_id': sid, 'surface_forms': sorted(list(forms))}
            for sid, forms in matched.items()
        ]
        return canon_resume
    
    def _canonicalize_jd_skills(self, jd_sections: Dict[str, str]) -> Dict[str, str]:
        """Canonicalize skills in job description sections."""
        canon_jd = {}
        for section, content in jd_sections.items():
            canon_content = content
            for skill_id, surface_forms in self.skill_taxonomy.items():
                for surface_form in surface_forms:
                    if surface_form.lower() in content.lower():
                        canon_content = re.sub(
                            re.escape(surface_form), 
                            skill_id, 
                            canon_content, 
                            flags=re.IGNORECASE
                        )
            canon_jd[section] = canon_content
        
        return canon_jd
    
    def _compute_section_tfidf_scores_canonical(self, resumes: List[ParsedResume], jd_sections: Dict[str, str]):
        """Compute TF-IDF scores for canonicalized content (true TF-IDF + cosine)."""
        section_corpus: Dict[str, List[str]] = {section: [] for section in self.section_weights.keys()}

        # JD first per section
        for section, content in jd_sections.items():
            if section in section_corpus:
                section_corpus[section].append(content or '')

        # Then resume texts
        for resume in resumes:
            for section, content in resume.sections.items():
                if section in section_corpus:
                    section_corpus[section].append(content or '')

        # Per section TF-IDF + cosine (JD vs resumes)
        per_resume_scores = [0.0] * len(resumes)
        for section, corpus in section_corpus.items():
            if len(corpus) <= 1:
                continue
            try:
                vec = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), min_df=1, max_df=0.95)
                X = vec.fit_transform(corpus)
                jd_vec = X[0:1]
                res_vecs = X[1:]
                sims = cosine_similarity(jd_vec, res_vecs).flatten()
                weight = self.section_weights.get(section, 0.0)
                for i, s in enumerate(sims):
                    per_resume_scores[i] += weight * float(s)
            except Exception as e:
                logger.warning(f"Canonical TF-IDF error in section '{section}': {e}")
                jd_text = corpus[0] if corpus else ''
                for i, resume in enumerate(resumes):
                    sim = self._compute_text_similarity(jd_text, resume.sections.get(section, ''))
                    per_resume_scores[i] += self.section_weights.get(section, 0.0) * sim

        # Store into the canonical resume objects’ tfidf_section_score (pipeline expects this)
        for i, resume in enumerate(resumes):
            resume.scores.tfidf_section_score = float(min(1.0, per_resume_scores[i]))
    
    def _compute_text_similarity(self, text1: str, text2: str) -> float:
        """Compute basic text similarity between two strings."""
        if not text1 or not text2:
            return 0.0
        
        # Simple word overlap similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    

    def _compute_improved_sbert_scores(self, resumes: List[ParsedResume], jd_sections: Dict[str, str]):
        """Compute SBERT scores with chunking, preserved tokens, and max-over-chunks scoring."""
        try:
            import numpy as np
            
            # Reuse already-loaded model from self.semantic_embedding (avoids 15-20s reload)
            if not self.semantic_embedding or not self.semantic_embedding.model_available:
                logger.warning("SBERT not available for improved scoring")
                for resume in resumes:
                    resume.scores.sbert_score = 0.0
                    resume.scores.semantic_score = 0.0
                return
            
            model = self.semantic_embedding.model
            
            # Build JD text with preserved tokens (same structure as TF-IDF but less normalized)
            jd_sections_text = []
            for section_name in ['experience', 'skills', 'education', 'misc']:
                section_text = jd_sections.get(section_name, '')
                if section_text:
                    # Use dense model preprocessing (preserves tokens like Node.js)
                    processed_text = preprocess_text_for_dense_models(section_text)
                    if processed_text:
                        jd_sections_text.append(processed_text)
            
            jd_text = ' '.join(jd_sections_text)
            if not jd_text:
                jd_text = "No job description provided"
            
            # Apply PII scrubbing but keep important tokens
            jd_text = scrub_pii_and_boilerplate(jd_text)
            
            # Chunk JD text and batch-encode for efficiency
            jd_chunks = [c for c in chunk_text_for_sbert(jd_text, max_tokens=512) if c.strip()]
            jd_embeddings = []
            if jd_chunks:
                try:
                    jd_embeddings = list(model.encode(jd_chunks, normalize_embeddings=True, batch_size=64))
                except Exception:
                    # Fallback to per-chunk encode if batch encode not supported
                    for chunk in jd_chunks:
                        emb = model.encode(chunk, normalize_embeddings=True)
                        jd_embeddings.append(emb)
            
            if not jd_embeddings:
                logger.warning("No valid JD embeddings generated")
                return
            
            # Process each resume with chunking
            for resume in resumes:
                try:
                    # Build resume text with same structure but preserved tokens
                    resume_sections_text = []
                    for section_name in ['experience', 'skills', 'education', 'misc']:
                        section_content = resume.sections.get(section_name, '')
                        if section_content:
                            # Use dense model preprocessing
                            processed_content = preprocess_text_for_dense_models(str(section_content))
                            if processed_content:
                                resume_sections_text.append(processed_content)
                    
                    resume_text = ' '.join(resume_sections_text)
                    if not resume_text:
                        resume_text = "No resume content available"
                    
                    # Apply PII scrubbing but keep important tokens
                    resume_text = scrub_pii_and_boilerplate(resume_text)
                    
                    # Chunk resume text
                    resume_chunks = [c for c in chunk_text_for_sbert(resume_text, max_tokens=512) if c.strip()]
                    
                    # Batch-encode resume chunks for efficiency
                    resume_chunk_embeddings = []
                    if resume_chunks:
                        try:
                            resume_chunk_embeddings = list(model.encode(resume_chunks, normalize_embeddings=True, batch_size=64))
                        except Exception:
                            # Fallback to per-chunk encode
                            for res_chunk in resume_chunks:
                                resume_chunk_embeddings.append(model.encode(res_chunk, normalize_embeddings=True))
                    
                    # Compute scores over all chunk pairs
                    max_score = 0.0
                    valid_scores = []
                    for res_emb in resume_chunk_embeddings:
                        for jd_emb in jd_embeddings:
                            score = float(np.dot(jd_emb, res_emb))
                            valid_scores.append(score)
                            if score > max_score:
                                max_score = score
                    
                    # Always use top-k mean for stability (never raw max)
                    if valid_scores:
                        # Use top-5 mean (or all scores if fewer than 5)
                        k = min(5, len(valid_scores))
                        valid_scores.sort(reverse=True)
                        final_score = np.mean(valid_scores[:k])
                    else:
                        final_score = 0.0
                    
                    # Update the resume score
                    resume.scores.sbert_score = final_score
                    resume.scores.semantic_score = final_score  # Keep legacy alias
                    
                except Exception as e:
                    logger.warning(f"Error processing resume {getattr(resume, 'id', 'unknown')} for improved SBERT: {e}")
                    resume.scores.sbert_score = 0.0
                    resume.scores.semantic_score = 0.0
            
            # Log improved semantic scores summary
            logger.info("Improved SBERT Scores:")
            for i, resume in enumerate(resumes):
                logger.info(f"  Resume {i}: {resume.scores.sbert_score:.4f}")
                
        except Exception as e:
            logger.warning(f"Improved SBERT computation failed: {e}")
            # Fall back to original scores
            for resume in resumes:
                if not hasattr(resume.scores, 'sbert_score') or resume.scores.sbert_score is None:
                    resume.scores.sbert_score = 0.0
                    resume.scores.semantic_score = 0.0

    def _generate_candidate_rationale(self, resume: ParsedResume, jd_criteria: Optional[Dict] = None) -> str:
        """Generate plain language rationale for a candidate using existing signals."""
        try:
            scores = resume.scores
            
            # Extract key metrics
            tfidf_score = getattr(scores, 'tfidf_norm', 0.0)
            semantic_score = getattr(scores, 'semantic_norm', 0.0)
            ce_score = getattr(scores, 'ce_norm', 0.0)
            coverage = getattr(scores, 'coverage', 0.0)
            matched_skills = getattr(scores, 'matched_required_skills', [])
            has_exp_match = getattr(scores, 'has_match_experience', False)
            has_skill_match = getattr(scores, 'has_match_skills', False)
            
            # Get JD criteria for context
            must_have_skills = []
            min_years = 0
            seniority = "mid"
            if jd_criteria:
                must_have_skills = jd_criteria.get('must_have_skills', [])
                min_years = jd_criteria.get('experience_min_years', 0)
                seniority = jd_criteria.get('seniority_level', 'mid')
            
            # Build rationale components
            rationale_parts = []
            
            # Overall assessment tier
            overall_score = getattr(scores, 'final_score', 0.0)
            if overall_score >= 0.8:
                overall_tier = "Excellent"
            elif overall_score >= 0.6:
                overall_tier = "Strong"
            elif overall_score >= 0.4:
                overall_tier = "Moderate"
            elif overall_score >= 0.2:
                overall_tier = "Limited"
            else:
                overall_tier = "Low"
            
            rationale_parts.append(f"Overall Match: {overall_tier}")
            
            # Skills assessment
            if must_have_skills:
                matched_count = len(matched_skills)
                total_count = len(must_have_skills)
                skill_coverage = matched_count / total_count if total_count > 0 else 0
                
                if skill_coverage >= 0.8:
                    skill_tier = "Excellent"
                elif skill_coverage >= 0.6:
                    skill_tier = "Strong"
                elif skill_coverage >= 0.4:
                    skill_tier = "Moderate"
                elif skill_coverage >= 0.2:
                    skill_tier = "Limited"
                else:
                    skill_tier = "Low"
                
                rationale_parts.append(f"Required Skills: {skill_tier} ({matched_count}/{total_count} matched)")
                
                if matched_skills:
                    skill_list = ", ".join(matched_skills[:3])
                    if len(matched_skills) > 3:
                        skill_list += f" and {len(matched_skills) - 3} more"
                    rationale_parts.append(f"Matched: {skill_list}")
                
                missing_skills = set(must_have_skills) - set(matched_skills)
                if missing_skills:
                    missing_list = ", ".join(list(missing_skills)[:3])
                    if len(missing_skills) > 3:
                        missing_list += f" and {len(missing_skills) - 3} more"
                    rationale_parts.append(f"Missing: {missing_list}")
            else:
                rationale_parts.append("Required Skills: Not specified")
            
            # Experience assessment
            if has_exp_match:
                rationale_parts.append("Experience Match: Yes")
            else:
                rationale_parts.append("Experience Match: No")
            
            # Content relevance (semantic similarity)
            if semantic_score >= 0.7:
                content_tier = "Excellent"
            elif semantic_score >= 0.5:
                content_tier = "Strong"
            elif semantic_score >= 0.3:
                content_tier = "Moderate"
            elif semantic_score >= 0.1:
                content_tier = "Limited"
            else:
                content_tier = "Low"
            
            rationale_parts.append(f"Content Relevance: {content_tier}")
            
            # Keyword match strength
            if tfidf_score >= 0.7:
                keyword_tier = "Excellent"
            elif tfidf_score >= 0.5:
                keyword_tier = "Strong"
            elif tfidf_score >= 0.3:
                keyword_tier = "Moderate"
            elif tfidf_score >= 0.1:
                keyword_tier = "Limited"
            else:
                keyword_tier = "Low"
            
            rationale_parts.append(f"Keyword Match: {keyword_tier}")
            
            # Gate reasons if applicable
            gate_reason = getattr(scores, 'gate_reason', '')
            if gate_reason:
                if 'skills_coverage' in gate_reason:
                    rationale_parts.append("Note: Below minimum skill requirements")
                elif 'exp_' in gate_reason:
                    rationale_parts.append("Note: Below minimum experience requirements")
                elif 'missing_skills_penalty' in gate_reason:
                    rationale_parts.append("Note: Penalty applied for missing required skills")
            
            # Combine all parts
            rationale = " | ".join(rationale_parts)
            
            return rationale
            
        except Exception as e:
            logger.warning(f"Error generating rationale for resume {getattr(resume, 'id', 'unknown')}: {e}")
            return "Overall Match: Unable to assess"

    def _z(self, arr):
        import numpy as np
        a = np.asarray(arr, dtype=float)
        return (a - a.mean()) / (a.std() + 1e-6)

    def _apply_cross_encoder(self, resumes: List[ParsedResume], job_description: str):
        from .cross_encoder_reranker import CrossEncoderReranker
        # Preselect top-200 by the 3-signal blend
        base_blend = []
        for r in resumes:
            base_blend.append(0.5*r.scores.semantic_score +
                              0.3*r.scores.tfidf_taxonomy_score +
                              0.2*r.scores.tfidf_section_score)
        # Pick top-200 indexes
        idxs = sorted(range(len(resumes)), key=lambda i: base_blend[i], reverse=True)[:200]
        subset = [resumes[i] for i in idxs]
        texts = [' '.join(r.sections.values()) for r in subset]
        ce = CrossEncoderReranker()
        ce_scores = ce.score_pairs(job_description, texts)
        for r, s in zip(subset, ce_scores):
            setattr(r.scores, 'cross_encoder', s)

        # Compute final_pre_llm across ALL resumes (CE missing -> use min CE)
        all_ce = [getattr(r.scores, 'cross_encoder', None) for r in resumes]
        min_ce = min([s for s in all_ce if s is not None], default=0.0)
        all_ce = [s if s is not None else min_ce for s in all_ce]

        z_ce = self._z(all_ce)
        z_sem = self._z([r.scores.semantic_score for r in resumes])
        z_tax = self._z([r.scores.tfidf_taxonomy_score for r in resumes])
        z_sec = self._z([r.scores.tfidf_section_score for r in resumes])

        final_pre = [0.6*z_ce[i] + 0.2*z_sem[i] + 0.15*z_tax[i] + 0.05*z_sec[i]
                     for i in range(len(resumes))]

        # Keep top-50 for LLM stage
        top50_idx = sorted(range(len(resumes)), key=lambda i: final_pre[i], reverse=True)[:50]
        # Store display-normalized 0-100 for UI
        import numpy as np
        arr = np.asarray(final_pre, dtype=float)
        min_v, max_v = float(arr.min()), float(arr.max())
        rng = max_v - min_v if max_v > min_v else 1.0
        display = [float(100 * (v - min_v) / rng) for v in final_pre]
        for i, r in enumerate(resumes):
            r.scores.final_pre_llm_display = display[i]
        return top50_idx, final_pre
    
    def _generate_llm_rankings(self, resumes: List[ParsedResume], job_description: str) -> List[Dict[str, Any]]:
        """Generate LLM-based rankings for all resumes."""
        try:
            if not self.llm_ranker or not getattr(self.llm_ranker, 'enabled', False):
                logger.warning("LLM ranker not initialized, using fallback ranking")
                return self._fallback_ranking(resumes, job_description)
            
            logger.info("Generating LLM-based rankings...")
            
            # Prepare candidate data for LLM ranking
            candidates_data = []
            for resume in resumes:
                candidate_data = {
                    'candidate_id': resume.id,
                    'resume_data': {
                        'sections': resume.sections,
                        'parsed': resume.parsed,
                        'meta': resume.meta
                    },
                    'computed_scores': {
                        'tfidf_section_score': resume.scores.tfidf_section_score,
                        'tfidf_taxonomy_score': resume.scores.tfidf_taxonomy_score,
                        'semantic_score': resume.scores.semantic_score
                    }
                }
                candidates_data.append(candidate_data)
            
            # Generate rankings using LLM
            ranking_results = self.llm_ranker.rank_candidates(job_description, candidates_data)
            
            # Convert to required format
            final_ranking = []
            for result in ranking_results:
                final_ranking.append({
                    'id': result.candidate_id,
                    'rank': result.rank,
                    'reasoning': result.reasoning,
                    'scores_snapshot': {
                        'tfidf_section': next((r.scores.tfidf_section_score for r in resumes if r.id == result.candidate_id), 0.0),
                        'tfidf_taxonomy': next((r.scores.tfidf_taxonomy_score for r in resumes if r.id == result.candidate_id), 0.0),
                        'semantic': next((r.scores.semantic_score for r in resumes if r.id == result.candidate_id), 0.0),
                        'cross_encoder': next((getattr(r.scores, 'cross_encoder', 0.0) for r in resumes if r.id == result.candidate_id), 0.0),
                        'final_pre_llm': next((getattr(r.scores, 'final_pre_llm', 0.0) for r in resumes if r.id == result.candidate_id), 0.0),
                        'final_pre_llm_display': next((getattr(r.scores, 'final_pre_llm_display', 0.0) for r in resumes if r.id == result.candidate_id), 0.0)
                    }
                })
            
            logger.info(f"LLM ranking completed for {len(final_ranking)} candidates")
            return final_ranking
            
        except Exception as e:
            logger.error(f"Error in LLM ranking: {str(e)}")
            logger.info("Falling back to computed score ranking")
            return self._fallback_ranking(resumes, job_description)
    
    def _fallback_ranking(self, resumes: List[ParsedResume], job_description: str) -> List[Dict[str, Any]]:
        """Generate fallback rankings based on computed scores when LLM is not available."""
        logger.info("Using fallback ranking based on computed scores")
        
        # Calculate composite scores for each resume
        scored_resumes = []
        for resume in resumes:
            # Weighted combination of scores
            composite_score = (
                resume.scores.tfidf_section_score * 0.4 +
                resume.scores.tfidf_taxonomy_score * 0.4 +
                resume.scores.semantic_score * 0.2
            )
            scored_resumes.append((resume, composite_score))
        
        # Sort by composite score (highest first)
        scored_resumes.sort(key=lambda x: x[1], reverse=True)
        
        # Generate ranking
        final_ranking = []
        for i, (resume, score) in enumerate(scored_resumes):
            final_ranking.append({
                'id': resume.id,
                'rank': i + 1,
                'reasoning': f"Ranked based on computed scores: TF-IDF Section ({resume.scores.tfidf_section_score:.2f}), TF-IDF Taxonomy ({resume.scores.tfidf_taxonomy_score:.2f}), Semantic ({resume.scores.semantic_score:.2f})",
                'scores_snapshot': {
                    'tfidf_section': resume.scores.tfidf_section_score,
                    'tfidf_taxonomy': resume.scores.tfidf_taxonomy_score,
                    'semantic': resume.scores.semantic_score
                }
            })
        
        return final_ranking
    
    def _generate_fallback_rankings(self, resumes: List[ParsedResume]) -> List[Dict[str, Any]]:
        """Generate fallback rankings based on computed scores."""
        # Sort by weighted average of scores
        scored_resumes = []
        for resume in resumes:
            weighted_score = (
                0.4 * resume.scores.tfidf_section_score +
                0.3 * resume.scores.tfidf_taxonomy_score +
                0.3 * resume.scores.semantic_score
            )
            scored_resumes.append((resume, weighted_score))
        
        # Sort by score (descending)
        scored_resumes.sort(key=lambda x: x[1], reverse=True)
        
        # Generate ranking
        final_ranking = []
        for i, (resume, score) in enumerate(scored_resumes):
            final_ranking.append({
                'id': resume.id,
                'rank': i + 1,
                'reasoning': f"Ranked based on computed scores: TF-IDF Section ({resume.scores.tfidf_section_score:.2f}), TF-IDF Taxonomy ({resume.scores.tfidf_taxonomy_score:.2f}), Semantic ({resume.scores.semantic_score:.2f})",
                'scores_snapshot': {
                    'tfidf_section': resume.scores.tfidf_section_score,
                    'tfidf_taxonomy': resume.scores.tfidf_taxonomy_score,
                    'semantic': resume.scores.semantic_score
                }
            })
        
        return final_ranking
    
    def _assemble_output(self, resumes: List[ParsedResume], final_ranking: List[Dict[str, Any]], job_description: str, jd_criteria: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Assemble the final output JSON."""
        # Convert resumes to required format
        resumes_output = []
        for resume in resumes:
            resume_output = {
                'id': resume.id,
                'scores': asdict(resume.scores),
                'matched_skills': resume.matched_skills,
                'parsed': resume.parsed,
                'meta': resume.meta
            }
            resumes_output.append(resume_output)
        
        # Generate batch summary
        batch_summary = self._generate_batch_summary(resumes, final_ranking)
        
        # Assemble final output
        jd_digest = {
            'tokens_summary': f"Job description with {len(job_description.split())} words",
            'top_skills': self._extract_top_skills(job_description),
            'embedding_info': {'model': 'sentence-bert' if self.semantic_embedding and self.semantic_embedding.model_available else 'fallback', 'dim': 384 if self.semantic_embedding and self.semantic_embedding.model_available else 0}
        }
        if isinstance(jd_criteria, dict) and jd_criteria:
            jd_digest['criteria'] = jd_criteria

        output = {
            'job_description_digest': jd_digest,
            'resumes': resumes_output,
            'final_ranking': final_ranking,
            'batch_summary': batch_summary
        }
        
        return output
    
    def _generate_batch_summary(self, resumes: List[ParsedResume], final_ranking: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate batch summary information."""
        # Top candidates
        top_candidates = [r['id'] for r in final_ranking[:3]] if final_ranking else []
        
        # Common gaps (simplified analysis)
        common_gaps = []
        if resumes:
            # Check for common missing skills
            missing_skills = set()
            for resume in resumes:
                if not resume.sections['skills'].strip():
                    missing_skills.add('missing skills section')
                if not resume.sections['experience'].strip():
                    missing_skills.add('missing experience section')
            
            common_gaps = list(missing_skills)
        
        # Processing notes
        notes = []
        failed_count = sum(1 for r in resumes if r.meta.get('processing_status') == 'failed')
        if failed_count > 0:
            notes.append(f"{failed_count} resumes failed to process")
        
        if not notes:
            notes.append("All resumes processed successfully")
        
        return {
            'top_candidates': top_candidates,
            'common_gaps': common_gaps,
            'notes': '; '.join(notes)
        }
    
    def _extract_top_skills(self, job_description: str) -> List[Dict[str, str]]:
        """Extract top skills from job description."""
        top_skills = []
        for skill_id, surface_forms in self.skill_taxonomy.items():
            for surface_form in surface_forms:
                if surface_form.lower() in job_description.lower():
                    top_skills.append({
                        'skill_id': skill_id,
                        'label': surface_forms[0].title()
                    })
                    break  # Only add each skill once
        
        return top_skills[:10]  # Limit to top 10 

    def _generate_classical_rankings(self, resumes: List[ParsedResume], job_description: str) -> List[Dict[str, Any]]:
        """Generate classical composite rankings without LLM."""
        from .text_processor import _norm, _std
        
        # Use absolute thresholds instead of batch normalization
        # This prevents unrelated resumes from getting artificially high scores
        TFIDF_THRESHOLD = 0.1  # Minimum TF-IDF score to be considered relevant
        SEMANTIC_THRESHOLD = 0.3  # Minimum semantic score to be considered relevant
        
        # Compute composite scores
        for resume in resumes:
            # Get raw scores
            S = getattr(resume.scores, 'tfidf_section_score', 0.0)
            K = getattr(resume.scores, 'tfidf_taxonomy_score', 0.0)
            B = getattr(resume.scores, 'semantic_score', 0.0)
            
            # Apply relevance thresholds - penalize low scores heavily
            if S < TFIDF_THRESHOLD:
                S = S * 0.1  # Heavy penalty for low TF-IDF
            if K < TFIDF_THRESHOLD:
                K = K * 0.1  # Heavy penalty for low skill TF-IDF
            if B < SEMANTIC_THRESHOLD:
                B = B * 0.2  # Penalty for low semantic similarity
            
            # Normalize to [0,1] using fixed ranges instead of batch min/max
            S_norm = min(1.0, max(0.0, S / 0.5))  # Expect max TF-IDF around 0.5
            K_norm = min(1.0, max(0.0, K / 0.5))  # Expect max skill TF-IDF around 0.5
            B_norm = min(1.0, max(0.0, B))  # Semantic scores already in [0,1]
            
            # Compute consistency bonus only for reasonably good scores
            available_scores = [s for s in [S_norm, K_norm, B_norm] if s > 0.1]
            if len(available_scores) >= 2:
                consistency_weight = 0.05  # Reduced bonus weight
                std_val = _std(available_scores)
                bonus = max(0.0, 1.0 - std_val) * consistency_weight
            else:
                bonus = 0.0
            
            # Weighted composite with emphasis on TF-IDF (lexical matching)
            wS, wK, wB = 0.5, 0.3, 0.2  # Favor lexical matching over semantic
            composite = wS * S_norm + wK * K_norm + wB * B_norm + bonus
            
            # Additional penalty for very low scores across all metrics
            if S < 0.01 and K < 0.01 and B < 0.2:
                composite *= 0.1  # Heavy penalty for completely unrelated content
            
            # Set final scores
            resume.scores.final_pre_llm = composite
            resume.scores.final_pre_llm_display = round(100 * max(0.0, min(1.0, composite)), 2)
            
            # Debug logging
            logger.info(f"Resume {resume.id}: S={S:.4f}->{S_norm:.4f}, K={K:.4f}->{K_norm:.4f}, B={B:.4f}->{B_norm:.4f}, composite={composite:.4f}")
        
        # Sort by final_pre_llm descending
        resumes.sort(key=lambda r: getattr(r.scores, 'final_pre_llm', 0.0), reverse=True)
        
        # Generate ranking results
        ranking_results = []
        for i, resume in enumerate(resumes):
            ranking_results.append({
                'id': resume.id,
                'rank': i + 1,
                'reasoning': f"Classical composite score: {getattr(resume.scores, 'final_pre_llm_display', 0.0):.2f}",
                'scores_snapshot': {
                    'tfidf_section': getattr(resume.scores, 'tfidf_section_score', 0.0),
                    'tfidf_taxonomy': getattr(resume.scores, 'tfidf_taxonomy_score', 0.0),
                    'semantic': getattr(resume.scores, 'semantic_score', 0.0),
                    'final_pre_llm': getattr(resume.scores, 'final_pre_llm', 0.0),
                    'final_pre_llm_display': getattr(resume.scores, 'final_pre_llm_display', 0.0)
                }
            })
        
        return ranking_results

    def _compute_fallback_score(self, jd_content: str, resume_content: str) -> float:
        """Compute a fallback similarity score when TF-IDF is not available."""
        try:
            # Simple text similarity as fallback
            return self._compute_text_similarity(jd_content, resume_content)
        except Exception as e:
            logger.warning(f"Fallback scoring failed: {e}")
            return 0.0 