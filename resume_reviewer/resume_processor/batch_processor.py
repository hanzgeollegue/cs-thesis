import json
import logging
import os
import uuid
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field, is_dataclass
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from concurrent.futures import ThreadPoolExecutor
import time
from contextlib import contextmanager
from datetime import datetime
import calendar

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
        USE_ENHANCED_NLG,
        PARSE_CONCURRENCY,
        BATCH_TIMEOUT_SEC,
        ENABLE_MATCH_DETECTION,
        REQUIRE_MATCH_FOR_CE,
        ENABLE_META_COMBINER,
        META_RIDGE_L2,
        META_MIN_LABELS,
    )
except ImportError:
    # Fallback to absolute imports (package or standalone context)
    try:
        from resume_processor.enhanced_pdf_parser import PDFParser
        from resume_processor.llm_ranker import LLMRanker, CEReranker
        from resume_processor.text_processor import SemanticEmbedding
        from resume_processor.text_processor import (
            clear_tfidf_caches,
            scrub_pii_and_boilerplate,
            preprocess_text_for_dense_models,
            chunk_text_for_sbert,
        )
        from resume_processor.config import (
            get_openai_config,
            get_llm_config,
            validate_config,
            USE_LLM_RANKER,
            USE_ENHANCED_NLG,
            PARSE_CONCURRENCY,
            BATCH_TIMEOUT_SEC,
            ENABLE_MATCH_DETECTION,
            REQUIRE_MATCH_FOR_CE,
            ENABLE_META_COMBINER,
            META_RIDGE_L2,
            META_MIN_LABELS,
        )
    except ImportError:
        try:
            from enhanced_pdf_parser import PDFParser
            from llm_ranker import LLMRanker, CEReranker
            from text_processor import SemanticEmbedding
            from text_processor import (
                clear_tfidf_caches,
                scrub_pii_and_boilerplate,
                preprocess_text_for_dense_models,
                chunk_text_for_sbert,
            )
            from config import (
                get_openai_config,
            get_llm_config,
            validate_config,
            USE_LLM_RANKER,
            USE_ENHANCED_NLG,
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
            'experience': [
                'experience', 'work experience', 'employment', 'career history', 'work history', 'professional experience',
                'career experience', 'employment history', 'relevant experience', 'seminars training experiences',
                'training experiences', 'training experience'
            ],
            'skills': [
                'skills', 'technical skills', 'technology skills', 'core competencies', 'competencies', 'expertise',
                'proficiencies', 'tools', 'technology stack', 'tech stack', 'toolset', 'technical proficiencies'
            ],
            'education': [
                'education', 'academic background', 'qualifications', 'academic qualifications', 'education history',
                'educational background', 'educational attainment', 'college', 'university', 'high school',
                'secondary education', 'tertiary education', 'academic history', 'academic profile'
            ],
            'misc': [
                'summary', 'professional summary', 'profile', 'objective', 'career objective', 'career objectives',
                'projects', 'certifications', 'awards', 'languages', 'interests', 'volunteer', 'leadership',
                'personal information', 'personal details', 'contact information', 'references'
            ]
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
    
    def clear_all_caches(self):
        """Clear all caches: PDF parser cache, TF-IDF caches, and Cross-Encoder cache."""
        try:
            # Clear TF-IDF in-memory caches
            clear_tfidf_caches()
            logger.info("[CACHE] Cleared TF-IDF vectorizer caches")
            
            # Clear PDF parser cache files
            if hasattr(self, 'pdf_parser') and self.pdf_parser:
                cache_dir = getattr(self.pdf_parser, 'cache_dir', None)
                if cache_dir and os.path.exists(cache_dir):
                    cache_files = [f for f in os.listdir(cache_dir) if f.endswith('_structured.json')]
                    for cache_file in cache_files:
                        try:
                            os.remove(os.path.join(cache_dir, cache_file))
                        except Exception as e:
                            logger.warning(f"[CACHE] Failed to remove {cache_file}: {e}")
                    logger.info(f"[CACHE] Cleared {len(cache_files)} PDF parser cache files from {cache_dir}")
            
            # Clear Cross-Encoder cache if available
            if hasattr(self, 'cross_encoder') and self.cross_encoder:
                if hasattr(self.cross_encoder, 'cache'):
                    try:
                        import threading
                        cache_lock = getattr(self.cross_encoder, 'cache_lock', None)
                        if cache_lock:
                            with cache_lock:
                                cache_size = len(self.cross_encoder.cache)
                                self.cross_encoder.cache.clear()
                                logger.info(f"[CACHE] Cleared Cross-Encoder cache ({cache_size} entries)")
                        else:
                            cache_size = len(self.cross_encoder.cache)
                            self.cross_encoder.cache.clear()
                            logger.info(f"[CACHE] Cleared Cross-Encoder cache ({cache_size} entries)")
                    except Exception as e:
                        logger.warning(f"[CACHE] Error clearing Cross-Encoder cache: {e}")
            
            logger.info("[CACHE] All caches cleared successfully")
        except Exception as e:
            logger.warning(f"[CACHE] Error clearing caches: {e}")

    def process_batch(self, resumes: List[str], job_description: str, jd_criteria: Optional[Dict[str, Any]] = None, clear_cache: bool = False) -> Dict[str, Any]:
        """Main batch processing pipeline.
        
        Args:
            resumes: List of file paths to PDF resumes
            job_description: Job description text
            jd_criteria: Optional pre-extracted JD criteria
            clear_cache: If True, clear all caches before processing
        """
        timing_bucket = {}
        batch_start_time = time.perf_counter()
        
        try:
            # Clear caches if requested
            if clear_cache:
                self.clear_all_caches()
            
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
                        parsed_resumes[i].scores.skill_tfidf = skill_tfidf_scores[i] if i < len(skill_tfidf_scores) else 0.0
                        parsed_resumes[i].scores.sbert_score = float(result.sbert_score)
                        parsed_resumes[i].scores.ce_score = float(result.ce_score)
                        # Store legacy aliases for backward compatibility
                        parsed_resumes[i].scores.tfidf_section_score = parsed_resumes[i].scores.section_tfidf
                        parsed_resumes[i].scores.tfidf_taxonomy_score = parsed_resumes[i].scores.skill_tfidf
                        parsed_resumes[i].scores.semantic_score = parsed_resumes[i].scores.sbert_score
                        parsed_resumes[i].scores.cross_encoder = parsed_resumes[i].scores.ce_score
                        # Store CE debug invariants if available (from EnhancedCrossEncoder)
                        if hasattr(result, 'ce_invariants') and result.ce_invariants:
                            setattr(parsed_resumes[i].scores, 'ce_entered_builder', result.ce_invariants.get('entered_ce_builder', False))
                            setattr(parsed_resumes[i].scores, 'ce_evidence_tokens', result.ce_invariants.get('evidence_tokens_available', 0))
                            setattr(parsed_resumes[i].scores, 'ce_pairs_before', result.ce_invariants.get('num_pairs_before_filter', 0))
                            setattr(parsed_resumes[i].scores, 'ce_pairs_after', result.ce_invariants.get('num_pairs_after_filter', 0))
                            setattr(parsed_resumes[i].scores, 'ce_raw', result.ce_invariants.get('ce_raw_score', 0.0))
                            setattr(parsed_resumes[i].scores, 'ce_blocked_reason', result.ce_invariants.get('ce_blocked_reason', ''))
                            setattr(parsed_resumes[i].scores, 'ce_timed_out', result.ce_invariants.get('ce_timed_out', False))
                        elif isinstance(result, dict) and 'ce_invariants' in result:
                            # Handle dict results from enhanced_cross_encoder
                            ce_inv = result.get('ce_invariants', {})
                            setattr(parsed_resumes[i].scores, 'ce_entered_builder', ce_inv.get('entered_ce_builder', False))
                            setattr(parsed_resumes[i].scores, 'ce_evidence_tokens', ce_inv.get('evidence_tokens_available', 0))
                            setattr(parsed_resumes[i].scores, 'ce_pairs_before', ce_inv.get('num_pairs_before_filter', 0))
                            setattr(parsed_resumes[i].scores, 'ce_pairs_after', ce_inv.get('num_pairs_after_filter', 0))
                            setattr(parsed_resumes[i].scores, 'ce_raw', ce_inv.get('ce_raw_score', 0.0))
                            setattr(parsed_resumes[i].scores, 'ce_blocked_reason', ce_inv.get('ce_blocked_reason', ''))
                            setattr(parsed_resumes[i].scores, 'ce_timed_out', ce_inv.get('ce_timed_out', False))
                        else:
                            # Default values for CEReranker (doesn't track these)
                            setattr(parsed_resumes[i].scores, 'ce_entered_builder', None)
                            setattr(parsed_resumes[i].scores, 'ce_evidence_tokens', None)
                            setattr(parsed_resumes[i].scores, 'ce_pairs_before', None)
                            setattr(parsed_resumes[i].scores, 'ce_pairs_after', None)
                            setattr(parsed_resumes[i].scores, 'ce_raw', None)
                            setattr(parsed_resumes[i].scores, 'ce_blocked_reason', None)
                            setattr(parsed_resumes[i].scores, 'ce_timed_out', None)
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
                            # Absolute scaling for SBERT - adjusted thresholds
                            # 0.656 raw should normalize to ~0.65-0.70, not 0.37
                            if sb_raw >= 0.80:
                                r.scores.semantic_norm = 1.0
                            elif sb_raw >= 0.65:
                                # Linear scaling from 0.65-0.80: maps 0.65→0.65, 0.80→1.0
                                r.scores.semantic_norm = 0.65 + (sb_raw - 0.65) / 0.15 * 0.35
                            elif sb_raw >= 0.50:
                                # Linear scaling from 0.50-0.65: maps 0.50→0.50, 0.65→0.65
                                r.scores.semantic_norm = 0.50 + (sb_raw - 0.50) / 0.15 * 0.15
                            else:
                                # For scores < 0.50, scale proportionally
                                r.scores.semantic_norm = max(0.0, sb_raw / 0.50 * 0.50)
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
                               f"CE_raw={ce_raw:.3f} -> CE_norm={r.scores.ce_norm:.3f} "
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
                
                # Initialize taxonomy for coverage calculation (needed regardless of ENABLE_MATCH_DETECTION)
                taxonomy = None
                try:
                    from .text_processor import SkillTaxonomy
                    taxonomy = SkillTaxonomy()
                except Exception as e:
                    logger.warning(f"Failed to initialize SkillTaxonomy for coverage calculation: {e}")
                    taxonomy = None
                
                for r in parsed_resumes:
                    filename = r.meta.get('source_file', 'unknown')
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
                                # Search all sections: experience, skills, education, misc, and also parsed projects
                                section_texts = [str(v) for v in r.sections.values()]
                                # Also include parsed projects text if available
                                if isinstance(r.parsed, dict):
                                    projects = r.parsed.get('projects', [])
                                    if projects:
                                        for proj in projects:
                                            if isinstance(proj, dict):
                                                proj_text = ' '.join([
                                                    str(proj.get('name', '')),
                                                    str(proj.get('summary', '')),
                                                    ' '.join(proj.get('technologies', []))
                                                ])
                                                if proj_text.strip():
                                                    section_texts.append(proj_text)
                                all_text = ' '.join(section_texts)
                                
                                # Try confidence-aware matching first, fallback to exact matching
                                try:
                                    # Use confidence-aware matching if fuzzy matching is enabled
                                    if taxonomy.fuzzy_enabled:
                                        matched_with_confidence = taxonomy.get_matched_required_skills_with_confidence(
                                            all_text, must_have, min_confidence=0.7
                                        )
                                        if matched_with_confidence:
                                            # Weight coverage by confidence scores
                                            total_confidence = sum(conf for _, conf in matched_with_confidence)
                                            coverage = total_confidence / len(must_have) if len(must_have) > 0 else 1.0
                                            matched_required = [skill for skill, _ in matched_with_confidence]
                                        else:
                                            # No matches above threshold, use exact matching
                                            matched_required = taxonomy.get_matched_required_skills(all_text, must_have)
                                            coverage = len(matched_required) / len(must_have) if len(must_have) > 0 else 1.0
                                    else:
                                        # Fuzzy matching disabled, use exact matching
                                        matched_required = taxonomy.get_matched_required_skills(all_text, must_have)
                                        coverage = len(matched_required) / len(must_have) if len(must_have) > 0 else 1.0
                                except Exception as conf_e:
                                    # Fallback to exact matching if confidence matching fails
                                    logger.warning(f"Confidence-aware matching failed, using exact matching: {conf_e}")
                                    matched_required = taxonomy.get_matched_required_skills(all_text, must_have)
                                    coverage = len(matched_required) / len(must_have) if len(must_have) > 0 else 1.0
                                
                                # Debug logging for skill matching
                                if coverage == 0.0 and len(must_have) > 0:
                                    logger.debug(f"[SKILL_COVERAGE] {filename}: coverage=0.0, "
                                               f"must_have={len(must_have)} skills, "
                                               f"matched={len(matched_required)}, "
                                               f"text_length={len(all_text)}")
                                    # Raw text fallback (pre-Nov fixes behavior)
                                    raw_text = ''
                                    if isinstance(r.parsed, dict):
                                        raw_text = r.parsed.get('raw_text', '') or ''
                                    if not raw_text:
                                        raw_text = ' '.join(section_texts)
                                    if raw_text.strip():
                                        fallback_matches = taxonomy.get_matched_required_skills(raw_text, must_have)
                                        if fallback_matches:
                                            matched_required = fallback_matches
                                            coverage = len(matched_required) / len(must_have) if len(must_have) > 0 else 1.0
                                            logger.info(f"[SKILL_COVERAGE] Raw text fallback rescued coverage with {len(matched_required)} skills for {filename}")
                                        else:
                                            logger.debug(f"[SKILL_COVERAGE] Raw text fallback found no matches for {filename}")
                                
                                if coverage < threshold:
                                    # Apply gamma=2 penalty: final = base * (coverage^2)
                                    score = base_composite * (coverage ** 2)
                                    gate_reason = f"skills_coverage_{coverage:.2f}<{threshold:.2f}"
                                else:
                                    score = base_composite
                            except Exception as e:
                                logger.warning(f"Error computing skill coverage for {filename}: {e}")
                                score = base_composite
                        else:
                            score = base_composite
                        
                        # Experience years gate (heuristic) - enhanced with structured parsing
                        exp_text = r.sections.get('experience', '') or ''
                        yrs_from_text = 0.0
                        try:
                            for m in re.findall(r"(\d+)\s+year", exp_text.lower()):
                                yrs_from_text = max(yrs_from_text, float(m))
                        except Exception:
                            yrs_from_text = 0.0

                        structured_entries = r.parsed.get('experience', []) if isinstance(r.parsed, dict) else []
                        structured_years = self._estimate_years_from_structured_experience(structured_entries)
                        yrs = max(yrs_from_text, structured_years)

                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(
                                f"[EXPERIENCE] {filename}: text_years={yrs_from_text:.2f}, "
                                f"struct_years={structured_years:.2f}, used={yrs:.2f}"
                            )

                        if min_years > 0:
                            try:
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
                        exp_content = str(exp_text).strip()
                        has_structured_experience = bool(structured_entries)
                        has_experience = (structured_years >= 0.25) or has_structured_experience or bool(exp_content)
                        r.scores.has_match_experience = has_experience
                        setattr(r.scores, 'estimated_experience_years', yrs)
                        
                        # Perfect-fit floor: if high coverage and strong signals, enforce floor
                        try:
                            tfidf_n = getattr(r.scores, 'tfidf_norm', 0.0)
                            ce_n = getattr(r.scores, 'ce_norm', 0.0)
                            # Relaxed criteria: high coverage (>=0.85) OR perfect coverage (1.0)
                            # AND (high CE score OR high TF-IDF OR experience match)
                            has_high_coverage = coverage >= 0.85
                            has_perfect_coverage = coverage == 1.0
                            has_strong_signal = (ce_n >= 0.9) or (tfidf_n >= 0.5) or r.scores.has_match_experience
                            
                            if (has_perfect_coverage or has_high_coverage) and has_strong_signal:
                                # Set floor based on coverage level
                                if has_perfect_coverage:
                                    score = max(score, 0.90)
                                elif coverage >= 0.90:
                                    score = max(score, 0.85)
                                else:  # coverage >= 0.85
                                    score = max(score, 0.80)
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
                        r.scores.has_match_experience = False
                        setattr(r.scores, 'estimated_experience_years', 0.0)
                    
                    r.scores.final_score = score
                    r.scores.final_score_display = round(100.0 * score, 2)
                    # Keep legacy display synchronized
                    r.scores.final_pre_llm = score
                    r.scores.final_pre_llm_display = r.scores.final_score_display
                    
                    # Calculate missing skills: normalize must_have to canonical forms for comparison
                    if must_have and taxonomy:
                        normalized_must_have = set([taxonomy.normalize_skill(skill) for skill in must_have])
                        missing_skills = list(normalized_must_have - set(matched_required))
                    elif must_have:
                        # Fallback if taxonomy not available: use original comparison
                        missing_skills = list(set(must_have) - set(matched_required))
                    else:
                        missing_skills = []
                    logger.info(f"[META] {filename} coverage={len(matched_required)}/{len(must_have) if must_have else 0} "
                               f"tf={r.scores.tfidf_norm:.3f} sb={r.scores.semantic_norm:.3f} ce={r.scores.ce_norm:.3f} "
                               f"ms={int(bool(r.scores.has_match_skills))} me={int(bool(r.scores.has_match_experience))} "
                               f"yrs={getattr(r.scores, 'estimated_experience_years', 0.0):.2f} "
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
            structured_data = self.pdf_parser._extract_structured_data(resume_path)
            
            if not structured_data.get('success', False):
                logger.warning(f"Failed to parse {resume_path}: {structured_data.get('error', 'Unknown error')}")
                fallback_structured = self._fallback_structured_parse(resume_path)
                if not fallback_structured:
                    return None
                structured_data = fallback_structured

            parsed_resume = self._build_parsed_resume(structured_data, resume_path)
            if not parsed_resume:
                logger.warning(f"No sections produced for {resume_path}; attempting fallback parser")
                fallback_structured = self._fallback_structured_parse(resume_path)
                if not fallback_structured:
                    return None
                parsed_resume = self._build_parsed_resume(fallback_structured, resume_path)
                if not parsed_resume:
                    return None

            raw_text_len = len((parsed_resume.parsed.get('raw_text') or '').strip())
            if raw_text_len == 0:
                logger.warning(f"No text extracted from {resume_path}; attempting fallback parser")
                fallback_structured = self._fallback_structured_parse(resume_path)
                if fallback_structured:
                    parsed_resume = self._build_parsed_resume(fallback_structured, resume_path)
                    if parsed_resume:
                        logger.info(f"Fallback parser succeeded for {resume_path}")

            return parsed_resume
            
        except Exception as e:
            logger.error(f"Error parsing {resume_path}: {str(e)}")
            return None

    def _build_parsed_resume(self, structured_data: Dict[str, Any], resume_path: str) -> Optional[ParsedResume]:
        """Construct a ParsedResume object from structured parser output."""
        if not structured_data or not structured_data.get('sections'):
            return None

        normalized_sections = self._normalize_sections(structured_data['sections'], structured_data)

        resume_id = str(uuid.uuid4())
        filename = os.path.basename(resume_path)

        structured_meta = structured_data.get('meta') or {}
        resume_meta = dict(structured_meta)
        resume_meta.setdefault("source_file", filename)
        resume_meta.setdefault("processing_status", structured_data.get('processing_status', 'completed'))

        pages_processed = resume_meta.get("pages_processed")
        pages_total = resume_meta.get("pages_total")
        if "pages" not in resume_meta:
            if pages_processed is not None:
                resume_meta["pages"] = pages_processed
            elif pages_total is not None:
                resume_meta["pages"] = pages_total
            else:
                resume_meta["pages"] = len(structured_data.get('layout_metadata', {}).get('text_elements', []))

        canonical_lengths = {
            key: len(normalized_sections.get(key, '').strip())
            for key in ('experience', 'skills', 'education')
        }
        resume_meta['canonical_section_lengths'] = canonical_lengths
        if all(length == 0 for length in canonical_lengths.values()):
            resume_meta['parsing_ok'] = False
            resume_meta['parse_reason'] = resume_meta.get('parse_reason') or 'no_canonical_sections'
        else:
            resume_meta.setdefault('parsing_ok', True)
            resume_meta.setdefault('parse_reason', 'ok')

        parsed_resume = ParsedResume(
            id=resume_id,
            sections=normalized_sections,
            meta=resume_meta,
            scores=ResumeScores(0.0, 0.0, 0.0),
            matched_skills=[],
            parsed=self._extract_parsed_data(structured_data, normalized_sections)
        )
        return parsed_resume

    def _fallback_structured_parse(self, resume_path: str) -> Optional[Dict[str, Any]]:
        """Fallback text extraction using PyMuPDF when primary parser yields no content."""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            logger.error("Fallback parsing requires PyMuPDF (fitz); module not available.")
            return None

        try:
            doc = fitz.open(resume_path)
        except Exception as e:
            logger.error(f"Fallback parser failed to open {resume_path}: {e}")
            return None

        content_lines: List[str] = []
        page_count = 0
        try:
            page_count = doc.page_count
            for page in doc:
                text = page.get_text("text") or ""
                if not text:
                    continue
                for line in text.splitlines():
                    cleaned = line.strip()
                    if cleaned:
                        content_lines.append(cleaned)
        finally:
            doc.close()

        if not content_lines:
            logger.warning(f"Fallback parser found no text in {resume_path}")
            return None

        sections = [{
            'header': 'Resume Content',
            'content': content_lines
        }]
        raw_text_length = sum(len(line) for line in content_lines)
        page_count = len(content_lines)  # approximate if actual count unavailable

        meta = {
            'parsing_ok': raw_text_length > 0,
            'parse_reason': 'fallback_simple' if raw_text_length > 0 else 'no_content',
            'pages_total': page_count,
            'pages_processed': page_count,
            'extracted_chars': raw_text_length,
            'section_count': len(sections),
            'source_file': os.path.basename(resume_path),
            'processing_status': 'fallback'
        }

        return {
            'success': True,
            'sections': sections,
            'summary': {},
            'layout_metadata': {},
            'error': None,
            'processing_status': 'fallback',
            'meta': meta
        }
    
    def _normalize_sections(self, sections: List[Dict[str, Any]], structured_data: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """Normalize section headers to canonical keys with aggregation."""
        # Log input shape for debugging
        logger.info(f"[NORMALIZE] _normalize_sections called with {len(sections) if isinstance(sections, list) else 'non-list'} sections")
        if isinstance(sections, list) and sections:
            top_headers = [s.get('header', '') for s in sections[:5]]
            logger.info(f"[NORMALIZE] Top 5 section headers: {top_headers}")
            total_content = sum(sum(len(str(item)) for item in s.get('content', [])) for s in sections)
            logger.info(f"[NORMALIZE] Total content length: {total_content} chars")
        elif not isinstance(sections, list):
            logger.warning(f"[NORMALIZE] Received non-list input: {type(sections)}")
        
        buckets: Dict[str, List[str]] = {
            'experience': [],
            'skills': [],
            'education': [],
            'misc': []
        }
        taxonomy = None
        taxonomy_error_logged = False

        def _clean_join(items: List[str]) -> str:
            cleaned = []
            for item in items or []:
                if not isinstance(item, str):
                    item = str(item)
                stripped = item.strip()
                if stripped:
                    cleaned.append(stripped)
            return ' '.join(cleaned).strip()

        for section in sections or []:
            header_raw = section.get('header', '')
            header = (header_raw or '').strip().lower()
            content_text = _clean_join(section.get('content', []))
            mapped = self._resolve_canonical_section(header, content_text)
            if mapped not in buckets:
                mapped = 'misc'
            header_skill_text = ''
            if mapped == 'skills':
                if taxonomy is None and not taxonomy_error_logged:
                    try:
                        from .text_processor import SkillTaxonomy
                        taxonomy = SkillTaxonomy()
                    except Exception as taxonomy_error:
                        taxonomy_error_logged = True
                        logger.warning(f"[NORMALIZE] Failed to initialize SkillTaxonomy for header extraction: {taxonomy_error}")
                if taxonomy:
                    header_skills = taxonomy.extract_skills_from_text(header_raw or '')
                    if header_skills:
                        header_skill_text = ' '.join(sorted(set(header_skills)))
            combined_text_parts = []
            if header_skill_text:
                combined_text_parts.append(header_skill_text)
            if content_text:
                combined_text_parts.append(content_text)
            combined_text = ' '.join(combined_text_parts).strip()
            if not combined_text:
                continue
            buckets[mapped].append(combined_text)

        # Augment skills bucket with layout metadata lines that contain skills but were filtered out
        if structured_data:
            layout_texts = ((structured_data.get('layout_metadata') or {}).get('text_elements') or [])
            if layout_texts:
                if taxonomy is None and not taxonomy_error_logged:
                    try:
                        from .text_processor import SkillTaxonomy
                        taxonomy = SkillTaxonomy()
                    except Exception as taxonomy_error:
                        taxonomy_error_logged = True
                        logger.warning(f"[NORMALIZE] Failed to initialize SkillTaxonomy for layout extraction: {taxonomy_error}")
                if taxonomy:
                    existing_skill_tokens = set()
                    if buckets['skills']:
                        try:
                            existing_skill_tokens = set(taxonomy.extract_skills_from_text(' '.join(buckets['skills'])))
                        except Exception as e:
                            logger.debug(f"[NORMALIZE] Failed to extract existing skill tokens: {e}")
                    added_layout_lines = 0
                    for elem in layout_texts:
                        text = str(elem.get('text', '')).strip()
                        if not text:
                            continue
                        # Ignore overly long lines that are likely paragraphs
                        if len(text) > 250:
                            continue
                        try:
                            skills_found = taxonomy.extract_skills_from_text(text)
                        except Exception:
                            skills_found = []
                        if not skills_found:
                            continue
                        if set(skills_found).issubset(existing_skill_tokens):
                            continue
                        buckets['skills'].append(text)
                        existing_skill_tokens.update(skills_found)
                        added_layout_lines += 1
                    if added_layout_lines:
                        logger.info(f"[NORMALIZE] Added {added_layout_lines} layout-derived skill lines to skills bucket")

        return {
            key: ' '.join(values).strip()
            for key, values in buckets.items()
        }

    def _resolve_canonical_section(self, header: str, content: str) -> str:
        """Resolve a section header to a canonical bucket."""
        header = (header or '').strip().lower()
        header_words = set(re.findall(r'\b[a-z]+\b', header))

        # Early heuristics for specific categories
        if any(word in header_words for word in {'skill', 'skills', 'technology', 'technologies', 'tool', 'tools'}):
            return 'skills'
        if any(word in header_words for word in {'education', 'educational', 'academic', 'college', 'university', 'school', 'degree', 'bachelor', 'masters', 'phd'}):
            return 'education'
        if any(word in header_words for word in {'training', 'trainings', 'seminar', 'seminars', 'workshop', 'workshops'}):
            return 'misc'
        if any(word in header_words for word in {'objective', 'objectives', 'summary', 'profile'}):
            return 'misc'
        if any(word in header_words for word in {'personal', 'information', 'contact'}):
            if 'contact' in header_words or 'personal' in header_words:
                return 'misc'

        def matches_alias(alias: str) -> bool:
            alias = alias.lower()
            alias_words = set(re.findall(r'\b[a-z]+\b', alias))
            if alias_words and alias_words.issubset(header_words):
                return True
            pattern = r'\b' + re.escape(alias) + r'\b'
            if re.search(pattern, header):
                return True
            if alias == 'experience' and ('experience' in header_words or 'experiences' in header_words):
                return True
            if alias.endswith('s') and alias[:-1] in header_words:
                return True
            if alias.endswith('ies') and alias[:-3] + 'y' in header_words:
                return True
            return False

        for canonical, aliases in self.section_mapping.items():
            if any(matches_alias(alias) for alias in aliases):
                return canonical

        # Additional heuristics
        if header in {'resume content', 'general information'}:
            # try to guess based on keywords when we only have a generic heading
            if any(kw in content.lower() for kw in ['experience', 'responsibilities', 'work history']):
                return 'experience'
            if any(kw in content.lower() for kw in ['skill', 'technologies', 'tech stack', 'proficient']):
                return 'skills'
            if any(kw in content.lower() for kw in ['university', 'college', 'school', 'bachelor', 'degree', 'education']):
                return 'education'
        return 'misc'
    
    def _extract_parsed_data(
        self,
        structured_data: Dict[str, Any],
        normalized_sections: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Extract structured parsed data from resume."""
        if normalized_sections is None:
            normalized_sections = self._normalize_sections(structured_data.get('sections', []), structured_data)
        else:
            normalized_sections = {
                'experience': (normalized_sections.get('experience') or '').strip(),
                'skills': (normalized_sections.get('skills') or '').strip(),
                'education': (normalized_sections.get('education') or '').strip(),
                'misc': (normalized_sections.get('misc') or '').strip()
            }
        
        normalized_sections.setdefault('misc', '')
        metadata = (structured_data.get('meta') or {}).copy()
        canonical_lengths = {
            key: len(normalized_sections.get(key, '').strip())
            for key in ('experience', 'skills', 'education')
        }
        metadata['canonical_section_lengths'] = canonical_lengths
        if all(length == 0 for length in canonical_lengths.values()):
            metadata['parsing_ok'] = False
            metadata['parse_reason'] = metadata.get('parse_reason') or 'no_canonical_sections'
        else:
            metadata.setdefault('parsing_ok', True)
            metadata.setdefault('parse_reason', 'ok')
        
        try:
            raw_text = self.pdf_parser.extract_plain_text(structured_data)
        except Exception as e:
            logger.warning(f"Failed to generate raw text from structured data: {e}")
            raw_text = ' '.join(normalized_sections.values()).strip()
        
        parsed = {
            'experience': [],
            'skills': [],
            'education': [],
            'projects': [],
            'certifications': [],
            'languages': [],
            'misc': normalized_sections.get('misc', ''),
            'sections': normalized_sections,
            'raw_text': raw_text,
            'metadata': metadata
        }
        fallback_flags: List[str] = []

        # --- Helper functions for structured extraction ---
        month_regex = r'(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)'
        date_range_re = re.compile(
            rf'({month_regex}\.?[\s/-]+\d{{2,4}}|\d{{1,2}}/\d{{2,4}}|\d{{4}})\s*(?:to|-|–|—)\s*(?:{month_regex}\.?[\s/-]+\d{{2,4}}|present|current|\d{{4}})',
            re.IGNORECASE
        )
        single_year_re = re.compile(r'\b(19|20)\d{2}\b')
        # Pattern to detect date-only lines like "March 10" or "March 14, 2025"
        date_only_re = re.compile(
            rf'^{month_regex}\s+\d{{1,2}}(?:\s*,\s*\d{{4}})?$',
            re.IGNORECASE
        )
        contact_noise_re = re.compile(r'(@|https?://|www\.|linkedin\.|github\.|portfolio)', re.IGNORECASE)
        job_keywords = [
            'engineer', 'developer', 'manager', 'director', 'analyst', 'consultant', 'specialist',
            'scientist', 'architect', 'lead', 'intern', 'founder', 'designer', 'technician', 'administrator',
            'supervisor', 'coordinator', 'advisor', 'executive', 'programmer', 'program manager', 'product manager',
            'product owner', 'project manager', 'qa', 'quality assurance', 'support engineer', 'data engineer',
            'data scientist', 'business analyst', 'scrum master', 'devops', 'sre', 'software', 'hardware',
            'security engineer', 'security analyst', 'solutions architect', 'systems engineer', 'network engineer'
        ]
        school_keywords = ['university', 'college', 'school', 'academy', 'institute', 'polytechnic', 'institute of', 'technology', 'polytechnic university']
        degree_keywords = [
            'bachelor', 'master', 'mba', 'bs', 'ba', 'ms', 'phd', 'associate', 'diploma',
            'certificate', 'b.s.', 'b.a.', 'm.s.', 'm.a.', 'high school', 'secondary', 'tertiary',
            'doctorate', 'bsc', 'msc', 'jd', 'md'
        ]

        non_role_keywords = {
            'seminar', 'seminars', 'workshop', 'workshops', 'bootcamp', 'boot camp', 'course', 'courses',
            'training', 'trainings', 'webinar', 'webinars', 'talk', 'talks', 'presentation', 'presentations',
            'orientation', 'tour', 'tours', 'conference', 'conferences', 'symposium', 'summit', 'forum'
        }
        heading_noise = {'role', 'roles', 'responsibilities', 'summary', 'objective', 'projects', 'activities'}
        company_markers = [
            ' inc', ' llc', ' ltd', ' plc', ' gmbh', ' s.r.l', ' company', ' corporation', ' corp', ' co.',
            ' technologies', ' technology', ' systems', ' solutions', ' studios', ' studio', ' labs',
            ' partners', ' holdings', ' consulting', ' international', ' global', ' limited', ' enterprises'
        ]

        training_keywords = non_role_keywords

        experience_promoted: List[str] = []
        experience_dropped: List[str] = []
        education_promoted: List[str] = []
        education_rerouted: List[str] = []
        rerouted_misc: List[str] = []

        def _split_lines(text: str) -> List[str]:
            if not text:
                return []
            parts = re.split(r'[\n\r•\u2022\u25CF]+', text)
            return [part.strip() for part in parts if part and part.strip()]

        def _split_tokens(text: str) -> List[str]:
            if not text:
                return []
            parts = re.split(r'[,;/\n\r•\u2022\u25CF]+', text)
            return [part.strip() for part in parts if part and part.strip()]

        def _collect_section_lines(keywords: List[str]) -> List[str]:
            lines: List[str] = []
            for section in structured_data.get('sections', []):
                header = (section.get('header') or '').lower()
                if any(keyword in header for keyword in keywords):
                    for line in section.get('content', []) or []:
                        cleaned = (line or '').strip()
                        if cleaned:
                            lines.append(cleaned)
            return lines

        def _is_training_like_line(text: str) -> bool:
            lower = text.lower()
            return any(keyword in lower for keyword in training_keywords)

        def _looks_like_role_header(line: str) -> bool:
            text = line.strip()
            if not text or contact_noise_re.search(text):
                return False
            text = re.sub(r'^[\s•\u2022\u25CF\-\*]+', '', text)
            lower = text.lower()
            if _is_training_like_line(lower):
                return False
            
            # Fix 1: Reject date-only lines (e.g., "March 10", "March 14, 2025")
            if date_only_re.match(text):
                return False
            
            keyword_hit = any(word in lower for word in job_keywords)
            has_sep = any(sep in lower for sep in [' - ', ' – ', ' — ', ' | ', ' at ', ' @ ', ', '])
            date_hit = bool(date_range_re.search(text) or single_year_re.search(text))
            company_marker = any(marker in lower for marker in company_markers)
            
            # Fix 1: Require job keyword OR company marker when only dates present
            # Dates alone should not be sufficient
            if date_hit and has_sep:
                # Only allow if we have job keyword or company marker
                if not keyword_hit and not company_marker:
                    return False
            
            if keyword_hit and (has_sep or date_hit or company_marker):
                return True
            # Fix 2: Add title length check - reject very long titles without structure
            if keyword_hit and len(text.split()) >= 2:
                # For longer titles (> 40 chars), require separators/dates/company markers
                if len(text) > 40:
                    if not (has_sep or date_hit or company_marker):
                        return False
                return True
            capitalized_tokens = sum(1 for token in text.split() if token[:1].isupper())
            if keyword_hit and capitalized_tokens >= 2:
                return True
            return False

        def _split_title_company(text: str) -> Tuple[str, str]:
            working = text.strip().strip('-|•')
            lowered = working.lower()
            separators = [' at ', ' @ ', ' - ', ' – ', ' — ', ' | ', ', ']
            for sep in separators:
                idx = lowered.find(sep)
                if idx != -1:
                    title = working[:idx].strip(' ,-|')
                    company = working[idx + len(sep):].strip(' ,-|')
                    return title, company
            return working, ''

        def _line_looks_like_company(text: str) -> bool:
            if not text:
                return False
            lower = text.lower()
            if _is_training_like_line(lower):
                return False
            if any(marker in lower for marker in company_markers):
                return True
            tokens = text.split()
            if len(tokens) <= 6 and sum(1 for token in tokens if token[:1].isupper()) >= 2:
                return True
            return False

        def _extract_dates_from_block(block_lines: List[str]) -> str:
            block_text = ' '.join(block_lines)
            match = date_range_re.search(block_text)
            if match:
                return match.group(0).strip()
            years = single_year_re.findall(block_text)
            if years:
                unique_years = []
                for y in years:
                    year_val = ''.join(y)
                    if year_val not in unique_years:
                        unique_years.append(year_val)
                if len(unique_years) >= 2:
                    return f"{unique_years[0]} - {unique_years[1]}"
                return unique_years[0]
            return ''

        def _build_experience_entries(lines: List[str]) -> List[Dict[str, Any]]:
            entries: List[Dict[str, Any]] = []
            current_block: List[str] = []
            for line in lines:
                cleaned = line.strip()
                if not cleaned:
                    continue
                if _looks_like_role_header(cleaned):
                    if current_block:
                        entries.extend(_finalize_experience_block(current_block))
                        current_block = []
                    current_block.append(cleaned)
                else:
                    if not current_block and _is_training_like_line(cleaned.lower()):
                        rerouted_misc.append(cleaned)
                        experience_dropped.append(cleaned[:80])
                        continue
                    if current_block:
                        current_block.append(cleaned)
            if current_block:
                entries.extend(_finalize_experience_block(current_block))
            return entries

        def _finalize_experience_block(block_lines: List[str]) -> List[Dict[str, Any]]:
            header = block_lines[0]
            if contact_noise_re.search(header):
                return []
            dates = _extract_dates_from_block(block_lines)
            header_without_dates = header.replace(dates, '').strip(' -–—|,') if dates else header
            title, company = _split_title_company(header_without_dates)
            title = title.strip()
            company = company.strip()
            title_lower = title.lower()
            if not title or contact_noise_re.search(title):
                if block_lines:
                    rerouted_misc.append(' '.join(block_lines))
                    experience_dropped.append(header[:80])
                return []
            if _is_training_like_line(title_lower):
                rerouted_misc.append(' '.join(block_lines))
                experience_dropped.append(header[:80])
                return []
            if title_lower.split()[0] in heading_noise:
                rerouted_misc.append(' '.join(block_lines))
                experience_dropped.append(header[:80])
                return []
            
            # Fix 2: Add title length and structure validation
            # Maximum title length check (80 chars)
            if len(title) > 80:
                rerouted_misc.append(' '.join(block_lines))
                experience_dropped.append(header[:80])
                return []
            
            # Detect sentence-style bullets: starts with verb, ends with period, > 50 chars
            description_verbs = ['integrated', 'developed', 'designed', 'implemented', 'created', 'built', 'managed', 'led', 'worked', 'utilized', 'redesigned', 'rebuilt']
            if len(title) > 50 and title.endswith('.') and any(title_lower.startswith(verb + ' ') for verb in description_verbs):
                rerouted_misc.append(' '.join(block_lines))
                experience_dropped.append(header[:80])
                return []
            
            # Negative patterns: reject lines that look like descriptions
            description_patterns = ['with test applications', 'to conduct', 'for future use', 'within the', 'for future']
            if any(pattern in title_lower for pattern in description_patterns):
                rerouted_misc.append(' '.join(block_lines))
                experience_dropped.append(header[:80])
                return []
            
            if not company:
                for ln in block_lines[1:3]:
                    candidate = ln.strip(' -•\u2022\u25CF')
                    if _line_looks_like_company(candidate):
                        company = candidate
                        break
            normalized_title = title_lower
            has_job_keyword = any(keyword in normalized_title for keyword in job_keywords)
            company_lower = company.lower()
            has_company_marker = any(marker in company_lower for marker in company_markers) if company else False
            
            # For titles > 40 chars, require separators/dates/company markers
            if len(title) > 40:
                has_sep = any(sep in header.lower() for sep in [' - ', ' – ', ' — ', ' | ', ' at ', ' @ ', ', '])
                if not (has_sep or dates or has_company_marker or has_job_keyword):
                    rerouted_misc.append(' '.join(block_lines))
                    experience_dropped.append(header[:80])
                    return []
            
            if not has_job_keyword and not dates and not has_company_marker:
                rerouted_misc.append(' '.join(block_lines))
                experience_dropped.append(header[:80])
                return []
            if len(title.split()) < 2 and not has_job_keyword:
                rerouted_misc.append(' '.join(block_lines))
                experience_dropped.append(header[:80])
                return []
            
            # Fix: Merge multi-line bullets intelligently
            bullet_chars = ('•', '●', '▪', '‣', '◦', '∙', '-', '*')
            bullet_pattern = re.compile(rf"^[{re.escape(''.join(bullet_chars))}]+\s*")
            description_verbs = ['integrated', 'developed', 'designed', 'implemented', 'created', 'built', 'managed', 'led', 'worked', 'utilized', 'redesigned', 'rebuilt', 'wrote', 'collaborated', 'monitored', 'set', 'participated']
            
            raw_bullet_lines = [
                ln for ln in block_lines[1:]
                if not contact_noise_re.search(ln) and not _is_training_like_line(ln.lower())
            ]
            
            merged_bullets = []
            current_bullet = None
            
            for line in raw_bullet_lines:
                line_stripped = line.strip()
                if not line_stripped:
                    if current_bullet:
                        merged_bullets.append(current_bullet)
                        current_bullet = None
                    continue
                
                # Check if line starts with bullet marker
                bullet_match = bullet_pattern.match(line_stripped)
                has_bullet_marker = bullet_match is not None
                
                # Extract text without bullet marker
                if has_bullet_marker:
                    text_without_bullet = line_stripped[bullet_match.end():].strip()
                else:
                    text_without_bullet = line_stripped
                
                if not text_without_bullet:
                    continue
                
                # Determine if this is a new bullet or continuation
                is_new_bullet = False
                if has_bullet_marker:
                    is_new_bullet = True
                elif current_bullet is None:
                    # First line without bullet marker - could be a bullet
                    is_new_bullet = True
                else:
                    # Check if line looks like a new bullet (starts with verb, capitalizes first letter)
                    text_lower = text_without_bullet.lower()
                    starts_with_verb = any(text_lower.startswith(verb + ' ') for verb in description_verbs)
                    starts_with_capital = text_without_bullet and text_without_bullet[0].isupper()
                    ends_with_punctuation = text_without_bullet.endswith(('.', '!', '?'))
                    prev_ends_with_punctuation = current_bullet.endswith(('.', '!', '?'))
                    
                    # If previous bullet doesn't end with punctuation, likely continuation
                    if not prev_ends_with_punctuation:
                        # Merge unless this line is very long and clearly a new bullet
                        is_new_bullet = starts_with_verb and len(text_without_bullet) > 40
                    # If previous bullet ended with punctuation, check if this is a new bullet
                    else:
                        # If line is very short (< 20 chars), likely continuation (rare but possible)
                        if len(text_without_bullet) < 20:
                            is_new_bullet = False
                        # If starts with verb and is reasonably long, likely new bullet
                        elif starts_with_verb and len(text_without_bullet) > 20:
                            is_new_bullet = True
                        # If capitalized and reasonably long, likely new bullet
                        elif starts_with_capital and len(text_without_bullet) > 30:
                            is_new_bullet = True
                        # Otherwise, continuation
                        else:
                            is_new_bullet = False
                
                if is_new_bullet:
                    # Save previous bullet if exists
                    if current_bullet:
                        merged_bullets.append(current_bullet)
                    current_bullet = text_without_bullet
                else:
                    # Merge with current bullet
                    if current_bullet:
                        current_bullet = current_bullet + ' ' + text_without_bullet
                    else:
                        current_bullet = text_without_bullet
            
            # Add final bullet if exists
            if current_bullet:
                merged_bullets.append(current_bullet)
            
            bullets = merged_bullets
            
            experience_promoted.append(header[:80])
            return [{
                'title': title.strip(),
                'company': company.strip(),
                'dates': dates,
                'description': ' '.join(block_lines).strip(),
                'bullets': bullets
            }]

        def _looks_like_school_line(line: str) -> bool:
            lower = line.lower()
            return any(keyword in lower for keyword in school_keywords)

        def _looks_like_degree_line(line: str) -> bool:
            lower = line.lower()
            return any(keyword in lower for keyword in degree_keywords)

        def _build_education_entries(lines: List[str]) -> List[Dict[str, Any]]:
            entries: List[Dict[str, Any]] = []
            current_block: List[str] = []
            for line in lines:
                cleaned = line.strip()
                if not cleaned:
                    continue
                lower = cleaned.lower()
                if _is_training_like_line(lower):
                    education_rerouted.append(cleaned[:80])
                    rerouted_misc.append(cleaned)
                    if current_block:
                        entries.append(_finalize_education_block(current_block))
                        current_block = []
                    continue
                if _looks_like_school_line(cleaned) or _looks_like_degree_line(cleaned):
                    if current_block:
                        entries.append(_finalize_education_block(current_block))
                        current_block = []
                    current_block.append(cleaned)
                else:
                    if current_block:
                        current_block.append(cleaned)
            if current_block:
                entries.append(_finalize_education_block(current_block))
            return [entry for entry in entries if entry]

        def _finalize_education_block(block_lines: List[str]) -> Optional[Dict[str, Any]]:
            if not block_lines:
                return None
            school = ''
            degree = ''
            year = ''
            for line in block_lines:
                if not school and _looks_like_school_line(line):
                    school = line.strip()
                if not degree and _looks_like_degree_line(line):
                    degree = line.strip()
                if not year:
                    dates = _extract_dates_from_block([line])
                    if dates:
                        year = dates
            if not school:
                for line in block_lines:
                    candidate = line.strip()
                    if candidate == degree:
                        continue
                    if _line_looks_like_company(candidate):
                        school = candidate
                        break
            if not degree:
                degree_candidates = [ln for ln in block_lines if _looks_like_degree_line(ln)]
                if degree_candidates:
                    degree = degree_candidates[0].strip()
            
            # Fix 3: Validate education field quality
            # Maximum length checks: degree <= 100 chars, school <= 150 chars
            if degree and len(degree) > 100:
                degree = ''
            if school and len(school) > 150:
                school = ''
            
            # Filter address patterns (street names, city/state patterns)
            address_patterns = [
                r'\b(street|st|avenue|ave|road|rd|boulevard|blvd|drive|dr|lane|ln|way|circle|cir)\b',
                r'\b\d+\s+(street|st|avenue|ave|road|rd|boulevard|blvd|drive|dr)\b',
                r'\b(city|state|province|region|country)\b',
                r'\b\d{4,5}\b',  # ZIP codes
                r'\b(ph|tel|phone|contact|address):',  # Contact info markers
            ]
            if degree:
                for pattern in address_patterns:
                    if re.search(pattern, degree, re.IGNORECASE):
                        degree = ''
                        break
            if school:
                for pattern in address_patterns:
                    if re.search(pattern, school, re.IGNORECASE):
                        school = ''
                        break
            
            # Filter objective text patterns
            objective_patterns = [
                r'\b(seeking|looking to|aims to|willing to|eager to|want to|hope to)\b',
                r'\b(entry-level|entry level|position|role|job|opportunity)\b',
                r'\b(where i can|where i will|while contributing|while learning)\b',
                r'\b(student|applicant|candidate)\b',
            ]
            # Filter organization activity patterns (NEW)
            organization_activity_patterns = [
                r'\b(provide|organize|assist|mentor|tutor|volunteer|coordinate)\b',
                r'\b(mentorship|tutoring|social events|activities|outreach|programming)\b',
                r'\b(freshman|underrepresented|majors|transition)\b',
                r'\b(1st year|2nd year|stimulate|creativity|fundamental concepts)\b',
            ]
            if degree:
                for pattern in objective_patterns:
                    if re.search(pattern, degree, re.IGNORECASE):
                        degree = ''
                        break
                # Check organization patterns
                if degree:
                    for pattern in organization_activity_patterns:
                        if re.search(pattern, degree, re.IGNORECASE):
                            degree = ''
                            break
            if school:
                for pattern in objective_patterns:
                    if re.search(pattern, school, re.IGNORECASE):
                        school = ''
                        break
                # Check organization patterns
                if school:
                    for pattern in organization_activity_patterns:
                        if re.search(pattern, school, re.IGNORECASE):
                            school = ''
                            break
            
            # Require at least one clean field (degree OR school) after filtering
            if not degree and not school:
                rerouted_misc.append(' '.join(block_lines))
                education_rerouted.append(block_lines[0][:80])
                return None
            
            # Additional validation: if both fields exist but are very short or suspicious, reject
            if degree and school:
                if len(degree) < 5 and len(school) < 5:
                    rerouted_misc.append(' '.join(block_lines))
                    education_rerouted.append(block_lines[0][:80])
                    return None
            
            education_promoted.append(block_lines[0][:80])
            return {
                'degree': degree,
                'school': school,
                'year': year,
                'description': ' '.join(block_lines).strip()
            }

        def _parse_skill_items(lines: List[str]) -> List[Dict[str, Any]]:
            tokens: List[str] = []
            for line in lines:
                tokens.extend(_split_tokens(line))
            deduped = []
            seen = set()
            for token in tokens:
                lower = token.lower()
                if lower and lower not in seen:
                    deduped.append({'skill': token})
                    seen.add(lower)
            return deduped

        def _collect_lines_for_keywords(keywords: List[str]) -> List[str]:
            return _collect_section_lines(keywords)

        # --- Experience extraction ---
        experience_lines = _collect_lines_for_keywords(['experience', 'work', 'employment'])
        parsed['experience'] = _build_experience_entries(experience_lines)

        # --- Skills extraction ---
        skill_lines = _collect_lines_for_keywords(['skills', 'competencies', 'technologies'])
        parsed['skills'] = _parse_skill_items(skill_lines)

        # --- Education extraction ---
        education_lines = _collect_lines_for_keywords(['education', 'academic', 'school', 'college', 'university'])
        parsed['education'] = _build_education_entries(education_lines)

        # Extract projects (block-based parsing)
        project_section = None
        for section in structured_data.get('sections', []):
            if any(word in section['header'].lower() for word in ['project']):
                project_section = section
                break
        if project_section:
            project_items = []
            seen_titles = set()
            
            # Helper function to detect if a line looks like a project name/header
            def _looks_like_project_header(line: str) -> bool:
                cleaned = line.strip()
                if not cleaned or len(cleaned) < 5:
                    return False
                cleaned_lower = cleaned.lower()
                # Skip generic headers
                if cleaned_lower in ['project', 'projects', 'project:', 'projects:']:
                    return False
                
                # Exclude lines that start with action verbs (these are descriptions, not headers)
                action_verbs = ['developed', 'designed', 'created', 'built', 'implemented', 'formulated', 
                               'wrote', 'presented', 'integrated', 'collaborated', 'managed', 'led',
                               'worked', 'utilized', 'redesigned', 'rebuilt', 'monitored', 'set',
                               'provide', 'organized', 'conducted', 'established', 'maintained']
                first_word = cleaned_lower.split()[0] if cleaned_lower.split() else ''
                if first_word in action_verbs:
                    return False
                
                # Check for date patterns (common in project headers)
                has_date = bool(date_range_re.search(cleaned) or single_year_re.search(cleaned))
                # Check for project-like keywords
                project_keywords = ['project', 'course', 'class', 'design', 'development', 'system', 
                                   'application', 'app', 'tool', 'platform', 'programming', 'personal']
                has_keyword = any(keyword in cleaned_lower for keyword in project_keywords)
                # Check if it looks like a title (capitalized words, reasonable length)
                words = cleaned.split()
                capitalized_count = sum(1 for w in words if w and w[0].isupper())
                is_title_like = capitalized_count >= 3 and len(words) >= 3  # Made more strict: need 3+ capitalized words
                
                # Must have at least 2 of these 3 criteria to be considered a header
                criteria_met = sum([has_date, has_keyword, is_title_like])
                return criteria_met >= 2 and len(cleaned) >= 15  # Increased minimum length
            
            # Helper function to detect if a line is date-only
            def _is_date_only_line(line: str) -> bool:
                cleaned = line.strip()
                # Check for single date patterns (month day, year)
                if date_only_re.match(cleaned):
                    return True
                # Check for date range patterns - if match covers entire line, it's date-only
                date_match = date_range_re.search(cleaned)
                if date_match and date_match.group(0).strip() == cleaned:
                    return True
                # Check for simple year ranges like "2021 - 2022" or "April 2021 - June 2021"
                if re.match(r'^(?:\d{4}|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4})\s*(?:-|to|–|—)\s*(?:\d{4}|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}|present|current)$', cleaned, re.IGNORECASE):
                    return True
                # Check for month-year patterns like "September 2019 – Present"
                if re.match(r'^(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}\s*(?:–|-|—)\s*(?:present|current)$', cleaned, re.IGNORECASE):
                    return True
                return False
            
            # Group lines into project blocks
            current_project = None
            project_lines = project_section.get('content', [])
            
            # Helper to check if line starts with action verb (likely a bullet)
            def _starts_with_action_verb(line: str) -> bool:
                action_verbs = ['developed', 'designed', 'created', 'built', 'implemented', 'formulated', 
                               'wrote', 'presented', 'integrated', 'collaborated', 'managed', 'led',
                               'worked', 'utilized', 'redesigned', 'rebuilt', 'monitored', 'set',
                               'provide', 'organized', 'conducted', 'established', 'maintained',
                               'analyzed', 'configured', 'deployed', 'optimized', 'enhanced']
                first_word = line.lower().split()[0] if line.lower().split() else ''
                return first_word in action_verbs
            
            for i, line in enumerate(project_lines):
                cleaned = line.strip()
                if not cleaned:
                    # Empty line - finalize current project if exists
                    if current_project and current_project.get('name'):
                        project_items.append(current_project)
                        current_project = None
                    continue
                
                cleaned_lower = cleaned.lower()
                # Skip generic headers
                if cleaned_lower in ['project', 'projects', 'project:', 'projects:']:
                    continue
                
                # Check if this is a date-only line
                if _is_date_only_line(cleaned):
                    # Attach to previous project if exists, otherwise skip
                    if current_project:
                        # Add date to project name or summary
                        if not current_project.get('summary'):
                            current_project['summary'] = cleaned
                        else:
                            current_project['summary'] = current_project['summary'] + ' ' + cleaned
                    continue
                
                # Check if this looks like a new project header
                if _looks_like_project_header(cleaned):
                    # Finalize previous project
                    if current_project and current_project.get('name'):
                        project_items.append(current_project)
                    
                    # Start new project
                    # Keep the full line as name (including dates) but limit length
                    full_name = cleaned[:150].strip()
                    # Extract dates from the name line for potential use
                    date_match = date_range_re.search(full_name)
                    dates_text = ''
                    if date_match:
                        dates_text = date_match.group(0).strip()
                        # Keep dates in name, but we can also store them separately if needed
                        name = full_name  # Keep full name with dates
                    else:
                        name = full_name
                    
                    # Check for duplicates (use name without dates for deduplication)
                    name_for_dedup = name
                    if date_match:
                        name_for_dedup = name[:date_match.start()].strip(' ,-–—')
                    name_lower = name_for_dedup.lower()
                    if name_lower not in seen_titles and len(name_for_dedup) >= 5:
                        seen_titles.add(name_lower)
                        current_project = {
                            'name': name,
                            'summary': '',
                            'bullets': [],  # NEW: store bullets separately
                            'technologies': []
                        }
                    else:
                        current_project = None
                else:
                    # This is likely a description line or bullet
                    if current_project:
                        # If line starts with action verb, treat as bullet
                        if _starts_with_action_verb(cleaned):
                            current_project['bullets'].append(cleaned)
                        else:
                            # Add to summary
                            if current_project['summary']:
                                current_project['summary'] = current_project['summary'] + ' ' + cleaned
                            else:
                                current_project['summary'] = cleaned
                    else:
                        # No current project - only create if it looks like a substantial title
                        # Don't create projects from action verb lines
                        if not _starts_with_action_verb(cleaned) and len(cleaned) >= 15:
                            name_lower = cleaned[:100].lower().strip()
                            if name_lower not in seen_titles:
                                seen_titles.add(name_lower)
                                current_project = {
                                    'name': cleaned[:100].strip(),
                                    'summary': '',
                                    'bullets': [],
                                    'technologies': []
                                }
            
            # Finalize last project if exists
            if current_project and current_project.get('name'):
                project_items.append(current_project)
            
            # Clean up project summaries and bullets (limit length, remove excessive whitespace)
            for project in project_items:
                if project.get('summary'):
                    summary = ' '.join(project['summary'].split())
                    project['summary'] = summary[:500]  # Limit summary length
                # Clean up bullets
                if project.get('bullets'):
                    project['bullets'] = [' '.join(bullet.split()) for bullet in project['bullets']]
            
            parsed['projects'] = project_items
        
        # Extract candidate name from summary
        summary = structured_data.get('summary', {})
        candidate_name = summary.get('candidate_name', '')
        
        # Filter out job titles and invalid names
        if candidate_name:
            candidate_name_lower = candidate_name.lower()
            # Reject common job title patterns
            job_title_keywords = [
                'coordinator', 'practicum', 'supervisor', 'manager', 'director', 'analyst',
                'engineer', 'developer', 'specialist', 'consultant', 'architect', 'scientist',
                'lead', 'intern', 'founder', 'designer', 'technician', 'administrator',
                'officer', 'programmer', 'strategist', 'editor', 'writer', 'producer',
                'tester', 'trainer', 'teacher', 'mentor', 'assistant', 'associate',
                'executive', 'president', 'vice', 'chief', 'head', 'senior', 'junior',
                'principal', 'staff'
            ]
            # Check for acronyms (BSIT, IT, CS, etc.)
            words = candidate_name.split()
            has_acronym = any(len(word) <= 4 and word.isupper() for word in words)
            # Check for academic patterns
            academic_patterns = ['bsit', 'bscs', 'bs ', 'ba ', 'ma ', 'ms ', 'phd', 'mba']
            
            if (any(keyword in candidate_name_lower for keyword in job_title_keywords) or
                has_acronym or
                any(acad in candidate_name_lower for acad in academic_patterns)):
                candidate_name = ''  # Reject invalid name
        
        # Set candidate name if valid, otherwise fall back to filename
        if candidate_name:
            parsed['candidate_name'] = candidate_name
        else:
            # Fall back to filename-based name extraction
            source_file = metadata.get('source_file', '')
            if source_file:
                # Extract name from filename (remove extension, clean up)
                base_name = os.path.splitext(source_file)[0]
                # Remove common prefixes/suffixes and clean
                base_name = base_name.replace('_', ' ').replace('-', ' ').strip()
                # Title case
                parsed['candidate_name'] = ' '.join(word.capitalize() for word in base_name.split())
            else:
                parsed['candidate_name'] = 'Unknown Candidate'

        # Fallback population using normalized sections when primary extraction is empty
        normalized_experience = normalized_sections.get('experience', '')
        if not parsed['experience'] and normalized_experience:
            exp_lines = _split_lines(normalized_experience)
            fallback_entries = _build_experience_entries(exp_lines)
            if fallback_entries:
                parsed['experience'] = fallback_entries
                fallback_flags.append('experience')

        normalized_skills = normalized_sections.get('skills', '')
        if not parsed['skills'] and normalized_skills:
            skill_items = _parse_skill_items([normalized_skills])
            if skill_items:
                parsed['skills'] = skill_items
                fallback_flags.append('skills')

        normalized_education = normalized_sections.get('education', '')
        if not parsed['education'] and normalized_education:
            edu_lines = _split_lines(normalized_education)
            fallback_edu = _build_education_entries(edu_lines)
            if fallback_edu:
                parsed['education'] = fallback_edu
                fallback_flags.append('education')

        if fallback_flags:
            metadata['section_fallback'] = fallback_flags
        if rerouted_misc:
            existing_misc = parsed.get('misc', '')
            combined_misc = ' '.join([existing_misc] + rerouted_misc).strip()
            parsed['misc'] = combined_misc or existing_misc

        if logger.isEnabledFor(logging.DEBUG):
            if experience_promoted:
                logger.debug("Experience headers accepted: %s", experience_promoted[:5])
            if experience_dropped:
                logger.debug("Experience headers dropped: %s", experience_dropped[:5])
            if education_promoted:
                logger.debug("Education blocks accepted: %s", education_promoted[:5])
            if education_rerouted:
                logger.debug("Education items rerouted to misc: %s", education_rerouted[:5])
        
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
            
            # Handle both tuple and single list return values
            if isinstance(result, tuple):
                section_scores, skill_scores = result
            else:
                # Single list returned - use as section scores, skill scores are 0
                section_scores = result if isinstance(result, list) else [0.0] * len(resumes)
                skill_scores = [0.0] * len(resumes)
            
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
                return self._create_fallback_ce_results(resumes, section_tfidf_scores, sbert_scores, skill_tfidf_scores)
            
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
                                                      section_tfidf_scores, sbert_scores)
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
                return self._create_fallback_ce_results(resumes, section_tfidf_scores, sbert_scores, skill_tfidf_scores)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in _apply_cross_encoder_reranking: {e}")
            return self._create_fallback_ce_results(resumes, section_tfidf_scores, sbert_scores, skill_tfidf_scores)

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

    def _parse_date_token(self, token: str) -> Optional[datetime]:
        """Parse tokens like 'June 2022' or '2022' into a datetime object (month precision)."""
        try:
            token_clean = (token or "").strip()
            if not token_clean:
                return None
            token_lower = token_clean.lower()
            if 'present' in token_lower or 'current' in token_lower:
                return datetime.now()

            year_match = re.search(r'\b(?:19|20)\d{2}\b', token_lower)
            if not year_match:
                return None
            year = int(year_match.group(0))

            month = 1
            for idx in range(1, 13):
                name = calendar.month_name[idx].lower()
                abbr = calendar.month_abbr[idx].lower()
                if name and name in token_lower:
                    month = idx
                    break
                if abbr and abbr in token_lower:
                    month = idx
                    break

            return datetime(year, month, 1)
        except Exception:
            return None

    def _estimate_years_from_structured_experience(self, experience_entries: List[Dict[str, Any]]) -> float:
        """
        Estimate total experience duration (in years) from structured experience entries.
        Falls back to heuristic values when dates are missing or incomplete.
        """
        if not experience_entries:
            return 0.0

        current_dt = datetime.now()
        total_months = 0

        for entry in experience_entries:
            dates = (entry.get('dates') or '').strip()
            months = None

            if dates:
                parts = re.split(r'\s*(?:to|–|—|-)\s*', dates, maxsplit=1)
                if len(parts) == 2:
                    start_token, end_token = parts[0], parts[1]
                else:
                    start_token, end_token = dates, ''

                start_dt = self._parse_date_token(start_token)
                end_dt = self._parse_date_token(end_token)

                if start_dt and not end_dt:
                    end_dt = current_dt
                if start_dt and end_dt:
                    if end_dt < start_dt:
                        end_dt = start_dt
                    months = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month) + 1
                elif end_dt and not start_dt:
                    start_dt = end_dt
                    months = 1

            if months is None:
                # Heuristic fallback: treat entries without reliable dates as half a year
                months = 6

            total_months += max(0, months)

        if total_months == 0:
            total_months = 6 * len(experience_entries)

        years = total_months / 12.0
        return round(years, 2)

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
                                  sbert_scores: List[float], skill_tfidf_scores: Optional[List[float]] = None) -> List:
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
                    sbert_score = sbert_scores[i] if i < len(sbert_scores) else 0.0
                    
                    # Simple weighted combination
                    skill_score = 0.0
                    if skill_tfidf_scores is not None:
                        skill_score = skill_tfidf_scores[i] if i < len(skill_tfidf_scores) else 0.0
                    final_score = (0.4 * section_score +
                                   0.25 * skill_score +
                                   0.35 * sbert_score)
                    
                    results.append(CERerankerResult(
                        candidate_id=getattr(resume, 'id', f'fallback_{i}'),
                        ce_score=0.0,
                        section_tfidf=section_score,
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
        def _json_safe(value: Any) -> Any:
            if is_dataclass(value):
                return _json_safe(asdict(value))
            if isinstance(value, dict):
                return {k: _json_safe(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [_json_safe(v) for v in value]
            if hasattr(value, "to_dict"):
                try:
                    return _json_safe(value.to_dict())
                except Exception:
                    pass
            if isinstance(value, (str, int, float, bool)) or value is None:
                return value
            return str(value)

        analysis_results: Dict[str, Dict[str, Any]] = {}
        if USE_ENHANCED_NLG and resumes:
            try:
                from .nlg_generator_enhanced import generate_candidate_analysis_enhanced
                statistics_module = None
                try:
                    import numpy as np  # type: ignore
                except Exception:
                    np = None
                    import statistics as _statistics
                    statistics_module = _statistics

                final_scores = [
                    float(getattr(resume.scores, 'final_score', 0.0) or 0.0)
                    for resume in resumes
                ]
                if final_scores:
                    if np is not None:
                        median_score = float(np.median(final_scores))  # type: ignore[arg-type]
                    else:
                        median_score = float(statistics_module.median(final_scores))  # type: ignore[union-attr]
                else:
                    median_score = 0.0

                batch_stats = {
                    'batch_id': str(uuid.uuid4()),
                    'candidate_count': len(resumes),
                    'all_final_scores': final_scores,
                    'median_score': median_score * 100
                }

                jd_payload = jd_criteria or {}
                for resume in resumes:
                    try:
                        candidate_payload = {
                            'id': resume.id,
                            'scores': asdict(resume.scores),
                            'parsed': resume.parsed,
                            'meta': resume.meta,
                            'matched_skills': resume.matched_skills,
                            'sections': resume.sections,
                            'parsing_ok': resume.parsed.get('metadata', {}).get('parsing_ok', True)
                        }
                        analysis = generate_candidate_analysis_enhanced(
                            candidate_payload,
                            jd_payload,
                            batch_stats
                        )
                        if analysis:
                            analysis_results[resume.id] = _json_safe(analysis)
                    except Exception as inner_e:
                        logger.warning(f"Enhanced NLG generation failed for {resume.id}: {inner_e}")
            except Exception as e:
                logger.warning(f"Enhanced NLG generation unavailable: {e}")
                analysis_results = {}

        # Sort resumes by final_score (descending) before creating output
        # This ensures the output order matches the ranking
        try:
            sorted_resumes = sorted(resumes, key=lambda r: float(getattr(r.scores, 'final_score', 0.0) or 0.0), reverse=True)
        except Exception as e:
            logger.warning(f"Error sorting resumes by final_score: {e}, using original order")
            sorted_resumes = resumes
        
        # Convert resumes to required format
        resumes_output = []
        for resume in sorted_resumes:
            scores_dict = asdict(resume.scores)
            analysis = analysis_results.get(resume.id) if analysis_results else None
            if analysis:
                scores_dict['analysis_text'] = analysis.get('text', '')
                scores_dict['analysis_bullets'] = analysis.get('bullets', [])
                scores_dict['analysis_facts'] = analysis.get('facts', {})
                scores_dict['analysis_metadata'] = analysis.get('metadata', {})

            analysis_block = None
            if analysis:
                analysis_block = {
                    'text': analysis.get('text', ''),
                    'bullets': analysis.get('bullets', []),
                    'facts': analysis.get('facts', {}),
                    'metadata': analysis.get('metadata', {})
                }

            resume_output = {
                'id': resume.id,
                'scores': scores_dict,
                'matched_skills': resume.matched_skills,
                'parsed': resume.parsed,
                'meta': resume.meta
            }
            if analysis_block is not None:
                resume_output['analysis'] = analysis_block
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
