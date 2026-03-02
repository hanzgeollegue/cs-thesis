"""Batch processor for resume ranking pipeline.

Handles PDF parsing, section normalization, structured data extraction,
and output assembly.  Scoring is delegated to the hybrid ranking pipeline
(to be implemented in hybrid_ranker.py).
"""
import os
import re
import uuid
import json
import time
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict, is_dataclass
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

try:
    from .enhanced_pdf_parser import PDFParser
except ImportError:
    try:
        from enhanced_pdf_parser import PDFParser
    except ImportError:
        from resume_processor.enhanced_pdf_parser import PDFParser

try:
    from .text_processor import (
        normalize_job_description,
        scrub_pii_and_boilerplate,
        preprocess_text_for_dense_models,
        chunk_text_for_sbert,
    )
except ImportError:
    try:
        from text_processor import (
            normalize_job_description,
            scrub_pii_and_boilerplate,
            preprocess_text_for_dense_models,
            chunk_text_for_sbert,
        )
    except ImportError:
        from resume_processor.text_processor import (
            normalize_job_description,
            scrub_pii_and_boilerplate,
            preprocess_text_for_dense_models,
            chunk_text_for_sbert,
        )

try:
    from .config import (
        USE_ENHANCED_NLG,
        PARSE_CONCURRENCY,
        BATCH_TIMEOUT_SEC,
        RESUME_TIMEOUT_SEC,
    )
except ImportError:
    try:
        from config import (
            USE_ENHANCED_NLG,
            PARSE_CONCURRENCY,
            BATCH_TIMEOUT_SEC,
            RESUME_TIMEOUT_SEC,
        )
    except ImportError:
        USE_ENHANCED_NLG = True
        PARSE_CONCURRENCY = 4
        BATCH_TIMEOUT_SEC = 300
        RESUME_TIMEOUT_SEC = 60

try:
    from .hybrid_ranker import HybridRanker, SBERT_MODEL_NAME
except ImportError:
    try:
        from hybrid_ranker import HybridRanker, SBERT_MODEL_NAME
    except ImportError:
        from resume_processor.hybrid_ranker import HybridRanker, SBERT_MODEL_NAME

logger = logging.getLogger(__name__)

TIMING_ENABLED = os.getenv("TIMING", "0") in {"1", "true", "True"}


def _safe_log_str(text: str, max_len: int = 50) -> str:
    """Sanitize a string for safe logging on Windows console (cp1252).
    
    Removes or replaces Unicode characters that can't be encoded in cp1252.
    """
    if not text:
        return ""
    # Replace common problematic characters
    replacements = {
        '\u202d': '',  # Left-to-Right Override
        '\u202c': '',  # Pop Directional Formatting
        '\u25cf': '*',  # Black Circle (bullet)
        '\u2022': '*',  # Bullet
        '\u2013': '-',  # En Dash
        '\u2014': '-',  # Em Dash
        '\u2018': "'",  # Left Single Quote
        '\u2019': "'",  # Right Single Quote
        '\u201c': '"',  # Left Double Quote
        '\u201d': '"',  # Right Double Quote
        '\u2026': '...',  # Ellipsis
        '\x11': ' ',   # Device Control 1
    }
    result = text
    for char, replacement in replacements.items():
        result = result.replace(char, replacement)
    # Remove any remaining non-ASCII characters
    result = result.encode('ascii', 'replace').decode('ascii')
    # Truncate if needed
    if max_len and len(result) > max_len:
        result = result[:max_len] + '...'
    return result


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
    verified_skills: List[str] = field(default_factory=list)  # Skills found in experience/projects
    skills_only_skills: List[str] = field(default_factory=list)  # Skills found ONLY in skills section (50% credit)
    coverage: float = 0.0
    gate_threshold: float = 0.0
    gate_reason: str = ""
    rationale: str = ""
    final_score: float = 0.0
    final_score_display: float = 0.0
    score_breakdown: Dict[str, Any] = field(default_factory=dict)

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
        """Initialize BatchProcessor.

        Args:
            api_key: Unused (retained for backward compatibility).
            model: Unused (retained for backward compatibility).
            disable_ocr: If True, skip OCR for image-based PDFs.
        """
        output_dir = os.path.join(os.getcwd(), 'batch_processing_output')
        self.pdf_parser = PDFParser(output_dir=output_dir, disable_ocr=disable_ocr)

        # Hybrid retrieve-and-rerank pipeline (BM25 → SBERT → RRF → Cross-Encoder)
        self.ranker = HybridRanker()

        # Section weights and canonical mapping (used by _normalize_sections)
        self.section_weights = {
            'experience': 0.45,
            'skills': 0.35,
            'education': 0.15,
            'misc': 0.05
        }
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

        # Simple skill taxonomy used by _extract_top_skills in output assembly
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

    def process_batch(self, resumes: List[str], job_description: str,
                      jd_criteria: Optional[Dict[str, Any]] = None,
                      clear_cache: bool = False) -> Dict[str, Any]:
        """Main batch processing pipeline.

        Currently a stub that parses PDFs and returns results with
        final_score = 0.  Scoring will be handled by hybrid_ranker.py
        (BM25 -> SBERT -> RRF -> Cross-Encoder).
        """
        timing_bucket = {}
        batch_start_time = time.perf_counter()

        try:
            # Validate inputs
            if not job_description.strip():
                return {"error": "job_description_required"}

            if len(resumes) > 25:
                return {"error": f"Batch limit exceeded. Maximum 25 resumes allowed, got {len(resumes)}"}

            # Step 1: Parse PDFs to JSON
            logger.info(f"Starting batch processing of {len(resumes)} resumes")
            with time_phase("parse_all", timing_bucket):
                parsed_resumes = self._parse_pdfs_to_json(resumes)

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
                jd_normalized = normalize_job_description(job_description)

            # -- Hybrid Retrieve-and-Rerank pipeline -------------------------
            logger.info("Running hybrid ranking pipeline")
            with time_phase("hybrid_rank", timing_bucket):
                final_ranking = self.ranker.rank(
                    parsed_resumes, jd_normalized,
                    top_k=min(10, len(parsed_resumes)),
                )

            # Step 3: Assemble output (includes NLG if enabled)
            with time_phase("save_results", timing_bucket):
                try:
                    output = self._assemble_output(
                        parsed_resumes, final_ranking, job_description,
                        jd_criteria=jd_criteria,
                    )
                except Exception as e:
                    logger.error(f"Error assembling final output: {e}")
                    output = {
                        'success': False,
                        'error': f'Error assembling output: {str(e)}',
                        'resumes': parsed_resumes,
                        'final_ranking': final_ranking,
                        'batch_summary': {'error': True}
                    }

            logger.info("Batch processing completed successfully")
            if TIMING_ENABLED and timing_bucket:
                total_time = sum(timing_bucket.values())
                logger.info(f"[TIMING] SUMMARY batch({len(resumes)} resumes): {total_time:.3f}s total")

            return output

        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

            try:
                fallback_resumes = [
                    self._create_fallback_resume(p, str(e)) for p in resumes
                ]
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
            executor = ThreadPoolExecutor(max_workers=max_workers)
            try:
                # Submit all tasks and collect results in deterministic order
                # This ensures identical inputs produce identical outputs
                futures = [executor.submit(self._parse_single_resume, path) for path in resume_paths]
                
                for i, future in enumerate(futures):
                    path = resume_paths[i]  # Use original order for deterministic results
                    filename = os.path.basename(path)
                    try:
                        logger.info(f"[PARSE] Waiting for result: {filename} ({i+1}/{len(resume_paths)})")
                        # Add timeout to prevent infinite blocking on problematic PDFs
                        parsed_resume = future.result(timeout=RESUME_TIMEOUT_SEC)
                        if parsed_resume:
                            parsed_resumes.append(parsed_resume)
                            logger.info(f"[PARSE] Completed: {filename}")
                    except FuturesTimeoutError:
                        logger.error(f"[PARSE] TIMEOUT after {RESUME_TIMEOUT_SEC}s: {filename}")
                        future.cancel()  # Try to cancel the timed-out future
                        
                        # Try quick fallback extraction instead of empty resume
                        logger.info(f"[PARSE] Attempting quick fallback extraction for: {filename}")
                        fallback_resume = self._quick_fallback_parse(path)
                        if fallback_resume:
                            parsed_resumes.append(fallback_resume)
                            logger.info(f"[PARSE] Fallback extraction succeeded for: {filename}")
                        else:
                            parsed_resumes.append(self._create_fallback_resume(path, f"Parsing timed out after {RESUME_TIMEOUT_SEC}s"))
                    except Exception as e:
                        logger.error(f"Error parsing {path}: {str(e)}")
                        # Try quick fallback extraction before giving up
                        fallback_resume = self._quick_fallback_parse(path)
                        if fallback_resume:
                            parsed_resumes.append(fallback_resume)
                        else:
                            parsed_resumes.append(self._create_fallback_resume(path, str(e)))
            finally:
                # Don't wait for hung threads - cancel and move on
                logger.info("[PARSE] Shutting down executor (not waiting for hung threads)")
                executor.shutdown(wait=False, cancel_futures=True)
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
        """Fallback text extraction using PyMuPDF when primary parser yields no content.
        Includes OCR fallback for image-based PDFs."""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            logger.error("Fallback parsing requires PyMuPDF (fitz); module not available.")
            return None

        try:
            from .config import OCR_ALLOWED, MAX_OCR_PAGES
        except ImportError:
            OCR_ALLOWED = True
            MAX_OCR_PAGES = 2

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

        # If text extraction yielded < 200 chars, try OCR fallback
        total_chars = sum(len(line) for line in content_lines)
        if total_chars < 200 and OCR_ALLOWED:
            logger.warning(f"[FALLBACK] Only {total_chars} chars extracted, attempting OCR fallback for {os.path.basename(resume_path)}")
            try:
                import pdfplumber
                import pytesseract
                from PIL import Image
                
                ocr_content_lines = []
                with pdfplumber.open(resume_path) as pdf:
                    for page_num, page in enumerate(pdf.pages[:MAX_OCR_PAGES]):
                        page_text = page.extract_text() or ""
                        if len(page_text.strip()) >= 80:
                            # Page has readable text
                            for line in page_text.splitlines():
                                cleaned = line.strip()
                                if cleaned:
                                    ocr_content_lines.append(cleaned)
                        else:
                            # Try OCR
                            try:
                                pil_img = page.to_image(resolution=200).original
                                config = "--oem 1 --psm 6 -c preserve_interword_spaces=1"
                                ocr_text = pytesseract.image_to_string(pil_img, config=config)
                                if ocr_text and len(ocr_text.strip()) > len(page_text.strip()):
                                    logger.info(f"[FALLBACK] OCR extracted {len(ocr_text.strip())} chars from page {page_num + 1}")
                                    for line in ocr_text.splitlines():
                                        cleaned = line.strip()
                                        if cleaned:
                                            ocr_content_lines.append(cleaned)
                                elif page_text.strip():
                                    for line in page_text.splitlines():
                                        cleaned = line.strip()
                                        if cleaned:
                                            ocr_content_lines.append(cleaned)
                            except Exception as ocr_e:
                                logger.warning(f"[FALLBACK] OCR failed for page {page_num + 1}: {ocr_e}")
                                if page_text.strip():
                                    for line in page_text.splitlines():
                                        cleaned = line.strip()
                                        if cleaned:
                                            ocr_content_lines.append(cleaned)
                
                ocr_total_chars = sum(len(line) for line in ocr_content_lines)
                if ocr_total_chars > total_chars:
                    logger.info(f"[FALLBACK] OCR improved extraction: {ocr_total_chars} chars (was {total_chars})")
                    content_lines = ocr_content_lines
                else:
                    logger.info(f"[FALLBACK] OCR did not improve extraction ({ocr_total_chars} vs {total_chars})")
            except ImportError as import_e:
                logger.warning(f"[FALLBACK] OCR dependencies not available: {import_e}")
            except Exception as ocr_e:
                logger.error(f"[FALLBACK] OCR fallback failed: {ocr_e}")

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
            top_headers = [_safe_log_str(s.get('header', ''), 40) for s in sections[:5]]
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
    
    def _quick_fallback_parse(self, path: str) -> Optional[ParsedResume]:
        """Quick fallback extraction for timed-out PDFs using simple PyMuPDF text extraction.
        
        This is a simplified extraction that:
        1. Opens the PDF with PyMuPDF
        2. Extracts plain text from all pages (no layout analysis)
        3. If < 200 chars extracted, attempts OCR fallback (for image-based PDFs)
        4. Puts all text into a 'misc' section
        
        This ensures we get SOME text even for complex PDFs that timeout during full parsing.
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            logger.warning("PyMuPDF not available for fallback parsing")
            return None
        
        try:
            from .config import OCR_ALLOWED, MAX_OCR_PAGES
        except ImportError:
            OCR_ALLOWED = True
            MAX_OCR_PAGES = 2
        
        try:
            doc = fitz.open(path)
            all_text_lines = []
            page_count = min(doc.page_count, 3)  # Limit to 3 pages
            
            for page_num in range(page_count):
                page = doc[page_num]
                # Use simple text extraction - no layout analysis
                text = page.get_text("text") or ""
                if text:
                    for line in text.splitlines():
                        cleaned = line.strip()
                        if cleaned and len(cleaned) > 1:
                            all_text_lines.append(cleaned)
            
            doc.close()
            
            raw_text = '\n'.join(all_text_lines) if all_text_lines else ""
            raw_text_len = len(raw_text)
            
            # If text extraction yielded < 200 chars, try OCR fallback for image-based PDFs
            if raw_text_len < 200 and OCR_ALLOWED:
                logger.warning(f"[FALLBACK] Only {raw_text_len} chars extracted, attempting OCR fallback for {os.path.basename(path)}")
                try:
                    import pdfplumber
                    import pytesseract
                    from PIL import Image
                    
                    ocr_text_lines = []
                    with pdfplumber.open(path) as pdf:
                        for page_num, page in enumerate(pdf.pages[:MAX_OCR_PAGES]):
                            # Check if page has readable text first
                            page_text = page.extract_text() or ""
                            if len(page_text.strip()) >= 80:
                                # Page has readable text, use it
                                for line in page_text.splitlines():
                                    cleaned = line.strip()
                                    if cleaned and len(cleaned) > 1:
                                        ocr_text_lines.append(cleaned)
                            else:
                                # Try OCR
                                try:
                                    pil_img = page.to_image(resolution=200).original
                                    config = "--oem 1 --psm 6 -c preserve_interword_spaces=1"
                                    ocr_text = pytesseract.image_to_string(pil_img, config=config)
                                    if ocr_text and len(ocr_text.strip()) > len(page_text.strip()):
                                        logger.info(f"[FALLBACK] OCR extracted {len(ocr_text.strip())} chars from page {page_num + 1}")
                                        for line in ocr_text.splitlines():
                                            cleaned = line.strip()
                                            if cleaned and len(cleaned) > 1:
                                                ocr_text_lines.append(cleaned)
                                    elif page_text.strip():
                                        # Use original text if OCR didn't help
                                        for line in page_text.splitlines():
                                            cleaned = line.strip()
                                            if cleaned and len(cleaned) > 1:
                                                ocr_text_lines.append(cleaned)
                                except Exception as ocr_e:
                                    logger.warning(f"[FALLBACK] OCR failed for page {page_num + 1}: {ocr_e}")
                                    # Fall back to original text if available
                                    if page_text.strip():
                                        for line in page_text.splitlines():
                                            cleaned = line.strip()
                                            if cleaned and len(cleaned) > 1:
                                                ocr_text_lines.append(cleaned)
                    
                    ocr_text = '\n'.join(ocr_text_lines) if ocr_text_lines else ""
                    ocr_text_len = len(ocr_text)
                    
                    if ocr_text_len > raw_text_len:
                        logger.info(f"[FALLBACK] OCR improved extraction: {ocr_text_len} chars (was {raw_text_len})")
                        raw_text = ocr_text
                        raw_text_len = ocr_text_len
                    else:
                        logger.info(f"[FALLBACK] OCR did not improve extraction ({ocr_text_len} vs {raw_text_len}), keeping original")
                except ImportError as import_e:
                    logger.warning(f"[FALLBACK] OCR dependencies not available: {import_e}")
                except Exception as ocr_e:
                    logger.error(f"[FALLBACK] OCR fallback failed: {ocr_e}")
            
            if not raw_text or raw_text_len == 0:
                logger.warning(f"[FALLBACK] No text extracted from {os.path.basename(path)}")
                return None
            
            logger.info(f"[FALLBACK] Extracted {raw_text_len} chars from {os.path.basename(path)}")
            
            # Create a minimal but functional ParsedResume
            return ParsedResume(
                id=str(uuid.uuid4()),
                sections={
                    'experience': '',
                    'skills': '',
                    'education': '',
                    'misc': raw_text  # Put all content in misc
                },
                meta={
                    "source_file": os.path.basename(path),
                    "pages": page_count,
                    "processing_status": "fallback",
                    "parse_method": "quick_fallback"
                },
                scores=ResumeScores(0.0, 0.0, 0.0),
                matched_skills=[],
                parsed={
                    'experience': [],
                    'skills': [],
                    'education': [],
                    'misc': raw_text,
                    'raw_text': raw_text  # Important for TF-IDF and SBERT
                }
            )
            
        except Exception as e:
            logger.error(f"[FALLBACK] Quick fallback parse failed for {path}: {e}")
            return None
    
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
                scores_dict['score_breakdown'] = analysis.get('score_breakdown', {})

            analysis_block = None
            if analysis:
                analysis_block = {
                    'text': analysis.get('text', ''),
                    'bullets': analysis.get('bullets', []),
                    'facts': analysis.get('facts', {}),
                    'metadata': analysis.get('metadata', {}),
                    'score_breakdown': analysis.get('score_breakdown', {})
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
            'embedding_info': {'model': SBERT_MODEL_NAME, 'dim': 384}
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

