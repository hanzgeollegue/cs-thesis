import pdfplumber
import pytesseract
from PIL import Image
import json
import os
import signal
from django.conf import settings
from django.utils import timezone
import logging
from typing import List, Dict, Any, Tuple, Optional
import re
import hashlib
import unicodedata
from concurrent.futures import ProcessPoolExecutor, as_completed
import difflib
import math
import time
import contextlib
from contextlib import contextmanager
import warnings

# Suppress harmless pdfminer warnings about color spaces (e.g., named patterns like 'P1', 'P2')
# These occur when PDFs have pattern color spaces that pdfminer doesn't fully understand
warnings.filterwarnings(
    'ignore',
    message=r".*Cannot set .* color because.*invalid float value.*",
    category=UserWarning
)

# Suppress pdfminer/pdfplumber logging warnings (color space issues, etc.)
# These are harmless but clutter the logs
for _logger_name in ['pdfminer', 'pdfminer.pdfpage', 'pdfminer.converter', 
                      'pdfminer.pdfinterp', 'pdfminer.cmapdb', 'pdfminer.psparser',
                      'pdfminer.pdfparser', 'pdfminer.pdfdocument', 'pdfminer.pdfcolor',
                      'pdfplumber']:
    logging.getLogger(_logger_name).setLevel(logging.ERROR)

# Optional fast PDF text extraction
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

logger = logging.getLogger(__name__)


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


def preprocess_text(text: str) -> str:
    """
    Preprocess text by cleaning and normalizing it.
    
    Args:
        text (str): Raw text to preprocess.
        
    Returns:
        str: Preprocessed text.
    """
    try:
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower().strip()
        
        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)
        
        # Remove special characters but keep alphanumeric, spaces, and common punctuation
        text = re.sub(r"[^\w\s.,;:!?-]", "", text)
        
        # Remove extra punctuation
        text = re.sub(r"[.,;:!?]+", " ", text)
        
        # Final whitespace cleanup
        text = re.sub(r"\s+", " ", text).strip()
        
        return text
        
    except Exception as e:
        logger.warning(f"Error preprocessing text: {e}")
        return ""


# Timeout helper
@contextlib.contextmanager
def time_limit(seconds: int, label: str = "resume"):
    if hasattr(signal, "SIGALRM"):
        def _alarm(_s, _f): 
            raise TimeoutError(f"{label} timed out after {seconds}s")
        old = signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(seconds)
        try: 
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)
    else:
        # Fallback: cooperative check (callers should check elapsed)
        start = time.perf_counter()
        yield
        if time.perf_counter() - start > seconds:
            raise TimeoutError(f"{label} timed out after {seconds}s")

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

# --- Configuration toggles ---
USE_PYMUPDF = True  # Try PyMuPDF (fitz) first if available
# Import configuration from config module
try:
    from .config import MAX_PAGES, MAX_OCR_PAGES, OCR_ALLOWED, RESUME_TIMEOUT_SEC, PARSER_BACKEND
except ImportError:
    MAX_PAGES = 3
    MAX_OCR_PAGES = 2
    OCR_ALLOWED = True
    RESUME_TIMEOUT_SEC = 120
    PARSER_BACKEND = "pymupdf4llm"

# Try to import pymupdf4llm if configured
try:
    if PARSER_BACKEND == "pymupdf4llm":
        import pymupdf4llm
        PYMUPDF4LLM_AVAILABLE = True
    else:
        PYMUPDF4LLM_AVAILABLE = False
except ImportError:
    PYMUPDF4LLM_AVAILABLE = False
    if PARSER_BACKEND == "pymupdf4llm":
        logger.warning("pymupdf4llm not available, falling back to fitz/pdfplumber")

# --- Precompiled regexes & inline synonyms ---
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
URL_RE = re.compile(r"https?://|www\.")
PHONE_RE = re.compile(r"\b\d{3}[-\.]?\d{3}[-\.]?\d{4}\b")
FOOTER_RE = re.compile(r"^(Page\s+)?\d+(\s+of\s+\d+)?$", re.IGNORECASE)
TRAILING_PERIOD_RE = re.compile(r"\.[\s]*$")
LEADING_BULLET_COLON_RE = re.compile(r"^[\u2022\u25CF\-\*\s]*([^:\n]+?)(:?)[\s]*$")
HEADER_CANDIDATE_THRESHOLD = 0.44  # Minimum header score to treat a line as a real section break

CANONICAL_SECTION_PRIORITY = [
    'experience', 'skills', 'education', 'projects', 'certifications',
    'languages', 'awards', 'volunteer', 'summary', 'contact', 'misc'
]

CONTENT_HINTS: Dict[str, List[str]] = {
    'experience': [
        'experience', 'work history', 'employment', 'professional experience',
        'responsible for', 'managed', 'developed', 'engineer', 'company',
        'intern', 'team lead', 'product', 'project manager', 'delivered'
    ],
    'skills': [
        'skills', 'technical', 'technologies', 'tech stack', 'frameworks',
        'tools', 'proficient', 'expertise', 'competencies', 'languages',
        'platforms', 'libraries'
    ],
    'education': [
        'education', 'university', 'college', 'school', 'academy', 'bachelor',
        'master', 'phd', 'gpa', 'degree', 'curriculum', 'diploma'
    ],
    'projects': [
        'project', 'projects', 'built', 'designed', 'implemented', 'prototype',
        'hackathon', 'case study'
    ],
    'certifications': [
        'certification', 'certifications', 'certified', 'license', 'credential'
    ],
    'languages': [
        'language', 'languages', 'fluent', 'bilingual', 'native', 'speak'
    ],
    'awards': [
        'award', 'awards', 'honor', 'recognition', 'achievement', 'distinction'
    ],
    'volunteer': [
        'volunteer', 'community', 'service', 'nonprofit', 'ngo', 'outreach'
    ],
    'summary': [
        'summary', 'objective', 'profile', 'statement', 'overview'
    ],
    'contact': [
        'contact', 'information', 'phone', 'email', 'address', 'linkedin'
    ]
}

# Small inline synonym set (canonical -> synonyms)
SECTION_SYNONYMS: Dict[str, List[str]] = {
    'experience': [
        'experience', 'work experience', 'employment', 'career history', 'work history',
        'professional experience', 'roles', 'professional background', 'employment history'
    ],
    'skills': [
        'skills', 'technical skills', 'core competencies', 'competencies', 'expertise',
        'proficiencies', 'technical proficiencies', 'toolset', 'technology skills'
    ],
    'education': [
        'education', 'academic background', 'qualifications', 'academic qualifications',
        'educational background', 'educational attainment', 'college', 'university',
        'secondary education', 'high school', 'tertiary education', 'academic history'
    ],
    'projects': [
        'projects', 'project experience', 'project work', 'key projects', 'notable projects',
        'portfolio projects'
    ],
    'certifications': [
        'certifications', 'licenses', 'professional certifications', 'accreditations'
    ],
    'awards': [
        'awards', 'honors', 'achievements', 'recognition', 'accomplishments', 'distinctions'
    ],
    'summary': [
        'summary', 'professional summary', 'career summary', 'profile', 'professional profile',
        'personal statement', 'objective', 'career objective', 'career objectives'
    ],
    'contact': [
        'contact', 'contact information', 'contact info', 'personal information',
        'personal details', 'address', 'phone', 'email'
    ],
    'training': [
        'training', 'professional development', 'courses', 'workshops', 'seminars',
        'seminars and training experiences', 'training experiences'
    ],
    'references': [
        'references', 'professional references', 'character references', 'referees'
    ],
    'languages': [
        'languages', 'language skills', 'linguistic skills', 'foreign languages'
    ],
    'interests': [
        'interests', 'hobbies', 'personal interests', 'activities', 'extracurricular'
    ],
    'volunteer': [
        'volunteer', 'volunteer work', 'volunteer experience', 'community service'
    ],
    'leadership': [
        'leadership', 'leadership roles', 'leadership experience', 'leadership positions'
    ]
}
ALL_SYNONYMS: List[str] = sorted({syn for syns in SECTION_SYNONYMS.values() for syn in syns}, key=len, reverse=True)

class PDFParser:
    """Enhanced PDF parser that accurately extracts resume data with proper header detection and content grouping."""
    
    def __init__(self, output_dir: str = None, disable_ocr: bool = False):
        # Handle output directory setup
        if output_dir:
            self.text_output_dir = os.path.join(output_dir, 'extracted_text')
            self.structured_output_dir = os.path.join(output_dir, 'structured_data')
        else:
            # Try to get Django settings, fallback to current directory
            try:
                self.text_output_dir = os.path.join(settings.MEDIA_ROOT, 'extracted_text')
                self.structured_output_dir = os.path.join(settings.MEDIA_ROOT, 'structured_data')
            except (ImportError, AttributeError):
                # Fallback to current directory
                current_dir = os.getcwd()
                self.text_output_dir = os.path.join(current_dir, 'extracted_text')
                self.structured_output_dir = os.path.join(current_dir, 'structured_data')
        
        # Create directories if they don't exist
        os.makedirs(self.text_output_dir, exist_ok=True)
        os.makedirs(self.structured_output_dir, exist_ok=True)
        
        # Control OCR usage
        self.disable_ocr = disable_ocr

        # Cache location (by file hash)
        self.cache_dir = self.text_output_dir
        # Tracking of last extraction positions to aid header scoring
        self._last_line_positions: List[Tuple[Optional[float], Optional[bool], Optional[float], Optional[float], Optional[int]]] = []  # (font_size, is_bold, x, y, page)
        self._left_margin_by_page: Dict[int, float] = {}
        self._font_median_by_page: Dict[int, float] = {}
        
        # Comprehensive list of resume section headers
        self.section_headers = {
            'contact': ['contact', 'contact information', 'personal information', 'personal details', 'address', 'phone', 'email'],
            'summary': ['summary', 'profile', 'professional summary', 'career summary', 'objective', 'career objective', 'professional profile', 'personal statement'],
            'experience': ['experience', 'work experience', 'professional experience', 'employment', 'employment history', 'career history', 'work history', 'professional background'],
            'education': ['education', 'educational background', 'academic background', 'qualifications', 'academic qualifications', 'academic history'],
            'skills': ['skills', 'technical skills', 'core competencies', 'competencies', 'expertise', 'proficiencies', 'capabilities', 'abilities'],
            'certifications': ['certifications', 'certificates', 'professional certifications', 'licenses', 'accreditations'],
            'projects': ['projects', 'key projects', 'notable projects', 'project experience', 'project work'],
            'awards': ['awards', 'honors', 'achievements', 'recognition', 'accomplishments', 'distinctions'],
            'publications': ['publications', 'research', 'papers', 'articles', 'papers published'],
            'languages': ['languages', 'language skills', 'linguistic skills', 'foreign languages'],
            'interests': ['interests', 'hobbies', 'personal interests', 'activities', 'extracurricular'],
            'references': ['references', 'professional references', 'referees', 'character references'],
            'availability': ['availability', 'available', 'availability status', 'when available'],
            'leadership': ['leadership', 'leadership roles', 'leadership experience', 'leadership positions', 'leadership activities'],
            'volunteer': ['volunteer', 'volunteer work', 'volunteer experience', 'community service'],
            'training': ['training', 'professional development', 'courses', 'workshops', 'seminars']
        }
        
        # Content to filter out (page headers, tips, etc.)
        self.filter_patterns = [
            r'^(Page\s+)?\d+(\s+of\s+\d+)?$',  # Page numbers
            r'^(Resume|CV|Curriculum\s+Vitae)$',  # Document titles
            r'^Tips?\s*:',  # Tips
            r'^Note\s*:',  # Notes
            r'^Remember\s*:',  # Reminders
            r'^Keep\s+in\s+mind\s*:',  # Instructions
            r'^supervisors?\s+after\s+completing\s+work\s+experience',  # Specific problematic content
            r'^This\s+is\s+a\s+sample',  # Sample text
            r'^For\s+best\s+results',  # Instructions
            r'^Make\s+sure\s+to',  # Instructions
            r'^Include\s+your',  # Instructions
            r'^Don\'t\s+forget\s+to',  # Instructions
        ]
        # Precompute canonical lookup for faster section tagging
        self._section_alias_lookup = self._build_section_alias_lookup()
        self._inline_header_keywords = set([
            'experience', 'work experience', 'employment', 'professional experience',
            'education', 'academic background', 'skills', 'technical skills',
            'projects', 'project experience', 'certifications', 'achievements',
            'awards', 'volunteer', 'activities', 'summary', 'objective', 'profile',
            'languages', 'contact'
        ])

    def extract_to_json(self, pdf_file_path: str, original_filename: str) -> str:
        structured_data = self._extract_structured_data(pdf_file_path)
        json_file_path = self._save_json(structured_data, original_filename)
        return json_file_path

    def _extract_structured_data(self, pdf_file_path: str) -> Dict[str, Any]:
        structured_data = {
            'success': True,
            'sections': [],
            'summary': {},
            'layout_metadata': {},  # New: Store layout information for ML processing
            'error': None,
            'processing_status': 'processing'
        }
        
        timing_bucket = {}
        
        try:
            with time_limit(RESUME_TIMEOUT_SEC, f"resume_{os.path.basename(pdf_file_path)}"):
                return self._extract_structured_data_impl(pdf_file_path, timing_bucket)
        except TimeoutError as e:
            logger.warning(f"Resume processing timed out: {e}")
            structured_data['processing_status'] = 'timeout'
            structured_data['error'] = str(e)
            return structured_data
        except Exception as e:
            logger.error(f"Error extracting structured data: {str(e)}")
            structured_data['success'] = False
            structured_data['error'] = str(e)
            structured_data['processing_status'] = 'failed'
            return structured_data

    def _extract_structured_data_impl(self, pdf_file_path: str, timing_bucket: Dict[str, float]) -> Dict[str, Any]:
        structured_data = {
            'success': True,
            'sections': [],
            'summary': {},
            'layout_metadata': {},  # New: Store layout information for ML processing
            'error': None,
            'processing_status': 'completed'
        }
        
        # Log backend configuration
        logger.info(f"[PARSER] Starting extraction for {os.path.basename(pdf_file_path)}")
        logger.info(f"[PARSER] PARSER_BACKEND={PARSER_BACKEND}, PYMUPDF4LLM_AVAILABLE={PYMUPDF4LLM_AVAILABLE}")
        
        # Quick cache: return if we have a cached structured output for this file hash
        with time_phase("probe_cache", timing_bucket):
            file_hash = self._hash_file(pdf_file_path)
            cache_path = os.path.join(self.cache_dir, f"{file_hash}_structured.json")
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, 'r', encoding='utf-8') as cf:
                        cached = json.load(cf)
                        if isinstance(cached, dict) and cached.get('sections'):
                            if 'canonical_sections' not in cached:
                                tagged, canonical = self._tag_sections_with_canonical_labels(cached.get('sections', []))
                                cached['sections'] = tagged
                                cached['canonical_sections'] = canonical
                            logger.info(f"[PARSER] Using cached result with {len(cached.get('sections', []))} sections")
                            return cached
                except Exception:
                    pass

        # Try pymupdf4llm first if configured
        extraction_method = None
        page_widths = {}  # Initialize for column detection
        if PARSER_BACKEND == "pymupdf4llm" and PYMUPDF4LLM_AVAILABLE:
            logger.info("[PARSER] Attempting pymupdf4llm extraction")
            extraction_method = "pymupdf4llm"
            with time_phase("extract_pymupdf4llm", timing_bucket):
                text_elements, all_lines_markdown = self._extract_text_pymupdf4llm(pdf_file_path)
            
            if text_elements:
                logger.info(f"[PARSER] pymupdf4llm extracted {len(text_elements)} text elements")
                # Use markdown lines directly for better section detection
                pages_total = len(set(te.get('page', 0) for te in text_elements))
                pages_processed = min(pages_total, MAX_PAGES)
                # Get page widths from fitz for column detection
                if fitz is not None:
                    try:
                        doc = fitz.open(pdf_file_path)
                        for page_num, page in enumerate(doc[:MAX_PAGES]):
                            page_widths[page_num] = float(page.rect.width)
                        doc.close()
                    except Exception:
                        pass
            else:
                logger.warning("[PARSER] pymupdf4llm returned no elements, falling back to fitz/pdfplumber")
                extraction_method = "fitz_fallback"
                # Fallback to standard extraction
                with pdfplumber.open(pdf_file_path) as pdf:
                    pages_total = len(pdf.pages)
                    pages_processed = min(pages_total, MAX_PAGES)
                    
                    with time_phase("extract_fast", timing_bucket):
                        fast_result = self._extract_text_fast(pdf_file_path)
                    
                    if not fast_result:
                        logger.info("[PARSER] fitz fast path returned no elements, using pdfplumber")
                        extraction_method = "pdfplumber"
                        page_widths = {}
                        with time_phase("extract_pdfplumber", timing_bucket):
                            text_elements = self._extract_text_with_layout(pdf)
                    else:
                        text_elements, page_widths = fast_result
                        extraction_method = "fitz"
                    
                    all_lines_markdown = None
        else:
            # Standard extraction path
            logger.info(f"[PARSER] Using standard extraction (pymupdf4llm not available or not configured)")
            with pdfplumber.open(pdf_file_path) as pdf:
                # Set meta.pages_total from actual PDF length
                pages_total = len(pdf.pages)
                pages_processed = min(pages_total, MAX_PAGES)
                
                # Capture page widths from pdfplumber for column detection
                for page_num, page in enumerate(pdf.pages[:MAX_PAGES]):
                    page_widths[page_num] = float(page.width)
                
                # Extract text with layout metadata (prefer PyMuPDF when available)
                with time_phase("extract_fast", timing_bucket):
                    fast_result = self._extract_text_fast(pdf_file_path)
                
                if not fast_result:
                    logger.info("[PARSER] fitz fast path returned no elements, using pdfplumber")
                    extraction_method = "pdfplumber"
                    with time_phase("extract_pdfplumber", timing_bucket):
                        text_elements = self._extract_text_with_layout(pdf)
                else:
                    text_elements, page_widths = fast_result
                    extraction_method = "fitz"
                
                all_lines_markdown = None
        
        if not text_elements:
            structured_data['success'] = False
            structured_data['error'] = 'No text could be extracted from PDF'
            logger.error("[PARSER] No text elements extracted from PDF")
            return structured_data
        
        # Check if extracted text is too short (likely image-based PDF)
        total_chars = sum(len(te.get('text', '')) for te in text_elements)
        MIN_CHARS_THRESHOLD = 200  # If less than 200 chars, try OCR
        
        if total_chars < MIN_CHARS_THRESHOLD and OCR_ALLOWED and not self.disable_ocr:
            logger.warning(f"[PARSER] Only {total_chars} chars extracted, attempting OCR fallback")
            try:
                with pdfplumber.open(pdf_file_path) as pdf:
                    ocr_text_elements = []
                    for page_num, page in enumerate(pdf.pages[:MAX_PAGES]):
                        ocr_text = self._ocr_page(page)
                        if ocr_text:
                            for line_num, line in enumerate(ocr_text.split('\n')):
                                if line.strip():
                                    ocr_text_elements.append({
                                        'text': line.strip(),
                                        'page': page_num,
                                        'y': float(line_num * 12),  # Approximate y position
                                        'x': 0.0,
                                        'font_size': 12.0,
                                        'font_name': 'ocr',
                                        'is_bold': False,
                                        'is_italic': False
                                    })
                    
                    ocr_chars = sum(len(te.get('text', '')) for te in ocr_text_elements)
                    if ocr_chars > total_chars:
                        logger.info(f"[PARSER] OCR improved extraction: {ocr_chars} chars (was {total_chars})")
                        text_elements = ocr_text_elements
                        extraction_method = "ocr_fallback"
                    else:
                        logger.info(f"[PARSER] OCR did not improve extraction, keeping original")
            except Exception as e:
                logger.error(f"[PARSER] OCR fallback failed: {e}")
        
        # Log extraction results and font info availability
        logger.info(f"[PARSER] Extraction method: {extraction_method or 'unknown'}, extracted {len(text_elements)} text elements")
        font_info_available = sum(1 for te in text_elements[:10] if te.get('font_size') and te.get('font_size') != 12) > 0
        bold_info_available = any(te.get('is_bold') for te in text_elements[:10])
        logger.info(f"[PARSER] Font info available: font_size={font_info_available}, is_bold={bold_info_available}")
        
        # Column detection and reordering for two-column layouts
        # CRITICAL: This ensures each column is read top-to-bottom before moving to next column
        if text_elements:
            text_elements = self._detect_and_group_by_columns(text_elements, page_widths)
            logger.info(f"[PARSER] After column detection: {len(text_elements)} text elements")
        
        if text_elements:
            sample_elements = text_elements[:3]
            for i, te in enumerate(sample_elements):
                logger.debug(f"[PARSER] Sample element {i}: text='{te.get('text', '')[:50]}...', font_size={te.get('font_size')}, is_bold={te.get('is_bold')}, x={te.get('x')}, y={te.get('y')}")
        
        # Store layout metadata for ML processing
        structured_data['layout_metadata'] = {
            'text_elements': text_elements,
            'font_statistics': self._calculate_font_statistics(text_elements),
            'layout_analysis': self._analyze_layout(text_elements)
        }

        # Build positional context lists for header scoring
        self._last_line_positions = []
        self._left_margin_by_page = {}
        self._font_median_by_page = {}
        
        # Compute per-page left margin and font median
        by_page: Dict[int, List[Dict[str, Any]]] = {}
        for te in text_elements:
            by_page.setdefault(int(te.get('page', 0)), []).append(te)
        for pg, elems in by_page.items():
            xs = [float(e.get('x', 0) or 0.0) for e in elems if e.get('x') is not None]
            fsz = [float(e.get('font_size', 0) or 0.0) for e in elems if e.get('font_size')]
            if xs:
                self._left_margin_by_page[pg] = min(xs)
            if fsz:
                fsz_sorted = sorted(f for f in fsz if f > 0)
                if fsz_sorted:
                    self._font_median_by_page[pg] = fsz_sorted[len(fsz_sorted)//2]
        
        # Extract clean text lines for current processing
        if all_lines_markdown:
            # Use markdown lines directly from pymupdf4llm
            all_lines = all_lines_markdown
            # Still need to build position tracking for header scoring
            self._last_line_positions = [
                (
                    te.get('font_size'), te.get('is_bold'), te.get('x'), te.get('y'), int(te.get('page', 0))
                ) for te in text_elements if te['text'].strip()
            ]
        else:
            all_lines = [elem['text'] for elem in text_elements if elem['text'].strip()]
            # Align positions with lines
            self._last_line_positions = [
                (
                    te.get('font_size'), te.get('is_bold'), te.get('x'), te.get('y'), int(te.get('page', 0))
                ) for te in text_elements if te['text'].strip()
            ]
        
        with time_phase("normalize_lines", timing_bucket):
            all_lines = self._normalize_lines(all_lines)
            all_lines = self._strip_repeated_headers_footers(all_lines)
        
        # Clean and preprocess lines
        with time_phase("clean_lines", timing_bucket):
            cleaned_lines = self._clean_lines(all_lines)
        
        # Detect actual section headers and group content
        with time_phase("detect_sections", timing_bucket):
            sections = self._detect_sections_and_group_content(cleaned_lines)
        
        # Apply final quality check
        with time_phase("final_qc", timing_bucket):
            sections = self._final_quality_check(sections)

        sections, canonical_sections = self._tag_sections_with_canonical_labels(sections)
        
        # Debug: Log Experience section length
        experience_text = canonical_sections.get('experience', '')
        education_text = canonical_sections.get('education', '')
        misc_text = canonical_sections.get('misc', '')
        logger.info(f"[PARSER] Experience section length: {len(experience_text)} characters")
        logger.info(f"[PARSER] Education section length: {len(education_text)} characters")
        logger.info(f"[PARSER] Misc section length: {len(misc_text)} characters")
        
        # BUCKET LEAK DETECTION: Experience empty but Education/Misc bloated
        bucket_leak_detected = False
        bloated_section = None
        
        if len(experience_text.strip()) < 100:
            if len(education_text) > 2000:
                bucket_leak_detected = True
                bloated_section = 'education'
                logger.warning(f"[BUCKET_LEAK_PARSER] Detected: Experience={len(experience_text)} chars, Education={len(education_text)} chars")
            elif len(misc_text) > 2000:
                bucket_leak_detected = True
                bloated_section = 'misc'
                logger.warning(f"[BUCKET_LEAK_PARSER] Detected: Experience={len(experience_text)} chars, Misc={len(misc_text)} chars")
        
        if bucket_leak_detected and bloated_section:
            logger.warning(f"[BUCKET_LEAK_PARSER] Attempting to re-split {bloated_section} section")
            
            # Find the bloated section and try to resplit it
            new_sections = []
            new_canonical_sections = dict(canonical_sections)
            
            for section in sections:
                if section.get('canonical') == bloated_section:
                    content = section.get('content', [])
                    if content:
                        # Try to re-split this bloated section
                        resplit_result = self._resplit_bloated_section(content, bloated_section)
                        
                        if len(resplit_result) > 1:
                            logger.info(f"[BUCKET_LEAK_PARSER] Successfully re-split {bloated_section} into {len(resplit_result)} sections")
                            
                            for resplit in resplit_result:
                                resplit_header = resplit.get('header', '').lower()
                                resplit_content = resplit.get('content', [])
                                resplit_text = ' '.join(str(c) for c in resplit_content)
                                
                                # Determine canonical based on header
                                if 'experience' in resplit_header or 'work' in resplit_header or 'employment' in resplit_header:
                                    resplit_canonical = 'experience'
                                elif 'education' in resplit_header or 'academic' in resplit_header:
                                    resplit_canonical = 'education'
                                else:
                                    resplit_canonical = bloated_section
                                
                                new_sections.append({
                                    'header': resplit.get('header', ''),
                                    'content': resplit_content,
                                    'canonical': resplit_canonical
                                })
                                
                                # Update canonical sections
                                if resplit_canonical in new_canonical_sections:
                                    new_canonical_sections[resplit_canonical] = (
                                        new_canonical_sections.get(resplit_canonical, '') + ' ' + resplit_text
                                    ).strip()
                                else:
                                    new_canonical_sections[resplit_canonical] = resplit_text
                        else:
                            # Resplit didn't work, keep original
                            new_sections.append(section)
                    else:
                        new_sections.append(section)
                else:
                    new_sections.append(section)
            
            # Check if resplit helped
            new_experience = new_canonical_sections.get('experience', '')
            if len(new_experience.strip()) > len(experience_text.strip()):
                logger.info(f"[BUCKET_LEAK_PARSER] Resplit successful! Experience now has {len(new_experience)} chars (was {len(experience_text)})")
                sections = new_sections
                canonical_sections = new_canonical_sections
            else:
                logger.warning("[BUCKET_LEAK_PARSER] Resplit did not improve Experience section, keeping original")
        
        # Fallback: If still empty after resplit attempt, use combined section
        experience_text = canonical_sections.get('experience', '')
        if len(experience_text.strip()) == 0:
            logger.warning(f"[PARSER] Experience still empty after resplit attempts")
            logger.warning("[PARSER] Attempting final fallback: re-reading entire raw text as Combined Section")
            
            # Re-read entire raw text without splitting
            all_raw_text = ' '.join([elem.get('text', '') for elem in text_elements if elem.get('text', '').strip()])
            
            if all_raw_text.strip():
                # Create a single "Combined Section" with all content
                structured_data['sections'] = [{
                    'header': 'Combined Section',
                    'content': [all_raw_text],
                    'canonical': 'misc'
                }]
                structured_data['canonical_sections'] = {
                    'misc': all_raw_text,
                    'experience': all_raw_text,  # Also put in experience for scoring
                    'education': canonical_sections.get('education', all_raw_text)
                }
                logger.info(f"[PARSER] Final fallback applied: Created Combined Section with {len(all_raw_text)} characters")
            else:
                # Even fallback failed - keep original sections
                logger.error("[PARSER] Fallback failed: no raw text available")
                structured_data['sections'] = sections
                structured_data['canonical_sections'] = canonical_sections
        else:
            # Normal case: sections are populated
            structured_data['sections'] = sections
            structured_data['canonical_sections'] = canonical_sections
        
        # Log section detection results
        logger.info(f"[PARSER] Final section count: {len(structured_data['sections'])}")
        if structured_data['sections']:
            section_headers = [_safe_log_str(s.get('header', ''), 40) for s in structured_data['sections'][:5]]
            logger.info(f"[PARSER] Section headers (first 5): {section_headers}")
            for i, section in enumerate(structured_data['sections'][:5]):
                content_len = sum(len(str(item)) for item in section.get('content', []))
                canonical = section.get('canonical', 'unknown')
                logger.info(f"[PARSER] Section {i}: header='{_safe_log_str(section.get('header', ''), 60)}', canonical='{canonical}', content_length={content_len}")
        else:
            logger.warning("[PARSER] No sections detected!")
        
        if structured_data.get('canonical_sections'):
            logger.info(f"[PARSER] Canonical sections present: {list(structured_data['canonical_sections'].keys())}")
            for key, value in structured_data['canonical_sections'].items():
                logger.debug(f"[PARSER] Canonical '{key}': {len(value)} chars")
        else:
            logger.warning("[PARSER] No canonical_sections generated!")
        
        with time_phase("summary_build", timing_bucket):
            structured_data['summary'] = self._generate_accurate_summary(sections, cleaned_lines)
        
        # Set meta information
        structured_data['meta'] = {
            'pages_total': pages_total,
            'pages_processed': pages_processed,
            'source_file': os.path.basename(pdf_file_path)
        }

        # Save cache
        with time_phase("save_cache", timing_bucket):
            try:
                with open(cache_path, 'w', encoding='utf-8') as cf:
                    json.dump(structured_data, cf, indent=2, ensure_ascii=False)
            except Exception:
                pass
        
        return structured_data

    def _extract_text_with_layout(self, pdf) -> List[Dict[str, Any]]:
        """Extract text with layout metadata (font size, weight, coordinates)."""
        text_elements = []
        
        for page_num, page in enumerate(pdf.pages[:MAX_PAGES]):
            try:
                # Extract text objects with layout information
                chars = page.chars
                if not chars:
                    # Fallback to basic text extraction
                    text = page.extract_text()
                    if not text and not self.disable_ocr and OCR_ALLOWED:
                        text = self._ocr_page(page)
                    if text:
                        # Create basic text elements without layout info
                        lines = text.split('\n')
                        for i, line in enumerate(lines):
                            if line.strip():
                                text_elements.append({
                                    'text': line.strip(),
                                    'page': page_num,
                                    'y': i * 20,  # Approximate position
                                    'x': 0,
                                    'font_size': 12,  # Default
                                    'font_name': 'unknown',
                                    'is_bold': False,
                                    'is_italic': False
                                })
                    continue
                
                # Group characters into text elements by proximity and formatting
                text_blocks = self._group_chars_into_blocks(chars, page_num)
                text_elements.extend(text_blocks)
                
            except Exception as e:
                logger.warning(f"Error processing page {page_num}: {str(e)}")
                continue
                
        return text_elements

    def _extract_text_fast(self, pdf_file_path: str) -> Optional[Tuple[List[Dict[str, Any]], Dict[int, float]]]:
        """Fast path: use PyMuPDF to extract text blocks with font information.
        
        IMPORTANT: Does NOT pre-sort elements by (y, x) to preserve layout information.
        Column detection happens later and handles proper reading order.
        
        Returns:
            Tuple of (text_elements, page_widths) or None if extraction fails.
            page_widths maps page number to page width in points.
        """
        if not (USE_PYMUPDF and fitz is not None):
            return None
        try:
            doc = fitz.open(pdf_file_path)
            text_elements: List[Dict[str, Any]] = []
            page_widths: Dict[int, float] = {}
            font_info_available = False
            
            for page_num, page in enumerate(doc[:MAX_PAGES]):
                # Store page width for column detection
                page_widths[page_num] = float(page.rect.width)
                
                # Try to get text with font information using "dict" format
                try:
                    text_dict = page.get_text("dict")
                    if text_dict and text_dict.get('blocks'):
                        font_info_available = True
                        for block in text_dict['blocks']:
                            if block.get('type') == 0:  # Text block
                                bbox = block.get('bbox', [0, 0, 0, 0])
                                x0, y0, x1, y1 = bbox[0], bbox[1], bbox[2], bbox[3]
                                block_width = x1 - x0
                                
                                # Extract text spans with font info
                                for line in block.get('lines', []):
                                    line_bbox = line.get('bbox', [x0, y0, x1, y1])
                                    line_y = line_bbox[1]
                                    line_x = line_bbox[0]
                                    line_width = line_bbox[2] - line_bbox[0]
                                    
                                    # Collect spans in this line
                                    line_text_parts = []
                                    line_font_size = 12.0
                                    line_is_bold = False
                                    line_is_italic = False
                                    
                                    for span in line.get('spans', []):
                                        span_text = span.get('text', '').strip()
                                        if span_text:
                                            line_text_parts.append(span_text)
                                            # Use font size from span if available
                                            span_font_size = span.get('size', 12.0)
                                            if span_font_size and span_font_size > 0:
                                                line_font_size = max(line_font_size, span_font_size)
                                            # Check for bold/italic
                                            font_flags = span.get('flags', 0)
                                            if font_flags & 16:  # Bold flag
                                                line_is_bold = True
                                            if font_flags & 1:  # Italic flag
                                                line_is_italic = True
                                    
                                    if line_text_parts:
                                        line_text = ' '.join(line_text_parts)
                                        norm = self._normalize_text(line_text)
                                        if norm.strip():
                                            text_elements.append({
                                                'text': norm.strip(),
                                                'page': page_num,
                                                'y': float(line_y),
                                                'x': float(line_x),
                                                'width': float(line_width),  # Track element width
                                                'font_size': line_font_size,
                                                'font_name': 'unknown',
                                                'is_bold': line_is_bold,
                                                'is_italic': line_is_italic
                                            })
                        continue  # Successfully used dict format, skip blocks fallback
                except Exception as e:
                    logger.debug(f"Failed to extract font info from page {page_num}: {e}")
                
                # Fallback to blocks if dict format failed
                blocks = page.get_text("blocks") or []
                for blk in blocks:
                    if len(blk) < 5:
                        continue
                    x0, y0, x1, y1, text = blk[0], blk[1], blk[2], blk[3], blk[4]
                    if not text or not str(text).strip():
                        continue
                    block_width = x1 - x0
                    # Normalize and split by lines
                    norm = self._normalize_text(str(text))
                    for ln in norm.split('\n'):
                        if ln.strip():
                            text_elements.append({
                                'text': ln.strip(),
                                'page': page_num,
                                'y': float(y0),
                                'x': float(x0),
                                'width': float(block_width),
                                'font_size': 12,
                                'font_name': 'unknown',
                                'is_bold': False,
                                'is_italic': False
                            })
            
            doc.close()
            
            # NOTE: Do NOT sort here! Column detection will handle proper reading order.
            # Sorting by (y, x) destroys two-column layouts by reading horizontally.
            
            if font_info_available:
                logger.debug(f"[FONT_EXTRACT] Extracted font information for {len(text_elements)} elements")
            else:
                logger.debug(f"[FONT_EXTRACT] No font information available, using defaults")
            
            logger.info(f"[PARSER] fitz fast path extracted {len(text_elements)} text elements")
            
            # Quick probe: if we have reasonable amount of text, return it
            if sum(len(te['text']) for te in text_elements) > 50:
                return (text_elements, page_widths)
            return None
        except Exception as e:
            logger.debug(f"PyMuPDF fast extract failed: {e}")
            return None

    def _extract_text_pymupdf4llm(self, pdf_file_path: str) -> Tuple[Optional[List[Dict[str, Any]]], Optional[List[str]]]:
        """Extract text using pymupdf4llm for markdown-formatted output."""
        try:
            import pymupdf4llm
            markdown_text = pymupdf4llm.to_markdown(pdf_file_path, pages=range(MAX_PAGES))
            
            if not markdown_text or not markdown_text.strip():
                logger.debug(f"[PARSER] PyMuPDF4LLM returned empty markdown (length: {len(markdown_text) if markdown_text else 0})")
                return None, None
            
            # Split markdown into lines for section detection
            markdown_lines = markdown_text.split('\n')
            
            # Also create text_elements for compatibility with existing code
            text_elements: List[Dict[str, Any]] = []
            page_num = 0
            y_pos = 0
            
            for line in markdown_lines:
                if line.strip():
                    # Markdown headers (##) are strong section indicators
                    is_header = line.strip().startswith('##')
                    text_elements.append({
                        'text': line.strip(),
                        'page': page_num,
                        'y': float(y_pos),
                        'x': 0.0,
                        'font_size': 14.0 if is_header else 12.0,
                        'font_name': 'unknown',
                        'is_bold': is_header,
                        'is_italic': False
                    })
                    y_pos += 20
            
            logger.info(f"[PARSER] PyMuPDF4LLM extracted {len(text_elements)} elements from {len(markdown_lines)} markdown lines")
            return text_elements, markdown_lines
        except Exception as e:
            logger.warning(f"[PARSER] pymupdf4llm extract failed: {e}")
            import traceback
            logger.debug(f"[PARSER] PyMuPDF4LLM traceback: {traceback.format_exc()}")
            return None, None

    def _group_chars_into_blocks(self, chars: List[Dict], page_num: int) -> List[Dict[str, Any]]:
        """Group individual characters into text blocks based on proximity and formatting."""
        if not chars:
            return []
        
        # Sort characters by y-coordinate, then x-coordinate
        sorted_chars = sorted(chars, key=lambda c: (c['top'], c['x0']))
        
        text_blocks = []
        current_block = {
            'chars': [],
            'font_size': None,
            'font_name': None,
            'is_bold': False,
            'is_italic': False,
            'page': page_num
        }
        
        for char in sorted_chars:
            # Check if this character belongs to the current block
            if current_block['chars']:
                last_char = current_block['chars'][-1]
                
                # Check if characters are close enough to be in same line
                y_distance = abs(char['top'] - last_char['top'])
                x_distance = char['x0'] - last_char['x1']
                
                # Same line if y-distance is small and x-distance is reasonable
                # Reduced x_distance threshold for better fragment detection
                if y_distance < 5 and x_distance < 20:
                    current_block['chars'].append(char)
                else:
                    # Start new block
                    if current_block['chars']:
                        text_blocks.append(self._create_text_element(current_block))
                    current_block = {
                        'chars': [char],
                        'font_size': None,
                        'font_name': None,
                        'is_bold': False,
                        'is_italic': False,
                        'page': page_num
                    }
            else:
                current_block['chars'].append(char)
        
        # Add the last block
        if current_block['chars']:
            text_blocks.append(self._create_text_element(current_block))
        
        # CRITICAL: Post-process text blocks to fix fragmentation
        return self._reconstruct_fragmented_text(text_blocks)

    def _reconstruct_fragmented_text(self, text_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Reconstruct fragmented text, especially ordinals and bullet points."""
        if not text_blocks:
            return text_blocks
        
        reconstructed_blocks = []
        i = 0
        
        while i < len(text_blocks):
            current_block = text_blocks[i]
            current_text = current_block.get('text', '').strip()
            
            # Look ahead for fragments to reconstruct
            reconstructed_text = self._reconstruct_text_fragments(text_blocks, i)
            
            if reconstructed_text != current_text:
                # Text was reconstructed, update the block
                current_block['text'] = reconstructed_text
                # Skip the blocks that were merged
                fragments_used = self._count_fragments_used(text_blocks, i, reconstructed_text)
                i += fragments_used
            else:
                i += 1
            
            reconstructed_blocks.append(current_block)
        
        return reconstructed_blocks

    def _reconstruct_text_fragments(self, text_blocks: List[Dict[str, Any]], start_index: int) -> str:
        """Reconstruct text fragments starting from a given index."""
        if start_index >= len(text_blocks):
            return ""
        
        current_text = text_blocks[start_index].get('text', '').strip()
        
        # Pattern 1: Ordinal number reconstruction (1 + st, 2 + nd, etc.)
        if re.match(r'^\d+$', current_text):
            reconstructed = self._reconstruct_ordinal(text_blocks, start_index)
            if reconstructed != current_text:
                return reconstructed
        
        # Pattern 2: Bullet point reconstruction
        if current_text in ['st', 'nd', 'rd', 'th'] and start_index > 0:
            prev_text = text_blocks[start_index - 1].get('text', '').strip()
            if re.match(r'^\d+$', prev_text):
                # This is part of an ordinal that should have been caught earlier
                return current_text
        
        # Pattern 3: Fragmented bullet points
        if current_text in ['•', '●', '-', '*'] or re.match(r'^[•●\-\*]', current_text):
            reconstructed = self._reconstruct_bullet_point(text_blocks, start_index)
            if reconstructed != current_text:
                return reconstructed
        
        # Pattern 4: Fragmented sentences (look for continuation patterns)
        if start_index < len(text_blocks) - 1:
            next_text = text_blocks[start_index + 1].get('text', '').strip()
            if self._should_merge_fragments(current_text, next_text):
                reconstructed = self._merge_sentence_fragments(text_blocks, start_index)
                if reconstructed != current_text:
                    return reconstructed
        
        return current_text

    def _reconstruct_ordinal(self, text_blocks: List[Dict[str, Any]], start_index: int) -> str:
        """Reconstruct ordinal numbers (1st, 2nd, 3rd, etc.)."""
        if start_index >= len(text_blocks):
            return ""
        
        current_text = text_blocks[start_index].get('text', '').strip()
        
        # Check if current text is a number
        if not re.match(r'^\d+$', current_text):
            return current_text
        
        # Look for ordinal suffix in next blocks
        ordinal_suffixes = ['st', 'nd', 'rd', 'th']
        reconstructed_parts = [current_text]
        
        for i in range(start_index + 1, min(start_index + 3, len(text_blocks))):
            next_text = text_blocks[i].get('text', '').strip()
            
            if next_text in ordinal_suffixes:
                # Found ordinal suffix
                reconstructed_parts.append(next_text)
                
                # Check for additional fragments (like "and 2nd")
                if i + 1 < len(text_blocks):
                    following_text = text_blocks[i + 1].get('text', '').strip()
                    if following_text.lower() in ['and', '&', ',']:
                        reconstructed_parts.append(' ' + following_text)
                        
                        # Look for next number
                        if i + 2 < len(text_blocks):
                            next_num_text = text_blocks[i + 2].get('text', '').strip()
                            if re.match(r'^\d+$', next_num_text):
                                reconstructed_parts.append(' ' + next_num_text)
                                
                                # Look for its suffix
                                if i + 3 < len(text_blocks):
                                    next_suffix = text_blocks[i + 3].get('text', '').strip()
                                    if next_suffix in ordinal_suffixes:
                                        reconstructed_parts.append(next_suffix)
                
                break
            elif next_text and not next_text.isspace():
                # If we hit non-whitespace that's not an ordinal suffix, stop
                break
        
        return ''.join(reconstructed_parts)

    def _reconstruct_bullet_point(self, text_blocks: List[Dict[str, Any]], start_index: int) -> str:
        """Reconstruct bullet points with proper formatting."""
        if start_index >= len(text_blocks):
            return ""
        
        current_text = text_blocks[start_index].get('text', '').strip()
        
        # Standardize bullet character
        if current_text in ['•', '●', '-', '*', '▪', '▫']:
            bullet_char = '●'
        elif re.match(r'^[•●\-\*]', current_text):
            bullet_char = '●'
            # Keep any text after the bullet
            remaining_text = current_text[1:].strip()
            if remaining_text:
                return bullet_char + ' ' + remaining_text
        else:
            return current_text
        
        # Look for content following the bullet
        content_parts = []
        
        for i in range(start_index + 1, min(start_index + 5, len(text_blocks))):
            next_text = text_blocks[i].get('text', '').strip()
            
            if next_text:
                # Check if this looks like bullet content
                if not re.match(r'^[•●\-\*]', next_text):
                    content_parts.append(next_text)
                else:
                    # Hit another bullet, stop
                    break
            else:
                # Empty text, might be spacing
                continue
        
        if content_parts:
            return bullet_char + ' ' + ' '.join(content_parts)
        else:
            return bullet_char + ' '

    def _should_merge_fragments(self, current_text: str, next_text: str) -> bool:
        """Determine if two text fragments should be merged."""
        if not current_text or not next_text:
            return False
        
        # Merge if current ends with incomplete word and next starts with completion
        if (current_text[-1].isalnum() and next_text[0].isalnum() and 
            not current_text.endswith('.') and not next_text[0].isupper()):
            return True
        
        # Merge ordinal fragments
        if re.match(r'^\d+$', current_text) and next_text in ['st', 'nd', 'rd', 'th']:
            return True
        
        # Merge if current is a fragment and next continues the sentence
        if (len(current_text) < 3 and current_text.isalnum() and 
            next_text[0].islower()):
            return True
        
        return False

    def _merge_sentence_fragments(self, text_blocks: List[Dict[str, Any]], start_index: int) -> str:
        """Merge sentence fragments into coherent text."""
        if start_index >= len(text_blocks):
            return ""
        
        merged_parts = [text_blocks[start_index].get('text', '').strip()]
        
        for i in range(start_index + 1, min(start_index + 5, len(text_blocks))):
            next_text = text_blocks[i].get('text', '').strip()
            
            if next_text and self._should_merge_fragments(merged_parts[-1], next_text):
                merged_parts.append(next_text)
            else:
                break
        
        return ' '.join(merged_parts)

    def _count_fragments_used(self, text_blocks: List[Dict[str, Any]], start_index: int, reconstructed_text: str) -> int:
        """Count how many text blocks were used in reconstruction."""
        original_text = text_blocks[start_index].get('text', '').strip()
        
        if reconstructed_text == original_text:
            return 1
        
        # Count based on the complexity of reconstruction
        fragments_count = 1
        
        # Check for ordinal reconstruction
        if re.match(r'^\d+(st|nd|rd|th)', reconstructed_text):
            fragments_count = 2
            
            # Check for compound ordinals (1st and 2nd)
            if ' and ' in reconstructed_text or ' & ' in reconstructed_text:
                fragments_count = 5
        
        # Check for bullet point reconstruction
        elif reconstructed_text.startswith('● '):
            content_words = len(reconstructed_text.split()) - 1  # Minus the bullet
            fragments_count = min(content_words, 3)
        
        return fragments_count

    def _create_text_element(self, block: Dict) -> Dict[str, Any]:
        """Create a text element from a block of characters."""
        if not block['chars']:
            return {}
        
        # Combine text
        text = ''.join(char['text'] for char in block['chars'])
        
        # Calculate position (use first character's position)
        first_char = block['chars'][0]
        x = first_char['x0']
        y = first_char['top']
        
        # Determine font properties (use most common)
        font_sizes = [char.get('size', 12) for char in block['chars']]
        font_names = [char.get('fontname', 'unknown') for char in block['chars']]
        
        # Use median font size to avoid outliers
        font_sizes.sort()
        font_size = font_sizes[len(font_sizes) // 2] if font_sizes else 12
        
        # Most common font name
        font_name = max(set(font_names), key=font_names.count) if font_names else 'unknown'
        
        # Check for bold/italic (simplified detection)
        is_bold = any('bold' in fn.lower() for fn in font_names)
        is_italic = any('italic' in fn.lower() for fn in font_names)
        
        return {
            'text': text.strip(),
            'page': block['page'],
            'x': x,
            'y': y,
            'font_size': font_size,
            'font_name': font_name,
            'is_bold': is_bold,
            'is_italic': is_italic,
            'char_count': len(block['chars'])
        }

    def _calculate_font_statistics(self, text_elements: List[Dict]) -> Dict[str, Any]:
        """Calculate font statistics for ML feature extraction."""
        if not text_elements:
            return {}
        
        font_sizes = [elem.get('font_size', 12) for elem in text_elements]
        font_sizes = [size for size in font_sizes if size > 0]
        
        if not font_sizes:
            return {}
        
        return {
            'mean_font_size': sum(font_sizes) / len(font_sizes),
            'median_font_size': sorted(font_sizes)[len(font_sizes) // 2],
            'max_font_size': max(font_sizes),
            'min_font_size': min(font_sizes),
            'font_size_std': (sum((x - sum(font_sizes) / len(font_sizes)) ** 2 for x in font_sizes) / len(font_sizes)) ** 0.5,
            'bold_elements_count': sum(1 for elem in text_elements if elem.get('is_bold', False)),
            'italic_elements_count': sum(1 for elem in text_elements if elem.get('is_italic', False))
        }

    def _analyze_layout(self, text_elements: List[Dict]) -> Dict[str, Any]:
        """Analyze layout patterns for ML feature extraction."""
        if not text_elements:
            return {}
        
        # Analyze positioning patterns
        y_positions = [elem.get('y', 0) for elem in text_elements]
        x_positions = [elem.get('x', 0) for elem in text_elements]
        
        # Detect potential columns
        x_positions_sorted = sorted(x_positions)
        column_boundaries = self._detect_column_boundaries(x_positions_sorted)
        
        return {
            'total_elements': len(text_elements),
            'y_range': max(y_positions) - min(y_positions) if y_positions else 0,
            'x_range': max(x_positions) - min(x_positions) if x_positions else 0,
            'column_boundaries': column_boundaries,
            'estimated_columns': len(column_boundaries) + 1
        }

    def _detect_column_boundaries(self, x_positions: List[float]) -> List[float]:
        """Detect column boundaries using clustering of x-positions."""
        if len(x_positions) < 2:
            return []
        
        # Simple clustering: find gaps larger than average
        gaps = [x_positions[i+1] - x_positions[i] for i in range(len(x_positions)-1)]
        if not gaps:
            return []
        
        avg_gap = sum(gaps) / len(gaps)
        threshold = avg_gap * 2  # Gap must be 2x average to be a column boundary
        
        boundaries = []
        for i, gap in enumerate(gaps):
            if gap > threshold:
                boundaries.append(x_positions[i] + gap / 2)
        
        return boundaries

    def _extract_all_lines(self, pdf) -> List[str]:
        """Extract all lines from PDF with OCR fallback."""
        all_lines = []
        
        for page_num, page in enumerate(pdf.pages[:MAX_PAGES]):
            try:
                # Try text extraction first
                text = page.extract_text()
                if (not text or text.strip() == '') and not self.disable_ocr and OCR_ALLOWED:
                    # OCR fallback for image-based PDFs
                    text = self._ocr_page(page)
                
                if text:
                    # Handle multi-column layouts
                    lines = self._handle_columns(page, text)
                    all_lines.extend(lines)
                    
            except Exception as e:
                logger.warning(f"Error processing page {page_num}: {str(e)}")
                continue
                
        return all_lines

    def _handle_columns(self, page, text: str) -> List[str]:
        """Handle multi-column layouts by detecting and merging columns properly."""
        lines = text.split('\n')
        
        # Try to detect if this is a multi-column layout
        if page.width > 700:  # Wide page, might have columns
            try:
                # Split page into left and right halves
                mid_point = page.width / 2
                left_bbox = (0, 0, mid_point, page.height)
                right_bbox = (mid_point, 0, page.width, page.height)
                
                left_text = page.within_bbox(left_bbox).extract_text() or ""
                right_text = page.within_bbox(right_bbox).extract_text() or ""
                
                # If both columns have substantial content, merge them properly
                if len(left_text.strip()) > 50 and len(right_text.strip()) > 50:
                    left_lines = [line.strip() for line in left_text.split('\n') if line.strip()]
                    right_lines = [line.strip() for line in right_text.split('\n') if line.strip()]
                    
                    # Merge columns by interleaving based on vertical position
                    merged_lines = []
                    merged_lines.extend(left_lines)
                    merged_lines.extend(right_lines)
                    return merged_lines
                    
            except Exception as e:
                logger.warning(f"Column detection failed: {str(e)}")
        
        return [line.strip() for line in lines if line.strip()]

    def _clean_lines(self, lines: List[str]) -> List[str]:
        """Clean and filter lines, removing noise and page artifacts."""
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Skip filtered patterns
            if any(re.match(pattern, line, re.IGNORECASE) for pattern in self.filter_patterns):
                continue
                
            # Skip very short lines that are likely artifacts (but keep bullet points)
            if len(line) < 2 and not re.match(r'^[•\-\*]', line):
                continue
                
            cleaned_lines.append(line)
            
        return cleaned_lines
    
    @staticmethod
    def _normalize_header_candidate(text: str) -> str:
        candidate = (text or '').strip().lower()
        candidate = re.sub(r'^[^a-z0-9]+', '', candidate)
        candidate = re.sub(r'[^a-z0-9]+$', '', candidate)
        candidate = re.sub(r'\s+', ' ', candidate)
        return candidate

    def _build_section_alias_lookup(self) -> Dict[str, str]:
        lookup: Dict[str, str] = {}
        for canonical, aliases in SECTION_SYNONYMS.items():
            for alias in aliases:
                norm_alias = self._normalize_header_candidate(alias)
                if norm_alias:
                    lookup[norm_alias] = canonical
        return lookup

    def _canonicalize_header(self, header: str, content: List[str]) -> str:
        candidate = self._normalize_header_candidate(header)
        if candidate and candidate in self._section_alias_lookup:
            return self._section_alias_lookup[candidate]

        # Fuzzy match against known aliases
        best_label = None
        best_ratio = 0.0
        for canonical, aliases in SECTION_SYNONYMS.items():
            for alias in aliases:
                alias_norm = self._normalize_header_candidate(alias)
                if not alias_norm:
                    continue
                if alias_norm == candidate and alias_norm:
                    return canonical
                ratio = difflib.SequenceMatcher(None, candidate or '', alias_norm).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_label = canonical
        if best_ratio >= 0.90 and best_label:
            return best_label

        # Content-based heuristics - but be more conservative
        # Only use content hints if we have substantial content and clear signals
        content_blob = ' '.join(content or []).lower()
        if content_blob and len(content_blob) > 50:  # Require substantial content
            hint_scores: Dict[str, int] = {}
            for canonical in CANONICAL_SECTION_PRIORITY:
                hints = CONTENT_HINTS.get(canonical, [])
                if not hints:
                    continue
                hits = sum(1 for hint in hints if hint in content_blob)
                if hits:
                    hint_scores[canonical] = hits
            if hint_scores:
                best_canonical, hits = max(hint_scores.items(), key=lambda kv: kv[1])
                # Require more hits for experience to avoid false positives
                minimum_hits = 3 if best_canonical == 'experience' else 2
                if hits >= minimum_hits:
                    return best_canonical

        return 'misc'

    def _tag_sections_with_canonical_labels(self, sections: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        if not sections:
            return [], {}

        aggregated_lists: Dict[str, List[str]] = {key: [] for key in CANONICAL_SECTION_PRIORITY}
        tagged_sections: List[Dict[str, Any]] = []

        for section in sections:
            header = section.get('header', '')
            content = section.get('content', []) or []
            canonical = self._canonicalize_header(header, content)
            enriched = dict(section)
            enriched['canonical'] = canonical
            tagged_sections.append(enriched)
            if content:
                aggregated_lists.setdefault(canonical, []).extend(content)

        aggregated = {
            key: ' '.join(values).strip()
            for key, values in aggregated_lists.items()
            if values
        }
        
        # FALLBACK: Check if Education has swallowed Experience
        # If Experience is empty but Education is >2000 chars, try to re-split
        experience_text = aggregated.get('experience', '')
        education_text = aggregated.get('education', '')
        
        if len(experience_text.strip()) == 0 and len(education_text) > 2000:
            logger.warning(f"[BUCKET_LEAK] Experience is empty but Education has {len(education_text)} chars - attempting re-split")
            
            # Find the Education section(s) and try to re-split
            new_tagged_sections = []
            new_aggregated_lists: Dict[str, List[str]] = {key: [] for key in CANONICAL_SECTION_PRIORITY}
            
            for section in tagged_sections:
                if section.get('canonical') == 'education':
                    content = section.get('content', [])
                    if content:
                        # Try to re-split this bloated section
                        resplit_sections = self._resplit_bloated_section(content, 'education')
                        
                        for resplit in resplit_sections:
                            resplit_header = resplit.get('header', '')
                            resplit_content = resplit.get('content', [])
                            resplit_canonical = self._canonicalize_header(resplit_header, resplit_content)
                            
                            # Override canonical if header explicitly says Experience/Education
                            if 'experience' in resplit_header.lower():
                                resplit_canonical = 'experience'
                            elif 'education' in resplit_header.lower():
                                resplit_canonical = 'education'
                            
                            enriched_resplit = dict(resplit)
                            enriched_resplit['canonical'] = resplit_canonical
                            new_tagged_sections.append(enriched_resplit)
                            
                            if resplit_content:
                                new_aggregated_lists.setdefault(resplit_canonical, []).extend(resplit_content)
                else:
                    new_tagged_sections.append(section)
                    canonical = section.get('canonical', 'misc')
                    content = section.get('content', [])
                    if content:
                        new_aggregated_lists.setdefault(canonical, []).extend(content)
            
            # Check if re-split was successful
            new_experience = ' '.join(new_aggregated_lists.get('experience', []))
            if len(new_experience.strip()) > 0:
                logger.info(f"[BUCKET_LEAK] Re-split successful! Experience now has {len(new_experience)} chars")
                tagged_sections = new_tagged_sections
                aggregated = {
                    key: ' '.join(values).strip()
                    for key, values in new_aggregated_lists.items()
                    if values
                }
            else:
                logger.warning("[BUCKET_LEAK] Re-split did not recover Experience content")
        
        return tagged_sections, aggregated

    def _enforce_core_sections(self, sections: List[Dict[str, Any]], lines: List[str]) -> List[Dict[str, Any]]:
        """Ensure core sections (experience, skills, education) exist before returning."""
        if not sections:
            return sections

        label_counts: Dict[str, int] = {}
        total_content_length = 0
        
        for section in sections:
            canonical = self._canonicalize_header(section.get('header', ''), section.get('content', []))
            section['canonical'] = canonical
            if canonical:
                label_counts[canonical] = label_counts.get(canonical, 0) + 1
            # Track total content to detect if one section has everything
            content = section.get('content', [])
            if isinstance(content, list):
                total_content_length += sum(len(str(item)) for item in content)
            elif isinstance(content, str):
                total_content_length += len(content)

        # Check if one section has too much content (indicates misclassification)
        for section in sections:
            canonical = section.get('canonical', '')
            content = section.get('content', [])
            section_length = 0
            if isinstance(content, list):
                section_length = sum(len(str(item)) for item in content)
            elif isinstance(content, str):
                section_length = len(content)
            
            # If one section has >80% of content and it's not clearly labeled, redistribute
            if total_content_length > 0 and section_length / total_content_length > 0.8:
                logger.warning(f"[ENFORCE_CORE] Section '{canonical}' has {section_length}/{total_content_length} chars ({100*section_length/total_content_length:.1f}%) - attempting to split")
                if canonical == 'experience' and len(sections) == 1:
                    # This is the problem case - everything in experience
                    # Try to split it properly
                    logger.warning(f"[ENFORCE_CORE] Detected single large section with {section_length} chars, attempting to split")
                    # Don't just recover - try to split the existing section
                    split_sections = self._split_large_section(section, lines)
                    if split_sections and len(split_sections) > 1:
                        logger.info(f"[ENFORCE_CORE] Successfully split large section into {len(split_sections)} sections")
                        return split_sections
                    else:
                        logger.warning(f"[ENFORCE_CORE] Failed to split large section")

        core_labels = ['experience', 'skills', 'education']
        missing = [label for label in core_labels if label_counts.get(label, 0) == 0]
        if not missing:
            return sections

        # Only recover if we don't have a single massive section
        if len(sections) > 1 or total_content_length < 1000:
            recovered = self._recover_sections_from_keywords(lines, missing)
            if recovered:
                sections.extend(recovered)
        return sections
    
    def _split_large_section(self, section: Dict[str, Any], all_lines: List[str]) -> Optional[List[Dict[str, Any]]]:
        """Split a large section that likely contains multiple sections."""
        content = section.get('content', [])
        if not content:
            return None
        
        # Convert content to lines if needed
        if isinstance(content, list):
            content_lines = []
            for item in content:
                if isinstance(item, str):
                    content_lines.extend(item.split('\n'))
                else:
                    content_lines.append(str(item))
        else:
            content_lines = content.split('\n')
        
        # Try aggressive detection on this content
        split_sections = self._detect_inline_sections_aggressive(content_lines)
        if not split_sections or len(split_sections) <= 1:
            split_sections = self._split_by_keywords(content_lines)
        
        return split_sections if split_sections and len(split_sections) > 1 else None

    def _recover_sections_from_keywords(self, lines: List[str], missing_labels: List[str]) -> List[Dict[str, Any]]:
        """Recover sections by scanning for inline headers when parser missed them."""
        if not lines or not missing_labels:
            return []

        missing = set(missing_labels)
        markers: List[Tuple[int, str, str]] = []
        for idx, raw_line in enumerate(lines):
            candidate = raw_line.strip()
            if not candidate:
                continue
            canonical = self._canonicalize_header(candidate, [])
            if canonical in missing:
                markers.append((idx, candidate, canonical))
                missing.discard(canonical)
                if not missing:
                    break

        if not markers:
            return []

        recovered_sections: List[Dict[str, Any]] = []
        markers.append((len(lines), '', ''))
        for i in range(len(markers) - 1):
            start_idx = markers[i][0] + 1
            end_idx = markers[i + 1][0]
            chunk = [ln.strip() for ln in lines[start_idx:end_idx] if ln.strip()]
            grouped = self._group_content_lines(chunk)
            canonical = markers[i][2]
            if grouped and canonical:
                recovered_sections.append({
                    'header': markers[i][1] or canonical.title(),
                    'content': grouped,
                    'canonical': canonical
                })
        return recovered_sections

    def _looks_like_inline_header(self, candidate: str) -> bool:
        cand = (candidate or '').strip()
        if not cand:
            return False

        cand = re.sub(r'^[\u2022\u25CF\-\*\s]+', '', cand)
        cand = cand.strip()
        if not cand:
            return False

        if self._header_score(cand) >= HEADER_CANDIDATE_THRESHOLD:
            return True

        words = cand.split()
        alpha_chars = sum(1 for ch in cand if ch.isalpha())
        uppercase_chars = sum(1 for ch in cand if ch.isupper())
        uppercase_ratio = (uppercase_chars / alpha_chars) if alpha_chars else 0.0
        candidate_lower = cand.lower()
        keyword_hit = any(keyword in candidate_lower for keyword in self._inline_header_keywords)

        if (len(cand) <= 80 and len(words) <= 8 and uppercase_ratio >= 0.6):
            return True
        if (len(cand) <= 60 and keyword_hit):
            return True
        return False

    def _normalize_lines(self, lines: List[str]) -> List[str]:
        """Normalization: NFKC, fix ligatures, de-hyphen at line breaks, unify bullets, split inline headers."""
        normed: List[str] = []
        for ln in lines[:]:
            t = self._normalize_text(ln)
            if t:
                # Handle case where normalize_text inserted newlines (for inline headers)
                if '\n' in t:
                    for sub_line in t.split('\n'):
                        sub_line = sub_line.strip()
                        if sub_line:
                            normed.append(sub_line)
                else:
                    normed.append(t)
        return normed

    def _normalize_text(self, text: str) -> str:
        t = unicodedata.normalize('NFKC', text or '')
        # Ligatures
        t = (t.replace('ﬁ', 'fi').replace('ﬂ', 'fl').replace('ﬀ', 'ff')
               .replace('ﬃ', 'ffi').replace('ﬄ', 'ffl'))
        # Fix spaced headers: "E X P E R I E N C E" -> "EXPERIENCE"
        # Pattern: 3+ uppercase letters separated by spaces
        spaced_header_pattern = r'\b([A-Z]\s){3,}[A-Z]\b'
        def fix_spaced_header(match):
            spaced_text = match.group(0)
            # Remove spaces between letters
            fixed = re.sub(r'\s+', '', spaced_text)
            return fixed
        t = re.sub(spaced_header_pattern, fix_spaced_header, t)
        
        # FIX: Split inline section headers that appear mid-line (from two-column layouts)
        # These patterns detect when a section header is embedded in content
        inline_header_patterns = [
            # "...text EXPERIENCE Job Title..."
            (r'(\S)\s+(EXPERIENCE|WORK EXPERIENCE|PROFESSIONAL EXPERIENCE)\s+([A-Z])', r'\1\n\2\n\3'),
            # "...text EDUCATION University..."
            (r'(\S)\s+(EDUCATION|ACADEMIC BACKGROUND|QUALIFICATIONS)\s+([A-Z])', r'\1\n\2\n\3'),
            # "...text SKILLS Programming..."
            (r'(\S)\s+(SKILLS|TECHNICAL SKILLS|CORE COMPETENCIES)\s+([A-Z])', r'\1\n\2\n\3'),
            # "...text PROJECTS Project Name..."
            (r'(\S)\s+(PROJECTS|PROJECT EXPERIENCE|PROJECTS & PUBLICATIONS)\s+([A-Z])', r'\1\n\2\n\3'),
            # "...text CERTIFICATIONS AWS..."
            (r'(\S)\s+(CERTIFICATIONS|CERTIFICATES|LICENSES)\s+([A-Z])', r'\1\n\2\n\3'),
        ]
        for pattern, replacement in inline_header_patterns:
            if re.search(pattern, t, re.IGNORECASE):
                t = re.sub(pattern, replacement, t, flags=re.IGNORECASE)
                logger.debug(f"[INLINE_HEADER_SPLIT] Split inline header in: {t[:80]}...")
        
        # De-hyphenation across line breaks (best effort when provided raw blocks)
        t = re.sub(r'-\s*\n\s*', '', t)
        # Unify bullets
        t = re.sub(r'^[•▪▫\-\*]+\s*', '● ', t)
        return t.strip()

    def _strip_repeated_headers_footers(self, lines: List[str]) -> List[str]:
        """Remove lines that repeat on most pages (likely headers/footers)."""
        # Heuristic: count duplicates and drop those appearing > 50% of lines window-sized by pages
        if not lines:
            return lines
        freq: Dict[str, int] = {}
        for ln in lines:
            k = ln.strip().lower()
            freq[k] = freq.get(k, 0) + 1
        threshold = max(3, int(0.5 * MAX_PAGES))  # rough
        return [ln for ln in lines if freq.get(ln.strip().lower(), 0) <= threshold]

    def _split_sections_on_embedded_headers(self, sections):
        """Split sections if a new header is found within the content."""
        new_sections = []
        for section in sections:
            content = section['content']
            current_header = section['header']
            current_content = []
            for line in content:
                line_clean = line.strip()
                if not line_clean:
                    if current_content:
                        current_content.append(line)
                    continue

                is_header = self._looks_like_inline_header(line_clean) or self._is_actual_section_header(line_clean, 0, [line_clean])
                if is_header:
                    if current_content:
                        new_sections.append({'header': current_header, 'content': current_content})
                    current_header = line_clean
                    current_content = []
                else:
                    current_content.append(line)
            if current_content:
                new_sections.append({'header': current_header, 'content': current_content})
        # Recursively apply until no more splits
        if len(new_sections) == len(sections):
            return new_sections
        else:
            return self._split_sections_on_embedded_headers(new_sections)

    def _detect_sections_and_group_content(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Detect actual section headers and group content properly."""
        sections = []
        logger.info(f"[SECTION_DETECT] Starting section detection on {len(lines)} lines")
        
        # First pass: score potential headers
        candidates: List[Tuple[int, str, float, Dict[str, Any]]] = []  # (idx, text, score, feat)
        prev_y = None
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Check for markdown headers first (## or ###)
            if line_stripped.startswith('##'):
                # Markdown header - very high confidence
                header_text = line_stripped.lstrip('#').strip()
                if header_text:
                    candidates.append((i, header_text, 1.0, {'font_size': 16.0, 'x': 0.0, 'is_markdown': True}))
                    continue
            
            font_size = is_bold = x = y = page = left_margin = None
            if i < len(self._last_line_positions):
                font_size, is_bold, x, y, page = self._last_line_positions[i]
                if page is not None and page in self._left_margin_by_page:
                    left_margin = self._left_margin_by_page[page]
            score = self._header_score(line, font_size=font_size, is_bold=is_bold, x=x, y=y, prev_y=prev_y, left_margin=left_margin)
            prev_y = y if y is not None else prev_y
            # Standalone constraint: allow trailing ':' but not '.'
            standalone_ok = not TRAILING_PERIOD_RE.search(line_stripped) or line_stripped.endswith(':')
            if score >= HEADER_CANDIDATE_THRESHOLD and standalone_ok:
                candidates.append((i, line_stripped, score, {'font_size': font_size or 0.0, 'x': x or 0.0}))
        
        logger.info(f"[SECTION_DETECT] Found {len(candidates)} header candidates (threshold={HEADER_CANDIDATE_THRESHOLD})")
        if candidates:
            top_candidates = sorted(candidates, key=lambda x: x[2], reverse=True)[:5]
            for idx, text, score, feat in top_candidates:
                logger.info(f"[SECTION_DETECT] Top candidate: line {idx}, text='{_safe_log_str(text)}', score={score:.3f}, font_size={feat.get('font_size', 'N/A')}")
        
        # If no clear headers found, try harder to find inline headers
        if not candidates:
            logger.warning("[SECTION_DETECT] No header candidates found, using fallback detection")
            # Look for contact info at the beginning
            contact_lines = []
            content_lines = []
            contact_end_idx = 0
            
            for i, line in enumerate(lines[:15]):  # Check first 15 lines for contact info
                if self._contains_contact_info(line):
                    contact_lines.append(line)
                    contact_end_idx = i + 1
                elif contact_lines and i < contact_end_idx + 2:
                    # Allow 2 lines after contact info to be part of contact section
                    contact_lines.append(line)
                    contact_end_idx = i + 1
            
            # Add remaining lines to content
            content_lines = lines[contact_end_idx:] if contact_end_idx > 0 else lines
            
            if contact_lines:
                sections.append({
                    'header': 'Contact Information',
                    'content': contact_lines
                })
            
            # Try much harder to find inline headers in content
            if content_lines:
                logger.info(f"[SECTION_DETECT] Attempting aggressive inline detection on {len(content_lines)} content lines")
                inline_sections = self._detect_inline_sections_aggressive(content_lines)
                logger.info(f"[SECTION_DETECT] Aggressive inline detection found {len(inline_sections)} sections")
                if inline_sections and len(inline_sections) > 1:
                    sections.extend(inline_sections)
                else:
                    # Last resort: try keyword-based section splitting
                    logger.info(f"[SECTION_DETECT] Attempting keyword-based splitting on {len(content_lines)} content lines")
                    keyword_sections = self._split_by_keywords(content_lines)
                    logger.info(f"[SECTION_DETECT] Keyword-based splitting found {len(keyword_sections)} sections")
                    if keyword_sections and len(keyword_sections) > 1:
                        sections.extend(keyword_sections)
                    else:
                        # Only create single section if we truly can't find any structure
                        logger.warning(f"[SECTION_DETECT] All detection methods failed, creating single 'Resume Content' blob with {len(content_lines)} lines")
                        sections.append({
                            'header': 'Resume Content',
                            'content': content_lines
                        })

            sections = self._split_sections_on_embedded_headers(sections)
            return self._enforce_core_sections(sections, lines)
        
        # Tie-break & deduplicate adjacent headers (keep higher score)
        deduped: List[Tuple[int, str, float, Dict[str, Any]]] = []
        last_kept = None
        for c in candidates:
            if last_kept and (c[0] - last_kept[0]) <= 2:
                # prefer higher score, then larger font, then leftmost x, then later index
                a = last_kept
                b = c
                pick_better = (b[2] > a[2]) or (
                    math.isclose(b[2], a[2], rel_tol=1e-6) and (
                        (b[3]['font_size'] > a[3]['font_size']) or (
                            math.isclose(b[3]['font_size'], a[3]['font_size']) and (b[3]['x'] < a[3]['x'] or (math.isclose(b[3]['x'], a[3]['x']) and b[0] > a[0]))
                        )
                    )
                )
                last_kept = b if pick_better else a
                if deduped:
                    deduped[-1] = last_kept
                else:
                    deduped.append(last_kept)
            else:
                deduped.append(c)
                last_kept = c

        # Process sections based on finalized headers
        for i, (header_index, header_text, hdr_score, _feat) in enumerate(deduped):
            # Determine content range for this section
            next_header_index = deduped[i + 1][0] if i + 1 < len(deduped) else len(lines)
            
            # Collect content lines for this section
            content_lines = []
            for j in range(header_index + 1, next_header_index):
                if j < len(lines):
                    content_line = lines[j].strip()
                    if content_line:
                        content_lines.append(content_line)
            
            # Back-to-back header handling: require minimum content to keep previous
            if i > 0 and len([ln for ln in content_lines if ln.strip()]) < 2 and sum(len(ln) for ln in content_lines) < 60:
                # Not enough content; compare this header with previous, keep higher score (already tie-broken above)
                continue

            # Group content properly
            grouped_content = self._group_content_lines(content_lines)
            
            if grouped_content:  # Only add sections with content
                sections.append({
                    'header': header_text.strip(),
                    'content': grouped_content
                })
        
        # At the end, add this post-processing step:
        sections = self._split_sections_on_embedded_headers(sections)
        sections = self._enforce_core_sections(sections, lines)
        return sections

    def _detect_inline_sections(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Detect probable headers within a flat list of lines and split into sections."""
        sections: List[Dict[str, Any]] = []
        current_header: Optional[str] = None
        current_content: List[str] = []

        def flush():
            if current_header and current_content:
                sections.append({'header': current_header.strip(), 'content': current_content.copy()})

        for line in lines:
            candidate = line.strip()
            if not candidate:
                if current_content:
                    current_content.append(line)
                continue

            if self._looks_like_inline_header(candidate):
                flush()
                current_header = candidate
                current_content = []
            else:
                if current_header is None:
                    current_header = 'Resume Content'
                    current_content = []
                current_content.append(line)

        flush()
        return sections

    def _detect_inline_sections_aggressive(self, lines: List[str]) -> List[Dict[str, Any]]:
        """More aggressive inline section detection with lower thresholds."""
        sections: List[Dict[str, Any]] = []
        current_header: Optional[str] = None
        current_content: List[str] = []

        def flush():
            if current_header and current_content:
                sections.append({'header': current_header.strip(), 'content': current_content.copy()})

        for i, line in enumerate(lines):
            candidate = line.strip()
            if not candidate:
                if current_content:
                    current_content.append(line)
                continue

            # Lower threshold for header detection
            is_header = False
            candidate_lower = candidate.lower()
            
            # Check against section synonyms with lower threshold - more flexible matching
            for canonical, aliases in SECTION_SYNONYMS.items():
                for alias in aliases:
                    alias_lower = alias.lower()
                    # Exact match
                    if alias_lower == candidate_lower:
                        is_header = True
                        candidate = alias.title()
                        break
                    # Starts with alias + colon or space
                    if candidate_lower.startswith(alias_lower + ':') or candidate_lower.startswith(alias_lower + ' '):
                        is_header = True
                        candidate = alias.title()
                        break
                    # Contains alias as whole word (for "Work Experience", "Technical Skills", etc.)
                    if re.search(r'\b' + re.escape(alias_lower) + r'\b', candidate_lower):
                        # Make sure it's not buried in a long sentence
                        if len(candidate) <= 60:
                            is_header = True
                            candidate = alias.title()
                            break
                if is_header:
                    break
            
            # Also check if it looks like a header (short, uppercase/title case, ends with colon)
            if not is_header:
                # More lenient: accept lines ending with colon that are reasonably short
                if candidate.endswith(':') and len(candidate) <= 60:
                    # Check if next line doesn't look like a header continuation
                    if i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        if not (next_line and (next_line.isupper() or len(next_line.split()) <= 3)):
                            is_header = True
                            # Try to extract the header text (remove colon)
                            candidate = candidate[:-1].strip()
                # Also accept short uppercase/title case lines
                elif (len(candidate) <= 50 and 
                      (candidate.isupper() or candidate.istitle()) and 
                      len(candidate.split()) <= 6):
                    # Check if next line doesn't look like a header continuation
                    if i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        if not (next_line and (next_line.isupper() or len(next_line.split()) <= 3)):
                            is_header = True

            if is_header:
                flush()
                current_header = candidate
                current_content = []
            else:
                if current_header is None:
                    current_header = 'Resume Content'
                    current_content = []
                current_content.append(line)

        flush()
        return sections

    def _split_by_keywords(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Split content by looking for common section keywords.
        
        FIX: Aggressive header detection and strict "stop" logic to prevent
        one section from swallowing another (e.g., Education eating Experience).
        """
        sections: List[Dict[str, Any]] = []
        current_section: Optional[str] = None
        current_content: List[str] = []
        
        # Expanded keywords with AGGRESSIVE Experience detection (case-insensitive)
        # Order matters: Experience patterns checked first to prevent Education from swallowing it
        section_keywords = {
            'experience': [
                'experience', 'work experience', 'professional experience', 'employment',
                'employment history', 'work history', 'career history', 'career',
                'professional background', 'positions held', 'roles', 'work',
                'job history', 'relevant experience', 'professional history',
                'industry experience', 'history'  # "History" as standalone header
            ],
            'skills': [
                'skills', 'technical skills', 'competencies', 'expertise', 'proficiencies', 
                'technologies', 'tools', 'tech stack', 'technical proficiencies', 'capabilities',
                'programming languages', 'software', 'platforms', 'core competencies'
            ],
            'education': [
                'education', 'academic background', 'academic qualifications',
                'educational background', 'academic', 'qualifications'
                # Removed 'university', 'college', 'bachelor', 'master' to avoid false positives
            ],
            'projects': ['projects', 'project experience', 'portfolio', 'key projects', 'notable projects'],
            'certifications': ['certifications', 'certificates', 'licenses', 'accreditations', 'credentials'],
            'awards': ['awards', 'honors', 'achievements', 'recognition', 'accomplishments', 'distinctions']
        }
        
        # Compile regex patterns for each section (case-insensitive)
        section_patterns = {}
        for section_name, keywords in section_keywords.items():
            # Create pattern that matches any keyword as whole word
            patterns = [r'\b' + re.escape(kw) + r'\b' for kw in keywords]
            section_patterns[section_name] = re.compile('|'.join(patterns), re.IGNORECASE)
        
        def flush():
            if current_section and current_content:
                sections.append({
                    'header': current_section.title(),
                    'content': current_content.copy()
                })
        
        def detect_section_header(line_text: str) -> Optional[str]:
            """Detect if a line is a section header. Returns section name or None."""
            line_stripped = line_text.strip()
            if not line_stripped:
                return None
            
            line_lower = line_stripped.lower()
            
            # Skip if line is too long (likely content, not header)
            if len(line_stripped) > 60 and not line_stripped.endswith(':'):
                return None
            
            # Check each section pattern (order: experience first to catch it before education)
            section_order = ['experience', 'skills', 'education', 'projects', 'certifications', 'awards']
            for section_name in section_order:
                pattern = section_patterns[section_name]
                
                # Exact match (whole line is just the keyword)
                for keyword in section_keywords[section_name]:
                    if line_lower == keyword or line_lower == keyword + ':':
                        return section_name
                
                # Line starts with keyword + colon or is short enough to be a header
                if pattern.search(line_stripped):
                    # Short lines are likely headers
                    if len(line_stripped) <= 40:
                        return section_name
                    # Lines ending with colon are likely headers
                    if line_stripped.endswith(':'):
                        return section_name
                    # Lines that are ALL CAPS or Title Case with keyword
                    if line_stripped.isupper() or line_stripped.istitle():
                        return section_name
            
            return None
        
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                if current_content:
                    current_content.append(line)
                continue
            
            # STRICT STOP LOGIC: Always check for new section header
            # This ensures we don't let one section swallow another
            found_section = detect_section_header(line_stripped)
            
            if found_section:
                # Close current section and start new one
                flush()
                current_section = found_section
                current_content = []
                logger.debug(f"[SPLIT_KEYWORDS] Found header '{line_stripped}' -> section '{found_section}'")
            else:
                if current_section is None:
                    current_section = 'Resume Content'
                current_content.append(line)
        
        flush()
        return sections
    
    def _resplit_bloated_section(self, section_content: List[str], section_name: str) -> List[Dict[str, Any]]:
        """Re-split a bloated section that may have swallowed other sections.
        
        Used when Education section is >2000 chars but Experience is empty.
        Looks for date ranges, Experience keywords, and job titles to split the content.
        """
        if not section_content:
            return []
        
        # Join content for analysis
        full_text = ' '.join(section_content)
        
        # Patterns that indicate Experience content
        experience_patterns = [
            r'\bwork(?:ed|ing)?\s+(?:at|for|with)\b',  # "worked at", "working for"
            r'\b(?:19|20)\d{2}\s*[-–—]\s*(?:19|20)\d{2}|present|current\b',  # Date ranges
            r'\b(?:senior|junior|lead|staff|principal)?\s*(?:software|web|mobile|data|full[- ]?stack)?\s*(?:engineer|developer|architect|analyst|manager|director)\b',
            r'\b(?:responsible for|managed|developed|designed|implemented|built|created|led)\b',
            r'\bexperience\b',
            r'\bemployment\b',
            r'\bhistory\b'
        ]
        
        # Check if content looks like it contains Experience data
        has_experience_content = any(
            re.search(pattern, full_text, re.IGNORECASE) 
            for pattern in experience_patterns
        )
        
        if not has_experience_content:
            # No experience content found, return as-is
            return [{'header': section_name.title(), 'content': section_content}]
        
        logger.info(f"[RESPLIT] Attempting to re-split bloated {section_name} section ({len(full_text)} chars)")
        
        # Try to find where Experience section should start
        # Look for lines that look like Experience headers or job entries
        new_sections: List[Dict[str, Any]] = []
        current_section = section_name
        current_content: List[str] = []
        
        # EXPANDED: More experience header patterns including spaced versions
        experience_header_patterns = [
            r'^(?:work\s+)?experience\s*:?\s*$',
            r'^professional\s+experience\s*:?\s*$',
            r'^employment\s+(?:history)?\s*:?\s*$',
            r'^work\s+history\s*:?\s*$',
            r'^career\s+(?:history)?\s*:?\s*$',
            r'^(?:relevant\s+)?experience\s*:?\s*$',
            # Spaced versions: "E X P E R I E N C E"
            r'^e\s*x\s*p\s*e\s*r\s*i\s*e\s*n\s*c\s*e\s*:?\s*$',
            r'^w\s*o\s*r\s*k\s*:?\s*$',
            # ALL CAPS versions
            r'^EXPERIENCE\s*:?\s*$',
            r'^WORK\s+EXPERIENCE\s*:?\s*$',
            r'^PROFESSIONAL\s+EXPERIENCE\s*:?\s*$',
            r'^EMPLOYMENT\s*:?\s*$',
        ]
        experience_header_re = re.compile('|'.join(experience_header_patterns), re.IGNORECASE)
        
        # Pattern to detect job title + company lines (common experience entry format)
        job_entry_pattern = re.compile(
            r'^(?:'
            # Job Title at/| Company
            r'(?:(?:senior|junior|lead|staff|principal|associate|assistant)?\s*)?'
            r'(?:software|web|mobile|data|full[- ]?stack|frontend|backend|devops|cloud|systems?)?\s*'
            r'(?:engineer|developer|architect|analyst|manager|director|specialist|consultant|designer|scientist|administrator)\s*'
            r'(?:[@|,\-–—]\s*[A-Z][A-Za-z\s&.,]+)?'
            r'|'
            # Company Name | Job Title format
            r'[A-Z][A-Za-z\s&.,]+\s*[@|,\-–—]\s*'
            r'(?:(?:senior|junior|lead|staff|principal|associate|assistant)?\s*)?'
            r'(?:software|web|mobile|data|full[- ]?stack|frontend|backend|devops|cloud|systems?)?\s*'
            r'(?:engineer|developer|architect|analyst|manager|director|specialist|consultant|designer|scientist|administrator)'
            r')',
            re.IGNORECASE
        )
        
        found_experience_start = False
        experience_start_index = None
        
        for i, line in enumerate(section_content):
            line_stripped = line.strip()
            
            # Check if this line is an Experience header
            if experience_header_re.match(line_stripped):
                # Save current content
                if current_content:
                    new_sections.append({
                        'header': current_section.title(),
                        'content': current_content.copy()
                    })
                current_section = 'experience'
                current_content = []
                found_experience_start = True
                experience_start_index = i
                logger.info(f"[RESPLIT] Found embedded Experience header: '{line_stripped}'")
            else:
                current_content.append(line)
        
        # Save final section
        if current_content:
            new_sections.append({
                'header': current_section.title(),
                'content': current_content
            })
        
        if len(new_sections) > 1:
            logger.info(f"[RESPLIT] Successfully split into {len(new_sections)} sections")
            return new_sections
        
        # Fallback 1: Try to split by job entry patterns (job title + company)
        logger.info("[RESPLIT] Header-based split failed, trying job-entry detection")
        
        # Look for job titles that indicate start of experience content
        job_title_patterns = [
            r'\b(?:senior|junior|lead|staff|principal|associate)?\s*(?:software|web|mobile|data|full[- ]?stack|frontend|backend|devops|cloud|systems?)?\s*(?:engineer|developer|architect|analyst|manager|director|specialist|consultant|designer|scientist|administrator)\b',
            r'\b(?:intern|internship)\b',
            r'\b(?:CEO|CTO|CFO|COO|VP|President|Founder)\b',
        ]
        job_title_re = re.compile('|'.join(job_title_patterns), re.IGNORECASE)
        
        # Date patterns that appear in job entries
        date_in_line_pattern = re.compile(
            r'(?:'
            r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*(?:\d{4}|\d{2})'  # "Jan 2020" or "Jan '20"
            r'|\d{1,2}/\d{2,4}'  # "01/2020" or "01/20"
            r'|\d{4}\s*[-–—]\s*(?:\d{4}|present|current|now)'  # "2020 - 2022"
            r'|(?:present|current|now)\b'  # Just "Present" or "Current"
            r')',
            re.IGNORECASE
        )
        
        split_index = None
        
        # First pass: Look for lines with both job title AND date (strongest signal)
        for i, line in enumerate(section_content):
            if i < 2:  # Skip first 2 lines (likely education header content)
                continue
            line_stripped = line.strip()
            
            has_job_title = job_title_re.search(line_stripped)
            has_date = date_in_line_pattern.search(line_stripped)
            
            if has_job_title and has_date:
                split_index = i
                logger.info(f"[RESPLIT] Found job entry line (title+date) at index {i}: '{line_stripped[:60]}...'")
                break
        
        # Second pass: Look for job title followed by date on next line
        if split_index is None:
            for i, line in enumerate(section_content):
                if i < 2:
                    continue
                line_stripped = line.strip()
                
                if job_title_re.search(line_stripped):
                    # Check if next line has a date or looks like a company name
                    if i + 1 < len(section_content):
                        next_line = section_content[i + 1].strip()
                        if date_in_line_pattern.search(next_line):
                            split_index = i
                            logger.info(f"[RESPLIT] Found job title at index {i}, date on next line")
                            break
                        # Check for company patterns (Inc, LLC, Corp, Ltd)
                        if re.search(r'\b(?:Inc|LLC|Corp|Ltd|Limited|Company|Co\.|Corporation)\b', next_line, re.IGNORECASE):
                            split_index = i
                            logger.info(f"[RESPLIT] Found job title at index {i}, company on next line")
                            break
        
        # Third pass: Look for date ranges as standalone split points
        if split_index is None:
            date_pattern = re.compile(
                r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*\d{4}\s*[-–—]\s*(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)?[a-z]*\.?\s*(?:\d{4}|present|current|now)',
                re.IGNORECASE
            )
            
            for i, line in enumerate(section_content):
                if i < 3:
                    continue
                if date_pattern.search(line.strip()):
                    split_index = i
                    logger.info(f"[RESPLIT] Found date range at index {i}")
                    break
        
        if split_index:
            education_content = section_content[:split_index]
            experience_content = section_content[split_index:]
            logger.info(f"[RESPLIT] Split at line {split_index}: Education={len(education_content)} lines, Experience={len(experience_content)} lines")
            return [
                {'header': 'Education', 'content': education_content},
                {'header': 'Experience', 'content': experience_content}
            ]
        
        # Last resort: If we detected job-related content but couldn't find a clean split,
        # put most content in Experience (since we know it has experience data)
        if has_experience_content:
            # Check if education content is minimal (just degree + school at start)
            education_lines = []
            experience_lines = []
            
            # Look for education-specific content at the start
            education_keywords = ['bachelor', 'master', 'phd', 'doctorate', 'degree', 'university', 'college', 'gpa', 'graduated']
            found_experience_indicator = False
            
            for i, line in enumerate(section_content):
                line_lower = line.strip().lower()
                
                if not found_experience_indicator:
                    # Check if this looks like education content
                    is_education_line = any(kw in line_lower for kw in education_keywords)
                    # Check if this looks like start of experience
                    is_experience_line = job_title_re.search(line) or date_in_line_pattern.search(line)
                    
                    if is_experience_line and i > 0:
                        found_experience_indicator = True
                        experience_lines.append(line)
                    elif is_education_line or i < 3:
                        education_lines.append(line)
                    else:
                        # Ambiguous - check context
                        if len(education_lines) < 5:
                            education_lines.append(line)
                        else:
                            found_experience_indicator = True
                            experience_lines.append(line)
                else:
                    experience_lines.append(line)
            
            if experience_lines and len(experience_lines) > len(education_lines):
                logger.info(f"[RESPLIT] Content-based split: Education={len(education_lines)} lines, Experience={len(experience_lines)} lines")
                return [
                    {'header': 'Education', 'content': education_lines},
                    {'header': 'Experience', 'content': experience_lines}
                ]
        
        # No split found, return original
        logger.warning("[RESPLIT] Could not find split point, returning original section")
        return [{'header': section_name.title(), 'content': section_content}]

    def _is_actual_section_header(self, line: str, line_index: int, all_lines: List[str]) -> bool:
        line_clean = (line or '').strip()
        if not line_clean:
            return False
        font_size = is_bold = x = y = prev_y = left_margin = None
        page = None
        if line_index < len(self._last_line_positions):
            font_size, is_bold, x, y, page = self._last_line_positions[line_index]
            if line_index > 0 and self._last_line_positions[line_index - 1][3] is not None and y is not None:
                prev_y = self._last_line_positions[line_index - 1][3]
            if page is not None and page in self._left_margin_by_page:
                left_margin = self._left_margin_by_page[page]
        score = self._header_score(line_clean, font_size=font_size, is_bold=is_bold, x=x, y=y, prev_y=prev_y, left_margin=left_margin)
        standalone_ok = not TRAILING_PERIOD_RE.search(line_clean) or line_clean.endswith(':')
        return score >= HEADER_CANDIDATE_THRESHOLD and standalone_ok

    def _header_score(self, line_text: str, *, font_size: float | None = None, is_bold: bool | None = None,
                      x: float | None = None, y: float | None = None, prev_y: float | None = None,
                      left_margin: float | None = None) -> float:
        """Compute numeric header confidence in [0,1]."""
        text = (line_text or '').strip()
        if not text:
            return 0.0
        text_lower = text.lower()
        score = 0.0
        
        # Check if font info is missing/unknown
        font_info_missing = (font_size is None or font_size == 12.0) and (is_bold is None or is_bold is False)

        # Synonym match (strip bullets/colon)
        m = LEADING_BULLET_COLON_RE.match(text)
        core = (m.group(1) if m else text).strip().lower()
        if core:
            # Strip surrounding punctuation/markup (e.g., "**SKILLS**", "__EXPERIENCE__")
            core = re.sub(r'^[\W_]+', '', core)
            core = re.sub(r'[\W_]+$', '', core)
            core = re.sub(r'\s+', ' ', core).strip()

        normalized_core = re.sub(r'[^a-z0-9\s]+', ' ', core).strip() if core else ""

        # Exact synonym match - boost score if font info is missing
        if normalized_core in ALL_SYNONYMS:
            score += 0.65 if font_info_missing else 0.55
        elif core in ALL_SYNONYMS:
            score += 0.65 if font_info_missing else 0.55
        else:
            # High similarity to any synonym (allow partial containment)
            best_sim = 0.0
            for syn in ALL_SYNONYMS:
                if normalized_core and (normalized_core in syn or syn in normalized_core):
                    best_sim = max(best_sim, 0.92)
                    break
                r = difflib.SequenceMatcher(None, normalized_core or core, syn).ratio()
                if r > best_sim:
                    best_sim = r
                if best_sim >= 0.90:
                    break
            if best_sim >= 0.85:
                # Boost similarity score if font info is missing
                score += 0.40 if font_info_missing else 0.30
            elif best_sim >= 0.75 and font_info_missing:
                # Lower threshold when font info is missing
                score += 0.25

        # Font size vs page median (only if font info is available)
        size_bonus = 0.0
        if font_size and font_size != 12.0 and y is not None:
            # find median for nearest page if known
            # page index recovered from nearest stored record
            page_idx = None
            # try to infer page_idx from current line index mapping already built; if not available, skip
            # (we rely on _font_median_by_page keyed by page during extraction)
            if self._last_line_positions:
                # best-effort: find same y entry back
                for fs, _b, _x, yy, pg in self._last_line_positions:
                    if yy == y:
                        page_idx = pg
                        break
            if page_idx is not None and page_idx in self._font_median_by_page:
                med = self._font_median_by_page.get(page_idx, 0.0) or 0.0
            else:
                # global fallback median from map values
                meds = [v for v in self._font_median_by_page.values() if v > 0]
                med = (sorted(meds)[len(meds)//2] if meds else 0.0)
            if med and med > 0:
                ratio = max(0.0, min(2.0, float(font_size) / float(med)))
                # map ratio in [1.0 .. 2.0] to [0 .. 0.35]
                size_bonus = max(0.0, min(0.35, 0.35 * (ratio - 1.0)))
        score += size_bonus

        # Bold bonus (only if actually bold)
        if is_bold is True:
            score += 0.15
        
        # If font info is missing, boost score for lines ending with colon (common header pattern)
        if font_info_missing and text.endswith(':'):
            score += 0.20
        # Also boost for short uppercase/title case lines (likely headers)
        if font_info_missing and len(text) <= 50 and (text.isupper() or text.istitle()):
            score += 0.15

        # Vertical gap bonus
        if y is not None and prev_y is not None:
            dy = max(0.0, float(y) - float(prev_y))
            # normalize by rough line height; assume 12px baseline; cap contribution
            gap_norm = max(0.0, min(1.0, dy / 24.0))
            score += 0.15 * gap_norm

        # Left alignment bonus
        if x is not None and left_margin is not None:
            if abs(float(x) - float(left_margin)) <= 10.0:
                score += 0.10

        # All-caps short line bonus
        if text.isupper() and len(text.split()) <= 5:
            score += 0.20

        # Noise penalties
        if EMAIL_RE.search(text) or URL_RE.search(text) or PHONE_RE.search(text) or FOOTER_RE.search(text):
            score -= 0.30
        if len(text) > 100:
            score -= 0.30
        if TRAILING_PERIOD_RE.search(text) and not text.endswith(':'):
            score -= 0.15

        # Clamp
        return max(0.0, min(1.0, score))

    def _is_clearly_content(self, line: str) -> bool:
        """Determine if a line is clearly content (not a header)."""
        # Bullet points
        if re.match(r'^[•\-\*]', line):
            return True
            
        # Numbered items
        if re.match(r'^\d+\.', line):
            return True
            
        # Long lines are likely content
        if len(line) > 80:
            return True
            
        # Contains dates
        if re.search(r'\b\d{4}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b', line):
            return True
            
        # Contains email or phone
        if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', line) or re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', line):
            return True
            
        # Contains common content indicators
        content_indicators = ['responsible for', 'managed', 'developed', 'created', 'implemented', 'achieved', 'improved', 'led', 'supervised', 'coordinated', 'worked', 'assisted', 'helped', 'supported']
        if any(indicator in line.lower() for indicator in content_indicators):
            return True
            
        # Contains specific problematic content
        problematic_content = ['supervisors after completing work experience', 'tips', 'note', 'remember', 'keep in mind']
        if any(content in line.lower() for content in problematic_content):
            return True
            
        return False

    def _group_content_lines(self, content_lines: List[str]) -> List[str]:
        """Group content lines properly, keeping bullet points and related text together."""
        if not content_lines:
            return []
        
        # First, apply text reconstruction to fix fragmentation
        reconstructed_lines = self._apply_line_level_reconstruction(content_lines)
        
        grouped_content = []
        current_item = ""
        
        for line in reconstructed_lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if this is a bullet point or numbered item
            if re.match(r'^[•●\-\*]', line) or re.match(r'^\d+\.', line):
                # Save previous item if exists
                if current_item:
                    # Validate and clean the item before adding
                    cleaned_item = self._validate_and_clean_text(current_item.strip())
                    if cleaned_item:
                        grouped_content.append(cleaned_item)
                # Start new item
                current_item = line
            else:
                # This is a continuation of the current item or a standalone line
                if current_item:
                    # If it looks like a continuation (not a new sentence), append to current item
                    if not line[0].isupper() or len(line.split()) < 3:
                        current_item += " " + line
                    else:
                        # This looks like a new item, save current and start new
                        cleaned_item = self._validate_and_clean_text(current_item.strip())
                        if cleaned_item:
                            grouped_content.append(cleaned_item)
                        current_item = line
                else:
                    # No current item, this is a standalone line
                    current_item = line
        
        # Add the last item
        if current_item:
            cleaned_item = self._validate_and_clean_text(current_item.strip())
            if cleaned_item:
                grouped_content.append(cleaned_item)
        
        return grouped_content

    def _apply_line_level_reconstruction(self, lines: List[str]) -> List[str]:
        """Apply text reconstruction at the line level to fix common fragmentation issues."""
        if not lines:
            return lines
        
        reconstructed_lines = []
        i = 0
        
        while i < len(lines):
            current_line = lines[i].strip()
            
            # Check for ordinal reconstruction across lines
            if re.match(r'^\d+$', current_line) and i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line in ['st', 'nd', 'rd', 'th']:
                    # Reconstruct ordinal
                    reconstructed_line = current_line + next_line
                    
                    # Check for continuation (like "and 2nd")
                    if i + 2 < len(lines) and lines[i + 2].strip().lower() in ['and', '&']:
                        reconstructed_line += ' ' + lines[i + 2].strip()
                        if i + 3 < len(lines) and re.match(r'^\d+$', lines[i + 3].strip()):
                            reconstructed_line += ' ' + lines[i + 3].strip()
                            if i + 4 < len(lines) and lines[i + 4].strip() in ['st', 'nd', 'rd', 'th']:
                                reconstructed_line += lines[i + 4].strip()
                                i += 5
                            else:
                                i += 4
                        else:
                            i += 3
                    else:
                        i += 2
                    
                    reconstructed_lines.append(reconstructed_line)
                    continue
            
            # Check for bullet point reconstruction
            if current_line in ['•', '●', '-', '*']:
                bullet_content = '● '
                if i + 1 < len(lines):
                    bullet_content += lines[i + 1].strip()
                    i += 2
                else:
                    i += 1
                reconstructed_lines.append(bullet_content)
                continue
            
            # Check for fragmented sentences
            if (i + 1 < len(lines) and 
                len(current_line) < 50 and 
                not current_line.endswith('.') and 
                not lines[i + 1].strip().startswith('•')):
                
                # Merge with next line if it seems to be a continuation
                next_line = lines[i + 1].strip()
                if next_line and not next_line[0].isupper():
                    reconstructed_lines.append(current_line + ' ' + next_line)
                    i += 2
                    continue
            
            reconstructed_lines.append(current_line)
            i += 1
        
        return reconstructed_lines

    def _validate_and_clean_text(self, text: str) -> str:
        """Validate and clean text to ensure quality and fix common issues."""
        if not text:
            return ""
        
        # Step 0: Reject contact info that shouldn't be in content sections
        if self._contains_contact_info(text) and len(text) < 100:
            # Short lines with contact info are likely misclassified headers
            return ""
        
        # Step 1: Fix corrupted ordinals
        text = self._fix_corrupted_ordinals(text)
        
        # Step 2: Fix bullet point formatting
        text = self._fix_bullet_formatting(text)
        
        # Step 3: Fix spacing issues
        text = self._fix_spacing_issues(text)
        
        # Step 4: Remove nonsensical character combinations
        text = self._remove_nonsensical_combinations(text)
        
        # Step 5: Validate sentence coherence
        if not self._is_coherent_text(text):
            return ""
        
        return text.strip()

    def _fix_corrupted_ordinals(self, text: str) -> str:
        """Fix corrupted ordinal numbers in text."""
        # Fix patterns like "1stnd" or "2ndrd" or "stnd●"
        text = re.sub(r'(\d+)(st|nd|rd|th)(nd|rd|th|st)', r'\1\2', text)
        
        # Fix standalone ordinal fragments
        text = re.sub(r'\b(st|nd|rd|th)(nd|rd|th|st)\b', r'\1', text)
        
        # Fix ordinals that got corrupted with symbols
        text = re.sub(r'(st|nd|rd|th)[●•\-\*]+', r'\1', text)
        
        # Fix number + ordinal separated by space
        text = re.sub(r'(\d+)\s+(st|nd|rd|th)\b', r'\1\2', text)
        
        return text

    def _fix_bullet_formatting(self, text: str) -> str:
        """Fix bullet point formatting issues."""
        # Standardize bullet characters
        text = re.sub(r'^[•▪▫\-\*]+\s*', '● ', text)
        
        # Fix multiple bullets
        text = re.sub(r'●\s*●\s*', '● ', text)
        
        # Ensure space after bullet
        text = re.sub(r'^●([^\s])', r'● \1', text)
        
        # Remove corrupted bullet combinations
        text = re.sub(r'(st|nd|rd|th)●', r'\1 ●', text)
        
        return text

    def _fix_spacing_issues(self, text: str) -> str:
        """Fix spacing issues in text."""
        # Fix multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([,.;:!?])', r'\1', text)
        text = re.sub(r'([,.;:!?])\s*([a-zA-Z])', r'\1 \2', text)
        
        # Fix spacing around parentheses
        text = re.sub(r'\s*\(\s*', ' (', text)
        text = re.sub(r'\s*\)\s*', ') ', text)
        
        return text.strip()

    def _remove_nonsensical_combinations(self, text: str) -> str:
        """Remove nonsensical character combinations that indicate corruption."""
        # Remove patterns like "stnd●" or "rdth●"
        text = re.sub(r'(st|nd|rd|th)(nd|rd|th|st)[●•\-\*]*', r'\1', text)
        
        # Remove standalone fragments that don't make sense
        text = re.sub(r'\b(st|nd|rd|th)\b(?!\s+\w)', '', text)
        
        # Remove corrupted bullet combinations
        text = re.sub(r'[●•\-\*]{2,}', '●', text)
        
        return text.strip()

    def _is_coherent_text(self, text: str) -> bool:
        """Check if text is coherent and meaningful."""
        if not text or len(text.strip()) < 2:
            return False
        
        # Check for corrupted patterns
        corrupted_patterns = [
            r'(st|nd|rd|th){2,}',  # Multiple ordinal suffixes
            r'[●•\-\*]{3,}',       # Multiple bullets
            r'\b(st|nd|rd|th)\b$', # Standalone ordinal suffix
            r'^[●•\-\*]+$'         # Only bullets
        ]
        
        for pattern in corrupted_patterns:
            if re.search(pattern, text):
                return False
        
        # Check for minimum word content
        words = re.findall(r'\b\w+\b', text)
        if len(words) < 1:
            return False
        
        return True

    def _final_quality_check(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Perform final quality check on all sections."""
        quality_checked_sections = []
        
        for section in sections:
            cleaned_content = []
            
            for content_item in section.get('content', []):
                # Apply final validation
                cleaned_item = self._validate_and_clean_text(content_item)
                
                if cleaned_item and self._is_coherent_text(cleaned_item):
                    cleaned_content.append(cleaned_item)
            
            # Only add section if it has valid content
            if cleaned_content:
                quality_checked_sections.append({
                    'header': section['header'],
                    'content': cleaned_content
                })
        
        return quality_checked_sections

    def _contains_contact_info(self, line: str) -> bool:
        """Check if a line contains contact information."""
        line_lower = line.lower()
        line_stripped = line.strip()
        
        # Email pattern
        if EMAIL_RE.search(line):
            return True
            
        # Phone pattern
        if PHONE_RE.search(line):
            return True
            
        # LinkedIn/GitHub URLs
        if URL_RE.search(line) and ('linkedin' in line_lower or 'github' in line_lower):
            return True
            
        # Address indicators
        address_indicators = ['street', 'avenue', 'road', 'drive', 'lane', 'blvd', 'ave', 'st', 'rd', 'dr']
        if any(indicator in line_lower for indicator in address_indicators):
            return True
            
        # City, State ZIP pattern
        if re.search(r'\b[A-Za-z\s]+,\s*[A-Za-z]{2}\s*\d{5}\b', line):
            return True
        
        # Common contact patterns: email | phone | location
        if '|' in line_stripped or '•' in line_stripped:
            parts = re.split(r'[|•]', line_stripped)
            contact_count = 0
            for part in parts:
                part = part.strip()
                if EMAIL_RE.search(part) or PHONE_RE.search(part):
                    contact_count += 1
                elif len(part) < 30 and any(word in part.lower() for word in ['ca', 'ny', 'tx', 'fl', 'il', 'pa', 'oh', 'ga', 'nc', 'mi']):
                    contact_count += 1
            if contact_count >= 2:
                return True
            
        return False

    def _extract_candidate_name(self, all_lines: List[str]) -> str:
        """Extract candidate name from the first few lines of the resume."""
        if not all_lines:
            return ""
        
        # Look at the first 10 lines for the name
        for i, line in enumerate(all_lines[:10]):
            line = line.strip()
            if not line:
                continue
                
            # Skip lines that are clearly not names
            if any(skip_word in line.lower() for skip_word in [
                'resume', 'cv', 'curriculum vitae', 'phone', 'email', 'address', 
                'objective', 'summary', 'profile', 'experience', 'education',
                'skills', 'certifications', 'projects', 'achievements'
            ]):
                continue
            
            # Look for lines that look like names (2-4 words, mostly letters, title case)
            words = line.split()
            if 2 <= len(words) <= 4:
                # Check if all words are mostly letters and at least one is title case
                has_title_case = any(word[0].isupper() and word[1:].islower() for word in words if len(word) > 1)
                mostly_letters = all(word.replace('-', '').replace("'", '').isalpha() for word in words)
                
                if has_title_case and mostly_letters:
                    # Additional check: avoid common non-name patterns (job titles, organizations, etc.)
                    job_title_patterns = [
                        'university', 'college', 'school', 'company', 'inc', 'llc', 'corp',
                        'software', 'engineer', 'developer', 'manager', 'director', 'analyst',
                        'coordinator', 'practicum', 'supervisor', 'specialist', 'consultant',
                        'architect', 'scientist', 'lead', 'intern', 'founder', 'designer',
                        'technician', 'administrator', 'officer', 'programmer', 'strategist',
                        'editor', 'writer', 'producer', 'tester', 'trainer', 'teacher', 'mentor',
                        'assistant', 'associate', 'executive', 'president', 'vice', 'chief',
                        'head', 'senior', 'junior', 'principal', 'staff'
                    ]
                    # Also check for acronyms that are likely not names (BSIT, IT, CS, etc.)
                    has_acronym = any(len(word) <= 4 and word.isupper() for word in words)
                    # Check for common degree/academic patterns
                    academic_patterns = ['bsit', 'bscs', 'bs', 'ba', 'ma', 'ms', 'phd', 'mba']
                    
                    if (not any(pattern in line.lower() for pattern in job_title_patterns) and
                        not has_acronym and
                        not any(acad in line.lower() for acad in academic_patterns)):
                        return line
        
        return ""

    def _generate_accurate_summary(self, sections: List[Dict[str, Any]], all_lines: List[str]) -> Dict[str, Any]:
        """Generate accurate summary metadata by examining actual content."""
        section_headers = [section['header'].lower() for section in sections]
        all_content = ' '.join([' '.join(section['content']) for section in sections]).lower()
        
        # Check for contact information
        has_contact_info = False
        for line in all_lines[:15]:  # Check first 15 lines
            if (re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', line) or 
                re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', line) or
                any(indicator in line.lower() for indicator in ['street', 'avenue', 'road', 'drive', 'phone', 'email', 'address'])):
                has_contact_info = True
                break
        
        # Check for education
        has_education = False
        education_indicators = ['degree', 'university', 'college', 'school', 'bachelor', 'master', 'phd', 'diploma', 'certification', 'graduated']
        if (any('education' in header for header in section_headers) or
            any(indicator in all_content for indicator in education_indicators)):
            has_education = True
        
        # Check for experience
        has_experience = False
        experience_indicators = ['experience', 'employment', 'work', 'position', 'job', 'company', 'employer', 'responsible for', 'managed', 'developed']
        if (any(indicator in ' '.join(section_headers) for indicator in ['experience', 'employment', 'work', 'career']) or
            any(indicator in all_content for indicator in experience_indicators)):
            has_experience = True
        
        # Check for skills
        has_skills = False
        skills_indicators = ['skills', 'competencies', 'proficient', 'expertise', 'technical', 'programming', 'software']
        if (any('skill' in header for header in section_headers) or
            any(indicator in all_content for indicator in skills_indicators)):
            has_skills = True
        
        # Extract candidate name
        candidate_name = self._extract_candidate_name(all_lines)
        
        return {
            'total_sections': len(sections),
            'section_names': [section['header'] for section in sections],
            'has_contact_info': has_contact_info,
            'has_education': has_education,
            'has_experience': has_experience,
            'has_skills': has_skills,
            'candidate_name': candidate_name
        }

    def _ocr_page(self, page) -> str:
        """OCR fallback for image-based PDFs with performance optimizations.
        
        Args:
            page: A pdfplumber page object
        """
        if not OCR_ALLOWED:
            logger.debug("[OCR] Skipping OCR - OCR_ALLOWED is False")
            return ""
        
        if self.disable_ocr:
            logger.debug("[OCR] Skipping OCR - disable_ocr flag is set")
            return ""
            
        try:
            # Check if page has readable text first (pdfplumber uses extract_text())
            text = page.extract_text() or ""
            if len(text.strip()) >= 80:  # Skip OCR if page has readable text
                logger.info(f"[OCR] Skipping OCR - page has {len(text.strip())} chars of readable text")
                return text
            
            logger.info(f"[OCR] Running OCR on page (only {len(text.strip())} chars extracted)")
            
            # Reduced resolution for speed
            pil_img = page.to_image(resolution=200).original
            # Tuned Tesseract flags for resume text
            config = "--oem 1 --psm 6 -c preserve_interword_spaces=1"
            ocr_text = pytesseract.image_to_string(pil_img, config=config)
            
            if ocr_text and len(ocr_text.strip()) > len(text.strip()):
                logger.info(f"[OCR] OCR extracted {len(ocr_text.strip())} chars (vs {len(text.strip())} from text extraction)")
                return ocr_text
            else:
                logger.info(f"[OCR] OCR did not improve extraction, using original text")
                return text
        except Exception as e:
            logger.error(f"[OCR] OCR failed: {str(e)}")
            return ""

    @staticmethod
    def _hash_file(path: str) -> str:
        h = hashlib.md5()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                h.update(chunk)
        return h.hexdigest()

    def _save_json(self, structured_data: Dict[str, Any], original_filename: str) -> str:
        """Save structured data to JSON file."""
        base_name = os.path.splitext(os.path.basename(original_filename))[0]
        json_filename = f"{base_name}_structured.json"
        json_file_path = os.path.join(self.structured_output_dir, json_filename)
        
        with open(json_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(structured_data, json_file, indent=2, ensure_ascii=False)
            
        return json_file_path

    def _find_column_boundary(self, xs_sorted: List[float], page_width: float = 612.0) -> Optional[float]:
        """Find column boundary using adaptive gap detection.
        
        Uses multiple strategies:
        1. Large gap detection (> 15% of page width)
        2. Bimodal distribution detection (two clusters of x-coordinates)
        3. Page center proximity check (columns typically divide near center)
        
        Args:
            xs_sorted: Sorted list of unique x-coordinates
            page_width: Width of the page in points (default 612 = letter size)
            
        Returns:
            Midpoint of the gap if a column boundary is detected, None otherwise
        """
        if len(xs_sorted) < 4:  # Need enough elements for column detection
            return None
        
        # Calculate gaps between consecutive x-coordinates
        gaps = []
        for i in range(len(xs_sorted) - 1):
            gap = xs_sorted[i + 1] - xs_sorted[i]
            gaps.append((gap, xs_sorted[i], xs_sorted[i + 1]))
        
        if not gaps:
            return None
        
        # Find the largest gap
        largest_gap, left_x, right_x = max(gaps, key=lambda x: x[0])
        
        # Adaptive threshold: 15% of page width OR 50 pixels, whichever is larger
        # This works for both narrow and wide PDFs
        adaptive_threshold = max(page_width * 0.15, 50.0)
        
        # The gap must also be in the "middle third" of the page to be a column boundary
        # (not just a left margin or right margin gap)
        gap_midpoint = (left_x + right_x) / 2.0
        page_center = page_width / 2.0
        center_tolerance = page_width * 0.35  # Allow columns to divide within 35% of center
        
        is_center_gap = abs(gap_midpoint - page_center) < center_tolerance
        
        if largest_gap > adaptive_threshold and is_center_gap:
            boundary = gap_midpoint
            logger.debug(f"[COLUMN_DETECT] Found column boundary at x={boundary:.1f} (gap={largest_gap:.1f}px, threshold={adaptive_threshold:.1f}px)")
            return boundary
        
        # Fallback: Check for bimodal distribution using k-means style clustering
        # If x-coords naturally cluster into two groups, that indicates columns
        if len(xs_sorted) >= 6:
            boundary = self._detect_bimodal_columns(xs_sorted, page_width)
            if boundary:
                return boundary
        
        return None
    
    def _detect_bimodal_columns(self, xs_sorted: List[float], page_width: float) -> Optional[float]:
        """Detect two-column layout using bimodal clustering of x-coordinates.
        
        If x-coordinates cluster into two distinct groups (left and right halves),
        returns the boundary between them.
        """
        # Simple 2-means clustering
        left_half = [x for x in xs_sorted if x < page_width / 2]
        right_half = [x for x in xs_sorted if x >= page_width / 2]
        
        # Need significant content in both halves
        if len(left_half) < 3 or len(right_half) < 3:
            return None
        
        # Calculate cluster centers
        left_center = sum(left_half) / len(left_half)
        right_center = sum(right_half) / len(right_half)
        
        # Check if clusters are well-separated (gap > 20% of page width)
        cluster_gap = right_center - left_center
        if cluster_gap > page_width * 0.20:
            # Boundary is between the rightmost left element and leftmost right element
            boundary = (max(left_half) + min(right_half)) / 2.0
            logger.debug(f"[COLUMN_DETECT] Bimodal clustering found boundary at x={boundary:.1f} (cluster gap={cluster_gap:.1f}px)")
            return boundary
        
        return None
    
    def _detect_and_group_by_columns(self, text_elements: List[Dict[str, Any]], page_widths: Optional[Dict[int, float]] = None) -> List[Dict[str, Any]]:
        """Detect two-column layouts and reorder text elements to read columns vertically.
        
        CRITICAL for 2-column resumes: Reads each column top-to-bottom before moving
        to the next column. This prevents horizontal line merging that destroys 
        multi-column layouts.
        
        Args:
            text_elements: List of text elements with 'x', 'y', 'page', 'text' keys
            page_widths: Optional dict mapping page number to page width in points
            
        Returns:
            Reordered list of text elements (left column fully, then right column)
        """
        if not text_elements:
            return text_elements
        
        # Default page width (letter size = 612 points)
        default_width = 612.0
        if page_widths is None:
            page_widths = {}
        
        # Group by page
        by_page: Dict[int, List[Dict[str, Any]]] = {}
        for elem in text_elements:
            page = int(elem.get('page', 0))
            by_page.setdefault(page, []).append(elem)
        
        reordered = []
        for page in sorted(by_page.keys()):
            elems = by_page[page]
            
            # Get page width (from metadata or estimate from max x-coordinate)
            page_width = page_widths.get(page, default_width)
            
            # If no explicit page width, estimate from content
            if page not in page_widths:
                max_x = max((float(e.get('x', 0) or 0) + float(e.get('width', 100) or 100)) 
                           for e in elems if e.get('x') is not None)
                if max_x > 400:  # Reasonable page width detected
                    page_width = max(max_x * 1.1, default_width)  # Add 10% margin
            
            # Extract x-coordinates for all elements
            xs = [float(e.get('x', 0) or 0.0) for e in elems if e.get('x') is not None]
            
            if not xs:
                # No x-coordinates available, sort by original order
                reordered.extend(elems)
                continue
            
            # Find column boundary using adaptive detection
            xs_sorted = sorted(set(xs))
            column_boundary = self._find_column_boundary(xs_sorted, page_width)
            
            if column_boundary:
                # Two-column layout detected
                left_col = [e for e in elems if float(e.get('x', 0) or 0.0) < column_boundary]
                right_col = [e for e in elems if float(e.get('x', 0) or 0.0) >= column_boundary]
                
                # Sort each column by y (top to bottom), then by x for sub-line ordering
                left_col.sort(key=lambda e: (float(e.get('y', 0) or 0.0), float(e.get('x', 0) or 0.0)))
                right_col.sort(key=lambda e: (float(e.get('y', 0) or 0.0), float(e.get('x', 0) or 0.0)))
                
                # CRITICAL: Process left column FULLY before right column
                # This ensures vertical reading order within each column
                reordered.extend(left_col)
                reordered.extend(right_col)
                
                logger.info(f"[COLUMN_DETECT] Page {page}: Two-column layout (boundary={column_boundary:.1f}px). Left: {len(left_col)}, Right: {len(right_col)} elements")
            else:
                # Single column - sort by y (top to bottom), then by x
                elems.sort(key=lambda e: (float(e.get('y', 0) or 0.0), float(e.get('x', 0) or 0.0)))
                reordered.extend(elems)
                logger.debug(f"[COLUMN_DETECT] Page {page}: Single column layout")
        
        return reordered

    def extract_plain_text(self, structured_data: Dict[str, Any]) -> str:
        """Extract plain text from structured data."""
        plain_text = ""
        for section in structured_data.get('sections', []):
            plain_text += f"\n--- {section['header']} ---\n"
            for line in section['content']:
                plain_text += line + "\n"
        return plain_text.strip()

    def clean_json(self, data):
        """Clean JSON data, replacing null values with empty strings."""
        if isinstance(data, dict):
            return {k: self.clean_json(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.clean_json(item) for item in data]
        elif data is None:
            return ""
        else:
            return data

    def add_classification_results(self, json_file_path: str, classification_results: Dict[str, Any]):
        """Add classification results to existing JSON file."""
        with open(json_file_path, 'r+', encoding='utf-8') as f:
            data = json.load(f)
            data['classification'] = classification_results
            f.seek(0)
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.truncate() 
