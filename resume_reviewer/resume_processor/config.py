"""
Configuration for Resume Processor.

Performance guardrails, NLG settings, file processing, and parser options.
Scoring/ranking config will live in hybrid_ranker.py once implemented.
"""

import os

# ---------------------------------------------------------------------------
# NLG System
# ---------------------------------------------------------------------------
NLG_VERSION = os.getenv('NLG_VERSION', '2')  # '1' for legacy, '2' for enhanced
USE_ENHANCED_NLG = NLG_VERSION == '2'

# Evidence-based analysis settings
EVIDENCE_CONFIDENCE_THRESHOLD = float(os.getenv('EVIDENCE_CONFIDENCE_THRESHOLD', '0.6'))
EVIDENCE_MAX_ITEMS = int(os.getenv('EVIDENCE_MAX_ITEMS', '3'))
CONCRETE_EXAMPLE_QUOTE_THRESHOLD = float(os.getenv('CONCRETE_EXAMPLE_QUOTE_THRESHOLD', '0.85'))
SHORTLIST_RANK_THRESHOLD = int(os.getenv('SHORTLIST_RANK_THRESHOLD', '3'))
SHORTLIST_PERCENTILE_THRESHOLD = int(os.getenv('SHORTLIST_PERCENTILE_THRESHOLD', '75'))

# ---------------------------------------------------------------------------
# Batch Processing Limits
# ---------------------------------------------------------------------------
MAX_BATCH_SIZE = 25
MAX_WORKERS = 4

# ---------------------------------------------------------------------------
# Performance Guardrails
# ---------------------------------------------------------------------------
RESUME_TIMEOUT_SEC = int(os.getenv("RESUME_TIMEOUT_SEC", "30"))
BATCH_TIMEOUT_SEC = int(os.getenv("BATCH_TIMEOUT_SEC", "300"))
MAX_PAGES = int(os.getenv("MAX_PAGES", "3"))
MAX_OCR_PAGES = int(os.getenv("MAX_OCR_PAGES", "2"))
OCR_ALLOWED = True
USE_OCR = False  # Not used directly, only as fallback
PARSE_CONCURRENCY = int(os.getenv("PARSE_CONCURRENCY", "8"))

# ---------------------------------------------------------------------------
# Feature Flags
# ---------------------------------------------------------------------------
DEBUG_SCORES = os.getenv("DEBUG_SCORES", "0") in {"1", "true", "True"}
DEBUG_AUDIT = os.getenv("DEBUG_AUDIT", "0") in {"1", "true", "True"}
SENIORITY_POLICY = os.getenv("SENIORITY_POLICY", "entry_mid_as_hard").strip().lower()

# ---------------------------------------------------------------------------
# Threading
# ---------------------------------------------------------------------------
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
if "MKL_NUM_THREADS" in os.environ:
    os.environ["MKL_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"]

# ---------------------------------------------------------------------------
# File Processing
# ---------------------------------------------------------------------------
SUPPORTED_FORMATS = ['.pdf']
MAX_FILE_SIZE_MB = 50

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# ---------------------------------------------------------------------------
# Parser Backend
# ---------------------------------------------------------------------------
PARSER_BACKEND = os.getenv("PARSER_BACKEND", "fitz").strip().lower()
