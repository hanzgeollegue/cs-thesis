"""
Configuration file for Resume Processor application.
Contains API keys, model settings, and other configuration options.
"""

import os
from typing import Optional

# OpenAI Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4')

# Alternative: You can set the API key directly here (not recommended for production)
# OPENAI_API_KEY = 'your-api-key-here'

# LLM Ranking Configuration
LLM_RANKING_ENABLED = bool(OPENAI_API_KEY)
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 1000

# Google Gemini Configuration
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', '')
# Optional: hard-code your Gemini key here for development (leave empty for safety)
HARD_CODED_GOOGLE_API_KEY = ''
# Use the user's requested model name; fall back to commonly available one if not set
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')

# Provider selection: 'openai' or 'google'
LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'openai').lower()

# If a hard-coded Google key is present, prefer Google provider automatically
if HARD_CODED_GOOGLE_API_KEY:
    LLM_PROVIDER = 'google'

# Optional flag to force-disable LLM usage
LLM_DISABLED = os.getenv('LLM_DISABLED', '').lower() in ('1', 'true', 'yes')
USE_LLM_RANKER = os.getenv("USE_LLM_RANKER", "0") in {"1", "true", "True"}

# TF-IDF Configuration
TFIDF_MAX_FEATURES = int(os.getenv("TFIDF_MAX_FEATURES", "20000"))
if TFIDF_MAX_FEATURES <= 1:
    TFIDF_MAX_FEATURES = 20000
TFIDF_MIN_DF = 1
TFIDF_MAX_DF = 0.95
TFIDF_NGRAM_RANGE = (1, 2)

# Section Weights for Scoring
SECTION_WEIGHTS = {
    'experience': 0.45,
    'skills': 0.35,
    'education': 0.15,
    'misc': 0.05
}

# Batch Processing Limits
MAX_BATCH_SIZE = 25
MAX_WORKERS = 4

# Performance Guardrails
RESUME_TIMEOUT_SEC = int(os.getenv("RESUME_TIMEOUT_SEC", "120"))
BATCH_TIMEOUT_SEC = int(os.getenv("BATCH_TIMEOUT_SEC", "600"))
MAX_PAGES = int(os.getenv("MAX_PAGES", "3"))
MAX_OCR_PAGES = int(os.getenv("MAX_OCR_PAGES", "2"))
# Disable OCR by default for performance; can be re-enabled via env
OCR_ALLOWED = os.getenv("OCR_ALLOWED", "0") in {"1", "true", "True"}
PARSE_CONCURRENCY = int(os.getenv("PARSE_CONCURRENCY", "8"))  # Use more cores for faster parallel parsing

# Normalization strategy
# If true, use absolute (batch-invariant) scaling for TF-IDF/SBERT/CE; else use relative batch min-max (with guardrails)
ABSOLUTE_NORMALIZATION = os.getenv("ABSOLUTE_NORMALIZATION", "1") in {"1", "true", "True"}

# Threading Configuration
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
if "MKL_NUM_THREADS" in os.environ:
    os.environ["MKL_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"]

# File Processing
SUPPORTED_FORMATS = ['.pdf']
MAX_FILE_SIZE_MB = 50

# Logging Configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

def get_openai_config() -> dict:
    """Get OpenAI configuration dictionary."""
    return {
        'api_key': OPENAI_API_KEY,
        'model': OPENAI_MODEL,
        'temperature': LLM_TEMPERATURE,
        'max_tokens': LLM_MAX_TOKENS,
        'enabled': LLM_RANKING_ENABLED
    }

def get_llm_config() -> dict:
    """Provider-agnostic LLM configuration (OpenAI or Google Gemini)."""
    # Prefer hard-coded key if provided
    effective_google_key = HARD_CODED_GOOGLE_API_KEY or GOOGLE_API_KEY
    if LLM_PROVIDER == 'google':
        return {
            'provider': 'google',
            'api_key': effective_google_key,
            'model': GEMINI_MODEL,
            'temperature': LLM_TEMPERATURE,
            'max_tokens': LLM_MAX_TOKENS,
            'enabled': (not LLM_DISABLED) and bool(effective_google_key)
        }
    # default to OpenAI
    return {
        'provider': 'openai',
        'api_key': OPENAI_API_KEY,
        'model': OPENAI_MODEL,
        'temperature': LLM_TEMPERATURE,
        'max_tokens': LLM_MAX_TOKENS,
        'enabled': (not LLM_DISABLED) and bool(OPENAI_API_KEY)
    }

def validate_config() -> list:
    """Validate configuration and return list of issues."""
    issues = []
    
    # Validate keys depending on provider
    if LLM_PROVIDER == 'google':
        # Consider hard-coded key as effective
        effective_google_key = HARD_CODED_GOOGLE_API_KEY or GOOGLE_API_KEY
        if not effective_google_key:
            issues.append("Google API key not set. Set GOOGLE_API_KEY environment variable.")
        # Basic format check (Google keys vary; skip strict validation)
    else:
        if not OPENAI_API_KEY:
            issues.append("OpenAI API key not set. LLM ranking will use fallback methods.")
            issues.append("Set OPENAI_API_KEY environment variable or add to config.py")
        if OPENAI_API_KEY and not OPENAI_API_KEY.startswith('sk-'):
            issues.append("OpenAI API key format appears invalid (should start with 'sk-')")
    
    return issues 

# --- Meta/Matching/Combiner Toggles ---
# Enable lightweight raw token match detection for explainability flags
ENABLE_MATCH_DETECTION = os.getenv("ENABLE_MATCH_DETECTION", "1") in {"1", "true", "True"}

# If enabled, and there is no experience nor skills token match, skip CE compute (set ce=0)
REQUIRE_MATCH_FOR_CE = os.getenv("REQUIRE_MATCH_FOR_CE", "0") in {"1", "true", "True"}

# Enable tiny in-process meta combiner (ridge on normalized features)
ENABLE_META_COMBINER = os.getenv("ENABLE_META_COMBINER", "1") in {"1", "true", "True"}
META_RIDGE_L2 = float(os.getenv("META_RIDGE_L2", "0.1"))
META_MIN_LABELS = int(os.getenv("META_MIN_LABELS", "50"))

# Testing / Caching
DISABLE_TFIDF_CACHE = os.getenv("DISABLE_TFIDF_CACHE", "0") in {"1", "true", "True"}