import json
import re
import numpy as np
import hashlib
from typing import List, Dict, Any, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from collections import defaultdict
import pickle
import os
import time
import math
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# --- Timing configuration ---
TIMING_ENABLED = os.getenv("TIMING", "0") in {"1", "true", "True"}


def normalize_job_description(jd_text: str) -> Dict[str, str]:
    """
    Normalize a job description into a standardized format with predefined sections.
    
    Args:
        jd_text (str): Raw job description text.
        
    Returns:
        dict: Normalized job description with predefined sections.
    """
    try:
        if not jd_text or not isinstance(jd_text, str):
            logger.warning("Invalid JD text provided for normalization")
            return {
                "job_title": "",
                "experience_required": "",
                "skills_required": "",
                "responsibilities": "",
                "education": "",
                "company_description": "",
                "location": ""
            }
        
        jd_normalized = {}
        jd_text_lower = jd_text.lower().strip()
        
        # Extract sections using regex patterns
        jd_normalized["job_title"] = _extract_job_title(jd_text_lower)
        jd_normalized["experience_required"] = _extract_section(jd_text_lower, "experience")
        jd_normalized["skills_required"] = _extract_section(jd_text_lower, "skills")
        jd_normalized["responsibilities"] = _extract_section(jd_text_lower, "responsibilities")
        jd_normalized["education"] = _extract_section(jd_text_lower, "education")
        jd_normalized["company_description"] = _extract_section(jd_text_lower, "company")
        jd_normalized["location"] = _extract_section(jd_text_lower, "location")
        
        # Fallback: if skills section is empty, try to extract from experience section
        if not jd_normalized["skills_required"].strip():
            logger.info("Skills section empty, attempting to extract from experience section")
            experience_text = jd_normalized["experience_required"]
            if experience_text.strip():
                # Look for technical skills in experience text
                tech_skills = _extract_technical_skills(experience_text)
                if tech_skills:
                    jd_normalized["skills_required"] = tech_skills
                    logger.info(f"Extracted skills from experience: {tech_skills[:100]}...")
        
        return jd_normalized
        
    except Exception as e:
        logger.error(f"Error normalizing job description: {e}")
        return {
            "job_title": "",
            "experience_required": "",
            "skills_required": "",
            "responsibilities": "",
            "education": "",
            "company_description": "",
            "location": ""
        }


def _extract_job_title(jd_text: str) -> str:
    """Extract job title from JD text."""
    try:
        # Look for common job title patterns
        title_patterns = [
            r"job title[:\s]+([^\n]+)",
            r"position[:\s]+([^\n]+)",
            r"role[:\s]+([^\n]+)",
            r"we are looking for[:\s]+([^\n]+)",
            r"seeking[:\s]+([^\n]+)",
            r"hiring[:\s]+([^\n]+)"
        ]
        
        for pattern in title_patterns:
            match = re.search(pattern, jd_text)
            if match:
                title = match.group(1).strip()
                # Clean up the title
                title = re.sub(r"[^\w\s-]", "", title)
                if len(title) > 3:  # Valid title should be at least 3 characters
                    return title
        
        # If no specific pattern found, try to extract from first line
        first_line = jd_text.split('\n')[0].strip()
        if len(first_line) < 100 and len(first_line) > 3:  # Reasonable title length
            return re.sub(r"[^\w\s-]", "", first_line)
        
        return ""
        
    except Exception as e:
        logger.warning(f"Error extracting job title: {e}")
        return ""


def _extract_section(jd_text: str, section_name: str) -> str:
    """
    Extract a section of the JD based on the section name.
    
    Args:
        jd_text (str): Raw JD text (lowercase).
        section_name (str): The name of the section to extract.
        
    Returns:
        str: The content of the section or an empty string if not found.
    """
    try:
        section_patterns = {
            "experience": [
                r"experience[\s\S]*?(?=skills|responsibilities|education|requirements|qualifications|$)",
                r"years? of experience[\s\S]*?(?=skills|responsibilities|education|requirements|qualifications|$)",
                r"required experience[\s\S]*?(?=skills|responsibilities|education|requirements|qualifications|$)",
                r"experience level[\s\S]*?(?=skills|responsibilities|education|requirements|qualifications|$)",
                r"experience in[\s\S]*?(?=requirements|qualifications|education|responsibilities|$)",
                r"with experience in[\s\S]*?(?=requirements|qualifications|education|responsibilities|$)",
                r"looking for[\s\S]*?(?=requirements|qualifications|education|responsibilities|$)",
                r"seeking[\s\S]*?(?=requirements|qualifications|education|responsibilities|$)",
                r"candidate with[\s\S]*?(?=requirements|qualifications|education|responsibilities|$)"
            ],
            "skills": [
                r"skills[\s\S]*?(?=responsibilities|education|requirements|qualifications|experience|$)",
                r"technical skills[\s\S]*?(?=responsibilities|education|requirements|qualifications|experience|$)",
                r"required skills[\s\S]*?(?=responsibilities|education|requirements|qualifications|experience|$)",
                r"qualifications[\s\S]*?(?=responsibilities|education|requirements|qualifications|experience|$)",
                r"experience in[\s\S]*?(?=requirements|qualifications|education|responsibilities|$)",
                r"with experience in[\s\S]*?(?=requirements|qualifications|education|responsibilities|$)",
                r"looking for[\s\S]*?(?=requirements|qualifications|education|responsibilities|$)",
                r"proficiency in[\s\S]*?(?=requirements|qualifications|education|responsibilities|$)",
                r"knowledge of[\s\S]*?(?=responsibilities|education|requirements|qualifications|experience|$)",
                r"must have[\s\S]*?(?=requirements|qualifications|education|responsibilities|$)",
                r"should have[\s\S]*?(?=requirements|qualifications|education|responsibilities|$)",
                r"candidate should[\s\S]*?(?=requirements|qualifications|education|responsibilities|$)"
            ],
            "responsibilities": [
                r"responsibilities[\s\S]*?(?=education|requirements|qualifications|experience|skills|$)",
                r"duties[\s\S]*?(?=education|requirements|qualifications|experience|skills|$)",
                r"what you'll do[\s\S]*?(?=education|requirements|qualifications|experience|skills|$)",
                r"key responsibilities[\s\S]*?(?=education|requirements|qualifications|experience|skills|$)"
            ],
            "education": [
                r"education[\s\S]*?(?=requirements|qualifications|experience|skills|responsibilities|$)",
                r"degree[\s\S]*?(?=requirements|qualifications|experience|skills|responsibilities|$)",
                r"educational requirements[\s\S]*?(?=requirements|qualifications|experience|skills|responsibilities|$)",
                r"academic[\s\S]*?(?=requirements|qualifications|experience|skills|responsibilities|$)"
            ],
            "company": [
                r"about us[\s\S]*?(?=requirements|qualifications|experience|skills|responsibilities|education|$)",
                r"company[\s\S]*?(?=requirements|qualifications|experience|skills|responsibilities|education|$)",
                r"our team[\s\S]*?(?=requirements|qualifications|experience|skills|responsibilities|education|$)",
                r"we are[\s\S]*?(?=requirements|qualifications|experience|skills|responsibilities|education|$)"
            ],
            "location": [
                r"location[\s\S]*?(?=requirements|qualifications|experience|skills|responsibilities|education|company|$)",
                r"based in[\s\S]*?(?=requirements|qualifications|experience|skills|responsibilities|education|company|$)",
                r"remote[\s\S]*?(?=requirements|qualifications|experience|skills|responsibilities|education|company|$)",
                r"office[\s\S]*?(?=requirements|qualifications|experience|skills|responsibilities|education|company|$)"
            ]
        }
        
        patterns = section_patterns.get(section_name, [])
        for pattern in patterns:
            match = re.search(pattern, jd_text, re.IGNORECASE)
            if match:
                content = match.group(0).strip()
                # Clean up the content
                content = re.sub(r"^(experience|skills|responsibilities|education|company|location)[:\s]*", "", content, flags=re.IGNORECASE)
                content = re.sub(r"\s+", " ", content)  # Normalize whitespace
                if len(content) > 10:  # Valid content should be substantial
                    return content
        
        return ""
        
    except Exception as e:
        logger.warning(f"Error extracting section '{section_name}': {e}")
        return ""


def preprocess_text(text: str) -> str:
    """
    Preprocess text by cleaning and normalizing it (for TF-IDF).
    
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

def preprocess_text_for_dense_models(text: str) -> str:
    """
    Preprocess text for dense models (SBERT/CE) while preserving important tokens.
    Keeps capitalization and punctuation in skill terms like Node.js, OpenAPI, Chart.js, RBAC.
    
    Args:
        text (str): Raw text to preprocess.
        
    Returns:
        str: Preprocessed text with preserved skill tokens.
    """
    try:
        if not text or not isinstance(text, str):
            return ""
        
        # Basic cleanup without lowercasing
        text = text.strip()
        
        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)
        
        # Remove problematic characters but preserve dots, hashes, pluses in skill terms
        # Keep: letters, numbers, spaces, dots, hashes, pluses, hyphens, underscores
        text = re.sub(r"[^\w\s.#+/-]", " ", text)
        
        # Final whitespace cleanup
        text = re.sub(r"\s+", " ", text).strip()
        
        return text
        
    except Exception as e:
        logger.warning(f"Error preprocessing text for dense models: {e}")
        return ""


def build_canonical_resume_text(sections: Dict[str, Any]) -> str:
    """Build a single normalized string for downstream models (TF-IDF/SBERT/CE)."""
    try:
        if not isinstance(sections, dict):
            return ""
        parts: List[str] = []
        for key in ("experience", "skills", "education", "misc"):
            val = sections.get(key, "")
            if isinstance(val, list):
                parts.extend([str(x) for x in val])
            else:
                parts.append(str(val))
        text = " \n ".join([p for p in parts if p])
        return preprocess_text(text)
    except Exception as e:
        logger.warning(f"Error building canonical text: {e}")
        return ""


def _extract_technical_skills(text: str) -> str:
    """Extract technical-sounding skill phrases from free text for JD fallback.
    Returns a comma-separated string of detected skills.
    """
    try:
        taxonomy = SkillTaxonomy()
        tokens = taxonomy.extract_skills_from_text(text)
        if not tokens:
            return ""
        return ', '.join(sorted(set(tokens)))
    except Exception as e:
        logger.warning(f"Error extracting technical skills: {e}")
        return ""

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

def _norm(x, lo, hi):
    """Normalize value to [0,1] range."""
    if hi is None or lo is None or hi <= lo: 
        return 0.5
    v = (x - lo) / (hi - lo)
    return 0.0 if v < 0 else 1.0 if v > 1 else float(v)

def _std(vals):
    """Compute standard deviation of values."""
    vals = [v for v in vals if isinstance(v, (int, float))]
    if not vals: 
        return 0.0
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))

def _dedup(items, key_fn):
    """Deduplicate items based on key function."""
    seen, out = set(), []
    for it in items:
        k = key_fn(it)
        if k not in seen:
            seen.add(k)
            out.append(it)
    return out

def _jd_hash(text: str, max_features: int, ngram: tuple[int, int]) -> str:
    """Generate hash for job description caching."""
    h = hashlib.sha256()
    h.update((text or "").lower().encode("utf-8"))
    h.update(f"{max_features}-{ngram}".encode("utf-8"))
    return h.hexdigest()

# Module-level caches for TF-IDF vectorizers
_section_vectorizer_cache = {}
_skill_vectorizer_cache = {}

def clear_tfidf_caches() -> None:
    """Clear in-memory TF-IDF vectorizer caches (for testing)."""
    try:
        _section_vectorizer_cache.clear()
    except Exception:
        pass
    try:
        _skill_vectorizer_cache.clear()
    except Exception:
        pass

def chunk_text_for_sbert(text: str, max_tokens: int = 512) -> List[str]:
    """
    Chunk text into segments for SBERT processing (~512-768 tokens).
    
    Args:
        text (str): Input text to chunk.
        max_tokens (int): Maximum tokens per chunk.
        
    Returns:
        List[str]: List of text chunks.
    """
    try:
        if not text or not isinstance(text, str):
            return []
        
        # Simple word-based chunking (approximation: ~1.3 tokens per word)
        words = text.split()
        if not words:
            return []
        
        max_words = max(1, int(max_tokens / 1.3))
        chunks = []
        
        for i in range(0, len(words), max_words):
            chunk_words = words[i:i + max_words]
            chunk_text = ' '.join(chunk_words)
            if chunk_text.strip():
                chunks.append(chunk_text)
        
        return chunks if chunks else [text]
        
    except Exception as e:
        logger.warning(f"Error chunking text: {e}")
        return [text] if text else []

def scrub_pii_and_boilerplate(text: str) -> str:
    """Remove PII and common boilerplate from text for cleaner SBERT input."""
    if not text or not isinstance(text, str):
        return ""
    
    import re
    
    # Remove emails
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    
    # Remove phone numbers (various formats)
    text = re.sub(r'(\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}', '[PHONE]', text)
    
    # Remove URLs
    text = re.sub(r'https?://[^\s]+', '[URL]', text)
    text = re.sub(r'www\.[^\s]+', '[URL]', text)
    
    # Remove common headers/footers
    headers_to_remove = [
        r'^(resume|cv|curriculum vitae)$',
        r'^(personal information|contact information)$',
        r'^(objective|summary|profile)$',
        r'^(references available upon request)$',
        r'^(page \d+ of \d+)$',
        r'^(confidential|private)$'
    ]
    
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Check if line matches any header pattern
        is_header = False
        for pattern in headers_to_remove:
            if re.match(pattern, line, re.IGNORECASE):
                is_header = True
                break
        if not is_header:
            cleaned_lines.append(line)
    
    return ' '.join(cleaned_lines)

class SectionAwareTFIDF:
    """Section-aware TF-IDF implementation for resume processing."""
    
    def __init__(self):
        from .config import TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, TFIDF_MIN_DF, TFIDF_MAX_DF
        
        # Section weights for different resume sections
        self.section_weights = {
            'skills': 0.35,
            'experience': 0.45,
            'education': 0.15,
            'summary': 0.25,
            'projects': 0.30,
            'certifications': 0.20,
            'awards': 0.15,
            'languages': 0.10,
            'interests': 0.05,
            'volunteer': 0.15,
            'leadership': 0.20,
            'availability': 0.05,
            'references': 0.05,
            'contact': 0.00,  # No weight for contact info
            'default': 0.10   # Default weight for unknown sections
        }
        
        # TF-IDF configuration
        self.tfidf_config = {
            'max_features': TFIDF_MAX_FEATURES,
            'ngram_range': TFIDF_NGRAM_RANGE,
            'min_df': TFIDF_MIN_DF,
            'max_df': TFIDF_MAX_DF
        }
        
        # TF-IDF vectorizers for different section types
        self.vectorizers = {}
        self.section_vectors = {}
        
    def build_section_vectors(self, resume_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Build TF-IDF vectors for each section of a resume."""
        sections = resume_data.get('sections', [])
        section_vectors = {}
        
        for section in sections:
            header = section['header'].lower()
            content = section['content']
            
            # Determine section type
            section_type = self._classify_section(header)
            
            # Combine content into single text
            section_text = ' '.join(content)
            
            # Create or get vectorizer for this section type
            if section_type not in self.vectorizers:
                self.vectorizers[section_type] = TfidfVectorizer(
                    max_features=1000,
                    stop_words='english',
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.95
                )
            
            # Fit and transform the section text
            try:
                vector = self.vectorizers[section_type].fit_transform([section_text])
                section_vectors[section_type] = vector.toarray()[0]
            except Exception as e:
                logger.warning(f"Error vectorizing section {section_type}: {str(e)}")
                section_vectors[section_type] = np.zeros(1000)  # Default empty vector
        
        return section_vectors
    
    def _classify_section(self, header: str) -> str:
        """Classify a section header into a standard category."""
        header_lower = header.lower()
        
        # Map headers to standard categories
        if any(word in header_lower for word in ['skill', 'competency', 'expertise', 'proficiency']):
            return 'skills'
        elif any(word in header_lower for word in ['experience', 'employment', 'work', 'career']):
            return 'experience'
        elif any(word in header_lower for word in ['education', 'academic', 'qualification']):
            return 'education'
        elif any(word in header_lower for word in ['summary', 'profile', 'objective']):
            return 'summary'
        elif any(word in header_lower for word in ['project']):
            return 'projects'
        elif any(word in header_lower for word in ['certification', 'certificate', 'license']):
            return 'certifications'
        elif any(word in header_lower for word in ['award', 'honor', 'achievement']):
            return 'awards'
        elif any(word in header_lower for word in ['language']):
            return 'languages'
        elif any(word in header_lower for word in ['interest', 'hobby']):
            return 'interests'
        elif any(word in header_lower for word in ['volunteer']):
            return 'volunteer'
        elif any(word in header_lower for word in ['leadership']):
            return 'leadership'
        elif any(word in header_lower for word in ['availability']):
            return 'availability'
        elif any(word in header_lower for word in ['reference', 'referee']):
            return 'references'
        elif any(word in header_lower for word in ['contact', 'personal']):
            return 'contact'
        else:
            return 'default'
    
    def combine_weighted_vectors(self, section_vectors: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine section vectors with appropriate weights."""
        if not section_vectors:
            return np.zeros(1000)
        
        # Get the maximum vector length
        max_length = max(len(vector) for vector in section_vectors.values())
        
        # Initialize combined vector
        combined_vector = np.zeros(max_length)
        total_weight = 0
        
        for section_type, vector in section_vectors.items():
            weight = self.section_weights.get(section_type, self.section_weights['default'])
            
            # Pad or truncate vector to match max_length
            if len(vector) < max_length:
                padded_vector = np.pad(vector, (0, max_length - len(vector)), 'constant')
            else:
                padded_vector = vector[:max_length]
            
            combined_vector += weight * padded_vector
            total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            combined_vector /= total_weight
        
        return combined_vector
    
    def calculate_similarity(self, resume1_vectors: Dict[str, np.ndarray], 
                           resume2_vectors: Dict[str, np.ndarray]) -> float:
        """Calculate similarity between two resumes using weighted section vectors."""
        timing_bucket = {}
        
        with time_phase("vectorize", timing_bucket):
            combined1 = self.combine_weighted_vectors(resume1_vectors)
            combined2 = self.combine_weighted_vectors(resume2_vectors)
            
            # Reshape for cosine similarity
            combined1 = combined1.reshape(1, -1)
            combined2 = combined2.reshape(1, -1)
        
        with time_phase("cosine", timing_bucket):
            similarity = cosine_similarity(combined1, combined2)[0][0]
        
        return similarity

    def compute_section_tfidf_scores(self, resumes: List[Dict[str, Any]], jd_sections: Dict[str, str]) -> Tuple[List[float], List[float]]:
        """Compute section TF-IDF scores with separate channels and normalization."""
        try:
            logger.info(f"Starting TF-IDF computation with {len(resumes)} resumes")
            if not resumes:
                logger.warning("No resumes provided for TF-IDF scoring")
                return [], []
            
            # Normalize JD sections if they're raw text
            if isinstance(jd_sections, str):
                logger.info("Normalizing raw JD text")
                jd_normalized = normalize_job_description(jd_sections)
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
            
            # Channel 1: Section TF-IDF - separate vectorizer for section content
            section_texts = []
            for resume in resumes:
                try:
                    sections = resume.get('sections', {})
                    if isinstance(sections, dict):
                        section_text = ' '.join([
                            str(sections.get('experience', '')),
                            str(sections.get('skills', '')),
                            str(sections.get('education', '')),
                            str(sections.get('misc', ''))
                        ])
                        # Fallback: if the expected keys are sparse/empty, concatenate all section values
                        if not section_text.strip():
                            try:
                                section_text = ' '.join([str(v) for v in sections.values()])
                            except Exception:
                                section_text = ''
                    else:
                        logger.warning(f"Invalid sections format for resume: {type(sections)}")
                        section_text = ''
                    
                    # Preprocess the text (preserve skill tokens like Node.js, C++, C#)
                    section_text = preprocess_text_for_dense_models(section_text)
                    section_texts.append(section_text)
                except Exception as e:
                    logger.warning(f"Error processing resume sections: {e}")
                    section_texts.append('')
            
            jd_section_text = ' '.join([
                str(jd_sections.get('experience', '')),
                str(jd_sections.get('skills', '')),
                str(jd_sections.get('education', '')),
                str(jd_sections.get('misc', ''))
            ])
            
            # Preprocess JD text (preserve skill tokens)
            jd_section_text = preprocess_text_for_dense_models(jd_section_text)
            
            # Handle empty text case
            if not jd_section_text.strip():
                logger.warning("Empty job description sections, returning zero scores")
                return [0.0] * len(resumes), [0.0] * len(resumes)
            
            # Check cache for section vectorizer
            jd_hash = _jd_hash(jd_section_text, self.tfidf_config['max_features'], self.tfidf_config['ngram_range'])
            if jd_hash in _section_vectorizer_cache:
                logger.info(f"[PERF] Using cached section vectorizer for JD hash: {jd_hash[:8]}")
                vectorizer = _section_vectorizer_cache[jd_hash]
                all_texts = [jd_section_text] + section_texts
                vectors = vectorizer.transform(all_texts)
            else:
                # Fit TF-IDF on section texts with separate config
                section_config = self.tfidf_config.copy()
                section_config['max_features'] = 1000  # Smaller for sections
                vectorizer = TfidfVectorizer(**section_config)
                all_texts = [jd_section_text] + section_texts
                vectors = vectorizer.fit_transform(all_texts)
                _section_vectorizer_cache[jd_hash] = vectorizer
                logger.info(f"[PERF] Cached section vectorizer for JD hash: {jd_hash[:8]}")
            
            # Compute similarities
            jd_vector = vectors[0:1]
            resume_vectors = vectors[1:]
            similarities = cosine_similarity(jd_vector, resume_vectors).flatten()
            
            # Normalize to [0,1]
            raw_scores = similarities.tolist()
            if raw_scores and len(raw_scores) > 0:
                min_score, max_score = min(raw_scores), max(raw_scores)
                if max_score > min_score:
                    norm_scores = [_norm(score, min_score, max_score) for score in raw_scores]
                else:
                    norm_scores = [0.0] * len(raw_scores)
            else:
                norm_scores = [0.0] * len(resumes)
            
            # Channel 2: Skill/Taxonomy TF-IDF - separate vectorizer for skill tokens
            taxonomy = SkillTaxonomy()
            skill_texts = []
            for resume in resumes:
                try:
                    skills = resume.get('matched_skills', [])
                    derived_tokens = []
                    if isinstance(skills, list) and len(skills) > 0:
                        # Normalize existing skill_id tokens to canonical taxonomy terms
                        for skill in skills:
                            if isinstance(skill, dict):
                                token = str(skill.get('skill_id', '')).strip()
                                if token:
                                    derived_tokens.append(taxonomy.normalize_skill(token))
                    else:
                        # Backfill: derive skills from resume text using taxonomy
                        sections = resume.get('sections', {})
                        resume_text = ''
                        if isinstance(sections, dict):
                            try:
                                resume_text = ' '.join([str(v) for v in sections.values()])
                            except Exception:
                                resume_text = ''
                        derived_tokens = taxonomy.extract_skills_from_text(resume_text)
                    
                    skill_text = ' '.join(sorted(set([t for t in derived_tokens if t])))
                    # Preprocess the skill text
                    skill_text = preprocess_text(skill_text)
                    skill_texts.append(skill_text)
                except Exception as e:
                    logger.warning(f"Error processing resume skills: {e}")
                    skill_texts.append('')
            
            # Canonicalize JD skills to taxonomy tokens; fallback to entire JD sections if empty
            jd_skills_raw = str(jd_sections.get('skills', ''))
            jd_tokens = taxonomy.extract_skills_from_text(jd_skills_raw)
            if not jd_tokens:
                try:
                    jd_all_text = ' '.join([
                        str(jd_sections.get('experience', '')),
                        str(jd_sections.get('skills', '')),
                        str(jd_sections.get('education', '')),
                        str(jd_sections.get('misc', ''))
                    ])
                except Exception:
                    jd_all_text = jd_skills_raw
                jd_tokens = taxonomy.extract_skills_from_text(jd_all_text)
            jd_skills = preprocess_text(' '.join(sorted(set(jd_tokens))))
            
            # Check cache for skill vectorizer
            skill_jd_hash = _jd_hash(jd_skills, self.tfidf_config['max_features'], self.tfidf_config['ngram_range'])
            if skill_jd_hash in _skill_vectorizer_cache:
                logger.info(f"[PERF] Using cached skill vectorizer for JD hash: {skill_jd_hash[:8]}")
                skill_vectorizer = _skill_vectorizer_cache[skill_jd_hash]
                all_skill_texts = [jd_skills] + skill_texts
                skill_vectors = skill_vectorizer.transform(all_skill_texts)
            else:
                # Fit TF-IDF on skill tokens with separate config
                skill_config = self.tfidf_config.copy()
                skill_config['max_features'] = 500  # Smaller for skills
                skill_config['ngram_range'] = (1, 2)  # Focus on unigrams and bigrams for skills
                skill_vectorizer = TfidfVectorizer(**skill_config)
                all_skill_texts = [jd_skills] + skill_texts
                skill_vectors = skill_vectorizer.fit_transform(all_skill_texts)
                _skill_vectorizer_cache[skill_jd_hash] = skill_vectorizer
                logger.info(f"[PERF] Cached skill vectorizer for JD hash: {skill_jd_hash[:8]}")
            
            # Compute skill similarities
            jd_skill_vector = skill_vectors[0:1]
            resume_skill_vectors = skill_vectors[1:]
            skill_similarities = cosine_similarity(jd_skill_vector, resume_skill_vectors).flatten()
            
            # Normalize skill scores to [0,1]
            skill_raw_scores = skill_similarities.tolist()
            if skill_raw_scores and len(skill_raw_scores) > 0:
                skill_min, skill_max = min(skill_raw_scores), max(skill_raw_scores)
                if skill_max > skill_min:
                    skill_norm_scores = [_norm(score, skill_min, skill_max) for score in skill_raw_scores]
                else:
                    skill_norm_scores = [0.0] * len(skill_raw_scores)
            else:
                skill_norm_scores = [0.0] * len(resumes)
            
            logger.info(f"TF-IDF computation completed successfully. Section scores: {len(norm_scores)}, Skill scores: {len(skill_norm_scores)}")
            return norm_scores, skill_norm_scores
            
        except Exception as e:
            logger.error(f"Error in compute_section_tfidf_scores: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return [0.0] * len(resumes), [0.0] * len(resumes)


class SkillTaxonomy:
    """Skill taxonomy integration for normalizing skill variations."""
    
    def __init__(self):
        # Skill taxonomy mapping (simplified version - can be expanded)
        self.skill_mapping = {
            # Programming Languages
            'python': ['python', 'python3', 'python 3', 'py'],
            'javascript': ['javascript', 'js', 'ecmascript', 'es6', 'es2015'],
            'java': ['java', 'j2ee', 'j2se'],
            'c++': ['c++', 'cpp', 'c plus plus'],
            'c#': ['c#', 'csharp', 'c sharp'],
            'php': ['php', 'php7', 'php8'],
            'ruby': ['ruby', 'ruby on rails', 'rails'],
            'go': ['go', 'golang'],
            'rust': ['rust'],
            'swift': ['swift', 'swiftui'],
            'kotlin': ['kotlin'],
            'scala': ['scala'],
            
            # Frameworks and Libraries
            'react': ['react', 'reactjs', 'react.js', 'reactjs', 'react native'],
            'angular': ['angular', 'angularjs', 'angular.js'],
            'vue': ['vue', 'vuejs', 'vue.js'],
            'nodejs': ['nodejs', 'node.js', 'node', 'express'],
            'django': ['django', 'django framework'],
            'flask': ['flask'],
            'spring': ['spring', 'spring boot', 'spring framework'],
            'laravel': ['laravel'],
            'asp.net': ['asp.net', 'aspnet', 'asp .net'],
            'jquery': ['jquery', 'jquery.js'],
            'bootstrap': ['bootstrap', 'bootstrap css'],
            'tailwind': ['tailwind', 'tailwind css'],
            
            # Databases
            'mysql': ['mysql', 'mariadb'],
            'postgresql': ['postgresql', 'postgres', 'psql'],
            'mongodb': ['mongodb', 'mongo'],
            'redis': ['redis'],
            'sqlite': ['sqlite', 'sqlite3'],
            'oracle': ['oracle', 'oracle db'],
            'sql server': ['sql server', 'mssql', 'microsoft sql server'],
            
            # Cloud and DevOps
            'aws': ['aws', 'amazon web services', 'amazon aws'],
            'azure': ['azure', 'microsoft azure'],
            'gcp': ['gcp', 'google cloud', 'google cloud platform'],
            'docker': ['docker', 'docker container'],
            'kubernetes': ['kubernetes', 'k8s'],
            'jenkins': ['jenkins'],
            'git': ['git', 'github', 'gitlab', 'bitbucket'],
            'ci/cd': ['ci/cd', 'continuous integration', 'continuous deployment'],
            
            # Data Science and ML
            'tensorflow': ['tensorflow', 'tf'],
            'pytorch': ['pytorch', 'torch'],
            'scikit-learn': ['scikit-learn', 'sklearn', 'scikit learn'],
            'pandas': ['pandas', 'pd'],
            'numpy': ['numpy', 'np'],
            'matplotlib': ['matplotlib', 'plt'],
            'seaborn': ['seaborn'],
            'jupyter': ['jupyter', 'jupyter notebook'],
            
            # Other Technologies
            'html': ['html', 'html5'],
            'css': ['css', 'css3'],
            'sass': ['sass', 'scss'],
            'less': ['less'],
            'webpack': ['webpack'],
            'babel': ['babel'],
            'typescript': ['typescript', 'ts'],
            'graphql': ['graphql'],
            'rest': ['rest', 'rest api', 'restful'],
            'soap': ['soap', 'soap api'],
            'microservices': ['microservices', 'micro service'],
            'api': ['api', 'apis'],
            'linux': ['linux', 'ubuntu', 'centos', 'debian'],
            'chart.js': ['chartjs', 'chart.js', 'chart .js', 'chart. js'],
        }
        
        # Create reverse mapping for quick lookup
        self.reverse_mapping = {}
        for canonical, variations in self.skill_mapping.items():
            for variation in variations:
                self.reverse_mapping[variation.lower()] = canonical
    
    def normalize_skill(self, skill: str) -> str:
        """Normalize a skill to its canonical form."""
        skill_lower = skill.lower().strip()
        
        # Direct match
        if skill_lower in self.reverse_mapping:
            return self.reverse_mapping[skill_lower]
        
        # Partial match (for multi-word skills)
        for variation, canonical in self.reverse_mapping.items():
            if skill_lower in variation or variation in skill_lower:
                return canonical
        
        # Return original if no match found
        return skill
    
    def extract_skills_from_text(self, text: str) -> List[str]:
        """Extract and normalize skills from text content using taxonomy variations.
        Matches any known variation as a whole token (punctuation-aware) and returns
        canonical skill tokens.
        """
        if not text or not isinstance(text, str):
            return []
        text_lower = text.lower()
        found: set[str] = set()
        # Use punctuation-aware boundaries so terms like "c++", "c#", "asp.net" match
        left = r"(?<![A-Za-z0-9])"
        right = r"(?![A-Za-z0-9])"
        for variation, canonical in self.reverse_mapping.items():
            try:
                pattern = left + re.escape(variation) + right
                if re.search(pattern, text_lower):
                    found.add(canonical)
            except Exception:
                # Fallback: simple substring contains
                if variation in text_lower:
                    found.add(canonical)
        return sorted(found)
    
    def get_skill_taxonomy_score(self, resume_skills: List[str], job_skills: List[str]) -> float:
        """Calculate skill taxonomy overlap score."""
        if not resume_skills or not job_skills:
            return 0.0
        
        # Normalize all skills
        normalized_resume = [self.normalize_skill(skill) for skill in resume_skills]
        normalized_job = [self.normalize_skill(skill) for skill in job_skills]
        
        # Calculate overlap
        overlap = set(normalized_resume) & set(normalized_job)
        total_required = len(set(normalized_job))
        
        if total_required == 0:
            return 0.0
        
        return len(overlap) / total_required
    
    def get_matched_required_skills(self, resume_text: str, required_skills: List[str]) -> List[str]:
        """Get canonical list of required skills that are matched in resume text."""
        if not resume_text or not required_skills:
            return []
        
        # Extract skills from resume text
        resume_skills = self.extract_skills_from_text(resume_text)
        
        # Normalize required skills
        normalized_required = [self.normalize_skill(skill) for skill in required_skills]
        
        # Find matches
        matched = set(resume_skills) & set(normalized_required)
        return sorted(list(matched))


class SemanticEmbedding:
    """Semantic embedding generation for resume content."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.model_available = True
        except ImportError:
            logger.warning("SentenceTransformer not available. Install with: pip install sentence-transformers")
            self.model_available = False
            self.model = None
    
    def generate_section_embeddings(self, resume_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Generate embeddings for each section of a resume."""
        if not self.model_available:
            logger.error("SentenceTransformer model not available")
            return {}
        
        timing_bucket = {}
        
        with time_phase("vectorize", timing_bucket):
            sections = resume_data.get('sections', [])
            embeddings = {}
            
            for section in sections:
                header = section['header']
                content = section['content']
                
                # Combine content into single text
                section_text = ' '.join(content)
                
                if section_text.strip():
                    try:
                        # Generate embedding
                        embedding = self.model.encode(section_text)
                        embeddings[header] = embedding
                    except Exception as e:
                        logger.warning(f"Error generating embedding for section {header}: {str(e)}")
        
        return embeddings
    
    def generate_resume_embedding(self, resume_data: Dict[str, Any]) -> np.ndarray:
        """Generate a single embedding for the entire resume."""
        if not self.model_available:
            logger.error("SentenceTransformer model not available")
            return np.zeros(384)  # Default embedding size
        
        timing_bucket = {}
        
        with time_phase("vectorize", timing_bucket):
            # Combine all content
            all_content = []
            for section in resume_data.get('sections', []):
                all_content.extend(section['content'])
            
            combined_text = ' '.join(all_content)
            
            if combined_text.strip():
                try:
                    embedding = self.model.encode(combined_text)
                    return embedding
                except Exception as e:
                    logger.error(f"Error generating resume embedding: {str(e)}")
                    return np.zeros(384)
            
            return np.zeros(384)
    
    def calculate_semantic_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate semantic similarity between two embeddings."""
        if embedding1.size == 0 or embedding2.size == 0:
            return 0.0
        
        timing_bucket = {}
        
        with time_phase("cosine", timing_bucket):
            # Ensure same dimensions
            if embedding1.size != embedding2.size:
                min_size = min(embedding1.size, embedding2.size)
                embedding1 = embedding1[:min_size]
                embedding2 = embedding2[:min_size]
            
            # Calculate cosine similarity
            similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        
        return float(similarity)
    
    def compute_sbert_scores(self, resumes: List[Dict[str, Any]], job_description: str) -> List[float]:
        """Compute SBERT semantic similarity scores for all resumes."""
        try:
            if not self.model_available:
                logger.warning("SBERT model not available, returning zero scores")
                return [0.0] * len(resumes)
            
            timing_bucket = {}
            scores = []
            
            with time_phase("sbert_encode", timing_bucket):
                # Normalize and preprocess job description
                try:
                    if isinstance(job_description, str):
                        # Normalize JD if it's raw text
                        jd_normalized = normalize_job_description(job_description)
                        # Combine all JD sections for SBERT
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
                    
                    # Preprocess the JD text
                    jd_text = preprocess_text(jd_text)
                    
                    if not jd_text.strip():
                        logger.warning("Empty job description after preprocessing, returning zero scores")
                        return [0.0] * len(resumes)
                    
                    jd_embedding = self.model.encode(jd_text)
                except Exception as e:
                    logger.warning(f"Error encoding job description: {e}")
                    return [0.0] * len(resumes)
                
                for i, resume in enumerate(resumes):
                    try:
                        # Generate resume embedding from preprocessed text
                        resume_text = self._extract_preprocessed_text(resume)
                        # Preprocess resume text
                        resume_text = preprocess_text(resume_text)
                        
                        if not resume_text.strip():
                            logger.warning(f"Empty resume text for resume {i}, using zero score")
                            scores.append(0.0)
                            continue
                        
                        resume_embedding = self.model.encode(resume_text)
                        
                        # Calculate similarity
                        similarity = self.calculate_semantic_similarity(jd_embedding, resume_embedding)
                        scores.append(similarity)
                    except Exception as e:
                        logger.warning(f"Error processing resume {i} for SBERT: {e}")
                        scores.append(0.0)
            
            # Normalize scores to [0,1]
            if scores:
                min_score, max_score = min(scores), max(scores)
                if max_score > min_score:
                    scores = [_norm(score, min_score, max_score) for score in scores]
                else:
                    scores = [0.5] * len(scores)  # All same score
            
            return scores
            
        except Exception as e:
            logger.error(f"Error in compute_sbert_scores: {e}")
            return [0.0] * len(resumes)
    
    def _extract_preprocessed_text(self, resume: Dict[str, Any]) -> str:
        """Extract preprocessed text from resume for SBERT scoring."""
        try:
            sections = resume.get('sections', {})
            all_content = []
            
            if isinstance(sections, dict):
                # Handle dict format sections (most common)
                for section_name, section_content in sections.items():
                    try:
                        if isinstance(section_content, list):
                            all_content.extend([str(item) for item in section_content])
                        else:
                            all_content.append(str(section_content))
                    except Exception as e:
                        logger.warning(f"Error processing section {section_name}: {e}")
                        continue
            elif isinstance(sections, list):
                # Handle list format sections
                for section in sections:
                    try:
                        if isinstance(section, dict):
                            content = section.get('content', [])
                            if isinstance(content, list):
                                all_content.extend([str(item) for item in content])
                            else:
                                all_content.append(str(content))
                        else:
                            logger.warning(f"Invalid section format: {type(section)}")
                    except Exception as e:
                        logger.warning(f"Error processing section: {e}")
                        continue
            else:
                logger.warning(f"Invalid sections format: {type(sections)}")
            
            return ' '.join(all_content)
            
        except Exception as e:
            logger.warning(f"Error extracting preprocessed text: {e}")
            return ""


class ResumeProcessor:
    """Main processor that combines all text processing components."""
    
    def __init__(self):
        self.tfidf_processor = SectionAwareTFIDF()
        self.skill_taxonomy = SkillTaxonomy()
        self.semantic_embedding = SemanticEmbedding()
    
    def process_resume(self, resume_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a resume through all text processing components."""
        processed_data = {
            'original_data': resume_data,
            'tfidf_vectors': {},
            'skill_analysis': {},
            'semantic_embeddings': {},
            'combined_scores': {}
        }
        
        try:
            # Generate TF-IDF vectors
            section_vectors = self.tfidf_processor.build_section_vectors(resume_data)
            combined_vector = self.tfidf_processor.combine_weighted_vectors(section_vectors)
            processed_data['tfidf_vectors'] = {
                'section_vectors': section_vectors,
                'combined_vector': combined_vector.tolist()
            }
            
            # Extract and normalize skills
            all_content = []
            for section in resume_data.get('sections', []):
                all_content.extend(section['content'])
            
            combined_text = ' '.join(all_content)
            extracted_skills = self.skill_taxonomy.extract_skills_from_text(combined_text)
            
            processed_data['skill_analysis'] = {
                'extracted_skills': extracted_skills,
                'skill_count': len(extracted_skills)
            }
            
            # Generate semantic embeddings
            section_embeddings = self.semantic_embedding.generate_section_embeddings(resume_data)
            resume_embedding = self.semantic_embedding.generate_resume_embedding(resume_data)
            
            processed_data['semantic_embeddings'] = {
                'section_embeddings': {k: v.tolist() for k, v in section_embeddings.items()},
                'resume_embedding': resume_embedding.tolist()
            }
            
            # Calculate combined scores
            processed_data['combined_scores'] = {
                'tfidf_score': float(np.linalg.norm(combined_vector)),
                'skill_diversity': len(extracted_skills),
                'semantic_richness': float(np.linalg.norm(resume_embedding))
            }
            
        except Exception as e:
            logger.error(f"Error processing resume: {str(e)}")
            processed_data['error'] = str(e)
        
        return processed_data
    
    def compare_resumes(self, resume1_data: Dict[str, Any], resume2_data: Dict[str, Any]) -> Dict[str, float]:
        """Compare two resumes using all similarity metrics."""
        # Process both resumes
        processed1 = self.process_resume(resume1_data)
        processed2 = self.process_resume(resume2_data)
        
        comparison_scores = {}
        
        try:
            # TF-IDF similarity
            if 'tfidf_vectors' in processed1 and 'tfidf_vectors' in processed2:
                tfidf_sim = self.tfidf_processor.calculate_similarity(
                    processed1['tfidf_vectors']['section_vectors'],
                    processed2['tfidf_vectors']['section_vectors']
                )
                comparison_scores['tfidf_similarity'] = tfidf_sim
            
            # Skill taxonomy similarity
            skills1 = processed1['skill_analysis'].get('extracted_skills', [])
            skills2 = processed2['skill_analysis'].get('extracted_skills', [])
            skill_sim = self.skill_taxonomy.get_skill_taxonomy_score(skills1, skills2)
            comparison_scores['skill_similarity'] = skill_sim
            
            # Semantic similarity
            if 'semantic_embeddings' in processed1 and 'semantic_embeddings' in processed2:
                emb1 = np.array(processed1['semantic_embeddings']['resume_embedding'])
                emb2 = np.array(processed2['semantic_embeddings']['resume_embedding'])
                semantic_sim = self.semantic_embedding.calculate_semantic_similarity(emb1, emb2)
                comparison_scores['semantic_similarity'] = semantic_sim
            
            # Overall similarity (weighted average)
            similarities = [v for v in comparison_scores.values() if isinstance(v, (int, float))]
            if similarities:
                comparison_scores['overall_similarity'] = sum(similarities) / len(similarities)
            
        except Exception as e:
            logger.error(f"Error comparing resumes: {str(e)}")
            comparison_scores['error'] = str(e)
        
        return comparison_scores 