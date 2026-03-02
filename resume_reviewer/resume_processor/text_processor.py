import json
import re
import numpy as np
import hashlib
import unicodedata
from typing import List, Dict, Any, Tuple, Optional
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


def dehyphenate_text(text: str) -> str:
    """Join soft-wrapped lines (e.g., "develop-\nment" → "development")."""
    try:
        if not text:
            return ""
        
        # Pattern: word followed by hyphen, newline, and continuation
        # This handles cases like "develop-\nment" or "soft-\nware"
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        
        return text
    except Exception as e:
        logger.warning(f"Error dehyphenating text: {e}")
        return text


def normalize_bullets(text: str) -> str:
    """Collapse bullet variants (•/●/-/*) to standard marker."""
    try:
        if not text:
            return ""
        
        # Replace various bullet characters with standard bullet
        bullet_chars = ['•', '●', '◦', '▪', '▫', '‣', '⁃']
        for char in bullet_chars:
            text = text.replace(char, '•')
        
        # Also handle dash bullets at start of line
        text = re.sub(r'^\s*-\s+', '• ', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\*\s+', '• ', text, flags=re.MULTILINE)
        
        return text
    except Exception as e:
        logger.warning(f"Error normalizing bullets: {e}")
        return text


def join_wrapped_lines(text: str) -> str:
    """Merge lines that end mid-sentence (no punctuation)."""
    try:
        if not text:
            return ""
        
        lines = text.split('\n')
        joined_lines = []
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                joined_lines.append('')
                i += 1
                continue
            
            # Check if line ends without punctuation and next line starts with lowercase
            if (i + 1 < len(lines) and 
                not re.search(r'[.!?:;]$', line) and 
                lines[i + 1].strip() and 
                lines[i + 1].strip()[0].islower()):
                # Join with space
                line += ' ' + lines[i + 1].strip()
                i += 2
            else:
                i += 1
            
            joined_lines.append(line)
        
        return '\n'.join(joined_lines)
    except Exception as e:
        logger.warning(f"Error joining wrapped lines: {e}")
        return text


def strip_boilerplate_headers(lines: List[str]) -> List[str]:
    """Remove "Resume", "CV", "Page X of Y", etc."""
    try:
        if not lines:
            return []
        
        # Common boilerplate patterns to remove
        boilerplate_patterns = [
            r'^(resume|cv|curriculum vitae)$',
            r'^(personal information|contact information)$',
            r'^(objective|summary|profile)$',
            r'^(references available upon request)$',
            r'^(page \d+ of \d+)$',
            r'^(confidential|private)$',
            r'^(draft|template)$'
        ]
        
        filtered_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                filtered_lines.append('')
                continue
            
            # Check if line matches any boilerplate pattern
            is_boilerplate = False
            for pattern in boilerplate_patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    is_boilerplate = True
                    break
            
            if not is_boilerplate:
                filtered_lines.append(line)
        
        return filtered_lines
    except Exception as e:
        logger.warning(f"Error stripping boilerplate: {e}")
        return lines


def preprocess_resume_text(raw_text: str) -> str:
    """Full preprocessing pipeline for resume text."""
    try:
        if not raw_text:
            return ""
        
        # Unicode normalization
        import unicodedata
        text = unicodedata.normalize('NFKC', raw_text)
        
        # Apply preprocessing steps
        text = dehyphenate_text(text)
        text = join_wrapped_lines(text)
        text = normalize_bullets(text)
        
        # Collapse whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    except Exception as e:
        logger.warning(f"Error in resume text preprocessing: {e}")
        return raw_text


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

class SemanticSkillMatcher:
    """Semantic matcher for fuzzy skill matching using SBERT.
    
    This class provides fuzzy matching capabilities for skills that don't have
    exact matches in the taxonomy. It uses sentence transformers to compute
    semantic similarity between input skills and canonical taxonomy skills.
    
    Attributes:
        canonical_skills: List of canonical skill names to match against
        model: SentenceTransformer model (lazy-loaded)
        model_available: Whether the model is available
        embeddings: Pre-computed embeddings for all canonical skills
    """
    
    def __init__(self, canonical_skills: List[str]):
        """Initialize with list of canonical skills to match against."""
        self.canonical_skills = canonical_skills
        self.model = None
        self.model_available = False
        self.embeddings = None
        
        try:
            from sentence_transformers import SentenceTransformer
            # Use lightweight model for speed
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.model_available = True
            # Pre-embed all canonical skills
            if canonical_skills:
                self.embeddings = self.model.encode(canonical_skills, show_progress_bar=False)
            logger.info(f"SemanticSkillMatcher initialized with {len(canonical_skills)} skills")
        except ImportError:
            logger.warning("SentenceTransformer not available for fuzzy matching")
            self.model_available = False
        except Exception as e:
            logger.warning(f"Failed to initialize semantic matcher: {e}")
            self.model_available = False
    
    def find_closest_skill(self, skill: str, threshold: float = 0.7) -> Optional[Tuple[str, float]]:
        """Find closest canonical skill using semantic similarity."""
        if not self.model_available or self.embeddings is None:
            return None
        
        try:
            # Encode the input skill
            skill_embedding = self.model.encode([skill], show_progress_bar=False)[0]
            
            # Compute cosine similarity with all canonical skills
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity([skill_embedding], self.embeddings)[0]
            
            # Find best match
            best_idx = similarities.argmax()
            best_score = float(similarities[best_idx])
            
            if best_score >= threshold:
                return (self.canonical_skills[best_idx], best_score)
            return None
        except Exception as e:
            logger.warning(f"Error in semantic skill matching: {e}")
            return None


class SkillTaxonomy:
    """Skill taxonomy integration for normalizing skill variations.
    
    This class provides a multi-domain skill taxonomy system that:
    - Maintains core tech skills (hard-coded for performance)
    - Loads domain-specific skills from JSON files (Marketing, Healthcare, Finance, etc.)
    - Supports fuzzy matching using semantic similarity for unknown skills
    - Provides confidence scoring for exact vs fuzzy matches
    
    The taxonomy automatically loads skills from the `skill_taxonomy/` directory
    based on configuration in `config.json`. Domains can be enabled/disabled
    without code changes.
    
    Attributes:
        skill_mapping: Dictionary mapping canonical skill names to variations
        reverse_mapping: Reverse lookup from variations to canonical names
        config: Configuration loaded from config.json
        fuzzy_enabled: Whether fuzzy matching is enabled
        fuzzy_threshold: Minimum similarity threshold for fuzzy matches
        loaded_domains: List of successfully loaded domain names
    """
    
    def __init__(self, taxonomy_dir: Optional[str] = None):
        """Initialize skill taxonomy with core tech skills and optional domain-specific skills."""
        import os
        import json
        
        # Core tech skills (hard-coded for performance)
        self.skill_mapping = {
            # Programming Languages
            'python': ['python', 'python3', 'python 3', 'py', 'pythonic', 'pypy'],
            'javascript': ['javascript', 'js', 'ecmascript', 'es6', 'es2015', 'es2016', 'es2017', 'es2018', 'es2019', 'es2020', 'es2021', 'es2022', 'es2023'],
            'java': ['java', 'j2ee', 'j2se', 'jdk', 'jre', 'jvm', 'javase', 'javaee'],
            'c++': ['c++', 'cpp', 'c plus plus', 'cplusplus', 'cxx'],
            'c#': ['c#', 'csharp', 'c sharp', 'csharp', 'dotnet', '.net'],
            'php': ['php', 'php7', 'php8', 'php5', 'php4'],
            'ruby': ['ruby', 'ruby on rails', 'rails', 'ror'],
            'go': ['go', 'golang', 'go lang'],
            'rust': ['rust', 'rustlang'],
            'swift': ['swift', 'swiftui', 'swift ui'],
            'kotlin': ['kotlin', 'kotlin android'],
            'scala': ['scala', 'scala lang'],
            'r': ['r', 'r language', 'r programming'],
            'matlab': ['matlab', 'matlab programming'],
            'perl': ['perl', 'perl5', 'perl6'],
            'haskell': ['haskell', 'haskell programming'],
            'clojure': ['clojure', 'clojure programming'],
            'erlang': ['erlang', 'erlang programming'],
            'elixir': ['elixir', 'elixir programming'],
            'dart': ['dart', 'dart programming'],
            'c': ['c', 'c programming', 'ansi c'],
            'cobol': ['cobol', 'cobol programming'],
            'fortran': ['fortran', 'fortran programming'],
            'assembly': ['assembly', 'asm', 'x86', 'x64', 'arm'],
            
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
            'chart.js': ['chartjs', 'chart.js'],
        }
        
        # Load domain-specific taxonomies from JSON files
        self.config = {}
        self.fuzzy_matcher = None
        self.fuzzy_cache = {}
        self.loaded_domains = []
        
        if taxonomy_dir is None:
            # Default to skill_taxonomy directory relative to this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            taxonomy_dir = os.path.join(current_dir, 'skill_taxonomy')
        
        self.taxonomy_dir = taxonomy_dir
        
        # Load configuration
        config_path = os.path.join(taxonomy_dir, 'config.json')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
                logger.info(f"Loaded skill taxonomy config from {config_path}")
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON in taxonomy config {config_path}: {e}")
                self.config = {}
            except Exception as e:
                logger.warning(f"Failed to load taxonomy config: {e}")
                self.config = {}
        else:
            logger.debug(f"Taxonomy config not found at {config_path}, using defaults")
        
        # Load enabled domain taxonomies
        enabled_domains = self.config.get('enabled_domains', [])
        total_loaded_skills = 0
        for domain in enabled_domains:
            domain_file = f"{domain}.json"
            domain_path = os.path.join(taxonomy_dir, domain_file)
            if os.path.exists(domain_path):
                try:
                    domain_skills = self._load_domain_taxonomy(domain_path)
                    if domain_skills:
                        self.skill_mapping.update(domain_skills)
                        total_loaded_skills += len(domain_skills)
                        self.loaded_domains.append(domain)
                        logger.info(f"Loaded {len(domain_skills)} skills from {domain_file}")
                    else:
                        logger.warning(f"Domain taxonomy {domain_file} is empty or invalid")
                except Exception as e:
                    logger.warning(f"Failed to load domain taxonomy {domain_file}: {e}")
            else:
                logger.debug(f"Domain taxonomy file not found: {domain_path}")
        
        if total_loaded_skills > 0:
            logger.info(f"Skill taxonomy initialized: {len(self.skill_mapping)} total skills "
                       f"({len(self.skill_mapping) - total_loaded_skills} core tech + "
                       f"{total_loaded_skills} from {len(self.loaded_domains)} domains)")
        else:
            logger.info(f"Skill taxonomy initialized with {len(self.skill_mapping)} core tech skills only")
        
        # Create reverse mapping for quick lookup
        self.reverse_mapping = {}
        for canonical, variations in self.skill_mapping.items():
            for variation in variations:
                self.reverse_mapping[variation.lower()] = canonical
        
        # Initialize fuzzy matcher if enabled (lazy-loaded)
        self.fuzzy_enabled = self.config.get('fuzzy_matching', {}).get('enabled', False)
        self.fuzzy_threshold = self.config.get('fuzzy_matching', {}).get('threshold', 0.7)
        self.confidence_weights = self.config.get('confidence_weights', {
            'exact_match': 1.0,
            'fuzzy_match': 0.6
        })
    
    def _load_domain_taxonomy(self, domain_file: str) -> Dict[str, List[str]]:
        """Load domain-specific taxonomy from JSON file.
        
        Args:
            domain_file: Path to JSON file containing domain skills
            
        Returns:
            Dictionary mapping canonical skill names to lists of variations
        """
        try:
            with open(domain_file, 'r', encoding='utf-8') as f:
                domain_data = json.load(f)
            
            # Validate structure: should be {canonical: [variations]}
            if not isinstance(domain_data, dict):
                logger.warning(f"Invalid taxonomy structure in {domain_file}: expected dict, got {type(domain_data)}")
                return {}
            
            # Validate each entry
            validated_data = {}
            for canonical, variations in domain_data.items():
                if isinstance(canonical, str) and isinstance(variations, list):
                    # Ensure all variations are strings
                    validated_variations = [str(v) for v in variations if v]
                    if validated_variations:
                        validated_data[canonical] = validated_variations
                    else:
                        logger.debug(f"Skipping {canonical} in {domain_file}: no valid variations")
                else:
                    logger.debug(f"Skipping invalid entry in {domain_file}: {canonical}")
            
            return validated_data
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in domain taxonomy {domain_file}: {e}")
            return {}
        except FileNotFoundError:
            logger.warning(f"Domain taxonomy file not found: {domain_file}")
            return {}
        except Exception as e:
            logger.warning(f"Error loading domain taxonomy {domain_file}: {e}")
            return {}
    
    def _get_fuzzy_matcher(self) -> Optional[SemanticSkillMatcher]:
        """Lazy-load fuzzy matcher if enabled."""
        if not self.fuzzy_enabled:
            return None
        
        if self.fuzzy_matcher is None:
            canonical_skills = list(self.skill_mapping.keys())
            self.fuzzy_matcher = SemanticSkillMatcher(canonical_skills)
        
        return self.fuzzy_matcher if self.fuzzy_matcher.model_available else None
    
    def _fuzzy_match_skill(self, skill: str, threshold: Optional[float] = None) -> Optional[Tuple[str, float]]:
        """Use semantic similarity to find closest canonical skill."""
        if threshold is None:
            threshold = self.fuzzy_threshold
        
        # Check cache first
        cache_key = f"{skill.lower()}_{threshold}"
        if cache_key in self.fuzzy_cache:
            return self.fuzzy_cache[cache_key]
        
        matcher = self._get_fuzzy_matcher()
        if matcher is None:
            return None
        
        result = matcher.find_closest_skill(skill, threshold)
        
        # Cache result
        if result is not None:
            self.fuzzy_cache[cache_key] = result
        
        return result
    
    def normalize_skill(self, skill: str, use_fuzzy: bool = True) -> str:
        """Normalize a skill to its canonical form with enhanced punctuation cleanup.
        
        Args:
            skill: The skill name to normalize
            use_fuzzy: If True, use fuzzy matching as fallback when exact match fails
        
        Returns:
            Canonical skill name
        """
        if not skill or not isinstance(skill, str):
            return ""
        
        # Enhanced cleanup with punctuation normalization
        skill_clean = self._normalize_skill_punctuation(skill.strip())
        skill_lower = skill_clean.lower()
        
        # Direct match - exact match in taxonomy
        if skill_lower in self.reverse_mapping:
            return self.reverse_mapping[skill_lower]
        
        # Partial match: ONLY match if the INPUT contains a known taxonomy variation
        # This prevents generic terms like "project management" from matching 
        # tool-specific terms like "asana project management" -> "asana"
        for variation, canonical in self.reverse_mapping.items():
            # Skip very short variations (< 3 chars) for substring matching
            if len(variation) < 3:
                continue
            # ONLY match if the input CONTAINS the variation (input is longer or equal)
            # This ensures "asana project management" -> "asana", but NOT "project management" -> "asana"
            if variation in skill_lower and variation != skill_lower:
                # Require variation to be at least 50% of input length
                if len(variation) >= len(skill_lower) * 0.5:
                    return canonical
        
        # Fuzzy matching as fallback - but only if enabled and skill is long enough
        if use_fuzzy and self.fuzzy_enabled and len(skill_lower) >= 3:
            fuzzy_result = self._fuzzy_match_skill(skill_clean)
            if fuzzy_result:
                return fuzzy_result[0]
        
        # Return cleaned version if no match found - PRESERVE USER INPUT
        return skill_clean
    
    def normalize_skill_with_confidence(self, skill: str) -> Tuple[str, float]:
        """Normalize a skill and return confidence score.
        
        Returns:
            Tuple of (canonical_skill, confidence) where:
            - confidence = 1.0 for exact matches
            - confidence = fuzzy_score * fuzzy_weight for fuzzy matches
            - confidence = 0.0 for no matches
        """
        if not skill or not isinstance(skill, str):
            return ("", 0.0)
        
        skill_clean = self._normalize_skill_punctuation(skill.strip())
        skill_lower = skill_clean.lower()
        
        # Direct match - highest confidence
        if skill_lower in self.reverse_mapping:
            return (self.reverse_mapping[skill_lower], self.confidence_weights['exact_match'])
        
        # Partial match - high confidence, but ONLY if lengths are similar
        # This prevents "r" from matching "system monitoring" just because "r" is in the string
        for variation, canonical in self.reverse_mapping.items():
            # Skip very short variations (like single letters) for substring matching
            if len(variation) < 3:
                continue
            # Only match if one is a substantial substring of the other (>50% length)
            if skill_lower in variation:
                if len(skill_lower) >= len(variation) * 0.5:
                    return (canonical, self.confidence_weights['exact_match'])
            elif variation in skill_lower:
                if len(variation) >= len(skill_lower) * 0.5:
                    return (canonical, self.confidence_weights['exact_match'])
        
        # Fuzzy matching - lower confidence
        if self.fuzzy_enabled:
            fuzzy_result = self._fuzzy_match_skill(skill_clean)
            if fuzzy_result:
                canonical, fuzzy_score = fuzzy_result
                fuzzy_weight = self.confidence_weights.get('fuzzy_match', 0.6)
                confidence = fuzzy_score * fuzzy_weight
                return (canonical, confidence)
        
        # No match
        return (skill_clean, 0.0)
    
    def _normalize_skill_punctuation(self, skill: str) -> str:
        """Normalize punctuation and spacing in skill names with enhanced thesis-ready normalization.
        
        FIX: Strips ALL punctuation (periods, hyphens) and lowercases for consistent matching.
        This ensures "Node.js" matches "nodejs", "React.js" matches "reactjs", etc.
        """
        if not skill:
            return ""
        
        # First, handle special cases that need to be preserved before stripping
        # These are skills where punctuation is meaningful (C++, C#, .NET)
        skill_lower = skill.lower().strip()
        
        # Handle C++ variations (must check before stripping +)
        if 'c++' in skill_lower or 'cpp' in skill_lower or 'c plus plus' in skill_lower:
            return 'cpp'
        
        # Handle C# variations (must check before stripping #)
        if 'c#' in skill_lower or 'csharp' in skill_lower or 'c sharp' in skill_lower:
            return 'csharp'
        
        # Enhanced canonical mappings for common tech stack variations
        # Check these BEFORE stripping punctuation
        canonical_map = {
            # JavaScript frameworks (with and without dots)
            'node.js': 'nodejs',
            'nodejs': 'nodejs',
            'node': 'nodejs',  # Common shorthand
            'react.js': 'react',
            'reactjs': 'react',
            'react': 'react',
            'vue.js': 'vue',
            'vuejs': 'vue',
            'vue': 'vue',
            'angular.js': 'angular',
            'angularjs': 'angular',
            'angular': 'angular',
            'chart.js': 'chartjs',
            'chartjs': 'chartjs',
            'd3.js': 'd3',
            'd3': 'd3',
            'three.js': 'threejs',
            'threejs': 'threejs',
            # .NET variations
            '.net': 'dotnet',
            'dotnet': 'dotnet',
            'asp.net': 'aspnet',
            'aspnet': 'aspnet',
            'asp net': 'aspnet',
            # Common variations
            'js': 'javascript',
            'javascript': 'javascript',
            'ts': 'typescript',
            'typescript': 'typescript',
            'py': 'python',
            'python': 'python',
            'sql': 'sql',
            'nosql': 'nosql',
            'html5': 'html',
            'html': 'html',
            'css3': 'css',
            'css': 'css',
            # Cloud/AWS
            'aws': 'amazon web services',
            'gcp': 'google cloud platform',
            'azure': 'microsoft azure',
            # Databases
            'postgresql': 'postgres',
            'postgres': 'postgres',
            'mongodb': 'mongo',
            'mongo': 'mongo',
            'mysql': 'mysql',
            # Tools
            'git': 'git',
            'github': 'git',
            'gitlab': 'git',
        }
        
        # Check canonical map first (handles variations with punctuation)
        if skill_lower in canonical_map:
            return canonical_map[skill_lower]
        
        # FIX: Strip ALL punctuation (periods, hyphens, etc.) for consistent matching
        # This ensures "Node.js" -> "nodejs", "React.js" -> "reactjs", etc.
        # Remove all punctuation except alphanumeric and spaces
        skill_clean = re.sub(r'[^\w\s]', '', skill_lower)
        
        # Collapse multiple spaces to single space
        skill_clean = re.sub(r'\s+', ' ', skill_clean).strip()
        
        # Check canonical map again with cleaned version
        if skill_clean in canonical_map:
            return canonical_map[skill_clean]
        
        # Remove common prefixes/suffixes that don't affect skill identity
        skill_clean = re.sub(r'^(the|a|an)\s+', '', skill_clean)
        skill_clean = re.sub(r'\s+(framework|library|tool|technology|platform)$', '', skill_clean)
        
        return skill_clean.strip()
    
    def extract_skills_from_text(self, text: str) -> List[str]:
        """Extract and normalize skills from text content using taxonomy variations.
        Matches any known variation as a whole token (punctuation-aware) and returns
        canonical skill tokens.
        """
        if not text or not isinstance(text, str):
            return []
        # Normalize unicode (handles fancy bullets/dashes and half-width variants)
        text_normalized = unicodedata.normalize('NFKC', text)
        # Replace common bullet/dash characters with spaces for cleaner boundaries
        replacements = {
            ord(ch): ' '
            for ch in [
                '\u2022',  # •
                '\u25CF',  # ●
                '\u25AA',  # ▪
                '\u25AB',  # ▫
                '\u25E6',  # ◦
                '\u2219',  # ∙
                '\u00B7',  # ·
                '\u2023',  # ‣
                '\u2043',  # ⁃
                '\u2212',  # −
                '\u2013',  # –
                '\u2014',  # —
                '\u2015',  # ―
                '\u2010',  # ‐
                '\u204C',  # ⁌
                '\u204D',  # ⁍
            ]
        }
        text_normalized = text_normalized.translate(replacements)
        # Remove zero-width and non-breaking spaces that break regex boundaries
        text_normalized = text_normalized.replace('\u200b', ' ').replace('\xa0', ' ')
        # Fix common "chart. js" -> "chart.js" style spacing
        text_normalized = re.sub(r'\b([a-z0-9]+)\.\s+(js)\b', r'\1.\2', text_normalized, flags=re.IGNORECASE)
        # Collapse multiple whitespace
        text_normalized = re.sub(r'\s+', ' ', text_normalized)
        text_lower = text_normalized.lower()
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
    
    def extract_must_have_skills_from_jd(self, jd_text: str, responsibilities: str = "", max_skills: int = 8) -> List[str]:
        """Intelligently extract truly must-have skills from JD text.
        
        Unlike extract_skills_from_text() which extracts ALL mentioned skills,
        this method uses contextual signals to identify only required skills:
        - Skills near "required", "must have", "essential", "mandatory"
        - Skills mentioned multiple times (high frequency = importance)
        - Skills in requirements/qualifications sections
        - Filters out skills near "preferred", "nice to have", "bonus", "plus"
        
        Args:
            jd_text: Full job description text
            responsibilities: Additional responsibilities text
            max_skills: Maximum number of skills to return (default 8)
            
        Returns:
            List of canonical skill names that are truly required
        """
        if not jd_text and not responsibilities:
            return []
        
        # Combine all JD content
        full_text = f"{jd_text}\n{responsibilities}".lower()
        
        # Define context patterns for must-have vs nice-to-have
        must_have_patterns = [
            r'required[:\s]',
            r'must[\s-]?have',
            r'essential',
            r'mandatory',
            r'requirements?[:\s]',
            r'qualifications?[:\s]',
            r'minimum[\s]requirements?',
            r'key[\s]requirements?',
            r'you[\s]must[\s]have',
            r'we[\s]require',
            r'strong[\s](?:knowledge|experience|skills?)',
            r'proven[\s](?:experience|track)',
            r'proficiency[\s]in',
            r'expert[\s]in',
            r'hands[\s-]?on[\s]experience',
        ]
        
        nice_to_have_patterns = [
            r'nice[\s-]?to[\s-]?have',
            r'preferred',
            r'bonus',
            r'plus',
            r'advantageous',
            r'desirable',
            r'ideally',
            r'a[\s]plus',
            r'would[\s]be[\s](?:nice|great|good)',
            r'familiarity[\s]with',  # Often indicates nice-to-have
            r'exposure[\s]to',
        ]
        
        # Extract all skills first
        all_skills = self.extract_skills_from_text(full_text)
        
        if not all_skills:
            return []
        
        # Score each skill based on context
        skill_scores: Dict[str, float] = {}
        
        for skill in all_skills:
            score = 0.0
            
            # Find all occurrences of the skill
            skill_pattern = r'\b' + re.escape(skill) + r'\b'
            try:
                occurrences = list(re.finditer(skill_pattern, full_text, re.IGNORECASE))
            except Exception:
                occurrences = []
            
            if not occurrences:
                # Skill matched via variation, give base score
                score = 0.3
            else:
                # Frequency bonus (more mentions = more important)
                score += min(len(occurrences) * 0.15, 0.6)
                
                # Check context around each occurrence
                for match in occurrences:
                    start = max(0, match.start() - 100)
                    end = min(len(full_text), match.end() + 50)
                    context = full_text[start:end]
                    
                    # Boost for must-have context
                    for pattern in must_have_patterns:
                        if re.search(pattern, context):
                            score += 0.25
                            break
                    
                    # Penalty for nice-to-have context
                    for pattern in nice_to_have_patterns:
                        if re.search(pattern, context):
                            score -= 0.4
                            break
            
            # Check if skill appears in first half of text (usually requirements)
            first_half = full_text[:len(full_text)//2]
            if skill.lower() in first_half:
                score += 0.15
            
            skill_scores[skill] = score
        
        # Filter to only positive-scoring skills
        must_have_skills = [skill for skill, score in skill_scores.items() if score > 0.3]
        
        # Sort by score and limit
        must_have_skills.sort(key=lambda s: skill_scores.get(s, 0), reverse=True)
        result = must_have_skills[:max_skills]
        
        if result:
            logger.info(f"[SKILL_EXTRACT] Auto-extracted {len(result)} must-have skills from JD: {result}")
        
        return result
    
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
    
    def get_matched_nice_to_have_skills(self, resume_text: str, nice_to_have_skills: List[str]) -> List[str]:
        """Get canonical list of nice-to-have skills that are matched in resume text.
        
        THESIS-READY FIX #1: Enhanced normalization ensures variations like "Node.js" 
        match "nodejs" in the config.
        
        FIX: Added super-normalized matching to guarantee matches even with punctuation variations.
        """
        if not resume_text or not nice_to_have_skills:
            return []
        
        # Extract skills from resume text using enhanced normalization
        resume_skills = self.extract_skills_from_text(resume_text)
        
        # Normalize nice-to-have skills with enhanced normalization
        normalized_nice_to_have = [self.normalize_skill(skill) for skill in nice_to_have_skills]
        
        # Find matches using taxonomy extraction
        matched = set(resume_skills) & set(normalized_nice_to_have)
        
        # FIX: Force matching with super-normalized text (removes all non-alphanumeric)
        # This guarantees "Node.js" matches "nodejs" and "C++.net" matches "cppnet"
        if len(matched) == 0:
            # Create super-normalized version of resume text (alphanumeric only)
            clean_text = re.sub(r'[^a-z0-9]', '', resume_text.lower())
            
            # Check each nice-to-have skill in super-normalized form
            for skill in nice_to_have_skills:
                normalized_skill = self.normalize_skill(skill)
                # Super-normalize the skill (remove all non-alphanumeric)
                clean_skill = re.sub(r'[^a-z0-9]', '', normalized_skill.lower())
                
                # Check if super-normalized skill exists in super-normalized text
                if clean_skill and clean_skill in clean_text:
                    matched.add(normalized_skill)
                    logger.debug(f"[NICE_TO_HAVE] Super-normalized match: '{skill}' -> '{normalized_skill}' (clean: '{clean_skill}')")
        
        return sorted(list(matched))
    
    def get_matched_required_skills_with_confidence(
        self, 
        resume_text: str, 
        required_skills: List[str], 
        min_confidence: float = 0.7
    ) -> List[Tuple[str, float]]:
        """Get matched required skills with confidence scores.
        
        This method normalizes required skills and checks if they appear in the resume text,
        returning matches with their confidence scores. Useful for weighted coverage calculations.
        
        Args:
            resume_text: Text content from resume (all sections combined)
            required_skills: List of required skills from job description
            min_confidence: Minimum confidence threshold for matches (0.0-1.0)
        
        Returns:
            List of (matched_skill, confidence) tuples, sorted by confidence descending.
            Confidence scores:
            - 1.0 for exact matches (skill found in taxonomy)
            - 0.6-0.9 for fuzzy matches (semantic similarity, if enabled)
            - 0.0 for no matches
        """
        if not resume_text or not required_skills:
            return []
        
        # Extract skills from resume text
        resume_skills = self.extract_skills_from_text(resume_text)
        resume_skills_set = set(resume_skills)
        
        # Normalize required skills with confidence
        matched_with_confidence = []
        for req_skill in required_skills:
            if not req_skill or not isinstance(req_skill, str):
                continue
                
            canonical, confidence = self.normalize_skill_with_confidence(req_skill)
            
            # Check if this canonical skill is in resume
            if canonical in resume_skills_set and confidence >= min_confidence:
                matched_with_confidence.append((canonical, confidence))
        
        # Sort by confidence descending
        matched_with_confidence.sort(key=lambda x: x[1], reverse=True)
        return matched_with_confidence


