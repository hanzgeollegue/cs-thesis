import re
import logging
from typing import Any, Dict, List

from .config import DEBUG_AUDIT

logger = logging.getLogger(__name__)

SENIORITY_CHOICES = [
    "intern",
    "junior",
    "mid",
    "senior",
    "lead",
    "staff",
]


def canonicalize_list(seq: List[str]) -> List[str]:
    """Lower, trim, dedupe, stable-sort for consistent downstream use."""
    if not isinstance(seq, list):
        return []
    cleaned = []
    seen = set()
    for item in seq:
        if not isinstance(item, str):
            continue
        val = item.strip().lower()
        if not val:
            continue
        if val not in seen:
            seen.add(val)
            cleaned.append(val)
    return sorted(cleaned)
def _apply_jd_hygiene(criteria: Dict[str, Any]) -> Dict[str, Any]:
    """Clean JD skills: lowercase+NFKC, alias map, expand broad terms, dedupe, disjoint sets."""
    import unicodedata
    
    try:
        from .text_processor import SkillTaxonomy
        taxonomy = SkillTaxonomy()
    except Exception:
        taxonomy = None

    def _norm_token(tok: str) -> str:
        """Normalize skill token with enhanced punctuation/spacing cleanup."""
        if not tok:
            return ""
        
        # Basic cleanup
        t = unicodedata.normalize('NFKC', str(tok).strip())
        
        # Collapse spacing and punctuation variants
        t = _normalize_skill_punctuation(t)
        
        # Apply taxonomy normalization if available
        if taxonomy:
            try:
                t = taxonomy.normalize_skill(t)
            except Exception:
                pass
        
        return t.lower() if t else ""

    def _normalize_skill_punctuation(skill: str) -> str:
        """Normalize punctuation and spacing in skill names."""
        if not skill:
            return ""
        
        # Remove extra spaces around punctuation
        skill = re.sub(r'\s*([.,;:!?])\s*', r'\1', skill)
        
        # Collapse multiple spaces
        skill = re.sub(r'\s+', ' ', skill)
        
        # Handle common variants
        skill = skill.replace(' .', '.').replace('. ', '.')
        skill = skill.replace(' -', '-').replace('- ', '-')
        skill = skill.replace(' /', '/').replace('/ ', '/')
        
        return skill.strip()

    # Small expansion map for broad terms
    BROAD_EXPANSIONS = {
        'api': ['openapi', 'swagger', 'rest'],
    }

    def _expand_and_clean(skills: List[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for s in skills or []:
            tok = _norm_token(str(s))
            logger.debug(f"[HYGIENE] Input skill: '{s}' -> normalized: '{tok}'")
            if not tok:
                logger.warning(f"[HYGIENE] Skill '{s}' was normalized to empty string - LOST!")
                continue
            expanded = BROAD_EXPANSIONS.get(tok, [tok])
            for e in expanded:
                e2 = _norm_token(e)
                if e2 and e2 not in seen:
                    seen.add(e2)
                    out.append(e2)
        logger.debug(f"[HYGIENE] Final skills after expand_and_clean: {out}")
        return sorted(out)

    # Store original for audit logging
    original_must_have = criteria.get('must_have_skills', [])
    original_nice_to_have = criteria.get('nice_to_have_skills', [])

    must_have = _expand_and_clean(original_must_have)
    nice_to_have = _expand_and_clean(original_nice_to_have)

    # Ensure disjoint: remove any overlaps from nice_to_have
    must_set = set(must_have)
    nice_clean = [s for s in nice_to_have if s not in must_set]
    
    # Log overlap removal
    overlap = set(nice_to_have) & must_set
    if overlap:
        logger.warning(f"Removed {len(overlap)} overlapping skills from nice-to-have: {overlap}")

    # Audit logging
    if DEBUG_AUDIT:
        logger.info(f"[AUDIT] JD Skills Hygiene:")
        logger.info(f"  Must-have before: {len(original_must_have)} skills")
        logger.info(f"  Must-have after: {len(must_have)} skills")
        logger.info(f"  Nice-to-have before: {len(original_nice_to_have)} skills") 
        logger.info(f"  Nice-to-have after: {len(nice_clean)} skills")
        logger.info(f"  Overlaps removed: {len(overlap)}")
        if overlap:
            logger.info(f"  Overlap details: {list(overlap)}")

    criteria['must_have_skills'] = must_have
    criteria['nice_to_have_skills'] = sorted(nice_clean)
    
    # Validate disjoint sets
    final_must_set = set(must_have)
    final_nice_set = set(nice_clean)
    if final_must_set & final_nice_set:
        logger.error(f"CRITICAL: Skills still overlap after hygiene: {final_must_set & final_nice_set}")
        # Force remove overlaps from nice-to-have
        criteria['nice_to_have_skills'] = [s for s in nice_clean if s not in final_must_set]
    
    return criteria



def _split_comma_list(value: str) -> List[str]:
    if not isinstance(value, str):
        return []
    parts = [p.strip() for p in value.split(',')]
    return [p for p in parts if p]


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        s = str(value).strip()
        if not s:
            return default
        return int(re.sub(r"[^0-9-]", "", s) or default)
    except Exception:
        return default


def parse_criteria_from_post(post_like: Dict[str, Any]) -> Dict[str, Any]:
    """Parse and normalize JD criteria from POST data (dict-like)."""
    if not isinstance(post_like, dict):
        return {}

    # DEBUG: Log raw input
    must_have_raw = post_like.get("must_have_skills", "")
    nice_to_have_raw = post_like.get("nice_to_have_skills", "")
    logger.info(f"[JD_PARSE] Raw must_have_skills input: '{must_have_raw[:200] if must_have_raw else 'EMPTY'}'...")
    logger.info(f"[JD_PARSE] Raw nice_to_have_skills input: '{nice_to_have_raw[:200] if nice_to_have_raw else 'EMPTY'}'...")
    
    # Check if this looks like natural language input (sentences in skill fields)
    is_natural_language = _detect_natural_language_input(must_have_raw)
    
    if is_natural_language:
        logger.info("[JD_PARSE] Detected natural language JD input, using enhanced processing")
        result = _parse_criteria_enhanced(post_like)
    else:
        logger.info("[JD_PARSE] Detected traditional JD input, using standard processing")
        result = _parse_criteria_standard(post_like)
    
    # DEBUG: Log output
    logger.info(f"[JD_PARSE] Output must_have_skills: {result.get('must_have_skills', [])}")
    logger.info(f"[JD_PARSE] Output nice_to_have_skills: {result.get('nice_to_have_skills', [])}")
    
    return result


def _detect_natural_language_input(must_have_skills: Any) -> bool:
    """Detect if the input contains natural language sentences instead of clean skill tokens."""
    if not must_have_skills:
        return False
    
    # If it's a list, check if any items look like sentences
    if isinstance(must_have_skills, list):
        for item in must_have_skills:
            if isinstance(item, str) and _looks_like_sentence(item):
                return True
        return False
    
    # If it's a string, check if it contains sentence-like patterns
    if isinstance(must_have_skills, str):
        return _looks_like_sentence(must_have_skills)
    
    return False


def _looks_like_sentence(text: str) -> bool:
    """Check if text looks like a sentence rather than a skill token.
    
    IMPORTANT: Be conservative - only flag as sentence if we're VERY confident.
    Short skill phrases like "system monitoring" or "project management" should NOT
    be flagged as sentences.
    """
    if not text or len(text) < 20:  # Increased threshold - short phrases are likely skills
        return False
    
    # Check for STRONG sentence indicators (removed overly common words like 'and', 'or', 'for')
    sentence_indicators = [
        'we are', 'we\'re', 'looking for', 'should be', 'must have',
        'experience with', 'knowledge of', 'proficiency in', 'familiar with',
        'comfortable with', 'working with', 'you will', 'the candidate',
        'responsible for', 'ability to', 'understanding of',
        'essential', 'required', 'preferred'
    ]
    
    text_lower = text.lower()
    for indicator in sentence_indicators:
        if indicator in text_lower:
            logger.debug(f"[SENTENCE_DETECT] '{text[:50]}...' flagged as sentence (indicator: '{indicator}')")
            return True
    
    # Only flag as sentence if it's LONG (>6 words) AND has sentence punctuation
    words = text.split()
    if len(words) > 6 and text.rstrip()[-1] in '.!?':
        logger.debug(f"[SENTENCE_DETECT] '{text[:50]}...' flagged as sentence (long with punctuation)")
        return True
    
    return False


def _parse_criteria_enhanced(post_like: Dict[str, Any]) -> Dict[str, Any]:
    """Parse criteria using enhanced processing for natural language inputs."""
    try:
        from .enhanced_jd_processor import process_jd_criteria_enhanced
        
        # Convert to the format expected by enhanced processor
        raw_criteria = {
            "position_title": str(post_like.get("position_title", "")).strip(),
            "experience_min_years": _safe_int(post_like.get("experience_min_years", 0), 0),
            "education_requirements": str(post_like.get("education_requirements", "")).strip(),
            "must_have_skills": _extract_list_from_input(post_like.get("must_have_skills", "")),
            "nice_to_have_skills": _extract_list_from_input(post_like.get("nice_to_have_skills", "")),
            "industry_experience": _extract_list_from_input(post_like.get("industry_experience", "")),
            # Responsibilities is descriptive text - prefer newline splitting to preserve sentences
            "responsibilities": _extract_list_from_input(post_like.get("responsibilities", ""), prefer_newlines=True),
            "keywords": _extract_list_from_input(post_like.get("keywords", "")),
            "location": str(post_like.get("location", "")).strip(),
            "seniority_level": str(post_like.get("seniority_level", "")).strip().lower(),
            "certifications": _extract_list_from_input(post_like.get("certifications", "")),
            "languages": _extract_list_from_input(post_like.get("languages", ""))
        }
        
        # Validate seniority level
        if raw_criteria["seniority_level"] not in SENIORITY_CHOICES:
            raw_criteria["seniority_level"] = ""
        
        # Process with enhanced processor
        processed_criteria = process_jd_criteria_enhanced(raw_criteria)
        
        # Remove debug fields for backward compatibility
        processed_criteria.pop('must_have_skills_debug', None)
        processed_criteria.pop('nice_to_have_skills_debug', None)
        processed_criteria.pop('industry_experience_debug', None)
        
        # Validate no overlap between required and nice-to-have skills
        must_set = set(processed_criteria.get('must_have_skills', []))
        nice_set = set(processed_criteria.get('nice_to_have_skills', []))
        overlap = must_set & nice_set
        if overlap:
            logger.error(f"Skills cannot be both required and nice-to-have: {overlap}")
            # Remove overlaps from nice-to-have (required takes priority)
            processed_criteria['nice_to_have_skills'] = [s for s in processed_criteria.get('nice_to_have_skills', []) if s not in overlap]
            logger.info(f"Removed overlapping skills from nice-to-have: {overlap}")
        
        return processed_criteria
        
    except Exception as e:
        logger.error(f"Error in enhanced parsing, falling back to standard: {e}")
        return _parse_criteria_standard(post_like)


def _parse_criteria_standard(post_like: Dict[str, Any]) -> Dict[str, Any]:
    """Parse criteria using standard processing for traditional inputs."""
    position_title = str(post_like.get("position_title", "")).strip()
    experience_min_years = _safe_int(post_like.get("experience_min_years", 0), 0)
    education_requirements = str(post_like.get("education_requirements", "")).strip()
    must_have_skills = canonicalize_list(_split_comma_list(post_like.get("must_have_skills", "")))
    nice_to_have_skills = canonicalize_list(_split_comma_list(post_like.get("nice_to_have_skills", "")))
    industry_experience = canonicalize_list(_split_comma_list(post_like.get("industry_experience", "")))
    # Responsibilities should preserve original case and order (not canonicalized like skills)
    responsibilities_raw = str(post_like.get("responsibilities", "")).strip()
    responsibilities = [ln.strip() for ln in responsibilities_raw.splitlines() if ln.strip()]
    # Don't canonicalize responsibilities - they're descriptive text, not skill tokens
    keywords = canonicalize_list(_split_comma_list(post_like.get("keywords", "")))
    location = str(post_like.get("location", "")).strip()
    seniority_level = str(post_like.get("seniority_level", "")).strip().lower()
    if seniority_level not in SENIORITY_CHOICES:
        seniority_level = ""
    certifications = canonicalize_list(_split_comma_list(post_like.get("certifications", "")))
    languages = canonicalize_list(_split_comma_list(post_like.get("languages", "")))

    criteria = {
        "position_title": position_title,
        "experience_min_years": experience_min_years,
        "education_requirements": education_requirements,
        "must_have_skills": must_have_skills,
        "nice_to_have_skills": nice_to_have_skills,
        "industry_experience": industry_experience,
        "responsibilities": responsibilities,
        "keywords": keywords,
        "location": location,
        "seniority_level": seniority_level,
        "certifications": certifications,
        "languages": languages,
    }
    
    # Generate JD summary for NLG compatibility
    criteria["jd_summary"] = _generate_jd_summary(criteria)
    
    # Apply JD hygiene: canonicalize, expand broad terms, dedupe, disjoint sets
    try:
        criteria = _apply_jd_hygiene(criteria)
    except Exception as e:
        logger.warning(f"JD hygiene failed: {e}")
    
    # Validate no overlap between required and nice-to-have skills
    must_set = set(criteria.get('must_have_skills', []))
    nice_set = set(criteria.get('nice_to_have_skills', []))
    overlap = must_set & nice_set
    if overlap:
        logger.error(f"Skills cannot be both required and nice-to-have: {overlap}")
        # Remove overlaps from nice-to-have (required takes priority)
        criteria['nice_to_have_skills'] = [s for s in criteria.get('nice_to_have_skills', []) if s not in overlap]
        logger.info(f"Removed overlapping skills from nice-to-have: {overlap}")
    
    # Generate JD hash for cache tracking
    jd_hash = _generate_jd_hash(criteria)
    criteria['jd_hash'] = jd_hash
    
    if DEBUG_AUDIT:
        logger.info(f"[AUDIT] JD Hash: {jd_hash}")
    
    return criteria


def _extract_list_from_input(input_value: Any, prefer_newlines: bool = False) -> List[str]:
    """Extract a list from various input formats.
    
    Args:
        input_value: The input to parse (string, list, or other)
        prefer_newlines: If True, prefer newline splitting over comma splitting.
                        Use for descriptive text like responsibilities.
    """
    if not input_value:
        return []
    
    if isinstance(input_value, list):
        return [str(item).strip() for item in input_value if str(item).strip()]
    
    if isinstance(input_value, str):
        # For descriptive text (responsibilities), prefer newline splitting
        if prefer_newlines:
            if '\n' in input_value:
                return [item.strip() for item in input_value.split('\n') if item.strip()]
            # Fallback to comma only if no newlines
            elif ',' in input_value:
                return [item.strip() for item in input_value.split(',') if item.strip()]
            else:
                return [input_value.strip()] if input_value.strip() else []
        else:
            # For skill-like inputs, prefer comma splitting
            if ',' in input_value:
                return [item.strip() for item in input_value.split(',') if item.strip()]
            elif '\n' in input_value:
                return [item.strip() for item in input_value.split('\n') if item.strip()]
            else:
                return [input_value.strip()] if input_value.strip() else []
    
    return [str(input_value).strip()] if str(input_value).strip() else []


def _generate_jd_summary(criteria: Dict[str, Any]) -> str:
    """
    Generate a natural language JD summary from structured criteria.
    This is a shared utility for both standard and enhanced JD processing.
    
    Args:
        criteria: JD criteria dictionary
        
    Returns:
        Natural language JD summary suitable for NLG and scoring
    """
    try:
        summary_parts = []
        
        # Job title and seniority
        title = criteria.get('position_title', '').strip()
        seniority = criteria.get('seniority_level', '').strip()
        if title:
            if seniority:
                summary_parts.append(f"We are seeking a {seniority} {title}.")
            else:
                summary_parts.append(f"We are seeking a {title}.")
        
        # Experience requirements
        exp_years = criteria.get('experience_min_years', 0)
        if exp_years > 0:
            summary_parts.append(f"The role requires {exp_years}+ years of experience.")
        
        # Education
        education = criteria.get('education_requirements', '').strip()
        if education:
            summary_parts.append(f"Education requirements: {education}")
        
        # Location
        location = criteria.get('location', '').strip()
        if location:
            summary_parts.append(f"Location: {location}")
        
        # Collect all text content for skills and responsibilities
        all_text_content = []
        
        # Add skill content
        must_have = criteria.get('must_have_skills', [])
        if must_have:
            all_text_content.extend([str(item) for item in must_have])
        
        nice_to_have = criteria.get('nice_to_have_skills', [])
        if nice_to_have:
            all_text_content.extend([str(item) for item in nice_to_have])
        
        industry = criteria.get('industry_experience', [])
        if industry:
            all_text_content.extend([str(item) for item in industry])
        
        # Add responsibilities (preserve as separate sentences)
        responsibilities = criteria.get('responsibilities', [])
        if responsibilities:
            # Responsibilities are descriptive text - join them properly as sentences
            responsibilities_text = '. '.join(str(item).strip() for item in responsibilities if str(item).strip())
            # Ensure each responsibility ends with a period
            if responsibilities_text and not responsibilities_text.endswith('.'):
                responsibilities_text += '.'
            if responsibilities_text:
                summary_parts.append(f"Key responsibilities: {responsibilities_text}")
        
        # Add keywords
        keywords = criteria.get('keywords', [])
        if keywords:
            all_text_content.extend([str(item) for item in keywords])
        
        # Combine remaining text content (skills, keywords)
        if all_text_content:
            combined_text = ' '.join(all_text_content)
            # Clean up the text
            combined_text = re.sub(r'\s+', ' ', combined_text).strip()
            summary_parts.append(combined_text)
        
        # Join all parts
        jd_summary = ' '.join(summary_parts)
        
        # Clean up
        jd_summary = re.sub(r'\s+', ' ', jd_summary).strip()
        
        return jd_summary
        
    except Exception as e:
        logger.error(f"Error creating JD summary: {e}")
        return str(criteria.get('position_title', ''))


def _generate_jd_hash(criteria: Dict[str, Any]) -> str:
    """Generate a hash of the JD criteria for cache tracking."""
    import hashlib
    
    # Create a deterministic string from key criteria
    key_parts = [
        str(sorted(criteria.get('must_have_skills', []))),
        str(sorted(criteria.get('nice_to_have_skills', []))),
        str(criteria.get('experience_min_years', 0))
    ]
    key_string = '|'.join(key_parts)
    
    # Generate hash
    return hashlib.md5(key_string.encode('utf-8')).hexdigest()[:12]


def build_jd_text(criteria: Dict[str, Any]) -> str:
    """Synthesize a compact JD text from structured criteria for downstream scoring.
    No PII is introduced; outputs a plain text JD summary.
    """
    if not isinstance(criteria, dict):
        return ""
    lines: List[str] = []
    title = criteria.get("position_title")
    if title:
        lines.append(f"Job Title: {title}")
    sen = criteria.get("seniority_level")
    if sen:
        lines.append(f"Seniority: {sen}")
    yrs = criteria.get("experience_min_years")
    try:
        if isinstance(yrs, int) and yrs > 0:
            lines.append(f"Experience: {yrs}+ years")
    except Exception:
        pass
    edu = criteria.get("education_requirements")
    if edu:
        lines.append(f"Education: {edu}")
    loc = criteria.get("location")
    if loc:
        lines.append(f"Location: {loc}")
    def _fmt_list(key: str, label: str):
        arr = criteria.get(key) or []
        if isinstance(arr, list) and arr:
            lines.append(f"{label}: " + ", ".join(arr))
    _fmt_list("must_have_skills", "Must-Have Skills")
    _fmt_list("nice_to_have_skills", "Nice-To-Have Skills")
    _fmt_list("industry_experience", "Industry Experience")
    _fmt_list("certifications", "Certifications")
    _fmt_list("languages", "Languages")
    # Responsibilities as bullets (compact)
    resp = criteria.get("responsibilities") or []
    if isinstance(resp, list) and resp:
        lines.append("Responsibilities:")
        for r in resp[:10]:
            lines.append(f"- {r}")
    # Keywords
    _fmt_list("keywords", "Keywords")
    return "\n".join(lines).strip()


