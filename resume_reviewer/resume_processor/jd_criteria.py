import re
from typing import Any, Dict, List


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

    position_title = str(post_like.get("position_title", "")).strip()
    experience_min_years = _safe_int(post_like.get("experience_min_years", 0), 0)
    education_requirements = str(post_like.get("education_requirements", "")).strip()
    must_have_skills = canonicalize_list(_split_comma_list(post_like.get("must_have_skills", "")))
    nice_to_have_skills = canonicalize_list(_split_comma_list(post_like.get("nice_to_have_skills", "")))
    industry_experience = canonicalize_list(_split_comma_list(post_like.get("industry_experience", "")))
    responsibilities_raw = str(post_like.get("responsibilities", "")).strip()
    responsibilities = [ln.strip() for ln in responsibilities_raw.splitlines() if ln.strip()]
    responsibilities = canonicalize_list(responsibilities)
    keywords = canonicalize_list(_split_comma_list(post_like.get("keywords", "")))
    location = str(post_like.get("location", "")).strip()
    seniority_level = str(post_like.get("seniority_level", "")).strip().lower()
    if seniority_level not in SENIORITY_CHOICES:
        seniority_level = ""
    certifications = canonicalize_list(_split_comma_list(post_like.get("certifications", "")))
    languages = canonicalize_list(_split_comma_list(post_like.get("languages", "")))

    return {
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


