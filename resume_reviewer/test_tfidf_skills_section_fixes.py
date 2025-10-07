#!/usr/bin/env python3
"""
Validate Section TF-IDF and Skill TF-IDF after JD normalization + taxonomy fixes.

Run in venv311 from thesis root:
  .\\venv311\\Scripts\\Activate.ps1
  cd .\\resume_reviewer
  python -u test_tfidf_skills_section_fixes.py
"""

from typing import List, Dict, Any

from resume_processor.text_processor import (
    SectionAwareTFIDF,
    normalize_job_description,
)


def make_resume(sections: Dict[str, str], matched: List[Dict[str, str]] | None = None) -> Dict[str, Any]:
    return {
        "sections": sections,
        "matched_skills": matched or [],
    }


def pretty(scores: List[float]) -> str:
    return ", ".join(f"{s:.3f}" for s in scores)


def test_overlap_vs_no_overlap():
    print("\n=== Test: Overlap vs No Overlap ===")
    jd = (
        "Software Engineer Position. We are looking for a skilled software engineer with "
        "experience in: Python, React, JavaScript, SQL, AWS, Docker."
    )

    resumes = [
        make_resume(
            {
                "experience": "Built React + Node apps on AWS with Docker; Python automation; SQL reports.",
                "skills": "React, JavaScript, AWS, Docker, Python, SQL",
                "education": "BS CS",
                "misc": "",
            }
        ),
        make_resume(
            {
                "experience": "Mobile Swift iOS only. No web or cloud mentioned.",
                "skills": "Swift, UIKit",
                "education": "BS IT",
                "misc": "",
            }
        ),
    ]

    proc = SectionAwareTFIDF()
    section_scores, skill_scores = proc.compute_section_tfidf_scores(resumes, jd)

    print("Section TF-IDF:", pretty(section_scores))
    print("Skill   TF-IDF:", pretty(skill_scores))

    # Expect first resume > second for both channels if normalization had contrast
    assert len(section_scores) == 2 and len(skill_scores) == 2
    assert section_scores[0] >= section_scores[1], "Expected overlap resume to score >= no-overlap (section)"
    assert skill_scores[0] >= skill_scores[1], "Expected overlap resume to score >= no-overlap (skills)"


def test_backfill_when_matched_skills_empty():
    print("\n=== Test: Backfill Skills When matched_skills Empty ===")
    jd = (
        "We need experience in React, JavaScript, AWS, Docker, and Python."
    )

    resumes = [
        # matched_skills empty; should backfill from section text via taxonomy
        make_resume(
            {
                "experience": "React and JavaScript frontend; AWS + Docker deployments; Python ETL.",
                "skills": "",
                "education": "",
                "misc": "",
            },
            matched=[],
        ),
        # No overlap case
        make_resume(
            {
                "experience": "C++ Qt desktop only.",
                "skills": "C++, Qt",
                "education": "",
                "misc": "",
            },
            matched=[],
        ),
    ]

    proc = SectionAwareTFIDF()
    _, skill_scores = proc.compute_section_tfidf_scores(resumes, jd)
    print("Skill TF-IDF (backfill):", pretty(skill_scores))

    assert len(skill_scores) == 2
    assert skill_scores[0] >= skill_scores[1], "Backfilled resume should score >= non-overlap"


def test_jd_skills_from_experience_fallback():
    print("\n=== Test: JD Skills From Experience Fallback ===")
    # No explicit 'skills' wording; skills appear under prose/experience phrasing
    jd = (
        "We are looking for someone with experience in Python, Django, Kubernetes, AWS, and SQL."
    )

    resumes = [
        make_resume(
            {
                "experience": "Developed APIs with Django and deployed on AWS EKS (Kubernetes).",
                "skills": "Django, Python, AWS, Kubernetes",
                "education": "",
                "misc": "",
            }
        ),
        make_resume(
            {
                "experience": "Ruby on Rails and Heroku only.",
                "skills": "Ruby, Rails",
                "education": "",
                "misc": "",
            }
        ),
    ]

    proc = SectionAwareTFIDF()
    section_scores, skill_scores = proc.compute_section_tfidf_scores(resumes, jd)

    print("Section TF-IDF (fallback):", pretty(section_scores))
    print("Skill   TF-IDF (fallback):", pretty(skill_scores))

    assert skill_scores[0] >= skill_scores[1], "JD skills fallback should enable non-zero contrast"


if __name__ == "__main__":
    print("Running TF-IDF skills/sections normalization tests...")
    test_overlap_vs_no_overlap()
    test_backfill_when_matched_skills_empty()
    test_jd_skills_from_experience_fallback()
    print("All checks executed.")


