#!/usr/bin/env python3
"""Simple test for JD normalization."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from resume_processor.text_processor import normalize_job_description
    
    jd = """
    Software Engineer Position
    
    We are looking for a skilled software engineer with experience in:
    - Python programming
    - React and JavaScript
    - Database design and SQL
    - Cloud platforms (AWS preferred)
    - Git version control
    """
    
    print("Testing JD normalization...")
    result = normalize_job_description(jd)
    
    print("Job Title:", result['job_title'])
    print("Skills:", result['skills_required'])
    print("Experience:", result['experience_required'])
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()