#!/usr/bin/env python3
"""
Test script to verify skills extraction in JD normalization.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from resume_processor.text_processor import normalize_job_description

def test_skills_extraction():
    """Test skills extraction from various JD formats."""
    
    # Test JD with explicit skills section
    jd1 = """
    Software Engineer Position
    
    We are looking for a skilled software engineer with experience in:
    - Python programming
    - React and JavaScript
    - Database design and SQL
    - Cloud platforms (AWS preferred)
    - Git version control
    
    Requirements:
    - Bachelor's degree in Computer Science or related field
    - 3+ years of software development experience
    - Experience with modern web technologies
    - Strong problem-solving skills
    - Team collaboration experience
    """
    
    # Test JD with skills in different format
    jd2 = """
    Senior Developer
    
    Required Skills:
    - Java, Spring Boot
    - Microservices architecture
    - Docker and Kubernetes
    - PostgreSQL, Redis
    - RESTful API design
    
    Technical Qualifications:
    - 5+ years Java development
    - Experience with cloud platforms
    - Knowledge of CI/CD pipelines
    """
    
    # Test JD with skills embedded in experience section
    jd3 = """
    Full Stack Developer
    
    We need someone with experience in:
    - Frontend: React, Vue.js, TypeScript
    - Backend: Node.js, Python, Django
    - Databases: MongoDB, MySQL
    - DevOps: Docker, AWS, Jenkins
    
    Must have proficiency in:
    - Agile methodologies
    - Code review processes
    - Testing frameworks
    """
    
    test_cases = [
        ("Explicit skills section", jd1),
        ("Skills in requirements", jd2),
        ("Skills in experience", jd3)
    ]
    
    print("Testing JD Skills Extraction")
    print("=" * 50)
    
    for name, jd in test_cases:
        print(f"\n{name}:")
        print("-" * 30)
        
        normalized = normalize_job_description(jd)
        
        print(f"Job Title: {normalized['job_title']}")
        print(f"Experience: {normalized['experience_required'][:100]}...")
        print(f"Skills: {normalized['skills_required']}")
        print(f"Education: {normalized['education'][:100]}...")
        print(f"Misc: {normalized['company_description'][:100]}...")
        
        # Check if skills were extracted
        skills = normalized['skills_required']
        if skills and len(skills.strip()) > 10:
            print("✅ Skills extracted successfully")
        else:
            print("❌ Skills extraction failed or empty")
        
        print()

if __name__ == "__main__":
    try:
        print("Starting skills extraction test...")
        test_skills_extraction()
        print("Test completed successfully!")
    except Exception as e:
        print(f"Error running test: {e}")
        import traceback
        traceback.print_exc()
