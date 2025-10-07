#!/usr/bin/env python3
"""
Test script for the deterministic ranking engine.
"""

import json
import sys
from pathlib import Path

# Add the resume_processor directory to Python path
current_dir = Path(__file__).parent
resume_processor_dir = current_dir / 'resume_processor'
sys.path.insert(0, str(resume_processor_dir))

from deterministic_ranker import DeterministicRanker, process_ranking_payload

def create_test_payload():
    """Create a test payload for the ranking engine."""
    return {
        "job_profile": {
            "title": "Senior Software Engineer",
            "must_have_skills": ["python", "react", "sql"],
            "nice_to_have_skills": ["aws", "docker", "kubernetes"]
        },
        "weights": {
            "section_tfidf": 0.4,
            "skill_tfidf": 0.3,
            "sbert": 0.2,
            "consistency_bonus": 0.1
        },
        "penalties": {
            "missing_must_have": 0.1,
            "low_evidence": 0.05
        },
        "hard_rules": {
            "disqualify_if_missing_must_have": False
        },
        "batch_stats": {
            "section_tfidf_min": 0.1,
            "section_tfidf_max": 0.9,
            "skill_tfidf_min": 0.0,
            "skill_tfidf_max": 1.0,
            "sbert_min": 0.2,
            "sbert_max": 0.8
        },
        "candidates": [
            {
                "candidate_id": "candidate_001",
                "raw_scores": {
                    "section_tfidf_overall": 0.8,
                    "section_tfidf_by_section": {
                        "experience": 0.85,
                        "skills": 0.75,
                        "education": 0.65,
                        "projects": 0.70
                    },
                    "skill_tfidf_overall": 0.9,
                    "skill_tfidf_by_cluster": {
                        "python": 0.95,
                        "react": 0.85,
                        "sql": 0.90,
                        "aws": 0.60
                    },
                    "sbert_overall": 0.75
                },
                "meta": {
                    "missing_must_have_count": 0,
                    "low_evidence": False,
                    "years_experience": 5,
                    "seniority_hint": "senior"
                }
            },
            {
                "candidate_id": "candidate_002",
                "raw_scores": {
                    "section_tfidf_overall": 0.6,
                    "section_tfidf_by_section": {
                        "experience": 0.70,
                        "skills": 0.50,
                        "education": 0.60,
                        "projects": 0.55
                    },
                    "skill_tfidf_overall": 0.7,
                    "skill_tfidf_by_cluster": {
                        "python": 0.80,
                        "react": 0.60,
                        "sql": 0.70,
                        "docker": 0.75
                    },
                    "sbert_overall": 0.65
                },
                "meta": {
                    "missing_must_have_count": 1,
                    "low_evidence": True,
                    "years_experience": 3,
                    "seniority_hint": "mid"
                }
            },
            {
                "candidate_id": "candidate_003",
                "raw_scores": {
                    "section_tfidf_overall": 0.4,
                    "section_tfidf_by_section": {
                        "experience": 0.45,
                        "skills": 0.35,
                        "education": 0.40,
                        "projects": 0.30
                    },
                    "skill_tfidf_overall": 0.5,
                    "skill_tfidf_by_cluster": {
                        "python": 0.60,
                        "javascript": 0.40,
                        "sql": 0.50
                    },
                    "sbert_overall": 0.3
                },
                "meta": {
                    "missing_must_have_count": 2,
                    "low_evidence": False,
                    "years_experience": 1,
                    "seniority_hint": "junior"
                }
            }
        ]
    }

def test_deterministic_ranking():
    """Test the deterministic ranking engine."""
    print("Testing Deterministic Ranking Engine")
    print("=" * 50)
    
    # Create test payload
    test_payload = create_test_payload()
    print(f"Created test payload with {len(test_payload['candidates'])} candidates")
    
    # Test the ranking engine
    try:
        ranker = DeterministicRanker()
        result = ranker.rank_candidates(test_payload)
        
        print("\nRanking Results:")
        print(f"Job Title: {result['job_title']}")
        print(f"Number of candidates: {len(result['ranked_candidates'])}")
        
        print("\nRanked Candidates:")
        for i, candidate in enumerate(result['ranked_candidates']):
            print(f"{i+1}. {candidate['candidate_id']}")
            print(f"   Composite Score: {candidate['composite_score']}")
            print(f"   Normalized Scores: {candidate['normalized']}")
            print(f"   Penalties: {candidate['penalties_applied']}")
            print(f"   Disqualified: {candidate['disqualified']}")
            print(f"   Rationale: {candidate['rationale']}")
            print()
        
        print("Batch Notes:")
        print(f"Weights Used: {result['batch_notes']['weights_used']}")
        print(f"Hard Rules: {result['batch_notes']['hard_rules']}")
        print(f"Normalization: {result['batch_notes']['normalization_used']}")
        
        return True
        
    except Exception as e:
        print(f"Error in ranking test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_json_processing():
    """Test JSON payload processing."""
    print("\nTesting JSON Payload Processing")
    print("=" * 50)
    
    try:
        # Create test payload and convert to JSON
        test_payload = create_test_payload()
        payload_json = json.dumps(test_payload, indent=2)
        
        print(f"Payload JSON size: {len(payload_json)} characters")
        
        # Process through JSON interface
        result_json = process_ranking_payload(payload_json)
        result = json.loads(result_json)
        
        print("JSON Processing successful!")
        print(f"Result contains {len(result['ranked_candidates'])} ranked candidates")
        
        # Verify schema compliance
        required_keys = ["job_title", "ranked_candidates", "batch_notes"]
        for key in required_keys:
            if key not in result:
                print(f"ERROR: Missing required key: {key}")
                return False
        
        print("Schema validation passed!")
        return True
        
    except Exception as e:
        print(f"Error in JSON processing test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("Deterministic Ranking Engine Test Suite")
    print("=" * 60)
    
    tests = [
        ("Deterministic Ranking", test_deterministic_ranking),
        ("JSON Processing", test_json_processing)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        if test_func():
            passed += 1
            print(f"✅ {test_name}: PASSED")
        else:
            print(f"❌ {test_name}: FAILED")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests passed! Deterministic ranking engine is ready.")
        print("\nAPI Endpoint: POST /resume/api/deterministic-ranking/")
        print("Content-Type: application/json")
    else:
        print("Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()
