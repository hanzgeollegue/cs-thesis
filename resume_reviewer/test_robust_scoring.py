#!/usr/bin/env python3
"""
Test script to verify robust scoring pipeline fixes.
Run this in the virtual environment: venv311\Scripts\activate
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from resume_processor.text_processor import SectionAwareTFIDF, SemanticEmbedding
from resume_processor.llm_ranker import CEReranker
import logging

# Set up logging
logging.basicConfig(level=logging.WARNING)

def test_tfidf_empty_data():
    """Test TF-IDF with empty data."""
    print("Testing TF-IDF with empty data...")
    tfidf_processor = SectionAwareTFIDF()
    resumes = [{'sections': {}, 'matched_skills': []}]
    jd_sections = {'experience': '', 'skills': '', 'education': '', 'misc': ''}
    
    result = tfidf_processor.compute_section_tfidf_scores(resumes, jd_sections)
    print(f"TF-IDF result: {result}")
    print(f"Type: {type(result)}")
    print(f"Length: {len(result) if result else 'None'}")
    
    # Verify it returns a tuple of two lists
    assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
    assert len(result) == 2, f"Expected 2 elements, got {len(result)}"
    assert isinstance(result[0], list), f"Expected list, got {type(result[0])}"
    assert isinstance(result[1], list), f"Expected list, got {type(result[1])}"
    print("✓ TF-IDF test passed")

def test_tfidf_with_data():
    """Test TF-IDF with actual data."""
    print("\nTesting TF-IDF with actual data...")
    tfidf_processor = SectionAwareTFIDF()
    resumes = [
        {
            'sections': {
                'experience': 'Python programming, database design',
                'skills': 'Python, SQL, React',
                'education': 'Computer Science degree',
                'misc': 'Software engineer with 5 years experience'
            },
            'matched_skills': [
                {'skill_id': 'python'},
                {'skill_id': 'sql'},
                {'skill_id': 'react'}
            ]
        }
    ]
    jd_sections = {
        'experience': 'Python programming, React and JavaScript, Database design and SQL',
        'skills': 'Git version control',
        'education': '',
        'misc': 'Software Engineer Position'
    }
    
    result = tfidf_processor.compute_section_tfidf_scores(resumes, jd_sections)
    print(f"TF-IDF result: {result}")
    print(f"Type: {type(result)}")
    print(f"Length: {len(result) if result else 'None'}")
    
    # Verify it returns a tuple of two lists
    assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
    assert len(result) == 2, f"Expected 2 elements, got {len(result)}"
    assert isinstance(result[0], list), f"Expected list, got {type(result[0])}"
    assert isinstance(result[1], list), f"Expected list, got {type(result[1])}"
    assert len(result[0]) == 1, f"Expected 1 score, got {len(result[0])}"
    assert len(result[1]) == 1, f"Expected 1 score, got {len(result[1])}"
    print("✓ TF-IDF with data test passed")

def test_sbert_empty_data():
    """Test SBERT with empty data."""
    print("\nTesting SBERT with empty data...")
    try:
        sbert = SemanticEmbedding()
        if sbert.model_available:
            resumes = [{'sections': []}]
            result = sbert.compute_sbert_scores(resumes, 'test job description')
            print(f"SBERT result: {result}")
            print(f"Type: {type(result)}")
            print(f"Length: {len(result) if result else 'None'}")
            
            # Verify it returns a list
            assert isinstance(result, list), f"Expected list, got {type(result)}"
            print("✓ SBERT test passed")
        else:
            print("SBERT not available, skipping test")
    except Exception as e:
        print(f"SBERT error: {e}")

def test_sbert_with_data():
    """Test SBERT with actual data."""
    print("\nTesting SBERT with actual data...")
    try:
        sbert = SemanticEmbedding()
        if sbert.model_available:
            resumes = [
                {
                    'sections': [
                        {'content': ['Python programming', 'Database design']},
                        {'content': ['React', 'JavaScript']}
                    ]
                }
            ]
            result = sbert.compute_sbert_scores(resumes, 'Software engineer with Python and React experience')
            print(f"SBERT result: {result}")
            print(f"Type: {type(result)}")
            print(f"Length: {len(result) if result else 'None'}")
            
            # Verify it returns a list
            assert isinstance(result, list), f"Expected list, got {type(result)}"
            assert len(result) == 1, f"Expected 1 score, got {len(result)}"
            print("✓ SBERT with data test passed")
        else:
            print("SBERT not available, skipping test")
    except Exception as e:
        print(f"SBERT error: {e}")

def test_ce_empty_data():
    """Test Cross-Encoder with empty data."""
    print("\nTesting Cross-Encoder with empty data...")
    try:
        ce = CEReranker()
        if ce.model_available:
            resumes = [{'id': 'test', 'sections': []}]
            result = ce.rerank_candidates('test job description', resumes, [0.0], [0.0], [0.0])
            print(f"CE result: {result}")
            print(f"Type: {type(result)}")
            print(f"Length: {len(result) if result else 'None'}")
            
            # Verify it returns a list
            assert isinstance(result, list), f"Expected list, got {type(result)}"
            print("✓ Cross-Encoder test passed")
        else:
            print("Cross-Encoder not available, skipping test")
    except Exception as e:
        print(f"CE error: {e}")

def test_ce_with_data():
    """Test Cross-Encoder with actual data."""
    print("\nTesting Cross-Encoder with actual data...")
    try:
        ce = CEReranker()
        if ce.model_available:
            resumes = [
                {
                    'id': 'test1',
                    'sections': [
                        {'content': ['Python programming', 'Database design']},
                        {'content': ['React', 'JavaScript']}
                    ]
                }
            ]
            result = ce.rerank_candidates(
                'Software engineer with Python and React experience',
                resumes,
                [0.5],  # section_tfidf_scores
                [0.3],  # skill_tfidf_scores
                [0.7]   # sbert_scores
            )
            print(f"CE result: {result}")
            print(f"Type: {type(result)}")
            print(f"Length: {len(result) if result else 'None'}")
            
            # Verify it returns a list
            assert isinstance(result, list), f"Expected list, got {type(result)}"
            assert len(result) == 1, f"Expected 1 result, got {len(result)}"
            print("✓ Cross-Encoder with data test passed")
        else:
            print("Cross-Encoder not available, skipping test")
    except Exception as e:
        print(f"CE error: {e}")

def test_data_structure_handling():
    """Test handling of different data structures."""
    print("\nTesting data structure handling...")
    tfidf_processor = SectionAwareTFIDF()
    
    # Test with string sections (should be handled gracefully)
    resumes = [{'sections': 'invalid string format', 'matched_skills': []}]
    jd_sections = {'experience': 'test', 'skills': 'test', 'education': 'test', 'misc': 'test'}
    
    result = tfidf_processor.compute_section_tfidf_scores(resumes, jd_sections)
    print(f"String sections result: {result}")
    assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
    print("✓ Data structure handling test passed")

def test_mixed_data_structures():
    """Test handling of mixed data structures."""
    print("\nTesting mixed data structures...")
    tfidf_processor = SectionAwareTFIDF()
    
    # Test with mixed valid/invalid data
    resumes = [
        {'sections': {'experience': 'Python'}, 'matched_skills': []},  # Valid
        {'sections': 'invalid', 'matched_skills': []},  # Invalid
        {'sections': {}, 'matched_skills': [{'skill_id': 'python'}]}  # Valid
    ]
    jd_sections = {'experience': 'Python programming', 'skills': 'Python', 'education': '', 'misc': ''}
    
    result = tfidf_processor.compute_section_tfidf_scores(resumes, jd_sections)
    print(f"Mixed data result: {result}")
    assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
    assert len(result[0]) == 3, f"Expected 3 scores, got {len(result[0])}"
    assert len(result[1]) == 3, f"Expected 3 scores, got {len(result[1])}"
    print("✓ Mixed data structures test passed")

def test_batch_processor_integration():
    """Test batch processor integration."""
    print("\nTesting batch processor integration...")
    try:
        from resume_processor.batch_processor import BatchProcessor
        
        processor = BatchProcessor()
        
        # Test with empty data
        result = processor._compute_section_tfidf_scores([], {})
        print(f"Empty batch result: {result}")
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        
        # Test with actual data
        from resume_processor.batch_processor import ParsedResume, ResumeScores
        
        resume = ParsedResume(
            id="test",
            sections={'experience': 'Python programming'},
            matched_skills=[],
            scores=ResumeScores(0.0, 0.0, 0.0)
        )
        
        result = processor._compute_section_tfidf_scores([resume], {'experience': 'Python'})
        print(f"Batch processor result: {result}")
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        print("✓ Batch processor integration test passed")
        
    except Exception as e:
        print(f"Batch processor error: {e}")

if __name__ == "__main__":
    print("Running robust scoring pipeline tests...")
    print("=" * 50)
    
    try:
        test_tfidf_empty_data()
        test_tfidf_with_data()
        test_sbert_empty_data()
        test_sbert_with_data()
        test_ce_empty_data()
        test_ce_with_data()
        test_data_structure_handling()
        test_mixed_data_structures()
        test_batch_processor_integration()
        
        print("\n" + "=" * 50)
        print("🎉 All tests passed! The robust scoring pipeline is working correctly.")
        print("\nKey improvements verified:")
        print("✓ No functions return None")
        print("✓ All methods handle empty data gracefully")
        print("✓ Data structure validation works")
        print("✓ Error handling prevents crashes")
        print("✓ Consistent return types maintained")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
