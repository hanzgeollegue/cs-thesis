"""
Unit tests for the deterministic NLG system.
Tests determinism, edge cases, and template logic.
"""

import unittest
import sys
import os

# Add the resume_processor module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'resume_processor'))

from nlg_generator import (
    CandidateAnalyzer, 
    PairwiseComparator, 
    FactExtractor,
    generate_candidate_analysis,
    generate_candidate_facts,
    generate_pairwise_comparison
)


class TestNLGSystem(unittest.TestCase):
    """Test the deterministic NLG system."""
    
    def setUp(self):
        """Set up test data."""
        self.sample_jd_criteria = {
            'position_title': 'Software Engineer',
            'experience_min_years': 3,
            'must_have_skills': ['python', 'javascript', 'react'],
            'nice_to_have_skills': ['docker', 'aws'],
            'jd_summary': 'We are seeking a Software Engineer with 3+ years experience in python, javascript, react.'
        }
        
        self.sample_candidate_excellent = {
            'id': 'candidate_1',
            'scores': {
                'final_score': 0.85,
                'coverage': 0.9,
                'has_match_skills': True,
                'has_match_experience': True,
                'matched_required_skills': ['python', 'javascript', 'react'],
                'matched_nice_skills': ['docker'],
                'missing_skills': []
            },
            'parsed': {
                'experience': [
                    {'title': 'Senior Developer', 'company': 'Tech Corp', 'bullets': ['Built React apps', 'Python backend']}
                ],
                'education': [{'degree': 'Computer Science', 'school': 'University'}],
                'projects': [{'name': 'Web App', 'technologies': ['React', 'Python']}]
            },
            'meta': {'source_file': 'excellent_candidate.pdf'}
        }
        
        self.sample_candidate_weak = {
            'id': 'candidate_2',
            'scores': {
                'final_score': 0.25,
                'coverage': 0.3,
                'has_match_skills': False,
                'has_match_experience': False,
                'matched_required_skills': ['python'],
                'matched_nice_skills': [],
                'missing_skills': ['javascript', 'react']
            },
            'parsed': {
                'experience': [],
                'education': [{'degree': 'Business', 'school': 'College'}],
                'projects': []
            },
            'meta': {'source_file': 'weak_candidate.pdf'}
        }
    
    def test_determinism(self):
        """Test that the same input always produces the same output."""
        analyzer = CandidateAnalyzer()
        
        # Generate analysis multiple times
        analysis1 = analyzer.generate_analysis(self.sample_candidate_excellent, self.sample_jd_criteria)
        analysis2 = analyzer.generate_analysis(self.sample_candidate_excellent, self.sample_jd_criteria)
        analysis3 = analyzer.generate_analysis(self.sample_candidate_excellent, self.sample_jd_criteria)
        
        # Should be identical
        self.assertEqual(analysis1, analysis2)
        self.assertEqual(analysis2, analysis3)
        self.assertIsInstance(analysis1, str)
        self.assertGreater(len(analysis1), 10)  # Should be substantial text
    
    def test_score_tier_logic(self):
        """Test that different score tiers produce different analysis styles."""
        analyzer = CandidateAnalyzer()
        
        # Test excellent tier (85%)
        excellent_analysis = analyzer.generate_analysis(self.sample_candidate_excellent, self.sample_jd_criteria)
        self.assertIn('Strong candidate', excellent_analysis)
        
        # Test weak tier (25%)
        weak_analysis = analyzer.generate_analysis(self.sample_candidate_weak, self.sample_jd_criteria)
        self.assertIn('Limited match', weak_analysis)
        
        # Should be different
        self.assertNotEqual(excellent_analysis, weak_analysis)
    
    def test_fact_extraction(self):
        """Test that facts are extracted correctly."""
        extractor = FactExtractor()
        facts = extractor.extract_facts(self.sample_candidate_excellent, self.sample_jd_criteria)
        
        # Check structure
        self.assertIn('scores', facts)
        self.assertIn('skills', facts)
        self.assertIn('experience', facts)
        self.assertIn('education', facts)
        self.assertIn('projects', facts)
        self.assertIn('metadata', facts)
        
        # Check specific values
        self.assertEqual(facts['scores']['final_score_percentage'], 85.0)
        self.assertEqual(len(facts['skills']['matched_required']), 3)
        self.assertEqual(facts['experience']['count'], 1)
        self.assertEqual(facts['education']['count'], 1)
    
    def test_pairwise_comparison(self):
        """Test pairwise comparison generation."""
        comparator = PairwiseComparator()
        comparison = comparator.compare(
            self.sample_candidate_excellent, 
            self.sample_candidate_weak, 
            self.sample_jd_criteria
        )
        
        # Check structure
        self.assertIsInstance(comparison.candidate_a_id, str)
        self.assertIsInstance(comparison.candidate_b_id, str)
        self.assertIn(comparison.winner, ['a', 'b', None])
        self.assertIsInstance(comparison.margin, float)
        self.assertIsInstance(comparison.comparison_text, str)
        self.assertIsInstance(comparison.key_differences, list)
        
        # Should identify winner correctly
        self.assertEqual(comparison.winner, 'a')  # Excellent candidate should win
        self.assertGreater(comparison.margin, 0)
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        analyzer = CandidateAnalyzer()
        extractor = FactExtractor()
        
        # Test with empty candidate data
        empty_candidate = {
            'id': 'empty',
            'scores': {'final_score': 0.0, 'coverage': 0.0, 'has_match_skills': False, 'has_match_experience': False, 'matched_required_skills': [], 'matched_nice_skills': [], 'missing_skills': []},
            'parsed': {},
            'meta': {}
        }
        
        # Should not crash
        analysis = analyzer.generate_analysis(empty_candidate, self.sample_jd_criteria)
        self.assertIsInstance(analysis, str)
        self.assertGreater(len(analysis), 0)
        
        facts = extractor.extract_facts(empty_candidate, self.sample_jd_criteria)
        self.assertIsInstance(facts, dict)
        self.assertIn('scores', facts)
    
    def test_convenience_functions(self):
        """Test the convenience functions work correctly."""
        # Test generate_candidate_analysis
        analysis = generate_candidate_analysis(self.sample_candidate_excellent, self.sample_jd_criteria)
        self.assertIsInstance(analysis, str)
        self.assertGreater(len(analysis), 10)
        
        # Test generate_candidate_facts
        facts = generate_candidate_facts(self.sample_candidate_excellent, self.sample_jd_criteria)
        self.assertIsInstance(facts, dict)
        self.assertIn('scores', facts)
        
        # Test generate_pairwise_comparison
        comparison = generate_pairwise_comparison(
            self.sample_candidate_excellent, 
            self.sample_candidate_weak, 
            self.sample_jd_criteria
        )
        self.assertIsInstance(comparison.comparison_text, str)
        self.assertGreater(len(comparison.comparison_text), 10)
    
    def test_template_consistency(self):
        """Test that templates produce consistent, readable output."""
        analyzer = CandidateAnalyzer()
        
        # Test that analysis contains expected elements
        analysis = analyzer.generate_analysis(self.sample_candidate_excellent, self.sample_jd_criteria)
        
        # Should contain score information
        self.assertIn('85%', analysis)  # Score percentage
        self.assertIn('3/3', analysis)  # Skills match ratio
        
        # Should contain skill information
        self.assertIn('python', analysis.lower())
        self.assertIn('javascript', analysis.lower())
        
        # Should be properly formatted (end with period)
        self.assertTrue(analysis.endswith('.'))
        
        # Should not contain placeholder text
        self.assertNotIn('{', analysis)
        self.assertNotIn('}', analysis)
    
    def test_enhanced_nlg_contract(self):
        """Test contract compatibility with enhanced NLG system."""
        try:
            from nlg_generator_enhanced import generate_candidate_analysis_enhanced
            
            # Test enhanced version
            result = generate_candidate_analysis_enhanced(
                self.sample_candidate, self.sample_jd_criteria, self.sample_batch_stats
            )
            
            # Should have enhanced structure
            self.assertIn('text', result)
            self.assertIn('bullets', result)
            self.assertIn('facts', result)
            self.assertIn('metadata', result)
            
            # Text should be compatible with original
            self.assertIsInstance(result['text'], str)
            self.assertGreater(len(result['text']), 10)
            
            # Bullets should be a list
            self.assertIsInstance(result['bullets'], list)
            self.assertGreater(len(result['bullets']), 0)
            
            # Facts should be a dict
            self.assertIsInstance(result['facts'], dict)
            
            # Metadata should have required fields
            metadata = result['metadata']
            self.assertIn('version', metadata)
            self.assertIn('sentences', metadata)
            
        except ImportError:
            # Enhanced version not available, skip test
            self.skipTest("Enhanced NLG system not available")
    
    def test_backward_compatibility(self):
        """Test that original API still works."""
        # Test original function still works
        analysis = generate_candidate_analysis(self.sample_candidate, self.sample_jd_criteria)
        self.assertIsInstance(analysis, str)
        self.assertGreater(len(analysis), 10)
        
        # Test facts extraction still works
        facts = generate_candidate_facts(self.sample_candidate, self.sample_jd_criteria)
        self.assertIsInstance(facts, dict)
        self.assertIn('scores', facts)
        self.assertIn('skills', facts)
    
    def test_enhanced_facts_contract(self):
        """Test that enhanced facts maintain compatibility."""
        try:
            from nlg_generator_enhanced import generate_candidate_facts_enhanced
            
            # Test enhanced facts
            enhanced_facts = generate_candidate_facts_enhanced(
                self.sample_candidate, self.sample_jd_criteria, self.sample_batch_stats
            )
            
            # Should have all original fields
            self.assertIn('scores', enhanced_facts)
            self.assertIn('skills', enhanced_facts)
            self.assertIn('experience', enhanced_facts)
            
            # Should have enhanced fields
            self.assertIn('seniority', enhanced_facts)
            self.assertIn('progression', enhanced_facts)
            self.assertIn('specializations', enhanced_facts)
            
            # Batch position should be present
            if self.sample_batch_stats:
                self.assertIn('batch_position', enhanced_facts)
                batch_pos = enhanced_facts['batch_position']
                self.assertIn('percentile', batch_pos)
                self.assertIn('gap_from_top', batch_pos)
            
        except ImportError:
            # Enhanced version not available, skip test
            self.skipTest("Enhanced NLG system not available")


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
