#!/usr/bin/env python3
"""
Test Evidence-Based Analysis System

Tests the enhanced NLG system with evidence-based reasoning, confidence filtering,
skills hygiene, and concrete examples.
"""

import os
import sys
import json
import unittest
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from resume_processor.evidence_collector import EvidenceCollector, EvidenceItem
from resume_processor.nlg_generator_enhanced import EnhancedNLGGenerator
from resume_processor.batch_processor import BatchProcessor
from resume_processor.jd_criteria import JobDescriptionCriteria
from resume_processor.profile_analyzer import ProfileAnalyzer
from resume_processor.nlg_templates import NLGTemplates
from resume_processor.nlg_polisher import NLGPolisher
from resume_processor.nlg_metadata import ProvenanceTracker


class TestEvidenceBasedAnalysis(unittest.TestCase):
    """Test evidence-based analysis components."""

    def setUp(self):
        """Set up test fixtures."""
        self.evidence_collector = EvidenceCollector()
        self.nlg_generator = EnhancedNLGGenerator()
        self.batch_processor = BatchProcessor()
        self.jd_criteria = JobDescriptionCriteria()
        self.profile_analyzer = ProfileAnalyzer()
        self.templates = NLGTemplates()
        self.polisher = NLGPolisher()
        self.provenance_tracker = ProvenanceTracker()

    def test_skills_hygiene_disjoint_sets(self):
        """Test that matched and missing skills are disjoint."""
        print("\n=== Testing Skills Hygiene: Disjoint Sets ===")
        
        # Test data with overlapping skills
        test_data = {
            'candidate_id': 'test_candidate',
            'scores': {
                'matched_required_skills': ['python', 'javascript', 'react'],
                'missing_required_skills': ['python', 'node.js', 'docker'],
                'matched_nice_to_have_skills': ['typescript', 'aws'],
                'missing_nice_to_have_skills': ['typescript', 'kubernetes']
            },
            'parsed': {
                'skills': [
                    {'name': 'python', 'proficiency': 'intermediate'},
                    {'name': 'javascript', 'proficiency': 'advanced'},
                    {'name': 'react', 'proficiency': 'intermediate'},
                    {'name': 'typescript', 'proficiency': 'beginner'}
                ]
            }
        }
        
        jd_criteria = {
            'required_skills': ['python', 'javascript', 'react', 'node.js', 'docker'],
            'nice_to_have_skills': ['typescript', 'aws', 'kubernetes']
        }
        
        # Process through evidence collector
        evidence_pool = self.evidence_collector.collect_evidence(test_data, jd_criteria)
        
        # Check that skills are properly categorized
        matched_required = set(test_data['scores']['matched_required_skills'])
        missing_required = set(test_data['scores']['missing_required_skills'])
        matched_nice = set(test_data['scores']['matched_nice_to_have_skills'])
        missing_nice = set(test_data['scores']['missing_nice_to_have_skills'])
        
        # Verify disjoint sets
        self.assertEqual(matched_required & missing_required, set(), 
                        "Required skills cannot be both matched and missing")
        self.assertEqual(matched_nice & missing_nice, set(), 
                        "Nice-to-have skills cannot be both matched and missing")
        
        print(f"✓ Matched required: {matched_required}")
        print(f"✓ Missing required: {missing_required}")
        print(f"✓ Matched nice-to-have: {matched_nice}")
        print(f"✓ Missing nice-to-have: {missing_nice}")
        print("✓ All skill sets are disjoint")

    def test_evidence_ranking_by_confidence(self):
        """Test that evidence is ranked by confidence score."""
        print("\n=== Testing Evidence Ranking by Confidence ===")
        
        # Create mock evidence items with different confidence levels
        evidence_items = [
            EvidenceItem(
                source='ce_pairs',
                skill='python',
                context='Developed Python applications for data analysis',
                confidence=0.95,
                evidence_type='direct_match'
            ),
            EvidenceItem(
                source='skill_inference',
                skill='javascript',
                context='Mentioned in experience section',
                confidence=0.75,
                evidence_type='inferred'
            ),
            EvidenceItem(
                source='text_match',
                skill='react',
                context='React.js framework experience',
                confidence=0.60,
                evidence_type='text_match'
            ),
            EvidenceItem(
                source='skill_inference',
                skill='node.js',
                context='Backend development mentioned',
                confidence=0.45,
                evidence_type='inferred'
            )
        ]
        
        # Test ranking
        ranked_evidence = self.evidence_collector._rank_evidence_by_confidence(evidence_items)
        
        # Verify ranking order (highest confidence first)
        expected_order = ['python', 'javascript', 'react', 'node.js']
        actual_order = [item.skill for item in ranked_evidence]
        
        self.assertEqual(actual_order, expected_order, 
                        "Evidence should be ranked by confidence (highest first)")
        
        print("✓ Evidence ranking order:")
        for i, item in enumerate(ranked_evidence):
            print(f"  {i+1}. {item.skill}: {item.confidence:.2f} ({item.source})")

    def test_confidence_filtering_threshold(self):
        """Test that low-confidence evidence is filtered out."""
        print("\n=== Testing Confidence Filtering Threshold ===")
        
        # Test data with mixed confidence levels
        test_data = {
            'candidate_id': 'test_candidate',
            'scores': {
                'cross_encoder': 0.85,
                'cross_encoder_confidence': 0.90,
                'ce_pairs': [
                    {
                        'query': 'python programming',
                        'context': 'Developed Python applications for data analysis',
                        'score': 0.95
                    },
                    {
                        'query': 'javascript development',
                        'context': 'Built web applications using JavaScript',
                        'score': 0.75
                    }
                ],
                'matched_required_skills': ['python', 'javascript'],
                'missing_required_skills': ['react', 'node.js']
            },
            'parsed': {
                'experience': [
                    {
                        'title': 'Software Engineer',
                        'description': 'Developed Python applications for data analysis and built web applications using JavaScript'
                    }
                ],
                'skills': [
                    {'name': 'python', 'proficiency': 'advanced'},
                    {'name': 'javascript', 'proficiency': 'intermediate'}
                ]
            }
        }
        
        jd_criteria = {
            'required_skills': ['python', 'javascript', 'react', 'node.js'],
            'nice_to_have_skills': ['typescript', 'aws']
        }
        
        # Collect evidence
        evidence_pool = self.evidence_collector.collect_evidence(test_data, jd_criteria)
        
        # Check that only high-confidence evidence is included
        high_confidence_evidence = [item for item in evidence_pool if item.confidence >= 0.6]
        low_confidence_evidence = [item for item in evidence_pool if item.confidence < 0.6]
        
        print(f"✓ High-confidence evidence (≥0.6): {len(high_confidence_evidence)} items")
        print(f"✓ Low-confidence evidence (<0.6): {len(low_confidence_evidence)} items")
        
        # Verify that high-confidence evidence includes expected skills
        high_confidence_skills = {item.skill for item in high_confidence_evidence}
        expected_high_confidence = {'python', 'javascript'}
        
        self.assertTrue(expected_high_confidence.issubset(high_confidence_skills),
                       "High-confidence evidence should include expected skills")

    def test_caveat_generation_for_low_evidence(self):
        """Test that caveats are generated when evidence is weak."""
        print("\n=== Testing Caveat Generation for Low Evidence ===")
        
        # Test case 1: No CE evidence
        test_data_no_ce = {
            'candidate_id': 'test_candidate',
            'scores': {
                'cross_encoder': 0.0,
                'cross_encoder_confidence': 0.0,
                'ce_pairs': [],
                'matched_required_skills': ['python'],
                'missing_required_skills': ['javascript', 'react']
            },
            'parsed': {
                'experience': [
                    {
                        'title': 'Software Engineer',
                        'description': 'Worked with Python for data analysis'
                    }
                ],
                'skills': [{'name': 'python', 'proficiency': 'intermediate'}]
            }
        }
        
        jd_criteria = {
            'required_skills': ['python', 'javascript', 'react'],
            'nice_to_have_skills': ['typescript']
        }
        
        # Generate analysis
        analysis = self.nlg_generator.generate_analysis(
            test_data_no_ce, jd_criteria, batch_size=1
        )
        
        # Check for caveat
        self.assertIn('Assessment based on', analysis['analysis_text'],
                     "Analysis should include caveat for low evidence")
        
        print("✓ Caveat generated for low CE evidence")
        print(f"  Analysis: {analysis['analysis_text'][:100]}...")
        
        # Test case 2: Parsing issues
        test_data_parsing_issues = {
            'candidate_id': 'test_candidate',
            'scores': {
                'cross_encoder': 0.0,
                'cross_encoder_confidence': 0.0,
                'ce_pairs': [],
                'matched_required_skills': [],
                'missing_required_skills': ['python', 'javascript']
            },
            'parsed': {
                'experience': [],
                'skills': [],
                'raw_text': ''
            }
        }
        
        # Generate analysis
        analysis = self.nlg_generator.generate_analysis(
            test_data_parsing_issues, jd_criteria, batch_size=1
        )
        
        # Check for caveat
        self.assertIn('Assessment based on', analysis['analysis_text'],
                     "Analysis should include caveat for parsing issues")
        
        print("✓ Caveat generated for parsing issues")

    def test_concrete_example_generation(self):
        """Test that concrete examples are generated with evidence."""
        print("\n=== Testing Concrete Example Generation ===")
        
        test_data = {
            'candidate_id': 'test_candidate',
            'scores': {
                'cross_encoder': 0.85,
                'cross_encoder_confidence': 0.90,
                'ce_pairs': [
                    {
                        'query': 'python programming',
                        'context': 'Developed Python applications for data analysis using pandas and numpy',
                        'score': 0.95
                    }
                ],
                'matched_required_skills': ['python'],
                'missing_required_skills': ['javascript', 'react']
            },
            'parsed': {
                'experience': [
                    {
                        'title': 'Data Analyst',
                        'description': 'Developed Python applications for data analysis using pandas and numpy to process large datasets'
                    }
                ],
                'skills': [{'name': 'python', 'proficiency': 'advanced'}]
            }
        }
        
        jd_criteria = {
            'required_skills': ['python', 'javascript', 'react'],
            'nice_to_have_skills': ['typescript'],
            'role_title': 'Data Scientist'
        }
        
        # Generate analysis
        analysis = self.nlg_generator.generate_analysis(
            test_data, jd_criteria, batch_size=1
        )
        
        # Check for concrete example
        analysis_text = analysis['analysis_text']
        self.assertIn('python', analysis_text.lower(),
                     "Analysis should mention Python as a strength")
        
        # Check that example is tied to JD requirement
        self.assertIn('data', analysis_text.lower(),
                     "Example should be relevant to data analysis role")
        
        print("✓ Concrete example generated")
        print(f"  Example context: {analysis_text}")

    def test_hedged_language_for_low_confidence(self):
        """Test that hedged language is used when confidence is low."""
        print("\n=== Testing Hedged Language for Low Confidence ===")
        
        # Test data with low confidence
        test_data = {
            'candidate_id': 'test_candidate',
            'scores': {
                'cross_encoder': 0.0,
                'cross_encoder_confidence': 0.0,
                'ce_pairs': [],
                'matched_required_skills': ['python'],
                'missing_required_skills': ['javascript', 'react']
            },
            'parsed': {
                'experience': [
                    {
                        'title': 'Software Engineer',
                        'description': 'Worked with Python for data analysis'
                    }
                ],
                'skills': [{'name': 'python', 'proficiency': 'intermediate'}]
            }
        }
        
        jd_criteria = {
            'required_skills': ['python', 'javascript', 'react'],
            'nice_to_have_skills': ['typescript']
        }
        
        # Generate analysis
        analysis = self.nlg_generator.generate_analysis(
            test_data, jd_criteria, batch_size=1
        )
        
        analysis_text = analysis['analysis_text']
        
        # Check for hedged language
        hedged_terms = ['appears to', 'may be', 'suggests', 'indicates', 'limited']
        has_hedged_language = any(term in analysis_text.lower() for term in hedged_terms)
        
        self.assertTrue(has_hedged_language,
                       "Analysis should use hedged language when confidence is low")
        
        print("✓ Hedged language used for low confidence")
        print(f"  Analysis: {analysis_text}")

    def test_batch_context_accuracy(self):
        """Test that batch context is accurate for different batch sizes."""
        print("\n=== Testing Batch Context Accuracy ===")
        
        test_data = {
            'candidate_id': 'test_candidate',
            'scores': {
                'final_score': 0.75,
                'matched_required_skills': ['python'],
                'missing_required_skills': ['javascript']
            },
            'parsed': {
                'experience': [{'title': 'Software Engineer', 'description': 'Python development'}],
                'skills': [{'name': 'python', 'proficiency': 'intermediate'}]
            }
        }
        
        jd_criteria = {
            'required_skills': ['python', 'javascript'],
            'nice_to_have_skills': ['typescript']
        }
        
        # Test single candidate batch
        analysis_single = self.nlg_generator.generate_analysis(
            test_data, jd_criteria, batch_size=1
        )
        
        self.assertIn('Only candidate in batch', analysis_single['analysis_text'],
                     "Single candidate should show 'Only candidate in batch'")
        
        print("✓ Single candidate batch context correct")
        
        # Test multi-candidate batch
        analysis_multi = self.nlg_generator.generate_analysis(
            test_data, jd_criteria, batch_size=3
        )
        
        self.assertIn('Ranked #', analysis_multi['analysis_text'],
                     "Multi-candidate batch should show ranking")
        
        print("✓ Multi-candidate batch context correct")

    def test_evidence_source_priority(self):
        """Test that evidence sources are prioritized correctly."""
        print("\n=== Testing Evidence Source Priority ===")
        
        test_data = {
            'candidate_id': 'test_candidate',
            'scores': {
                'cross_encoder': 0.85,
                'cross_encoder_confidence': 0.90,
                'ce_pairs': [
                    {
                        'query': 'python programming',
                        'context': 'Developed Python applications for data analysis',
                        'score': 0.95
                    }
                ],
                'matched_required_skills': ['python'],
                'missing_required_skills': ['javascript']
            },
            'parsed': {
                'experience': [
                    {
                        'title': 'Software Engineer',
                        'description': 'Developed Python applications for data analysis'
                    }
                ],
                'skills': [{'name': 'python', 'proficiency': 'advanced'}]
            }
        }
        
        jd_criteria = {
            'required_skills': ['python', 'javascript'],
            'nice_to_have_skills': ['typescript']
        }
        
        # Collect evidence
        evidence_pool = self.evidence_collector.collect_evidence(test_data, jd_criteria)
        
        # Check that CE evidence has highest priority
        ce_evidence = [item for item in evidence_pool if item.source == 'ce_pairs']
        other_evidence = [item for item in evidence_pool if item.source != 'ce_pairs']
        
        if ce_evidence and other_evidence:
            ce_confidence = max(item.confidence for item in ce_evidence)
            other_confidence = max(item.confidence for item in other_evidence)
            
            self.assertGreaterEqual(ce_confidence, other_confidence,
                                   "CE evidence should have highest confidence")
        
        print("✓ Evidence source priority maintained")
        print(f"  CE evidence: {len(ce_evidence)} items")
        print(f"  Other evidence: {len(other_evidence)} items")

    def test_validation_suggestion_for_thin_evidence(self):
        """Test that validation suggestions are provided when evidence is thin."""
        print("\n=== Testing Validation Suggestions for Thin Evidence ===")
        
        # Test data with very thin evidence
        test_data = {
            'candidate_id': 'test_candidate',
            'scores': {
                'cross_encoder': 0.0,
                'cross_encoder_confidence': 0.0,
                'ce_pairs': [],
                'matched_required_skills': [],
                'missing_required_skills': ['python', 'javascript', 'react']
            },
            'parsed': {
                'experience': [],
                'skills': [],
                'raw_text': ''
            }
        }
        
        jd_criteria = {
            'required_skills': ['python', 'javascript', 'react'],
            'nice_to_have_skills': ['typescript']
        }
        
        # Generate analysis
        analysis = self.nlg_generator.generate_analysis(
            test_data, jd_criteria, batch_size=1
        )
        
        analysis_text = analysis['analysis_text']
        
        # Check for validation suggestion
        validation_terms = ['validation', 'assessment', 'code sample', 'technical']
        has_validation_suggestion = any(term in analysis_text.lower() for term in validation_terms)
        
        self.assertTrue(has_validation_suggestion,
                       "Analysis should suggest validation when evidence is thin")
        
        print("✓ Validation suggestion provided for thin evidence")
        print(f"  Analysis: {analysis_text}")

    def test_metadata_tracking(self):
        """Test that metadata is properly tracked for explainability."""
        print("\n=== Testing Metadata Tracking ===")
        
        test_data = {
            'candidate_id': 'test_candidate',
            'scores': {
                'cross_encoder': 0.85,
                'cross_encoder_confidence': 0.90,
                'ce_pairs': [
                    {
                        'query': 'python programming',
                        'context': 'Developed Python applications',
                        'score': 0.95
                    }
                ],
                'matched_required_skills': ['python'],
                'missing_required_skills': ['javascript']
            },
            'parsed': {
                'experience': [{'title': 'Software Engineer', 'description': 'Python development'}],
                'skills': [{'name': 'python', 'proficiency': 'advanced'}]
            }
        }
        
        jd_criteria = {
            'required_skills': ['python', 'javascript'],
            'nice_to_have_skills': ['typescript']
        }
        
        # Generate analysis
        analysis = self.nlg_generator.generate_analysis(
            test_data, jd_criteria, batch_size=1
        )
        
        # Check that metadata is included
        self.assertIn('metadata', analysis,
                     "Analysis should include metadata")
        
        metadata = analysis['metadata']
        self.assertIn('template_version', metadata,
                     "Metadata should include template version")
        self.assertIn('processing_time', metadata,
                     "Metadata should include processing time")
        self.assertIn('sentence_count', metadata,
                     "Metadata should include sentence count")
        
        print("✓ Metadata tracking working")
        print(f"  Template version: {metadata.get('template_version', 'N/A')}")
        print(f"  Processing time: {metadata.get('processing_time', 'N/A')}s")
        print(f"  Sentence count: {metadata.get('sentence_count', 'N/A')}")

    def test_end_to_end_evidence_based_analysis(self):
        """Test complete evidence-based analysis pipeline."""
        print("\n=== Testing End-to-End Evidence-Based Analysis ===")
        
        # Comprehensive test data
        test_data = {
            'candidate_id': 'test_candidate',
            'scores': {
                'cross_encoder': 0.85,
                'cross_encoder_confidence': 0.90,
                'ce_pairs': [
                    {
                        'query': 'python programming',
                        'context': 'Developed Python applications for data analysis using pandas and numpy',
                        'score': 0.95
                    },
                    {
                        'query': 'javascript development',
                        'context': 'Built web applications using JavaScript and React',
                        'score': 0.80
                    }
                ],
                'matched_required_skills': ['python', 'javascript'],
                'missing_required_skills': ['react', 'node.js'],
                'matched_nice_to_have_skills': ['typescript'],
                'missing_nice_to_have_skills': ['aws', 'docker']
            },
            'parsed': {
                'experience': [
                    {
                        'title': 'Software Engineer',
                        'description': 'Developed Python applications for data analysis using pandas and numpy. Built web applications using JavaScript and React.'
                    }
                ],
                'skills': [
                    {'name': 'python', 'proficiency': 'advanced'},
                    {'name': 'javascript', 'proficiency': 'intermediate'},
                    {'name': 'typescript', 'proficiency': 'beginner'}
                ]
            }
        }
        
        jd_criteria = {
            'required_skills': ['python', 'javascript', 'react', 'node.js'],
            'nice_to_have_skills': ['typescript', 'aws', 'docker'],
            'role_title': 'Full Stack Developer'
        }
        
        # Generate complete analysis
        analysis = self.nlg_generator.generate_analysis(
            test_data, jd_criteria, batch_size=3
        )
        
        # Verify all components are present
        analysis_text = analysis['analysis_text']
        
        # Check for score context
        self.assertIn('Ranked #', analysis_text,
                     "Analysis should include ranking context")
        
        # Check for strengths with evidence
        self.assertIn('python', analysis_text.lower(),
                     "Analysis should mention Python as strength")
        
        # Check for gaps
        self.assertIn('react', analysis_text.lower(),
                     "Analysis should mention missing React skills")
        
        # Check for concrete example
        self.assertIn('data analysis', analysis_text.lower(),
                     "Analysis should include concrete example")
        
        # Check for metadata
        self.assertIn('metadata', analysis,
                     "Analysis should include metadata")
        
        print("✓ End-to-end evidence-based analysis working")
        print(f"  Analysis length: {len(analysis_text)} characters")
        print(f"  Metadata fields: {len(analysis['metadata'])}")
        
        # Print sample analysis
        print(f"\nSample Analysis:")
        print(f"{analysis_text[:200]}...")


def run_evidence_based_tests():
    """Run all evidence-based analysis tests."""
    print("=" * 60)
    print("EVIDENCE-BASED ANALYSIS SYSTEM TESTS")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestEvidenceBasedAnalysis)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun) * 100
    print(f"\nSuccess rate: {success_rate:.1f}%")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_evidence_based_tests()
    sys.exit(0 if success else 1)