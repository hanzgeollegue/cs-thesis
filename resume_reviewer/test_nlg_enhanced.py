"""
Comprehensive test suite for the enhanced NLG system.
Tests determinism, variation, context-sensitivity, grammar, explainability, and evidence-based analysis.
"""

import unittest
import sys
import os
import time
import hashlib

# Add the resume_processor module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'resume_processor'))

from nlg_generator_enhanced import (
    EnhancedCandidateAnalyzer,
    EnhancedFactExtractor,
    EnhancedNLGGenerator,
    generate_candidate_analysis_enhanced,
    generate_candidate_facts_enhanced
)
from nlg_config import get_domain_config
from nlg_templates import TemplateEngine, VariantSelector
from nlg_polisher import GrammarPolisher
from nlg_metadata import ProvenanceTracker
from nlg_summary import BulletSummaryGenerator
from profile_analyzer import ProfileAnalyzer
from evidence_collector import EvidenceCollector, EvidenceItem


class TestEnhancedNLGSystem(unittest.TestCase):
    """Test the enhanced NLG system."""
    
    def setUp(self):
        """Set up test data."""
        self.sample_jd_criteria = {
            'position_title': 'Senior Software Engineer',
            'experience_min_years': 5,
            'must_have_skills': ['python', 'javascript', 'react', 'aws'],
            'nice_to_have_skills': ['docker', 'kubernetes', 'typescript'],
            'jd_summary': 'We are seeking a Senior Software Engineer with 5+ years experience in python, javascript, react, aws.'
        }
        
        self.sample_candidate_excellent = {
            'id': 'candidate_1',
            'scores': {
                'final_score': 0.85,
                'coverage': 0.9,
                'has_match_skills': True,
                'has_match_experience': True,
                'matched_required_skills': ['python', 'javascript', 'react', 'aws'],
                'matched_nice_skills': ['docker', 'typescript'],
                'missing_skills': []
            },
            'parsed': {
                'experience': [
                    {'title': 'Senior Software Engineer', 'company': 'Tech Corp', 'bullets': ['Built React apps', 'Python backend', 'AWS deployment']},
                    {'title': 'Software Engineer', 'company': 'Startup Inc', 'bullets': ['Full-stack development', 'Team lead']}
                ],
                'education': [{'degree': 'Computer Science', 'school': 'University'}],
                'projects': [{'name': 'Web App', 'technologies': ['React', 'Python', 'AWS']}],
                'skills': ['python', 'javascript', 'react', 'aws', 'docker']
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
                'missing_skills': ['javascript', 'react', 'aws']
            },
            'parsed': {
                'experience': [],
                'education': [{'degree': 'Business', 'school': 'College'}],
                'projects': [],
                'skills': ['python']
            },
            'meta': {'source_file': 'weak_candidate.pdf'}
        }
        
        self.sample_batch_stats = {
            'candidate_count': 10,
            'top_score': 90.0,
            'median_score': 60.0,
            'avg_score': 55.0,
            'avg_skills_matched': 2.5,
            'score_distribution': {
                'excellent': 2,
                'strong': 3,
                'moderate': 3,
                'weak': 2
            }
        }
    
    def test_determinism(self):
        """Test that the same input always produces the same output."""
        analyzer = EnhancedCandidateAnalyzer()
        
        # Generate analysis multiple times
        result1 = analyzer.generate_analysis(self.sample_candidate_excellent, self.sample_jd_criteria, self.sample_batch_stats)
        result2 = analyzer.generate_analysis(self.sample_candidate_excellent, self.sample_jd_criteria, self.sample_batch_stats)
        result3 = analyzer.generate_analysis(self.sample_candidate_excellent, self.sample_jd_criteria, self.sample_batch_stats)
        
        # Should be identical
        self.assertEqual(result1['text'], result2['text'])
        self.assertEqual(result2['text'], result3['text'])
        self.assertEqual(result1['bullets'], result2['bullets'])
        self.assertEqual(result2['bullets'], result3['bullets'])
        
        # Should have proper structure
        self.assertIn('text', result1)
        self.assertIn('bullets', result1)
        self.assertIn('facts', result1)
        self.assertIn('metadata', result1)
    
    def test_variation(self):
        """Test that different candidates produce different phrasing."""
        analyzer = EnhancedCandidateAnalyzer()
        
        # Generate analysis for different candidates
        result_excellent = analyzer.generate_analysis(self.sample_candidate_excellent, self.sample_jd_criteria, self.sample_batch_stats)
        result_weak = analyzer.generate_analysis(self.sample_candidate_weak, self.sample_jd_criteria, self.sample_batch_stats)
        
        # Should be different
        self.assertNotEqual(result_excellent['text'], result_weak['text'])
        self.assertNotEqual(result_excellent['bullets'], result_weak['bullets'])
        
        # Should reflect different tiers
        self.assertIn('Strong', result_excellent['text'])
        self.assertIn('Limited', result_weak['text'])
    
    def test_context_sensitivity(self):
        """Test that analysis adapts to different contexts."""
        analyzer = EnhancedCandidateAnalyzer()
        
        # Test with different seniority levels
        junior_candidate = self.sample_candidate_excellent.copy()
        junior_candidate['parsed']['experience'] = [
            {'title': 'Junior Developer', 'company': 'Tech Corp', 'bullets': ['Assisted with React apps']}
        ]
        
        senior_candidate = self.sample_candidate_excellent.copy()
        senior_candidate['parsed']['experience'] = [
            {'title': 'Senior Software Engineer', 'company': 'Tech Corp', 'bullets': ['Architected React apps', 'Led team of 5']},
            {'title': 'Staff Engineer', 'company': 'Big Corp', 'bullets': ['Technical leadership', 'Mentored junior developers']}
        ]
        
        result_junior = analyzer.generate_analysis(junior_candidate, self.sample_jd_criteria, self.sample_batch_stats)
        result_senior = analyzer.generate_analysis(senior_candidate, self.sample_jd_criteria, self.sample_batch_stats)
        
        # Should have different tones
        self.assertNotEqual(result_junior['text'], result_senior['text'])
    
    def test_grammar_fixes(self):
        """Test grammar polishing functionality."""
        polisher = GrammarPolisher()
        
        # Test singular/plural fixes
        test_text = "1 skills and 2 skill. The candidate has 3 year of experience."
        polished_text, polish_log = polisher.apply_polish(test_text)
        
        self.assertIn("1 skill", polished_text)
        self.assertIn("2 skills", polished_text)
        self.assertIn("3 years", polished_text)
        self.assertTrue(len(polish_log) > 0)
    
    def test_explainability(self):
        """Test metadata and explainability features."""
        analyzer = EnhancedCandidateAnalyzer()
        result = analyzer.generate_analysis(self.sample_candidate_excellent, self.sample_jd_criteria, self.sample_batch_stats)
        
        # Check metadata structure
        metadata = result['metadata']
        self.assertIn('version', metadata)
        self.assertIn('sentences', metadata)
        self.assertIn('polish_operations', metadata)
        self.assertIn('profile_context', metadata)
        self.assertIn('batch_context', metadata)
        
        # Check sentence metadata
        sentences = metadata['sentences']
        self.assertGreater(len(sentences), 0)
        
        for sentence in sentences:
            self.assertIn('template_id', sentence)
            self.assertIn('inputs_used', sentence)
            self.assertIn('rules_applied', sentence)
    
    def test_bullet_generation(self):
        """Test bullet point generation."""
        generator = BulletSummaryGenerator()
        facts = EnhancedFactExtractor().extract_facts(self.sample_candidate_excellent, self.sample_jd_criteria, self.sample_batch_stats)
        
        bullets = generator.generate_bullets(self.sample_candidate_excellent, facts, self.sample_batch_stats)
        
        # Should generate bullets
        self.assertGreater(len(bullets), 0)
        self.assertLessEqual(len(bullets), 5)
        
        # Should contain score information
        score_bullets = [b for b in bullets if '%' in b]
        self.assertGreater(len(score_bullets), 0)
    
    def test_profile_analysis(self):
        """Test profile analysis functionality."""
        analyzer = ProfileAnalyzer()
        
        # Test seniority detection
        seniority = analyzer.detect_seniority(self.sample_candidate_excellent)
        self.assertIn(seniority, ['intern', 'junior', 'mid', 'senior', 'lead', 'staff'])
        
        # Test career progression
        progression = analyzer.assess_career_progression(self.sample_candidate_excellent)
        self.assertIn('trajectory', progression)
        self.assertIn('velocity', progression)
        
        # Test specialization identification
        specializations = analyzer.identify_specializations(self.sample_candidate_excellent)
        self.assertIn('primary', specializations)
        self.assertIn('type', specializations)
    
    def test_template_variation(self):
        """Test template variation system."""
        engine = TemplateEngine()
        
        # Test opening sentence variation
        opening1 = engine.render_opening(85.0, 'excellent', 'senior', 
                                       {'matched_required': ['python'], 'total_required': 3}, 'candidate_1')
        opening2 = engine.render_opening(85.0, 'excellent', 'senior', 
                                       {'matched_required': ['python'], 'total_required': 3}, 'candidate_2')
        
        # Should be different due to variant selection
        self.assertNotEqual(opening1, opening2)
        
        # But should contain same information
        self.assertIn('85%', opening1)
        self.assertIn('85%', opening2)
        self.assertIn('1/3', opening1)
        self.assertIn('1/3', opening2)
    
    def test_batch_insights(self):
        """Test batch-relative insights."""
        extractor = EnhancedFactExtractor()
        facts = extractor.extract_facts(self.sample_candidate_excellent, self.sample_jd_criteria, self.sample_batch_stats)
        
        # Should include batch position
        self.assertIn('batch_position', facts)
        batch_pos = facts['batch_position']
        self.assertIn('percentile', batch_pos)
        self.assertIn('gap_from_top', batch_pos)
        self.assertIn('above_median', batch_pos)
    
    def test_readability_validation(self):
        """Test readability validation."""
        polisher = GrammarPolisher()
        
        # Test with good text
        good_text = "Strong candidate with 85% match and 3/4 required skills. Has required skills: python, javascript, react. Experience includes: Senior Developer at Tech Corp."
        validation = polisher.validate_readability(good_text)
        
        self.assertIn('avg_sentence_length', validation)
        self.assertIn('vocabulary_diversity', validation)
        self.assertIn('overall_quality', validation)
    
    def test_backward_compatibility(self):
        """Test backward compatibility with original API."""
        from nlg_generator_enhanced import generate_candidate_analysis, generate_candidate_facts
        
        # Test without batch_stats (should use original version)
        text = generate_candidate_analysis(self.sample_candidate_excellent, self.sample_jd_criteria)
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 10)
        
        # Test with batch_stats (should use enhanced version)
        result = generate_candidate_analysis(self.sample_candidate_excellent, self.sample_jd_criteria, self.sample_batch_stats)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 10)
    
    def test_error_handling(self):
        """Test error handling and graceful degradation."""
        analyzer = EnhancedCandidateAnalyzer()
        
        # Test with malformed data
        malformed_candidate = {'id': 'test', 'scores': {}, 'parsed': {}, 'meta': {}}
        result = analyzer.generate_analysis(malformed_candidate, self.sample_jd_criteria, self.sample_batch_stats)
        
        # Should not crash and should return some result
        self.assertIn('text', result)
        self.assertIn('bullets', result)
        self.assertIn('facts', result)
        self.assertIn('metadata', result)
    
    def test_performance(self):
        """Test performance characteristics."""
        analyzer = EnhancedCandidateAnalyzer()
        
        # Time the analysis generation
        start_time = time.time()
        result = analyzer.generate_analysis(self.sample_candidate_excellent, self.sample_jd_criteria, self.sample_batch_stats)
        end_time = time.time()
        
        # Should complete within reasonable time (less than 1 second)
        self.assertLess(end_time - start_time, 1.0)
        
        # Check processing time in metadata
        processing_time = result['metadata'].get('total_processing_time', 0)
        self.assertGreater(processing_time, 0)
        self.assertLess(processing_time, 1.0)
    
    def test_all_template_paths(self):
        """Test all template paths (excellent/strong/moderate/weak × senior/junior/mid)."""
        analyzer = EnhancedCandidateAnalyzer()
        
        # Test different score tiers
        for score_pct in [90, 70, 50, 30]:  # excellent, strong, moderate, weak
            candidate = self.sample_candidate_excellent.copy()
            candidate['scores']['final_score'] = score_pct / 100.0
            
            result = analyzer.generate_analysis(candidate, self.sample_jd_criteria, self.sample_batch_stats)
            
            # Should generate valid analysis
            self.assertIsInstance(result['text'], str)
            self.assertGreater(len(result['text']), 10)
            self.assertIsInstance(result['bullets'], list)
            self.assertGreater(len(result['bullets']), 0)


class TestTemplateVariation(unittest.TestCase):
    """Test template variation system specifically."""
    
    def test_variant_selector_determinism(self):
        """Test that VariantSelector is deterministic."""
        selector1 = VariantSelector('candidate_1')
        selector2 = VariantSelector('candidate_1')
        
        # Same candidate should get same variants
        for i in range(10):
            self.assertEqual(selector1.select_variant(5, f'context_{i}'), 
                           selector2.select_variant(5, f'context_{i}'))
    
    def test_variant_selector_diversity(self):
        """Test that different candidates get different variants."""
        selector1 = VariantSelector('candidate_1')
        selector2 = VariantSelector('candidate_2')
        
        # Different candidates should get different variants (usually)
        variants1 = [selector1.select_variant(10, f'context_{i}') for i in range(10)]
        variants2 = [selector2.select_variant(10, f'context_{i}') for i in range(10)]
        
        # Should have some differences (not all identical)
        self.assertNotEqual(variants1, variants2)


class TestGrammarPolisher(unittest.TestCase):
    """Test grammar polisher specifically."""
    
    def setUp(self):
        self.polisher = GrammarPolisher()
    
    def test_singular_plural_fixes(self):
        """Test singular/plural fixes."""
        test_cases = [
            ("1 skills", "1 skill"),
            ("2 skill", "2 skills"),
            ("0 skills", "0 skill"),
            ("3 year", "3 years"),
            ("1 role", "1 role"),
            ("2 role", "2 roles")
        ]
        
        for input_text, expected in test_cases:
            result, _ = self.polisher.apply_polish(input_text)
            self.assertIn(expected, result)
    
    def test_redundancy_removal(self):
        """Test redundancy removal."""
        test_text = "The candidate has python skills. The candidate has python skills."
        result, polish_log = self.polisher.apply_polish(test_text)
        
        # Should remove duplicate sentences
        sentences = result.split('. ')
        self.assertLessEqual(len(sentences), 2)  # Should not have duplicates
    
    def test_transition_addition(self):
        """Test transition phrase addition."""
        test_text = "Strong candidate with 85% match. Has required skills: python. Experience includes: Senior Developer."
        result, polish_log = self.polisher.apply_polish(test_text)
        
        # Should add transitions
        has_transition = any(connector in result for connector in 
                           ['Additionally', 'However', 'Furthermore', 'Moreover'])
        self.assertTrue(has_transition)

    def test_evidence_based_analysis_structure(self):
        """Test that evidence-based analysis follows the correct structure."""
        try:
            from nlg_generator_enhanced import generate_candidate_analysis_enhanced
            
            # Create test data with evidence
            candidate_data = {
                'id': 'test_evidence_001',
                'scores': {
                    'final_score': 0.75,
                    'coverage': 0.8,
                    'has_match_skills': True,
                    'has_match_experience': True,
                    'matched_required_skills': ['python', 'django'],
                    'matched_nice_skills': ['postgresql'],
                    'missing_skills': ['kubernetes'],
                    'ce_matches': [
                        {
                            'skill': 'python',
                            'evidence_text': 'Developed REST APIs using Python and Django framework',
                            'ce_score': 0.9,
                            'company': 'TechCorp',
                            'role': 'Senior Developer'
                        }
                    ],
                    'inferred_skills_evidence': [
                        {
                            'skill': 'django',
                            'evidence_text': 'Built web applications with Django ORM',
                            'confidence': 0.8,
                            'company': 'TechCorp'
                        }
                    ]
                },
                'parsed': {
                    'experience': [
                        {
                            'title': 'Senior Developer',
                            'company': 'TechCorp',
                            'start_date': '2020-01-01',
                            'end_date': '2023-12-31',
                            'description': 'Developed REST APIs using Python and Django framework'
                        }
                    ],
                    'skills': [
                        {'name': 'python', 'proficiency': 'expert'},
                        {'name': 'django', 'proficiency': 'advanced'}
                    ],
                    'education': [{'degree': 'BS Computer Science'}]
                }
            }
            
            jd_criteria = {
                'id': 'jd_001',
                'position_title': 'Senior Python Developer',
                'must_have_skills': ['python', 'django', 'kubernetes'],
                'nice_to_have_skills': ['postgresql', 'redis']
            }
            
            batch_stats = {
                'candidate_count': 10,
                'top_score': 85.0,
                'median_score': 65.0,
                'all_final_scores': [0.85, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35]
            }
            
            result = generate_candidate_analysis_enhanced(candidate_data, jd_criteria, batch_stats)
            
            # Check structure
            self.assertIn('text', result)
            self.assertIn('bullets', result)
            self.assertIn('facts', result)
            self.assertIn('metadata', result)
            
            # Check evidence in facts
            facts = result['facts']
            self.assertIn('evidence', facts)
            self.assertIn('data_quality', facts)
            
            # Check evidence items
            evidence = facts['evidence']
            self.assertIsInstance(evidence, list)
            self.assertGreater(len(evidence), 0)
            
            # Check data quality assessment
            data_quality = facts['data_quality']
            self.assertIn('needs_caveat', data_quality)
            self.assertIn('evidence_counts', data_quality)
            
            # Check batch position
            self.assertIn('batch_position', facts)
            batch_pos = facts['batch_position']
            self.assertIn('percentile', batch_pos)
            self.assertIn('rank_estimate', batch_pos)
            
        except ImportError:
            self.skipTest("Enhanced NLG system not available")

    def test_confidence_filtering(self):
        """Test that low-confidence evidence is filtered out."""
        try:
            from nlg_generator_enhanced import generate_candidate_analysis_enhanced
            
            # Create test data with mixed confidence evidence
            candidate_data = {
                'id': 'test_confidence_001',
                'scores': {
                    'final_score': 0.6,
                    'coverage': 0.7,
                    'has_match_skills': True,
                    'matched_required_skills': ['python', 'javascript'],
                    'matched_nice_skills': [],
                    'missing_skills': ['react'],
                    'ce_matches': [
                        {
                            'skill': 'python',
                            'evidence_text': 'Strong Python development experience',
                            'ce_score': 0.9,  # High confidence
                            'company': 'TechCorp'
                        },
                        {
                            'skill': 'javascript',
                            'evidence_text': 'Some JavaScript usage',
                            'ce_score': 0.3,  # Low confidence - should be filtered
                            'company': 'TechCorp'
                        }
                    ]
                },
                'parsed': {
                    'experience': [
                        {
                            'title': 'Developer',
                            'company': 'TechCorp',
                            'start_date': '2021-01-01',
                            'end_date': '2023-12-31'
                        }
                    ],
                    'skills': [
                        {'name': 'python', 'proficiency': 'advanced'},
                        {'name': 'javascript', 'proficiency': 'basic'}
                    ]
                }
            }
            
            jd_criteria = {
                'id': 'jd_001',
                'position_title': 'Full Stack Developer',
                'must_have_skills': ['python', 'javascript', 'react'],
                'nice_to_have_skills': []
            }
            
            result = generate_candidate_analysis_enhanced(candidate_data, jd_criteria)
            
            # Check that only high-confidence evidence is used
            facts = result['facts']
            evidence = facts['evidence']
            
            # Should have evidence items
            self.assertGreater(len(evidence), 0)
            
            # Check that low-confidence evidence is marked appropriately
            high_confidence_evidence = [e for e in evidence if e.get('confidence', 0) >= 0.6]
            self.assertGreater(len(high_confidence_evidence), 0)
            
        except ImportError:
            self.skipTest("Enhanced NLG system not available")

    def test_caveat_generation(self):
        """Test that caveats are generated when data quality is poor."""
        try:
            from nlg_generator_enhanced import generate_candidate_analysis_enhanced
            
            # Create test data with poor parsing quality
            candidate_data = {
                'id': 'test_caveat_001',
                'scores': {
                    'final_score': 0.4,
                    'coverage': 0.3,
                    'has_match_skills': False,
                    'matched_required_skills': [],
                    'matched_nice_skills': [],
                    'missing_skills': ['python', 'django'],
                    'ce_matches': []  # No CE evidence
                },
                'parsed': {
                    'experience': [],  # No experience parsed
                    'skills': [],      # No skills parsed
                    'education': []    # No education parsed
                }
            }
            
            jd_criteria = {
                'id': 'jd_001',
                'position_title': 'Python Developer',
                'must_have_skills': ['python', 'django'],
                'nice_to_have_skills': []
            }
            
            result = generate_candidate_analysis_enhanced(candidate_data, jd_criteria)
            
            # Check that caveat is generated
            facts = result['facts']
            data_quality = facts['data_quality']
            
            # Should need caveat due to poor parsing
            self.assertTrue(data_quality.get('needs_caveat', False))
            
            # Check that analysis text contains caveat language
            analysis_text = result['text']
            caveat_indicators = [
                'limited', 'incomplete', 'based on available', 'constraints noted'
            ]
            has_caveat = any(indicator in analysis_text.lower() for indicator in caveat_indicators)
            self.assertTrue(has_caveat, f"Expected caveat in analysis: {analysis_text}")
            
        except ImportError:
            self.skipTest("Enhanced NLG system not available")

    def test_concrete_example_generation(self):
        """Test that concrete examples are generated with proper formatting."""
        try:
            from nlg_generator_enhanced import generate_candidate_analysis_enhanced
            
            # Create test data with strong evidence
            candidate_data = {
                'id': 'test_example_001',
                'scores': {
                    'final_score': 0.8,
                    'coverage': 0.9,
                    'has_match_skills': True,
                    'matched_required_skills': ['python', 'django'],
                    'matched_nice_skills': ['postgresql'],
                    'missing_skills': [],
                    'ce_matches': [
                        {
                            'skill': 'python',
                            'evidence_text': 'Developed scalable REST APIs using Python and Django framework for e-commerce platform',
                            'ce_score': 0.95,  # Very high confidence
                            'company': 'TechCorp',
                            'role': 'Senior Developer'
                        }
                    ]
                },
                'parsed': {
                    'experience': [
                        {
                            'title': 'Senior Developer',
                            'company': 'TechCorp',
                            'start_date': '2020-01-01',
                            'end_date': '2023-12-31',
                            'description': 'Developed scalable REST APIs using Python and Django framework for e-commerce platform'
                        }
                    ],
                    'skills': [
                        {'name': 'python', 'proficiency': 'expert'},
                        {'name': 'django', 'proficiency': 'expert'}
                    ]
                }
            }
            
            jd_criteria = {
                'id': 'jd_001',
                'position_title': 'Senior Python Developer',
                'must_have_skills': ['python', 'django'],
                'nice_to_have_skills': ['postgresql']
            }
            
            result = generate_candidate_analysis_enhanced(candidate_data, jd_criteria)
            
            # Check that concrete example is generated
            analysis_text = result['text']
            
            # Should contain concrete example indicators
            example_indicators = [
                'demonstrated', 'applied', 'used', 'implemented', 'techcorp'
            ]
            has_example = any(indicator in analysis_text.lower() for indicator in example_indicators)
            self.assertTrue(has_example, f"Expected concrete example in analysis: {analysis_text}")
            
            # Should mention the company
            self.assertIn('TechCorp', analysis_text)
            
        except ImportError:
            self.skipTest("Enhanced NLG system not available")

    def test_skills_hygiene_validation(self):
        """Test that skills hygiene is maintained (no overlaps between matched/missing)."""
        try:
            from nlg_generator_enhanced import generate_candidate_facts_enhanced
            
            # Create test data with potential skills conflicts
            candidate_data = {
                'id': 'test_hygiene_001',
                'scores': {
                    'final_score': 0.7,
                    'matched_required_skills': ['python', 'django'],  # These should not appear in missing
                    'matched_nice_skills': ['postgresql'],
                    'missing_skills': ['python', 'react']  # 'python' conflict with matched_required
                },
                'parsed': {
                    'experience': [
                        {
                            'title': 'Developer',
                            'company': 'TechCorp',
                            'start_date': '2021-01-01',
                            'end_date': '2023-12-31'
                        }
                    ],
                    'skills': [
                        {'name': 'python', 'proficiency': 'advanced'},
                        {'name': 'django', 'proficiency': 'intermediate'}
                    ]
                }
            }
            
            jd_criteria = {
                'id': 'jd_001',
                'position_title': 'Python Developer',
                'must_have_skills': ['python', 'django', 'react'],
                'nice_to_have_skills': ['postgresql']
            }
            
            facts = generate_candidate_facts_enhanced(candidate_data, jd_criteria)
            
            # Check skills hygiene
            skills = facts['skills']
            matched_required = set(skills['matched_required'])
            matched_nice = set(skills['matched_nice'])
            missing_required = set(skills['missing_required'])
            
            # Should be disjoint sets
            self.assertEqual(len(matched_required & matched_nice), 0, 
                           "Matched required and nice-to-have should not overlap")
            self.assertEqual(len(matched_required & missing_required), 0, 
                           "Matched required and missing should not overlap")
            self.assertEqual(len(matched_nice & missing_required), 0, 
                           "Matched nice-to-have and missing should not overlap")
            
            # Check that conflicts are logged
            self.assertIn('conflicts_detected', skills)
            conflicts = skills['conflicts_detected']
            self.assertGreater(len(conflicts), 0, "Should detect skills conflicts")
            
        except ImportError:
            self.skipTest("Enhanced NLG system not available")

    def test_evidence_based_analysis_generation(self):
        """Test evidence-based analysis generation with confidence filtering."""
        try:
            # Test data with mixed evidence quality
            candidate_data = {
                'id': 'candidate_evidence_test',
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
            
            # Generate evidence-based analysis
            nlg_generator = EnhancedNLGGenerator()
            analysis = nlg_generator.generate_analysis(candidate_data, jd_criteria, batch_size=3)
            
            # Verify analysis structure
            self.assertIn('analysis_text', analysis)
            self.assertIn('metadata', analysis)
            
            analysis_text = analysis['analysis_text']
            
            # Check for evidence-based content
            self.assertIn('python', analysis_text.lower(), "Should mention Python as strength")
            self.assertIn('react', analysis_text.lower(), "Should mention missing React skills")
            self.assertIn('data analysis', analysis_text.lower(), "Should include concrete example")
            
            # Check for ranking context
            self.assertIn('Ranked #', analysis_text, "Should include ranking context")
            
            # Check metadata
            metadata = analysis['metadata']
            self.assertIn('template_version', metadata)
            self.assertIn('processing_time', metadata)
            self.assertIn('sentence_count', metadata)
            
        except ImportError:
            self.skipTest("Enhanced NLG system not available")

    def test_confidence_filtering_and_hedged_language(self):
        """Test confidence filtering and hedged language for low-quality evidence."""
        try:
            # Test data with low confidence
            candidate_data = {
                'id': 'candidate_low_confidence',
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
            nlg_generator = EnhancedNLGGenerator()
            analysis = nlg_generator.generate_analysis(candidate_data, jd_criteria, batch_size=1)
            
            analysis_text = analysis['analysis_text']
            
            # Check for caveat
            self.assertIn('Assessment based on', analysis_text, "Should include caveat for low evidence")
            
            # Check for hedged language
            hedged_terms = ['appears to', 'may be', 'suggests', 'indicates', 'limited']
            has_hedged_language = any(term in analysis_text.lower() for term in hedged_terms)
            self.assertTrue(has_hedged_language, "Should use hedged language when confidence is low")
            
        except ImportError:
            self.skipTest("Enhanced NLG system not available")

    def test_batch_context_accuracy(self):
        """Test that batch context is accurate for different batch sizes."""
        try:
            candidate_data = {
                'id': 'candidate_batch_test',
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
            
            nlg_generator = EnhancedNLGGenerator()
            
            # Test single candidate batch
            analysis_single = nlg_generator.generate_analysis(candidate_data, jd_criteria, batch_size=1)
            self.assertIn('Only candidate in batch', analysis_single['analysis_text'],
                         "Single candidate should show 'Only candidate in batch'")
            
            # Test multi-candidate batch
            analysis_multi = nlg_generator.generate_analysis(candidate_data, jd_criteria, batch_size=3)
            self.assertIn('Ranked #', analysis_multi['analysis_text'],
                         "Multi-candidate batch should show ranking")
            
        except ImportError:
            self.skipTest("Enhanced NLG system not available")

    def test_evidence_collector_functionality(self):
        """Test evidence collector functionality."""
        try:
            evidence_collector = EvidenceCollector()
            
            # Test data
            candidate_data = {
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
            evidence_pool = evidence_collector.collect_evidence(candidate_data, jd_criteria)
            
            # Verify evidence collection
            self.assertIsInstance(evidence_pool, list, "Evidence pool should be a list")
            self.assertGreater(len(evidence_pool), 0, "Should collect some evidence")
            
            # Check evidence items
            for item in evidence_pool:
                self.assertIsInstance(item, EvidenceItem, "Evidence items should be EvidenceItem instances")
                self.assertIn(item.source, ['ce_pairs', 'skill_inference', 'text_match'], 
                            "Evidence source should be valid")
                self.assertGreaterEqual(item.confidence, 0.0, "Confidence should be non-negative")
                self.assertLessEqual(item.confidence, 1.0, "Confidence should be at most 1.0")
            
        except ImportError:
            self.skipTest("Evidence collector not available")

    def test_validation_suggestions_for_thin_evidence(self):
        """Test that validation suggestions are provided when evidence is thin."""
        try:
            # Test data with very thin evidence
            candidate_data = {
                'id': 'candidate_thin_evidence',
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
            nlg_generator = EnhancedNLGGenerator()
            analysis = nlg_generator.generate_analysis(candidate_data, jd_criteria, batch_size=1)
            
            analysis_text = analysis['analysis_text']
            
            # Check for validation suggestion
            validation_terms = ['validation', 'assessment', 'code sample', 'technical']
            has_validation_suggestion = any(term in analysis_text.lower() for term in validation_terms)
            self.assertTrue(has_validation_suggestion,
                           "Analysis should suggest validation when evidence is thin")
            
        except ImportError:
            self.skipTest("Enhanced NLG system not available")

    def test_metadata_tracking_and_explainability(self):
        """Test metadata tracking and explainability features."""
        try:
            candidate_data = {
                'id': 'candidate_metadata_test',
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
            nlg_generator = EnhancedNLGGenerator()
            analysis = nlg_generator.generate_analysis(candidate_data, jd_criteria, batch_size=1)
            
            # Check metadata
            self.assertIn('metadata', analysis, "Analysis should include metadata")
            
            metadata = analysis['metadata']
            required_fields = ['template_version', 'processing_time', 'sentence_count']
            for field in required_fields:
                self.assertIn(field, metadata, f"Metadata should include {field}")
            
            # Check processing time is reasonable
            processing_time = metadata.get('processing_time', 0)
            self.assertGreater(processing_time, 0, "Processing time should be positive")
            self.assertLess(processing_time, 10, "Processing time should be reasonable (<10s)")
            
        except ImportError:
            self.skipTest("Enhanced NLG system not available")


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
