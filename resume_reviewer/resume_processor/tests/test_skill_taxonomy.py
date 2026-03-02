"""
Unit tests for SkillTaxonomy class with multi-domain and fuzzy matching support.
"""

import unittest
import os
import sys
import tempfile
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from resume_processor.text_processor import SkillTaxonomy
except ImportError:
    # Try relative import
    from ..text_processor import SkillTaxonomy


class TestSkillTaxonomy(unittest.TestCase):
    """Test cases for SkillTaxonomy class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.taxonomy = SkillTaxonomy()
    
    def test_core_tech_skills_loaded(self):
        """Test that core tech skills are loaded."""
        # Test some known tech skills
        self.assertEqual(self.taxonomy.normalize_skill('python'), 'python')
        self.assertEqual(self.taxonomy.normalize_skill('react'), 'react')
        self.assertEqual(self.taxonomy.normalize_skill('javascript'), 'javascript')
        self.assertEqual(self.taxonomy.normalize_skill('aws'), 'aws')
    
    def test_skill_variations(self):
        """Test that skill variations are normalized correctly."""
        # Test variations
        self.assertEqual(self.taxonomy.normalize_skill('python3'), 'python')
        self.assertEqual(self.taxonomy.normalize_skill('reactjs'), 'react')
        self.assertEqual(self.taxonomy.normalize_skill('js'), 'javascript')
        self.assertEqual(self.taxonomy.normalize_skill('node.js'), 'nodejs')
    
    def test_extract_skills_from_text(self):
        """Test skill extraction from text."""
        text = "I have experience with Python, React, and AWS cloud services."
        skills = self.taxonomy.extract_skills_from_text(text)
        self.assertIn('python', skills)
        self.assertIn('react', skills)
        self.assertIn('aws', skills)
    
    def test_domain_taxonomy_loading(self):
        """Test loading domain-specific taxonomies."""
        # Create temporary taxonomy directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test domain file
            test_domain = {
                "test_skill": ["test skill", "test-skill", "testskill"],
                "another_skill": ["another skill", "another-skill"]
            }
            domain_file = os.path.join(temp_dir, "test_domain.json")
            with open(domain_file, 'w', encoding='utf-8') as f:
                json.dump(test_domain, f)
            
            # Create config
            config = {
                "enabled_domains": ["test_domain"],
                "fuzzy_matching": {"enabled": False}
            }
            config_file = os.path.join(temp_dir, "config.json")
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f)
            
            # Load taxonomy with test directory
            test_taxonomy = SkillTaxonomy(taxonomy_dir=temp_dir)
            
            # Test that domain skills are loaded
            self.assertEqual(test_taxonomy.normalize_skill('test skill'), 'test_skill')
            self.assertEqual(test_taxonomy.normalize_skill('test-skill'), 'test_skill')
            self.assertEqual(test_taxonomy.normalize_skill('another skill'), 'another_skill')
    
    def test_normalize_skill_with_confidence(self):
        """Test confidence scoring for skill normalization."""
        # Exact match should have confidence 1.0
        canonical, confidence = self.taxonomy.normalize_skill_with_confidence('python')
        self.assertEqual(canonical, 'python')
        self.assertEqual(confidence, 1.0)
        
        # Variation should also have confidence 1.0
        canonical, confidence = self.taxonomy.normalize_skill_with_confidence('python3')
        self.assertEqual(canonical, 'python')
        self.assertEqual(confidence, 1.0)
        
        # Unknown skill should have confidence 0.0 (if fuzzy matching disabled)
        canonical, confidence = self.taxonomy.normalize_skill_with_confidence('unknown_skill_xyz')
        self.assertEqual(confidence, 0.0)
    
    def test_get_matched_required_skills(self):
        """Test matching required skills against resume text."""
        resume_text = "I have experience with Python, React, JavaScript, and AWS."
        required_skills = ['python', 'react', 'java', 'docker']
        
        matched = self.taxonomy.get_matched_required_skills(resume_text, required_skills)
        
        # Should match python and react
        self.assertIn('python', matched)
        self.assertIn('react', matched)
        # Should not match java or docker (not in resume)
        self.assertNotIn('java', matched)
        self.assertNotIn('docker', matched)
    
    def test_get_matched_required_skills_with_confidence(self):
        """Test confidence-aware skill matching."""
        resume_text = "I have experience with Python and React."
        required_skills = ['python', 'react', 'docker']
        
        matched = self.taxonomy.get_matched_required_skills_with_confidence(
            resume_text, required_skills, min_confidence=0.7
        )
        
        # Should have matches for python and react
        matched_skills = [skill for skill, _ in matched]
        self.assertIn('python', matched_skills)
        self.assertIn('react', matched_skills)
        self.assertNotIn('docker', matched_skills)
        
        # All confidences should be >= 0.7
        for skill, confidence in matched:
            self.assertGreaterEqual(confidence, 0.7)
    
    def test_invalid_domain_file_handling(self):
        """Test handling of invalid domain files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create invalid JSON file
            invalid_file = os.path.join(temp_dir, "invalid.json")
            with open(invalid_file, 'w', encoding='utf-8') as f:
                f.write("invalid json content {")
            
            # Create config pointing to invalid domain
            config = {
                "enabled_domains": ["invalid"],
                "fuzzy_matching": {"enabled": False}
            }
            config_file = os.path.join(temp_dir, "config.json")
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f)
            
            # Should not crash, just log warning
            test_taxonomy = SkillTaxonomy(taxonomy_dir=temp_dir)
            # Should still have core tech skills
            self.assertEqual(test_taxonomy.normalize_skill('python'), 'python')
    
    def test_missing_config_file(self):
        """Test behavior when config file is missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # No config file
            test_taxonomy = SkillTaxonomy(taxonomy_dir=temp_dir)
            # Should still work with defaults (core tech only)
            self.assertEqual(test_taxonomy.normalize_skill('python'), 'python')
            self.assertEqual(len(test_taxonomy.loaded_domains), 0)
    
    def test_punctuation_normalization(self):
        """Test punctuation normalization in skill names."""
        # Test various punctuation formats
        self.assertEqual(self.taxonomy.normalize_skill('node.js'), 'nodejs')
        self.assertEqual(self.taxonomy.normalize_skill('react.js'), 'react')
        self.assertEqual(self.taxonomy.normalize_skill('c++'), 'c++')
        self.assertEqual(self.taxonomy.normalize_skill('asp.net'), 'asp.net')


if __name__ == '__main__':
    unittest.main()

