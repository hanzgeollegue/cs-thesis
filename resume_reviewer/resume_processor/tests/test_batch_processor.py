#!/usr/bin/env python3
"""
Tests for the batch processing pipeline.
"""

import os
import sys
import json
from pathlib import Path
from django.test import TestCase
from django.conf import settings

from ..batch_processor import BatchProcessor

class BatchProcessorTestCase(TestCase):
    """Test cases for the BatchProcessor class."""
    
    def setUp(self):
        """Set up test data."""
        self.processor = BatchProcessor()
        
        # Sample job description
        self.sample_jd = """
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
    
    def test_section_normalization(self):
        """Test section normalization functionality."""
        # Test section mapping
        test_sections = [
            {'header': 'Work Experience', 'content': ['Software Engineer at Tech Corp']},
            {'header': 'Technical Skills', 'content': ['Python', 'React', 'SQL']},
            {'header': 'Education', 'content': ['BS Computer Science']},
            {'header': 'Projects', 'content': ['Built a web app']}
        ]
        
        normalized = self.processor._normalize_sections(test_sections)
        
        # Check that sections are properly normalized
        self.assertIn('experience', normalized)
        self.assertIn('skills', normalized)
        self.assertIn('education', normalized)
        self.assertIn('misc', normalized)
        
        # Check content
        self.assertIn('Software Engineer at Tech Corp', normalized['experience'])
        self.assertIn('Python', normalized['skills'])
    
    def test_skill_taxonomy(self):
        """Test skill taxonomy functionality."""
        # Check that skill taxonomy is loaded
        self.assertIsInstance(self.processor.skill_taxonomy, dict)
        self.assertGreater(len(self.processor.skill_taxonomy), 0)
        
        # Check that skills have surface forms
        for skill_id, surface_forms in self.processor.skill_taxonomy.items():
            self.assertIsInstance(surface_forms, list)
            self.assertGreater(len(surface_forms), 0)
    
    def test_section_weights(self):
        """Test section weights configuration."""
        expected_weights = {
            'experience': 0.45,
            'skills': 0.35,
            'education': 0.15,
            'misc': 0.05
        }
        
        for section, weight in expected_weights.items():
            self.assertEqual(self.processor.section_weights[section], weight)
    
    def test_text_similarity(self):
        """Test text similarity computation."""
        # Test identical text
        similarity = self.processor._compute_text_similarity("hello world", "hello world")
        self.assertEqual(similarity, 1.0)
        
        # Test similar text
        similarity = self.processor._compute_text_similarity("hello world", "hello there world")
        self.assertGreater(similarity, 0.0)
        self.assertLess(similarity, 1.0)
        
        # Test different text
        similarity = self.processor._compute_text_similarity("hello world", "goodbye universe")
        self.assertEqual(similarity, 0.0)
        
        # Test empty text
        similarity = self.processor._compute_text_similarity("", "hello world")
        self.assertEqual(similarity, 0.0)
    
    def test_skill_extraction(self):
        """Test skill extraction from job description."""
        top_skills = self.processor._extract_top_skills(self.sample_jd)
        
        # Should extract some skills
        self.assertIsInstance(top_skills, list)
        
        # Check skill structure
        if top_skills:
            skill = top_skills[0]
            self.assertIn('skill_id', skill)
            self.assertIn('label', skill)
    
    def test_batch_summary_generation(self):
        """Test batch summary generation."""
        # Mock resumes for testing
        from unittest.mock import Mock
        
        mock_resume = Mock()
        mock_resume.id = "test-123"
        mock_resume.meta = {"processing_status": "success"}
        mock_resume.sections = {"skills": "Python", "experience": "Developer"}
        
        mock_ranking = [{"id": "test-123", "rank": 1}]
        
        summary = self.processor._generate_batch_summary([mock_resume], mock_ranking)
        
        self.assertIn('top_candidates', summary)
        self.assertIn('common_gaps', summary)
        self.assertIn('notes', summary)
        
        # Check that top candidate is included
        self.assertIn("test-123", summary['top_candidates']) 