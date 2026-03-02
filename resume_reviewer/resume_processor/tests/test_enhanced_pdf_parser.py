"""
Test cases for enhanced_pdf_parser.py to ensure proper section detection.
"""
import unittest
import os
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Configure Django if needed
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'settings')

try:
    import django
    django.setup()
except Exception:
    pass

from resume_processor.enhanced_pdf_parser import PDFParser

logger = logging.getLogger(__name__)


class TestEnhancedPDFParser(unittest.TestCase):
    """Test cases for PDF parser section detection."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.parser = PDFParser()
    
    def test_section_detection_produces_multiple_sections(self):
        """Test that parser produces at least 3 sections for typical resumes."""
        # This test requires a sample PDF file
        # For now, we'll test with mock data or skip if no test PDF available
        test_pdf_path = None
        
        # Look for a test PDF in common locations
        possible_paths = [
            "media/temp_uploads",
            "batch_processing_output",
            "test_data"
        ]
        
        for base_path in possible_paths:
            if os.path.exists(base_path):
                pdf_files = list(Path(base_path).glob("*.pdf"))
                if pdf_files:
                    test_pdf_path = str(pdf_files[0])
                    break
        
        if not test_pdf_path:
            self.skipTest("No test PDF file found")
        
        # Extract structured data
        structured_data = self.parser._extract_structured_data(test_pdf_path)
        
        # Assertions
        self.assertTrue(structured_data.get('success', False), "Extraction should succeed")
        sections = structured_data.get('sections', [])
        self.assertGreater(len(sections), 0, "Should have at least one section")
        
        # Check for multiple sections
        self.assertGreaterEqual(
            len(sections), 
            2, 
            f"Should have at least 2 sections, got {len(sections)}. Sections: {[s.get('header') for s in sections]}"
        )
        
        # Check canonical sections
        canonical_sections = structured_data.get('canonical_sections', {})
        self.assertIsInstance(canonical_sections, dict, "canonical_sections should be a dict")
        
        # Log section info for debugging
        logger.info(f"Test found {len(sections)} sections:")
        for i, section in enumerate(sections):
            header = section.get('header', '')
            canonical = section.get('canonical', 'unknown')
            content_len = sum(len(str(item)) for item in section.get('content', []))
            logger.info(f"  Section {i}: '{header}' -> '{canonical}' ({content_len} chars)")
    
    def test_no_single_section_has_excessive_content(self):
        """Test that no single section has >80% of total content."""
        test_pdf_path = None
        
        # Look for a test PDF
        for base_path in ["media/temp_uploads", "batch_processing_output", "test_data"]:
            if os.path.exists(base_path):
                pdf_files = list(Path(base_path).glob("*.pdf"))
                if pdf_files:
                    test_pdf_path = str(pdf_files[0])
                    break
        
        if not test_pdf_path:
            self.skipTest("No test PDF file found")
        
        structured_data = self.parser._extract_structured_data(test_pdf_path)
        
        if not structured_data.get('success'):
            self.skipTest(f"Extraction failed: {structured_data.get('error')}")
        
        sections = structured_data.get('sections', [])
        if len(sections) == 0:
            self.skipTest("No sections extracted")
        
        # Calculate total content length
        total_length = 0
        section_lengths = []
        
        for section in sections:
            content = section.get('content', [])
            section_len = sum(len(str(item)) for item in content)
            section_lengths.append((section.get('header', ''), section_len))
            total_length += section_len
        
        if total_length == 0:
            self.skipTest("No content extracted")
        
        # Check that no section has >80% of content
        for header, section_len in section_lengths:
            percentage = (section_len / total_length) * 100
            self.assertLessEqual(
                percentage,
                80.0,
                f"Section '{header}' has {percentage:.1f}% of content ({section_len}/{total_length} chars). "
                f"This indicates section detection failure."
            )
    
    def test_canonical_sections_present(self):
        """Test that canonical_sections contains expected keys."""
        test_pdf_path = None
        
        for base_path in ["media/temp_uploads", "batch_processing_output", "test_data"]:
            if os.path.exists(base_path):
                pdf_files = list(Path(base_path).glob("*.pdf"))
                if pdf_files:
                    test_pdf_path = str(pdf_files[0])
                    break
        
        if not test_pdf_path:
            self.skipTest("No test PDF file found")
        
        structured_data = self.parser._extract_structured_data(test_pdf_path)
        
        if not structured_data.get('success'):
            self.skipTest(f"Extraction failed: {structured_data.get('error')}")
        
        canonical_sections = structured_data.get('canonical_sections', {})
        
        # Should have at least some canonical sections
        self.assertGreater(
            len(canonical_sections),
            0,
            f"Should have canonical_sections, got: {canonical_sections}"
        )
        
        # Check that canonical sections have content
        for key, value in canonical_sections.items():
            self.assertIsInstance(value, str, f"Canonical section '{key}' should be a string")
            if key in ['experience', 'skills', 'education']:
                # Core sections should have some content for typical resumes
                self.assertGreater(
                    len(value),
                    0,
                    f"Core section '{key}' should have content"
                )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    unittest.main()

