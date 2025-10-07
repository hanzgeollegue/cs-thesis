#!/usr/bin/env python3
"""
Test script for batch upload functionality.
This script tests the batch processor without needing the web interface.
"""

import os
import sys
import django
from pathlib import Path

# Add the project directory to Python path
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))

# Setup Django - use the correct settings module
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'settings')
django.setup()

from resume_processor.batch_processor import BatchProcessor
from resume_processor.config import get_openai_config, validate_config

def test_batch_processor():
    """Test the batch processor functionality."""
    print("🧪 Testing Batch Processor")
    print("=" * 50)
    
    # Check configuration
    config = get_openai_config()
    config_issues = validate_config()
    
    print(f"OpenAI API Key configured: {bool(config['api_key'])}")
    print(f"Model: {config['model']}")
    
    if config_issues:
        print("Configuration issues:")
        for issue in config_issues:
            print(f"  - {issue}")
    else:
        print("✅ Configuration looks good!")
    
    # Test batch processor initialization
    try:
        processor = BatchProcessor()
        print("✅ Batch processor initialized successfully")
        
        # Test with a sample job description
        job_description = """
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
        
        print(f"\n📝 Job Description: {len(job_description.split())} words")
        
        # Test with empty resume list (should work)
        try:
            result = processor.process_batch([], job_description)
            if "error" in result:
                print(f"✅ Empty batch handling: {result['error']}")
            else:
                print("✅ Empty batch handling: Success")
        except Exception as e:
            print(f"❌ Empty batch handling failed: {e}")
        
        # Test with invalid job description
        try:
            result = processor.process_batch([], "")
            if result.get("error") == "job_description_required":
                print("✅ Job description validation: Working")
            else:
                print(f"⚠️  Job description validation: Unexpected result - {result}")
        except Exception as e:
            print(f"❌ Job description validation failed: {e}")
        
        # Test with too many resumes
        try:
            fake_resumes = [f"resume_{i}.pdf" for i in range(30)]
            result = processor.process_batch(fake_resumes, job_description)
            if "error" in result and "Batch limit exceeded" in result["error"]:
                print("✅ Batch size validation: Working")
            else:
                print(f"⚠️  Batch size validation: Unexpected result - {result}")
        except Exception as e:
            print(f"❌ Batch size validation failed: {e}")
        
        print("\n🎉 Basic functionality tests completed!")
        
    except Exception as e:
        print(f"❌ Failed to initialize batch processor: {e}")
        import traceback
        traceback.print_exc()

def test_pdf_parser():
    """Test the PDF parser functionality."""
    print("\n📄 Testing PDF Parser")
    print("=" * 50)
    
    try:
        from resume_processor.enhanced_pdf_parser import PDFParser
        
        parser = PDFParser()
        print("✅ PDF parser initialized successfully")
        
        # Test method availability
        methods = [method for method in dir(parser) if not method.startswith('_')]
        print(f"Available methods: {methods}")
        
        # Check if required method exists
        if hasattr(parser, '_extract_structured_data'):
            print("✅ _extract_structured_data method exists")
        else:
            print("❌ _extract_structured_data method missing")
            
    except Exception as e:
        print(f"❌ PDF parser test failed: {e}")
        import traceback
        traceback.print_exc()

def test_llm_ranker():
    """Test the LLM ranker functionality."""
    print("\n🤖 Testing LLM Ranker")
    print("=" * 50)
    
    try:
        from resume_processor.llm_ranker import LLMRanker
        
        # Test without API key
        ranker = LLMRanker()
        print("✅ LLM ranker initialized without API key")
        
        # Test with API key if available
        config = get_openai_config()
        if config['api_key']:
            ranker_with_key = LLMRanker(api_key=config['api_key'])
            print("✅ LLM ranker initialized with API key")
        else:
            print("ℹ️  No API key available for testing")
            
    except Exception as e:
        print(f"❌ LLM ranker test failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run all tests."""
    print("🚀 Starting Batch Processor Tests")
    print("=" * 60)
    
    test_batch_processor()
    test_pdf_parser()
    test_llm_ranker()
    
    print("\n" + "=" * 60)
    print("🏁 All tests completed!")
    
    print("\n💡 To test the web interface:")
    print("1. Start the server: python manage.py runserver")
    print("2. Open: http://127.0.0.1:8000")
    print("3. Try the batch upload functionality")

if __name__ == "__main__":
    main()
