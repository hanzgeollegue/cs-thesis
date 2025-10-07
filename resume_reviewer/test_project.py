#!/usr/bin/env python3
"""
Simple test script to verify project setup and basic functionality.
Run this after setting up the project to ensure everything works.
"""

import os
import sys
import django
from pathlib import Path

def test_django_setup():
    """Test Django setup and basic imports."""
    print("🧪 Testing Django Setup...")
    
    try:
        # Add project root to Python path
        project_root = Path(__file__).parent
        sys.path.insert(0, str(project_root))
        
        # Set Django settings
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'settings')
        django.setup()
        
        print("✅ Django setup successful")
        return True
        
    except Exception as e:
        print(f"❌ Django setup failed: {e}")
        return False

def test_imports():
    """Test importing key modules."""
    print("\n🧪 Testing Imports...")
    
    try:
        from resume_processor.models import Resume, JobPosting, RankingSession
        print("✅ Models imported successfully")
        
        from resume_processor.views import upload_resume, batch_upload_view
        print("✅ Views imported successfully")
        
        from resume_processor.batch_processor import BatchProcessor
        print("✅ BatchProcessor imported successfully")
        
        from resume_processor.enhanced_pdf_parser import PDFParser
        print("✅ PDFParser imported successfully")
        
        from resume_processor.llm_ranker import LLMRanker
        print("✅ LLMRanker imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_batch_processor():
    """Test BatchProcessor initialization."""
    print("\n🧪 Testing BatchProcessor...")
    
    try:
        from resume_processor.batch_processor import BatchProcessor
        
        processor = BatchProcessor()
        
        # Test configuration
        assert processor.section_weights['experience'] == 0.45
        assert processor.section_weights['skills'] == 0.35
        assert len(processor.skill_taxonomy) > 0
        
        print("✅ BatchProcessor initialized successfully")
        print(f"   - Section weights: {processor.section_weights}")
        print(f"   - Skill taxonomy entries: {len(processor.skill_taxonomy)}")
        
        return True
        
    except Exception as e:
        print(f"❌ BatchProcessor test failed: {e}")
        return False

def test_models():
    """Test model definitions."""
    print("\n🧪 Testing Models...")
    
    try:
        from resume_processor.models import Resume, JobPosting, RankingSession
        
        # Test Resume model
        resume = Resume()
        assert hasattr(resume, 'candidate_id')
        assert hasattr(resume, 'filename')
        assert hasattr(resume, 'parsed_data')
        
        # Test JobPosting model
        job = JobPosting()
        assert hasattr(job, 'job_id')
        assert hasattr(job, 'title')
        assert hasattr(job, 'requirements')
        
        # Test RankingSession model
        session = RankingSession()
        assert hasattr(session, 'session_id')
        assert hasattr(session, 'ranking_results')
        
        print("✅ Models defined correctly")
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def test_urls():
    """Test URL configuration."""
    print("\n🧪 Testing URLs...")
    
    try:
        from django.urls import reverse
        from django.test import Client
        
        client = Client()
        
        # Test basic URL patterns
        urls_to_test = [
            'resume_processor:upload_resume',
            'resume_processor:resume_list',
            'resume_processor:batch_upload',
        ]
        
        for url_name in urls_to_test:
            try:
                url = reverse(url_name)
                print(f"   ✅ {url_name}: {url}")
            except Exception as e:
                print(f"   ❌ {url_name}: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ URL test failed: {e}")
        return False

def test_settings():
    """Test Django settings."""
    print("\n🧪 Testing Settings...")
    
    try:
        from django.conf import settings
        
        # Check required settings
        assert hasattr(settings, 'MEDIA_ROOT')
        assert hasattr(settings, 'MEDIA_URL')
        assert hasattr(settings, 'STATIC_URL')
        assert 'resume_processor' in settings.INSTALLED_APPS
        
        print("✅ Django settings configured correctly")
        print(f"   - Media root: {settings.MEDIA_ROOT}")
        print(f"   - Media URL: {settings.MEDIA_URL}")
        print(f"   - Installed apps: {len(settings.INSTALLED_APPS)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Settings test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Resume Reviewer Project Test Suite")
    print("=" * 50)
    
    tests = [
        test_django_setup,
        test_imports,
        test_batch_processor,
        test_models,
        test_urls,
        test_settings,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Project is ready to use.")
        print("\nNext steps:")
        print("1. Run migrations: python manage.py migrate")
        print("2. Start server: python manage.py runserver")
        print("3. Test web interface: http://127.0.0.1:8000/resume/batch-upload/")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 