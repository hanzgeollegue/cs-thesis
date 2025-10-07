#!/usr/bin/env python3
"""
Test script to verify SBERT integration in the resume processing pipeline.
"""

import os
import sys
import logging

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_sbert_import():
    """Test if SBERT can be imported and initialized."""
    try:
        from resume_processor.text_processor import SemanticEmbedding
        logger.info("✅ Successfully imported SemanticEmbedding")
        
        # Test initialization
        semantic_embedding = SemanticEmbedding()
        if semantic_embedding.model_available:
            logger.info("✅ SBERT model loaded successfully")
            logger.info(f"Model available: {semantic_embedding.model_available}")
            return True
        else:
            logger.warning("⚠️ SBERT model not available (sentence-transformers not installed?)")
            return False
            
    except ImportError as e:
        logger.error(f"❌ Failed to import SemanticEmbedding: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Error initializing SBERT: {e}")
        return False

def test_batch_processor_sbert():
    """Test if BatchProcessor can initialize with SBERT."""
    try:
        from resume_processor.batch_processor import BatchProcessor
        logger.info("✅ Successfully imported BatchProcessor")
        
        # Test initialization
        processor = BatchProcessor()
        if hasattr(processor, 'semantic_embedding'):
            logger.info("✅ BatchProcessor has semantic_embedding attribute")
            if processor.semantic_embedding and processor.semantic_embedding.model_available:
                logger.info("✅ SBERT is available in BatchProcessor")
                return True
            else:
                logger.warning("⚠️ SBERT not available in BatchProcessor")
                return False
        else:
            logger.error("❌ BatchProcessor missing semantic_embedding attribute")
            return False
            
    except ImportError as e:
        logger.error(f"❌ Failed to import BatchProcessor: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Error initializing BatchProcessor: {e}")
        return False

def test_semantic_scoring():
    """Test semantic scoring functionality."""
    try:
        from resume_processor.text_processor import SemanticEmbedding
        
        semantic_embedding = SemanticEmbedding()
        if not semantic_embedding.model_available:
            logger.warning("⚠️ Skipping semantic scoring test - SBERT not available")
            return False
        
        # Test with sample texts
        text1 = "Python developer with machine learning experience"
        text2 = "Software engineer skilled in Python and AI"
        
        # Generate embeddings
        emb1 = semantic_embedding.model.encode(text1)
        emb2 = semantic_embedding.model.encode(text2)
        
        # Calculate similarity
        similarity = semantic_embedding.calculate_semantic_similarity(emb1, emb2)
        
        logger.info(f"✅ Semantic similarity test completed: {similarity:.3f}")
        logger.info(f"Embedding dimensions: {emb1.shape}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error in semantic scoring test: {e}")
        return False

def main():
    """Run all SBERT integration tests."""
    logger.info("🧪 Testing SBERT Integration")
    logger.info("=" * 50)
    
    tests = [
        ("SBERT Import Test", test_sbert_import),
        ("BatchProcessor SBERT Test", test_batch_processor_sbert),
        ("Semantic Scoring Test", test_semantic_scoring)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n🔍 Running {test_name}")
        logger.info("-" * 30)
        try:
            if test_func():
                passed += 1
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.info(f"❌ {test_name}: FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name}: ERROR - {e}")
    
    logger.info("\n" + "=" * 50)
    logger.info(f"🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All SBERT integration tests passed!")
        logger.info("\n💡 Next steps:")
        logger.info("1. Start Django server: python manage.py runserver")
        logger.info("2. Upload resumes and check for three scores:")
        logger.info("   - tfidf_section_score (Section-Aware TF-IDF)")
        logger.info("   - tfidf_taxonomy_score (Skill-Cluster TF-IDF)")
        logger.info("   - semantic_score (SBERT embeddings)")
    else:
        logger.warning("⚠️ Some tests failed. Check the errors above.")
        logger.info("\n🔧 Troubleshooting:")
        logger.info("1. Install sentence-transformers: pip install sentence-transformers")
        logger.info("2. Check if you're in the correct directory")
        logger.info("3. Verify Python version compatibility")

if __name__ == "__main__":
    main()
