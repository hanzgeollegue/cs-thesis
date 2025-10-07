#!/usr/bin/env python3
"""
Compatibility test script to check dependencies and versions.
Run this to diagnose any compatibility issues.
"""

import sys
import importlib

def test_import(module_name, min_version=None):
    """Test if a module can be imported and check version."""
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {module_name}: {version}")
        
        if min_version and version != 'unknown':
            try:
                from packaging import version as pkg_version
                if pkg_version.parse(version) < pkg_version.parse(min_version):
                    print(f"   ⚠️  Version {version} is below recommended {min_version}")
                else:
                    print(f"   ✅ Version {version} meets requirements")
            except ImportError:
                print(f"   ℹ️  Cannot check version compatibility (packaging not available)")
        
        return True
    except ImportError as e:
        print(f"❌ {module_name}: Import failed - {e}")
        return False
    except Exception as e:
        print(f"❌ {module_name}: Error - {e}")
        return False

def test_scikit_learn_compatibility():
    """Test scikit-learn specific compatibility."""
    print("\n🧪 Testing scikit-learn compatibility...")
    
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Test basic TfidfVectorizer initialization
        print("   Testing TfidfVectorizer initialization...")
        
        # Test with minimal parameters
        try:
            vectorizer = TfidfVectorizer()
            print("   ✅ Basic TfidfVectorizer() works")
        except Exception as e:
            print(f"   ❌ Basic TfidfVectorizer() failed: {e}")
            return False
        
        # Test with common parameters
        try:
            vectorizer = TfidfVectorizer(max_features=1000)
            print("   ✅ TfidfVectorizer(max_features=1000) works")
        except Exception as e:
            print(f"   ❌ TfidfVectorizer(max_features=1000) failed: {e}")
        
        # Test with ngram_range
        try:
            vectorizer = TfidfVectorizer(ngram_range=(1, 2))
            print("   ✅ TfidfVectorizer(ngram_range=(1, 2)) works")
        except Exception as e:
            print(f"   ❌ TfidfVectorizer(ngram_range=(1, 2)) failed: {e}")
        
        # Test with stop_words (might not be available in older versions)
        try:
            vectorizer = TfidfVectorizer(stop_words='english')
            print("   ✅ TfidfVectorizer(stop_words='english') works")
        except Exception as e:
            print(f"   ⚠️  TfidfVectorizer(stop_words='english') not available: {e}")
        
        # Test with random_state (added in newer versions)
        try:
            vectorizer = TfidfVectorizer(random_state=42)
            print("   ✅ TfidfVectorizer(random_state=42) works")
        except Exception as e:
            print(f"   ⚠️  TfidfVectorizer(random_state=42) not available: {e}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ scikit-learn compatibility test failed: {e}")
        return False

def test_dataclasses():
    """Test dataclasses compatibility."""
    print("\n🧪 Testing dataclasses compatibility...")
    
    try:
        from dataclasses import dataclass, asdict
        
        @dataclass
        class TestClass:
            name: str
            value: int
        
        test_obj = TestClass("test", 42)
        result = asdict(test_obj)
        
        if result == {"name": "test", "value": 42}:
            print("   ✅ dataclasses and asdict work correctly")
            return True
        else:
            print(f"   ❌ dataclasses/asdict returned unexpected result: {result}")
            return False
            
    except Exception as e:
        print(f"   ❌ dataclasses test failed: {e}")
        return False

def test_numpy():
    """Test numpy compatibility."""
    print("\n🧪 Testing numpy compatibility...")
    
    try:
        import numpy as np
        
        # Test basic numpy operations
        arr = np.array([1, 2, 3])
        if arr.sum() == 6:
            print("   ✅ numpy basic operations work")
            return True
        else:
            print("   ❌ numpy basic operations failed")
            return False
            
    except Exception as e:
        print(f"   ❌ numpy test failed: {e}")
        return False

def main():
    """Run all compatibility tests."""
    print("🔍 Resume Reviewer Compatibility Test")
    print("=" * 50)
    
    # Test basic imports
    print("🧪 Testing basic imports...")
    
    test_modules = [
        ("django", "4.2.0"),
        ("sklearn", "0.24.0"),
        ("numpy", "1.20.0"),
        ("pdfplumber", "0.9.0"),
        ("PIL", "9.5.0"),
        ("requests", "2.31.0"),
    ]
    
    import_success = 0
    for module, min_version in test_modules:
        if test_import(module, min_version):
            import_success += 1
    
    # Test specific compatibility
    print("\n🧪 Testing specific compatibility...")
    
    sklearn_ok = test_scikit_learn_compatibility()
    dataclasses_ok = test_dataclasses()
    numpy_ok = test_numpy()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Compatibility Test Results")
    print("=" * 50)
    
    print(f"Basic imports: {import_success}/{len(test_modules)} successful")
    print(f"scikit-learn compatibility: {'✅' if sklearn_ok else '❌'}")
    print(f"dataclasses compatibility: {'✅' if dataclasses_ok else '❌'}")
    print(f"numpy compatibility: {'✅' if numpy_ok else '❌'}")
    
    if import_success == len(test_modules) and sklearn_ok and dataclasses_ok and numpy_ok:
        print("\n🎉 All compatibility tests passed!")
        print("Your environment should work with the resume reviewer project.")
    else:
        print("\n⚠️  Some compatibility tests failed.")
        print("You may need to update some packages or check your environment.")
        
        if not sklearn_ok:
            print("\n💡 scikit-learn issues:")
            print("   - Try: pip install --upgrade scikit-learn")
            print("   - Minimum version: 0.24.0")
            print("   - The project will use fallback scoring if TF-IDF fails")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 