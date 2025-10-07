#!/usr/bin/env python3
"""
Simple test script to verify the website functionality.
Run this after starting the Django server to test the website.
"""

import requests
import time
import webbrowser
from urllib.parse import urljoin

def test_website():
    """Test the website functionality."""
    base_url = "http://127.0.0.1:8000"
    
    print("🌐 Testing Resume Reviewer Website")
    print("=" * 50)
    
    # Test endpoints
    endpoints = [
        "/",                    # Landing page
        "/resume/upload/",      # Single upload
        "/resume/batch-upload/", # Batch upload
        "/resume/list/",        # Resume list
        "/admin/",              # Admin panel
    ]
    
    print("Testing endpoints...")
    for endpoint in endpoints:
        try:
            url = urljoin(base_url, endpoint)
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                print(f"✅ {endpoint}: OK (Status: {response.status_code})")
            else:
                print(f"⚠️  {endpoint}: Status {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            print(f"❌ {endpoint}: Connection failed - Is the server running?")
        except Exception as e:
            print(f"❌ {endpoint}: Error - {e}")
    
    print("\n" + "=" * 50)
    print("🎯 Website Test Summary")
    print("=" * 50)
    
    print("\nTo test the website:")
    print("1. Make sure Django server is running: python manage.py runserver")
    print("2. Open your browser and go to: http://127.0.0.1:8000")
    print("3. Test the batch upload functionality")
    print("4. Upload some sample PDF resumes")
    
    print("\nExpected behavior:")
    print("✅ Landing page shows batch upload form prominently")
    print("✅ Can upload up to 25 PDF resumes")
    print("✅ Job description input is pre-filled")
    print("✅ Modern, clean design with gradient background")
    print("✅ Responsive design works on mobile")
    
    print("\nTest URLs:")
    for endpoint in endpoints:
        print(f"   {urljoin(base_url, endpoint)}")
    
    # Try to open the website in browser
    try:
        print(f"\n🌐 Opening website in browser: {base_url}")
        webbrowser.open(base_url)
    except Exception as e:
        print(f"Could not open browser automatically: {e}")
        print(f"Please manually open: {base_url}")

def check_server_status():
    """Check if the Django server is running."""
    try:
        response = requests.get("http://127.0.0.1:8000", timeout=5)
        return True
    except:
        return False

def main():
    """Main test function."""
    print("🚀 Resume Reviewer Website Test")
    print("=" * 50)
    
    # Check if server is running
    if check_server_status():
        print("✅ Django server is running")
        test_website()
    else:
        print("❌ Django server is not running")
        print("\nTo start the server:")
        print("1. cd resume_reviewer")
        print("2. python manage.py runserver")
        print("3. Wait for server to start")
        print("4. Run this test again")
        
        # Wait a bit and check again
        print("\nWaiting 5 seconds and checking again...")
        time.sleep(5)
        
        if check_server_status():
            print("✅ Server is now running!")
            test_website()
        else:
            print("❌ Server still not running. Please start it manually.")

if __name__ == "__main__":
    main() 