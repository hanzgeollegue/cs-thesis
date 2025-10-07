#!/usr/bin/env python3
"""
OpenAI API Setup Script for Resume Reviewer
This script helps you configure your OpenAI API key for LLM ranking functionality.
"""

import os
import sys
import getpass
from pathlib import Path

def setup_openai_api():
    """Interactive setup for OpenAI API key."""
    print("🔑 OpenAI API Setup for Resume Reviewer")
    print("=" * 50)
    
    print("\nTo enable AI-powered resume ranking, you need an OpenAI API key.")
    print("This will allow the system to use GPT models for intelligent candidate evaluation.")
    
    # Check if API key already exists
    current_key = os.getenv('OPENAI_API_KEY', '')
    if current_key:
        print(f"\n✅ API key already configured: {current_key[:8]}...{current_key[-4:]}")
        choice = input("Do you want to update it? (y/n): ").lower().strip()
        if choice != 'y':
            print("Keeping existing API key.")
            return
    
    print("\n📋 Steps to get your OpenAI API key:")
    print("1. Go to https://platform.openai.com/api-keys")
    print("2. Sign in or create an account")
    print("3. Click 'Create new secret key'")
    print("4. Copy the key (starts with 'sk-')")
    print("5. Keep it secure - you won't see it again!")
    
    print("\n⚠️  Important:")
    print("- Never share your API key publicly")
    print("- The key will be stored in environment variables")
    print("- You can also add it directly to config.py")
    
    # Get API key from user
    while True:
        api_key = getpass.getpass("\n🔑 Enter your OpenAI API key (will be hidden): ").strip()
        
        if not api_key:
            print("❌ API key cannot be empty. Please try again.")
            continue
        
        if not api_key.startswith('sk-'):
            print("❌ Invalid API key format. Should start with 'sk-'")
            continue
        
        if len(api_key) < 20:
            print("❌ API key seems too short. Please check and try again.")
            continue
        
        break
    
    # Verify the key format
    print(f"\n✅ API key format looks valid: {api_key[:8]}...{api_key[-4:]}")
    
    # Setup options
    print("\n🔧 Setup Options:")
    print("1. Set environment variable (recommended)")
    print("2. Add to config.py file")
    print("3. Both")
    
    while True:
        choice = input("Choose option (1-3): ").strip()
        if choice in ['1', '2', '3']:
            break
        print("Please enter 1, 2, or 3.")
    
    # Option 1: Environment variable
    if choice in ['1', '3']:
        setup_environment_variable(api_key)
    
    # Option 2: Config file
    if choice in ['2', '3']:
        setup_config_file(api_key)
    
    print("\n🎉 Setup completed!")
    print("\nTo test your configuration:")
    print("1. Restart your terminal/command prompt")
    print("2. Run: python test_compatibility.py")
    print("3. Start the server: python manage.py runserver")
    print("4. Visit: http://127.0.0.1:8000")

def setup_environment_variable(api_key):
    """Set up environment variable for the current session."""
    print("\n🔧 Setting environment variable...")
    
    # Set for current session
    os.environ['OPENAI_API_KEY'] = api_key
    
    # Platform-specific setup
    if sys.platform.startswith('win'):
        setup_windows_env(api_key)
    else:
        setup_unix_env(api_key)
    
    print("✅ Environment variable set for current session")

def setup_windows_env(api_key):
    """Set up Windows environment variable permanently."""
    try:
        import winreg
        
        print("🔧 Setting permanent Windows environment variable...")
        
        # Set for current user
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_SET_VALUE)
        winreg.SetValueEx(key, "OPENAI_API_KEY", 0, winreg.REG_SZ, api_key)
        winreg.CloseKey(key)
        
        print("✅ Windows environment variable set permanently")
        print("💡 You may need to restart your terminal for changes to take effect")
        
    except ImportError:
        print("⚠️  Could not set permanent Windows environment variable")
        print("💡 Please set it manually in System Properties > Environment Variables")
    except Exception as e:
        print(f"⚠️  Error setting Windows environment variable: {e}")

def setup_unix_env(api_key):
    """Set up Unix/Linux/macOS environment variable permanently."""
    home = Path.home()
    shell_rc = None
    
    # Find shell configuration file
    for rc_file in ['.bashrc', '.zshrc', '.bash_profile', '.profile']:
        if (home / rc_file).exists():
            shell_rc = home / rc_file
            break
    
    if shell_rc:
        print(f"🔧 Adding to {shell_rc}...")
        
        # Check if already exists
        with open(shell_rc, 'r') as f:
            content = f.read()
        
        if f'OPENAI_API_KEY={api_key}' in content:
            print("✅ API key already in shell configuration")
        else:
            # Add export statement
            with open(shell_rc, 'a') as f:
                f.write(f'\n# OpenAI API Key for Resume Reviewer\nexport OPENAI_API_KEY="{api_key}"\n')
            
            print(f"✅ Added to {shell_rc}")
            print("💡 Run 'source ~/.bashrc' or restart your terminal")
    else:
        print("⚠️  Could not find shell configuration file")
        print("💡 Please add manually: export OPENAI_API_KEY='your-key-here'")

def setup_config_file(api_key):
    """Add API key to config.py file."""
    config_path = Path("resume_processor/config.py")
    
    if not config_path.exists():
        print("❌ config.py not found. Please run this script from the resume_reviewer directory.")
        return
    
    print(f"🔧 Adding API key to {config_path}...")
    
    try:
        with open(config_path, 'r') as f:
            content = f.read()
        
        # Check if already exists
        if f"OPENAI_API_KEY = '{api_key}'" in content:
            print("✅ API key already in config.py")
            return
        
        # Replace the placeholder
        if "OPENAI_API_KEY = ''" in content:
            content = content.replace("OPENAI_API_KEY = ''", f"OPENAI_API_KEY = '{api_key}'")
        elif "OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')" in content:
            # Add fallback
            content = content.replace(
                "OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')",
                f"OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '{api_key}')"
            )
        else:
            print("⚠️  Could not find API key configuration in config.py")
            return
        
        # Write back
        with open(config_path, 'w') as f:
            f.write(content)
        
        print("✅ API key added to config.py")
        
    except Exception as e:
        print(f"❌ Error updating config.py: {e}")

def test_api_key():
    """Test if the API key is working."""
    print("\n🧪 Testing API key...")
    
    try:
        import requests
        
        api_key = os.getenv('OPENAI_API_KEY', '')
        if not api_key:
            print("❌ No API key found in environment")
            return False
        
        # Test with a simple API call
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 10
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=10
        )
        
        if response.status_code == 200:
            print("✅ API key is working!")
            return True
        else:
            print(f"❌ API test failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except ImportError:
        print("⚠️  requests library not available. Install with: pip install requests")
        return False
    except Exception as e:
        print(f"❌ Error testing API: {e}")
        return False

def main():
    """Main setup function."""
    try:
        setup_openai_api()
        
        # Test the API key
        if test_api_key():
            print("\n🎉 Everything is set up correctly!")
            print("You can now use AI-powered resume ranking!")
        else:
            print("\n⚠️  API key test failed. Please check your configuration.")
        
    except KeyboardInterrupt:
        print("\n\n❌ Setup cancelled by user.")
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        print("Please check your configuration and try again.")

if __name__ == "__main__":
    main() 