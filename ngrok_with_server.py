#!/usr/bin/env python3
"""
Launch script for Django Resume Reviewer with PUBLIC internet access.
Uses pyngrok to create a public URL for worldwide access.
No manual venv activation needed - everything handled automatically.
"""

import subprocess
import sys
import os
import time
import signal
from pathlib import Path

# Configuration
SCRIPT_DIR = Path(__file__).parent.absolute()
VENV_PYTHON = SCRIPT_DIR / "venv311" / "Scripts" / "python.exe"
MANAGE_PATH = SCRIPT_DIR / "resume_reviewer" / "manage.py"
LOCAL_PORT = 8000

def install_pyngrok():
    """Install pyngrok if not already installed."""
    try:
        import pyngrok
        return True
    except ImportError:
        print("📦 Installing pyngrok for public access...")
        try:
            subprocess.check_call([str(VENV_PYTHON), "-m", "pip", "install", "pyngrok"])
            print("✅ pyngrok installed successfully!")
            return True
        except Exception as e:
            print(f"❌ Failed to install pyngrok: {e}")
            return False

def start_django_server():
    """Start Django server using venv python (equivalent to activate.bat + runserver)"""
    print(f"📍 Using Python: {VENV_PYTHON}")
    print(f"📍 Starting Django on port {LOCAL_PORT}...")
    
    return subprocess.Popen(
        [str(VENV_PYTHON), str(MANAGE_PATH), "runserver", f"127.0.0.1:{LOCAL_PORT}"],
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
    )

def create_ngrok_tunnel():
    """Create ngrok tunnel to expose Django server worldwide"""
    try:
        from pyngrok import ngrok
        print("\n🌐 Creating public tunnel with ngrok...")
        
        # Connect with bind_tls=True for HTTPS
        public_url = ngrok.connect(LOCAL_PORT, bind_tls=True)
        return public_url, ngrok
        
    except Exception as e:
        if "authentication failed" in str(e) or "authtoken" in str(e):
            print("\n❌ ngrok requires authentication!")
            print("\n📝 Quick setup (one-time only):")
            print("  1. Sign up (free): https://dashboard.ngrok.com/signup")
            print("  2. Get your token: https://dashboard.ngrok.com/get-started/your-authtoken")
            print(f"  3. Run this command:")
            print(f"     {VENV_PYTHON} -m pyngrok config add-authtoken YOUR_TOKEN_HERE")
            print("\n💡 After setup, run this script again!")
            sys.exit(1)
        else:
            raise

def main():
    """Main launcher function."""
    print("🌍 Starting Django Resume Reviewer with PUBLIC ACCESS...")
    print("=" * 60)
    
    # Install pyngrok if needed
    if not install_pyngrok():
        sys.exit(1)
    
    processes = []
    tunnels = []
    
    try:
        # Start Django server
        print("\n🔧 Starting Django server...")
        django_process = start_django_server()
        processes.append(("Django", django_process))
        print(f"✅ Django started (PID: {django_process.pid})")
        
        # Wait for Django to initialize
        print("⏳ Waiting for Django to initialize...")
        time.sleep(4)
        
        # Create ngrok tunnel
        public_url, ngrok_mod = create_ngrok_tunnel()
        tunnels.append(public_url)
        
        print("\n" + "=" * 60)
        print("🎉 APPLICATION IS NOW PUBLICLY ACCESSIBLE!")
        print("=" * 60)
        print("\n📱 SHARE THIS LINK WITH ANYONE (even outside your network):")
        print(f"\n🌍 Public URL:  {public_url}")
        print(f"🏠 Local URL:   http://localhost:{LOCAL_PORT}")
        print("\n💡 This URL works from ANYWHERE on the internet!")
        print("⚠️  Note: Free ngrok URLs are temporary and change each restart.")
        print("\n🛑 Press Ctrl+C to stop server and close tunnel...")
        print("=" * 60)
        
        # Keep running until interrupted
        try:
            while True:
                # Check if Django process died
                if django_process.poll() is not None:
                    print(f"\n⚠️ Django stopped unexpectedly (exit code: {django_process.returncode})")
                    break
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n\n🛑 Shutdown requested...")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Close ngrok tunnels
        if tunnels:
            print("\n🔌 Closing public tunnel...")
            try:
                from pyngrok import ngrok as ngrok_mod
                for tunnel in tunnels:
                    try:
                        ngrok_mod.disconnect(tunnel.public_url)
                    except:
                        pass
                ngrok_mod.kill()
            except:
                pass
        
        # Stop Django server
        print("🛑 Stopping Django server...")
        for name, process in processes:
            try:
                if sys.platform == "win32":
                    process.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    process.terminate()
                
                # Wait for graceful shutdown
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                
                print(f"✅ {name} stopped")
            except Exception as e:
                print(f"⚠️ Error stopping {name}: {e}")
        
        print("\n👋 All services stopped. Goodbye!")

if __name__ == "__main__":
    main()
