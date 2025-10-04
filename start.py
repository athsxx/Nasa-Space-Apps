#!/usr/bin/env python3
"""Quick start script for Air Quality Monitoring Agent."""

import subprocess
import sys
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"Current version: Python {sys.version}")
        sys.exit(1)
    print(f"✅ Python version: {sys.version.split()[0]}")

def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        sys.exit(1)

def run_tests():
    """Run basic tests."""
    print("🧪 Running tests...")
    try:
        subprocess.check_call([sys.executable, "-m", "unittest", "discover", "tests", "-v"])
        print("✅ All tests passed")
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Some tests failed: {e}")
        print("Continuing with setup...")

def create_env_file():
    """Create .env file if it doesn't exist."""
    env_file = Path(".env")
    example_file = Path(".env.example")
    
    if not env_file.exists() and example_file.exists():
        print("📄 Creating .env file...")
        with open(example_file, 'r') as src, open(env_file, 'w') as dst:
            dst.write(src.read())
        print("✅ .env file created (please configure as needed)")

def start_server():
    """Start the FastAPI server."""
    print("🚀 Starting Air Quality Monitoring Agent...")
    print("📚 API Documentation will be available at: http://localhost:8000/docs")
    print("🏠 Home page will be available at: http://localhost:8000")
    print("\n💡 Press Ctrl+C to stop the server")
    
    try:
        subprocess.check_call([sys.executable, "main.py"])
    except KeyboardInterrupt:
        print("\n👋 Server stopped. Goodbye!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start server: {e}")
        sys.exit(1)

def main():
    """Main setup and start process."""
    print("🌍 Air Quality Monitoring Agent - Quick Start")
    print("=" * 50)
    
    # Check Python version
    check_python_version()
    
    # Change to script directory
    script_dir = Path(__file__).parent.absolute()
    os.chdir(script_dir)
    print(f"📁 Working directory: {script_dir}")
    
    # Install dependencies
    install_dependencies()
    
    # Create environment file
    create_env_file()
    
    # Run tests (optional)
    if "--skip-tests" not in sys.argv:
        run_tests()
    
    # Start server
    if "--no-server" not in sys.argv:
        start_server()
    else:
        print("✅ Setup complete! Run 'python main.py' to start the server.")

if __name__ == "__main__":
    main()