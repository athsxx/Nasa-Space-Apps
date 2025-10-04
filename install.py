"""
Air Quality Monitoring Agent - Native Installation Script
Installs dependencies and sets up the environment for native execution.
"""

import subprocess
import sys
import os
import platform
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Error: {result.stderr}")
            return False
        else:
            print(f"✅ {description} completed successfully")
            return True
    except Exception as e:
        print(f"❌ Error running command: {e}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True


def install_requirements():
    """Install Python requirements."""
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found")
        return False
    
    commands = [
        "pip install --upgrade pip",
        "pip install -r requirements.txt"
    ]
    
    for cmd in commands:
        if not run_command(cmd, f"Running: {cmd}"):
            return False
    return True


def create_environment_file():
    """Create a .env file with default settings."""
    env_content = """# Air Quality Monitoring Agent Configuration
# OpenAQ API Configuration
OPENAQ_API_KEY=
OPENAQ_BASE_URL=https://api.openaq.org/v2

# TEMPO Data Configuration (NASA Earthdata credentials required)
TEMPO_USERNAME=
TEMPO_PASSWORD=

# Location Service Configuration
DEFAULT_LOCATION_LAT=40.7128
DEFAULT_LOCATION_LON=-74.0060

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=true

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=logs/aqi_agent.log
"""
    
    if not os.path.exists(".env"):
        with open(".env", "w") as f:
            f.write(env_content)
        print("✅ Created .env configuration file")
    else:
        print("ℹ️  .env file already exists")


def create_logs_directory():
    """Create logs directory if it doesn't exist."""
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    print("✅ Created logs directory")


def run_tests():
    """Run basic tests to verify installation."""
    print("🧪 Running basic tests...")
    
    # Test core imports
    required_modules = [
        ("requests", "HTTP client library"),
        ("pandas", "Data analysis library"), 
        ("numpy", "Numerical computing library"),
        ("fastapi", "Web framework"),
        ("pydantic", "Data validation library")
    ]
    
    failed_imports = []
    for module_name, description in required_modules:
        try:
            __import__(module_name)
            print(f"✅ {module_name} ({description}) imported successfully")
        except ImportError as e:
            print(f"❌ {module_name} import error: {e}")
            failed_imports.append(module_name)
    
    if failed_imports:
        print(f"❌ Failed to import required modules: {', '.join(failed_imports)}")
        return False
    
    # Test basic functionality
    try:
        from aqi_agent import AQICalculator
        calculator = AQICalculator()
        aqi = calculator.calculate_aqi_for_pollutant("pm25", 25.0)
        print(f"✅ AQI calculation test passed (PM2.5=25.0 → AQI={aqi})")
    except Exception as e:
        print(f"❌ AQI calculation test failed: {e}")
        return False
    
    return True


def show_usage_instructions():
    """Show usage instructions."""
    print("\n" + "="*60)
    print("🚀 AIR QUALITY MONITORING AGENT - READY!")
    print("="*60)
    print("\n📋 Usage Instructions:")
    print("\n1. Start the Web API:")
    print("   python main.py")
    print("   Then visit: http://localhost:8000")
    print("\n2. Use the Command Line Interface:")
    print("   python cli.py --location \"New York, NY\"")
    print("   python cli.py --lat 40.7128 --lon -74.0060")
    print("   python cli.py --auto")
    print("\n3. Run tests:")
    print("   python -m pytest tests/")
    print("   python tests/test_aqi.py")
    print("\n📚 Documentation:")
    print("   - API Docs: http://localhost:8000/docs (after starting server)")
    print("   - README: Check README.md for detailed information")
    print("\n🔧 Configuration:")
    print("   - Edit .env file to configure API keys and settings")
    print("   - Check config/settings.py for advanced configuration")
    print("\n" + "="*60)


def main():
    """Main setup function."""
    print("🌍 Air Quality Monitoring Agent - Native Installation")
    print("=" * 55)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create necessary directories
    create_logs_directory()
    
    # Create environment file
    create_environment_file()
    
    # Install requirements
    if not install_requirements():
        print("❌ Failed to install requirements")
        sys.exit(1)
    
    # Run tests
    if not run_tests():
        print("⚠️  Some tests failed, but installation may still work")
    
    # Show usage instructions
    show_usage_instructions()
    
    print("\n✅ Installation completed successfully!")


if __name__ == "__main__":
    main()