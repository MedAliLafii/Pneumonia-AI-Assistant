#!/usr/bin/env python3
"""
Startup script for Pneumonia AI Assistant.
Checks dependencies and components before launching the application.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required. Current version: {version.major}.{version.minor}")
        return False
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def check_dependencies():
    """Check if required packages are installed."""
    print("\n📦 Checking dependencies...")
    required_packages = [
        'flask', 'tensorflow', 'langchain', 'sentence_transformers',
        'opencv-python', 'pillow', 'numpy'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - not installed")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    return True

def check_model_file():
    """Check if the CNN model file exists."""
    print("\n🤖 Checking CNN model...")
    model_path = Path("models/improved_pneumonia_cnn.h5")
    if not model_path.exists():
        print(f"❌ CNN model not found: {model_path}")
        print("Please ensure the model file is in the models/ directory")
        return False
    print(f"✅ CNN model found: {model_path}")
    return True

def check_env_file():
    """Check if environment file exists."""
    print("\n🔑 Checking environment configuration...")
    env_path = Path(".env")
    if not env_path.exists():
        print("⚠️  .env file not found")
        print("Please create a .env file with your GEMINI_API_KEY")
        print("Example: GEMINI_API_KEY=your_api_key_here")
        return False
    print("✅ .env file found")
    return True

def check_faiss_index():
    """Check if FAISS index exists."""
    print("\n📚 Checking knowledge base...")
    faiss_dir = Path("faiss_index")
    if not faiss_dir.exists():
        print("⚠️  FAISS index not found")
        print("The application will create a new index from PDF data on first run")
        return True  # Not critical, will be created automatically
    print("✅ FAISS index found")
    return True

def run_tests():
    """Run basic tests."""
    print("\n🧪 Running basic tests...")
    try:
        result = subprocess.run([sys.executable, "test_cnn.py"], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("✅ CNN model test passed")
            return True
        else:
            print("❌ CNN model test failed")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        return False

def main():
    """Main startup function."""
    print("🚀 Pneumonia AI Assistant - Startup Check")
    print("=" * 50)
    
    checks = [
        check_python_version(),
        check_dependencies(),
        check_model_file(),
        check_env_file(),
        check_faiss_index(),
        run_tests()
    ]
    
    print("\n" + "=" * 50)
    
    if all(checks):
        print("✅ All checks passed! Starting application...")
        print("\n🌐 The application will be available at: http://localhost:8080")
        print("📱 Use the Chat tab for questions and X-Ray Analysis tab for image uploads")
        print("\nPress Ctrl+C to stop the application")
        
        # Start the Flask application
        os.system(f"{sys.executable} app.py")
    else:
        print("❌ Some checks failed. Please fix the issues above before starting.")
        sys.exit(1)

if __name__ == "__main__":
    main()
