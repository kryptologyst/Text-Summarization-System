#!/usr/bin/env python3
"""
Setup Script for Modern Text Summarization System
================================================

Automated setup and installation script.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python version {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_dependencies():
    """Install Python dependencies"""
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found")
        return False
    
    return run_command("pip install -r requirements.txt", "Installing Python dependencies")

def download_spacy_model():
    """Download spaCy English model"""
    return run_command("python -m spacy download en_core_web_sm", "Downloading spaCy English model")

def download_nltk_data():
    """Download required NLTK data"""
    commands = [
        "python -c \"import nltk; nltk.download('punkt')\"",
        "python -c \"import nltk; nltk.download('stopwords')\"",
        "python -c \"import nltk; nltk.download('punkt_tab')\""
    ]
    
    success = True
    for cmd in commands:
        if not run_command(cmd, f"Downloading NLTK data"):
            success = False
    
    return success

def create_directories():
    """Create necessary directories"""
    directories = ["logs", "data", "models"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    return True

def test_installation():
    """Test if installation was successful"""
    test_script = """
import sys
try:
    from modern_summarizer import ModernTextSummarizer, SummarizationConfig
    print("✅ Core modules imported successfully")
    
    config = SummarizationConfig(max_sentences=2, method="tfidf")
    summarizer = ModernTextSummarizer(config)
    print("✅ Summarizer initialized successfully")
    
    # Test with sample text
    sample_text = "This is a test sentence. This is another test sentence."
    summary, scores = summarizer.summarize(sample_text)
    print("✅ Summarization test completed successfully")
    
    print("🎉 Installation test passed!")
    
except Exception as e:
    print(f"❌ Installation test failed: {e}")
    sys.exit(1)
"""
    
    return run_command(f'python -c "{test_script}"', "Testing installation")

def main():
    """Main setup function"""
    print("🧠 Modern Text Summarization System Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Download spaCy model
    if not download_spacy_model():
        print("⚠️  Warning: spaCy model download failed. Some features may not work.")
    
    # Download NLTK data
    if not download_nltk_data():
        print("⚠️  Warning: NLTK data download failed. Some features may not work.")
    
    # Create directories
    create_directories()
    
    # Test installation
    if not test_installation():
        print("❌ Installation test failed")
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run the web interface: streamlit run app.py")
    print("2. Or run the CLI tool: python cli.py demo")
    print("3. Or run tests: pytest test_summarizer.py -v")
    print("\nFor more information, see README.md")

if __name__ == "__main__":
    main()
