"""
Verify that all dependencies are installed correctly.
Run this script to check if your environment is set up properly.
"""

import sys
import importlib.util
import os
import platform

def check_module(module_name):
    """Check if a module is installed."""
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        print(f"❌ {module_name} is NOT installed")
        return False
    else:
        print(f"✅ {module_name} is installed")
        return True

def check_system_requirements():
    """Check if the system meets the minimum requirements for running the LLM."""
    # Check CPU
    print("Checking system specifications:")
    
    # Check OS
    os_name = platform.system()
    os_version = platform.version()
    print(f"✅ OS: {os_name} {os_version}")
    
    # Check Python version
    python_version = platform.python_version()
    print(f"✅ Python version: {python_version}")
    
    # Check if CUDA is available (if torch is installed)
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA is available - Version: {torch.version.cuda}")
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
            print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("⚠️ CUDA is not available - model will run on CPU (slower)")
    except ImportError:
        print("⚠️ Torch is not installed yet - cannot check CUDA availability")
    
    # Check memory
    try:
        import psutil
        total_ram = psutil.virtual_memory().total / (1024**3)  # Convert to GB
        print(f"✅ System RAM: {total_ram:.2f} GB")
        
        if total_ram < 16:
            print("⚠️ WARNING: Less than 16GB RAM detected. The Deepseek model may not run properly.")
            print("   Consider modifying the MODEL_ID in utils.py to use a smaller model.")
    except ImportError:
        print("⚠️ Cannot check RAM - psutil module not installed")

def main():
    """Main function to check dependencies."""
    print("Checking dependencies for Indian Stock Market Analyzer...")
    print("-" * 50)
    
    # List of required modules
    required_modules = [
        "streamlit",
        "yfinance",
        "pandas",
        "numpy",
        "matplotlib",
        "plotly",
        "transformers",
        "torch",
        "huggingface_hub",
        "accelerate",
        "dotenv"
    ]
    
    # Check each module
    all_installed = True
    for module in required_modules:
        if not check_module(module):
            all_installed = False
    
    print("-" * 50)
    
    # Check system requirements
    check_system_requirements()
    
    print("-" * 50)
    
    # Final verdict
    if all_installed:
        print("✅ All dependencies are installed correctly!")
        print("You can run the app with: streamlit run app.py")
    else:
        print("❌ Some dependencies are missing or not configured correctly.")
        print("Please install the missing dependencies with: pip install -r requirements.txt")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 