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
            
            # Recommend models based on GPU memory
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            if gpu_memory >= 12:
                print("✅ GPU Memory is sufficient for all models")
            elif gpu_memory >= 8:
                print("✅ GPU Memory is sufficient for most models except the largest (deepseek-coder-6.7b)")
            elif gpu_memory >= 4:
                print("✅ GPU Memory is sufficient for smaller models (phi-2, tinyllama-1.1b, deepseek-coder-1.3b)")
            else:
                print("⚠️ Limited GPU Memory - only the smallest models may work efficiently")
        else:
            print("⚠️ CUDA is not available - models will run on CPU (slower)")
    except ImportError:
        print("⚠️ Torch is not installed yet - cannot check CUDA availability")
    
    # Check memory
    try:
        import psutil
        total_ram = psutil.virtual_memory().total / (1024**3)  # Convert to GB
        print(f"✅ System RAM: {total_ram:.2f} GB")
        
        # Recommend models based on RAM
        if total_ram < 4:
            print("⚠️ WARNING: Less than 4GB RAM detected. LLMs may not run properly.")
            print("   Consider upgrading your system or using a cloud-based deployment.")
        elif total_ram < 6:
            print("⚠️ RAM is suitable for only the smallest models (tinyllama-1.1b)")
            print("   Recommended model: tinyllama-1.1b")
        elif total_ram < 12:
            print("✅ RAM is suitable for small to medium-sized models")
            print("   Recommended models: phi-2, tinyllama-1.1b, deepseek-coder-1.3b, stablelm-zephyr-3b")
        elif total_ram < 16:
            print("✅ RAM is suitable for most models")
            print("   All models should work, but the largest ones may be slower")
        else:
            print("✅ RAM is suitable for all models")
            print("   You can use any model, including the largest ones")
    except ImportError:
        print("⚠️ Cannot check RAM - psutil module not installed")

def print_available_models():
    """Print information about available models."""
    print("\nAvailable AI Models:")
    print("-" * 30)
    
    models = [
        {"name": "phi-2", "size": "2.7B", "ram": "6GB+", "description": "Microsoft's efficient model"},
        {"name": "tinyllama-1.1b", "size": "1.1B", "ram": "4GB+", "description": "Very small general model"},
        {"name": "deepseek-coder-1.3b", "size": "1.3B", "ram": "4GB+", "description": "Lightweight coding model"},
        {"name": "stablelm-zephyr-3b", "size": "3B", "ram": "6GB+", "description": "Stability AI's model"},
        {"name": "llama3-8b", "size": "8B", "ram": "16GB+", "description": "Meta's Llama 3 model"},
        {"name": "deepseek-coder-6.7b", "size": "6.7B", "ram": "16GB+", "description": "Powerful coding model"}
    ]
    
    for model in models:
        print(f"✓ {model['name']} ({model['size']} parameters)")
        print(f"  • System Requirements: {model['ram']} RAM")
        print(f"  • Description: {model['description']}")
        print()

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
    
    # Print model information
    print_available_models()
    
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