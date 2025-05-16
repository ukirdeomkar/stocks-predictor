"""
A utility script to verify the setup for the Indian Stock Market Analyzer application.
Checks for required dependencies and API access.
"""

import os
import sys
import subprocess
import pkg_resources
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def check_module(module_name):
    """Check if a Python module is installed."""
    try:
        pkg_resources.get_distribution(module_name)
        return True
    except pkg_resources.DistributionNotFound:
        return False

def check_api_connection():
    """Verify Hugging Face API connectivity and authentication."""
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    
    if not api_key:
        print("❌ No Hugging Face API key found in environment variables")
        print("   You must set up the API key to use this application")
        print("\n   To set up your API key:")
        print("   1. Create a free account at https://huggingface.co/")
        print("   2. Generate an API key from https://huggingface.co/settings/tokens")
        print("   3. Create a .env file in the app's directory with: HUGGINGFACE_API_KEY=your_api_key_here")
        print("   4. Restart the application")
        return False
    
    # Verify API connectivity
    try:
        # Use a small model for the test
        test_model = "microsoft/phi-2"
        api_url = os.getenv("HUGGINGFACE_API_URL", f"https://api-inference.huggingface.co/models/{test_model}")
        
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {"inputs": "Hello, are you available?"}
        
        print("Testing API connection to Hugging Face...", end="")
        response = requests.post(api_url, headers=headers, json=payload, timeout=10)
        
        if response.status_code == 200:
            print(" ✅ Success!")
            return True
        elif response.status_code == 401:
            print(" ❌ Failed: Invalid API key")
            return False
        else:
            print(f" ❌ Failed with status code: {response.status_code}")
            print(f"   Message: {response.text}")
            return False
    except Exception as e:
        print(f" ❌ Connection error: {str(e)}")
        return False

def check_required_modules():
    """Check for all required Python modules."""
    required_modules = [
        "streamlit", "yfinance", "pandas", "numpy", "plotly", 
        "requests", "python-dotenv", "matplotlib"
    ]
    
    missing_modules = []
    for module in required_modules:
        if not check_module(module):
            missing_modules.append(module)
    
    if missing_modules:
        print("❌ Missing required modules:", ", ".join(missing_modules))
        print("   Install them with: pip install " + " ".join(missing_modules))
        return False
    else:
        print("✅ All required Python modules are installed")
        return True

def print_available_models():
    """Print information about available AI models on Hugging Face."""
    models = {
        "deepseek-coder-6.7b": {
            "description": "A specialized model for financial and code analysis",
            "best_for": "Detailed technical analysis"
        },
        "mistral-7b": {
            "description": "Well-balanced general-purpose model",
            "best_for": "Balanced analysis"
        },
        "phi-2": {
            "description": "Small but efficient model with good financial understanding",
            "best_for": "Quick insights"
        },
        "tinyllama-1.1b": {
            "description": "Ultra-compact model for very basic analysis",
            "best_for": "Very basic summaries"
        }
    }
    
    print("\n📊 Available AI Models (via Hugging Face API):")
    print("╔══════════════════╦══════════════════════════════════════════╦══════════════════════╗")
    print("║ Model            ║ Description                              ║ Best For             ║")
    print("╠══════════════════╬══════════════════════════════════════════╬══════════════════════╣")
    
    for model_name, info in models.items():
        name_col = model_name + " " * (16 - len(model_name))
        desc_col = info["description"] + " " * (40 - len(info["description"]))
        best_col = info["best_for"] + " " * (20 - len(info["best_for"]))
        print(f"║ {name_col} ║ {desc_col} ║ {best_col} ║")
    
    print("╚══════════════════╩══════════════════════════════════════════╩══════════════════════╝")

def main():
    """Main function to verify setup."""
    print("\n🔍 Verifying setup for Indian Stock Market Analyzer...\n")
    
    modules_ok = check_required_modules()
    
    api_ok = check_api_connection()
    
    if modules_ok and api_ok:
        print_available_models()
        print("\n✅ Setup verification complete. You're ready to analyze stocks!")
        print("\n   Run the application with: streamlit run app.py")
        return True
    else:
        print("\n❌ Setup verification failed. Please fix the issues above before running the application.")
        return False

if __name__ == "__main__":
    sys.exit(0 if main() else 1) 