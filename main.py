"""
Indian Stock Market Analyzer - Main Entry Point
===============================================

This is the main entry point for the Indian Stock Market Analyzer application.
This file imports and runs the Streamlit app from the src directory.
"""

import os
import sys

# Add the src directory to Python path to make imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the app from src
from src.app import run_app

if __name__ == "__main__":
    # Run the Streamlit app
    run_app() 