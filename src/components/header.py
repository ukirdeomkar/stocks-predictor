"""
Header and footer components for the application
"""

import streamlit as st

def create_header():
    """Create the header for the application"""
    # Header
    st.markdown('<h1 class="main-header">Indian Stock Market Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('Analyze BSE and NSE stocks with AI-powered insights to maximize your investment profits')

def add_footer():
    """Add a footer to the application"""
    st.markdown("""
    ---
    Made with ❤️ using Streamlit, yfinance, and open-source LLMs | Indian Stock Market Analyzer
    """) 