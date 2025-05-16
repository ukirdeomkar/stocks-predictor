"""
Session state utilities for managing Streamlit's session state
"""

import streamlit as st

def initialize_session_state():
    """Initialize all session state variables"""
    
    # Initialize model selection if not already set
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "phi-2"  # Default model
    
    # Initialize model loaded state
    if "model_loaded" not in st.session_state:
        st.session_state.model_loaded = False
    
    # Always use API - no toggle needed
    if "use_api" not in st.session_state:
        st.session_state.use_api = True
    
    # Initialize analysis results dictionary
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = {} 