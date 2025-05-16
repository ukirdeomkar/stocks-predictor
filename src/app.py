"""
Main Streamlit application file for the Indian Stock Market Analyzer
"""

import streamlit as st
import os
from dotenv import load_dotenv
import importlib
import sys

# Fix for torch path issues with Streamlit's hot-reloading
try:
    # Only try to reload if the module exists
    if 'torch.classes' in sys.modules and sys.modules['torch.classes'] is not None:
        importlib.reload(sys.modules['torch.classes'])
except (ModuleNotFoundError, KeyError, AttributeError):
    # Silently continue if module can't be found or reloaded
    pass

# Import components and utilities
from .components.header import create_header, add_footer
from .components.sidebar import create_sidebar
from .components.stock_analysis import create_stock_analysis_tab
from .components.stock_comparison import create_stock_comparison_tab
from .components.portfolio_optimization import create_portfolio_optimization_tab
from .components.help_tab import create_help_tab
from .components.portfolio_allocation import show_portfolio_allocation_section
from .utils.session_state import initialize_session_state

# Load environment variables
load_dotenv()

def setup_page():
    """Set up the page configuration and styles"""
    st.set_page_config(
        page_title="Indian Stock Market Analyzer",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Add custom CSS
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            color: #1E88E5;
            text-align: center;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.5rem;
            color: #0D47A1;
            margin-top: 2rem;
            margin-bottom: 1rem;
        }
        .info-box {
            background-color: #E3F2FD;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .recommendation-box {
            background-color: #E8F5E9;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-top: 1rem;
            margin-bottom: 1rem;
            color: #263238 !important; 
            font-weight: normal !important;
        }
        .recommendation-box p {
            color: #263238 !important;
            margin-bottom: 0.8rem;
        }
        .recommendation-box h1, .recommendation-box h2, .recommendation-box h3, 
        .recommendation-box h4, .recommendation-box h5 {
            color: #0D47A1 !important;
            margin-top: 1rem;
            margin-bottom: 0.5rem;
        }
        .recommendation-box ul, .recommendation-box ol {
            color: #263238 !important;
            margin-bottom: 0.8rem;
            padding-left: 1.5rem;
        }
        .allocation-box {
            background-color: #E8F5E9;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-top: 1rem;
            margin-bottom: 1rem;
            color: #263238 !important;
        }
        /* Streamlit radio button styling */
        div.stRadio > div {
            padding-top: 0.3rem;
            padding-bottom: 0.3rem;
        }
        div.stRadio label {
            font-weight: bold;
            font-size: 1.05rem;
        }
        /* Reduce spacing between elements */
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)


def run_app():
    """Run the Streamlit application"""
    # Setup page configuration
    setup_page()
    
    # Initialize session state
    initialize_session_state()
    
    # Create header
    create_header()
    
    # Create sidebar
    selected_model = create_sidebar()
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Single Stock Analysis", 
        "🔄 Stock Comparison", 
        "💼 Portfolio Optimization",
        "ℹ️ Help"
    ])
    
    # Create content for tabs
    with tab1:
        create_stock_analysis_tab(selected_model)
    
    with tab2:
        create_stock_comparison_tab()
    
    with tab3:
        create_portfolio_optimization_tab(selected_model)
    
    with tab4:
        create_help_tab()
    
    # Show portfolio allocation section if applicable
    show_portfolio_allocation_section(selected_model)
    
    # Add footer
    add_footer()


if __name__ == "__main__":
    run_app() 