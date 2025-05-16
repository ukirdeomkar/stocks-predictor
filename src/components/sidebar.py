"""
Sidebar component for the application
"""

import streamlit as st
import os
from ..models.model_info import get_available_models

def create_sidebar():
    """
    Create the sidebar for the application
    
    Returns:
        str: The selected model
    """
    # Get available models
    available_models = get_available_models()
    
    # Model configuration in sidebar
    st.sidebar.markdown('## AI Model Settings')
    
    # Create model selection with categories
    st.sidebar.markdown("### Select Analysis Model")
    
    # Model selection
    selected_model = st.sidebar.selectbox(
        "Model",
        options=list(available_models.keys()),
        index=list(available_models.keys()).index("phi-2") if "phi-2" in available_models else 0,
        format_func=lambda x: f"{x} - {available_models[x]['description']}"
    )
    
    # Show model details
    model_info = available_models[selected_model]
    st.sidebar.info(f"""
    **{selected_model}**  
    {model_info['description']}
    """)
    
    # Update session state
    if selected_model != st.session_state.selected_model:
        st.session_state.selected_model = selected_model
    
    # Model details expander
    with st.sidebar.expander("About Selected Model", expanded=False):
        st.markdown(f"""
        **Model**: {selected_model}
        
        **Description**: {model_info['description']}
        
        **Size**: {model_info['size']}
        
        This model is accessed through the Hugging Face API.
        """)
    
    # Model execution options
    st.sidebar.markdown("### Model Execution Options")
    
    # Check if Hugging Face API key is available
    hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
    api_available = hf_api_key is not None and hf_api_key.strip() != ""
    
    if api_available:
        st.sidebar.success("✅ Using Hugging Face API for analysis")
        st.sidebar.info("API analysis is faster and requires less resources")
    else:
        # API key not available
        st.sidebar.error(
            "❌ No Hugging Face API key found. You need to set up an API key to use the app."
        )
        
        # Show instructions for setting up API key
        with st.sidebar.expander("How to set up your API key", expanded=True):
            st.markdown("""
            1. Create a free account at [Hugging Face](https://huggingface.co/)
            2. Generate an API key from [your settings page](https://huggingface.co/settings/tokens)
            3. Create a `.env` file in the app's directory with:
               ```
               HUGGINGFACE_API_KEY=your_api_key_here
               ```
            4. Restart the application
            """)
        
        # Add a link to get an API key
        st.sidebar.markdown(
            "[Get a free Hugging Face API key](https://huggingface.co/settings/tokens)"
        )
    
    # Sidebar for stock selection and controls
    st.sidebar.markdown('## Stock Selection')
    
    # Add example tickers for user convenience
    st.sidebar.markdown("""
    #### Example Tickers:
    - BSE: RELIANCE.BO, TCS.BO, HDFCBANK.BO, INFY.BO, ICICIBANK.BO
    - NSE: RELIANCE.NS, TCS.NS, HDFCBANK.NS, INFY.NS, ICICIBANK.NS
    """)
    
    # Input for stock ticker
    st.session_state.ticker = st.sidebar.text_input(
        'Enter Stock Symbol (append .BO for BSE or .NS for NSE)',
        value=st.session_state.get('ticker', '')
    )
    
    # Time period selection
    time_periods = {
        '1 Week': '1wk',
        '1 Month': '1mo',
        '3 Months': '3mo',
        '6 Months': '6mo',
        '1 Year': '1y',
        '2 Years': '2y',
        '5 Years': '5y',
        'Year to Date': 'ytd'
    }
    selected_period = st.sidebar.selectbox('Select Time Period', list(time_periods.keys()))
    st.session_state.period = time_periods[selected_period]
    
    # Data interval selection
    intervals = ['1d', '5d', '1wk', '1mo']
    interval_names = {'1d': 'Daily', '5d': '5-Day', '1wk': 'Weekly', '1mo': 'Monthly'}
    selected_interval = st.sidebar.selectbox(
        'Select Data Interval', 
        [interval_names[i] for i in intervals]
    )
    st.session_state.interval = [k for k, v in interval_names.items() if v == selected_interval][0]
    
    return selected_model 