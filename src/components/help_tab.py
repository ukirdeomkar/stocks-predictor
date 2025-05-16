"""
Help tab component for the application
"""

import streamlit as st
from ..models.model_info import get_available_models

def create_help_tab():
    """
    Create the help tab content with information and guidance
    """
    st.markdown('<h2 class="sub-header">Help & Information</h2>', unsafe_allow_html=True)
    
    # Stock symbol format explanation
    st.markdown("""
    ### Stock Symbol Format
    
    - For **BSE (Bombay Stock Exchange)** stocks, append **.BO** to the symbol
      - Example: `RELIANCE.BO`, `TCS.BO`, `HDFCBANK.BO`
    
    - For **NSE (National Stock Exchange)** stocks, append **.NS** to the symbol
      - Example: `RELIANCE.NS`, `TCS.NS`, `HDFCBANK.NS`
    """)
    
    # Features explanation
    st.markdown("""
    ### App Features
    
    1. **Single Stock Analysis**
       - Real-time market data
       - Technical indicators (RSI, MACD, Moving Averages, Bollinger Bands)
       - Stock price charts
       - AI-powered stock analysis and recommendations
    
    2. **Stock Comparison**
       - Compare multiple stocks side by side
       - Normalized price performance charts
       - Returns, volatility, and Sharpe ratio comparison
       - Risk vs. Return analysis
    
    3. **Portfolio Optimization**
       - AI-generated portfolio allocation recommendations
       - Risk assessment and diversification suggestions
       - Personalized investment strategy
    """)
    
    # Technical indicators explanation
    with st.expander("Understanding Technical Indicators"):
        st.markdown("""
        #### RSI (Relative Strength Index)
        - Measures the speed and change of price movements
        - RSI > 70: Potentially overbought
        - RSI < 30: Potentially oversold
        
        #### MACD (Moving Average Convergence Divergence)
        - Shows the relationship between two moving averages
        - MACD above Signal Line: Bullish signal
        - MACD below Signal Line: Bearish signal
        
        #### Moving Averages
        - 20-day MA: Short-term trend
        - 50-day MA: Medium-term trend
        - 200-day MA: Long-term trend
        
        #### Bollinger Bands
        - Consist of a middle band (20-day MA) and upper/lower bands (2 standard deviations)
        - Helps identify periods of high/low volatility
        - Price near upper band: Potentially overbought
        - Price near lower band: Potentially oversold
        """)
    
    # AI Model information
    with st.expander("About the AI Models"):
        st.markdown("""
        ### Available AI Models

        This application uses the Hugging Face Inference API to provide AI-powered stock analysis:
        
        **Model Options:**
        """)
        
        # List all available models with details
        available_models = get_available_models()
        for model_key, model_info in available_models.items():
            st.markdown(f"""
            #### {model_key}
            - **Description**: {model_info['description']}
            - **Size**: {model_info['size']}
            - **API Endpoint**: `{model_info['api_endpoint']}`
            """)
        
        st.markdown("""
        **Benefits of using the Hugging Face API:**
        - **Speed**: Get analysis in seconds instead of minutes
        - **No hardware requirements**: Run powerful AI models without a high-end GPU
        - **Generous free tier**: The Hugging Face API offers a free tier with reasonable quotas
        - **Access to powerful models**: Get analysis from various AI models without downloading them
        
        **API Setup:**
        The application requires a Hugging Face API key to function. You can get a free API key by:
        1. Creating an account at [Hugging Face](https://huggingface.co/)
        2. Generating an API key from your [settings page](https://huggingface.co/settings/tokens)
        3. Adding the key to a `.env` file in the app's directory
        """)
    
    # Tips for using AI analysis
    with st.expander("Tips for Using AI Analysis"):
        st.markdown("""
        1. **Choose the right model for your needs**:
           - Smaller models are faster but may provide more basic analysis
           - Larger models provide more detailed analysis but may take longer to process
        
        2. **Use AI recommendations as one input**, not the sole decision maker
        
        3. **Compare AI analysis with your own research** and other sources
        
        4. **Consider the fundamental factors** not captured in technical analysis
        
        5. **Be aware of market conditions** that may affect AI analysis accuracy
        
        6. **Regularly review your portfolio** as market conditions change
        """)
    
    # Disclaimer
    st.warning("""
    **Disclaimer**: This application provides analysis for educational purposes only. It is not intended to provide investment advice. 
    Always conduct your own research and consider consulting with a qualified financial advisor before making investment decisions.
    """) 