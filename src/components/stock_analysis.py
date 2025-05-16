"""
Component for the single stock analysis tab
"""

import streamlit as st
import pandas as pd
import re

from ..utils.stock_data import validate_ticker, get_stock_data, get_stock_info
from ..utils.plotting import create_stock_chart
from ..utils.analysis import analyze_with_llm
from ..models.model_info import get_available_models
import os

def create_stock_analysis_tab(selected_model):
    """
    Create the stock analysis tab content
    
    Args:
        selected_model (str): The selected model key
    """
    ticker = st.session_state.get('ticker', '')
    
    if ticker:
        # Check if ticker is valid
        if validate_ticker(ticker):
            st.markdown(f'<h2 class="sub-header">Analysis for {ticker}</h2>', unsafe_allow_html=True)
            
            # Create two columns for the layout
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Show progress message
                progress_msg = st.empty()
                progress_msg.info('Fetching stock data...')
                
                # Get stock data
                df = get_stock_data(ticker, period=st.session_state.period, interval=st.session_state.interval)
                
                if not df.empty:
                    # Create and display chart
                    fig = create_stock_chart(df, ticker)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display price statistics
                    st.markdown('<h3 class="sub-header">Price Statistics</h3>', unsafe_allow_html=True)
                    
                    # Calculate basic statistics
                    current_price = df['Close'].iloc[-1]
                    change = df['Close'].iloc[-1] - df['Close'].iloc[-2]
                    pct_change = (change / df['Close'].iloc[-2]) * 100
                    
                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                    metrics_col1.metric("Current Price", f"₹{current_price:.2f}", 
                                      f"{change:.2f} ({pct_change:.2f}%)")
                    metrics_col2.metric("52-Week High", f"₹{df['High'].max():.2f}")
                    metrics_col3.metric("52-Week Low", f"₹{df['Low'].min():.2f}")
                    
                    # Display technical indicators
                    st.markdown('<h3 class="sub-header">Technical Indicators</h3>', unsafe_allow_html=True)
                    
                    latest_indicators = df.iloc[-1]
                    
                    indicators_col1, indicators_col2, indicators_col3 = st.columns(3)
                    
                    # Display RSI
                    rsi = latest_indicators['RSI']
                    rsi_color = "red" if rsi > 70 else "green" if rsi < 30 else "black"
                    indicators_col1.markdown(f"**RSI (14):** <span style='color:{rsi_color}'>{rsi:.2f}</span>", 
                                           unsafe_allow_html=True)
                    
                    # Display MACD
                    macd = latest_indicators['MACD']
                    signal = latest_indicators['Signal']
                    macd_color = "green" if macd > signal else "red"
                    indicators_col2.markdown(f"**MACD:** <span style='color:{macd_color}'>{macd:.2f}</span>", 
                                           unsafe_allow_html=True)
                    indicators_col2.markdown(f"**Signal:** {signal:.2f}", unsafe_allow_html=True)
                    
                    # Display Moving Averages
                    ma20 = latest_indicators['MA20']
                    ma50 = latest_indicators['MA50']
                    ma200 = latest_indicators['MA200']
                    indicators_col3.markdown(f"**MA20:** {ma20:.2f}", unsafe_allow_html=True)
                    indicators_col3.markdown(f"**MA50:** {ma50:.2f}", unsafe_allow_html=True)
                    indicators_col3.markdown(f"**MA200:** {ma200:.2f}", unsafe_allow_html=True)
                    
                    # Clear progress message
                    progress_msg.empty()
                else:
                    st.error(f"Could not fetch data for {ticker}. Please check if the symbol is correct.")
            
            with col2:
                # Get stock info
                stock_info = get_stock_info(ticker)
                
                # Display general information
                st.markdown('<h3 class="sub-header">Stock Information</h3>', unsafe_allow_html=True)
                
                # Create a clean display of stock information - Convert all values to strings to avoid PyArrow errors
                info_df = pd.DataFrame({
                    'Attribute': [
                        'Name', 'Sector', 'Industry', 'Market Cap', 
                        'P/E Ratio', 'EPS', 'Dividend Yield', 'Avg Volume'
                    ],
                    'Value': [
                        str(stock_info['name']), 
                        str(stock_info['sector']), 
                        str(stock_info['industry']),
                        str(stock_info['market_cap']),
                        str(stock_info['pe_ratio']),
                        str(stock_info['eps']),
                        str(stock_info['dividend_yield']),
                        str(stock_info['avg_volume'])
                    ]
                })
                
                st.table(info_df)
                
                # AI Analysis section
                st.markdown('<h3 class="sub-header">AI Analysis</h3>', unsafe_allow_html=True)
                
                available_models = get_available_models()
                
                st.info(f"Using model: **{selected_model}** - {available_models[selected_model]['description']} (via Hugging Face API)")
                
                # Check if Hugging Face API key is available
                hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
                api_available = hf_api_key is not None and hf_api_key.strip() != ""

                if not api_available:
                    st.error("❌ No Hugging Face API key found. You must set up an API key to use the analysis features.")
                    st.markdown("""
                    To set up your API key:
                    1. Create a free account at [Hugging Face](https://huggingface.co/)
                    2. Generate an API key from [your settings page](https://huggingface.co/settings/tokens)
                    3. Create a `.env` file in the app's directory with: `HUGGINGFACE_API_KEY=your_api_key_here`
                    4. Restart the application
                    """)
                else:
                    if st.button("Generate AI Analysis"):
                        with st.spinner(f"Generating analysis using {selected_model} via Hugging Face API..."):
                            try:
                                # Proceed with analysis
                                analysis = analyze_with_llm(
                                    stock_info, 
                                    df, 
                                    model_key=selected_model
                                )
                                
                                # Check if analysis was successful
                                if analysis and not analysis.startswith("Error"):
                                    # Store the analysis in session state for portfolio allocation
                                    if 'analysis_results' not in st.session_state:
                                        st.session_state.analysis_results = {}
                                    st.session_state.analysis_results[ticker] = analysis
                                    
                                    # Create a container with proper styling for the analysis
                                    analysis_container = st.container()
                                    with analysis_container:
                                        st.markdown("<div class='recommendation-box'>", unsafe_allow_html=True)
                                        
                                        # Format the analysis text to ensure proper rendering
                                        formatted_analysis = analysis.replace("\n", "<br>")
                                        
                                        # Handle potential Markdown formatting in the analysis
                                        # First, escape any HTML tags in the analysis that aren't our <br> tags
                                        formatted_analysis = re.sub(r'<(?!br>|/br>)', '&lt;', formatted_analysis)
                                        formatted_analysis = re.sub(r'(?<!<br)>', '&gt;', formatted_analysis)
                                        
                                        # Now render with proper formatting
                                        st.markdown(f"{formatted_analysis}", unsafe_allow_html=True)
                                        st.markdown("</div>", unsafe_allow_html=True)
                                else:
                                    st.error(f"Error during analysis: {analysis}")
                            except Exception as e:
                                st.error(f"An error occurred: {str(e)}")
        else:
            st.error(f"Invalid ticker symbol: {ticker}. Please ensure you're using the correct format (e.g., RELIANCE.BO for BSE or RELIANCE.NS for NSE)")
    else:
        st.info("Enter a stock symbol in the sidebar to begin analysis") 