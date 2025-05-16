import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Import utility functions
from utils import (
    validate_ticker, 
    get_stock_data, 
    get_stock_info, 
    calculate_technical_indicators,
    create_stock_chart,
    analyze_with_llm,
    compare_stocks,
    suggest_portfolio_allocation,
    load_model,
    get_available_models
)

# Load environment variables
load_dotenv()

# App title and description
st.set_page_config(
    page_title="Indian Stock Market Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
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

# Header
st.markdown('<h1 class="main-header">Indian Stock Market Analyzer</h1>', unsafe_allow_html=True)
st.markdown('Analyze BSE and NSE stocks with AI-powered insights to maximize your investment profits')

# Get available models
available_models = get_available_models()

# Initialize session state for selected model
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "phi-2"  # Default model

# Model configuration in sidebar
st.sidebar.markdown('## AI Model Settings')

# Group models by size category
small_models = ["tinyllama-1.1b", "deepseek-coder-1.3b"]
medium_models = ["phi-2", "stablelm-zephyr-3b"]
large_models = ["llama3-8b", "deepseek-coder-6.7b"]

# Create model selection with categories
st.sidebar.markdown("### Select Analysis Model")

# Category-based selection
model_category = st.sidebar.radio(
    "Model Size",
    ["Small (4GB+ RAM)", "Medium (6GB+ RAM)", "Large (16GB+ RAM)"],
    index=1,  # Default to medium models
    help="Select model size based on your system capabilities"
)

# Get appropriate model options based on category
if model_category == "Small (4GB+ RAM)":
    category_models = small_models
elif model_category == "Large (16GB+ RAM)":
    category_models = large_models
else:  # Medium models as default
    category_models = medium_models

# Model selection within category
selected_model = st.sidebar.selectbox(
    "Model",
    options=category_models,
    index=0,
    format_func=lambda x: f"{x} - {available_models[x]['description']}"
)

# Show model details
model_info = available_models[selected_model]
st.sidebar.info(f"""
**{selected_model}**  
{model_info['description']}  
**System Requirements:** {model_info['ram_required']}
""")

# Update session state
if selected_model != st.session_state.selected_model:
    st.session_state.selected_model = selected_model

# Model details expander
with st.sidebar.expander("About Selected Model", expanded=False):
    st.markdown(f"""
    **Model**: {model_info['id']}
    
    **Description**: {model_info['description']}
    
    **System Requirements**: {model_info['ram_required']} RAM
    
    **Format**: {model_info['format']}
    
    This model will be downloaded and run locally on your machine.
    The first analysis may take time to download and load the model.
    """)

# Pre-load model option
if st.sidebar.button("Pre-load Selected Model"):
    with st.spinner(f"Loading {selected_model} model... This may take a few minutes depending on your hardware..."):
        try:
            load_model(selected_model)
            st.sidebar.success("✅ Model loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"❌ Error loading model: {str(e)}")

# Sidebar for stock selection and controls
st.sidebar.markdown('## Stock Selection')

# Add example tickers for user convenience
st.sidebar.markdown("""
#### Example Tickers:
- BSE: RELIANCE.BO, TCS.BO, HDFCBANK.BO, INFY.BO, ICICIBANK.BO
- NSE: RELIANCE.NS, TCS.NS, HDFCBANK.NS, INFY.NS, ICICIBANK.NS
""")

# Input for stock ticker
ticker = st.sidebar.text_input('Enter Stock Symbol (append .BO for BSE or .NS for NSE)')

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
period = time_periods[selected_period]

# Data interval selection
intervals = ['1d', '5d', '1wk', '1mo']
interval_names = {'1d': 'Daily', '5d': '5-Day', '1wk': 'Weekly', '1mo': 'Monthly'}
selected_interval = st.sidebar.selectbox('Select Data Interval', 
                                        [interval_names[i] for i in intervals])
interval = [k for k, v in interval_names.items() if v == selected_interval][0]

# Create tabs for different functionalities
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Single Stock Analysis", 
    "🔄 Stock Comparison", 
    "💼 Portfolio Optimization",
    "ℹ️ Help"
])

# Tab 1: Single Stock Analysis
with tab1:
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
                df = get_stock_data(ticker, period=period, interval=interval)
                
                # Calculate technical indicators
                df_with_indicators = calculate_technical_indicators(df)
                
                # Create and display chart
                fig = create_stock_chart(df_with_indicators, ticker)
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
                
                latest_indicators = df_with_indicators.iloc[-1]
                
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
            
            with col2:
                # Get stock info
                stock_info = get_stock_info(ticker)
                
                # Display general information
                st.markdown('<h3 class="sub-header">Stock Information</h3>', unsafe_allow_html=True)
                
                # Create a clean display of stock information
                info_df = pd.DataFrame({
                    'Attribute': [
                        'Name', 'Sector', 'Industry', 'Market Cap', 
                        'P/E Ratio', 'EPS', 'Dividend Yield', 'Avg Volume'
                    ],
                    'Value': [
                        stock_info['name'], 
                        stock_info['sector'], 
                        stock_info['industry'],
                        f"₹{stock_info['market_cap']:,}" if isinstance(stock_info['market_cap'], (int, float)) else stock_info['market_cap'],
                        f"{stock_info['pe_ratio']:.2f}" if isinstance(stock_info['pe_ratio'], (int, float)) else stock_info['pe_ratio'],
                        f"₹{stock_info['eps']:.2f}" if isinstance(stock_info['eps'], (int, float)) else stock_info['eps'],
                        f"{stock_info['dividend_yield']:.2f}%" if isinstance(stock_info['dividend_yield'], (int, float)) else stock_info['dividend_yield'],
                        f"{stock_info['avg_volume']:,}" if isinstance(stock_info['avg_volume'], (int, float)) else stock_info['avg_volume']
                    ]
                })
                
                st.table(info_df)
                
                # AI Analysis section
                st.markdown('<h3 class="sub-header">AI Analysis</h3>', unsafe_allow_html=True)
                
                st.info(f"Using model: **{selected_model}** - {available_models[selected_model]['description']}")
                
                if st.button("Generate AI Analysis"):
                    with st.spinner(f"Generating analysis using {selected_model}... This may take a few minutes on the first run as the model loads..."):
                        analysis = analyze_with_llm(stock_info, df_with_indicators, model_key=selected_model)
                        st.markdown(f'<div class="recommendation-box">{analysis}</div>', 
                                  unsafe_allow_html=True)
        else:
            st.error(f"Invalid ticker symbol: {ticker}. Please ensure you're using the correct format (e.g., RELIANCE.BO for BSE or RELIANCE.NS for NSE)")
    else:
        st.info("Enter a stock symbol in the sidebar to begin analysis")

# Tab 2: Stock Comparison
with tab2:
    st.markdown('<h2 class="sub-header">Compare Multiple Stocks</h2>', unsafe_allow_html=True)
    
    # Input for multiple tickers
    ticker_input = st.text_input('Enter multiple stock symbols separated by commas (e.g., RELIANCE.NS, TCS.NS, HDFCBANK.NS)')
    
    if ticker_input:
        tickers = [t.strip() for t in ticker_input.split(',')]
        
        # Validate all tickers
        invalid_tickers = [t for t in tickers if not validate_ticker(t)]
        valid_tickers = [t for t in tickers if validate_ticker(t)]
        
        if invalid_tickers:
            st.warning(f"Invalid ticker symbols: {', '.join(invalid_tickers)}")
        
        if valid_tickers:
            with st.spinner("Fetching data for comparison..."):
                # Get comparison data
                comparison_df, normalized_prices, performance_metrics = compare_stocks(valid_tickers, period=period)
                
                # Display comparison table
                st.markdown('<h3 class="sub-header">Performance Comparison</h3>', unsafe_allow_html=True)
                st.dataframe(comparison_df.style.format({
                    'Total Return (%)': '{:.2f}',
                    'Annualized Volatility (%)': '{:.2f}',
                    'Sharpe Ratio': '{:.2f}',
                    'Current Price': '{:.2f}',
                    'P/E Ratio': '{:.2f}',
                    'Dividend Yield (%)': '{:.2f}'
                }))
                
                # Display price chart
                st.markdown('<h3 class="sub-header">Normalized Price Comparison (Base=100)</h3>', unsafe_allow_html=True)
                
                # Create price comparison chart
                fig = px.line(normalized_prices, x=normalized_prices.index, y=normalized_prices.columns,
                              title="Price Performance Comparison (Normalized)",
                              labels={"value": "Normalized Price", "variable": "Stock"})
                st.plotly_chart(fig, use_container_width=True)
                
                # Display relative strength chart
                st.markdown('<h3 class="sub-header">Returns Comparison</h3>', unsafe_allow_html=True)
                
                # Create returns comparison chart
                returns_data = {ticker: performance_metrics[ticker]['Total Return (%)'] for ticker in valid_tickers}
                fig = px.bar(
                    x=list(returns_data.keys()),
                    y=list(returns_data.values()),
                    labels={'x': 'Stock', 'y': 'Total Return (%)'},
                    title="Total Returns Comparison"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Risk vs. Return scatter plot
                st.markdown('<h3 class="sub-header">Risk vs. Return</h3>', unsafe_allow_html=True)
                
                risk_return_data = pd.DataFrame({
                    'Stock': valid_tickers,
                    'Return (%)': [performance_metrics[t]['Total Return (%)'] for t in valid_tickers],
                    'Risk (%)': [performance_metrics[t]['Annualized Volatility (%)'] for t in valid_tickers],
                    'Sharpe Ratio': [performance_metrics[t]['Sharpe Ratio'] for t in valid_tickers]
                })
                
                fig = px.scatter(
                    risk_return_data,
                    x='Risk (%)',
                    y='Return (%)',
                    size='Sharpe Ratio',
                    hover_name='Stock',
                    text='Stock',
                    title="Risk vs. Return Analysis"
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Enter multiple stock symbols separated by commas to compare them")

# Tab 3: Portfolio Optimization
with tab3:
    st.markdown('<h2 class="sub-header">Portfolio Optimization</h2>', unsafe_allow_html=True)
    
    # Input for portfolio tickers
    portfolio_input = st.text_input('Enter stock symbols for your portfolio (separated by commas)')
    
    if portfolio_input:
        portfolio_tickers = [t.strip() for t in portfolio_input.split(',')]
        
        # Validate all tickers
        invalid_portfolio_tickers = [t for t in portfolio_tickers if not validate_ticker(t)]
        valid_portfolio_tickers = [t for t in portfolio_tickers if validate_ticker(t)]
        
        if invalid_portfolio_tickers:
            st.warning(f"Invalid ticker symbols: {', '.join(invalid_portfolio_tickers)}")
        
        if valid_portfolio_tickers:
            if len(valid_portfolio_tickers) < 2:
                st.warning("Please enter at least two valid stock symbols for portfolio analysis")
            else:
                st.markdown('<h3 class="sub-header">Portfolio Analysis</h3>', unsafe_allow_html=True)
                
                st.info(f"Using model: **{selected_model}** - {available_models[selected_model]['description']}")
                
                if st.button("Generate Portfolio Recommendations"):
                    with st.spinner(f"Analyzing your portfolio using {selected_model}... This may take a few minutes..."):
                        # Gather analysis results for each stock
                        analysis_results = {}
                        progress_bar = st.progress(0)
                        
                        for i, ticker in enumerate(valid_portfolio_tickers):
                            # Update progress
                            progress_bar.progress((i / len(valid_portfolio_tickers)))
                            
                            # Get stock data and calculate indicators
                            stock_info = get_stock_info(ticker)
                            df = get_stock_data(ticker, period='1y')
                            df_with_indicators = calculate_technical_indicators(df)
                            
                            # Get AI analysis
                            analysis = analyze_with_llm(stock_info, df_with_indicators, model_key=selected_model)
                            analysis_results[ticker] = analysis
                        
                        # Complete progress
                        progress_bar.progress(1.0)
                        
                        # Get portfolio allocation suggestion
                        allocation_suggestion = suggest_portfolio_allocation(
                            valid_portfolio_tickers, analysis_results, model_key=selected_model
                        )
                        
                        st.markdown(f'<div class="recommendation-box">{allocation_suggestion}</div>', 
                                  unsafe_allow_html=True)
    else:
        st.info("Enter the stock symbols for your portfolio to get optimization recommendations")

# Tab 4: Help
with tab4:
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

        This application offers multiple free, open-source large language models that run locally on your machine:
        
        **Model Options:**
        """)
        
        # List all available models with details
        for model_key, model_info in available_models.items():
            st.markdown(f"""
            #### {model_key}
            - **Description**: {model_info['description']}
            - **System Requirements**: {model_info['ram_required']} RAM
            - **Model ID**: `{model_info['id']}`
            """)
        
        st.markdown("""
        **Benefits:**
        - **Free to use**: No API costs or subscription fees
        - **Privacy**: All analysis is done locally on your machine
        - **Customizable**: You can choose the model that works best for your hardware
        
        **First-Time Use:**
        The first time you run an analysis with a specific model, the application will download and load it.
        This may take a few minutes depending on your internet connection and hardware.
        Subsequent analyses will be faster as the model will already be loaded.
        """)
    
    # Tips for using AI analysis
    with st.expander("Tips for Using AI Analysis"):
        st.markdown("""
        1. **Choose the right model for your hardware**:
           - Smaller models (1-3B parameters) work well on most computers
           - Larger models (5B+ parameters) provide better analysis but require more RAM/GPU memory
        
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

# Footer
st.markdown("""
---
Made with ❤️ using Streamlit, yfinance, and open-source LLMs | Indian Stock Market Analyzer
""") 