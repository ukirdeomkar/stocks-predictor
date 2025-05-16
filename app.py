import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import sys
import importlib

# Fix for torch path issues with Streamlit's hot-reloading - with proper existence check
try:
    # Only try to reload if the module exists
    if 'torch.classes' in sys.modules and sys.modules['torch.classes'] is not None:
        importlib.reload(sys.modules['torch.classes'])
except (ModuleNotFoundError, KeyError, AttributeError):
    # Silently continue if module can't be found or reloaded
    pass

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

# Initialize session state variables if they don't exist
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "phi-2"  # Default model
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
    
# Always use API - no toggle needed
st.session_state.use_api = True

# Initialize session state variables
if "use_api" not in st.session_state:
    st.session_state.use_api = True

if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = {}

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
                                    df_with_indicators, 
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
                                        import re
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
                        try:
                            # Import torch only when needed to avoid module path issues
                            import torch
                            
                            # Try to load the model first
                            load_successful = load_model(selected_model)
                            
                            if not load_successful:
                                st.error(f"Failed to load the model {selected_model}. Please try a different model or restart the application.")
                            else:
                                # Gather analysis results for each stock
                                analysis_results = {}
                                progress_bar = st.progress(0)
                                analysis_errors = []
                                
                                for i, ticker in enumerate(valid_portfolio_tickers):
                                    # Update progress
                                    progress_bar.progress((i / len(valid_portfolio_tickers)))
                                    
                                    try:
                                        # Get stock data and calculate indicators
                                        stock_info = get_stock_info(ticker)
                                        df = get_stock_data(ticker, period='1y')
                                        df_with_indicators = calculate_technical_indicators(df)
                                        
                                        # Get AI analysis
                                        analysis = analyze_with_llm(
                                            stock_info, 
                                            df_with_indicators, 
                                            model_key=selected_model,
                                            use_api=st.session_state.use_api
                                        )
                                        
                                        if analysis and not analysis.startswith("Error"):
                                            analysis_results[ticker] = analysis
                                        else:
                                            analysis_errors.append(f"Could not analyze {ticker}: {analysis}")
                                    except Exception as e:
                                        analysis_errors.append(f"Error analyzing {ticker}: {str(e)}")
                                
                                # Complete progress
                                progress_bar.progress(1.0)
                                
                                # Show any errors that occurred during analysis
                                if analysis_errors:
                                    st.warning(f"Some stocks could not be analyzed: {', '.join(analysis_errors)}")
                                
                                if analysis_results:
                                    # Get portfolio allocation suggestion
                                    try:
                                        allocation_suggestion = suggest_portfolio_allocation(
                                            list(analysis_results.keys()), 
                                            analysis_results, 
                                            model_key=selected_model,
                                            use_api=st.session_state.use_api
                                        )
                                        
                                        if allocation_suggestion and not allocation_suggestion.startswith("Error"):
                                            # Create a container with proper styling for the portfolio recommendation
                                            portfolio_container = st.container()
                                            with portfolio_container:
                                                st.markdown("<div class='recommendation-box'>", unsafe_allow_html=True)
                                                
                                                # Format the analysis text to ensure proper rendering
                                                formatted_suggestion = allocation_suggestion.replace("\n", "<br>")
                                                
                                                # Handle potential Markdown formatting in the analysis
                                                import re
                                                formatted_suggestion = re.sub(r'<(?!br>|/br>)', '&lt;', formatted_suggestion)
                                                formatted_suggestion = re.sub(r'(?<!<br)>', '&gt;', formatted_suggestion)
                                                
                                                # Now render with proper formatting
                                                st.markdown(f"{formatted_suggestion}", unsafe_allow_html=True)
                                                st.markdown("</div>", unsafe_allow_html=True)
                                        else:
                                            st.error(f"Error generating portfolio recommendations: {allocation_suggestion}")
                                    except Exception as e:
                                        st.error(f"Error during portfolio optimization: {str(e)}")
                                else:
                                    st.error("Could not analyze any of the selected stocks. Please try different stocks or another model.")
                        except ImportError:
                            st.error("❌ PyTorch is not properly installed. Please check your environment setup.")
                        except Exception as e:
                            st.error(f"An error occurred: {str(e)}. Please try a different model or restart the application.")
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

# Check for analysis_results to show portfolio allocation section
if 'analysis_results' in st.session_state and st.session_state.analysis_results:
    st.markdown('<h3 class="sub-header">Portfolio Allocation Suggestion</h3>', unsafe_allow_html=True)
    
    # Check if Hugging Face API key is available
    hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
    api_available = hf_api_key is not None and hf_api_key.strip() != ""
    
    if not api_available:
        st.error("❌ No Hugging Face API key found. You must set up an API key to use the portfolio allocation feature.")
    else:
        if st.button("Generate Portfolio Allocation Suggestion"):
            with st.spinner(f"Generating portfolio allocation using {selected_model} via Hugging Face API..."):
                try:
                    # Get the portfolio allocation suggestion
                    portfolio_allocation = suggest_portfolio_allocation(
                        st.session_state.analysis_results,
                        model_key=selected_model
                    )
                    
                    # Check if portfolio allocation was successful
                    if portfolio_allocation and not portfolio_allocation.startswith("Error"):
                        st.success("Portfolio Allocation Suggestion")
                        st.markdown("<div class='allocation-box'>", unsafe_allow_html=True)
                        
                        # Format the portfolio allocation text
                        formatted_allocation = portfolio_allocation.replace("\n", "<br>")
                        
                        # Handle potential Markdown formatting
                        import re
                        formatted_allocation = re.sub(r'<(?!br>|/br>)', '&lt;', formatted_allocation)
                        formatted_allocation = re.sub(r'(?<!<br)>', '&gt;', formatted_allocation)
                        
                        st.markdown(f"{formatted_allocation}", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        st.error(f"Error generating portfolio allocation: {portfolio_allocation}")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}") 