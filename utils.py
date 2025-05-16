import os
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, timedelta
from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load environment variables
load_dotenv()

# Define available models with their specifications
AVAILABLE_MODELS = {
    "deepseek-coder-1.3b": {
        "id": "deepseek-ai/deepseek-coder-1.3b-instruct",
        "description": "Lightweight coding model (1.3B parameters)",
        "ram_required": "4GB+",
        "format": "deepseek"
    },
    "tinyllama-1.1b": {
        "id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "description": "Very small general model (1.1B parameters)",
        "ram_required": "4GB+",
        "format": "llama"
    },
    "phi-2": {
        "id": "microsoft/phi-2",
        "description": "Microsoft's efficient model (2.7B parameters)",
        "ram_required": "6GB+",
        "format": "default"
    },
    "stablelm-zephyr-3b": {
        "id": "stabilityai/stablelm-zephyr-3b",
        "description": "Stability AI's 3B model tuned on Zephyr data",
        "ram_required": "6GB+",
        "format": "zephyr"
    },
    "llama3-8b": {
        "id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "description": "Meta's Llama 3 model (8B parameters)",
        "ram_required": "16GB+",
        "format": "llama"
    },
    "deepseek-coder-6.7b": {
        "id": "deepseek-ai/deepseek-coder-6.7b-instruct",
        "description": "Powerful coding model (6.7B parameters)",
        "ram_required": "16GB+",
        "format": "deepseek"
    }
}

# Default model if none selected
DEFAULT_MODEL = "phi-2"

# Initialize model variables as None - will be loaded on demand
current_model_key = None
tokenizer = None
model = None

def get_available_models():
    """
    Returns a dictionary of available models
    """
    return AVAILABLE_MODELS

def load_model(model_key=DEFAULT_MODEL):
    """
    Load the specified model and its tokenizer on demand
    
    Args:
        model_key (str): Key for the model to load from AVAILABLE_MODELS
    """
    global tokenizer, model, current_model_key
    
    # If the requested model is already loaded, do nothing
    if current_model_key == model_key and tokenizer is not None and model is not None:
        return
    
    # Get model info
    if model_key not in AVAILABLE_MODELS:
        model_key = DEFAULT_MODEL
    
    model_info = AVAILABLE_MODELS[model_key]
    model_id = model_info["id"]
    
    print(f"Loading model: {model_id}... This may take a moment.")
    
    # Free up memory if another model was loaded before
    if model is not None:
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    
    # Move model to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Update current model key
    current_model_key = model_key
    
    print(f"Model {model_id} loaded successfully!")
    return True

def validate_ticker(ticker):
    """
    Validate if the ticker exists and is valid for BSE or NSE
    
    Args:
        ticker (str): Stock ticker symbol with .BO (BSE) or .NS (NSE) suffix
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        # Check if we got valid info back
        if 'regularMarketPrice' in info and info['regularMarketPrice'] is not None:
            return True
        return False
    except Exception as e:
        return False

def get_stock_data(ticker, period="1y", interval="1d"):
    """
    Fetch historical stock data
    
    Args:
        ticker (str): Stock ticker symbol with .BO (BSE) or .NS (NSE) suffix
        period (str): Period to fetch data for (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval (str): Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        
    Returns:
        pd.DataFrame: DataFrame with stock data
    """
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)
    return df

def get_stock_info(ticker):
    """
    Get general information about a stock
    
    Args:
        ticker (str): Stock ticker symbol with .BO (BSE) or .NS (NSE) suffix
        
    Returns:
        dict: Stock information
    """
    stock = yf.Ticker(ticker)
    info = stock.info
    
    # Extract relevant information
    relevant_info = {
        'symbol': ticker,
        'name': info.get('shortName', 'N/A'),
        'sector': info.get('sector', 'N/A'),
        'industry': info.get('industry', 'N/A'),
        'current_price': info.get('currentPrice', info.get('regularMarketPrice', 'N/A')),
        'market_cap': info.get('marketCap', 'N/A'),
        'pe_ratio': info.get('trailingPE', 'N/A'),
        'eps': info.get('trailingEps', 'N/A'),
        'dividend_yield': info.get('dividendYield', 'N/A') * 100 if info.get('dividendYield') else 'N/A',
        '52_week_high': info.get('fiftyTwoWeekHigh', 'N/A'),
        '52_week_low': info.get('fiftyTwoWeekLow', 'N/A'),
        'avg_volume': info.get('averageVolume', 'N/A'),
    }
    
    return relevant_info

def calculate_technical_indicators(df):
    """
    Calculate technical indicators for stock data
    
    Args:
        df (pd.DataFrame): DataFrame with stock data
        
    Returns:
        pd.DataFrame: DataFrame with additional technical indicators
    """
    # Make a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Calculate moving averages
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()
    
    # Calculate RSI (Relative Strength Index)
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate MACD (Moving Average Convergence Divergence)
    data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    # Calculate Bollinger Bands
    data['MA20_std'] = data['Close'].rolling(window=20).std()
    data['upper_band'] = data['MA20'] + (data['MA20_std'] * 2)
    data['lower_band'] = data['MA20'] - (data['MA20_std'] * 2)
    
    return data

def create_stock_chart(df, ticker):
    """
    Create an interactive chart for stock data using Plotly
    
    Args:
        df (pd.DataFrame): DataFrame with stock data and indicators
        ticker (str): Stock ticker symbol
        
    Returns:
        go.Figure: Plotly figure object
    """
    # Create candlestick chart
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                                         open=df['Open'],
                                         high=df['High'],
                                         low=df['Low'],
                                         close=df['Close'],
                                         name='Price')])
    
    # Add volume as bar chart
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', yaxis='y2', opacity=0.3))
    
    # Add moving averages
    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], mode='lines', name='20-day MA', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], mode='lines', name='50-day MA', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA200'], mode='lines', name='200-day MA', line=dict(color='red')))
    
    # Add Bollinger Bands
    fig.add_trace(go.Scatter(x=df.index, y=df['upper_band'], mode='lines', name='Upper Band', line=dict(color='rgba(0,128,0,0.3)')))
    fig.add_trace(go.Scatter(x=df.index, y=df['lower_band'], mode='lines', name='Lower Band', line=dict(color='rgba(0,128,0,0.3)')))
    
    # Update layout
    fig.update_layout(
        title=f'{ticker} Stock Price and Indicators',
        yaxis_title='Price',
        xaxis_title='Date',
        yaxis2=dict(
            title='Volume',
            overlaying='y',
            side='right',
            showgrid=False
        ),
        height=600,
        legend=dict(orientation='h', y=1.05),
        margin=dict(l=50, r=50, t=100, b=50),
    )
    
    return fig

def analyze_with_llm(stock_info, technical_data, model_key=DEFAULT_MODEL, news=None):
    """
    Analyze stock data using the selected LLM model
    
    Args:
        stock_info (dict): General stock information
        technical_data (pd.DataFrame): Technical indicators data
        model_key (str): Key for the model to use from AVAILABLE_MODELS
        news (list, optional): Recent news about the stock
        
    Returns:
        str: Analysis results from the LLM
    """
    # Load the model if it's not already loaded
    try:
        load_model(model_key)
    except Exception as e:
        return f"Error loading model: {str(e)}\n\nPlease try another model or check your system resources."
    
    # Extract the most recent data point
    latest_data = technical_data.iloc[-1].to_dict()
    
    # Prepare a summary of the technical indicators
    rsi_value = latest_data.get('RSI', 'N/A')
    macd_value = latest_data.get('MACD', 'N/A')
    signal_value = latest_data.get('Signal', 'N/A')
    
    current_price = latest_data.get('Close', 'N/A')
    ma20 = latest_data.get('MA20', 'N/A')
    ma50 = latest_data.get('MA50', 'N/A')
    ma200 = latest_data.get('MA200', 'N/A')
    
    upper_band = latest_data.get('upper_band', 'N/A')
    lower_band = latest_data.get('lower_band', 'N/A')
    
    # Calculate price trends
    price_trend_30d = ((current_price / technical_data['Close'].iloc[-30]) - 1) * 100 if len(technical_data) >= 30 else 'N/A'
    price_trend_90d = ((current_price / technical_data['Close'].iloc[-90]) - 1) * 100 if len(technical_data) >= 90 else 'N/A'
    
    # Format price trends as strings
    price_trend_30d_str = f"{price_trend_30d:.2f}%" if isinstance(price_trend_30d, (int, float)) else price_trend_30d
    price_trend_90d_str = f"{price_trend_90d:.2f}%" if isinstance(price_trend_90d, (int, float)) else price_trend_90d
    
    # Get model format
    model_format = AVAILABLE_MODELS[model_key]["format"]
    
    # Base prompt content
    prompt_content = f"""
Analyze the following stock:

Stock Information:
- Symbol: {stock_info['symbol']}
- Name: {stock_info['name']}
- Sector: {stock_info['sector']}
- Industry: {stock_info['industry']}
- Current Price: {stock_info['current_price']}
- Market Cap: {stock_info['market_cap']}
- P/E Ratio: {stock_info['pe_ratio']}
- EPS: {stock_info['eps']}
- Dividend Yield: {stock_info['dividend_yield']}
- 52-Week High: {stock_info['52_week_high']}
- 52-Week Low: {stock_info['52_week_low']}
- Average Volume: {stock_info['avg_volume']}

Technical Indicators:
- Current Price: {current_price}
- 20-Day Moving Average: {ma20}
- 50-Day Moving Average: {ma50}
- 200-Day Moving Average: {ma200}
- RSI (14-day): {rsi_value}
- MACD: {macd_value}
- MACD Signal: {signal_value}
- Upper Bollinger Band: {upper_band}
- Lower Bollinger Band: {lower_band}
- 30-Day Price Change: {price_trend_30d_str}
- 90-Day Price Change: {price_trend_90d_str}

Provide a comprehensive analysis including:
1. Overall assessment of the stock's financial health
2. Technical analysis and key indicator interpretations
3. Potential short-term (1-3 months) and long-term (6-12 months) price movements
4. Key support and resistance levels to watch
5. Risk factors specific to this stock
6. Investment recommendation (Buy, Hold, or Sell) with rationale
7. Suggested position sizing and risk management approaches

Focus on Indian market context and factors specific to the BSE/NSE markets.
"""
    
    # Format prompt based on model type
    if model_format == "deepseek":
        prompt = f"""<｜begin▁of▁conversation｜>
<｜system｜>
You are a financial analyst specializing in Indian stock markets (BSE and NSE). Provide a comprehensive financial analysis based on the data provided.
</｜system｜>

<｜user｜>
{prompt_content}
</｜user｜>

<｜assistant｜>
"""
    elif model_format == "llama":
        prompt = f"""<s>[INST] <<SYS>>
You are a financial analyst specializing in Indian stock markets (BSE and NSE). Provide a comprehensive financial analysis based on the data provided.
<</SYS>>

{prompt_content} [/INST]
"""
    elif model_format == "zephyr":
        prompt = f"""<|system|>
You are a financial analyst specializing in Indian stock markets (BSE and NSE). Provide a comprehensive financial analysis based on the data provided.
</|system|>

<|user|>
{prompt_content}
</|user|>

<|assistant|>
"""
    else:  # default format
        prompt = f"""You are a financial analyst specializing in Indian stock markets (BSE and NSE). Provide a comprehensive financial analysis based on the data provided.

{prompt_content}
"""
    
    try:
        # Get device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Generate the response
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=1000,  # Reduced for faster generation and to fit in memory constraints
                temperature=0.3,
                top_p=0.95,
                do_sample=True,
            )
        
        # Decode the generated response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the model's response based on format
        if model_format == "deepseek":
            # Extract just the assistant's response part
            if "<｜assistant｜>" in full_response:
                assistant_response = full_response.split("<｜assistant｜>")[1].split("</｜assistant｜>")[0].strip()
            else:
                assistant_response = full_response
        elif model_format == "llama":
            # For Llama models, we typically just need to take everything after the prompt
            assistant_response = full_response.split(prompt_content)[1].strip()
            if "[/INST]" in assistant_response:
                assistant_response = assistant_response.split("[/INST]")[1].strip()
        elif model_format == "zephyr":
            # Extract assistant's response for Zephyr model
            if "<|assistant|>" in full_response:
                assistant_response = full_response.split("<|assistant|>")[1].split("</|assistant|>")[0].strip()
            else:
                assistant_response = full_response
        else:
            # For default format, just take everything after the prompt
            assistant_response = full_response.split(prompt_content)[-1].strip()
        
        # Failsafe: If parsing fails, return the full response
        if not assistant_response:
            assistant_response = full_response
        
        return assistant_response
    except Exception as e:
        return f"Error in analysis: {str(e)}"

def compare_stocks(tickers, period="1y"):
    """
    Compare multiple stocks based on key metrics
    
    Args:
        tickers (list): List of stock ticker symbols
        period (str): Period to fetch data for
        
    Returns:
        tuple: (comparison_df, normalized_price_df, performance_metrics)
    """
    comparison_data = {}
    normalized_prices = pd.DataFrame()
    performance_metrics = {}
    
    for ticker in tickers:
        if not validate_ticker(ticker):
            continue
            
        # Get stock information
        stock_info = get_stock_info(ticker)
        
        # Get historical data
        hist_data = get_stock_data(ticker, period=period)
        
        if hist_data.empty:
            continue
            
        # Calculate performance metrics
        start_price = hist_data['Close'].iloc[0]
        end_price = hist_data['Close'].iloc[-1]
        total_return = ((end_price / start_price) - 1) * 100
        
        # Calculate volatility (standard deviation of daily returns)
        daily_returns = hist_data['Close'].pct_change().dropna()
        volatility = daily_returns.std() * (252 ** 0.5) * 100  # Annualized volatility
        
        # Calculate Sharpe ratio (assuming risk-free rate of 5% for Indian markets)
        risk_free_rate = 0.05
        sharpe_ratio = (daily_returns.mean() * 252 - risk_free_rate) / (daily_returns.std() * (252 ** 0.5))
        
        # Store the metrics
        performance_metrics[ticker] = {
            'Total Return (%)': total_return,
            'Annualized Volatility (%)': volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Current Price': end_price,
            'Market Cap': stock_info.get('market_cap', 'N/A'),
            'P/E Ratio': stock_info.get('pe_ratio', 'N/A'),
            'Dividend Yield (%)': stock_info.get('dividend_yield', 'N/A'),
        }
        
        # Add normalized prices for comparison chart
        normalized_prices[ticker] = hist_data['Close'] / start_price * 100
    
    # Convert performance metrics to DataFrame
    comparison_df = pd.DataFrame(performance_metrics).T
    
    return comparison_df, normalized_prices, performance_metrics

def suggest_portfolio_allocation(tickers, analysis_results, model_key=DEFAULT_MODEL):
    """
    Suggest an optimal portfolio allocation based on LLM analysis
    
    Args:
        tickers (list): List of stock ticker symbols
        analysis_results (dict): Dictionary with LLM analysis for each ticker
        model_key (str): Key for the model to use from AVAILABLE_MODELS
        
    Returns:
        str: Suggested portfolio allocation and strategy
    """
    # Load the model if it's not already loaded
    try:
        load_model(model_key)
    except Exception as e:
        return f"Error loading model: {str(e)}\n\nPlease try another model or check your system resources."
    
    # Prepare a prompt with information about all stocks
    stocks_info = "\n\n".join([
        f"Stock: {ticker}\n{analysis}" 
        for ticker, analysis in analysis_results.items()
    ])
    
    # Base prompt content
    prompt_content = f"""
As a portfolio manager specializing in Indian markets, recommend an optimal allocation for a portfolio consisting of these stocks:

{stocks_info}

Provide:
1. Recommended percentage allocation for each stock
2. Rationale for each allocation
3. Overall portfolio strategy
4. Risk assessment for the portfolio
5. Suggestions for any additional diversification needed

Focus on maximizing returns while managing risk appropriately.
"""
    
    # Get model format
    model_format = AVAILABLE_MODELS[model_key]["format"]
    
    # Format prompt based on model type
    if model_format == "deepseek":
        prompt = f"""<｜begin▁of▁conversation｜>
<｜system｜>
You are a portfolio manager specializing in Indian stock markets. Provide detailed allocation advice for maximizing returns while managing risk.
</｜system｜>

<｜user｜>
{prompt_content}
</｜user｜>

<｜assistant｜>
"""
    elif model_format == "llama":
        prompt = f"""<s>[INST] <<SYS>>
You are a portfolio manager specializing in Indian stock markets. Provide detailed allocation advice for maximizing returns while managing risk.
<</SYS>>

{prompt_content} [/INST]
"""
    elif model_format == "zephyr":
        prompt = f"""<|system|>
You are a portfolio manager specializing in Indian stock markets. Provide detailed allocation advice for maximizing returns while managing risk.
</|system|>

<|user|>
{prompt_content}
</|user|>

<|assistant|>
"""
    else:  # default format
        prompt = f"""You are a portfolio manager specializing in Indian stock markets. Provide detailed allocation advice for maximizing returns while managing risk.

{prompt_content}
"""
    
    try:
        # Get device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Generate the response
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=1000,  # Reduced for faster generation
                temperature=0.3,
                top_p=0.95,
                do_sample=True,
            )
        
        # Decode the generated response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the model's response based on format
        if model_format == "deepseek":
            # Extract just the assistant's response part
            if "<｜assistant｜>" in full_response:
                assistant_response = full_response.split("<｜assistant｜>")[1].split("</｜assistant｜>")[0].strip()
            else:
                assistant_response = full_response
        elif model_format == "llama":
            # For Llama models, we typically just need to take everything after the prompt
            assistant_response = full_response.split(prompt_content)[1].strip()
            if "[/INST]" in assistant_response:
                assistant_response = assistant_response.split("[/INST]")[1].strip()
        elif model_format == "zephyr":
            # Extract assistant's response for Zephyr model
            if "<|assistant|>" in full_response:
                assistant_response = full_response.split("<|assistant|>")[1].split("</|assistant|>")[0].strip()
            else:
                assistant_response = full_response
        else:
            # For default format, just take everything after the prompt
            assistant_response = full_response.split(prompt_content)[-1].strip()
        
        # Failsafe: If parsing fails, return the full response
        if not assistant_response:
            assistant_response = full_response
        
        return assistant_response
    except Exception as e:
        return f"Error in portfolio allocation suggestion: {str(e)}" 