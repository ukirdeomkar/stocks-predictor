import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import requests
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Default model to use
DEFAULT_MODEL = "phi-2"

def get_available_models():
    """
    Get a dictionary of available AI models with their descriptions and API endpoints.
    
    Returns:
        dict: Dictionary of model information with keys being model names
    """
    return {
        "phi-2": {
            "description": "Small but efficient model with good financial understanding",
            "api_endpoint": "microsoft/phi-2",
            "size": "2.7B"
        },
        "tinyllama-1.1b": {
            "description": "Ultra-compact model for very basic analysis",
            "api_endpoint": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "size": "1.1B"
        },
        "mistral-7b": {
            "description": "Well-balanced general-purpose model with strong reasoning",
            "api_endpoint": "mistralai/Mistral-7B-Instruct-v0.2",
            "size": "7B"
        },
        "deepseek-coder-6.7b": {
            "description": "A specialized model for financial and code analysis",
            "api_endpoint": "deepseek-ai/deepseek-coder-6.7b-instruct",
            "size": "6.7B"
        }
    }

def get_stock_info(ticker):
    """
    Get basic information about a stock.
    
    Args:
        ticker (str): Stock ticker symbol
        
    Returns:
        dict: Dictionary containing basic stock information
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Get the basics
        stock_info = {
            "symbol": ticker,
            "name": info.get("longName", "Unknown"),
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
            "current_price": info.get("currentPrice", info.get("regularMarketPrice", "Unknown")),
            "market_cap": info.get("marketCap", "Unknown"),
            "pe_ratio": info.get("trailingPE", "Unknown"),
            "eps": info.get("trailingEps", "Unknown"),
            "dividend_yield": info.get("dividendYield", "Unknown"),
            "52_week_high": info.get("fiftyTwoWeekHigh", "Unknown"),
            "52_week_low": info.get("fiftyTwoWeekLow", "Unknown"),
            "avg_volume": info.get("averageVolume", "Unknown")
        }
        
        # Format the values for better readability
        for key, value in stock_info.items():
            if key == "market_cap" and isinstance(value, (int, float)):
                # Convert market cap to billions/millions
                if value >= 1e9:
                    stock_info[key] = f"₹{value/1e9:.2f}B"
                elif value >= 1e6:
                    stock_info[key] = f"₹{value/1e6:.2f}M"
            elif key == "dividend_yield" and isinstance(value, (int, float)):
                stock_info[key] = f"{value*100:.2f}%"
            elif isinstance(value, (int, float)) and key not in ["eps", "pe_ratio"]:
                # Format numbers with commas for thousands
                stock_info[key] = f"{value:,}"
                
        return stock_info
    except Exception as e:
        print(f"Error getting stock info: {str(e)}")
        return {"symbol": ticker, "error": str(e)}

def get_stock_data(ticker, period="1y"):
    """
    Get historical stock data with technical indicators.
    
    Args:
        ticker (str): Stock ticker symbol
        period (str, optional): Time period for data (e.g., "1y", "6mo", "1d"). Defaults to "1y".
        
    Returns:
        pd.DataFrame: DataFrame with stock data and technical indicators
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        
        if df.empty:
            raise ValueError(f"No data returned for ticker {ticker}")
            
        # Calculate technical indicators
        # Moving Averages
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['MA200'] = df['Close'].rolling(window=200).mean()
        
        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD (Moving Average Convergence Divergence)
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['std'] = df['Close'].rolling(window=20).std()
        df['upper_band'] = df['MA20'] + (df['std'] * 2)
        df['lower_band'] = df['MA20'] - (df['std'] * 2)
        
        # Fill NaN values
        df = df.fillna(method='bfill')
        
        return df
    except Exception as e:
        print(f"Error getting stock data: {str(e)}")
        return pd.DataFrame()

def analyze_with_llm(stock_info, df_with_indicators, model_key=DEFAULT_MODEL):
    """
    Analyze stock data using a LLM via the Hugging Face API.
    
    Args:
        stock_info (dict): Dictionary containing basic stock information
        df_with_indicators (pandas.DataFrame): DataFrame with stock data and technical indicators
        model_key (str): Key of the model to use from available_models
        
    Returns:
        str: Analysis of the stock by the LLM, or error message
    """
    # Get available models
    available_models = get_available_models()
    
    # Verify the Hugging Face API key is set
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    if not api_key:
        return "Error: Hugging Face API key not found. Please set up your API key in the .env file."
    
    try:
        # Get model information
        if model_key not in available_models:
            return f"Error: Model {model_key} not found in available models."
        model_info = available_models[model_key]
        
        # Prepare API endpoint
        api_endpoint = model_info["api_endpoint"]
        api_url = os.getenv("HUGGINGFACE_API_URL", f"https://api-inference.huggingface.co/models/{api_endpoint}")
        
        # Prepare the prompt based on the stock data
        prompt = create_analysis_prompt(stock_info, df_with_indicators)
        
        # Call the Hugging Face API
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {"inputs": prompt}
        
        # Make the API request
        response = requests.post(api_url, headers=headers, json=payload)
        
        # Process the response
        if response.status_code != 200:
            return f"Error: API request failed with status code {response.status_code}. Details: {response.text}"
        
        # Extract the generated text from the response
        response_data = response.json()
        
        # Handle different response formats from different models
        if isinstance(response_data, list) and len(response_data) > 0:
            if isinstance(response_data[0], dict) and "generated_text" in response_data[0]:
                # Format for models like Mistral and some others
                analysis = response_data[0]["generated_text"]
            else:
                # Some other list format
                analysis = str(response_data[0])
        elif isinstance(response_data, dict):
            if "generated_text" in response_data:
                # Format for some models
                analysis = response_data["generated_text"]
            else:
                # Some other dictionary format
                analysis = str(response_data)
        else:
            # Fallback - use the raw response
            analysis = str(response_data)
        
        # Clean up the response
        if prompt in analysis:
            # Remove the prompt from the beginning of the response if present
            analysis = analysis[len(prompt):].strip()
        
        # If the response is empty, return an error
        if not analysis:
            return "Error: The model returned an empty response. Please try a different model."
        
        return analysis
    
    except Exception as e:
        return f"Error during analysis: {str(e)}"

def compare_stocks(tickers, period="1y"):
    """
    Compare multiple stocks based on technical indicators and performance.
    
    Args:
        tickers (list): List of stock ticker symbols
        period (str, optional): Time period for data. Defaults to "1y".
        
    Returns:
        pd.DataFrame: DataFrame with comparison metrics
        dict: Dictionary with normalized price data for plotting
        dict: Dictionary with performance metrics
    """
    comparison_data = {}
    normalized_prices = {}
    performance_metrics = {}
    
    try:
        # Get data for each ticker
        for ticker in tickers:
            df = get_stock_data(ticker, period)
            if df.empty:
                continue
                
            # Store the normalized price data for plotting
            start_price = df['Close'].iloc[0]
            normalized_prices[ticker] = (df['Close'] / start_price) * 100
                
            # Calculate the latest metrics
            latest = df.iloc[-1]
            month_ago = df.iloc[-30] if len(df) > 30 else df.iloc[0]
            
            # Calculate returns
            daily_return = (df['Close'].pct_change().mean() * 100)
            monthly_return = ((latest['Close'] / month_ago['Close']) - 1) * 100
            total_return = ((latest['Close'] / df['Close'].iloc[0]) - 1) * 100
            
            # Calculate volatility
            volatility = df['Close'].pct_change().std() * 100 * (252 ** 0.5)  # Annualized
            
            # Calculate Sharpe Ratio (assuming risk-free rate of 4%)
            risk_free_rate = 0.04
            sharpe_ratio = (total_return/100 - risk_free_rate) / (volatility/100) if volatility != 0 else 0
            
            # Store the metrics
            comparison_data[ticker] = {
                'Current Price': latest['Close'],
                'RSI': latest['RSI'],
                'MACD': latest['MACD'],
                'Above MA200': 'Yes' if latest['Close'] > latest['MA200'] else 'No',
                'Daily Return %': daily_return,
                'Monthly Return %': monthly_return,
                'Total Return %': total_return,
                'Volatility %': volatility,
                'Sharpe Ratio': sharpe_ratio
            }
            
            # Store performance metrics over time
            performance_metrics[ticker] = {
                'daily_returns': df['Close'].pct_change().dropna(),
                'cumulative_returns': (df['Close'] / df['Close'].iloc[0]) - 1,
                'volatility': df['Close'].pct_change().rolling(window=20).std() * (252 ** 0.5)
            }
            
    except Exception as e:
        print(f"Error comparing stocks: {str(e)}")
        
    # Convert to DataFrame
    comparison_df = pd.DataFrame(comparison_data).T
    
    # Format the DataFrame
    for col in ['RSI', 'MACD', 'Daily Return %', 'Monthly Return %', 'Total Return %', 'Volatility %', 'Sharpe Ratio']:
        if col in comparison_df.columns:
            comparison_df[col] = comparison_df[col].round(2)
    
    return comparison_df, normalized_prices, performance_metrics

def suggest_portfolio_allocation(analysis_results, model_key=DEFAULT_MODEL):
    """
    Suggest a portfolio allocation based on the analyzed stocks.
    
    Args:
        analysis_results (dict): Dictionary with stock symbols as keys and analysis results as values
        model_key (str): Key of the model to use from available_models
        
    Returns:
        str: Portfolio allocation suggestion from the model, or error message
    """
    # Get available models
    available_models = get_available_models()
    
    # Verify the Hugging Face API key is set
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    if not api_key:
        return "Error: Hugging Face API key not found. Please set up your API key in the .env file."
    
    try:
        # Validate parameters
        if not analysis_results or len(analysis_results) < 2:
            return "Error: Need at least 2 analyzed stocks to suggest a portfolio allocation."
        
        # Get model information
        if model_key not in available_models:
            return f"Error: Model {model_key} not found in available models."
        model_info = available_models[model_key]
        
        # Create summary of each analyzed stock
        stock_summaries = []
        for symbol, analysis in analysis_results.items():
            # Split the analysis into lines and grab the relevant sections
            lines = analysis.split('\n')
            recommendation_line = None
            risk_factors = []
            
            # Look for recommendation and risk factors in the analysis
            in_risk_section = False
            for line in lines:
                line = line.strip()
                if "recommendation" in line.lower() and ("buy" in line.lower() or "sell" in line.lower() or "hold" in line.lower()):
                    recommendation_line = line
                elif "risk factor" in line.lower():
                    in_risk_section = True
                elif in_risk_section and line and not line.startswith('#'):
                    risk_factors.append(line)
                elif in_risk_section and (line.startswith('#') or "investment recommendation" in line.lower()):
                    in_risk_section = False
            
            # Summarize the risks
            risk_summary = '; '.join(risk_factors[:3]) if risk_factors else "No specific risks identified."
            
            # Create a summary for this stock
            summary = f"{symbol}: {recommendation_line if recommendation_line else 'No clear recommendation'} | Risks: {risk_summary}"
            stock_summaries.append(summary)
        
        # Prepare API endpoint
        api_endpoint = model_info["api_endpoint"]
        api_url = os.getenv("HUGGINGFACE_API_URL", f"https://api-inference.huggingface.co/models/{api_endpoint}")
        
        # Create prompt for portfolio allocation
        prompt = f"""You are a financial advisor specializing in portfolio optimization for Indian stock markets.

I have analyzed the following stocks:

{chr(10).join(stock_summaries)}

Based on these analyses, suggest an optimal portfolio allocation with specific percentages for each stock. 
Maximize returns while managing risk appropriately. Explain your rationale for each allocation decision.
Take into account the recommendations, risks, and potential of each stock.
The percentages must sum to 100%.
"""
        
        # Call the Hugging Face API
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {"inputs": prompt}
        
        # Make the API request
        response = requests.post(api_url, headers=headers, json=payload)
        
        # Process the response
        if response.status_code != 200:
            return f"Error: API request failed with status code {response.status_code}. Details: {response.text}"
        
        # Extract the generated text from the response
        response_data = response.json()
        
        # Handle different response formats from different models
        if isinstance(response_data, list) and len(response_data) > 0:
            if isinstance(response_data[0], dict) and "generated_text" in response_data[0]:
                # Format for models like Mistral and some others
                allocation = response_data[0]["generated_text"]
            else:
                # Some other list format
                allocation = str(response_data[0])
        elif isinstance(response_data, dict):
            if "generated_text" in response_data:
                # Format for some models
                allocation = response_data["generated_text"]
            else:
                # Some other dictionary format
                allocation = str(response_data)
        else:
            # Fallback - use the raw response
            allocation = str(response_data)
        
        # Clean up the response
        if prompt in allocation:
            # Remove the prompt from the beginning of the response if present
            allocation = allocation[len(prompt):].strip()
        
        # If the response is empty, return an error
        if not allocation:
            return "Error: The model returned an empty response. Please try a different model."
        
        return allocation
    
    except Exception as e:
        return f"Error during portfolio allocation: {str(e)}"

def create_analysis_prompt(stock_info, df_with_indicators):
    """
    Create a prompt for the AI model to analyze stock data.
    
    Args:
        stock_info (dict): Dictionary containing basic stock information
        df_with_indicators (pandas.DataFrame): DataFrame with stock data and technical indicators
        
    Returns:
        str: Formatted prompt for the AI model
    """
    # Extract the most recent data point
    try:
        latest_data = df_with_indicators.iloc[-1].to_dict()
        
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
        price_trend_30d = ((current_price / df_with_indicators['Close'].iloc[-30]) - 1) * 100 if len(df_with_indicators) >= 30 else 'N/A'
        price_trend_90d = ((current_price / df_with_indicators['Close'].iloc[-90]) - 1) * 100 if len(df_with_indicators) >= 90 else 'N/A'
        
        # Format price trends as strings
        price_trend_30d_str = f"{price_trend_30d:.2f}%" if isinstance(price_trend_30d, (int, float)) else price_trend_30d
        price_trend_90d_str = f"{price_trend_90d:.2f}%" if isinstance(price_trend_90d, (int, float)) else price_trend_90d
        
        # Create the prompt
        prompt = f"""You are a financial analyst specializing in Indian stock markets (BSE and NSE). Provide a comprehensive financial analysis based on the data provided.

Analyze the following stock:

Stock Information:
- Symbol: {stock_info.get('symbol', 'N/A')}
- Name: {stock_info.get('name', 'N/A')}
- Sector: {stock_info.get('sector', 'N/A')}
- Industry: {stock_info.get('industry', 'N/A')}
- Current Price: {stock_info.get('current_price', 'N/A')}
- Market Cap: {stock_info.get('market_cap', 'N/A')}
- P/E Ratio: {stock_info.get('pe_ratio', 'N/A')}
- EPS: {stock_info.get('eps', 'N/A')}
- Dividend Yield: {stock_info.get('dividend_yield', 'N/A')}
- 52-Week High: {stock_info.get('52_week_high', 'N/A')}
- 52-Week Low: {stock_info.get('52_week_low', 'N/A')}
- Average Volume: {stock_info.get('avg_volume', 'N/A')}

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
        return prompt
    except Exception as e:
        # Return a simple prompt if the detailed one can't be created
        return f"""You are a financial analyst specializing in Indian stock markets (BSE and NSE). 
Analyze the stock with symbol {stock_info.get('symbol', 'N/A')} based on the available information.
Provide an investment recommendation (Buy, Hold, or Sell) with rationale.""" 