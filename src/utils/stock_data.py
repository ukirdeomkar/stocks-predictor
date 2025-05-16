"""
Utility functions for fetching and processing stock data
"""

import pandas as pd
import yfinance as yf

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
    except Exception:
        return False

def get_stock_data(ticker, period="1y", interval="1d"):
    """
    Get historical stock data with technical indicators
    
    Args:
        ticker (str): Stock ticker symbol
        period (str, optional): Time period for data (e.g., "1y", "6mo", "1d"). Defaults to "1y".
        interval (str, optional): Data interval (e.g., "1d", "1wk"). Defaults to "1d".
        
    Returns:
        pd.DataFrame: DataFrame with stock data and technical indicators
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        
        if df.empty:
            raise ValueError(f"No data returned for ticker {ticker}")
            
        # Calculate technical indicators
        df = calculate_technical_indicators(df)
        
        return df
    except Exception as e:
        print(f"Error getting stock data: {str(e)}")
        return pd.DataFrame()

def get_stock_info(ticker):
    """
    Get basic information about a stock
    
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

def calculate_technical_indicators(df):
    """
    Calculate technical indicators for stock data
    
    Args:
        df (pd.DataFrame): DataFrame with stock data
        
    Returns:
        pd.DataFrame: DataFrame with additional technical indicators
    """
    # Calculate moving averages
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
    df['std'] = df['Close'].rolling(window=20).std()
    df['upper_band'] = df['MA20'] + (df['std'] * 2)
    df['lower_band'] = df['MA20'] - (df['std'] * 2)
    
    # Fill NaN values using forward fill followed by backward fill
    # This replaces the deprecated fillna(method='bfill')
    df = df.ffill().bfill()
    
    return df 