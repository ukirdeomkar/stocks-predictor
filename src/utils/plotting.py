"""
Utility functions for plotting stock data
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

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
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    )])
    
    # Add volume as bar chart
    fig.add_trace(go.Bar(
        x=df.index, 
        y=df['Volume'], 
        name='Volume', 
        yaxis='y2', 
        opacity=0.3
    ))
    
    # Add moving averages
    fig.add_trace(go.Scatter(
        x=df.index, 
        y=df['MA20'], 
        mode='lines', 
        name='20-day MA', 
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=df.index, 
        y=df['MA50'], 
        mode='lines', 
        name='50-day MA', 
        line=dict(color='orange')
    ))
    fig.add_trace(go.Scatter(
        x=df.index, 
        y=df['MA200'], 
        mode='lines', 
        name='200-day MA', 
        line=dict(color='red')
    ))
    
    # Add Bollinger Bands
    fig.add_trace(go.Scatter(
        x=df.index, 
        y=df['upper_band'], 
        mode='lines', 
        name='Upper Band', 
        line=dict(color='rgba(0,128,0,0.3)')
    ))
    fig.add_trace(go.Scatter(
        x=df.index, 
        y=df['lower_band'], 
        mode='lines', 
        name='Lower Band', 
        line=dict(color='rgba(0,128,0,0.3)')
    ))
    
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

def create_comparison_chart(normalized_prices):
    """
    Create a comparison chart for multiple stocks
    
    Args:
        normalized_prices (pd.DataFrame): DataFrame with normalized prices for multiple stocks
        
    Returns:
        go.Figure: Plotly figure object
    """
    fig = px.line(
        normalized_prices, 
        x=normalized_prices.index, 
        y=normalized_prices.columns,
        title="Price Performance Comparison (Normalized to 100)",
        labels={"value": "Normalized Price", "variable": "Stock"}
    )
    
    fig.update_layout(
        height=500,
        legend=dict(orientation='h', y=1.05),
        margin=dict(l=50, r=50, t=100, b=50),
    )
    
    return fig

def create_returns_comparison(returns_data):
    """
    Create a bar chart comparing returns for multiple stocks
    
    Args:
        returns_data (dict): Dictionary with stock tickers as keys and returns as values
        
    Returns:
        go.Figure: Plotly figure object
    """
    fig = px.bar(
        x=list(returns_data.keys()),
        y=list(returns_data.values()),
        labels={'x': 'Stock', 'y': 'Total Return (%)'},
        title="Total Returns Comparison"
    )
    
    fig.update_layout(
        height=400,
        margin=dict(l=50, r=50, t=100, b=50),
    )
    
    return fig

def create_risk_return_plot(risk_return_data):
    """
    Create a risk vs return scatter plot
    
    Args:
        risk_return_data (pd.DataFrame): DataFrame with 'Stock', 'Return (%)', 'Risk (%)', and 'Sharpe Ratio' columns
        
    Returns:
        go.Figure: Plotly figure object
    """
    fig = px.scatter(
        risk_return_data,
        x='Risk (%)',
        y='Return (%)',
        size='Sharpe Ratio',
        hover_name='Stock',
        text='Stock',
        title="Risk vs. Return Analysis"
    )
    
    fig.update_layout(
        height=500,
        margin=dict(l=50, r=50, t=100, b=50),
    )
    
    return fig 