"""
Component for the stock comparison tab
"""

import streamlit as st
import pandas as pd
from ..utils.stock_data import validate_ticker
from ..utils.analysis import compare_stocks
from ..utils.plotting import create_comparison_chart, create_returns_comparison, create_risk_return_plot

def create_stock_comparison_tab():
    """
    Create the stock comparison tab content
    """
    st.markdown('<h2 class="sub-header">Compare Multiple Stocks</h2>', unsafe_allow_html=True)
    
    # Input for multiple tickers
    ticker_input = st.text_input(
        'Enter multiple stock symbols separated by commas (e.g., RELIANCE.NS, TCS.NS, HDFCBANK.NS)',
        key='comparison_tickers'
    )
    
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
                comparison_df, normalized_prices, performance_metrics = compare_stocks(
                    valid_tickers, 
                    period=st.session_state.period
                )
                
                # Display comparison table
                st.markdown('<h3 class="sub-header">Performance Comparison</h3>', unsafe_allow_html=True)
                st.dataframe(comparison_df)
                
                # Display price chart
                st.markdown('<h3 class="sub-header">Normalized Price Comparison (Base=100)</h3>', unsafe_allow_html=True)
                
                # Create price comparison chart
                fig = create_comparison_chart(pd.DataFrame(normalized_prices))
                st.plotly_chart(fig, use_container_width=True)
                
                # Display relative strength chart
                st.markdown('<h3 class="sub-header">Returns Comparison</h3>', unsafe_allow_html=True)
                
                # Create returns comparison chart
                returns_data = {ticker: comparison_df.loc[ticker, 'Total Return %'] for ticker in valid_tickers}
                fig = create_returns_comparison(returns_data)
                st.plotly_chart(fig, use_container_width=True)
                
                # Risk vs. Return scatter plot
                st.markdown('<h3 class="sub-header">Risk vs. Return</h3>', unsafe_allow_html=True)
                
                risk_return_data = pd.DataFrame({
                    'Stock': valid_tickers,
                    'Return (%)': [comparison_df.loc[t, 'Total Return %'] for t in valid_tickers],
                    'Risk (%)': [comparison_df.loc[t, 'Volatility %'] for t in valid_tickers],
                    'Sharpe Ratio': [comparison_df.loc[t, 'Sharpe Ratio'] for t in valid_tickers]
                })
                
                fig = create_risk_return_plot(risk_return_data)
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Enter multiple stock symbols separated by commas to compare them") 