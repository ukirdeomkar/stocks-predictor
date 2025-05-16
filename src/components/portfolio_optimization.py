"""
Component for the portfolio optimization tab
"""

import streamlit as st
from ..utils.stock_data import validate_ticker, get_stock_data, get_stock_info
from ..utils.analysis import analyze_with_llm, suggest_portfolio_allocation
from ..models.model_info import get_available_models
import os

def create_portfolio_optimization_tab(selected_model):
    """
    Create the portfolio optimization tab content
    
    Args:
        selected_model (str): The selected model key
    """
    st.markdown('<h2 class="sub-header">Portfolio Optimization</h2>', unsafe_allow_html=True)
    
    # Input for portfolio tickers
    portfolio_input = st.text_input(
        'Enter stock symbols for your portfolio (separated by commas)',
        key='portfolio_tickers'
    )
    
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
                
                # Get available models
                available_models = get_available_models()
                
                st.info(f"Using model: **{selected_model}** - {available_models[selected_model]['description']} (via Hugging Face API)")
                
                # Check if Hugging Face API key is available
                hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
                api_available = hf_api_key is not None and hf_api_key.strip() != ""

                if not api_available:
                    st.error("❌ No Hugging Face API key found. You must set up an API key to use the portfolio optimization feature.")
                    st.markdown("""
                    To set up your API key:
                    1. Create a free account at [Hugging Face](https://huggingface.co/)
                    2. Generate an API key from [your settings page](https://huggingface.co/settings/tokens)
                    3. Create a `.env` file in the app's directory with: `HUGGINGFACE_API_KEY=your_api_key_here`
                    4. Restart the application
                    """)
                else:
                    if st.button("Generate Portfolio Recommendations"):
                        with st.spinner(f"Analyzing your portfolio using {selected_model}... This may take a few minutes..."):
                            try:
                                # Gather analysis results for each stock
                                analysis_results = {}
                                progress_bar = st.progress(0)
                                analysis_errors = []
                                
                                for i, ticker in enumerate(valid_portfolio_tickers):
                                    # Update progress
                                    progress_bar.progress((i / len(valid_portfolio_tickers)))
                                    
                                    try:
                                        # Get stock data and info
                                        stock_info = get_stock_info(ticker)
                                        df = get_stock_data(ticker, period='1y')
                                        
                                        if not df.empty:
                                            # Get AI analysis
                                            analysis = analyze_with_llm(
                                                stock_info, 
                                                df, 
                                                model_key=selected_model
                                            )
                                            
                                            if analysis and not analysis.startswith("Error"):
                                                analysis_results[ticker] = analysis
                                            else:
                                                analysis_errors.append(f"Could not analyze {ticker}: {analysis}")
                                        else:
                                            analysis_errors.append(f"Could not fetch data for {ticker}")
                                    except Exception as e:
                                        analysis_errors.append(f"Error analyzing {ticker}: {str(e)}")
                                
                                # Complete progress
                                progress_bar.progress(1.0)
                                
                                # Show any errors that occurred during analysis
                                if analysis_errors:
                                    st.warning(f"Some stocks could not be analyzed: {', '.join(analysis_errors)}")
                                
                                if analysis_results:
                                    # Store the analysis results in session state
                                    st.session_state.analysis_results = analysis_results
                                    
                                    # Get portfolio allocation suggestion
                                    try:
                                        allocation_suggestion = suggest_portfolio_allocation(
                                            analysis_results, 
                                            model_key=selected_model
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
                            except Exception as e:
                                st.error(f"An error occurred: {str(e)}")
    else:
        st.info("Enter the stock symbols for your portfolio to get optimization recommendations") 