"""
Portfolio allocation component for the main page
"""

import streamlit as st
import os
from ..utils.analysis import suggest_portfolio_allocation
import re

def show_portfolio_allocation_section(selected_model):
    """
    Show portfolio allocation section on the main page if analysis results are available
    
    Args:
        selected_model (str): Selected model key
    """
    # Check if we have analysis results to work with
    if 'analysis_results' in st.session_state and st.session_state.analysis_results:
        # Only show if we have at least 2 analyzed stocks
        if len(st.session_state.analysis_results) >= 2:
            st.markdown('<h3 class="sub-header">Portfolio Allocation Suggestion</h3>', unsafe_allow_html=True)
            
            # Check if Hugging Face API key is available
            hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
            api_available = hf_api_key is not None and hf_api_key.strip() != ""
            
            if not api_available:
                st.error("❌ No Hugging Face API key found. You must set up an API key to use the portfolio allocation feature.")
            else:
                st.info(f"Based on your previous stock analyses, you can generate an optimal portfolio allocation using {selected_model}.")
                st.write(f"Analyzed stocks: {', '.join(st.session_state.analysis_results.keys())}")
                
                if st.button("Generate Portfolio Allocation Suggestion", key="main_portfolio_button"):
                    with st.spinner(f"Generating portfolio allocation using {selected_model} via Hugging Face API..."):
                        try:
                            # Get the portfolio allocation suggestion
                            portfolio_allocation = suggest_portfolio_allocation(
                                st.session_state.analysis_results,
                                model_key=selected_model
                            )
                            
                            # Check if portfolio allocation was successful
                            if portfolio_allocation and not portfolio_allocation.startswith("Error"):
                                st.success("Portfolio Allocation Suggestion Generated")
                                
                                # Create a container with proper styling
                                st.markdown("<div class='allocation-box'>", unsafe_allow_html=True)
                                
                                # Format the portfolio allocation text
                                formatted_allocation = portfolio_allocation.replace("\n", "<br>")
                                
                                # Handle potential Markdown formatting
                                formatted_allocation = re.sub(r'<(?!br>|/br>)', '&lt;', formatted_allocation)
                                formatted_allocation = re.sub(r'(?<!<br)>', '&gt;', formatted_allocation)
                                
                                st.markdown(f"{formatted_allocation}", unsafe_allow_html=True)
                                st.markdown("</div>", unsafe_allow_html=True)
                            else:
                                st.error(f"Error generating portfolio allocation: {portfolio_allocation}")
                        except Exception as e:
                            st.error(f"An error occurred: {str(e)}") 