"""
Utility functions for stock analysis and portfolio recommendations
"""

import pandas as pd
from ..api.huggingface import analyze_with_huggingface_api
from ..models.model_info import DEFAULT_MODEL, get_available_models
from .stock_data import get_stock_data, get_stock_info, calculate_technical_indicators

def create_analysis_prompt(stock_info, df_with_indicators, model_key=DEFAULT_MODEL):
    """
    Create a prompt for the AI model to analyze stock data
    
    Args:
        stock_info (dict): Dictionary containing basic stock information
        df_with_indicators (pandas.DataFrame): DataFrame with stock data and technical indicators
        model_key (str): The key of the model to use
        
    Returns:
        str: Formatted prompt for the AI model
    """
    # Different model categories require different prompts
    # Classification and sentiment models
    classification_models = ["finbert", "finbert-tone", "finbert-pretrain", "bert-sentiment", "bert-base"]
    
    # Text generation models
    generation_models = ["bart-cnn", "flan-t5"]
    
    # Question answering models
    qa_models = ["roberta-squad"]
    
    # Zero-shot classification models
    zeroshot_models = ["bart-mnli", "deberta-zeroshot"]
    
    # Embedding and similarity models
    embedding_models = ["sentence-transformer", "mpnet-sentence", "codeberta"]
    
    # Long-form document models
    long_document_models = ["longformer"]
    
    try:
        # Extract the latest data point
        latest_data = df_with_indicators.iloc[-1].to_dict() if not df_with_indicators.empty else {}
        
        # Basic prompt information that's needed for all models
        stock_symbol = stock_info.get('symbol', 'N/A')
        stock_name = stock_info.get('name', 'N/A')
        current_price = latest_data.get('Close', stock_info.get('current_price', 'N/A'))
        sector = stock_info.get('sector', 'N/A')
        industry = stock_info.get('industry', 'N/A')
        
        # Technical indicators
        rsi_value = latest_data.get('RSI', 'N/A')
        macd_value = latest_data.get('MACD', 'N/A')
        signal_value = latest_data.get('Signal', 'N/A')
        ma20 = latest_data.get('MA20', 'N/A')
        ma50 = latest_data.get('MA50', 'N/A')
        ma200 = latest_data.get('MA200', 'N/A')
        
        # For classification and sentiment models (short, focused prompts)
        if model_key in classification_models:
            # Calculate performance metrics
            if not df_with_indicators.empty and len(df_with_indicators) > 1:
                start_price = df_with_indicators['Close'].iloc[0]
                percent_change = ((current_price - start_price) / start_price) * 100 if isinstance(current_price, (int, float)) and isinstance(start_price, (int, float)) else 'N/A'
            else:
                percent_change = 'N/A'
            
            # Check if price is above or below moving averages
            above_ma50 = current_price > ma50 if isinstance(current_price, (int, float)) and isinstance(ma50, (int, float)) else False
            above_ma200 = current_price > ma200 if isinstance(current_price, (int, float)) and isinstance(ma200, (int, float)) else False
            
            # Create a concise sentiment prompt
            sentiment_prompt = f"{stock_name} ({stock_symbol}) stock analysis: "
            
            # Add price movement
            if isinstance(percent_change, (int, float)):
                if percent_change > 0:
                    sentiment_prompt += f"Stock price increased by {percent_change:.2f}% during the period. "
                else:
                    sentiment_prompt += f"Stock price decreased by {abs(percent_change):.2f}% during the period. "
            
            # Add technical indicators
            if isinstance(rsi_value, (int, float)):
                if rsi_value > 70:
                    sentiment_prompt += f"RSI is overbought at {rsi_value:.2f}. "
                elif rsi_value < 30:
                    sentiment_prompt += f"RSI is oversold at {rsi_value:.2f}. "
                else:
                    sentiment_prompt += f"RSI is neutral at {rsi_value:.2f}. "
            
            if isinstance(macd_value, (int, float)) and isinstance(signal_value, (int, float)):
                if macd_value > signal_value:
                    sentiment_prompt += "MACD is above signal line indicating bullish momentum. "
                else:
                    sentiment_prompt += "MACD is below signal line indicating bearish momentum. "
            
            # Add moving average information
            if above_ma50 and above_ma200:
                sentiment_prompt += "Price is above both 50-day and 200-day moving averages. "
            elif above_ma50:
                sentiment_prompt += "Price is above 50-day but below 200-day moving average. "
            elif above_ma200:
                sentiment_prompt += "Price is below 50-day but above 200-day moving average. "
            else:
                sentiment_prompt += "Price is below both 50-day and 200-day moving averages. "
            
            # Add sector and market cap information
            if sector != "Unknown" and sector != "N/A":
                sentiment_prompt += f"The stock belongs to {sector} sector. "
            
            if stock_info.get('market_cap', 'N/A') not in ['Unknown', 'N/A']:
                sentiment_prompt += f"Market cap: {stock_info.get('market_cap')}. "
            
            # Add P/E ratio if available
            if stock_info.get('pe_ratio', 'N/A') not in ['Unknown', 'N/A']:
                sentiment_prompt += f"P/E ratio: {stock_info.get('pe_ratio')}. "
                
            return sentiment_prompt
        
        # For zero-shot classification models
        elif model_key in zeroshot_models:
            # Create a structured prompt for zero-shot models
            zeroshot_prompt = f"Based on the following financial data, should I buy, hold, or sell {stock_name} ({stock_symbol}) stock?\n\n"
            
            # Add structured data
            zeroshot_prompt += f"Stock: {stock_name} ({stock_symbol})\n"
            zeroshot_prompt += f"Sector: {sector}\n"
            zeroshot_prompt += f"Current Price: {current_price}\n"
            
            # Add technical indicators
            if isinstance(rsi_value, (int, float)):
                zeroshot_prompt += f"RSI: {rsi_value:.2f} "
                if rsi_value > 70:
                    zeroshot_prompt += "(Overbought)\n"
                elif rsi_value < 30:
                    zeroshot_prompt += "(Oversold)\n"
                else:
                    zeroshot_prompt += "(Neutral)\n"
            
            # MACD
            if isinstance(macd_value, (int, float)) and isinstance(signal_value, (int, float)):
                zeroshot_prompt += f"MACD: {macd_value:.2f}, Signal: {signal_value:.2f} "
                if macd_value > signal_value:
                    zeroshot_prompt += "(Bullish)\n"
                else:
                    zeroshot_prompt += "(Bearish)\n"
            
            # Moving Averages
            zeroshot_prompt += "Moving Averages: "
            if above_ma50 and above_ma200:
                zeroshot_prompt += "Price above 50-day and 200-day MAs (Strong Bullish)\n"
            elif above_ma50:
                zeroshot_prompt += "Price above 50-day MA but below 200-day MA (Moderately Bullish)\n"
            elif above_ma200:
                zeroshot_prompt += "Price below 50-day MA but above 200-day MA (Mixed Signals)\n"
            else:
                zeroshot_prompt += "Price below 50-day and 200-day MAs (Bearish)\n"
            
            # Additional metrics
            zeroshot_prompt += f"P/E Ratio: {stock_info.get('pe_ratio', 'N/A')}\n"
            zeroshot_prompt += f"EPS: {stock_info.get('eps', 'N/A')}\n"
            zeroshot_prompt += f"Dividend Yield: {stock_info.get('dividend_yield', 'N/A')}\n"
            
            return zeroshot_prompt
        
        # For QA models
        elif model_key in qa_models:
            # Create a context for QA models
            qa_prompt = f"Context: Analysis of {stock_name} ({stock_symbol}) stock.\n\n"
            
            # Add structured data as context
            qa_prompt += "Financial Data:\n"
            qa_prompt += f"- Current Price: {current_price}\n"
            qa_prompt += f"- Sector: {sector}\n"
            qa_prompt += f"- Industry: {industry}\n"
            qa_prompt += f"- P/E Ratio: {stock_info.get('pe_ratio', 'N/A')}\n"
            qa_prompt += f"- EPS: {stock_info.get('eps', 'N/A')}\n"
            qa_prompt += f"- Market Cap: {stock_info.get('market_cap', 'N/A')}\n"
            
            # Technical indicators
            qa_prompt += "\nTechnical Indicators:\n"
            qa_prompt += f"- RSI: {rsi_value}\n"
            qa_prompt += f"- MACD: {macd_value}\n"
            qa_prompt += f"- Signal Line: {signal_value}\n"
            qa_prompt += f"- 20-day MA: {ma20}\n"
            qa_prompt += f"- 50-day MA: {ma50}\n"
            qa_prompt += f"- 200-day MA: {ma200}\n"
            
            # Add analysis questions
            qa_prompt += "\nBased on this data, what is the investment recommendation for this stock? Should I buy, hold, or sell? Explain your reasoning based on the technical indicators and financial metrics."
            
            return qa_prompt
            
        # For embedding models
        elif model_key in embedding_models:
            # Create a simple, clean prompt for embedding models
            embedding_prompt = f"{stock_name} ({stock_symbol}) Stock Analysis:\n"
            embedding_prompt += f"Price: {current_price}, Sector: {sector}, Industry: {industry}\n"
            embedding_prompt += f"P/E: {stock_info.get('pe_ratio', 'N/A')}, EPS: {stock_info.get('eps', 'N/A')}\n"
            
            # Technical indicators
            embedding_prompt += f"RSI: {rsi_value}, MACD: {macd_value}, Signal: {signal_value}\n"
            embedding_prompt += f"MA20: {ma20}, MA50: {ma50}, MA200: {ma200}\n"
            
            # Add simple questions for similarity comparison
            embedding_prompt += "\nIs this stock a good investment?\n"
            embedding_prompt += "What are the key financial metrics for this stock?\n"
            embedding_prompt += "Should I buy, hold, or sell this stock?\n"
            embedding_prompt += "What are the risks associated with this stock?\n"
            
            return embedding_prompt
            
        # For long document models
        elif model_key in long_document_models:
            # Create a comprehensive prompt for models that can handle long contexts
            longdoc_prompt = f"# Comprehensive Analysis of {stock_name} ({stock_symbol})\n\n"
            
            # Company overview
            longdoc_prompt += "## Company Overview\n"
            longdoc_prompt += f"- Name: {stock_name}\n"
            longdoc_prompt += f"- Ticker Symbol: {stock_symbol}\n"
            longdoc_prompt += f"- Sector: {sector}\n"
            longdoc_prompt += f"- Industry: {industry}\n\n"
            
            # Financial metrics
            longdoc_prompt += "## Financial Metrics\n"
            longdoc_prompt += f"- Current Price: {current_price}\n"
            longdoc_prompt += f"- Market Cap: {stock_info.get('market_cap', 'N/A')}\n"
            longdoc_prompt += f"- P/E Ratio: {stock_info.get('pe_ratio', 'N/A')}\n"
            longdoc_prompt += f"- EPS: {stock_info.get('eps', 'N/A')}\n"
            longdoc_prompt += f"- Dividend Yield: {stock_info.get('dividend_yield', 'N/A')}\n"
            longdoc_prompt += f"- 52-Week High: {stock_info.get('52_week_high', 'N/A')}\n"
            longdoc_prompt += f"- 52-Week Low: {stock_info.get('52_week_low', 'N/A')}\n"
            longdoc_prompt += f"- Average Volume: {stock_info.get('avg_volume', 'N/A')}\n\n"
            
            # Technical analysis
            longdoc_prompt += "## Technical Indicators\n"
            longdoc_prompt += f"- Current Price: {current_price}\n"
            longdoc_prompt += f"- 20-Day Moving Average: {ma20}\n"
            longdoc_prompt += f"- 50-Day Moving Average: {ma50}\n"
            longdoc_prompt += f"- 200-Day Moving Average: {ma200}\n"
            longdoc_prompt += f"- RSI (14-day): {rsi_value}\n"
            longdoc_prompt += f"- MACD: {macd_value}\n"
            longdoc_prompt += f"- MACD Signal: {signal_value}\n\n"
            
            # Add price history summary if available
            if not df_with_indicators.empty and len(df_with_indicators) > 30:
                try:
                    # Get monthly data points
                    monthly_data = df_with_indicators.iloc[::20]  # Every 20 days ~ monthly
                    
                    longdoc_prompt += "## Price History\n"
                    for i, (date, row) in enumerate(monthly_data.iterrows()):
                        if i < 6:  # Limit to 6 months
                            longdoc_prompt += f"- {date.strftime('%Y-%m-%d')}: Close ${row['Close']:.2f}, Volume {row['Volume']:.0f}\n"
                    longdoc_prompt += "\n"
                except Exception as e:
                    print(f"Error creating price history: {str(e)}")
            
            # Analysis request
            longdoc_prompt += "## Analysis Request\n"
            longdoc_prompt += "Based on the comprehensive data provided above, please provide:\n\n"
            longdoc_prompt += "1. A detailed technical analysis of the stock's current position\n"
            longdoc_prompt += "2. An interpretation of all key indicators and what they suggest about future price movement\n"
            longdoc_prompt += "3. A fundamental analysis based on financial metrics\n"
            longdoc_prompt += "4. A clear investment recommendation (Buy, Hold, or Sell) with supporting rationale\n"
            longdoc_prompt += "5. Key risk factors that could impact this recommendation\n"
            longdoc_prompt += "6. Suggested entry/exit points and position sizing guidelines\n\n"
            
            longdoc_prompt += "Please provide this analysis with specific attention to Indian market context and factors relevant to BSE/NSE markets."
            
            return longdoc_prompt
        
        # For all other models (including text generation models)
        else:
            # Create a standard financial analysis prompt
            prompt = f"""Analyze the following stock from the Indian market:

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

Technical Indicators:
- Current Price: {current_price}
- 20-Day Moving Average: {ma20}
- 50-Day Moving Average: {ma50}
- 200-Day Moving Average: {ma200}
- RSI (14-day): {rsi_value}
- MACD: {macd_value}
- MACD Signal: {signal_value}

Based on this data, provide:
1. A technical analysis interpretation
2. Potential price movement forecast
3. Clear investment recommendation (Buy, Hold, or Sell)
4. Key risk factors

Focus on Indian market context and BSE/NSE factors.
"""
            return prompt
    
    except Exception as e:
        # Return a simple prompt if the detailed one can't be created
        print(f"Error creating analysis prompt: {str(e)}")
        return f"""Analyze the stock with symbol {stock_info.get('symbol', 'N/A')} based on the available information.
Provide an investment recommendation (Buy, Hold, or Sell) with rationale."""

def analyze_with_llm(stock_info, df_with_indicators, model_key=DEFAULT_MODEL):
    """
    Analyze stock data using a LLM via the Hugging Face API
    
    Args:
        stock_info (dict): Dictionary containing basic stock information
        df_with_indicators (pandas.DataFrame): DataFrame with stock data and technical indicators
        model_key (str): Key of the model to use
        
    Returns:
        str: Analysis of the stock by the LLM, or error message
    """
    # Prepare the prompt based on the stock data and model
    prompt = create_analysis_prompt(stock_info, df_with_indicators, model_key)
    
    # Call the Hugging Face API
    analysis = analyze_with_huggingface_api(prompt, model_key)
    
    # Clean up the response
    if prompt in analysis:
        # Remove the prompt from the beginning of the response if present
        analysis = analysis[len(prompt):].strip()
    
    return analysis

def suggest_portfolio_allocation(analysis_results, model_key=DEFAULT_MODEL):
    """
    Suggest a portfolio allocation based on the analyzed stocks
    
    Args:
        analysis_results (dict): Dictionary with stock symbols as keys and analysis results as values
        model_key (str): Key of the model to use
        
    Returns:
        str: Portfolio allocation suggestion from the model, or error message
    """
    # Validate parameters
    if not analysis_results or len(analysis_results) < 2:
        return "Error: Need at least 2 analyzed stocks to suggest a portfolio allocation."
    
    # Create summary of each analyzed stock
    stock_summaries = []
    for symbol, analysis in analysis_results.items():
        # For classification models, create a simpler summary
        if model_key in ["distilbert", "roberta", "finbert"]:
            stock_summaries.append(f"{symbol}: {analysis}")
            continue
            
        # For text generation models, extract key points from the analysis
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
    
    # Create appropriate prompt based on model type
    if model_key in ["distilbert", "roberta", "finbert"]:
        # For classification models, create a simpler prompt
        prompt = "Based on the following stock analyses, suggest how to allocate a portfolio:\n\n"
        prompt += "\n".join(stock_summaries)
    else:
        # For text generation models
        prompt = f"""You are a financial advisor specializing in portfolio optimization for Indian stock markets.

I have analyzed the following stocks:

{chr(10).join(stock_summaries)}

Based on these analyses, suggest an optimal portfolio allocation with specific percentages for each stock. 
Maximize returns while managing risk appropriately. Explain your rationale for each allocation decision.
Take into account the recommendations, risks, and potential of each stock.
The percentages must sum to 100%.
"""
    
    # Call the Hugging Face API
    allocation = analyze_with_huggingface_api(prompt, model_key)
    
    # Clean up the response
    if prompt in allocation:
        # Remove the prompt from the beginning of the response if present
        allocation = allocation[len(prompt):].strip()
    
    return allocation

def compare_stocks(tickers, period="1y"):
    """
    Compare multiple stocks based on technical indicators and performance
    
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