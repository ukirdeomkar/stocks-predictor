# Indian Stock Market Analyzer

A Streamlit application that fetches real-time data from Indian stock markets (BSE and NSE) using yfinance and provides AI-powered analysis to help maximize investment profits.

## Features

- Real-time stock data from BSE and NSE
- Historical price charts and technical indicators
- Multiple AI models to choose from for stock analysis, all running locally
- Comparison tools for multiple stocks
- Portfolio optimization suggestions

## Setup Instructions

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

## AI Models

This application offers multiple free, open-source LLMs that run locally on your machine:

| Model | Description | System Requirements |
|-------|-------------|---------------------|
| phi-2 | Microsoft's efficient model (2.7B parameters) | 6GB+ RAM |
| tinyllama-1.1b | Very small general model (1.1B parameters) | 4GB+ RAM |
| deepseek-coder-1.3b | Lightweight coding model (1.3B parameters) | 4GB+ RAM |
| stablelm-zephyr-3b | Stability AI's 3B model tuned on Zephyr data | 6GB+ RAM |
| llama3-8b | Meta's Llama 3 model (8B parameters) | 16GB+ RAM |
| deepseek-coder-6.7b | Powerful coding model (6.7B parameters) | 16GB+ RAM |

### Key Benefits

- **No API costs**: Unlike OpenAI models, there are no usage fees
- **Privacy**: All analysis is performed locally on your machine
- **Model flexibility**: Choose the model that works best for your hardware

The first analysis may take a few minutes as the selected model is downloaded and loaded. Subsequent analyses will be faster.

## How to Use

1. Select your preferred AI model from the sidebar based on your hardware capabilities
2. Enter the stock symbols you're interested in (use the format `SYMBOL.BO` for BSE stocks and `SYMBOL.NS` for NSE stocks)
3. Select the timeframe for analysis
4. Choose the type of analysis you want
5. View the results and AI-generated insights

## Notes

- Stock market investments involve risk, always do additional research before making investment decisions
- For optimal performance on Streamlit's free deployment tier, use the smaller models (phi-2, tinyllama-1.1b, or deepseek-coder-1.3b)
- If using more powerful models, you may need to deploy the application on your own infrastructure 