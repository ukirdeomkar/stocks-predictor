# Indian Stock Market Analyzer

A Streamlit application that fetches real-time data from Indian stock markets (BSE and NSE) using yfinance and provides AI-powered analysis to help maximize investment profits.

## Features

- Real-time stock data from BSE and NSE
- Historical price charts and technical indicators
- AI analysis of stock performance and predictions using Deepseek (free, open-source LLM)
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


## AI Model

This application uses the Deepseek Coder 6.7B model, a free open-source LLM that runs locally on your machine:

- **No API costs**: Unlike OpenAI models, there are no usage fees
- **Privacy**: All analysis is performed locally on your machine
- **Customizability**: You can change the model in the utils.py file if needed

### System Requirements

- **Minimum**: 16GB RAM, modern CPU
- **Recommended**: 32GB RAM, NVIDIA GPU with 8GB+ VRAM

The first analysis may take a few minutes as the model is downloaded and loaded. Subsequent analyses will be faster.

## How to Use

1. Enter the stock symbols you're interested in (use the format `SYMBOL.BO` for BSE stocks and `SYMBOL.NS` for NSE stocks)
2. Select the timeframe for analysis
3. Choose the type of analysis you want
4. View the results and AI-generated insights

## Notes

- Stock market investments involve risk, always do additional research before making investment decisions
- If you have limited system resources, you can modify the MODEL_ID in utils.py to use a smaller model like TinyLlama 