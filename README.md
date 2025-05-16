# Indian Stock Market Analyzer

An AI-powered Streamlit application for analyzing Indian stocks from BSE and NSE using real-time data from Yahoo Finance through the `yfinance` library. The application leverages the Hugging Face Inference API to provide AI-driven stock analysis and portfolio allocation recommendations.

## Features

- Real-time stock data from BSE and NSE
- Multiple technical indicators (RSI, MACD, Bollinger Bands, etc.)
- AI-driven stock analysis via Hugging Face's API
- Portfolio allocation recommendations
- Interactive visualization of stock prices and technical indicators
- Fully responsive UI for desktop and mobile use

## Analysis through Hugging Face API

This application uses the Hugging Face Inference API to provide AI-powered stock analysis. The API offers:

- **Speed**: Get analysis in seconds instead of minutes
- **No hardware requirements**: Run powerful AI models without a high-end GPU
- **Generous free tier**: The Hugging Face API offers a free tier with reasonable quotas
- **Access to powerful models**: Get analysis from various AI models without downloading them

The Hugging Face API is required to use this application, as it exclusively relies on API calls for AI analysis.

## Available AI Models

| Model | Description | Best For |
|-------|-------------|----------|
| deepseek-coder-6.7b | A specialized model for financial and code analysis | Detailed technical analysis |
| mistral-7b | Well-balanced general-purpose model | Balanced analysis |
| phi-2 | Small but efficient model with good financial understanding | Quick insights |
| tinyllama-1.1b | Ultra-compact model for very basic analysis | Very basic summaries |

## Setup

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. **Setup your Hugging Face API Key (Required)**:
   - Create a free account at [Hugging Face](https://huggingface.co/)
   - Generate an API key from your [settings page](https://huggingface.co/settings/tokens)
   - Create a `.env` file in the app's directory with:
     ```
     HUGGINGFACE_API_KEY=your_api_key_here
     ```

## How to Use

1. Run the application with:
   ```
   streamlit run app.py
   ```
2. Select an AI model from the dropdown in the sidebar
3. Enter a valid BSE or NSE ticker symbol:
   - For BSE stocks, add `.BO` suffix (e.g., `RELIANCE.BO`)
   - For NSE stocks, add `.NS` suffix (e.g., `RELIANCE.NS`)
4. Click "Analyze" to fetch stock data and visualize the charts
5. Click "Generate AI Analysis" to get AI-driven insights about the stock
6. For multi-stock analysis, enter multiple ticker symbols separated by commas

## Important Notes

- The application requires a working internet connection to fetch real-time data and access the Hugging Face API
- Stock market investments involve risk; use the analysis as one of many inputs for your investment decisions
- The API has usage limits on the free tier; for high-volume usage, consider upgrading your Hugging Face account
- For the best experience, select models appropriate for your analysis needs

## License

MIT License 