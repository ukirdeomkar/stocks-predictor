"""
Module to handle interactions with the Hugging Face API
"""

import os
import requests
import time
import json
from ..models.model_info import get_available_models

def analyze_with_huggingface_api(prompt, model_key):
    """
    Send a prompt to the Hugging Face API and get the response
    
    Args:
        prompt (str): The prompt to send to the API
        model_key (str): The key of the model to use
    
    Returns:
        str: The response from the API, or an error message
    """
    # Get API key
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    if not api_key or api_key.strip() == "":
        return "Error: No Hugging Face API key found. Please set up your API key in the .env file."
    
    # Get available models
    available_models = get_available_models()
    
    # Validate model key
    if model_key not in available_models:
        return f"Error: Model {model_key} not found in available models."
    
    try:
        # Get model info
        model_info = available_models[model_key]
        api_endpoint = model_info["api_endpoint"]
        
        # Prepare API URL
        api_url = f"https://api-inference.huggingface.co/models/{api_endpoint}"
        
        # Log which model we're using
        print(f"Using Hugging Face API with model: {api_endpoint}")
        
        # Prepare headers and payload
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Create model-specific payload
        payload = create_model_payload(model_key, prompt)
        
        # Make the API request with retry logic
        max_retries = 3
        retry_count = 0
        retry_delay = 2  # seconds
        
        while retry_count < max_retries:
            try:
                # Make the request with a timeout
                print(f"Sending request to {api_url} with payload: {payload}")
                response = requests.post(api_url, headers=headers, json=payload, timeout=120)
                
                # Print response details for debugging
                print(f"Response status code: {response.status_code}")
                
                # Check for 503 (service unavailable) or 429 (rate limit) which may require retries
                if response.status_code in [503, 429]:
                    retry_count += 1
                    print(f"Received status code {response.status_code}, retrying ({retry_count}/{max_retries})...")
                    if 'retry-after' in response.headers:
                        retry_delay = int(response.headers['retry-after'])
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                
                # Break for other status codes
                break
            except requests.exceptions.Timeout:
                retry_count += 1
                print(f"Request timed out, retrying ({retry_count}/{max_retries})...")
                time.sleep(retry_delay)
                retry_delay *= 2
            except requests.exceptions.ConnectionError:
                return "Error: Could not connect to the Hugging Face API. Please check your internet connection."
        
        # Check response status code
        if response.status_code == 200:
            # Parse the response
            try:
                result = response.json()
                print(f"Raw API response: {result}")
                
                # Process the response based on the model
                return process_model_response(model_key, result, prompt)
            except Exception as e:
                print(f"Error parsing response: {str(e)}")
                return f"Error parsing API response: {str(e)}. Raw response: {response.text}"
        elif response.status_code == 401:
            return "Error: Invalid API key. Please check your Hugging Face API key."
        elif response.status_code == 404:
            return f"Error: Model '{api_endpoint}' not found. This model may not be publicly available or may have been renamed. Please try another model."
        elif response.status_code == 400:
            try:
                error_data = response.json()
                error_message = error_data.get('error', response.text)
                return f"Error: The model rejected your request: {error_message}"
            except:
                return f"Error: Bad request. The API rejected your request: {response.text}"
        else:
            return f"Error: API returned status code {response.status_code}. {response.text}"
        
    except Exception as e:
        print(f"Unexpected error in analyze_with_huggingface_api: {str(e)}")
        return f"Error using Hugging Face API: {str(e)}"

def create_model_payload(model_key, prompt):
    """
    Create a payload appropriate for the model type
    
    Args:
        model_key (str): The model key
        prompt (str): The input prompt
        
    Returns:
        dict: The payload dictionary
    """
    # Classification models (including sentiment analysis)
    classification_models = ["finbert", "finbert-tone", "finbert-pretrain", 
                            "bert-sentiment", "bert-base"]
    
    # Text generation and summarization models
    generation_models = ["bart-cnn", "flan-t5"]
    
    # Question answering models
    qa_models = ["roberta-squad"]
    
    # Zero-shot classification models
    zeroshot_models = ["bart-mnli", "deberta-zeroshot"]
    
    # Sentence embedding models
    embedding_models = ["sentence-transformer", "mpnet-sentence", "codeberta"]
    
    # Long-form document models
    long_document_models = ["longformer"]
    
    # Special models
    special_models = ["whisper"]
    
    if model_key in classification_models:
        # For text classification/sentiment models
        return {
            "inputs": prompt,
            "options": {
                "wait_for_model": True
            }
        }
    elif model_key in generation_models:
        # For text generation/summarization models
        return {
            "inputs": prompt,
            "parameters": {
                "max_length": 512,
                "temperature": 0.7,
                "top_p": 0.95,
                "do_sample": True,
                "return_full_text": False
            },
            "options": {
                "wait_for_model": True
            }
        }
    elif model_key in qa_models:
        # For question answering models
        # Extract a question from the prompt or create a default one
        analysis_question = "What is the investment recommendation for this stock based on the data?"
        
        # Look for specific questions in the prompt
        if "investment recommendation" in prompt.lower():
            analysis_question = "What is the investment recommendation (Buy, Hold, or Sell) for this stock?"
        elif "technical indicator" in prompt.lower():
            analysis_question = "How should I interpret the technical indicators for this stock?"
        
        return {
            "inputs": {
                "question": analysis_question,
                "context": prompt
            },
            "options": {
                "wait_for_model": True
            }
        }
    elif model_key in zeroshot_models:
        # For zero-shot classification models
        candidate_labels = ["Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"]
        
        return {
            "inputs": prompt,
            "parameters": {
                "candidate_labels": candidate_labels
            },
            "options": {
                "wait_for_model": True
            }
        }
    elif model_key in embedding_models:
        # For sentence embedding models
        # Create specific questions for financial analysis
        questions = [
            "Is this stock a good investment?",
            "What are the key financial metrics for this stock?",
            "Should I buy, hold, or sell this stock?",
            "What are the risks associated with this stock?"
        ]
        
        return {
            "inputs": {
                "source_sentence": prompt,
                "sentences": questions
            },
            "options": {
                "wait_for_model": True
            }
        }
    elif model_key in long_document_models:
        # For long document models
        return {
            "inputs": prompt,
            "options": {
                "wait_for_model": True,
                "use_gpu": True
            }
        }
    elif model_key in special_models:
        if model_key == "whisper":
            # Since Whisper is for audio, we'll return a meaningful message
            return {
                "error": "Whisper is designed for audio transcription, not text analysis. Please select a different model for stock analysis."
            }
    else:
        # Default payload for other models
        return {
            "inputs": prompt,
            "options": {
                "wait_for_model": True
            }
        }

def process_model_response(model_key, result, prompt):
    """
    Process the API response based on the model type
    
    Args:
        model_key (str): The model key
        result: The API response result
        prompt (str): The original prompt
        
    Returns:
        str: The processed response
    """
    # Classification models (including sentiment analysis)
    classification_models = ["finbert", "finbert-tone", "finbert-pretrain", 
                            "bert-sentiment", "bert-base"]
    
    # Text generation and summarization models
    generation_models = ["bart-cnn", "flan-t5"]
    
    # Question answering models
    qa_models = ["roberta-squad"]
    
    # Zero-shot classification models
    zeroshot_models = ["bart-mnli", "deberta-zeroshot"]
    
    # Sentence embedding models
    embedding_models = ["sentence-transformer", "mpnet-sentence", "codeberta"]
    
    # Long-form document models
    long_document_models = ["longformer"]
    
    # Special models
    special_models = ["whisper"]
    
    # Error check
    if isinstance(result, dict) and "error" in result:
        return f"Error from model: {result['error']}"
    
    # Handle text generation models
    if model_key in generation_models:
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], dict) and "generated_text" in result[0]:
                return result[0]["generated_text"]
            else:
                return str(result[0])
        elif isinstance(result, dict) and "generated_text" in result:
            return result["generated_text"]
        return str(result)
    
    # Handle classification models
    elif model_key in classification_models:
        if isinstance(result, list) and len(result) > 0:
            # Convert sentiment/classification results to a readable format
            analysis = "Stock Analysis:\n\n"
            
            # Special handling for different sentiment model outputs
            if model_key in ["finbert", "finbert-tone"]:
                # Process FinBERT-specific outputs (positive/negative/neutral)
                for item in result[0]:
                    if isinstance(item, dict) and "label" in item and "score" in item:
                        label = item["label"]
                        score = item["score"]
                        sentiment = label.title()  # Capitalize first letter
                        
                        analysis += f"• {sentiment} sentiment: {score*100:.1f}% confidence\n"
                
                # Add financial interpretation
                positive_sentiment = any("positive" in item.get("label", "").lower() and item.get("score", 0) > 0.5 
                                       for item in result[0] if isinstance(item, dict))
                negative_sentiment = any("negative" in item.get("label", "").lower() and item.get("score", 0) > 0.5 
                                       for item in result[0] if isinstance(item, dict))
                
                analysis += "\nInvestment Recommendation:\n"
                
                if positive_sentiment and not negative_sentiment:
                    analysis += "• Based on sentiment analysis, this stock shows POSITIVE signals\n"
                    analysis += "• Recommendation: Consider BUY or HOLD positions\n"
                    analysis += "• Continue monitoring financial metrics and technical indicators for confirmation"
                elif negative_sentiment and not positive_sentiment:
                    analysis += "• Based on sentiment analysis, this stock shows NEGATIVE signals\n"
                    analysis += "• Recommendation: Consider HOLD or SELL positions\n"
                    analysis += "• Wait for sentiment improvement before establishing new positions"
                else:
                    analysis += "• Based on sentiment analysis, this stock shows MIXED signals\n"
                    analysis += "• Recommendation: HOLD positions and monitor closely\n"
                    analysis += "• Consider technical indicators and fundamental metrics for further guidance"
                
            elif model_key == "bert-sentiment":
                # Process BERT sentiment outputs (star ratings)
                ratings = []
                for item in result[0]:
                    if isinstance(item, dict) and "label" in item and "score" in item:
                        label = item["label"]
                        score = item["score"]
                        ratings.append((label, score))
                
                # Sort by score in descending order
                ratings.sort(key=lambda x: x[1], reverse=True)
                
                # Display top 3 ratings
                for i, (label, score) in enumerate(ratings[:3]):
                    analysis += f"• {label}: {score*100:.1f}% confidence\n"
                
                # Interpret the sentiment scores for investment
                top_rating = ratings[0][0] if ratings else None
                
                analysis += "\nInvestment Recommendation:\n"
                
                if top_rating in ["5 stars", "4 stars"]:
                    analysis += "• Based on sentiment analysis, this stock has a STRONG POSITIVE outlook\n"
                    analysis += "• Recommendation: Consider BUY positions\n"
                elif top_rating == "3 stars":
                    analysis += "• Based on sentiment analysis, this stock has a NEUTRAL outlook\n"
                    analysis += "• Recommendation: HOLD existing positions and monitor\n"
                else:
                    analysis += "• Based on sentiment analysis, this stock has a NEGATIVE outlook\n"
                    analysis += "• Recommendation: Consider SELL positions or avoid new purchases\n"
                
            else:
                # Generic classification label handling
                for item in result[0]:
                    if isinstance(item, dict) and "label" in item and "score" in item:
                        label = item["label"]
                        score = item["score"]
                        
                        analysis += f"• {label}: {score*100:.1f}% confidence\n"
                
                analysis += "\nNote: Further specific financial interpretation not available for this model type."
            
            return analysis
        
        return "No clear classification results were returned."
    
    # Handle question answering models
    elif model_key in qa_models:
        if isinstance(result, dict) and "answer" in result:
            answer = result["answer"]
            score = result.get("score", 0)
            
            analysis = "Investment Analysis:\n\n"
            analysis += f"• {answer} (confidence: {score:.1%})\n\n"
            
            # Add additional context
            analysis += "Note: This answer is based on the provided financial data. Consider additional market\n"
            analysis += "factors and consult multiple sources before making investment decisions."
            
            return analysis
        
        return str(result)
    
    # Handle zero-shot classification models
    elif model_key in zeroshot_models:
        if isinstance(result, dict):
            if "scores" in result and "labels" in result:
                scores = result["scores"]
                labels = result["labels"]
                
                analysis = "Investment Recommendations:\n\n"
                
                # Combine labels and scores, sort by score
                recommendations = sorted(zip(labels, scores), key=lambda x: x[1], reverse=True)
                
                # Format top 3 recommendations
                for label, score in recommendations[:3]:
                    analysis += f"• {label}: {score*100:.1f}% confidence\n"
                
                # Add reasoning based on top recommendation
                top_recommendation = recommendations[0][0] if recommendations else None
                
                analysis += "\nReasoning:\n"
                
                if "Buy" in top_recommendation:
                    analysis += "The model suggests a BUY recommendation based on the financial data provided.\n"
                    analysis += "This indicates positive sentiment toward the stock's future performance."
                elif "Hold" in top_recommendation:
                    analysis += "The model suggests a HOLD recommendation based on the financial data provided.\n"
                    analysis += "This indicates stable or uncertain outlook for the stock in the near term."
                elif "Sell" in top_recommendation:
                    analysis += "The model suggests a SELL recommendation based on the financial data provided.\n"
                    analysis += "This indicates concerns about the stock's future performance."
                
                return analysis
            
        return str(result)
    
    # Handle embedding model responses
    elif model_key in embedding_models:
        if isinstance(result, list) and isinstance(result[0], float):
            # We received embedding vectors, so generate a meaningful response
            analysis = "Semantic Analysis of Financial Data:\n\n"
            analysis += "The model has analyzed the financial data and created embeddings.\n"
            analysis += "These embeddings can be used for similarity comparisons with other financial documents.\n\n"
            analysis += "For direct investment advice, please use a classification or text generation model."
            
            return analysis
        elif isinstance(result, dict) and "similarities" in result:
            # We received similarity scores to questions
            similarities = result["similarities"]
            questions = [
                "Is this stock a good investment?",
                "What are the key financial metrics for this stock?",
                "Should I buy, hold, or sell this stock?",
                "What are the risks associated with this stock?"
            ]
            
            analysis = "Financial Data Analysis:\n\n"
            
            # Combine questions and similarity scores
            for i, (question, score) in enumerate(zip(questions, similarities)):
                analysis += f"{i+1}. {question}\n"
                analysis += f"   Confidence: {score*100:.1f}%\n\n"
            
            # Make a recommendation based on the highest similarity
            max_index = similarities.index(max(similarities))
            max_question = questions[max_index]
            
            analysis += "Based on semantic analysis, the most relevant question is:\n"
            analysis += f"'{max_question}'\n\n"
            analysis += "For direct investment advice, please use a classification or text generation model."
            
            return analysis
        
        return str(result)
    
    # Handle long document models
    elif model_key in long_document_models:
        return str(result)
    
    # Handle special models
    elif model_key in special_models:
        if model_key == "whisper":
            return "Whisper is designed for audio transcription, not text analysis. Please select a different model for stock analysis."
    
    # Default response handling for other models
    else:
        if isinstance(result, list):
            return str(result[0] if result else "No result")
        return str(result) 