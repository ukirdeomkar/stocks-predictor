"""
Module to manage model information and selection
"""

# Default model to use
DEFAULT_MODEL = "finbert"

def get_available_models():
    """
    Get a dictionary of available AI models with their descriptions and API endpoints.
    
    Returns:
        dict: Dictionary of model information with keys being model names
    """
    return {
        "finbert": {
            "description": "Specialized model for financial sentiment analysis",
            "api_endpoint": "ProsusAI/finbert",
            "size": "110M"
        },
        "finbert-tone": {
            "description": "Financial tone detection for market sentiment",
            "api_endpoint": "yiyanghkust/finbert-tone",
            "size": "110M"
        },
        "finbert-pretrain": {
            "description": "Pretrained on financial texts for classification",
            "api_endpoint": "yiyanghkust/finbert-pretrain",
            "size": "110M"
        },
        "roberta-squad": {
            "description": "Question answering model for financial inquiries",
            "api_endpoint": "deepset/roberta-base-squad2",
            "size": "125M"
        },
        "bert-sentiment": {
            "description": "Multilingual sentiment analysis for financial texts",
            "api_endpoint": "nlptown/bert-base-multilingual-uncased-sentiment",
            "size": "110M"
        },
        "codeberta": {
            "description": "Code and text embedding for financial documents",
            "api_endpoint": "huggingface/CodeBERTa-small-v1",
            "size": "84M"
        },
        "sentence-transformer": {
            "description": "Sentence embeddings for financial document similarity",
            "api_endpoint": "sentence-transformers/all-MiniLM-L6-v2",
            "size": "80M"
        },
        "bart-mnli": {
            "description": "Natural Language Inference for financial reasoning",
            "api_endpoint": "facebook/bart-large-mnli",
            "size": "1.6G"
        },
        "bart-cnn": {
            "description": "Summarization for financial reports and news",
            "api_endpoint": "facebook/bart-large-cnn",
            "size": "1.6G"
        },
        "deberta-zeroshot": {
            "description": "Zero-shot classification for financial analysis",
            "api_endpoint": "MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
            "size": "1.3G"
        },
        "mpnet-sentence": {
            "description": "State-of-the-art semantic search for financial documents",
            "api_endpoint": "sentence-transformers/all-mpnet-base-v2",
            "size": "420M"
        },
        "flan-t5": {
            "description": "Versatile model for financial summaries and Q&A",
            "api_endpoint": "google/flan-t5-large",
            "size": "780M"
        },
        "longformer": {
            "description": "Long document analysis for financial reports",
            "api_endpoint": "allenai/longformer-base-4096",
            "size": "430M"
        },
        "whisper": {
            "description": "Speech-to-text for earnings calls and presentations",
            "api_endpoint": "openai/whisper-large",
            "size": "1.5G"
        },
        "bert-base": {
            "description": "Classic BERT model for feature extraction",
            "api_endpoint": "bert-base-uncased",
            "size": "110M"
        }
    } 