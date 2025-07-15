import os
from pathlib import Path

class Config:
    """Configuration settings for the sentiment analysis project"""
    
    # Model settings
    MODEL_NAME = "distilbert-base-uncased"
    NUM_LABELS = 3
    MAX_LENGTH = 512
    
    # Training settings
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 3
    WARMUP_STEPS = 500
    WEIGHT_DECAY = 0.01
    
    # Data settings
    TEST_SIZE = 0.2
    VAL_SIZE = 0.1
    RANDOM_STATE = 42
    
    # Paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    MODEL_DIR = BASE_DIR / "models"
    LOGS_DIR = BASE_DIR / "logs"
    
    # Create directories if they don't exist
    DATA_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)
    
    # API settings
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    
    # Sentiment mapping
    SENTIMENT_MAP = {
        0: "Negative",
        1: "Neutral", 
        2: "Positive"
    }
    
    # Model performance thresholds
    MIN_CONFIDENCE = 0.6
    
    @classmethod
    def get_model_path(cls, model_name="sentiment_model"):
        """Get the full path to a saved model"""
        return cls.MODEL_DIR / model_name
    
    @classmethod
    def get_data_path(cls, filename):
        """Get the full path to a data file"""
        return cls.DATA_DIR / filename