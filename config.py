"""
Configuration file for Enhanced AI Service
"""

import os
from typing import Optional

class Config:
    """Configuration class for AI service settings"""
    
    # Model paths and configurations
    LLAMA_MODEL_PATH = os.getenv("LLAMA_MODEL_PATH", "meta-llama/Llama-2-7b-chat-hf")
    SENTENCE_TRANSFORMER_MODEL = os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
    
    # HuggingFace settings
    HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
    HUGGINGFACE_CACHE_DIR = os.getenv("HUGGINGFACE_CACHE_DIR", "./models")
    
    # Device configuration
    FORCE_CPU = os.getenv("FORCE_CPU", "false").lower() == "true"
    DEVICE = "cpu" if FORCE_CPU else ("cuda" if os.getenv("CUDA_AVAILABLE", "true").lower() == "true" else "cpu")
    
    # Model quantization settings
    USE_4BIT_QUANTIZATION = os.getenv("USE_4BIT_QUANTIZATION", "true").lower() == "true"
    USE_8BIT_QUANTIZATION = os.getenv("USE_8BIT_QUANTIZATION", "false").lower() == "true"
    
    # FAISS index settings
    DEFAULT_INDEX_TYPE = os.getenv("DEFAULT_INDEX_TYPE", "FlatIP")  # FlatIP for cosine similarity
    INDEX_SAVE_DIR = os.getenv("INDEX_SAVE_DIR", "./indices")
    
    # Text processing settings
    MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "4096"))
    DEFAULT_SUMMARY_LENGTH = int(os.getenv("DEFAULT_SUMMARY_LENGTH", "150"))
    MIN_TEXT_LENGTH = int(os.getenv("MIN_TEXT_LENGTH", "50"))
    
    # Medical entity extraction settings
    MEDICAL_KEYWORDS = [
        'diagnosis', 'symptoms', 'treatment', 'medication', 'procedure',
        'results', 'findings', 'recommendation', 'assessment', 'plan',
        'vital signs', 'lab values', 'blood pressure', 'heart rate',
        'temperature', 'oxygen saturation', 'respiratory rate'
    ]
    
    # LLaMA-2 generation settings
    LLAMA_GENERATION_CONFIG = {
        "max_new_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
        "pad_token_id": None,  # Will be set automatically
        "eos_token_id": None,  # Will be set automatically
    }
    
    # Sentence-BERT settings
    SENTENCE_BERT_BATCH_SIZE = int(os.getenv("SENTENCE_BERT_BATCH_SIZE", "32"))
    SENTENCE_BERT_DEVICE = DEVICE
    
    # Logging settings
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Performance settings
    ENABLE_CACHING = os.getenv("ENABLE_CACHING", "true").lower() == "true"
    CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour
    
    # Security settings
    ENABLE_RATE_LIMITING = os.getenv("ENABLE_RATE_LIMITING", "true").lower() == "true"
    MAX_REQUESTS_PER_MINUTE = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "60"))
    
    @classmethod
    def get_llama_config(cls) -> dict:
        """Get LLaMA-2 configuration"""
        if cls.USE_4BIT_QUANTIZATION:
            return {
                "load_in_4bit": True,
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": "bfloat16"
            }
        elif cls.USE_8BIT_QUANTIZATION:
            return {
                "load_in_8bit": True
            }
        else:
            return {}
    
    @classmethod
    def get_model_paths(cls) -> dict:
        """Get model paths configuration"""
        return {
            "llama": cls.LLAMA_MODEL_PATH,
            "sentence_transformer": cls.SENTENCE_TRANSFORMER_MODEL,
            "cache_dir": cls.HUGGINGFACE_CACHE_DIR
        }
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate configuration settings"""
        try:
            # Check required environment variables
            if not cls.HUGGINGFACE_TOKEN:
                print("Warning: HUGGINGFACE_TOKEN not set. LLaMA-2 model may not load.")
            
            # Validate numeric settings
            if cls.DEFAULT_SUMMARY_LENGTH <= 0:
                print("Error: DEFAULT_SUMMARY_LENGTH must be positive")
                return False
            
            if cls.MAX_TEXT_LENGTH <= cls.MIN_TEXT_LENGTH:
                print("Error: MAX_TEXT_LENGTH must be greater than MIN_TEXT_LENGTH")
                return False
            
            # Create directories if they don't exist
            os.makedirs(cls.INDEX_SAVE_DIR, exist_ok=True)
            os.makedirs(cls.HUGGINGFACE_CACHE_DIR, exist_ok=True)
            
            return True
            
        except Exception as e:
            print(f"Configuration validation error: {e}")
            return False

# Environment-specific configurations
class DevelopmentConfig(Config):
    """Development environment configuration"""
    LOG_LEVEL = "DEBUG"
    ENABLE_CACHING = False
    ENABLE_RATE_LIMITING = False

class ProductionConfig(Config):
    """Production environment configuration"""
    LOG_LEVEL = "WARNING"
    ENABLE_CACHING = True
    ENABLE_RATE_LIMITING = True
    FORCE_CPU = False

class TestConfig(Config):
    """Testing environment configuration"""
    LOG_LEVEL = "DEBUG"
    ENABLE_CACHING = False
    ENABLE_RATE_LIMITING = False
    FORCE_CPU = True

# Configuration factory
def get_config(environment: str = None) -> Config:
    """Get configuration based on environment"""
    if not environment:
        environment = os.getenv("ENVIRONMENT", "development").lower()
    
    configs = {
        "development": DevelopmentConfig,
        "production": ProductionConfig,
        "test": TestConfig
    }
    
    config_class = configs.get(environment, DevelopmentConfig)
    config = config_class()
    
    # Validate configuration
    if not config.validate_config():
        print("Warning: Configuration validation failed, using defaults")
    
    return config

# Default configuration instance
config = get_config() 