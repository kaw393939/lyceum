"""
Gutenberg Content Generation System - Configuration Module
=========================================================
Handles application configuration, environment settings, and logging setup.
"""

import os
import yaml
import logging
import traceback
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from functools import lru_cache

# Pydantic V2 compatible imports
from pydantic import BaseModel, Field, field_validator, SecretStr
# Import BaseSettings from pydantic_settings
from pydantic_settings import BaseSettings

# For secure environment variable handling
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("python-dotenv not installed. Run 'pip install python-dotenv'")

VERSION = "1.0.0"
DEFAULT_CONFIG_PATH = "config.yaml"

class EnvSettings(BaseSettings):
    """Environment-based settings with secure secrets handling."""
    openai_api_key: Optional[SecretStr] = None
    azure_openai_key: Optional[SecretStr] = None
    azure_openai_endpoint: Optional[str] = None
    mongo_password: Optional[SecretStr] = None
    jwt_secret: Optional[SecretStr] = None
    environment: str = "development"
    ptolemy_api_url: Optional[str] = None
    ptolemy_bearer_token: Optional[SecretStr] = None
    socrates_api_url: Optional[str] = None
    socrates_bearer_token: Optional[SecretStr] = None
    model_provider: Optional[str] = "openai"
    
    class Config:
        env_file = ".env"

# Load environment settings
env_settings = EnvSettings()

class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"

# Base class for all configuration models to support dict-like access
class DictLikeModel(BaseModel):
    """Base class for configuration models that supports dictionary-like access."""
    
    def get(self, key, default=None):
        """Get a configuration value with a fallback default."""
        return getattr(self, key, default)
    
    def __getitem__(self, key):
        """Support dictionary-like access."""
        return getattr(self, key)
    
    def __contains__(self, key):
        """Support 'in' operator."""
        return hasattr(self, key)

class MongoConfig(DictLikeModel):
    uri: str = "mongodb://gutenberg_user:gutenberg_password@mongodb:27017/gutenberg"
    database: str = "gutenberg"
    templates_collection: str = "templates"
    content_collection: str = "content"
    media_collection: str = "media"
    feedback_collection: str = "feedback"
    sessions_collection: str = "session_data"
    cached_collection: str = "cached_data"
    ttl_days: int = 30  # Default time to live for content in days

class RedisConfig(DictLikeModel):
    uri: str = "redis://redis:6379/0"
    password: Optional[str] = None
    prefix: str = "gutenberg:"
    ttl: int = 3600  # Default cache TTL in seconds

class LLMConfig(DictLikeModel):
    provider: str = "openai"  # openai, azure, anthropic, etc.
    default_model: str = "gpt-4o-mini"
    fallback_model: str = "gpt-3.5-turbo"
    model: str = "gpt-4o-mini"  # Active model to use
    temperature_content: float = 0.7
    temperature_rich_content: float = 0.8
    temperature_summaries: float = 0.5
    max_tokens_content: int = 4000
    max_tokens_rich_content: int = 8000
    max_tokens_summaries: int = 1000
    timeout: int = 60
    retry_count: int = 3
    retry_delay: int = 2
    use_azure: bool = False
    azure_deployment_id: Optional[str] = None
    azure_api_version: str = "2023-05-15"
    stream_responses: bool = False
    rate_limit_minute: int = 100  # Requests per minute limit
    max_concurrent_requests: int = 10
    request_timeout: int = 60
    context_window_size: int = 16000
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

class RAGConfig(DictLikeModel):
    enabled: bool = True
    vector_db_url: Optional[str] = None  # If using separate vector DB from Ptolemy
    vector_db_collection: str = "content_vectors"
    shared_ptolemy_vectors: bool = True  # Whether to use Ptolemy's vectors
    ptolemy_collection: str = "concepts"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_chunks_per_doc: int = 10
    similarity_top_k: int = 5
    similarity_threshold: float = 0.7
    use_reranking: bool = False
    reranker_model: Optional[str] = None
    reranker_top_k: int = 3

class TemplateConfig(DictLikeModel):
    templates_dir: str = "./templates"
    default_template: str = "default.json"
    template_version_control: bool = True
    max_template_versions: int = 5
    custom_template_prefix: str = "custom_"
    enable_dynamic_templates: bool = True
    template_validation: bool = True
    max_template_size_kb: int = 500
    allow_external_templates: bool = False

class MediaConfig(DictLikeModel):
    audio_enabled: bool = True
    video_enabled: bool = True
    image_enabled: bool = True
    diagram_enabled: bool = True
    chart_enabled: bool = True
    infographic_enabled: bool = True
    audio_format: str = "mp3"
    video_format: str = "mp4"
    image_format: str = "png"
    max_audio_length_seconds: int = 600
    max_video_length_seconds: int = 300
    max_image_resolution: str = "1024x1024"
    audio_bitrate: str = "128k"
    video_bitrate: str = "1000k"
    image_quality: int = 90
    storage_path: str = "./media"
    use_gridfs: bool = True
    image_api_enabled: bool = True
    image_api_url: str = "https://api.openai.com/v1/images/generations"
    audio_api_enabled: bool = True
    use_openai_tts: bool = True
    tts_voices: List[str] = ["nova", "echo", "alloy", "fable", "onyx", "shimmer"]
    default_voice: str = "nova"
    enable_mermaid_diagrams: bool = True
    enable_chartjs: bool = True

class PtolemyConfig(DictLikeModel):
    api_url: str = "http://ptolemy:8000"
    api_version: str = "v1"
    api_prefix: str = "/api/v1"
    timeout: int = 30
    retry_count: int = 3
    retry_delay: int = 2
    relationship_cache_ttl: int = 3600
    concept_cache_ttl: int = 3600
    graph_cache_ttl: int = 7200
    use_shared_vector_db: bool = True
    max_concept_depth: int = 3
    batch_size: int = 50

class SocratesConfig(DictLikeModel):
    api_url: str = "http://socrates:8000"
    api_version: str = "v1"
    api_prefix: str = "/api/v1"
    timeout: int = 30
    retry_count: int = 3
    retry_delay: int = 2
    learner_profile_cache_ttl: int = 1800
    activity_stream_enabled: bool = True
    feedback_forwarding: bool = True

class ApiConfig(DictLikeModel):
    host: str = "0.0.0.0"
    port: int = 8001  # Different from Ptolemy's 8000
    cors_origins: List[str] = ["*"]
    api_prefix: str = "/api/v1"
    docs_url: str = "/docs"
    redoc_url: str = "/redoc"
    debug: bool = False
    workers: int = 4
    request_timeout: int = 60
    enable_profiling: bool = False
    enable_metrics: bool = True
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_period: int = 60  # seconds
    jwt_algorithm: str = "HS256"
    jwt_expiration: int = 86400  # 24 hours
    require_auth: bool = False
    content_generation_timeout: int = 120

class LoggingConfig(DictLikeModel):
    level: LogLevel = LogLevel.INFO
    file: Optional[str] = "gutenberg.log"
    console: bool = True
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_api_requests: bool = True
    log_errors_only: bool = False
    enable_request_id: bool = True
    enable_structured_logging: bool = False
    max_file_size: int = 10485760  # 10MB
    backup_count: int = 5
    
    # Log analyzer settings
    enable_log_analyzer: bool = True
    log_analysis_interval: int = 3600  # 1 hour
    log_analysis_save_reports: bool = True
    log_analysis_reports_dir: str = "reports"
    auto_implement_recommendations: bool = False  # Whether to automatically apply simple recommendations
    log_analysis_retention_days: int = 30  # How long to keep reports

class ContentConfig(DictLikeModel):
    max_content_length: int = 50000
    min_content_length: int = 500
    default_age_range: str = "14-18"
    default_difficulty: str = "intermediate"
    enable_content_validation: bool = True
    content_validation_threshold: float = 0.7
    dynamic_content_adjustment: bool = True
    enable_content_versioning: bool = True
    max_content_versions: int = 3
    content_retention_days: int = 90
    default_content_format: str = "markdown"
    enable_html_content: bool = True
    enable_audio_generation: bool = True
    enable_image_generation: bool = True
    enable_interactive_elements: bool = True
    default_interactivity_level: str = "medium"
    feedback_threshold_for_regeneration: float = 0.3  # Regenerate if 30% negative feedback

class FeedbackConfig(DictLikeModel):
    enabled: bool = True
    anonymous_feedback: bool = True
    store_feedback_metadata: bool = True
    enable_detailed_ratings: bool = True
    feedback_categories: List[str] = ["clarity", "accuracy", "engagement", "relevance", "depth"]
    feedback_aggregation_interval: int = 86400  # Daily
    minimum_feedback_for_analysis: int = 5
    forward_to_socrates: bool = True
    forward_to_galileo: bool = True

class CacheConfig(DictLikeModel):
    """Configuration for the various caches used by the system."""
    content_cache_size: int = 1000
    content_cache_ttl: int = 3600  # 1 hour
    template_cache_size: int = 500
    template_cache_ttl: int = 7200  # 2 hours
    concept_cache_size: int = 1000
    concept_cache_ttl: int = 3600  # 1 hour
    media_cache_size: int = 200
    media_cache_ttl: int = 14400  # 4 hours
    use_redis: bool = False

class GeneralConfig(DictLikeModel):
    """General system-wide configuration."""
    use_threading: bool = True
    max_workers: int = 10
    debug_mode: bool = False
    enable_telemetry: bool = True
    default_language: str = "en"
    storage_path: str = "./storage"
    tmp_path: str = "./tmp"
    cleanup_tmp_interval: int = 3600  # 1 hour
    max_tmp_age: int = 86400  # 1 day

class Config(DictLikeModel):
    mongo: MongoConfig = MongoConfig()
    redis: RedisConfig = RedisConfig()
    llm: LLMConfig = LLMConfig()
    rag: RAGConfig = RAGConfig()
    templates: TemplateConfig = TemplateConfig()
    media: MediaConfig = MediaConfig()
    ptolemy: PtolemyConfig = PtolemyConfig()
    socrates: SocratesConfig = SocratesConfig()
    api: ApiConfig = ApiConfig()
    logging: LoggingConfig = LoggingConfig()
    content: ContentConfig = ContentConfig()
    feedback: FeedbackConfig = FeedbackConfig()
    cache: CacheConfig = CacheConfig()
    general: GeneralConfig = GeneralConfig()
    data_dir: str = "./data"
    environment: str = "development"
    version: str = VERSION

def create_default_config() -> Config:
    """Create default configuration."""
    return Config()

def load_config(config_path: str = DEFAULT_CONFIG_PATH, env: Optional[str] = None) -> Config:
    """
    Load configuration from file with environment-specific overrides.
    Falls back to default configuration if file not found.
    """
    if env is None:
        env = env_settings.environment

    # Look for environment-specific config first
    env_config_path = f"{os.path.splitext(config_path)[0]}.{env}.yaml"
    
    try:
        # Try environment-specific config first
        if os.path.exists(env_config_path):
            with open(env_config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
                logging.info(f"Loaded environment-specific config from {env_config_path}")
                config = Config.model_validate(config_dict)
        # Fall back to main config
        elif os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
                logging.info(f"Loaded config from {config_path}")
                config = Config.model_validate(config_dict)
        else:
            # Create default config
            config = create_default_config()
            os.makedirs(os.path.dirname(config_path) or '.', exist_ok=True)
            with open(config_path, 'w') as f:
                yaml.dump(config.model_dump(), f, default_flow_style=False)
            logging.info(f"Created default config at {config_path}")
    except Exception as e:
        logging.error(f"Error loading config: {e}")
        logging.error(traceback.format_exc())
        config = create_default_config()
    
    # Override with environment variables
    if env_settings.mongo_password:
        config.mongo.uri = config.mongo.uri.replace("mongodb://", f"mongodb://admin:{env_settings.mongo_password.get_secret_value()}@")
    if env_settings.ptolemy_api_url:
        config.ptolemy.api_url = env_settings.ptolemy_api_url
    if env_settings.ptolemy_bearer_token:
        config.ptolemy.bearer_token = env_settings.ptolemy_bearer_token.get_secret_value()
    if env_settings.socrates_api_url:
        config.socrates.api_url = env_settings.socrates_api_url
    if env_settings.socrates_bearer_token:
        config.socrates.bearer_token = env_settings.socrates_bearer_token.get_secret_value()
    if env_settings.azure_openai_endpoint:
        config.llm.use_azure = True
        config.llm.azure_deployment_id = config.llm.default_model
    
    # Update environment setting
    config.environment = env
    
    return config

def setup_logging(config: LoggingConfig):
    """Set up the application's logging configuration."""
    log_level = getattr(logging, config.level.value)
    handlers = []
    
    if config.file:
        try:
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                config.file,
                maxBytes=config.max_file_size,
                backupCount=config.backup_count
            )
            handlers.append(file_handler)
        except Exception as e:
            print(f"Failed to set up file logging: {e}")
    
    if config.console:
        handlers.append(logging.StreamHandler())
    
    if config.enable_structured_logging:
        try:
            import structlog
            structlog.configure(
                processors=[
                    structlog.stdlib.filter_by_level,
                    structlog.stdlib.add_logger_name,
                    structlog.stdlib.add_log_level,
                    structlog.stdlib.PositionalArgumentsFormatter(),
                    structlog.processors.TimeStamper(fmt="iso"),
                    structlog.processors.StackInfoRenderer(),
                    structlog.processors.format_exc_info,
                    structlog.processors.UnicodeDecoder(),
                    structlog.processors.JSONRenderer()
                ],
                context_class=dict,
                logger_factory=structlog.stdlib.LoggerFactory(),
                wrapper_class=structlog.stdlib.BoundLogger,
                cache_logger_on_first_use=True,
            )
        except ImportError:
            print("structlog not installed. Using standard logging. Run 'pip install structlog' for structured logging.")
    
    logging.basicConfig(level=log_level, format=config.format, handlers=handlers)
    
    # Silence noisy libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('openai').setLevel(logging.WARNING)

# Global config instance (lazy loaded)
_config_instance = None

@lru_cache
def get_config() -> Config:
    """Get the global configuration instance, loading it if necessary."""
    global _config_instance
    if _config_instance is None:
        _config_instance = load_config()
        setup_logging(_config_instance.logging)
    return _config_instance