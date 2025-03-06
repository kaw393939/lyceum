"""
Ptolemy Knowledge Map System - Configuration Module
==================================================
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

VERSION = "2.0.0"
DEFAULT_CONFIG_PATH = "config.yaml"

class EnvSettings(BaseSettings):
    """Environment-based settings with secure secrets handling."""
    openai_api_key: Optional[SecretStr] = None
    azure_openai_key: Optional[SecretStr] = None
    azure_openai_endpoint: Optional[str] = None
    neo4j_password: Optional[SecretStr] = None
    mongo_password: Optional[SecretStr] = None
    jwt_secret: Optional[SecretStr] = None
    environment: str = "development"
    ptolemy_api_url: Optional[str] = None
    ptolemy_bearer_token: Optional[SecretStr] = None
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

class Neo4jConfig(DictLikeModel):
    """Configuration for Neo4j graph database connection.
    
    Neo4j stores the knowledge graph structure including concepts and their relationships.
    This configuration manages connection parameters, authentication, and performance settings.
    """
    uri: str = "bolt://neo4j:7687"  # Neo4j server URI
    user: str = "neo4j"  # Database username
    password: str = "password"  # Database password
    database: Optional[str] = None  # Specific database name (None uses default)
    
    # Connection pool settings
    connection_timeout: int = 30  # Connection timeout in seconds
    max_connection_lifetime: int = 3600  # Maximum connection lifetime in seconds
    max_connection_pool_size: int = 50  # Maximum size of the connection pool
    connection_acquisition_timeout: int = 60  # Timeout for acquiring a connection from the pool
    
    @field_validator('uri')
    def validate_uri(cls, v):
        """Validate that the Neo4j URI has a proper protocol."""
        if not v.startswith(('bolt://', 'bolt+s://', 'bolt+ssc://', 'neo4j://', 'neo4j+s://')):
            raise ValueError(f"Invalid Neo4j URI protocol: {v}")
        return v

class QdrantConfig(DictLikeModel):
    """Configuration for Qdrant vector database connection.
    
    Qdrant stores vector embeddings for semantic search and retrieval.
    This class manages connection parameters, collection settings, and shared resource configuration.
    """
    # Connection settings
    url: str = "http://qdrant:6333"  # Qdrant server URL
    api_key: Optional[str] = None  # API key for authentication (if required)
    timeout: int = 30  # Connection timeout in seconds
    
    # Collection configuration
    collection_name: str = "goliath_vectors"  # Name of the vector collection
    vector_size: int = 1536  # Dimensionality of vectors (1536 for OpenAI embeddings)
    distance: str = "Cosine"  # Distance metric for vector similarity (Cosine, Euclid, Dot)
    
    # Resource identifier
    service_prefix: str = "ptolemy_"  # Prefix for vector IDs from this service
    
    # Scaling parameters
    shard_number: int = 1  # Number of shards for the collection
    replication_factor: int = 1  # Replication factor for high availability
    write_consistency_factor: int = 1  # Write consistency factor
    on_disk_payload: bool = True  # Whether to store payload on disk (vs in-memory)
    
    # Performance settings
    max_workers: int = 10  # Maximum number of worker threads for batch operations
    batch_size: int = 100  # Number of vectors per batch operation
    
    # Shared database settings between Ptolemy and Gutenberg
    shared_collections_enabled: bool = True  # Whether to use a shared collection
    namespacing_field: str = "service"  # Field name used for namespacing
    namespacing_value: str = "ptolemy"  # Value identifying vectors from this service
    indexing_threshold: int = 20000  # Threshold for reindexing
    
    @field_validator('distance')
    def validate_distance(cls, v):
        """Validate that the distance metric is supported by Qdrant."""
        valid_metrics = ["Cosine", "Euclid", "Dot"]
        if v not in valid_metrics:
            raise ValueError(f"Invalid distance metric: {v}. Must be one of {valid_metrics}")
        return v

class MongoConfig(DictLikeModel):
    """Configuration for MongoDB document database connection.
    
    MongoDB stores structured data like concept details, user information, and analytics.
    This configuration manages connection parameters and collection names.
    """
    # Connection settings
    uri: str = "mongodb://mongodb:27017"  # MongoDB connection URI
    database: str = "ptolemy"  # Database name
    
    # Core data collections
    concepts_collection: str = "concepts"  # Stores concept definitions
    domains_collection: str = "domains"  # Stores domain information
    relationships_collection: str = "relationships"  # Stores relationships between concepts
    learning_paths_collection: str = "learning_paths"  # Stores learning path sequences
    
    # User and activity data
    users_collection: str = "users"  # Stores user information
    activity_collection: str = "activity"  # Stores user activity logs
    
    # Support collections
    analytics_collection: str = "analytics"  # Stores usage analytics
    cache_collection: str = "cache"  # Stores cached data
    
    @field_validator('uri')
    def validate_mongo_uri(cls, v):
        """Validate that the MongoDB URI has a proper format."""
        if not v.startswith('mongodb://') and not v.startswith('mongodb+srv://'):
            raise ValueError(f"Invalid MongoDB URI format: {v}")
        return v

class RedisConfig(DictLikeModel):
    uri: str = "redis://redis:6379/0"
    password: Optional[str] = None
    prefix: str = "ptolemy:"
    ttl: int = 3600  # Default cache TTL in seconds

class EmbeddingsConfig(DictLikeModel):
    model_name: str = "all-MiniLM-L6-v2"
    cache_dir: Optional[str] = "./data/model_cache"
    batch_size: int = 32
    use_gpu: bool = False
    quantize: bool = False
    max_seq_length: int = 256
    default_pooling: str = "mean"

class LLMConfig(DictLikeModel):
    """Configuration for Language Model interactions.
    
    This class manages settings for different LLM providers, models, and their operational parameters.
    It controls temperature settings for different tasks, token limits, timeouts, and rate limiting.
    """
    provider: str = "openai"  # Options: "openai", "azure", "anthropic", etc.
    default_model: str = "gpt-4o-mini"  # Default model to use
    fallback_model: str = "gpt-3.5-turbo"  # Fallback model if default is unavailable
    model: str = "gpt-4o-mini"  # Currently active model
    
    # Temperature controls for different tasks (higher = more creative)
    temperature_analysis: float = 0.2  # Lower temperature for analytical tasks
    temperature_generation: float = 0.7  # Higher temperature for creative generation
    temperature_enrichment: float = 0.5  # Balanced temperature for enrichment
    
    # Token limits for different task types
    max_tokens_analysis: int = 2000
    max_tokens_generation: int = 4000
    max_tokens_enrichment: int = 1000
    
    # Request handling parameters
    timeout: int = 60  # Seconds to wait for response
    retry_count: int = 3  # Number of retries on failure
    retry_delay: int = 2  # Initial delay between retries (seconds)
    
    # Azure OpenAI specific configuration
    use_azure: bool = False  # Whether to use Azure OpenAI
    azure_deployment_id: Optional[str] = None  # Azure deployment ID
    azure_api_version: str = "2023-05-15"  # Azure API version
    
    # Advanced options
    stream_responses: bool = False  # Whether to stream responses
    rate_limit_minute: int = 100  # Requests per minute limit
    max_concurrent_requests: int = 10  # Maximum concurrent requests
    request_timeout: int = 60  # Request timeout in seconds
    
    # Model selection policies
    allow_gpt4: bool = True  # Allow usage of GPT-4 models
    allow_gpt4_for_domains: bool = True  # Allow GPT-4 for domain analysis
    allow_gpt4_for_learning_paths: bool = True  # Allow GPT-4 for learning path generation
    
    @field_validator('model')
    def validate_model(cls, value, info):
        """Validate that the selected model is supported by the provider."""
        provider = info.data.get('provider', 'openai')
        
        # Basic validation of model names by provider
        if provider == 'openai' and not (value.startswith('gpt-') or value.startswith('text-')):
            raise ValueError(f"Invalid OpenAI model name: {value}")
        elif provider == 'anthropic' and not value.startswith('claude-'):
            raise ValueError(f"Invalid Anthropic model name: {value}")
            
        return value
    
class ApiConfig(DictLikeModel):
    host: str = "0.0.0.0"
    port: int = 8000
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

class LoggingConfig(DictLikeModel):
    level: LogLevel = LogLevel.INFO
    file: Optional[str] = "ptolemy.log"
    console: bool = True
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_api_requests: bool = True
    log_errors_only: bool = False
    enable_request_id: bool = True
    enable_structured_logging: bool = False
    max_file_size: int = 10485760  # 10MB
    backup_count: int = 5

class PromptsConfig(DictLikeModel):
    domain_analysis_prompt: Optional[str] = None
    concept_generation_prompt: Optional[str] = None
    concept_enrichment_prompt: Optional[str] = None
    relationship_generation_prompt: Optional[str] = None
    learning_path_prompt: Optional[str] = None
    consistency_check_prompt: Optional[str] = None
    topic_extraction_prompt: Optional[str] = None
    knowledge_gap_prompt: Optional[str] = None
    default_concept_count: int = 15
    max_concepts_gpt35: int = 25
    max_concepts_general: int = 50
    generation_temperature: float = 0.7
    analysis_temperature: float = 0.2
    enrichment_temperature: float = 0.5

class ValidationConfig(DictLikeModel):
    enable_automated_checks: bool = True
    consistency_check_interval: int = 86400  # daily
    concept_relation_min_threshold: int = 1  # Each concept should have at least this many relationships
    max_orphaned_concepts: int = 10  # Maximum number of concepts without relationships allowed
    validate_on_write: bool = True
    detect_cycles: bool = True
    detect_contradictions: bool = True

class IntegrationConfig(DictLikeModel):
    gutenberg_api_url: Optional[str] = None
    gutenberg_api_key: Optional[str] = None
    socrates_api_url: Optional[str] = None
    socrates_api_key: Optional[str] = None
    enable_webhooks: bool = False
    webhook_urls: Dict[str, str] = {}
    enable_event_streaming: bool = False
    event_streaming_url: Optional[str] = None

class MetricsConfig(DictLikeModel):
    enabled: bool = True
    prometheus_port: int = 9090
    export_interval: int = 15  # seconds
    include_neo4j_metrics: bool = True
    include_mongo_metrics: bool = True
    include_qdrant_metrics: bool = True
    include_llm_metrics: bool = True
    include_api_metrics: bool = True
    labels: Dict[str, str] = {"service": "ptolemy"}

class CacheConfig(DictLikeModel):
    """Configuration for the various caches used by the system."""
    concept_cache_size: int = 1000
    concept_cache_ttl: int = 3600  # 1 hour
    relationship_cache_size: int = 1000
    relationship_cache_ttl: int = 3600  # 1 hour
    embedding_cache_size: int = 500
    embedding_cache_ttl: int = 7200  # 2 hours
    use_redis: bool = False

class GeneralConfig(DictLikeModel):
    """General system-wide configuration."""
    use_threading: bool = True
    max_workers: int = 10
    debug_mode: bool = False
    enable_telemetry: bool = True

class Config(DictLikeModel):
    neo4j: Neo4jConfig = Neo4jConfig()
    qdrant: QdrantConfig = QdrantConfig()
    mongo: MongoConfig = MongoConfig()
    redis: RedisConfig = RedisConfig()
    embeddings: EmbeddingsConfig = EmbeddingsConfig()
    llm: LLMConfig = LLMConfig()
    api: ApiConfig = ApiConfig()
    logging: LoggingConfig = LoggingConfig()
    prompts: PromptsConfig = PromptsConfig()
    validation: ValidationConfig = ValidationConfig() 
    integration: IntegrationConfig = IntegrationConfig()
    metrics: MetricsConfig = MetricsConfig()
    cache: CacheConfig = CacheConfig()
    general: GeneralConfig = GeneralConfig()
    data_dir: str = "./data"
    environment: str = "development"
    version: str = VERSION

def create_default_config() -> Config:
    """Create default configuration with essential prompts."""
    default_prompts = {
        "domain_analysis_prompt": (
            "Analyze the educational domain \"{domain}\" with this description:\n"
            "\"{description}\"\n\n"
            "{topics_text}\n\n"
            "Determine the appropriate number of distinct concepts that should be taught to provide:\n"
            "1. Comprehensive coverage of the domain\n"
            "2. Appropriate granularity\n"
            "3. A logical learning progression\n\n"
            "Respond with a JSON object containing:\n"
            "\"recommended_concept_count\": <integer>, \"justification\": \"<brief explanation>\", \"suggested_model\": \"<gpt-3.5-turbo or gpt-4-turbo>\""
        ),
        "concept_generation_prompt": (
            "Create a comprehensive knowledge graph with {num_concepts} concepts for teaching {domain}.\n\n"
            "Domain description: {description}\n\n"
            "{topics_text}\n\n"
            "Important guidelines:\n"
            "1. Ensure concepts have appropriate granularity.\n"
            "2. Cover both fundamental and advanced aspects.\n"
            "3. Create a logical progression from basic to advanced.\n"
            "4. Assign prerequisites to create a meaningful learning path.\n"
            "5. Include both theoretical concepts and practical applications.\n\n"
            "Return a JSON object with an array \"concepts\". Each concept must have:\n"
            "- id: a unique string\n"
            "- name: a clear name\n"
            "- description: a paragraph of 3-5 sentences covering definition, relevance, connections, and an example.\n"
            "- difficulty: a number from 0.0 to 1.0\n"
            "- complexity: a number from 0.0 to 1.0\n"
            "- importance: a number from 0.0 to 1.0\n"
            "- prerequisites: an array of concept IDs\n"
            "- keywords: an array of relevant keywords\n"
            "- estimated_learning_time_minutes: integer estimate of learning time\n"
            "Ensure all JSON syntax is valid."
        ),
        "concept_enrichment_prompt": (
            "Enrich each concept description by adding:\n"
            "1. A teaching approach suggestion (e.g., 'This concept can be effectively taught through hands-on exercises.')\n"
            "2. Common misconceptions students might have\n"
            "3. Real-world applications of the concept\n"
            "4. Assessment strategies to check understanding\n\n"
            "Keep the original description and seamlessly integrate these additions."
        ),
        "relationship_generation_prompt": (
            "Analyze these educational concepts and create meaningful relationships between them:\n\n"
            "{concepts_text}\n\n"
            "Create relationships that establish a coherent knowledge structure with these types:\n"
            "- prerequisite: Concept A must be understood before Concept B\n"
            "- builds_on: Concept B extends or enhances Concept A\n"
            "- related_to: Concepts share significant connections but neither is prerequisite\n"
            "- part_of: Concept A is a component of the broader Concept B\n"
            "- example_of: Concept A illustrates or instantiates Concept B\n"
            "- contrasts_with: Concepts highlight differences or alternative approaches\n\n"
            "For each relationship, specify:\n"
            "1. source_id: ID of the source concept\n"
            "2. target_id: ID of the target concept\n"
            "3. relationship_type: one of the types above\n"
            "4. strength: number from 0.0 to 1.0 indicating relationship strength\n"
            "5. description: brief explanation of the relationship\n\n"
            "Return a JSON object with an array \"relationships\" containing these relationships.\n"
            "Ensure all relationships are pedagogically meaningful and create a coherent learning structure."
        ),
        "learning_path_prompt": (
            "Create an optimal learning path through these concepts for a {level} learner:\n\n"
            "{concepts_text}\n\n"
            "The learner's goal is: {goal}\n\n"
            "Create a sequence of concepts that:\n"
            "1. Starts with appropriate foundational concepts\n"
            "2. Respects prerequisite relationships\n"
            "3. Builds complexity gradually\n"
            "4. Reaches the goal efficiently\n"
            "5. Includes only necessary concepts\n\n"
            "For each step in the path, specify:\n"
            "1. concept_id: ID of the concept to learn\n"
            "2. order: position in sequence (starting from 1)\n"
            "3. estimated_time_minutes: time to spend on this concept\n"
            "4. reason: brief explanation of why this concept is included at this point\n"
            "5. learning_activities: suggested activities to master this concept\n\n"
            "Return a JSON object with \"path\" array containing these steps.\n"
            "Ensure the path is coherent, properly sequenced, and achieves the learning goal."
        ),
        "consistency_check_prompt": (
            "Analyze this knowledge graph for consistency issues:\n\n"
            "Concepts:\n{concepts_text}\n\n"
            "Relationships:\n{relationships_text}\n\n"
            "Identify any of these problems:\n"
            "1. Circular prerequisites (A requires B, B requires C, C requires A)\n"
            "2. Contradictory relationships (A is prerequisite for B, but B is also prerequisite for A)\n"
            "3. Missing prerequisites (concept requires knowledge not represented in prerequisites)\n"
            "4. Isolated concepts (no incoming or outgoing relationships)\n"
            "5. Inconsistent difficulty levels (prerequisites have higher difficulty than dependents)\n"
            "6. Pedagogical sequencing issues (concepts that should be taught together are far apart)\n\n"
            "For each issue found, provide:\n"
            "- issue_type: category of problem\n"
            "- concepts_involved: IDs of relevant concepts\n"
            "- description: detailed explanation of the issue\n"
            "- recommendation: suggested fix\n\n"
            "Return a JSON object with an array \"issues\" containing these problems.\n"
            "If no issues are found, return an empty array."
        ),
        "topic_extraction_prompt": (
            "From this educational domain description:\n"
            "\"{description}\"\n\n"
            "Extract 8-12 key topics that form the core of this domain. For each topic:\n"
            "1. Provide a clear, concise name\n"
            "2. Give a brief (1-2 sentence) description\n"
            "3. Explain why this topic is essential to the domain\n"
            "4. Estimate its relative importance (high/medium/low)\n\n"
            "Ensure topics are:\n"
            "- Distinct yet comprehensive of the domain\n"
            "- At approximately the same level of granularity\n"
            "- Ordered in a logical learning sequence\n\n"
            "Return as a JSON array of topic objects."
        ),
        "knowledge_gap_prompt": (
            "Analyze the current structure of our knowledge domain on {domain}:\n\n"
            "{existing_structure}\n\n"
            "Recent learner questions and interests suggest these areas require attention:\n"
            "{learner_interests}\n\n"
            "Identify knowledge gaps in our current structure that should be addressed to better serve learners. For each gap:\n"
            "1. Describe the missing concept, topic, or relationship\n"
            "2. Explain why it's important based on learner interests\n"
            "3. Specify where it fits in the existing structure\n"
            "4. Suggest content or relationships that should be developed\n\n"
            "Return a JSON object with an array \"gaps\" containing these identified needs.\n"
            "Focus on substantive gaps that represent meaningful learning opportunities."
        ),
        "default_concept_count": 15,
        "max_concepts_gpt35": 25,
        "max_concepts_general": 50,
        "generation_temperature": 0.7,
        "analysis_temperature": 0.2,
        "enrichment_temperature": 0.5
    }
    
    # Create the PromptsConfig with default prompts
    prompts_config = PromptsConfig(**default_prompts)
    
    # Return the complete Config object with the prompts configured
    return Config(
        prompts=prompts_config,
        # Other fields have default values defined in their class declarations
    )

def load_config(config_path: str = DEFAULT_CONFIG_PATH, env: Optional[str] = None) -> Config:
    """Load configuration from file with environment-specific overrides.
    
    This function handles configuration loading with the following precedence:
    1. Environment-specific config file (e.g., config.development.yaml)
    2. Standard config file (e.g., config.yaml)
    3. Default configuration (created if no files exist)
    
    Args:
        config_path: Path to the configuration file
        env: Environment name (development, production, etc.)
        
    Returns:
        Config: Loaded and validated configuration object
    """
    # Use environment from settings if not specified
    if env is None:
        env = env_settings.environment

    # Create path for environment-specific config
    env_config_path = f"{os.path.splitext(config_path)[0]}.{env}.yaml"
    
    # Initialize logger for early logging
    logger = logging.getLogger("config")
    
    try:
        # Load from environment-specific config if it exists
        if os.path.exists(env_config_path):
            config = _load_config_from_file(env_config_path)
            logger.info(f"Loaded environment-specific config from {env_config_path}")
        
        # Fall back to standard config if it exists
        elif os.path.exists(config_path):
            config = _load_config_from_file(config_path)
            logger.info(f"Loaded config from {config_path}")
        
        # Create default config if no files exist
        else:
            config = create_default_config()
            # Ensure directory exists
            config_dir = os.path.dirname(config_path)
            if config_dir:
                os.makedirs(config_dir, exist_ok=True)
                
            # Write default config to file
            with open(config_path, 'w') as f:
                yaml.dump(config.model_dump(), f, default_flow_style=False)
            logger.info(f"Created default config at {config_path}")
            
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        logger.error(traceback.format_exc())
        logger.warning("Falling back to default configuration")
        config = create_default_config()
    
    # Apply environment variable overrides
    _apply_environment_overrides(config)
    
    # Set environment in config
    config.environment = env
    
    return config

def _load_config_from_file(file_path: str) -> Config:
    """Load configuration from a YAML file.
    
    Args:
        file_path: Path to the YAML configuration file
        
    Returns:
        Config: Validated configuration object
    """
    with open(file_path, 'r') as f:
        config_dict = yaml.safe_load(f)
        return Config.model_validate(config_dict)

def _apply_environment_overrides(config: Config) -> None:
    """Apply environment variable overrides to the configuration.
    
    This applies any configuration values that should come from environment
    variables rather than config files.
    
    Args:
        config: The configuration object to modify
    """
    # Apply secure passwords from environment
    if env_settings.neo4j_password:
        config.neo4j.password = env_settings.neo4j_password.get_secret_value()
    
    if env_settings.mongo_password:
        # Extract username and host from URI
        parts = config.mongo.uri.split('@')
        if len(parts) > 1:
            prefix = parts[0].split(':')[0]  # mongodb://user
            suffix = parts[1]  # host:port/db
            config.mongo.uri = f"{prefix}:{env_settings.mongo_password.get_secret_value()}@{suffix}"
    
    # Apply Azure OpenAI settings if endpoint is provided
    if env_settings.azure_openai_endpoint:
        config.llm.use_azure = True
        config.llm.azure_deployment_id = config.llm.default_model

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
    logging.getLogger('neo4j').setLevel(logging.WARNING)

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