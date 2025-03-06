"""
Ptolemy Knowledge Map System - LLM Generation Module
==================================================
Handles knowledge generation and enrichment operations using LLMs.
"""

import json
import logging
import time
import re
import uuid
import os
import traceback
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from datetime import datetime
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, as_completed

# OpenAI integration
try:
    from openai import OpenAI, AzureOpenAI, APIError, APIConnectionError, RateLimitError, APITimeoutError
except ImportError:
    logging.error("OpenAI package not installed. Run 'pip install openai'")

from config import LLMConfig, PromptsConfig
from models import (
    Concept, Relationship, ValidationIssue, ValidationResult,
    ConceptType, RelationshipType, DifficultyLevel, DomainStructureRequest
)

# Configure module-level logger
logger = logging.getLogger("llm.generation")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Function decorator for timing and logging
def log_execution_time(func):
    """Decorator to log execution time of functions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        logger.debug(f"{func.__name__} executed in {execution_time:.2f} seconds")
        return result
    return wrapper

# Function decorator for retrying on failure
def retry_on_exception(max_retries=3, retry_delay=1, allowed_exceptions=(Exception,)):
    """Decorator to retry a function on exception."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except allowed_exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        sleep_time = retry_delay * (2 ** attempt)
                        logger.warning(f"Attempt {attempt+1} failed: {e}. Retrying in {sleep_time} seconds...")
                        time.sleep(sleep_time)
                    else:
                        logger.error(f"All {max_retries+1} attempts failed. Last error: {e}")
            
            # If we get here, all retries failed
            raise last_exception
        return wrapper
    return decorator

class LLMService:
    """Service for LLM-based generation and enrichment."""
    
    def __init__(self, llm_config: LLMConfig, prompts_config: PromptsConfig):
        """Initialize the LLM service with configuration."""
        self.config = llm_config
        self.prompts = prompts_config
        self.client = self._initialize_client()
        
        # Cache for storing recent results to avoid duplicate calls
        self._cache = {}
        self._cache_ttl = 3600  # Cache TTL in seconds (1 hour)
        
        # Initialize thread pool for concurrent operations
        self._executor = ThreadPoolExecutor(max_workers=self.config.max_concurrent_requests or 5)
        
        # Load fallback templates
        self._load_fallback_templates()
        
        logger.info(f"LLM Service initialized with model: {self.config.model}")
    
    def _load_fallback_templates(self):
        """Load fallback templates in case prompt config is incomplete."""
        self._fallback_templates = {
            "domain_analysis_prompt": """
                Analyze the domain '{domain}' with the description: 
                {description}
                
                {topics_text}
                
                Determine the optimal number of concepts needed to adequately cover this domain for educational purposes.
                Consider factors like domain breadth, complexity, and target audience.
                
                Output JSON with the following structure:
                {
                    "recommended_concept_count": <integer>,
                    "suggested_model": <string>,  // either "gpt-3.5-turbo" or "gpt-4-turbo"
                    "justification": <string>
                }
            """,
            "concept_generation_prompt": """
                Generate {num_concepts} educational concepts for the domain '{domain}' with the description:
                {description}
                
                {topics_text}
                
                For each concept, provide:
                1. A descriptive name
                2. A detailed educational description suitable for teaching
                3. Difficulty level (beginner, intermediate, advanced)
                4. Importance score (0.0-1.0) where higher means more central to the domain
                5. Complexity score (0.0-1.0) where higher means more complex
                6. Keywords or tags
                7. Prerequisites (IDs of other concepts that should be learned first)
                8. Estimated learning time in minutes
                
                Assign each concept a unique ID (can be a short string).
                
                Output JSON with the following structure:
                {
                    "concepts": [
                        {
                            "id": <string>,
                            "name": <string>,
                            "description": <string>,
                            "difficulty": <string>,
                            "importance": <float>,
                            "complexity": <float>,
                            "keywords": [<string>],
                            "prerequisites": [<string>],
                            "estimated_learning_time_minutes": <integer>
                        }
                    ]
                }
            """
        }
    
    def _get_prompt(self, prompt_name):
        """Get prompt from config or fallback to template."""
        prompt = getattr(self.prompts, prompt_name, None)
        if not prompt and prompt_name in self._fallback_templates:
            prompt = self._fallback_templates[prompt_name]
        return prompt
    
    def _initialize_client(self) -> Union[OpenAI, AzureOpenAI, None]:
        """Initialize the OpenAI client based on configuration."""
        try:
            if self.config.use_azure:
                # Azure OpenAI setup
                api_key = os.getenv("AZURE_OPENAI_KEY")
                endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
                
                if not api_key or not endpoint:
                    logger.error("Azure OpenAI API key or endpoint not found in environment")
                    return None
                
                return AzureOpenAI(
                    api_key=api_key,
                    api_version=self.config.azure_api_version,
                    azure_endpoint=endpoint
                )
            else:
                # Standard OpenAI setup
                api_key = os.getenv("OPENAI_API_KEY")
                
                if not api_key:
                    logger.error("OpenAI API key not found in environment")
                    return None
                
                client = OpenAI(api_key=api_key)
                
                # Test connection with a simple model check
                try:
                    logger.debug("Testing OpenAI connection...")
                    # Typically would call list_models() but we'll skip the actual API call here
                    # to avoid unnecessary API usage
                    logger.info("OpenAI client initialized successfully")
                except Exception as e:
                    logger.error(f"Failed to test OpenAI connection: {e}")
                    return None
                
                return client
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            return None

    def _validate_and_normalize_concept_fields(self, concept):
        """Validate and normalize concept fields to ensure they match expected types."""
        if not isinstance(concept, dict):
            return concept
            
        # Validate and normalize difficulty field
        if "difficulty" in concept:
            difficulty = concept["difficulty"]
            # If difficulty is a number, map it to appropriate string enum
            if isinstance(difficulty, (int, float)):
                # Map numeric values to string enum values
                if difficulty <= 0.33:
                    concept["difficulty"] = "beginner"
                elif difficulty <= 0.66:
                    concept["difficulty"] = "intermediate"
                else:
                    concept["difficulty"] = "advanced"
            # If difficulty is a string but not a valid enum value
            elif isinstance(difficulty, str) and difficulty.lower() not in ["beginner", "intermediate", "advanced"]:
                # Default to intermediate for invalid string values
                concept["difficulty"] = "intermediate"
                
        # Ensure importance and complexity are float values between 0 and 1
        for field in ["importance", "complexity"]:
            if field in concept and not isinstance(concept[field], float):
                try:
                    # Try to convert to float
                    concept[field] = float(concept[field])
                    # Ensure value is between 0 and 1
                    concept[field] = max(0.0, min(1.0, concept[field]))
                except (ValueError, TypeError):
                    # Set default value if conversion fails
                    concept[field] = 0.5
                    
        return concept
    
    @log_execution_time
    def _call_openai_with_retry(self, messages: List[Dict[str, str]], model: str, 
                              temperature: float, max_tokens: int, 
                              json_mode: bool = True) -> Optional[str]:
        """Call OpenAI API with retry logic."""
        if not self.client:
            logger.error("LLM client not available")
            return None
        
        # Generate cache key from input parameters
        cache_key = self._generate_cache_key(messages, model, temperature, max_tokens, json_mode)
        
        # Check cache
        cached_result = self._check_cache(cache_key)
        if cached_result:
            logger.info("Using cached LLM response")
            return cached_result
        
        for attempt in range(self.config.retry_count + 1):
            try:
                # Response format for JSON mode
                response_format = {"type": "json_object"} if json_mode else None
                
                # Select correct deployment ID for Azure
                api_model = model
                if self.config.use_azure:
                    api_model = self.config.azure_deployment_id or model
                
                # Sanitize messages
                cleaned_messages = []
                for msg in messages:
                    # Ensure correct roles and content
                    if "role" not in msg or msg["role"] not in ["system", "user", "assistant"]:
                        continue
                    if "content" not in msg or not isinstance(msg["content"], str):
                        continue
                    cleaned_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
                
                # If no valid messages, return error
                if not cleaned_messages:
                    logger.error("No valid messages provided")
                    return None
                
                # Log request details (but not full content for privacy)
                logger.info(f"Calling {api_model} with {len(cleaned_messages)} messages, temp={temperature}, max_tokens={max_tokens}")
                
                # Track request start time
                start_time = time.time()
                
                # Make API call
                response = self.client.chat.completions.create(
                    model=api_model,
                    messages=cleaned_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                    timeout=self.config.timeout
                )
                
                # Log request duration
                duration = time.time() - start_time
                logger.info(f"LLM request completed in {duration:.2f} seconds")
                
                result = response.choices[0].message.content
                
                # Cache the result
                self._update_cache(cache_key, result)
                
                return result
            
            except RateLimitError as e:
                if attempt < self.config.retry_count:
                    # Exponential backoff
                    sleep_time = self.config.retry_delay * (2 ** attempt)
                    logger.warning(f"Rate limit exceeded. Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"Rate limit exceeded after {attempt+1} attempts")
                    return None
            
            except (APIConnectionError, APITimeoutError) as e:
                if attempt < self.config.retry_count:
                    sleep_time = self.config.retry_delay * (2 ** attempt)
                    logger.warning(f"Connection error: {e}. Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"Connection error after {attempt+1} attempts: {e}")
                    return None
            
            except APIError as e:
                logger.error(f"OpenAI API error: {e}")
                return None
            
            except Exception as e:
                logger.error(f"Unexpected error calling LLM: {e}")
                logger.debug(traceback.format_exc())
                return None
    
    def _generate_cache_key(self, messages, model, temperature, max_tokens, json_mode):
        """Generate a cache key based on request parameters."""
        # Only use message content for key, ignore other fields
        message_str = "|".join([msg.get("content", "") for msg in messages])
        return f"{model}-{temperature}-{max_tokens}-{json_mode}-{hash(message_str)}"
    
    def _check_cache(self, key):
        """Check if there's a valid cached response for key."""
        if key in self._cache:
            item = self._cache[key]
            if time.time() - item["timestamp"] < self._cache_ttl:
                return item["response"]
            else:
                # Expired
                del self._cache[key]
        return None
    
    def _update_cache(self, key, response):
        """Update cache with new response."""
        self._cache[key] = {
            "response": response,
            "timestamp": time.time()
        }
        
        # Clean old cache entries
        self._clean_cache()
    
    def _clean_cache(self):
        """Clean expired cache entries."""
        now = time.time()
        expired_keys = [k for k, v in self._cache.items() if now - v["timestamp"] > self._cache_ttl]
        for key in expired_keys:
            del self._cache[key]
    
    @log_execution_time
    def _repair_json(self, json_str: str) -> str:
        """Attempt to repair malformed JSON from LLM responses."""
        try:
            # Try to parse as is
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError as e:
            logger.warning(f"Attempting to repair JSON: {e}")
            
            # First try: Check if JSON is wrapped in markdown code blocks
            code_block_pattern = r"```(?:json)?(.*?)```"
            matches = re.findall(code_block_pattern, json_str, re.DOTALL)
            if matches:
                for match in matches:
                    try:
                        # Try each code block
                        candidate = match.strip()
                        json.loads(candidate)
                        logger.info("Found valid JSON in code block")
                        return candidate
                    except json.JSONDecodeError:
                        continue
            
            # Common JSON fixes
            # Remove any text before the first curly brace
            json_str = re.sub(r'^[^{]*', '', json_str)
            
            # Remove any text after the last curly brace
            json_str = re.sub(r'[^}]*$', '', json_str)
            
            # Fix unclosed quotes at end of string
            if "Unterminated string" in str(e) and ("at the end" in str(e) or e.pos >= len(json_str) - 10):
                json_str += '"'
            
            # Fix missing quotes on lines with odd number of quotes
            lines = json_str.split('\n')
            for i, line in enumerate(lines):
                if line.count('"') % 2 == 1:
                    lines[i] = line + '"'
            json_str = '\n'.join(lines)
            
            # Fix trailing commas
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*]', ']', json_str)
            
            # Fix missing quotes around property names
            json_str = re.sub(r'([{,]\s*)([a-zA-Z0-9_]+)(\s*:)', r'\1"\2"\3', json_str)
            
            # Try parsing again
            try:
                json.loads(json_str)
                logger.info("JSON repair successful")
                return json_str
            except json.JSONDecodeError:
                logger.error("JSON repair attempt failed")
                
                # Last resort: try a second repair pass with LLM
                try:
                    repaired = self._repair_json_with_llm(json_str)
                    if repaired:
                        return repaired
                except Exception as repair_e:
                    logger.error(f"LLM JSON repair failed: {repair_e}")
                
                # Return original string if all repairs fail
                return json_str
    
    def _repair_json_with_llm(self, json_str: str) -> Optional[str]:
        """Use LLM to repair malformed JSON as a last resort."""
        if not self.client:
            return None
            
        prompt = f"""
        The following JSON is malformed. Please fix it and return only valid JSON with no explanations or 
        markdown formatting. The content should be preserved as much as possible:
        
        {json_str}
        """
        
        messages = [
            {"role": "system", "content": "You are a JSON repair expert. Return only valid JSON, no explanations."},
            {"role": "user", "content": prompt}
        ]
        
        repaired = self._call_openai_with_retry(
            messages=messages,
            model="gpt-3.5-turbo",
            temperature=0.1,  # Low temperature for deterministic response
            max_tokens=4000,
            json_mode=False
        )
        
        if not repaired:
            return None
            
        # Extract valid JSON from the response
        try:
            # Attempt to parse as-is
            json.loads(repaired)
            logger.info("LLM JSON repair successful")
            return repaired
        except json.JSONDecodeError:
            # Try to extract JSON from markdown or text
            code_block_pattern = r"```(?:json)?(.*?)```"
            matches = re.findall(code_block_pattern, repaired, re.DOTALL)
            if matches:
                for match in matches:
                    try:
                        candidate = match.strip()
                        json.loads(candidate)
                        logger.info("Found valid JSON in LLM repair response")
                        return candidate
                    except json.JSONDecodeError:
                        continue
            
            # Failed to repair
            return None
    
    @log_execution_time
    def analyze_domain(self, domain: str, description: str, 
                      key_topics: Optional[List[str]] = None) -> Dict[str, Any]:
        """Analyze a domain to determine optimal concept count and structure."""
        if not self.client:
            logger.error("LLM client not available")
            return {"concept_count": self.prompts.default_concept_count, "model": "gpt-3.5-turbo"}
        
        # Prepare topics text
        topics_text = ""
        if key_topics:
            topics_text = "Key topics to consider:\n" + "\n".join(f"- {topic}" for topic in key_topics)
        
        # Get and format prompt
        prompt_text = self._get_prompt("domain_analysis_prompt").format(
            domain=domain,
            description=description,
            topics_text=topics_text
        )
        
        logger.info(f"Analyzing domain '{domain}' to determine optimal concept count")
        
        # Call LLM
        messages = [
            {"role": "system", "content": "You are an expert curriculum designer."},
            {"role": "user", "content": prompt_text}
        ]
        
        try:
            # Determine model to use - using a lighter model for analysis is fine
            analysis_model = "gpt-3.5-turbo"
            
            json_str = self._call_openai_with_retry(
                messages=messages,
                model=analysis_model,
                temperature=self.prompts.analysis_temperature,
                max_tokens=self.config.max_tokens_analysis,
                json_mode=True
            )
            
            if not json_str:
                logger.warning("Failed to get domain analysis. Using defaults.")
                return {"concept_count": self.prompts.default_concept_count, "model": "gpt-3.5-turbo"}
            
            # Parse and validate response
            repaired_json = self._repair_json(json_str)
            analysis = json.loads(repaired_json)
            
            # Extract and validate results
            concept_count = int(analysis.get("recommended_concept_count", self.prompts.default_concept_count))
            suggested_model = analysis.get("suggested_model", "gpt-3.5-turbo")
            justification = analysis.get("justification", "")
            
            # Apply sanity checks
            if concept_count < 3:
                logger.warning(f"Adjusted concept count from {concept_count} to minimum of 3")
                concept_count = 3
            elif concept_count > 50:
                logger.warning(f"Adjusted concept count from {concept_count} to maximum of 50")
                concept_count = 50
            
            # Validate model selection
            if suggested_model not in ["gpt-3.5-turbo", "gpt-4-turbo"]:
                suggested_model = "gpt-3.5-turbo"
            
            logger.info(f"Domain analysis complete. Recommended concepts: {concept_count}, Model: {suggested_model}")
            
            return {
                "concept_count": concept_count, 
                "model": suggested_model,
                "justification": justification
            }
        except Exception as e:
            logger.error(f"Error in domain analysis: {e}")
            logger.debug(traceback.format_exc())
            return {"concept_count": self.prompts.default_concept_count, "model": "gpt-3.5-turbo"}
    
    @log_execution_time
    def extract_topics(self, description: str) -> List[Dict[str, Any]]:
        """Extract key topics from a domain description."""
        if not self.client:
            logger.error("LLM client not available")
            return []
        
        # Format prompt
        prompt = self.prompts.topic_extraction_prompt.format(
            description=description
        )
        
        # Call LLM
        messages = [
            {"role": "system", "content": "You are an expert curriculum designer."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            json_str = self._call_openai_with_retry(
                messages=messages,
                model="gpt-3.5-turbo",
                temperature=self.prompts.analysis_temperature,
                max_tokens=self.config.max_tokens_analysis,
                json_mode=True
            )
            
            if not json_str:
                logger.warning("Failed to extract topics")
                return []
            
            # Parse and validate response
            repaired_json = self._repair_json(json_str)
            data = json.loads(repaired_json)
            
            # Handle different response formats
            topics = []
            if isinstance(data, list):
                topics = data
            elif isinstance(data, dict):
                for key in ["topics", "key_topics", "results"]:
                    if key in data and isinstance(data[key], list):
                        topics = data[key]
                        break
            
            # Standardize format
            standardized_topics = []
            for topic in topics:
                if isinstance(topic, str):
                    standardized_topics.append({"name": topic})
                elif isinstance(topic, dict) and "name" in topic:
                    standardized_topics.append(topic)
            
            logger.info(f"Extracted {len(standardized_topics)} topics from description")
            return standardized_topics
        except Exception as e:
            logger.error(f"Error extracting topics: {e}")
            logger.debug(traceback.format_exc())
            return []
    
    @log_execution_time
    def generate_concepts(self, domain: str, description: str, 
                         num_concepts: Optional[int] = None,
                         key_topics: Optional[List[str]] = None,
                         model: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Generate a structured knowledge graph of concepts for a domain."""
        if not self.client:
            logger.error("LLM client not available")
            return None
        
        # Determine concept count and model if not provided
        if num_concepts is None or model is None:
            analysis = self.analyze_domain(domain, description, key_topics)
            if num_concepts is None:
                num_concepts = analysis["concept_count"]
            if model is None:
                model = analysis["model"]
        
        # Apply model-specific limits
        if model.startswith("gpt-3.5") and num_concepts > self.prompts.max_concepts_gpt35:
            logger.warning(f"Reducing concept count from {num_concepts} to {self.prompts.max_concepts_gpt35} for GPT-3.5")
            num_concepts = self.prompts.max_concepts_gpt35
        elif num_concepts > self.prompts.max_concepts_general:
            logger.warning(f"Reducing concept count from {num_concepts} to {self.prompts.max_concepts_general} for reliability")
            num_concepts = self.prompts.max_concepts_general
        
        # Prepare topics text
        topics_text = ""
        if key_topics:
            topics_text = "Key topics to include:\n" + "\n".join(f"- {topic}" for topic in key_topics)
        
        # Format prompt
        prompt_text = self._get_prompt("concept_generation_prompt").format(
            num_concepts=num_concepts,
            domain=domain,
            description=description,
            topics_text=topics_text
        )
        
        logger.info(f"Generating {num_concepts} concepts for '{domain}' using {model}")
        
        # Call LLM
        messages = [
            {"role": "system", "content": "You are a curriculum design expert who produces valid JSON."},
            {"role": "user", "content": prompt_text}
        ]
        
        try:
            # Adjust token limit based on concept count
            max_tokens = self.config.max_tokens_generation
            if num_concepts > 25:
                max_tokens = 6000
            
            json_str = self._call_openai_with_retry(
                messages=messages,
                model=model,
                temperature=self.prompts.generation_temperature,
                max_tokens=max_tokens,
                json_mode=True
            )
            
            if not json_str:
                logger.error("Failed to generate concepts")
                return None
            
            # Parse and validate response
            repaired_json = self._repair_json(json_str)
            data = json.loads(repaired_json)
            
            if "concepts" in data and isinstance(data["concepts"], list) and len(data["concepts"]) > 0:
                logger.info(f"Successfully generated {len(data['concepts'])} concepts")
                
                # Ensure each concept has a valid ID and normalized fields
                for i, concept in enumerate(data["concepts"]):
                    if "id" not in concept or not concept["id"]:
                        concept["id"] = str(uuid.uuid4())
                    
                    # Validate and normalize fields
                    data["concepts"][i] = self._validate_and_normalize_concept_fields(concept)
                
                # Enrich with domain metadata
                data["domain"] = domain
                data["description"] = description
                data["generated_at"] = datetime.now().isoformat()
                data["model_used"] = model
                
                return data
            else:
                logger.error("Response valid JSON but missing 'concepts' array")
                return None
        except Exception as e:
            logger.error(f"Error generating concepts: {e}")
            logger.debug(traceback.format_exc())
            return None
    
    @log_execution_time
    def enrich_concepts(self, concepts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enrich concept descriptions with teaching approaches and other educational metadata."""
        if not self.client or not concepts:
            return concepts
        
        logger.info(f"Enriching {len(concepts)} concepts with educational metadata")
        
        # For large concept sets, break into chunks for better processing
        if len(concepts) > 15:
            return self._batch_process_concepts(concepts, self._enrich_concept_batch, "enrichment")
        
        # Prepare input for the LLM
        concepts_text = json.dumps({"concepts": concepts}, indent=2)
        
        # Format prompt for enrichment
        prompt_text = self.prompts.concept_enrichment_prompt + "\n\n" + concepts_text
        
        # Call LLM
        messages = [
            {"role": "system", "content": "You are an expert educational content developer."},
            {"role": "user", "content": prompt_text}
        ]
        
        try:
            json_str = self._call_openai_with_retry(
                messages=messages,
                model="gpt-3.5-turbo",
                temperature=self.prompts.enrichment_temperature,
                max_tokens=self.config.max_tokens_enrichment,
                json_mode=True
            )
            
            if not json_str:
                logger.warning("Failed to enrich concepts. Returning originals.")
                return concepts
            
            # Parse and validate response
            repaired_json = self._repair_json(json_str)
            data = json.loads(repaired_json)
            
            if "concepts" in data and isinstance(data["concepts"], list) and len(data["concepts"]) > 0:
                logger.info(f"Successfully enriched {len(data['concepts'])} concepts")
                
                # Merge with original concepts to ensure no data is lost
                enriched_concepts = self._merge_with_originals(concepts, data["concepts"])
                
                return enriched_concepts
            else:
                logger.warning("Response valid JSON but missing 'concepts' array. Returning originals.")
                return concepts
        except Exception as e:
            logger.error(f"Error enriching concepts: {e}")
            logger.debug(traceback.format_exc())
            return concepts
    
    def _enrich_concept_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of concepts for enrichment."""
        # Re-call enrich_concepts with the batch
        return self.enrich_concepts(batch)
    
    def _batch_process_concepts(self, concepts: List[Dict[str, Any]], 
                               batch_processor: Callable, operation_name: str,
                               batch_size: int = 10) -> List[Dict[str, Any]]:
        """Process concepts in batches using the provided processor function."""
        if not concepts:
            return []
            
        # Split into batches
        batches = [concepts[i:i+batch_size] for i in range(0, len(concepts), batch_size)]
        logger.info(f"Processing {len(concepts)} concepts in {len(batches)} batches for {operation_name}")
        
        results = []
        futures = []
        
        # Process batches in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(len(batches), 5)) as executor:
            for batch in batches:
                future = executor.submit(batch_processor, batch)
                futures.append(future)
            
            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    batch_result = future.result()
                    if batch_result:
                        results.extend(batch_result)
                except Exception as e:
                    logger.error(f"Error processing batch: {e}")
        
        # If some batches failed, ensure we return at least the original concepts
        if len(results) < len(concepts):
            logger.warning(f"Some {operation_name} batches failed, returning mix of processed and original concepts")
            
            # Create a map of processed concept IDs
            processed_ids = {c["id"]: c for c in results if "id" in c}
            
            # Add any concepts that weren't processed
            for concept in concepts:
                if "id" in concept and concept["id"] not in processed_ids:
                    results.append(concept)
        
        logger.info(f"Completed batch {operation_name} with {len(results)} concepts")
        return results
    
    def _merge_with_originals(self, original_concepts: List[Dict[str, Any]], 
                             enriched_concepts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge enriched concepts with originals to ensure no data is lost."""
        # Create a map of original concepts
        original_map = {c["id"]: c for c in original_concepts if "id" in c}
        
        # Create merged result
        merged = []
        
        # Process each enriched concept
        for enriched in enriched_concepts:
            if "id" in enriched and enriched["id"] in original_map:
                # Get the original concept
                original = original_map[enriched["id"]]
                
                # Create merged concept with original as base
                merged_concept = original.copy()
                
                # Update with enriched values, but don't overwrite with empty values
                for key, value in enriched.items():
                    if value is not None and value != "" and (key not in original or original[key] == ""):
                        merged_concept[key] = value
                
                # Validate fields
                merged_concept = self._validate_and_normalize_concept_fields(merged_concept)
                
                merged.append(merged_concept)
            else:
                # If no matching original, use the enriched but validate fields
                merged.append(self._validate_and_normalize_concept_fields(enriched))
        
        # Add any originals that weren't enriched
        enriched_ids = {c["id"] for c in enriched_concepts if "id" in c}
        for concept_id, concept in original_map.items():
            if concept_id not in enriched_ids:
                merged.append(concept)
        
        return merged
    
    @log_execution_time
    def generate_relationships(self, concepts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate meaningful relationships between concepts."""
        if not self.client or not concepts:
            return []
        
        logger.info(f"Generating relationships between {len(concepts)} concepts")
        
        # For large concept sets, process in smaller batches
        if len(concepts) > 20:
            return self._batch_generate_relationships(concepts)
        
        # Prepare concepts text (simplified to reduce token count)
        simplified_concepts = []
        for concept in concepts:
            if not isinstance(concept, dict) or "id" not in concept or "name" not in concept:
                continue
                
            simplified_concepts.append({
                "id": concept["id"],
                "name": concept["name"],
                "description": concept.get("description", "")[:200] + "..." if len(concept.get("description", "")) > 200 else concept.get("description", ""),
                "concept_type": concept.get("concept_type", "topic"),
                "difficulty": concept.get("difficulty", "intermediate")
            })
        
        concepts_text = json.dumps(simplified_concepts, indent=2)
        
        # Format prompt
        prompt_text = self.prompts.relationship_generation_prompt.format(
            concepts_text=concepts_text
        )
        
        # Call LLM
        messages = [
            {"role": "system", "content": "You are an expert knowledge graph designer."},
            {"role": "user", "content": prompt_text}
        ]
        
        try:
            max_tokens = self.config.max_tokens_generation
            if len(concepts) > 10:
                max_tokens = 6000
            
            # Use GPT-4 for better relationship quality when possible
            model = "gpt-4-turbo" if self.config.allow_gpt4 else "gpt-3.5-turbo"
            
            json_str = self._call_openai_with_retry(
                messages=messages,
                model=model,
                temperature=self.prompts.generation_temperature,
                max_tokens=max_tokens,
                json_mode=True
            )
            
            if not json_str:
                logger.error("Failed to generate relationships")
                return []
            
            # Parse and validate response
            repaired_json = self._repair_json(json_str)
            data = json.loads(repaired_json)
            
            if "relationships" in data and isinstance(data["relationships"], list):
                # Add unique IDs and created timestamps
                for relationship in data["relationships"]:
                    if "id" not in relationship:
                        relationship["id"] = str(uuid.uuid4())
                    if "created_at" not in relationship:
                        relationship["created_at"] = datetime.now().isoformat()
                    
                    # Ensure required fields are present
                    if "source_id" not in relationship or "target_id" not in relationship:
                        continue
                        
                    # Skip self-relationships
                    if relationship["source_id"] == relationship["target_id"]:
                        continue
                
                # Filter out invalid relationships
                valid_relationships = []
                concept_ids = {c["id"] for c in concepts if "id" in c}
                
                for rel in data["relationships"]:
                    if "source_id" in rel and "target_id" in rel:
                        # Check that both source and target exist in our concepts
                        if rel["source_id"] in concept_ids and rel["target_id"] in concept_ids:
                            valid_relationships.append(rel)
                
                logger.info(f"Successfully generated {len(valid_relationships)} valid relationships")
                return valid_relationships
            else:
                logger.error("Response valid JSON but missing 'relationships' array")
                return []
        except Exception as e:
            logger.error(f"Error generating relationships: {e}")
            logger.debug(traceback.format_exc())
            return []
    
    def _batch_generate_relationships(self, concepts: List[Dict[str, Any]], batch_size: int = 15) -> List[Dict[str, Any]]:
        """Generate relationships in batches for large concept sets."""
        logger.info(f"Using batch relationship generation for {len(concepts)} concepts")
        
        # Split concepts into overlapping batches
        all_relationships = []
        concept_count = len(concepts)
        
        if concept_count <= batch_size:
            # If fewer concepts than batch size, just process directly
            return self.generate_relationships(concepts)
        
        # Create batches with some overlap
        batches = []
        for i in range(0, concept_count, batch_size - 5):
            end_idx = min(i + batch_size, concept_count)
            batch = concepts[i:end_idx]
            batches.append(batch)
        
        logger.info(f"Split into {len(batches)} batches for relationship generation")
        
        # Process each batch
        futures = []
        with ThreadPoolExecutor(max_workers=min(len(batches), 3)) as executor:
            for batch in batches:
                future = executor.submit(self.generate_relationships, batch)
                futures.append(future)
            
            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    batch_relationships = future.result()
                    all_relationships.extend(batch_relationships)
                except Exception as e:
                    logger.error(f"Error processing relationship batch: {e}")
        
        # Deduplicate relationships
        unique_rels = {}
        for rel in all_relationships:
            # Create a key based on source, target and type
            if "source_id" in rel and "target_id" in rel and "relationship_type" in rel:
                key = f"{rel['source_id']}-{rel['target_id']}-{rel['relationship_type']}"
                if key not in unique_rels:
                    unique_rels[key] = rel
        
        logger.info(f"Generated {len(unique_rels)} unique relationships across all batches")
        return list(unique_rels.values())

    @log_execution_time
    def generate_learning_path(self, goal: str, concepts: List[Dict[str, Any]], 
                               level: str = "beginner") -> Dict[str, Any]:
        """Generate an optimal learning path through concepts for a specific goal."""
        if not self.client or not concepts:
            return {"path": []}
        
        logger.info(f"Generating learning path for goal: '{goal}' at {level} level")
        
        # Prepare simplified concepts text
        simplified_concepts = []
        for concept in concepts:
            if not isinstance(concept, dict) or "id" not in concept or "name" not in concept:
                continue
                
            simplified_concepts.append({
                "id": concept["id"],
                "name": concept["name"],
                "description": (concept.get("description", "")[:150] + "..."
                                if len(concept.get("description", "")) > 150
                                else concept.get("description", "")),
                "difficulty": concept.get("difficulty", "intermediate"),
                "prerequisites": concept.get("prerequisites", [])
            })
        
        concepts_text = json.dumps(simplified_concepts, indent=2)
        
        # Format prompt
        prompt_text = self.prompts.learning_path_prompt.format(
            level=level,
            concepts_text=concepts_text,
            goal=goal
        )
        
        # Call LLM
        messages = [
            {"role": "system", "content": "You are an expert educational pathway designer."},
            {"role": "user", "content": prompt_text}
        ]
        
        try:
            # Use GPT-4 for better pathway quality when possible
            model = "gpt-4-turbo" if self.config.allow_gpt4 else "gpt-3.5-turbo"
            
            json_str = self._call_openai_with_retry(
                messages=messages,
                model=model,
                temperature=self.prompts.generation_temperature,
                max_tokens=self.config.max_tokens_generation,
                json_mode=True
            )
            
            if not json_str:
                logger.error("Failed to generate learning path")
                return {"path": []}
            
            # Parse and validate response
            repaired_json = self._repair_json(json_str)
            data = json.loads(repaired_json)
            
            if "path" in data and isinstance(data["path"], list):
                # Validate and sanitize the path
                valid_path = []
                concept_ids = {c["id"] for c in concepts if "id" in c}
                concept_names = {c["id"]: c["name"] for c in concepts if "id" in c and "name" in c}
                
                for step in data["path"]:
                    if not isinstance(step, dict):
                        continue
                        
                    # Ensure step has a valid concept_id
                    if "concept_id" not in step or step["concept_id"] not in concept_ids:
                        continue
                    
                    # Add concept_name if missing
                    if "concept_name" not in step and step["concept_id"] in concept_names:
                        step["concept_name"] = concept_names[step["concept_id"]]
                    
                    # Ensure step has a description
                    if "description" not in step:
                        step["description"] = "Learn this concept to progress toward your goal."
                    
                    # --- FIX: Ensure learning_activities is a list ---
                    if "learning_activities" in step:
                        if not isinstance(step["learning_activities"], list):
                            step["learning_activities"] = [step["learning_activities"]]
                    else:
                        step["learning_activities"] = []
                    # ---------------------------------------------------
                    
                    valid_path.append(step)
                
                # Add metadata
                result = {
                    "goal": goal,
                    "level": level,
                    "generated_at": datetime.now().isoformat(),
                    "path": valid_path
                }
                
                # Calculate total time
                total_time = sum(step.get("estimated_time_minutes", 30) for step in valid_path)
                result["total_time_minutes"] = total_time
                
                logger.info(f"Successfully generated learning path with {len(valid_path)} steps")
                return result
            else:
                logger.error("Response valid JSON but missing 'path' array")
                return {"path": []}
        except Exception as e:
            logger.error(f"Error generating learning path: {e}")
            logger.debug(traceback.format_exc())
            return {"path": []}


    @log_execution_time
    def validate_knowledge_graph(self, concepts: List[Dict[str, Any]], 
                               relationships: List[Dict[str, Any]]) -> ValidationResult:
        """Validate a knowledge graph for consistency and quality issues."""
        if not self.client or not concepts or not relationships:
            return ValidationResult(valid=False, issues=[], warnings=[], timestamp=datetime.now())
        
        logger.info(f"Validating knowledge graph with {len(concepts)} concepts and {len(relationships)} relationships")
        
        # For large graphs, perform a simplified validation
        if len(concepts) > 30 or len(relationships) > 100:
            return self._simplified_validation(concepts, relationships)
        
        # Prepare simplified concepts and relationships text
        simplified_concepts = []
        for concept in concepts:
            if not isinstance(concept, dict) or "id" not in concept or "name" not in concept:
                continue
                
            simplified_concepts.append({
                "id": concept["id"],
                "name": concept["name"],
                "concept_type": concept.get("concept_type", "topic"),
                "difficulty": concept.get("difficulty", "intermediate")
            })
        
        simplified_relationships = []
        for relationship in relationships:
            if not isinstance(relationship, dict) or "source_id" not in relationship or "target_id" not in relationship:
                continue
                
            simplified_relationships.append({
                "source_id": relationship["source_id"],
                "target_id": relationship["target_id"],
                "relationship_type": relationship.get("relationship_type", "related_to"),
                "strength": relationship.get("strength", 0.5)
            })
        
        concepts_text = json.dumps(simplified_concepts, indent=2)
        relationships_text = json.dumps(simplified_relationships, indent=2)
        
        # Format prompt
        prompt_text = self.prompts.consistency_check_prompt.format(
            concepts_text=concepts_text,
            relationships_text=relationships_text
        )
        
        # Call LLM
        messages = [
            {"role": "system", "content": "You are an expert knowledge graph validator."},
            {"role": "user", "content": prompt_text}
        ]
        
        try:
            json_str = self._call_openai_with_retry(
                messages=messages,
                model="gpt-4-turbo" if self.config.allow_gpt4 else "gpt-3.5-turbo",
                temperature=self.prompts.analysis_temperature,
                max_tokens=self.config.max_tokens_analysis,
                json_mode=True
            )
            
            if not json_str:
                logger.error("Failed to validate knowledge graph")
                return ValidationResult(valid=False, issues=[], warnings=[], stats={
                    "concepts": len(concepts),
                    "relationships": len(relationships)
                }, timestamp=datetime.now())
            
            # Parse and validate response
            repaired_json = self._repair_json(json_str)
            data = json.loads(repaired_json)
            
            issues = []
            warnings = []
            
            if "issues" in data and isinstance(data["issues"], list):
                for issue_data in data["issues"]:
                    if not isinstance(issue_data, dict):
                        continue
                        
                    severity = issue_data.get("severity", "medium").lower()
                    
                    # Create ValidationIssue object
                    issue = ValidationIssue(
                        issue_type=issue_data.get("issue_type", "unknown"),
                        severity=severity,
                        concepts_involved=issue_data.get("concepts_involved", []),
                        description=issue_data.get("description", "No description provided"),
                        recommendation=issue_data.get("recommendation", "No recommendation provided")
                    )
                    
                    if severity in ["high", "critical"]:
                        issues.append(issue)
                    else:
                        warnings.append(issue)
            
            # Determine if valid based on critical issues
            valid = len(issues) == 0
            
            # Collect stats
            stats = {
                "concepts": len(concepts),
                "relationships": len(relationships),
                "issues": len(issues),
                "warnings": len(warnings)
            }
            
            result = ValidationResult(
                valid=valid,
                issues=issues,
                warnings=warnings,
                stats=stats,
                timestamp=datetime.now()
            )
            
            logger.info(f"Validation complete. Valid: {valid}, Issues: {len(issues)}, Warnings: {len(warnings)}")
            return result
        except Exception as e:
            logger.error(f"Error validating knowledge graph: {e}")
            logger.debug(traceback.format_exc())
            return ValidationResult(valid=False, issues=[], warnings=[], stats={
                "concepts": len(concepts),
                "relationships": len(relationships)
            }, timestamp=datetime.now())
    
    def _simplified_validation(self, concepts: List[Dict[str, Any]], 
                             relationships: List[Dict[str, Any]]) -> ValidationResult:
        """Perform simple programmatic validation for large graphs."""
        logger.info("Performing simplified validation for large graph")
        
        issues = []
        warnings = []
        
        # Create sets for checking
        concept_ids = {c["id"] for c in concepts if "id" in c}
        relationship_source_targets = set()
        
        # Check for duplicate relationships
        for rel in relationships:
            if "source_id" not in rel or "target_id" not in rel:
                continue
                
            source_id = rel["source_id"]
            target_id = rel["target_id"]
            rel_type = rel.get("relationship_type", "related_to")
            
            # Check that source and target concepts exist
            if source_id not in concept_ids:
                warnings.append(ValidationIssue(
                    issue_type="missing_source",
                    severity="medium",
                    concepts_involved=[source_id],
                    description=f"Relationship references source concept ID {source_id} which doesn't exist",
                    recommendation="Remove this relationship or create the missing concept"
                ))
            
            if target_id not in concept_ids:
                warnings.append(ValidationIssue(
                    issue_type="missing_target",
                    severity="medium",
                    concepts_involved=[target_id],
                    description=f"Relationship references target concept ID {target_id} which doesn't exist",
                    recommendation="Remove this relationship or create the missing concept"
                ))
            
            # Check for duplicate relationships
            rel_key = f"{source_id}-{target_id}-{rel_type}"
            if rel_key in relationship_source_targets:
                warnings.append(ValidationIssue(
                    issue_type="duplicate_relationship",
                    severity="low",
                    concepts_involved=[source_id, target_id],
                    description=f"Duplicate relationship from {source_id} to {target_id} of type {rel_type}",
                    recommendation="Remove the duplicate relationship"
                ))
            else:
                relationship_source_targets.add(rel_key)
            
            # Check for self-relationships
            if source_id == target_id:
                issues.append(ValidationIssue(
                    issue_type="self_relationship",
                    severity="high",
                    concepts_involved=[source_id],
                    description=f"Concept {source_id} has a relationship to itself",
                    recommendation="Remove this self-relationship"
                ))
        
        # Check for isolated concepts (no relationships)
        concepts_in_relationships = set()
        for rel in relationships:
            if "source_id" in rel:
                concepts_in_relationships.add(rel["source_id"])
            if "target_id" in rel:
                concepts_in_relationships.add(rel["target_id"])
        
        isolated_concepts = concept_ids - concepts_in_relationships
        if isolated_concepts:
            warnings.append(ValidationIssue(
                issue_type="isolated_concepts",
                severity="medium",
                concepts_involved=list(isolated_concepts),
                description=f"Found {len(isolated_concepts)} concepts with no relationships",
                recommendation="Add relationships to connect these concepts to the graph"
            ))
        
        # Determine if valid based on critical issues
        valid = len(issues) == 0
        
        # Collect stats
        stats = {
            "concepts": len(concepts),
            "relationships": len(relationships),
            "issues": len(issues),
            "warnings": len(warnings),
            "isolated_concepts": len(isolated_concepts)
        }
        
        result = ValidationResult(
            valid=valid,
            issues=issues,
            warnings=warnings,
            stats=stats,
            timestamp=datetime.now()
        )
        
        logger.info(f"Simplified validation complete. Valid: {valid}, Issues: {len(issues)}, Warnings: {len(warnings)}")
        return result
    
    @log_execution_time
    def identify_knowledge_gaps(self, domain: str, existing_structure: List[Dict[str, Any]], 
                              learner_interests: List[str]) -> List[Dict[str, Any]]:
        """Identify knowledge gaps based on learner interests and existing content."""
        if not self.client:
            logger.error("LLM client not available")
            return []
        
        logger.info(f"Identifying knowledge gaps for domain '{domain}' based on {len(learner_interests)} learner interests")
        
        # Prepare simplified structure
        simplified_structure = []
        for item in existing_structure:
            if not isinstance(item, dict) or "id" not in item or "name" not in item:
                continue
                
            simplified_structure.append({
                "id": item["id"],
                "name": item["name"],
                "concept_type": item.get("concept_type", "topic"),
                "description": item.get("description", "")[:100] + "..." if len(item.get("description", "")) > 100 else item.get("description", "")
            })
        
        # Limit the number of concepts for token efficiency
        if len(simplified_structure) > 50:
            logger.info(f"Limiting knowledge gap analysis to 50 concepts (from {len(simplified_structure)})")
            simplified_structure = simplified_structure[:50]
        
        existing_structure_text = json.dumps(simplified_structure, indent=2)
        learner_interests_text = "\n".join(f"- {interest}" for interest in learner_interests[:10])  # Limit to 10 interests
        
        # Format prompt
        prompt_text = self.prompts.knowledge_gap_prompt.format(
            domain=domain,
            existing_structure=existing_structure_text,
            learner_interests=learner_interests_text
        )
        
        # Call LLM
        messages = [
            {"role": "system", "content": "You are an expert educational content analyzer."},
            {"role": "user", "content": prompt_text}
        ]
        
        try:
            json_str = self._call_openai_with_retry(
                messages=messages,
                model="gpt-4-turbo" if self.config.allow_gpt4 else "gpt-3.5-turbo",
                temperature=self.prompts.analysis_temperature,
                max_tokens=self.config.max_tokens_analysis,
                json_mode=True
            )
            
            if not json_str:
                logger.error("Failed to identify knowledge gaps")
                return []
            
            # Parse and validate response
            repaired_json = self._repair_json(json_str)
            data = json.loads(repaired_json)
            
            if "gaps" in data and isinstance(data["gaps"], list):
                # Add metadata
                for gap in data["gaps"]:
                    if "id" not in gap:
                        gap["id"] = str(uuid.uuid4())
                    gap["domain_id"] = domain
                    gap["created_at"] = datetime.now().isoformat()
                
                logger.info(f"Successfully identified {len(data['gaps'])} knowledge gaps")
                return data["gaps"]
            else:
                logger.error("Response valid JSON but missing 'gaps' array")
                return []
        except Exception as e:
            logger.error(f"Error identifying knowledge gaps: {e}")
            logger.debug(traceback.format_exc())
            return []
# Modify the generate_complete_domain method to properly handle learning path generation
# Add this to the generate_complete_domain method before returning the result    
    @log_execution_time
    def generate_complete_domain(self, request: DomainStructureRequest) -> Dict[str, Any]:
        """Generate a complete domain structure including concepts and relationships."""
        if not self.client:
            logger.error("LLM client not available")
            return {"success": False, "error": "LLM client not available"}
        
        try:
            # Start tracking execution time
            start_time = time.time()
            
            # Step 1: Extract key topics if not provided
            key_topics = request.key_topics
            if not key_topics:
                topics_data = self.extract_topics(request.domain_description)
                if topics_data:
                    key_topics = [topic.get("name") for topic in topics_data if "name" in topic]
                    logger.info(f"Extracted {len(key_topics)} topics from description")
            
            # Step 2: Analyze domain to determine concept count and model
            concept_count = request.concept_count
            model = request.model
            
            if not concept_count or not model:
                analysis = self.analyze_domain(
                    request.domain_name,
                    request.domain_description,
                    key_topics
                )
                if not concept_count:
                    concept_count = analysis["concept_count"]
                if not model:
                    model = analysis["model"]
                    
                logger.info(f"Domain analysis recommends {concept_count} concepts using {model}")
            
            # Step 3: Generate concepts
            concepts_data = self.generate_concepts(
                request.domain_name,
                request.domain_description,
                concept_count,
                key_topics,
                model
            )
            
            if not concepts_data or "concepts" not in concepts_data:
                return {"success": False, "error": "Failed to generate concepts"}
            
            concepts = concepts_data["concepts"]
            logger.info(f"Generated {len(concepts)} concepts")
            
            # Step 4: Enrich concepts with educational metadata if requested
            if request.include_content_suggestions:
                logger.info("Enriching concepts with content suggestions")
                concepts = self.enrich_concepts(concepts)
            
            # Step 5: Generate relationships if requested
            relationships = []
            if request.generate_relationships:
                logger.info("Generating relationships between concepts")
                relationships = self.generate_relationships(concepts)
                logger.info(f"Generated {len(relationships)} relationships")
            
            # Step 6: Validate the knowledge graph if both concepts and relationships exist
            validation = None
            if concepts and relationships and len(concepts) > 2 and len(relationships) > 0:
                logger.info("Validating knowledge graph")
                validation = self.validate_knowledge_graph(concepts, relationships)
                validation_dict = validation.model_dump() if validation else None
            else:
                validation_dict = None
            
            # Step 7: Generate sample learning path if requested
            learning_paths = []
            if request.generate_learning_paths and concepts:
                logger.info("Generating sample learning paths")
                # Create a default learning path for domain exploration
                try:
                    sample_path = self.generate_learning_path(
                        goal=f"Explore the fundamentals of {request.domain_name}",
                        concepts=concepts,
                        level="beginner"
                    )
                    if sample_path and "path" in sample_path and sample_path["path"]:
                        learning_paths.append(sample_path)
                        logger.info(f"Generated learning path with {len(sample_path['path'])} steps")
                except Exception as path_error:
                    logger.error(f"Error generating learning path: {path_error}")
                    # Don't fail the entire request if just the learning path fails
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Step 8: Prepare response
            result = {
                "success": True,
                "domain": {
                    "name": request.domain_name,
                    "description": request.domain_description,
                    "key_topics": key_topics,
                    "id": str(uuid.uuid4())  # Generate domain ID if not provided
                },
                "concepts": concepts,
                "relationships": relationships,
                "learning_paths": learning_paths,
                "validation": validation_dict,
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "model_used": model,
                    "concept_count": len(concepts),
                    "relationship_count": len(relationships),
                    "learning_path_count": len(learning_paths),
                    "execution_time_seconds": execution_time,
                    "request_metadata": request.metadata
                }
            }
            
            logger.info(f"Successfully generated complete domain structure for '{request.domain_name}' in {execution_time:.2f} seconds")
            return result
        
        except Exception as e:
            logger.error(f"Error generating domain structure: {e}")
            logger.debug(traceback.format_exc())
            return {
                "success": False,
                "error": f"Error generating domain: {str(e)}"
            }

    def close(self):
        """Clean up resources."""
        if hasattr(self, '_executor') and self._executor:
            self._executor.shutdown(wait=False)
        self._cache.clear()
        logger.info("LLM Service closed")