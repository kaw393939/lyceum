"""
Ptolemy Client
=============
Client for interacting with the Ptolemy knowledge system API.
"""

import json
import logging
import httpx
import uuid
import asyncio
import os
import time
from typing import Dict, List, Optional, Any, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config.settings import get_config
from utils.logging_utils import get_logger

logger = get_logger(__name__)

class PtolemyClient:
    """Client for interacting with the Ptolemy knowledge mapping system.
    
    This client provides access to Ptolemy's concept graph, relationships, and learning paths.
    It supports both direct API communication and a mock mode for development/testing.
    Features include:
    - Fetch concepts, relationships, and learning paths
    - Search the knowledge graph semantically
    - Request learning path generation
    - Local caching for performance optimization
    - Automatic failover to mock data if API is unavailable
    - Batch and parallel operations for efficiency
    """
    
    def __init__(self):
        """Initialize the Ptolemy client with configuration and connection setup.
        
        The client will:
        1. Load configuration settings from environment or config file
        2. Set up caching mechanisms for performance
        3. Determine whether to use live API or mock mode
        4. Establish connection to Ptolemy API or initialize mock data
        """
        # Load configuration
        self.config = get_config()
        ptolemy_config = self.config.get("ptolemy", {})
        
        # Determine operating mode (real or mock) - prioritize environment variable
        self.mock = os.environ.get("PTOLEMY_USE_MOCK", "").lower() in ("true", "1") or ptolemy_config.get("use_mock", False)
        
        # API URL handling - prioritize environment variable
        self.api_url = os.environ.get("PTOLEMY_API_URL", ptolemy_config.get("api_url", "http://ptolemy:8000"))
        logger.debug(f"Using Ptolemy API URL: {self.api_url}")
        
        # API key handling - fall back to mock if no key available
        self.api_key = os.environ.get("PTOLEMY_API_KEY") or os.environ.get("PTOLEMY_BEARER_TOKEN")
        api_key_config = ptolemy_config.get("api_key") or ptolemy_config.get("bearer_token")
        
        if not self.api_key:
            self.api_key = api_key_config
            
        if not self.api_key:
            logger.warning("No Ptolemy API key found in environment or config, falling back to mock mode")
            self.mock = True
        
        # Initialize results cache
        self._cache = {}
        self._cache_ttl = ptolemy_config.get("cache_ttl", 300)  # Default: 5 minutes
        self._max_cache_size = ptolemy_config.get("max_cache_size", 1000)
        
        # Connection settings
        self.max_retries = ptolemy_config.get("retry_count", 3)
        self.retry_delay = ptolemy_config.get("retry_delay", 1.0)
        self.max_parallel_requests = ptolemy_config.get("parallel_requests", 5)
        self.batch_size = ptolemy_config.get("max_batch_size", 20)
        
        # Establish connection or setup mock data
        if not self.mock:
            self._init_api_connection()
            logger.info("PtolemyClient initialized with API connection")
        else:
            # Initialize mock data structures
            self.mock_concepts = self._initialize_mock_concepts()
            self.mock_relationships = self._initialize_mock_relationships()
            self.mock_learning_paths = self._initialize_mock_learning_paths()
            logger.info("PtolemyClient initialized in mock mode with synthetic data")
            
    def _init_api_connection(self):
        """Initialize the connection to the Ptolemy API service.
        
        This method:
        1. Sets up API connection parameters from environment or config
        2. Configures API endpoints for different operations
        3. Handles Docker networking and service discovery
        4. Falls back to mock mode on initialization errors
        """
        try:
            # Get connection parameters (prioritize environment variables)
            ptolemy_config = self.config.get("ptolemy", {})
            
            # Base URL - typically the service hostname in Docker network
            self.base_url = os.environ.get("PTOLEMY_API_URL", 
                            ptolemy_config.get("api_url", "http://ptolemy:8000"))
            
            # API authentication
            self.api_key = os.environ.get("PTOLEMY_API_KEY", 
                           ptolemy_config.get("api_key", ""))
            
            # Request parameters
            self.timeout = ptolemy_config.get("timeout", 30.0)
            
            # Define API endpoint paths
            self.endpoints = {
                "concepts": "/api/v1/concepts",
                "relationships": "/api/v1/relationships",
                "learning_paths": "/api/v1/learning_paths",
                "concept_graph": "/api/v1/concept_graph",
                "search": "/api/v1/search",
                "health": "/health",
                "domain": "/api/v1/domains",
                "batch": "/api/v1/batch"
            }
            
            # Validate connection settings
            if not self.api_key:
                logger.warning("No Ptolemy API key provided - authentication may fail")
            
            # Configure headers for API requests
            self.headers = {
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            
            # Add authorization if we have an API key
            if self.api_key:
                self.headers["Authorization"] = f"Bearer {self.api_key}"
            
            # Special handling for Docker networking
            if "ptolemy" in self.base_url or "localhost" not in self.base_url:
                logger.info(f"Using service discovery for Ptolemy at: {self.base_url}")
                
            # Set up the HTTP client with connection pooling
            self.client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=self.headers,
                timeout=self.timeout,
                follow_redirects=True,
                limits=httpx.Limits(
                    max_keepalive_connections=20,
                    max_connections=100
                )
            )
            
            logger.info(f"Ptolemy API client initialized with base URL: {self.base_url}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Ptolemy API client: {str(e)}")
            logger.debug(f"Initialization error details: {e.__class__.__name__}")
            
            # Fall back to mock mode
            self.mock = True
            
            # Initialize mock data structures
            self.mock_concepts = self._initialize_mock_concepts()
            self.mock_relationships = self._initialize_mock_relationships()
            self.mock_learning_paths = self._initialize_mock_learning_paths()
            
            logger.warning("Falling back to mock mode due to API initialization error")
    
    def _initialize_mock_concepts(self) -> Dict[str, Dict[str, Any]]:
        """Initialize mock concept data."""
        concepts = {}
        
        # Stoicism concepts
        concepts["stoicism"] = {
            "id": "stoicism",
            "name": "Stoicism",
            "description": "An ancient Greek philosophical school founded by Zeno of Citium focused on virtue, reason, and living in accordance with nature.",
            "category": "philosophy",
            "properties": {
                "origin": "Ancient Greece",
                "time_period": "3rd century BCE"
            }
        }
        
        concepts["dichotomy_of_control"] = {
            "id": "dichotomy_of_control",
            "name": "Dichotomy of Control",
            "description": "The Stoic principle that distinguishes between what we can control (our judgments, actions) and what we cannot control (external events).",
            "category": "philosophy",
            "properties": {
                "attribution": "Epictetus",
                "difficulty": "intermediate"
            }
        }
        
        concepts["virtue_ethics"] = {
            "id": "virtue_ethics",
            "name": "Virtue Ethics",
            "description": "A philosophical approach that emphasizes the development of character and virtues as the key to ethical behavior.",
            "category": "ethics",
            "properties": {
                "key_virtues": ["wisdom", "courage", "justice", "temperance"],
                "difficulty": "intermediate"
            }
        }
        
        concepts["stoic_virtues"] = {
            "id": "stoic_virtues",
            "name": "Stoic Virtues",
            "description": "The four cardinal virtues in Stoicism are wisdom (sophia), courage (andreia), justice (dikaiosyne), and temperance (sophrosyne).",
            "category": "philosophy",
            "properties": {
                "attribution": "Zeno, Epictetus, Marcus Aurelius",
                "difficulty": "intermediate",
                "key_components": ["wisdom", "courage", "justice", "temperance"]
            }
        }
        
        concepts["negative_visualization"] = {
            "id": "negative_visualization",
            "name": "Negative Visualization",
            "description": "A Stoic practice of contemplating potential adversities to develop gratitude and resilience.",
            "category": "philosophy",
            "properties": {
                "original_term": "premeditatio malorum",
                "difficulty": "beginner",
                "attributed_to": "Seneca"
            }
        }
        
        concepts["view_from_above"] = {
            "id": "view_from_above",
            "name": "View From Above",
            "description": "A Stoic meditation technique where one imagines viewing oneself and human affairs from a cosmic perspective.",
            "category": "philosophy",
            "properties": {
                "difficulty": "intermediate",
                "attributed_to": "Marcus Aurelius"
            }
        }
        
        # Mathematics concepts
        concepts["algebra"] = {
            "id": "algebra",
            "name": "Algebra",
            "description": "The branch of mathematics dealing with symbols and the rules for manipulating these symbols to solve equations.",
            "category": "mathematics",
            "properties": {
                "difficulty": "intermediate",
                "grade_level": "middle school to high school"
            }
        }
        
        concepts["calculus"] = {
            "id": "calculus",
            "name": "Calculus",
            "description": "The mathematical study of continuous change, dealing with limits, derivatives, integrals, and infinite series.",
            "category": "mathematics",
            "properties": {
                "difficulty": "advanced",
                "grade_level": "high school to college"
            }
        }
        
        # Computer Science concepts
        concepts["algorithms"] = {
            "id": "algorithms",
            "name": "Algorithms",
            "description": "Step-by-step procedures for solving problems or accomplishing tasks, forming the foundation of programming and computer science.",
            "category": "computer_science",
            "properties": {
                "difficulty": "intermediate",
                "types": ["sorting", "searching", "graph", "dynamic programming"]
            }
        }
        
        concepts["data_structures"] = {
            "id": "data_structures",
            "name": "Data Structures",
            "description": "Specialized formats for organizing, processing, retrieving and storing data efficiently.",
            "category": "computer_science",
            "properties": {
                "difficulty": "intermediate",
                "types": ["arrays", "linked lists", "trees", "graphs", "hash tables"]
            }
        }
        
        return concepts
    
    def _initialize_mock_relationships(self) -> List[Dict[str, Any]]:
        """Initialize mock relationship data."""
        relationships = [
            {
                "id": "rel_001",
                "source_id": "stoicism",
                "target_id": "dichotomy_of_control",
                "relationship_type": "includes",
                "description": "Stoicism includes the principle of dichotomy of control",
                "strength": 0.9
            },
            {
                "id": "rel_002",
                "source_id": "stoicism",
                "target_id": "virtue_ethics",
                "relationship_type": "is_type_of",
                "description": "Stoicism is a form of virtue ethics",
                "strength": 0.8
            },
            {
                "id": "rel_003",
                "source_id": "stoicism",
                "target_id": "stoic_virtues",
                "relationship_type": "includes",
                "description": "Stoicism includes the four cardinal virtues",
                "strength": 0.9
            },
            {
                "id": "rel_004",
                "source_id": "stoicism",
                "target_id": "negative_visualization",
                "relationship_type": "includes",
                "description": "Negative visualization is a Stoic practice",
                "strength": 0.8
            },
            {
                "id": "rel_005",
                "source_id": "stoicism",
                "target_id": "view_from_above",
                "relationship_type": "includes",
                "description": "View from above is a Stoic meditation technique",
                "strength": 0.7
            },
            {
                "id": "rel_006",
                "source_id": "virtue_ethics",
                "target_id": "stoic_virtues",
                "relationship_type": "includes",
                "description": "Virtue ethics encompasses the Stoic virtues",
                "strength": 0.9
            },
            {
                "id": "rel_007",
                "source_id": "algebra",
                "target_id": "calculus",
                "relationship_type": "prerequisite",
                "description": "Algebra is a prerequisite for learning calculus",
                "strength": 0.9
            },
            {
                "id": "rel_008",
                "source_id": "algorithms",
                "target_id": "data_structures",
                "relationship_type": "related",
                "description": "Algorithms and data structures are closely related concepts",
                "strength": 0.9
            }
        ]
        return relationships
    
    def _initialize_mock_learning_paths(self) -> Dict[str, Dict[str, Any]]:
        """Initialize mock learning paths."""
        paths = {}
        
        # Stoicism learning path
        paths["stoicism_intro"] = {
            "id": "stoicism_intro",
            "name": "Introduction to Stoicism",
            "description": "A beginner-friendly introduction to Stoic philosophy and its practical applications.",
            "goal": "Understand the core principles of Stoicism and apply them to daily life",
            "target_learner_level": "beginner",
            "estimated_time_minutes": 180,
            "steps": [
                {
                    "order": 1,
                    "concept_id": "stoicism",
                    "name": "Foundations of Stoicism",
                    "reason": "Building a solid foundation of Stoic history and principles",
                    "estimated_time_minutes": 60
                },
                {
                    "order": 2,
                    "concept_id": "dichotomy_of_control",
                    "name": "Understanding the Dichotomy of Control",
                    "reason": "This principle is fundamental to Stoic practice",
                    "estimated_time_minutes": 45
                },
                {
                    "order": 3,
                    "concept_id": "stoic_virtues",
                    "name": "The Four Stoic Virtues",
                    "reason": "Understanding the core virtues that guide Stoic ethics",
                    "estimated_time_minutes": 45
                },
                {
                    "order": 4,
                    "concept_id": "negative_visualization",
                    "name": "Practicing Negative Visualization",
                    "reason": "Learning a key Stoic practice for resilience and gratitude",
                    "estimated_time_minutes": 30
                }
            ]
        }
        
        paths["stoic_practices"] = {
            "id": "stoic_practices",
            "name": "Stoic Meditation Practices",
            "description": "A guide to practical Stoic meditation and reflection techniques.",
            "goal": "Develop a daily Stoic practice routine",
            "target_learner_level": "intermediate",
            "estimated_time_minutes": 120,
            "steps": [
                {
                    "order": 1,
                    "concept_id": "negative_visualization",
                    "name": "Mastering Negative Visualization",
                    "reason": "Developing a structured practice for contemplating adversity",
                    "estimated_time_minutes": 45
                },
                {
                    "order": 2,
                    "concept_id": "view_from_above",
                    "name": "The View From Above",
                    "reason": "Practicing cosmic perspective to reduce attachment",
                    "estimated_time_minutes": 45
                },
                {
                    "order": 3,
                    "concept_id": "dichotomy_of_control",
                    "name": "Daily Control Inventory",
                    "reason": "Applying the dichotomy of control to everyday situations",
                    "estimated_time_minutes": 30
                }
            ]
        }
        
        # Computer Science learning path
        paths["cs_fundamentals"] = {
            "id": "cs_fundamentals",
            "name": "Computer Science Fundamentals",
            "description": "An introduction to the foundational concepts of computer science.",
            "goal": "Understand the basic building blocks of computer science",
            "target_learner_level": "beginner",
            "estimated_time_minutes": 240,
            "steps": [
                {
                    "order": 1,
                    "concept_id": "algorithms",
                    "name": "Introduction to Algorithms",
                    "reason": "Algorithms are the foundation of problem-solving in CS",
                    "estimated_time_minutes": 120
                },
                {
                    "order": 2,
                    "concept_id": "data_structures",
                    "name": "Essential Data Structures",
                    "reason": "Understanding how to organize and store data efficiently",
                    "estimated_time_minutes": 120
                }
            ]
        }
        
        return paths
    
    @retry(
        stop=stop_after_attempt(3), 
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry_error_callback=lambda x: logger.error(f"Request failed after retries: {x}")
    )
    async def _make_request(self, method: str, endpoint: str, params: Dict = None, 
                         json_data: Dict = None) -> Dict[str, Any]:
        """
        Make a request to the Ptolemy API.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            params: Query parameters
            json_data: JSON request body
            
        Returns:
            Response data
        """
        url = f"{self.base_url}{endpoint}"
        
        # Set up headers with API key if available
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        if hasattr(self, 'api_key') and self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        try:
            # Log the request details at debug level
            logger.debug(f"Making request to Ptolemy API: {method} {url}")
            if params:
                logger.debug(f"Request params: {params}")
            if json_data:
                logger.debug(f"Request data: {json_data}")
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.request(
                    method=method,
                    url=url,
                    params=params,
                    json=json_data,
                    headers=headers
                )
                
                if response.status_code >= 400:
                    logger.error(f"HTTP error from Ptolemy API: {response.status_code} - {response.text}")
                    if response.status_code == 404:
                        return None
                
                response.raise_for_status()
                result = response.json()
                logger.debug(f"Ptolemy API response success: {method} {url}")
                return result
                
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error from Ptolemy API: {e.response.status_code} - {e.response.text}")
            if e.response.status_code == 404:
                return None
            raise
            
        except httpx.TimeoutException:
            logger.error(f"Timeout error from Ptolemy API: {url}")
            raise
            
        except Exception as e:
            logger.error(f"Error communicating with Ptolemy API: {str(e)}")
            raise
    
    # Cache utility methods
    def _add_to_cache(self, key: str, value: Any) -> None:
        """Add a value to the cache with expiration time."""
        # If cache is at max size, remove oldest entry
        if len(self._cache) >= self._max_cache_size:
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k]['timestamp'])
            del self._cache[oldest_key]
            
        self._cache[key] = {
            'value': value,
            'timestamp': time.time(),
            'expires': time.time() + self._cache_ttl
        }
    
    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get a value from the cache if it exists and is not expired."""
        if key not in self._cache:
            return None
            
        cache_entry = self._cache[key]
        # Check if entry is expired
        if cache_entry['expires'] < time.time():
            del self._cache[key]
            return None
            
        return cache_entry['value']
    
    def _clear_cache(self) -> None:
        """Clear the entire cache."""
        self._cache = {}
    
    async def get_concept(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """
        Get concept data by ID, with caching.
        
        Args:
            concept_id: ID of the concept
            
        Returns:
            Concept data or None if not found
        """
        # Check cache first
        cache_key = f"concept:{concept_id}"
        cached_value = self._get_from_cache(cache_key)
        if cached_value is not None:
            return cached_value
        
        # Not in cache, fetch from source
        if self.mock:
            # Simulate network latency for more realistic behavior
            await asyncio.sleep(0.1)
            result = self.mock_concepts.get(concept_id)
        else:
            endpoint = f"{self.endpoints['concepts']}/{concept_id}"
            result = await self._make_request("GET", endpoint)
        
        # Cache result if it exists
        if result:
            self._add_to_cache(cache_key, result)
        
        return result
    
    async def get_concept_with_relationships(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """
        Get concept data with its relationships.
        
        Args:
            concept_id: ID of the concept
            
        Returns:
            Dictionary with concept and relationships data
        """
        if self.mock:
            # Simulate network latency
            await asyncio.sleep(0.1)
            
            concept = self.mock_concepts.get(concept_id)
            if not concept:
                return None
            
            relationships = []
            for rel in self.mock_relationships:
                if rel["source_id"] == concept_id or rel["target_id"] == concept_id:
                    relationships.append(rel)
            
            return {
                "concept": concept,
                "relationships": relationships
            }
        else:
            endpoint = f"{self.endpoints['concepts']}/{concept_id}/with_relationships"
            return await self._make_request("GET", endpoint)
    
    async def get_concepts_batch(self, concept_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Get multiple concepts by ID using efficient batch operations.
        
        Args:
            concept_ids: List of concept IDs
            
        Returns:
            List of concept data
        """
        if not concept_ids:
            return []
            
        if self.mock:
            # Simulate network latency
            await asyncio.sleep(0.2)
            return [self.mock_concepts.get(cid) for cid in concept_ids if cid in self.mock_concepts]
        else:
            # Use bulk operation if available
            try:
                # Check if bulk endpoint exists
                endpoint = f"{self.endpoints['concepts']}/bulk"
                batch_size = self.config.get("ptolemy", {}).get("max_batch_size", 20)
                
                # Process in batches to avoid too large requests
                all_results = []
                for i in range(0, len(concept_ids), batch_size):
                    batch = concept_ids[i:i+batch_size]
                    try:
                        result = await self._make_request("POST", endpoint, json_data={"ids": batch})
                        if result and "concepts" in result:
                            all_results.extend(result["concepts"])
                    except Exception as e:
                        logger.warning(f"Batch request failed, falling back to individual requests: {e}")
                        # Fall back to individual requests for this batch
                        for concept_id in batch:
                            concept = await self.get_concept(concept_id)
                            if concept:
                                all_results.append(concept)
                
                return all_results
            except Exception as e:
                # Fall back to individual requests
                logger.warning(f"Batch operations not supported, using individual requests: {e}")
                results = []
                # Use gather for parallel requests with concurrency limit
                semaphore = asyncio.Semaphore(5)  # Limit concurrent requests
                
                async def get_with_semaphore(cid):
                    async with semaphore:
                        return await self.get_concept(cid)
                
                tasks = [get_with_semaphore(cid) for cid in concept_ids]
                for concept in await asyncio.gather(*tasks):
                    if concept:
                        results.append(concept)
                
                return results
    
    async def search_concepts(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for concepts.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of matching concepts
        """
        if self.mock:
            # Simulate network latency
            await asyncio.sleep(0.2)
            
            # Simple search implementation that checks if query is in name or description
            results = []
            query = query.lower()
            for concept_id, concept in self.mock_concepts.items():
                # Check for match in name, description, or category
                name_match = query in concept.get("name", "").lower()
                desc_match = query in concept.get("description", "").lower()
                category_match = query in concept.get("category", "").lower()
                
                # Also check properties for matches
                prop_match = False
                if "properties" in concept:
                    for prop_value in concept["properties"].values():
                        if isinstance(prop_value, str) and query in prop_value.lower():
                            prop_match = True
                            break
                        elif isinstance(prop_value, list):
                            for item in prop_value:
                                if isinstance(item, str) and query in item.lower():
                                    prop_match = True
                                    break
                
                if name_match or desc_match or category_match or prop_match:
                    # Add a relevance score based on match quality
                    match_score = 0
                    if name_match:
                        match_score += 1.0  # Highest priority for name matches
                    if desc_match:
                        match_score += 0.5  # Medium priority for description matches
                    if category_match:
                        match_score += 0.3  # Lower priority for category matches
                    if prop_match:
                        match_score += 0.2  # Lowest priority for property matches
                    
                    results.append({
                        **concept,
                        "relevance_score": match_score
                    })
            
            # Sort by relevance score and limit results
            results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
            return results[:limit]
        else:
            endpoint = f"{self.endpoints['search']}/concepts"
            params = {"query": query, "limit": limit}
            
            result = await self._make_request("GET", endpoint, params=params)
            return result.get("results", []) if result else []
    
    async def get_concept_relationships(self, concept_id: str, 
                                     relationship_types: List[str] = None) -> List[Dict[str, Any]]:
        """
        Get relationships for a concept.
        
        Args:
            concept_id: ID of the concept
            relationship_types: Optional list of relationship types to filter by
            
        Returns:
            List of relationships
        """
        if self.mock:
            # Simulate network latency
            await asyncio.sleep(0.1)
            
            if relationship_types:
                return [
                    rel for rel in self.mock_relationships
                    if (rel["source_id"] == concept_id or rel["target_id"] == concept_id) and
                    rel["relationship_type"] in relationship_types
                ]
            else:
                return [
                    rel for rel in self.mock_relationships
                    if rel["source_id"] == concept_id or rel["target_id"] == concept_id
                ]
        else:
            endpoint = f"{self.endpoints['concepts']}/{concept_id}/relationships"
            params = {}
            
            if relationship_types:
                params["types"] = ",".join(relationship_types)
            
            result = await self._make_request("GET", endpoint, params=params)
            return result.get("relationships", []) if result else []
    
    async def get_learning_path(self, path_id: str) -> Optional[Dict[str, Any]]:
        """
        Get learning path data.
        
        Args:
            path_id: ID of the learning path
            
        Returns:
            Learning path data or None if not found
        """
        if self.mock:
            # Simulate network latency
            await asyncio.sleep(0.1)
            return self.mock_learning_paths.get(path_id)
        else:
            endpoint = f"{self.endpoints['learning_paths']}/{path_id}"
            return await self._make_request("GET", endpoint)
    
    async def get_learning_paths_for_concept(self, concept_id: str) -> List[Dict[str, Any]]:
        """
        Get learning paths that include a concept.
        
        Args:
            concept_id: ID of the concept
            
        Returns:
            List of learning paths
        """
        if self.mock:
            # Simulate network latency
            await asyncio.sleep(0.2)
            
            paths = []
            for path_id, path in self.mock_learning_paths.items():
                if any(step.get("concept_id") == concept_id for step in path.get("steps", [])):
                    # Find the step with this concept
                    for step in path.get("steps", []):
                        if step.get("concept_id") == concept_id:
                            # Add step information to the path
                            path_copy = path.copy()
                            path_copy["relevant_step"] = step
                            paths.append(path_copy)
                            break
            
            return paths
        else:
            endpoint = f"{self.endpoints['concepts']}/{concept_id}/learning_paths"
            
            result = await self._make_request("GET", endpoint)
            return result.get("learning_paths", []) if result else []
    
    async def get_concept_graph(self, concept_id: str, depth: int = 1, 
                              relationship_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get concept graph starting from a central concept.
        
        Args:
            concept_id: ID of the central concept
            depth: Traversal depth
            relationship_types: Optional filter for relationship types
            
        Returns:
            Dictionary with nodes and edges
        """
        if self.mock:
            # Simulate network latency
            await asyncio.sleep(0.3)
            
            # Start with the central concept
            nodes = []
            edges = []
            visited = set()
            
            # Get the central concept
            central_concept = self.mock_concepts.get(concept_id)
            if not central_concept:
                return {"nodes": [], "edges": []}
            
            nodes.append(central_concept)
            visited.add(concept_id)
            
            # Helper function to recursively build the graph
            def expand_node(node_id: str, current_depth: int):
                if current_depth > depth:
                    return
                
                # Get relationships for this node
                for rel in self.mock_relationships:
                    # Check if relationship type matches filter
                    if relationship_types and rel["relationship_type"] not in relationship_types:
                        continue
                    
                    # Process outgoing relationships
                    if rel["source_id"] == node_id and rel["target_id"] not in visited:
                        target_concept = self.mock_concepts.get(rel["target_id"])
                        if target_concept:
                            nodes.append(target_concept)
                            edges.append(rel)
                            visited.add(rel["target_id"])
                            # Recursive call
                            expand_node(rel["target_id"], current_depth + 1)
                    
                    # Process incoming relationships
                    elif rel["target_id"] == node_id and rel["source_id"] not in visited:
                        source_concept = self.mock_concepts.get(rel["source_id"])
                        if source_concept:
                            nodes.append(source_concept)
                            edges.append(rel)
                            visited.add(rel["source_id"])
                            # Recursive call
                            expand_node(rel["source_id"], current_depth + 1)
            
            # Start expanding from the central concept
            expand_node(concept_id, 1)
            
            return {
                "nodes": nodes,
                "edges": edges
            }
        else:
            endpoint = f"{self.endpoints['concept_graph']}/{concept_id}"
            params = {"depth": depth}
            
            if relationship_types:
                params["relationship_types"] = ",".join(relationship_types)
            
            return await self._make_request("GET", endpoint, params=params)
    
    async def create_concept(self, concept_data: Dict[str, Any]) -> str:
        """
        Create a new concept.
        
        Args:
            concept_data: Concept data
            
        Returns:
            Concept ID
        """
        if self.mock:
            # Simulate network latency
            await asyncio.sleep(0.2)
            
            concept_id = concept_data.get("id", str(uuid.uuid4()))
            concept_data["id"] = concept_id
            self.mock_concepts[concept_id] = concept_data
            logger.info(f"Created concept with ID: {concept_id} (mock)")
            return concept_id
        else:
            endpoint = self.endpoints['concepts']
            result = await self._make_request("POST", endpoint, json_data=concept_data)
            return result.get("id") if result else None
    
    async def update_concept(self, concept_id: str, update_data: Dict[str, Any]) -> bool:
        """
        Update concept data.
        
        Args:
            concept_id: Concept ID
            update_data: Data to update
            
        Returns:
            True if successful
        """
        if self.mock:
            # Simulate network latency
            await asyncio.sleep(0.2)
            
            if concept_id not in self.mock_concepts:
                return False
            
            self.mock_concepts[concept_id].update(update_data)
            logger.info(f"Updated concept with ID: {concept_id} (mock)")
            return True
        else:
            endpoint = f"{self.endpoints['concepts']}/{concept_id}"
            result = await self._make_request("PUT", endpoint, json_data=update_data)
            return result is not None
    
    async def create_relationship(self, relationship_data: Dict[str, Any]) -> str:
        """
        Create a new relationship between concepts.
        
        Args:
            relationship_data: Relationship data
            
        Returns:
            Relationship ID if successful, None otherwise
        """
        if self.mock:
            # Simulate network latency
            await asyncio.sleep(0.2)
            
            # Check if source and target concepts exist
            source_id = relationship_data.get("source_id")
            target_id = relationship_data.get("target_id")
            if not source_id or not target_id:
                return None
            if source_id not in self.mock_concepts or target_id not in self.mock_concepts:
                return None
            
            # Generate ID if not provided
            relationship_id = relationship_data.get("id", f"rel_{str(uuid.uuid4())[:8]}")
            relationship_data["id"] = relationship_id
            
            self.mock_relationships.append(relationship_data)
            logger.info(f"Created relationship with ID: {relationship_id} between {source_id} and {target_id} (mock)")
            return relationship_id
        else:
            endpoint = self.endpoints['relationships']
            result = await self._make_request("POST", endpoint, json_data=relationship_data)
            return result.get("id") if result else None
    
    async def get_related_concepts(self, concept_id: str, 
                                 relationship_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get concepts related to a concept.
        
        Args:
            concept_id: ID of the concept
            relationship_type: Optional relationship type to filter by
            
        Returns:
            List of related concepts with relationship information
        """
        if self.mock:
            # Simulate network latency
            await asyncio.sleep(0.2)
            
            related_concepts = []
            
            # Get all relationships for this concept
            for rel in self.mock_relationships:
                # Apply relationship type filter if provided
                if relationship_type and rel["relationship_type"] != relationship_type:
                    continue
                
                # Process outgoing relationships
                if rel["source_id"] == concept_id:
                    target_concept = self.mock_concepts.get(rel["target_id"])
                    if target_concept:
                        related_concepts.append({
                            "concept": target_concept,
                            "relationship": rel,
                            "direction": "outgoing"
                        })
                
                # Process incoming relationships
                elif rel["target_id"] == concept_id:
                    source_concept = self.mock_concepts.get(rel["source_id"])
                    if source_concept:
                        related_concepts.append({
                            "concept": source_concept,
                            "relationship": rel,
                            "direction": "incoming"
                        })
            
            return related_concepts
        else:
            endpoint = f"{self.endpoints['concepts']}/{concept_id}/related"
            params = {}
            
            if relationship_type:
                params["relationship_type"] = relationship_type
            
            result = await self._make_request("GET", endpoint, params=params)
            return result.get("related_concepts", []) if result else []
    
    async def list_learning_paths(self, limit: int = 10, offset: int = 0, 
                                filters: Dict[str, Any] = None) -> Tuple[List[Dict[str, Any]], int]:
        """
        List learning paths.
        
        Args:
            limit: Maximum number of items
            offset: Number of items to skip
            filters: Optional filters
            
        Returns:
            Tuple of (paths, total count)
        """
        if self.mock:
            # Simulate network latency
            await asyncio.sleep(0.2)
            
            paths = list(self.mock_learning_paths.values())
            
            # Apply filters if provided
            if filters:
                filtered_paths = []
                for path in paths:
                    matches = True
                    for key, value in filters.items():
                        if path.get(key) != value:
                            matches = False
                            break
                    
                    if matches:
                        filtered_paths.append(path)
                
                paths = filtered_paths
            
            # Sort by name
            paths.sort(key=lambda x: x.get("name", ""))
            
            # Apply pagination
            total = len(paths)
            paginated_paths = paths[offset:offset + limit]
            
            return paginated_paths, total
        else:
            endpoint = self.endpoints['learning_paths']
            params = {
                "limit": limit,
                "offset": offset
            }
            
            if filters:
                params.update(filters)
            
            result = await self._make_request("GET", endpoint, params=params)
            if result:
                return result.get("paths", []), result.get("total", 0)
            return [], 0