"""
Gutenberg Content Generation System - Vector Store Service
========================================================
Provides access to vector-based retrieval of content for RAG implementation.
"""

import logging
import time
import traceback
from typing import Dict, List, Optional, Any, Union
import json
import asyncio
import numpy as np
from datetime import datetime
import os
from tenacity import retry, stop_after_attempt, wait_exponential

# Import Qdrant libraries with graceful degradation
try:
    from qdrant_client import QdrantClient, models
    from qdrant_client.http import models as rest_models
    from qdrant_client.http.exceptions import UnexpectedResponse
    QDRANT_AVAILABLE = True
except ImportError:
    print("Warning: Qdrant client library not installed. Vector store functionality will be limited.")
    QDRANT_AVAILABLE = False
    # Create placeholder classes for type checking
    class QdrantClient:
        pass
    class models:
        class VectorParams:
            pass
        class Distance:
            COSINE = "cosine"
        class Filter:
            pass
        class FieldCondition:
            pass
        class MatchAny:
            pass
        class MatchValue:
            pass
        class PointStruct:
            pass
        class PointIdsList:
            pass
    class UnexpectedResponse(Exception):
        pass

# Import OpenAI with graceful degradation
try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    print("Warning: OpenAI library not installed. Embedding functionality will be limited.")
    OPENAI_AVAILABLE = False
    # Create placeholder class for type checking
    class AsyncOpenAI:
        pass

from config.settings import get_config
from utils.logging_utils import get_logger

logger = get_logger(__name__)

class VectorStore:
    """
    Integration with Qdrant vector database for semantic search and retrieval.
    Supports knowledge retrieval for RAG applications with OpenAI embeddings.
    """
    
    def __init__(self):
        """Initialize the vector store client."""
        self.config = get_config()
        
        # Get configuration from environment or config file
        self.collection_name = self.config.get("vector_store", {}).get("collection_name", "gutenberg_content")
        self.embedding_dim = self.config.get("vector_store", {}).get("embedding_dim", 1536)  # OpenAI embedding dimension
        
        # Check for mock mode
        self.mock_mode = self.config.get("vector_store", {}).get("mock_mode", False)
        self.connected = False  # Default to not connected
        
        # Initialize OpenAI client for embeddings
        self.openai_client = AsyncOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", self.config.get("openai", {}).get("api_key", "")),
        )
        self.embedding_model = self.config.get("vector_store", {}).get("embedding_model", "text-embedding-3-small")
        
        # If in mock mode, set connected to True
        if self.mock_mode:
            logger.warning("Qdrant in mock mode. Using mock responses for vector store operations.")
            self.connected = True
        else:
            # Initialize Qdrant client
            self._initialize_connection()
        
        # Cache for recently accessed vectors to reduce API calls
        self._cache = {}
        
        logger.info(f"VectorStore initialized with collection '{self.collection_name}'")
    
    def _initialize_connection(self):
        """Initialize connection to the Qdrant vector store.
        
        This method connects to the Qdrant service using configuration parameters,
        verifies the connection, and sets up the required collection if needed.
        
        If the Qdrant client library is not available, the service will run in
        mock mode without actual vector storage capabilities.
        
        The method handles connection errors gracefully with appropriate logging.
        """
        # Check if mock mode is enabled in config
        mock_mode = self.config.get("vector_store", {}).get("mock_mode", False)
        if not QDRANT_AVAILABLE or mock_mode:
            logger.warning("Qdrant in mock mode. Using mock responses for vector store operations.")
            self.connected = True  # Pretend to be connected in mock mode
            return
            
        try:
            # Get configuration values with defaults, prioritizing environment variables
            self.qdrant_url = os.environ.get("QDRANT_URL", 
                            self.config.get("vector_store", {}).get("url", "http://qdrant:6333"))
            
            # Port can be part of URL in the environment variable, so only use the default when needed
            if ":" in self.qdrant_url and not self.qdrant_url.endswith(":"):
                self.qdrant_port = None  # Port is included in the URL
            else:
                self.qdrant_port = self.config.get("vector_store", {}).get("port", 6333)
                
            connection_timeout = self.config.get("vector_store", {}).get("connection_timeout", 10.0)
            
            logger.info(f"Connecting to Qdrant at {self.qdrant_url}")
            
            # Connect to Qdrant with appropriate options
            self.client = QdrantClient(
                url=self.qdrant_url,
                port=self.qdrant_port,
                prefer_grpc=True,
                timeout=connection_timeout
            )
            
            # Test connection before proceeding
            self.client.get_collections()
            
            # Set up the necessary collection
            self._ensure_collection_exists()
            
            # Mark connection as successful
            self.connected = True
            logger.info(f"Successfully connected to Qdrant at {self.qdrant_url}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {str(e)}")
            self.connected = False
            
            # Provide more detailed debug information
            if hasattr(e, "__traceback__"):
                logger.debug(f"Connection error details: {traceback.format_exc()}")
                
            # Implement reconnection logic
            reconnect_attempts = self.config.get("vector_store", {}).get("reconnect_attempts", 3)
            poll_interval = self.config.get("vector_store", {}).get("poll_interval", 0.1)
            
            if reconnect_attempts > 0:
                logger.info(f"Will attempt to reconnect {reconnect_attempts} times")
                # The reconnection will be handled by the retry decorator on actual operation calls
    
    def _ensure_collection_exists(self):
        """Ensure that the Qdrant vector collection exists, creating it if needed.
        
        This method:
        1. Checks if the configured collection already exists in Qdrant
        2. Creates the collection with proper vector configuration if needed
        3. Sets up appropriate payload indexes for efficient filtering
        4. Configures service namespacing for shared collections between Ptolemy and Gutenberg
        
        Raises:
            Exception: If there's an error creating the collection or indexes
        """
        try:
            # Check existing collections
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            # Create collection if it doesn't exist
            if self.collection_name not in collection_names:
                logger.info(f"Creating vector collection '{self.collection_name}'")
                
                # Configure indexing optimization
                threshold = self.config.get("vector_store", {}).get("indexing_threshold", 20000)
                optimizers_config = models.OptimizersConfigDiff(indexing_threshold=threshold)
                
                # Create the collection with proper vector configuration
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.embedding_dim,
                        distance=models.Distance.COSINE
                    ),
                    optimizers_config=optimizers_config
                )
                
                # Set up indexes
                self._create_payload_indexes()
                
                logger.info(f"Vector collection '{self.collection_name}' created successfully")
            else:
                # Collection exists, check compatibility
                logger.info(f"Vector collection '{self.collection_name}' already exists")
                
                # Verify collection parameters (optional enhancement)
                try:
                    collection_info = self.client.get_collection(self.collection_name)
                    actual_dim = collection_info.config.params.vectors.size
                    
                    if actual_dim != self.embedding_dim:
                        logger.warning(
                            f"Collection dimension mismatch: found {actual_dim}, expected {self.embedding_dim}. "
                            f"Using existing collection, but this may cause issues."
                        )
                except Exception as e:
                    logger.warning(f"Could not verify collection parameters: {e}")
                
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {str(e)}")
            if hasattr(e, "__traceback__"):
                logger.debug(f"Collection creation error details: {traceback.format_exc()}")
            raise
    
    def _create_payload_indexes(self):
        """Create payload indexes for efficient filtering in the vector collection.
        
        Sets up keyword and text indexes for common query fields to improve 
        search performance and filtering capabilities.
        """
        # Standard indexes for content metadata
        index_fields = [
            {"name": "category", "type": models.PayloadSchemaType.KEYWORD},
            {"name": "subcategory", "type": models.PayloadSchemaType.KEYWORD},
            {"name": "content_type", "type": models.PayloadSchemaType.KEYWORD},
            {"name": "title", "type": models.PayloadSchemaType.TEXT},
            {"name": "created_at", "type": models.PayloadSchemaType.DATETIME}
        ]
        
        # Add service namespacing index if shared collections are enabled
        shared_enabled = self.config.get("vector_store", {}).get("shared_collections_enabled", False)
        if shared_enabled:
            namespacing_field = self.config.get("vector_store", {}).get("namespacing_field", "service")
            index_fields.append({"name": namespacing_field, "type": models.PayloadSchemaType.KEYWORD})
            logger.info(f"Enabled shared collection namespacing using '{namespacing_field}' field")
        
        # Create each index with error handling
        for field in index_fields:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field["name"],
                    field_schema=field["type"]
                )
                logger.debug(f"Created index for field '{field['name']}'")
            except Exception as e:
                # This often happens when index already exists, which is fine
                logger.debug(f"Note: Index creation for '{field['name']}' returned: {str(e)}")
        
        logger.info(f"All payload indexes set up for collection '{self.collection_name}'")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def _get_embedding(self, text: str) -> List[float]:
        """Generate an embedding vector for the text using OpenAI API.
        
        This method converts text into a vector representation for semantic search.
        It includes retry logic for API failures and proper input validation.
        
        Args:
            text: The text to convert into a vector embedding
            
        Returns:
            List[float]: A vector representation of the input text
            
        Raises:
            Exception: If embedding generation fails after retries
        """
        # Validate input
        if not text or not text.strip():
            logger.warning("Attempted to embed empty text, returning zero vector")
            return [0.0] * self.embedding_dim
        
        # Process input text
        try:
            # OpenAI has a context length limit; truncate if needed
            model_token_limit = 8191  # Default for text-embedding-3-small
            
            # Approximate token count (rough estimate: 4 chars per token)
            if len(text) > model_token_limit * 4:
                truncated_length = model_token_limit * 4
                logger.warning(
                    f"Text length ({len(text)} chars) may exceed token limit, "
                    f"truncating to approximately {model_token_limit} tokens ({truncated_length} chars)"
                )
                text = text[:truncated_length]
            
            # Generate embedding via API
            start_time = time.time()
            response = await self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=text,
                encoding_format="float"  # Ensure we get float values
            )
            
            embedding = response.data[0].embedding
            
            # Verify embedding dimensions
            if len(embedding) != self.embedding_dim:
                logger.warning(
                    f"Embedding dimension mismatch: got {len(embedding)}, "
                    f"expected {self.embedding_dim}"
                )
            
            duration = time.time() - start_time
            logger.debug(f"Generated embedding in {duration:.2f}s")
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            if hasattr(e, "__traceback__"):
                logger.debug(f"Embedding error details: {traceback.format_exc()}")
            raise
    
    async def search(self, 
                   query: str, 
                   top_k: int = 3, 
                   limit: Optional[int] = None,
                   filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for semantically similar content in the vector store.
        
        This method performs a semantic similarity search using embeddings:
        1. Converts the query text to a vector embedding
        2. Finds the most similar vectors in the database
        3. Applies optional filters to narrow results
        4. Handles service namespacing for shared collections
        5. Returns formatted results with content and metadata
        
        Args:
            query: The text query to search for similar content
            top_k: Number of top results to return (default: 3)
            limit: Alternative name for top_k parameter (for API compatibility)
            filters: Optional dictionary of field:value filters to narrow results
            
        Returns:
            List of search results with content, score, and metadata
            Each result contains: id, content, score, and metadata keys
            
        Note:
            Returns an empty list on error rather than raising exceptions
        """
        start_time = time.time()
        
        # Use limit parameter if provided (for compatibility with other functions)
        if limit is not None:
            top_k = limit
            
        logger.info(f"Vector search: '{query[:50]}...' (top_k={top_k}, filters={filters})")
        
        # Check if in mock mode (either disconnected or explicitly configured)
        mock_mode = self.config.get("vector_store", {}).get("mock_mode", False)
        
        if (not self.connected and not mock_mode) or (not QDRANT_AVAILABLE and not mock_mode):
            logger.warning("Vector search unavailable - Qdrant not connected")
            return []
            
        # Return mock results if in mock mode
        if mock_mode or (not self.connected and QDRANT_AVAILABLE):
            logger.info("Returning mock search results")
            # Generate mock search results
            mock_results = []
            for i in range(min(top_k, 5)):  # Generate up to 5 mock results
                mock_results.append({
                    "id": f"mock-doc-{i}",
                    "content": f"This is mock content for search '{query[:20]}...' result {i+1}",
                    "score": 0.95 - (i * 0.1),
                    "metadata": {
                        "source": "mock",
                        "category": "test"
                    }
                })
            return mock_results
        
        try:
            # Convert query to vector embedding
            query_embedding = await self._get_embedding(query)
            
            # Build filter conditions list
            filter_conditions = []
            
            # Apply service namespacing for shared collections
            shared_enabled = self.config.get("vector_store", {}).get("shared_collections_enabled", False)
            if shared_enabled:
                namespacing_field = self.config.get("vector_store", {}).get("namespacing_field", "service")
                namespacing_value = self.config.get("vector_store", {}).get("namespacing_value", "gutenberg")
                filter_conditions.append(
                    models.FieldCondition(
                        key=namespacing_field,
                        match=models.MatchValue(value=namespacing_value)
                    )
                )
                logger.debug(f"Applied service namespace filter: {namespacing_field}={namespacing_value}")
            
            # Apply user-provided filters
            if filters:
                for field, value in filters.items():
                    if isinstance(value, list):
                        # For lists, match any value in the list
                        filter_conditions.append(
                            models.FieldCondition(
                                key=field,
                                match=models.MatchAny(any=value)
                            )
                        )
                    else:
                        # For single values, exact match
                        filter_conditions.append(
                            models.FieldCondition(
                                key=field,
                                match=models.MatchValue(value=value)
                            )
                        )
                logger.debug(f"Applied user filters: {filters}")
            
            # Create final filter query if we have conditions
            filter_query = None
            if filter_conditions:
                filter_query = models.Filter(must=filter_conditions)
            
            # Perform vector similarity search
            search_params = models.SearchParams(hnsw_ef=128)  # Customize search parameters
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                with_payload=True,
                with_vectors=False,  # Don't return vector data to save bandwidth
                search_params=search_params,
                filter=filter_query
            )
            
            # Format results for application use
            results = []
            for i, result in enumerate(search_results):
                payload = result.payload or {}
                
                # Extract content and metadata
                content = payload.get("content", "")
                # All fields except content go into metadata
                metadata = {k: v for k, v in payload.items() if k != "content"}
                
                # Format result with consistent structure
                formatted_result = {
                    "id": str(result.id),
                    "content": content,
                    "score": round(result.score, 4),
                    "metadata": metadata,
                    "rank": i + 1
                }
                
                results.append(formatted_result)
            
            search_time = time.time() - start_time
            logger.info(f"Vector search completed in {search_time:.2f}s with {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error in vector search: {str(e)}")
            if hasattr(e, "__traceback__"):
                logger.debug(f"Search error details: {traceback.format_exc()}")
            # Fall back to empty results on error
            return []
    
    async def index(self, 
                  content: str, 
                  metadata: Dict[str, Any],
                  id: Optional[str] = None) -> str:
        """Index content in the vector store for semantic search.
        
        This method:
        1. Converts the content text to a vector embedding
        2. Adds service namespacing for shared collections
        3. Stores both the vector and the content/metadata payload
        4. Generates or prefixes IDs appropriately
        
        Args:
            content: The text content to index
            metadata: Dictionary of metadata for the content
            id: Optional custom ID for the content (generated if not provided)
            
        Returns:
            String ID of the indexed content (for future reference or deletion)
            
        Raises:
            Exception: If indexing fails after retries
        """
        if not content:
            raise ValueError("Cannot index empty content")
            
        # Truncate content in log for readability
        content_preview = content[:50] + "..." if len(content) > 50 else content
        logger.info(f"Indexing content: '{content_preview}' with metadata: {metadata.keys()}")
        
        if not self.connected or not QDRANT_AVAILABLE:
            logger.error("Vector indexing unavailable - Qdrant not connected")
            raise RuntimeError("Vector store not connected")
        
        try:
            # Get vector embedding from text
            start_time = time.time()
            embedding = await self._get_embedding(content)
            
            # Generate a deterministic ID if not provided
            if not id:
                # Use timestamp and hash for unique ID generation
                timestamp = int(time.time())
                content_hash = abs(hash(content)) % 10000
                id = f"vec_{timestamp}_{content_hash}"
            
            # Prepare payload that combines content and metadata
            payload = {"content": content}
            
            # Add all metadata fields to payload
            if metadata:
                # Add creation timestamp if not provided
                if "created_at" not in metadata:
                    metadata["created_at"] = datetime.now().isoformat()
                payload.update(metadata)
            
            # Handle service namespacing for shared collections
            shared_enabled = self.config.get("vector_store", {}).get("shared_collections_enabled", False)
            if shared_enabled:
                # Add service field for filtering
                namespacing_field = self.config.get("vector_store", {}).get("namespacing_field", "service")
                namespacing_value = self.config.get("vector_store", {}).get("namespacing_value", "gutenberg")
                payload[namespacing_field] = namespacing_value
                
                # Prefix ID with service name for uniqueness across services
                service_prefix = self.config.get("vector_store", {}).get("service_prefix", "gutenberg_")
                vector_id = f"{service_prefix}{id}" if not id.startswith(service_prefix) else id
                logger.debug(f"Using shared collection with namespaced ID: {vector_id}")
            else:
                vector_id = id
            
            # Store in vector database
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=vector_id,
                        vector=embedding,
                        payload=payload
                    )
                ]
            )
            
            indexing_time = time.time() - start_time
            logger.info(f"Content indexed with ID: {id} in {indexing_time:.2f}s")
            return id
            
        except Exception as e:
            logger.error(f"Error indexing content: {str(e)}")
            if hasattr(e, "__traceback__"):
                logger.debug(f"Indexing error details: {traceback.format_exc()}")
            raise
    
    async def delete(self, id: str) -> bool:
        """Delete content from the vector store.
        
        This method removes vectors from the database by ID, handling service
        prefixing for shared collections.
        
        Args:
            id: ID of the content to delete
            
        Returns:
            bool: True if successful, False otherwise
            
        Note:
            Returns False instead of raising exceptions on error
        """
        logger.info(f"Deleting vector content with ID: {id}")
        
        if not self.connected or not QDRANT_AVAILABLE:
            logger.error("Vector deletion unavailable - Qdrant not connected")
            return False
        
        try:
            # Handle service prefixing for shared collections
            shared_enabled = self.config.get("vector_store", {}).get("shared_collections_enabled", False)
            if shared_enabled:
                service_prefix = self.config.get("vector_store", {}).get("service_prefix", "gutenberg_")
                vector_id = f"{service_prefix}{id}" if not id.startswith(service_prefix) else id
                logger.debug(f"Using namespaced ID for deletion: {vector_id}")
            else:
                vector_id = id
            
            # Delete from Qdrant
            result = self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=[vector_id]
                )
            )
            
            # Check if deletion was successful
            if result and hasattr(result, 'status') and result.status == 'completed':
                logger.info(f"Vector content successfully deleted: {id}")
                return True
            else:
                logger.warning(f"Vector content deletion may not have completed: {id}")
                return True  # Qdrant returns successfully even if ID doesn't exist
            
        except Exception as e:
            logger.error(f"Error deleting vector content: {str(e)}")
            if hasattr(e, "__traceback__"):
                logger.debug(f"Deletion error details: {traceback.format_exc()}")
            return False
    
    async def update(self, 
                   id: str, 
                   content: str, 
                   metadata: Dict[str, Any]) -> bool:
        """
        Update content in the vector store.
        
        Args:
            id: ID of the content to update
            content: New content
            metadata: New metadata
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Updating content with ID: {id}")
        
        try:
            # Generate new embedding
            embedding = await self._get_embedding(content)
            
            # Prepare payload with content and metadata
            payload = {"content": content}
            payload.update(metadata)
            
            # Update in Qdrant (upsert will create or update)
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=id,
                        vector=embedding,
                        payload=payload
                    )
                ]
            )
            
            logger.info(f"Content updated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error updating content: {str(e)}")
            return False
    
    async def get_by_id(self, id: str) -> Optional[Dict[str, Any]]:
        """
        Get content by ID.
        
        Args:
            id: ID of the content to retrieve
            
        Returns:
            Content and metadata or None if not found
        """
        logger.info(f"Getting content with ID: {id}")
        
        try:
            # Retrieve from Qdrant
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[id],
                with_vectors=False,
                with_payload=True
            )
            
            if not points:
                logger.info(f"Content not found")
                return None
                
            point = points[0]
            payload = point.payload
            
            # Extract content and metadata
            content = payload.get("content", "")
            metadata = {k: v for k, v in payload.items() if k != "content"}
            
            # Format the result
            result = {
                "id": str(point.id),
                "content": content,
                "metadata": metadata
            }
            
            logger.info(f"Content retrieved successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error getting content: {str(e)}")
            return None
    
    async def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for text - public wrapper for _get_embedding.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        # Check if in mock mode
        mock_mode = self.config.get("vector_store", {}).get("mock_mode", False)
        if mock_mode:
            # Return a mock embedding (just random numbers)
            import random
            return [random.random() for _ in range(self.embedding_dim)]
            
        return await self._get_embedding(text)
    
    async def store_embedding(self, doc_id: str, text: str, embedding: List[float], metadata: Dict[str, Any]) -> bool:
        """
        Store an embedding with its text and metadata.
        
        Args:
            doc_id: Unique document ID
            text: Original text
            embedding: Embedding vector
            metadata: Additional metadata
            
        Returns:
            True if successful
        """
        logger.info(f"Storing embedding for document: {doc_id}")
        
        # Check if in mock mode
        mock_mode = self.config.get("vector_store", {}).get("mock_mode", False)
        if mock_mode:
            logger.info(f"Mock mode: Simulating storage of embedding for document: {doc_id}")
            return True
        
        try:
            # Prepare payload with content and metadata
            payload = {"content": text}
            payload.update(metadata)
            
            # Index in Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=doc_id,
                        vector=embedding,
                        payload=payload
                    )
                ]
            )
            
            logger.info(f"Embedding stored successfully for document: {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing embedding: {str(e)}")
            return False
    
    async def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document from the vector store.
        
        Args:
            doc_id: Document ID to delete
            
        Returns:
            True if successful
        """
        return await self.delete(doc_id)
    
    async def health_check(self) -> bool:
        """
        Check if the vector store is available and ready.
        
        Returns:
            True if healthy
        """
        # Check if mock mode is enabled in config
        mock_mode = self.config.get("vector_store", {}).get("mock_mode", False)
        if not QDRANT_AVAILABLE or mock_mode:
            logger.warning("Qdrant in mock mode. Health check returning mock status.")
            return True  # Return true for mock mode
            
        try:
            if not self.connected:
                logger.warning("Vector store is not connected")
                return False
                
            # Check if we can connect and the collection exists
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            result = self.collection_name in collection_names
            logger.info(f"Vector store health check: {'passed' if result else 'failed'}")
            return result
            
        except Exception as e:
            logger.error(f"Vector store health check failed: {str(e)}")
            return False