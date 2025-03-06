"""
Ptolemy Knowledge Map System - Embedding Service
=============================================
Service for generating and managing vector embeddings of concepts.
"""

import logging
import os
import time
import traceback
import threading
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple, Set, Callable
from functools import lru_cache
from datetime import datetime

# Import required libraries with proper error handling
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import DBSCAN
    EMBEDDING_LIBRARIES_AVAILABLE = True
except ImportError as e:
    EMBEDDING_LIBRARIES_AVAILABLE = False
    error_msg = f"Required packages not installed: {e}. Run 'pip install sentence-transformers scikit-learn'"
    logging.error(error_msg)

from config import EmbeddingsConfig
from models import Concept

# Configure module-level logger
logger = logging.getLogger("embedding.service")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class EmbeddingService:
    """Service for generating vector embeddings for knowledge concepts."""
    
    def __init__(self, config: EmbeddingsConfig):
        """Initialize the embedding service with the given configuration.
        
        Args:
            config: Configuration for embedding generation
        """
        self.config = config
        self.model = None
        
        # Thread-local storage for per-thread embeddings cache
        self._local = threading.local()
        
        # Shared cache for batch operations
        self._batch_cache = {}
        self._batch_cache_timestamps = {}
        self._cache_lock = threading.RLock()
        
        # Metrics for monitoring
        self._metrics = {
            "embeddings_generated": 0,
            "batch_embeddings_generated": 0,
            "cache_hits": 0,
            "errors": 0,
            "last_error_time": None,
            "model_load_time": 0
        }
        
        # Load the model if dependencies are available
        if EMBEDDING_LIBRARIES_AVAILABLE:
            self.load_model()
        else:
            logger.error("Embedding service initialized in limited mode - required libraries not available")
    
    def load_model(self) -> bool:
        """Load the embedding model.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        if not EMBEDDING_LIBRARIES_AVAILABLE:
            logger.error("Cannot load model - required libraries not available")
            return False
            
        try:
            start_time = time.time()
            logger.info(f"Loading embedding model: {self.config.model_name}")
            
            # Configure GPU usage if requested
            if self.config.use_gpu:
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
            
            # Load model with appropriate settings
            device = "cuda" if self.config.use_gpu else "cpu"
            model_kwargs = {
                "cache_folder": self.config.cache_dir
            }
            
            if self.config.quantize:
                model_kwargs["quantize"] = True
            
            self.model = SentenceTransformer(
                self.config.model_name,
                device=device,
                **model_kwargs
            )
            
            # Set max sequence length
            if self.config.max_seq_length:
                self.model.max_seq_length = self.config.max_seq_length
            
            load_time = time.time() - start_time
            self._metrics["model_load_time"] = load_time
            
            logger.info(f"Loaded embedding model: {self.config.model_name} on {device} in {load_time:.2f}s")
            return True
        except Exception as e:
            self._metrics["errors"] += 1
            self._metrics["last_error_time"] = datetime.now().isoformat()
            logger.error(f"Failed to load embedding model: {e}")
            logger.debug(traceback.format_exc())
            self.model = None
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded embedding model.
        
        Returns:
            Dict containing model information
        """
        if not EMBEDDING_LIBRARIES_AVAILABLE:
            return {
                "status": "unavailable", 
                "error": "Required libraries not installed",
                "libraries_available": False
            }
            
        if not self.model:
            return {
                "status": "unavailable", 
                "error": "Model not loaded",
                "libraries_available": True
            }
        
        try:
            # Get model information
            info = {
                "status": "loaded",
                "model_name": self.config.model_name,
                "vector_size": self.model.get_sentence_embedding_dimension(),
                "max_seq_length": self.model.max_seq_length,
                "device": self.model.device.type,
                "quantized": self.config.quantize,
                "metrics": self._metrics.copy(),
                "libraries_available": True
            }
            
            # Add additional information if available
            if hasattr(self.model, "_model_card_vars") and hasattr(self.model._model_card_vars, "get"):
                if self.model._model_card_vars.get("language"):
                    info["language"] = self.model._model_card_vars.get("language")
                if self.model._model_card_vars.get("eval_results"):
                    info["eval_results"] = self.model._model_card_vars.get("eval_results")
            
            return info
        except Exception as e:
            self._metrics["errors"] += 1
            self._metrics["last_error_time"] = datetime.now().isoformat()
            logger.error(f"Error getting model info: {e}")
            return {"status": "error", "error": str(e), "libraries_available": True}
    
    def _get_per_thread_cache(self):
        """Get thread-local cache to prevent threading issues with lru_cache."""
        if not hasattr(self._local, 'embedding_cache'):
            self._local.embedding_cache = {}
        return self._local.embedding_cache
    
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate an embedding for a text string.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding, or None if generation failed
        """
        if not self.model:
            if not EMBEDDING_LIBRARIES_AVAILABLE:
                logger.error("Cannot generate embedding - required libraries not available")
            else:
                logger.error("Embedding model not available - call load_model() first")
            return None
        
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return None
        
        # Check thread-local cache
        cache = self._get_per_thread_cache()
        cache_key = hash(text)
        
        if cache_key in cache:
            self._metrics["cache_hits"] += 1
            return cache[cache_key]
        
        try:
            # Normalize text
            text = self._normalize_text(text)
            
            # Generate embedding
            start_time = time.time()
            embedding = self.model.encode(text)
            
            # Record metrics
            generation_time = time.time() - start_time
            self._metrics["embeddings_generated"] += 1
            
            # Log for long generations
            if generation_time > 1.0:
                logger.debug(f"Embedding generation took {generation_time:.2f}s for text of length {len(text)}")
            
            # Convert to list and add to cache
            result = embedding.tolist()
            cache[cache_key] = result
            
            return result
        except Exception as e:
            self._metrics["errors"] += 1
            self._metrics["last_error_time"] = datetime.now().isoformat()
            logger.error(f"Error generating embedding: {e}")
            logger.debug(traceback.format_exc())
            return None
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for consistent embedding generation.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        # Remove excess whitespace
        text = " ".join(text.split())
        
        # Truncate if too long
        max_length = self.config.max_seq_length or 512
        if len(text.split()) > max_length:
            words = text.split()[:max_length]
            text = " ".join(words)
            logger.debug(f"Text truncated to {max_length} tokens")
        
        return text
    
    def generate_concept_embedding(self, concept: Concept) -> Optional[List[float]]:
        """Generate an embedding for a concept, using name, description, and keywords.
        
        Args:
            concept: Concept to embed
            
        Returns:
            List of floats representing the embedding, or None if generation failed
        """
        if not self.model:
            logger.error("Embedding model not available")
            return None
        
        if not concept or not concept.name:
            logger.warning("Invalid concept provided for embedding")
            return None
        
        try:
            # Create a text representation of the concept
            text_parts = [concept.name]
            
            if concept.description:
                text_parts.append(concept.description)
            
            if concept.keywords and len(concept.keywords) > 0:
                keywords_text = ", ".join(concept.keywords)
                text_parts.append(f"Keywords: {keywords_text}")
            
            # Join all parts
            text = ". ".join(text_parts)
            
            # Generate embedding
            return self.generate_embedding(text)
        except Exception as e:
            self._metrics["errors"] += 1
            self._metrics["last_error_time"] = datetime.now().isoformat()
            logger.error(f"Error generating concept embedding: {e}")
            logger.debug(traceback.format_exc())
            return None
    
    def generate_batch_embeddings(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Generate embeddings for multiple texts in a batch.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings, or None if generation failed
        """
        if not self.model:
            logger.error("Embedding model not available")
            return None
        
        if not texts:
            logger.warning("Empty text list provided for batch embedding")
            return []
        
        try:
            # Clean texts and filter empty ones
            cleaned_texts = []
            indices = []
            
            for i, text in enumerate(texts):
                if text and text.strip():
                    cleaned_texts.append(self._normalize_text(text))
                    indices.append(i)
                else:
                    logger.warning(f"Empty text at index {i} in batch - skipping")
            
            if not cleaned_texts:
                logger.warning("No valid texts in batch after filtering")
                return []
            
            # Check cache for batch
            cache_key = self._get_batch_cache_key(cleaned_texts)
            cached_result = self._get_from_batch_cache(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Generate embeddings in batch
            start_time = time.time()
            embeddings = self.model.encode(
                cleaned_texts,
                batch_size=self.config.batch_size,
                show_progress_bar=len(cleaned_texts) > 100
            )
            
            # Update metrics
            batch_time = time.time() - start_time
            self._metrics["batch_embeddings_generated"] += 1
            
            # Log for long generations
            if batch_time > 2.0:
                logger.info(f"Batch embedding generation took {batch_time:.2f}s for {len(cleaned_texts)} texts")
            
            # Convert to list
            result = embeddings.tolist()
            
            # Add to cache
            self._add_to_batch_cache(cache_key, result)
            
            return result
        except Exception as e:
            self._metrics["errors"] += 1
            self._metrics["last_error_time"] = datetime.now().isoformat()
            logger.error(f"Error generating batch embeddings: {e}")
            logger.debug(traceback.format_exc())
            return None
    
    def _get_batch_cache_key(self, texts: List[str]) -> str:
        """Generate a cache key for a batch of texts.
        
        Args:
            texts: List of texts
            
        Returns:
            Cache key
        """
        # Use a hash of the concatenated texts
        return str(hash("".join(texts)))
    
    def _get_from_batch_cache(self, key: str) -> Optional[List[List[float]]]:
        """Get a result from the batch cache if it exists and is fresh.
        
        Args:
            key: Cache key
            
        Returns:
            Cached result or None
        """
        with self._cache_lock:
            if key in self._batch_cache:
                # Check if cache entry is still valid (TTL: 1 hour)
                timestamp = self._batch_cache_timestamps.get(key, 0)
                if time.time() - timestamp < 3600:  # 1 hour TTL
                    self._metrics["cache_hits"] += 1
                    return self._batch_cache[key]
                else:
                    # Clean expired entry
                    del self._batch_cache[key]
                    del self._batch_cache_timestamps[key]
            return None
    
    def _add_to_batch_cache(self, key: str, result: List[List[float]]) -> None:
        """Add a result to the batch cache.
        
        Args:
            key: Cache key
            result: Result to cache
        """
        with self._cache_lock:
            self._batch_cache[key] = result
            self._batch_cache_timestamps[key] = time.time()
            
            # Clean cache if it gets too large
            if len(self._batch_cache) > 100:
                self._clean_batch_cache()
    
    def _clean_batch_cache(self) -> None:
        """Clean expired entries from the batch cache."""
        with self._cache_lock:
            current_time = time.time()
            expired_keys = [
                k for k, t in self._batch_cache_timestamps.items() 
                if current_time - t > 3600  # 1 hour TTL
            ]
            
            for key in expired_keys:
                del self._batch_cache[key]
                del self._batch_cache_timestamps[key]
            
            # Log cache cleaning
            if expired_keys:
                logger.debug(f"Cleaned {len(expired_keys)} expired entries from batch cache")
    
    def generate_batch_concept_embeddings(self, concepts: List[Concept]) -> Dict[str, List[float]]:
        """Generate embeddings for multiple concepts in a batch.
        
        Args:
            concepts: List of concepts to embed
            
        Returns:
            Dictionary mapping concept IDs to embeddings
        """
        if not self.model:
            logger.error("Embedding model not available")
            return {}
        
        if not concepts:
            logger.warning("Empty concept list provided for batch embedding")
            return {}
        
        try:
            # Prepare texts and track concept IDs
            texts = []
            concept_ids = []
            
            for concept in concepts:
                if not concept or not concept.id or not concept.name:
                    logger.warning(f"Invalid concept in batch - skipping: {getattr(concept, 'id', 'unknown')}")
                    continue
                
                # Create a text representation of the concept
                text_parts = [concept.name]
                
                if concept.description:
                    text_parts.append(concept.description)
                
                if concept.keywords and len(concept.keywords) > 0:
                    keywords_text = ", ".join(concept.keywords)
                    text_parts.append(f"Keywords: {keywords_text}")
                
                # Join all parts
                text = ". ".join(text_parts)
                texts.append(text)
                concept_ids.append(concept.id)
            
            if not texts:
                logger.warning("No valid concepts in batch after filtering")
                return {}
            
            # Generate embeddings
            batch_embeddings = self.generate_batch_embeddings(texts)
            if not batch_embeddings:
                return {}
            
            # Map concept IDs to embeddings
            result = {}
            for i, concept_id in enumerate(concept_ids):
                if i < len(batch_embeddings):
                    result[concept_id] = batch_embeddings[i]
            
            logger.info(f"Generated embeddings for {len(result)} concepts in batch")
            return result
        except Exception as e:
            self._metrics["errors"] += 1
            self._metrics["last_error_time"] = datetime.now().isoformat()
            logger.error(f"Error generating batch concept embeddings: {e}")
            logger.debug(traceback.format_exc())
            return {}
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        if not embedding1 or not embedding2:
            return 0.0
        
        try:
            # Convert to numpy arrays if needed
            v1 = np.array(embedding1, dtype=np.float32)
            v2 = np.array(embedding2, dtype=np.float32)
            
            # Check dimensions
            if v1.ndim != 1 or v2.ndim != 1:
                v1 = v1.flatten()
                v2 = v2.flatten()
            
            # Check if shapes match
            if v1.shape != v2.shape:
                logger.warning(f"Embedding shapes don't match: {v1.shape} vs {v2.shape}")
                # Try to make them match by truncating the longer one
                min_dim = min(v1.shape[0], v2.shape[0])
                v1 = v1[:min_dim]
                v2 = v2[:min_dim]
            
            # Reshape embeddings for cosine_similarity
            v1 = v1.reshape(1, -1)
            v2 = v2.reshape(1, -1)
            
            similarity = cosine_similarity(v1, v2)[0][0]
            
            # Ensure result is within bounds
            similarity = max(0.0, min(1.0, float(similarity)))
            
            return similarity
        except Exception as e:
            self._metrics["errors"] += 1
            self._metrics["last_error_time"] = datetime.now().isoformat()
            logger.error(f"Error calculating similarity: {e}")
            logger.debug(traceback.format_exc())
            return 0.0
    
    def calculate_concept_similarities(self, concept_embeddings: Dict[str, List[float]]) -> Dict[Tuple[str, str], float]:
        """Calculate pairwise similarities between multiple concept embeddings.
        
        Args:
            concept_embeddings: Dictionary mapping concept IDs to embeddings
            
        Returns:
            Dictionary mapping concept ID pairs to similarity scores
        """
        if not concept_embeddings:
            return {}
        
        try:
            # Extract concept IDs and embeddings
            concept_ids = list(concept_embeddings.keys())
            
            # Skip if less than 2 concepts
            if len(concept_ids) < 2:
                logger.warning("Need at least 2 concepts to calculate similarities")
                return {}
            
            # Convert embeddings to numpy array
            embeddings_list = [concept_embeddings[cid] for cid in concept_ids]
            
            # Check if all embeddings have the same dimensionality
            dimensions = [len(emb) for emb in embeddings_list]
            if len(set(dimensions)) > 1:
                # Handle different dimensions by using the minimum dimension
                min_dim = min(dimensions)
                logger.warning(f"Embeddings have different dimensions. Using minimum: {min_dim}")
                embeddings = np.array([emb[:min_dim] for emb in embeddings_list])
            else:
                embeddings = np.array(embeddings_list)
            
            # Calculate pairwise similarities
            start_time = time.time()
            similarity_matrix = cosine_similarity(embeddings)
            
            # Convert to dictionary of pairs
            result = {}
            for i in range(len(concept_ids)):
                for j in range(i+1, len(concept_ids)):
                    pair = (concept_ids[i], concept_ids[j])
                    similarity = float(similarity_matrix[i][j])
                    # Ensure similarity is within bounds
                    similarity = max(0.0, min(1.0, similarity))
                    result[pair] = similarity
            
            calculation_time = time.time() - start_time
            if calculation_time > 1.0:
                logger.info(f"Calculated {len(result)} concept similarities in {calculation_time:.2f}s")
            
            return result
        except Exception as e:
            self._metrics["errors"] += 1
            self._metrics["last_error_time"] = datetime.now().isoformat()
            logger.error(f"Error calculating concept similarities: {e}")
            logger.debug(traceback.format_exc())
            return {}
    
    def find_similar_texts(self, query: str, texts: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """Find texts most similar to a query text.
        
        Args:
            query: Query text
            texts: List of texts to search
            top_k: Number of results to return
            
        Returns:
            List of dictionaries with index, text, and similarity score
        """
        if not self.model:
            logger.error("Embedding model not available")
            return []
        
        if not query or not texts:
            logger.warning("Empty query or texts provided")
            return []
        
        try:
            # Normalize query and texts
            query = self._normalize_text(query)
            cleaned_texts = [self._normalize_text(text) for text in texts if text and text.strip()]
            
            if not cleaned_texts:
                logger.warning("No valid texts after filtering")
                return []
            
            # Generate embedding for query
            query_embedding = self.model.encode(query)
            
            # Generate embeddings for all texts
            corpus_embeddings = self.model.encode(cleaned_texts)
            
            # Calculate similarities
            similarities = cosine_similarity([query_embedding], corpus_embeddings)[0]
            
            # Get top-k results
            top_k = min(top_k, len(cleaned_texts))
            top_indices = similarities.argsort()[-top_k:][::-1]
            results = []
            
            for idx in top_indices:
                similarity = float(similarities[idx])
                # Ensure similarity is within bounds
                similarity = max(0.0, min(1.0, similarity))
                
                results.append({
                    "index": int(idx),
                    "text": cleaned_texts[idx],
                    "similarity": similarity
                })
            
            return results
        except Exception as e:
            self._metrics["errors"] += 1
            self._metrics["last_error_time"] = datetime.now().isoformat()
            logger.error(f"Error finding similar texts: {e}")
            logger.debug(traceback.format_exc())
            return []
    
    def detect_semantic_clusters(self, embeddings: List[List[float]], 
                              threshold: float = 0.8,
                              min_samples: int = 2) -> List[List[int]]:
        """Detect clusters of semantically similar embeddings.
        
        Args:
            embeddings: List of embeddings
            threshold: Similarity threshold (0.0 to 1.0)
            min_samples: Minimum number of samples to form a cluster
            
        Returns:
            List of clusters, where each cluster is a list of indices
        """
        if not embeddings:
            return []
        
        # For very small sets, use simple approach
        if len(embeddings) < 5:
            return self._detect_clusters_simple(embeddings, threshold)
        
        try:
            # Convert embeddings to numpy array
            embeddings_array = np.array(embeddings)
            
            # Check for inconsistent dimensions
            dimensions = [len(emb) for emb in embeddings]
            if len(set(dimensions)) > 1:
                # Handle different dimensions by using the minimum dimension
                min_dim = min(dimensions)
                logger.warning(f"Embeddings have different dimensions. Using minimum: {min_dim}")
                embeddings_array = np.array([emb[:min_dim] for emb in embeddings])
            
            # Calculate distance matrix (1 - similarity)
            similarity_matrix = cosine_similarity(embeddings_array)
            distance_matrix = 1 - similarity_matrix
            
            # Use DBSCAN for clustering
            eps = 1 - threshold  # Convert similarity threshold to distance threshold
            clustering = DBSCAN(
                eps=eps,
                min_samples=min_samples,
                metric='precomputed'
            ).fit(distance_matrix)
            
            # Get cluster labels
            labels = clustering.labels_
            
            # Organize into clusters
            clusters = {}
            for i, label in enumerate(labels):
                if label >= 0:  # Ignore noise points (label -1)
                    if label not in clusters:
                        clusters[label] = []
                    clusters[label].append(i)
            
            # Convert to list of clusters
            result = list(clusters.values())
            
            logger.info(f"Detected {len(result)} semantic clusters with threshold {threshold}")
            return result
        except Exception as e:
            self._metrics["errors"] += 1
            self._metrics["last_error_time"] = datetime.now().isoformat()
            logger.error(f"Error detecting semantic clusters: {e}")
            logger.debug(traceback.format_exc())
            
            # Fallback to simple approach
            return self._detect_clusters_simple(embeddings, threshold)
    
    def _detect_clusters_simple(self, embeddings: List[List[float]], 
                             threshold: float = 0.8) -> List[List[int]]:
        """Simple clustering approach for small datasets.
        
        Args:
            embeddings: List of embeddings
            threshold: Similarity threshold
            
        Returns:
            List of clusters
        """
        try:
            # Create similarity matrix
            embeddings_array = np.array(embeddings)
            similarity_matrix = cosine_similarity(embeddings_array)
            
            # Find clusters using a simple threshold-based approach
            n = len(embeddings)
            visited = [False] * n
            clusters = []
            
            for i in range(n):
                if visited[i]:
                    continue
                
                cluster = [i]
                visited[i] = True
                
                for j in range(n):
                    if not visited[j] and similarity_matrix[i][j] >= threshold:
                        cluster.append(j)
                        visited[j] = True
                
                if len(cluster) > 1:  # Only keep clusters with multiple items
                    clusters.append(cluster)
            
            return clusters
        except Exception as e:
            logger.error(f"Error in simple cluster detection: {e}")
            return []
    
    def find_semantic_duplicates(self, concepts: List[Concept], 
                              threshold: float = 0.9) -> List[List[str]]:
        """Find potentially duplicate concepts based on semantic similarity.
        
        Args:
            concepts: List of concepts
            threshold: Similarity threshold (0.0 to 1.0)
            
        Returns:
            List of clusters, where each cluster is a list of concept IDs
        """
        if not self.model or not concepts:
            return []
        
        try:
            # Generate embeddings for concepts
            concept_embeddings = self.generate_batch_concept_embeddings(concepts)
            if not concept_embeddings:
                return []
            
            # Extract concept IDs and embeddings
            concept_ids = list(concept_embeddings.keys())
            embeddings = [concept_embeddings[cid] for cid in concept_ids]
            
            # Detect clusters
            clusters = self.detect_semantic_clusters(embeddings, threshold)
            
            # Map cluster indices to concept IDs
            result = []
            for cluster in clusters:
                concept_cluster = [concept_ids[idx] for idx in cluster]
                result.append(concept_cluster)
            
            logger.info(f"Found {len(result)} potential duplicate clusters with threshold {threshold}")
            return result
        except Exception as e:
            self._metrics["errors"] += 1
            self._metrics["last_error_time"] = datetime.now().isoformat()
            logger.error(f"Error finding semantic duplicates: {e}")
            logger.debug(traceback.format_exc())
            return []
    
    def find_concept_gaps(self, domain_concepts: List[Concept], 
                        seed_phrases: List[str], 
                        threshold: float = 0.75) -> List[str]:
        """Find potential gaps in domain concepts based on seed phrases.
        
        Args:
            domain_concepts: List of existing domain concepts
            seed_phrases: List of seed phrases to check for coverage
            threshold: Similarity threshold for considering a phrase covered
            
        Returns:
            List of seed phrases that may represent gaps
        """
        if not self.model:
            logger.error("Embedding model not available")
            return []
        
        if not domain_concepts or not seed_phrases:
            return []
        
        try:
            # Generate embeddings for concepts
            concept_embeddings = self.generate_batch_concept_embeddings(domain_concepts)
            if not concept_embeddings:
                return []
            
            # Extract concept embeddings as a list
            concept_embs = list(concept_embeddings.values())
            
            # Check each seed phrase
            gaps = []
            for phrase in seed_phrases:
                if not phrase or not phrase.strip():
                    continue
                
                # Generate embedding for phrase
                phrase_emb = self.generate_embedding(phrase)
                if not phrase_emb:
                    continue
                
                # Calculate max similarity with any concept
                max_similarity = 0.0
                for concept_emb in concept_embs:
                    similarity = self.calculate_similarity(phrase_emb, concept_emb)
                    max_similarity = max(max_similarity, similarity)
                
                # If max similarity is below threshold, consider it a gap
                if max_similarity < threshold:
                    gaps.append(phrase)
            
            logger.info(f"Found {len(gaps)} potential concept gaps out of {len(seed_phrases)} seed phrases")
            return gaps
        except Exception as e:
            self._metrics["errors"] += 1
            self._metrics["last_error_time"] = datetime.now().isoformat()
            logger.error(f"Error finding concept gaps: {e}")
            logger.debug(traceback.format_exc())
            return []
    
    def close(self):
        """Clean up resources."""
        logger.info("Closing embedding service")
        
        # Clean caches
        self._local = threading.local()
        with self._cache_lock:
            self._batch_cache.clear()
            self._batch_cache_timestamps.clear()
        
        # Clear model reference to free memory
        self.model = None