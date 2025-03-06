"""
RAG Processor
============
Retrieval-Augmented Generation processor for enhancing LLM outputs with relevant knowledge.
"""

import time
import logging
import json
from typing import Dict, List, Optional, Any, Union
import asyncio
import re

from config.settings import get_config
from integrations.vector_store import VectorStore
from integrations.llm_service import LLMService, LLMResponse
from utils.logging_utils import get_logger

logger = get_logger(__name__)

class RAGProcessor:
    """
    Retrieval-Augmented Generation processor for enhancing content generation
    with relevant knowledge from external sources.
    """
    
    def __init__(self):
        """Initialize the RAG processor."""
        self.config = get_config()
        
        # Check if we should use mock mode
        self.use_mock = self.config.get("vector_store", {}).get("use_mock", False)
        
        if self.use_mock:
            logger.info("RAG processor initializing in mock mode")
            self.vector_store = None
            self.llm_service = LLMService()
        else:
            self.vector_store = VectorStore()
            self.llm_service = LLMService()
        
        # Configuration
        self.max_sources = self.config.get("rag", {}).get("max_sources", 5)
        self.relevance_threshold = self.config.get("rag", {}).get("relevance_threshold", 0.7)
        self.chunk_size = self.config.get("rag", {}).get("chunk_size", 300)
        
        logger.info("RAGProcessor initialized")
    
    async def enhance_prompt(self, prompt: str, context: Dict[str, Any]) -> str:
        """
        Enhance a prompt with relevant knowledge.
        
        Args:
            prompt: The original prompt
            context: Context information including domain, topic, etc.
            
        Returns:
            Enhanced prompt with additional knowledge
        """
        logger.info(f"Enhancing prompt: {prompt[:50]}...")
        start_time = time.time()
        
        try:
            # Extract key terms from the prompt and context
            key_terms = await self._extract_key_terms(prompt, context)
            logger.info(f"Extracted key terms: {', '.join(key_terms)}")
            
            # Retrieve knowledge based on extracted terms
            all_knowledge = []
            for term in key_terms:
                term_knowledge = await self.retrieve_knowledge(term, n_results=3)
                all_knowledge.extend(term_knowledge)
            
            # Sort by relevance and deduplicate
            all_knowledge.sort(key=lambda x: x.get("score", 0), reverse=True)
            
            # Deduplicate
            seen_content = set()
            unique_knowledge = []
            for item in all_knowledge:
                content_hash = hash(item.get("content", "")[:100])  # Use first 100 chars as signature
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    unique_knowledge.append(item)
            
            # Limit to top sources that meet relevance threshold
            filtered_knowledge = [
                k for k in unique_knowledge[:self.max_sources] 
                if k.get("score", 0) >= self.relevance_threshold
            ]
            
            # Format the knowledge into context
            knowledge_context = self._format_knowledge_for_context(filtered_knowledge)
            
            # Combine with original prompt
            if knowledge_context:
                enhanced_prompt = f"{prompt}\n\nAdditional context:\n{knowledge_context}"
                
                logger.info(f"Prompt enhanced with {len(filtered_knowledge)} knowledge items in {time.time() - start_time:.2f}s")
                return enhanced_prompt
            else:
                logger.info(f"No relevant knowledge found for prompt enhancement in {time.time() - start_time:.2f}s")
                return prompt
            
        except Exception as e:
            logger.error(f"Error enhancing prompt: {str(e)}")
            return prompt  # Fall back to original prompt on error
    
    async def retrieve_knowledge(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant knowledge for a query.
        
        Args:
            query: The search query
            n_results: Number of results to return
            
        Returns:
            List of knowledge items with content and metadata
        """
        logger.info(f"Retrieving knowledge for: {query[:50]}... (n_results={n_results})")
        start_time = time.time()
        
        # If in mock mode, return mock data
        if self.use_mock:
            logger.info(f"Using mock knowledge retrieval for: {query}")
            mock_results = [
                {
                    "content": f"Mock content about {query}",
                    "metadata": {"type": "mock", "query": query},
                    "source": "Mock Source",
                    "score": 0.95
                },
                {
                    "content": f"Additional information about {query} with details",
                    "metadata": {"type": "mock", "query": query},
                    "source": "Mock Reference",
                    "score": 0.85
                }
            ]
            return mock_results[:n_results]
        
        try:
            # Search the vector store
            results = await self.vector_store.search(query, top_k=n_results)
            
            processed_results = []
            for item in results:
                processed_results.append({
                    "content": item.get("content", ""),
                    "metadata": item.get("metadata", {}),
                    "source": item.get("metadata", {}).get("source", "Unknown"),
                    "score": item.get("score", 0)
                })
            
            logger.info(f"Retrieved {len(processed_results)} knowledge items in {time.time() - start_time:.2f}s")
            return processed_results
            
        except Exception as e:
            logger.error(f"Error retrieving knowledge: {str(e)}")
            return []  # Return empty list on error
    
    async def generate_with_rag(self, 
                              prompt: str, 
                              system_message: Optional[str] = None,
                              context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate content using RAG approach.
        
        Args:
            prompt: The user prompt
            system_message: Optional system message
            context: Optional context information
            
        Returns:
            Generated content with metadata about the sources used
        """
        logger.info(f"Generating with RAG: {prompt[:50]}...")
        start_time = time.time()
        
        context = context or {}
        
        try:
            # 1. Extract key terms from the prompt
            key_terms = await self._extract_key_terms(prompt, context)
            
            # 2. Retrieve relevant knowledge for each key term
            retrieved_knowledge = []
            retrieval_tasks = [self.retrieve_knowledge(term, n_results=3) for term in key_terms]
            retrieval_results = await asyncio.gather(*retrieval_tasks)
            
            for term_knowledge in retrieval_results:
                retrieved_knowledge.extend(term_knowledge)
            
            # 3. Sort by relevance and deduplicate
            retrieved_knowledge.sort(key=lambda x: x.get("score", 0), reverse=True)
            
            # Deduplicate by content
            seen_content = set()
            unique_knowledge = []
            for item in retrieved_knowledge:
                content = item.get("content", "")
                content_hash = hash(content[:100])  # Use first 100 chars as signature
                if content_hash and content_hash not in seen_content:
                    seen_content.add(content_hash)
                    unique_knowledge.append(item)
            
            # Limit to top N relevant items
            filtered_knowledge = [
                k for k in unique_knowledge[:self.max_sources] 
                if k.get("score", 0) >= self.relevance_threshold
            ]
            
            # 4. Format the retrieved knowledge
            knowledge_context = self._format_knowledge(filtered_knowledge)
            
            # 5. Enhance the system message with the knowledge
            enhanced_system = system_message or "You are a helpful assistant."
            enhanced_system += f"\n\nUse the following additional information to inform your response, but respond in your own words without directly quoting these sources:\n{knowledge_context}"
            
            # 6. Generate the response using the enhanced prompt
            response = await self.llm_service.generate_content(
                prompt=prompt,
                system_message=enhanced_system,
                temperature=0.5
            )
            
            # 7. Prepare the result
            sources = []
            for item in filtered_knowledge:
                source_info = {
                    "content": item.get("content", ""),
                    "source": item.get("source", "Unknown"),
                    "score": item.get("score", 0)
                }
                if "metadata" in item and isinstance(item["metadata"], dict):
                    source_info.update({
                        k: v for k, v in item["metadata"].items() 
                        if k not in ("content", "source", "score")
                    })
                sources.append(source_info)
            
            result = {
                "content": response.content,
                "structured_output": response.structured_output,
                "sources": sources,
                "processing_time": time.time() - start_time,
                "tokens": {
                    "prompt": response.prompt_tokens,
                    "completion": response.completion_tokens,
                    "total": response.total_tokens
                }
            }
            
            logger.info(f"RAG generation completed in {result['processing_time']:.2f}s with {len(sources)} sources")
            return result
            
        except Exception as e:
            logger.error(f"Error in RAG generation: {str(e)}")
            
            # Fall back to standard generation
            try:
                response = await self.llm_service.generate_content(
                    prompt=prompt,
                    system_message=system_message,
                    temperature=0.7
                )
                
                return {
                    "content": response.content,
                    "structured_output": response.structured_output,
                    "sources": [],
                    "error": str(e),
                    "processing_time": time.time() - start_time,
                    "tokens": {
                        "prompt": response.prompt_tokens,
                        "completion": response.completion_tokens,
                        "total": response.total_tokens
                    }
                }
            except Exception as fallback_error:
                logger.error(f"Error in fallback generation: {str(fallback_error)}")
                return {
                    "content": f"Error generating content: {str(e)}",
                    "sources": [],
                    "error": str(e),
                    "fallback_error": str(fallback_error),
                    "processing_time": time.time() - start_time
                }
    
    async def _extract_key_terms(self, text: str, context: Dict[str, Any] = None) -> List[str]:
        """
        Extract key terms from text and context for knowledge retrieval.
        
        Args:
            text: The text to analyze
            context: Additional context with domain, topic, etc.
            
        Returns:
            List of key terms
        """
        # Get terms from context if available
        context = context or {}
        context_terms = []
        
        if "topic" in context:
            context_terms.append(context["topic"])
            
        if "concept_name" in context:
            context_terms.append(context["concept_name"])
            
        if "domain" in context:
            context_terms.append(context["domain"])
        
        # Extract terms from text using LLM
        try:
            # Use LLM to extract key terms
            system_message = """
            You are a keyword extraction specialist. Extract the 3-5 most important search terms or phrases
            from the text that would be useful for retrieving relevant knowledge. Focus on domain-specific 
            concepts, named entities, and technical terms. Return only a JSON array of strings with no additional text.
            """
            
            response = await self.llm_service.generate_content(
                prompt=f"Extract key search terms from this text:\n\n{text}",
                system_message=system_message,
                temperature=0.3,
                output_format={"key_terms": ["string"]}
            )
            
            if response.structured_output and "key_terms" in response.structured_output:
                llm_terms = response.structured_output["key_terms"]
                
                # Combine with context terms and deduplicate
                all_terms = context_terms + llm_terms
                unique_terms = list(dict.fromkeys(all_terms))  # Preserve order while deduplicating
                
                logger.info(f"Extracted {len(unique_terms)} key terms")
                return unique_terms
                
            else:
                # Fall back to regex-based extraction if LLM failed
                logger.warning("LLM key term extraction failed, falling back to regex")
                return self._extract_key_terms_regex(text, context_terms)
                
        except Exception as e:
            logger.error(f"Error extracting key terms with LLM: {str(e)}")
            return self._extract_key_terms_regex(text, context_terms)
    
    def _extract_key_terms_regex(self, text: str, context_terms: List[str]) -> List[str]:
        """
        Extract key terms using regex as a fallback.
        
        Args:
            text: Text to extract terms from
            context_terms: Terms already extracted from context
            
        Returns:
            List of key terms
        """
        terms = list(context_terms)  # Start with context terms
        
        # Extract capitalized phrases (potentially named entities)
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        terms.extend(capitalized[:3])  # Take up to 3 capitalized terms
        
        # Extract potential key phrases
        phrases = re.findall(r'\b\w+\s+(?:of|for|in|on|with)\s+\w+\b', text)
        terms.extend(phrases[:3])  # Take up to 3 prepositional phrases
        
        # Extract longer words (potentially domain-specific terms)
        words = text.split()
        long_words = [word.strip('.,()[]{}:;"\'-') for word in words if len(word) > 5 and word.isalpha()]
        terms.extend(long_words[:3])  # Take up to 3 longer words
        
        # Ensure we have at least one term
        if not terms and text:
            terms.append(text.split()[0])
            
        # Remove duplicates while preserving order
        unique_terms = list(dict.fromkeys(terms))
        
        return unique_terms[:5]  # Limit to 5 terms
    
    def _format_knowledge(self, knowledge_items: List[Dict[str, Any]]) -> str:
        """
        Format retrieved knowledge into a context string for LLM prompting.
        
        Args:
            knowledge_items: List of knowledge items
            
        Returns:
            Formatted knowledge context
        """
        if not knowledge_items:
            return ""
        
        context_parts = []
        
        for i, item in enumerate(knowledge_items, 1):
            content = item.get("content", "")
            source = item.get("source", "Unknown Source")
            
            if content:
                context_parts.append(f"Source {i}: {content} (Reference: {source})")
        
        return "\n\n".join(context_parts)
    
    def _format_knowledge_for_context(self, knowledge_items: List[Dict[str, Any]]) -> str:
        """
        Format retrieved knowledge into a context string for prompt enhancement.
        
        Args:
            knowledge_items: List of knowledge items
            
        Returns:
            Formatted knowledge context
        """
        if not knowledge_items:
            return ""
        
        context_parts = []
        
        for item in knowledge_items:
            content = item.get("content", "")
            source = item.get("source", "")
            
            if content:
                context_parts.append(f"- {content}")
                if source:
                    context_parts[-1] += f" (Source: {source})"
        
        return "\n".join(context_parts)