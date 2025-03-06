"""
RAG (Retrieval-Augmented Generation) Processor
=============================================
Component for enhancing content generation with retrieved knowledge.
"""

import logging
import time
import asyncio
from typing import Dict, List, Optional, Any, Union

from pydantic import BaseModel, Field

from integrations.vector_store import VectorStore
from integrations.ptolemy_client import PtolemyClient
from integrations.llm_service import LLMService
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class RetrievalResult(BaseModel):
    """Model for a retrieval result."""
    content: str = Field(..., description="Retrieved content")
    source: str = Field(..., description="Source of the content")
    relevance: float = Field(..., description="Relevance score (0-1)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class RAGResult(BaseModel):
    """Model for RAG processing result."""
    content: str = Field(..., description="Generated content")
    retrieval_results: List[RetrievalResult] = Field(default_factory=list, description="Retrieved items used")
    query: str = Field(..., description="Original query")
    citations: List[Dict[str, Any]] = Field(default_factory=list, description="Citations for the content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class RAGProcessor:
    """
    Processor for Retrieval-Augmented Generation to enhance content
    generation with knowledge retrieval.
    """
    
    def __init__(self):
        """Initialize the RAG processor."""
        self.vector_store = VectorStore()
        self.ptolemy = PtolemyClient()
        self.llm_service = LLMService()
        
        logger.info("RAG processor initialized")
    
    async def process_query(self, query: str, context: Dict[str, Any] = None, 
                          temperature: float = 0.7) -> RAGResult:
        """
        Process a query using RAG to generate enhanced content.
        
        Args:
            query: The query or prompt to process
            context: Additional context data
            temperature: Temperature for generation
            
        Returns:
            RAG result with content and retrieval metadata
        """
        logger.info(f"Processing RAG query: {query[:50]}...")
        start_time = time.time()
        
        # Get retrieval results
        retrieval_results = await self._retrieve_knowledge(query, context)
        
        # Prepare content generation with retrieved knowledge
        retrieval_context = self._prepare_retrieval_context(retrieval_results)
        
        # Combine with the original query and generate content
        enhanced_prompt = self._create_enhanced_prompt(query, retrieval_context)
        
        # Generate content with LLM
        response = await self.llm_service.generate_content(
            prompt=enhanced_prompt,
            system_message="You are an educational content creator using retrieved information to answer questions accurately. Cite sources where appropriate.",
            temperature=temperature
        )
        
        # Extract and format citations
        citations = self._extract_citations(response.content, retrieval_results)
        
        # Create result
        result = RAGResult(
            content=response.content,
            retrieval_results=retrieval_results,
            query=query,
            citations=citations,
            metadata={
                "processing_time": time.time() - start_time,
                "retrieval_count": len(retrieval_results),
                "sources": [r.source for r in retrieval_results]
            }
        )
        
        logger.info(f"RAG processing complete: {len(retrieval_results)} retrieval results in {time.time() - start_time:.2f}s")
        return result
    
    async def _retrieve_knowledge(self, query: str, context: Dict[str, Any] = None) -> List[RetrievalResult]:
        """
        Retrieve relevant knowledge for a query.
        
        Args:
            query: The query to retrieve knowledge for
            context: Additional context data
            
        Returns:
            List of retrieval results
        """
        retrieval_results = []
        
        # Retrieve from vector store
        vector_results = await self._retrieve_from_vector_store(query)
        retrieval_results.extend(vector_results)
        
        # Retrieve from Ptolemy if context contains concept info
        if context and "concept_id" in context:
            ptolemy_results = await self._retrieve_from_ptolemy(query, context["concept_id"])
            retrieval_results.extend(ptolemy_results)
        
        # Sort by relevance
        retrieval_results.sort(key=lambda x: x.relevance, reverse=True)
        
        # Limit to most relevant results
        return retrieval_results[:5]
    
    async def _retrieve_from_vector_store(self, query: str, top_k: int = 3) -> List[RetrievalResult]:
        """
        Retrieve knowledge from the vector store.
        
        Args:
            query: The query to retrieve knowledge for
            top_k: Number of results to retrieve
            
        Returns:
            List of retrieval results
        """
        try:
            # Check vector store health before querying
            is_healthy = await self.vector_store.health_check()
            if not is_healthy:
                logger.warning("Vector store health check failed, skipping retrieval")
                return []
            
            # Perform search with retry logic
            max_retries = 2
            retry_count = 0
            results = []
            
            while retry_count <= max_retries:
                try:
                    results = await self.vector_store.search(query, top_k=top_k)
                    break  # Success, exit retry loop
                except Exception as search_error:
                    retry_count += 1
                    if retry_count <= max_retries:
                        logger.warning(f"Vector search failed, retrying ({retry_count}/{max_retries}): {search_error}")
                        await asyncio.sleep(1)  # Wait before retry
                    else:
                        logger.error(f"Vector search failed after {max_retries} retries: {search_error}")
                        raise  # Re-raise to be caught by outer exception handler
            
            # Process results
            return [
                RetrievalResult(
                    content=result["content"],
                    source=result.get("source", "Vector Store"),
                    relevance=result.get("score", 0.0),
                    metadata=result.get("metadata", {})
                )
                for result in results
                if result.get("content")  # Filter out empty results
            ]
            
        except Exception as e:
            logger.error(f"Error retrieving from vector store: {e}")
            # Return empty list instead of raising to ensure graceful degradation
            return []
    
    async def _retrieve_from_ptolemy(self, query: str, concept_id: str) -> List[RetrievalResult]:
        """
        Retrieve knowledge from Ptolemy.
        
        Args:
            query: The query to retrieve knowledge for
            concept_id: ID of the concept
            
        Returns:
            List of retrieval results
        """
        try:
            # Concurrent retrieval for better performance
            tasks = []
            
            # Get the concept
            concept_task = asyncio.create_task(self.ptolemy.get_concept(concept_id))
            tasks.append(concept_task)
            
            # Get related concepts concurrently
            related_task = asyncio.create_task(self.ptolemy.get_related_concepts(concept_id))
            tasks.append(related_task)
            
            # Wait for all tasks to complete
            await asyncio.gather(*tasks)
            
            # Get results
            concept = await concept_task
            related = await related_task
            
            if not concept:
                logger.warning(f"Concept not found in Ptolemy: {concept_id}")
                return []
            
            # Create retrieval results
            results = []
            
            # Add primary concept with valid check
            concept_desc = concept.get("description", "")
            if concept_desc and len(concept_desc) > 20:  # Ensure meaningful content
                results.append(
                    RetrievalResult(
                        content=concept_desc,
                        source=f"Ptolemy Concept: {concept.get('name', concept_id)}",
                        relevance=1.0,
                        metadata={
                            "concept_id": concept_id, 
                            "type": "primary_concept",
                            "category": concept.get("category", ""),
                            "properties": concept.get("properties", {})
                        }
                    )
                )
            
            # Add related concepts with relevance scoring
            for rel_concept in related:
                # Extract concept data either directly or from the nested structure
                if "concept" in rel_concept:
                    # New API format
                    rel_data = rel_concept.get("concept", {})
                    rel_id = rel_data.get("id")
                    rel_name = rel_data.get("name", rel_id)
                    rel_desc = rel_data.get("description", "")
                    # Get relationship data
                    rel_info = rel_concept.get("relationship", {})
                    rel_type = rel_info.get("type", "related")
                    rel_direction = rel_info.get("direction", "unknown")
                    rel_strength = rel_info.get("strength", 0.5)
                else:
                    # Older API format
                    rel_id = rel_concept.get("concept_id")
                    rel_name = rel_concept.get("name", rel_id)
                    rel_desc = rel_concept.get("description", "")
                    rel_type = rel_concept.get("relationship_type", "related")
                    rel_direction = "unknown"
                    rel_strength = 0.5
                
                # Skip if no meaningful description
                if not rel_desc or len(rel_desc) < 20 or not rel_id:
                    continue
                
                # Calculate relevance based on relationship type and strength
                base_relevance = 0.7  # Default for related concepts
                if rel_type == "prerequisite":
                    base_relevance = 0.85  # Higher relevance for prerequisites
                elif rel_type == "example_of":
                    base_relevance = 0.8  # Higher relevance for examples
                
                # Adjust by explicit relationship strength if available
                relevance = base_relevance * rel_strength
                
                results.append(
                    RetrievalResult(
                        content=rel_desc,
                        source=f"Ptolemy Related Concept ({rel_type}): {rel_name}",
                        relevance=relevance,
                        metadata={
                            "concept_id": rel_id, 
                            "type": "related_concept", 
                            "relationship": rel_type,
                            "direction": rel_direction,
                            "strength": rel_strength
                        }
                    )
                )
            
            # Sort by relevance
            results.sort(key=lambda x: x.relevance, reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving from Ptolemy: {e}")
            # Log stack trace for debugging
            logger.debug(f"Ptolemy retrieval exception details: {str(e)}", exc_info=True)
            return []
    
    def _prepare_retrieval_context(self, retrieval_results: List[RetrievalResult]) -> str:
        """
        Prepare retrieval context for content generation.
        
        Args:
            retrieval_results: List of retrieval results
            
        Returns:
            Formatted retrieval context
        """
        if not retrieval_results:
            return "No relevant information found."
        
        context_parts = ["Here is some relevant information:"]
        
        for i, result in enumerate(retrieval_results, 1):
            content = result.content.strip()
            source = result.source
            
            context_parts.append(f"[{i}] From {source}:\n{content}")
        
        return "\n\n".join(context_parts)
    
    def _create_enhanced_prompt(self, query: str, retrieval_context: str) -> str:
        """
        Create an enhanced prompt with retrieval context.
        
        Args:
            query: Original query
            retrieval_context: Formatted retrieval context
            
        Returns:
            Enhanced prompt
        """
        return f"""

{retrieval_context}

Using the information above, please respond to the following:

{query}

Please cite your sources using the numbers in square brackets, like [1], [2], etc.
"""
    
    def _extract_citations(self, content: str, retrieval_results: List[RetrievalResult]) -> List[Dict[str, Any]]:
        """
        Extract citations from generated content.
        
        Args:
            content: Generated content
            retrieval_results: List of retrieval results
            
        Returns:
            List of citations
        """
        citations = []
        
        # Improved citation extraction with smarter detection
        for i, result in enumerate(retrieval_results, 1):
            citation_marker = f"[{i}]"
            
            # Get all occurrences of this citation marker
            start_pos = 0
            while True:
                pos = content.find(citation_marker, start_pos)
                if pos == -1:
                    break
                
                # Extract surrounding context (50 chars before and after)
                context_start = max(0, pos - 50)
                context_end = min(len(content), pos + 50)
                citation_context = content[context_start:context_end]
                
                # Add to citations if not already present with this context
                context_hash = hash(citation_context)
                if not any(c.get("context_hash") == context_hash for c in citations):
                    citations.append({
                        "citation_number": i,
                        "source": result.source,
                        "content_snippet": result.content[:100] + "..." if len(result.content) > 100 else result.content,
                        "citation_context": citation_context,
                        "context_hash": context_hash,
                        "position": pos
                    })
                
                start_pos = pos + len(citation_marker)
        
        # Sort citations by position in the content
        citations.sort(key=lambda x: x.get("position", 0))
        
        # Remove technical fields used for processing
        for citation in citations:
            if "context_hash" in citation:
                del citation["context_hash"]
            if "position" in citation:
                del citation["position"]
        
        return citations