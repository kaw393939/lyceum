"""
RAG Service
==========
Retrieval-Augmented Generation service that combines knowledge from Ptolemy and Gutenberg.
Includes diagnostic information and service health monitoring.
"""

import os
import time
import json
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from openai import AsyncOpenAI

from ..clients.ptolemy_client import PtolemyClient
from ..clients.gutenberg_client import GutenbergClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_service")

class RAGService:
    """Service for Retrieval-Augmented Generation."""
    
    def __init__(self):
        """Initialize the RAG service."""
        # Initialize OpenAI client
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not self.openai_api_key:
            logger.warning("No OpenAI API key found. LLM functionality will be limited.")
            
        self.openai_client = AsyncOpenAI(
            api_key=self.openai_api_key
        )
        
        # Initialize content sources
        self.ptolemy_client = PtolemyClient()
        self.gutenberg_client = GutenbergClient()
        
        # Configure default parameters
        self.model = os.environ.get("MODEL", "gpt-4o")
        self.temperature = float(os.environ.get("RAG_TEMPERATURE", "0.7"))
        self.top_k = int(os.environ.get("RAG_TOP_K", "5"))
        
        # Initialize diagnostic information
        self.init_time = datetime.now().isoformat()
        self.service_status = {
            "initialized_at": self.init_time,
            "openai_api_key_available": bool(self.openai_api_key),
            "model": self.model,
            "services": {
                "ptolemy": {
                    "url": os.environ.get("PTOLEMY_URL", "http://ptolemy:8000"),
                    "mock_mode": os.environ.get("PTOLEMY_USE_MOCK", "true").lower() in ("true", "1"),
                    "health_status": "unknown",
                    "last_checked": None
                },
                "gutenberg": {
                    "url": os.environ.get("GUTENBERG_URL", "http://gutenberg:8001"),
                    "mock_mode": os.environ.get("GUTENBERG_USE_MOCK", "true").lower() in ("true", "1"),
                    "health_status": "unknown",
                    "last_checked": None
                }
            },
            "request_stats": {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "average_response_time": 0,
                "average_token_usage": 0
            }
        }
        
        # Setup health check task
        self.health_check_task = None
        
        # Start automatic health checks in a more controlled way
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                self.health_check_task = asyncio.create_task(self._periodic_health_check())
            else:
                # In environments where there's no running event loop (like Streamlit script first launch),
                # don't create the task to avoid warnings
                logger.info("No running event loop detected, skipping health check task creation")
        except RuntimeError:
            logger.info("No event loop running, skipping health check task creation")
        
    async def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant content from all sources.
        
        Args:
            query: Search query
            top_k: Maximum number of results to retrieve (per source)
            
        Returns:
            List of relevant content items
        """
        if top_k is None:
            top_k = self.top_k
            
        # Search in parallel from all sources
        results = await asyncio.gather(
            self.ptolemy_client.search_content(query, top_k),
            self.gutenberg_client.search_content(query, top_k)
        )
        
        # Combine and sort results by score
        combined_results = []
        for source_results in results:
            combined_results.extend(source_results)
            
        # Sort by score (descending)
        combined_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        # Return top results across all sources
        return combined_results[:top_k]
    
    async def generate_completion(self, 
                                user_message: str, 
                                system_prompt: str,
                                chat_history: List[Dict[str, str]],
                                retrieve: bool = True) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        """
        Generate a completion with RAG.
        
        Args:
            user_message: User's message/query
            system_prompt: System prompt for the LLM
            chat_history: Chat history as a list of message objects
            retrieve: Whether to perform retrieval for this message
            
        Returns:
            Tuple of (generated response, relevant sources, request metadata)
        """
        start_time = time.time()
        request_stats = {
            "start_time": start_time,
            "retrieve_enabled": retrieve,
            "query": user_message,
            "retrieval_stats": {
                "sources_count": 0,
                "retrieval_time": 0,
                "sources": []
            },
            "llm_stats": {
                "model": self.model,
                "temperature": self.temperature,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "llm_time": 0
            },
            "total_time": 0,
            "success": False
        }
        
        # Track request in metrics
        self.service_status["request_stats"]["total_requests"] += 1
        
        try:
            # Perform retrieval if enabled
            sources = []
            enhanced_system_prompt = system_prompt
            
            retrieval_start = time.time()
            if retrieve:
                # Get relevant content
                sources = await self.retrieve(user_message)
                
                if sources:
                    # Format sources for inclusion in prompt
                    sources_text = "\n\n".join([
                        f"SOURCE {i+1}: {item['content']}\nTITLE: {item['metadata'].get('title', 'Unknown')}\nTYPE: {item['metadata'].get('type', 'Unknown')}\nSOURCE: {item['metadata'].get('source', 'Unknown')}"
                        for i, item in enumerate(sources[:5])  # Limit to top 5 sources
                    ])
                    
                    # Append sources to system prompt
                    enhancement = f"""
                    
                    Use the following sources when responding to the user. Incorporate this information seamlessly without explicitly mentioning "according to source 1" etc.
                    
                    {sources_text}
                    
                    When using the provided sources:
                    1. Respond accurately, synthesizing information from the sources
                    2. If the sources contain contradictions, acknowledge them
                    3. If the sources don't contain information to answer the query, state that you don't have specific information on the topic
                    """
                    
                    enhanced_system_prompt = system_prompt + enhancement
            
            retrieval_end = time.time()
            retrieval_time = retrieval_end - retrieval_start
            
            # Update retrieval stats
            request_stats["retrieval_stats"]["sources_count"] = len(sources)
            request_stats["retrieval_stats"]["retrieval_time"] = retrieval_time
            request_stats["retrieval_stats"]["sources"] = [
                {"title": s["metadata"].get("title", "Unknown"), 
                 "source": s["metadata"].get("source", "Unknown"),
                 "score": s.get("score", 0.0)}
                for s in sources[:5]
            ]
            
            # Prepare messages for the API call
            messages = [
                {"role": "system", "content": enhanced_system_prompt}
            ]
            
            # Add chat history
            messages.extend(chat_history)
            
            # Generate completion
            llm_start = time.time()
            response = await self.openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature
            )
            llm_end = time.time()
            llm_time = llm_end - llm_start
            
            # Update LLM stats
            request_stats["llm_stats"]["prompt_tokens"] = response.usage.prompt_tokens
            request_stats["llm_stats"]["completion_tokens"] = response.usage.completion_tokens
            request_stats["llm_stats"]["total_tokens"] = response.usage.total_tokens
            request_stats["llm_stats"]["llm_time"] = llm_time
            
            # Update total time and success
            end_time = time.time()
            request_stats["total_time"] = end_time - start_time
            request_stats["success"] = True
            
            # Update service stats with running averages
            self._update_service_stats(request_stats)
            
            return response.choices[0].message.content, sources, request_stats
            
        except Exception as e:
            logger.error(f"Error generating completion: {str(e)}")
            
            # Update failure statistics
            end_time = time.time()
            request_stats["total_time"] = end_time - start_time
            request_stats["error"] = str(e)
            self.service_status["request_stats"]["failed_requests"] += 1
            
            # Return minimal response with error information
            return f"I encountered an error: {str(e)}", [], request_stats
    
    async def _periodic_health_check(self):
        """Run periodic health checks on services."""
        try:
            while True:
                await self._check_services_health()
                # Sleep for 60 seconds between checks
                await asyncio.sleep(60)
        except asyncio.CancelledError:
            # Handle cancellation gracefully
            logger.info("Health check task cancelled")
        except Exception as e:
            logger.error(f"Error in health check task: {str(e)}")
    
    async def _check_services_health(self):
        """Check the health of connected services."""
        # Check Ptolemy
        try:
            ptolemy_url = os.environ.get("PTOLEMY_URL", "http://ptolemy:8000")
            ptolemy_health_url = f"{ptolemy_url}/health"
            
            async with self.ptolemy_client._create_client() as client:
                response = await client.get(ptolemy_health_url, timeout=5.0)
                if response.status_code == 200:
                    self.service_status["services"]["ptolemy"]["health_status"] = "healthy"
                else:
                    self.service_status["services"]["ptolemy"]["health_status"] = f"unhealthy ({response.status_code})"
        except Exception as e:
            self.service_status["services"]["ptolemy"]["health_status"] = f"error ({str(e)})"
        
        self.service_status["services"]["ptolemy"]["last_checked"] = datetime.now().isoformat()
        
        # Check Gutenberg
        try:
            gutenberg_url = os.environ.get("GUTENBERG_URL", "http://gutenberg:8001")
            gutenberg_health_url = f"{gutenberg_url}/health"
            
            async with self.gutenberg_client._create_client() as client:
                response = await client.get(gutenberg_health_url, timeout=5.0)
                if response.status_code == 200:
                    self.service_status["services"]["gutenberg"]["health_status"] = "healthy"
                else:
                    self.service_status["services"]["gutenberg"]["health_status"] = f"unhealthy ({response.status_code})"
        except Exception as e:
            self.service_status["services"]["gutenberg"]["health_status"] = f"error ({str(e)})"
        
        self.service_status["services"]["gutenberg"]["last_checked"] = datetime.now().isoformat()
    
    def _update_service_stats(self, request_stats: Dict[str, Any]):
        """Update service statistics with new request data."""
        stats = self.service_status["request_stats"]
        
        # Update success count
        if request_stats["success"]:
            stats["successful_requests"] += 1
        
        # Update rolling averages
        n = stats["total_requests"]
        
        # Update average response time
        current_avg_time = stats["average_response_time"]
        new_time = request_stats["total_time"]
        stats["average_response_time"] = (current_avg_time * (n-1) + new_time) / n
        
        # Update average token usage if available
        if "llm_stats" in request_stats and "total_tokens" in request_stats["llm_stats"]:
            current_avg_tokens = stats["average_token_usage"]
            new_tokens = request_stats["llm_stats"]["total_tokens"]
            stats["average_token_usage"] = (current_avg_tokens * (n-1) + new_tokens) / n
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information about service status and performance."""
        # Add current time to the diagnostics
        diagnostics = dict(self.service_status)
        diagnostics["current_time"] = datetime.now().isoformat()
        diagnostics["uptime_seconds"] = (datetime.now() - datetime.fromisoformat(self.init_time)).total_seconds()
        
        return diagnostics
        
    async def cleanup(self):
        """Clean up resources and cancel background tasks."""
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                # Wait for task to be cancelled
                await self.health_check_task
            except asyncio.CancelledError:
                # This is expected
                pass
    
    async def start_health_check(self):
        """Start the health check task if not already running."""
        if not self.health_check_task:
            self.health_check_task = asyncio.create_task(self._periodic_health_check())