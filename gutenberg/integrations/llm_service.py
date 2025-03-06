"""
Gutenberg Content Generation System - LLM Service
================================================
Provides a service for generating content using large language models.
Supports multiple providers and models with consistent interfaces.
"""

import logging
import time
import json
import asyncio
import httpx
import random
import os
from typing import Dict, List, Optional, Any, Union
import re
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from pydantic import BaseModel, Field
from config.settings import get_config
from utils.logging_utils import get_logger

# Import appropriate libraries for different providers
try:
    from openai import AsyncOpenAI, RateLimitError as OpenAIRateLimitError
except ImportError:
    print("Warning: OpenAI library not installed. OpenAI functionality will be unavailable.")
    # Define placeholder classes/exceptions for type checking
    class AsyncOpenAI:
        pass
    class OpenAIRateLimitError(Exception):
        pass

try:
    from anthropic import Anthropic, AsyncAnthropic, RateLimitError as AnthropicRateLimitError
except ImportError:
    print("Warning: Anthropic library not installed. Anthropic functionality will be unavailable.")
    # Define placeholder classes/exceptions for type checking
    class Anthropic:
        pass
    class AsyncAnthropic:
        pass
    class AnthropicRateLimitError(Exception):
        pass

logger = get_logger(__name__)


class LLMResponse(BaseModel):
    """Model for LLM API responses."""
    content: str = Field(..., description="Generated content")
    structured_output: Optional[Dict[str, Any]] = Field(None, description="Structured output if requested")
    prompt_tokens: Optional[int] = Field(None, description="Number of prompt tokens used")
    completion_tokens: Optional[int] = Field(None, description="Number of completion tokens used")
    total_tokens: Optional[int] = Field(None, description="Total tokens used")
    model: Optional[str] = Field(None, description="Model used")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class LLMService:
    """
    Service for generating content using large language models.
    Supports multiple providers (OpenAI, Anthropic, etc.) and
    manages context handling, rate limiting, and error handling.
    """
    
    def __init__(self):
        """Initialize the LLM service with configuration."""
        self.config = get_config()
        
        # Track API usage for rate limiting
        self.usage_tracker = {
            "total_tokens": 0,
            "requests": 0,
            "last_request_time": 0,
            "minute_window": {
                "tokens": 0,
                "requests": 0,
                "start_time": time.time()
            }
        }
        
        # LLM client instances (lazy-loaded)
        self._clients = {}
        
        # Default model settings
        self.default_model = self.config.get("llm", {}).get("default_model", "gpt-4o")
        self.default_provider = self.config.get("llm", {}).get("default_provider", "openai")
        self.structured_output_model = self.config.get("llm", {}).get("structured_output_model", "gpt-4o")
        self.structured_output_provider = self.config.get("llm", {}).get("structured_output_provider", "openai")
        self.long_context_model = self.config.get("llm", {}).get("long_context_model", "gpt-4o-32k")
        self.long_context_provider = self.config.get("llm", {}).get("long_context_provider", "openai")
        
        logger.info("LLMService initialized")
        
    async def check_connection(self) -> bool:
        """
        Check if the LLM service is available and properly configured.
        
        Returns:
            True if connection is successful, False otherwise
        """
        logger.info("Checking LLM service connection")
        
        try:
            # Try to initialize the default client
            provider = self.default_provider
            
            if provider == "openai":
                # Check for OpenAI API key
                api_key = os.environ.get("OPENAI_API_KEY") or self.config.get("openai", {}).get("api_key", "")
                if not api_key:
                    logger.warning("OpenAI API key not found")
                    return False
                
                # Initialize client (lightweight operation, doesn't make API call)
                client = AsyncOpenAI(api_key=api_key)
                
                # Make a simple models list call to verify connection
                models = await client.models.list()
                if not models or not models.data:
                    logger.warning("Failed to retrieve OpenAI models")
                    return False
                
                logger.info(f"Successfully connected to OpenAI API. Available models: {len(models.data)}")
                return True
                
            elif provider == "anthropic":
                # Check for Anthropic API key
                api_key = os.environ.get("ANTHROPIC_API_KEY") or self.config.get("anthropic", {}).get("api_key", "")
                if not api_key:
                    logger.warning("Anthropic API key not found")
                    return False
                
                # Initialize client 
                client = AsyncAnthropic(api_key=api_key)
                
                # For Anthropic, we'll attempt a very small completion to check connectivity
                response = await client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=1,
                    messages=[
                        {"role": "user", "content": "Hello"}
                    ]
                )
                
                if not response or not response.content:
                    logger.warning("Failed to get response from Anthropic API")
                    return False
                
                logger.info("Successfully connected to Anthropic API")
                return True
                
            else:
                logger.warning(f"Unsupported provider: {provider}")
                return False
            
        except Exception as e:
            logger.error(f"Error checking LLM service connection: {str(e)}")
            return False
    
    @retry(
        stop=stop_after_attempt(3), 
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((OpenAIRateLimitError, AnthropicRateLimitError))
    )
    async def generate_content(self, 
                         prompt: str, 
                         system_message: Optional[str] = None,
                         temperature: float = 0.7,
                         max_tokens: int = 1000,
                         output_format: Optional[Union[str, Dict[str, Any]]] = None) -> LLMResponse:
        """
        Generate content using LLM.
        
        Args:
            prompt: The user prompt for generation
            system_message: Optional system message for context
            temperature: Creativity parameter (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            output_format: Optional format for JSON structured output
            
        Returns:
            LLMResponse with generated content and metadata
        """
        start_time = time.time()
        logger.info(f"Generating content: {prompt[:50]}... (temp={temperature}, max_tokens={max_tokens})")
        
        try:
            # Apply rate limiting if enabled
            await self._apply_rate_limiting()
            
            # Select the appropriate model based on config and request
            format_str = output_format if isinstance(output_format, str) else "json" if output_format else None
            model_info = self._select_model(prompt, max_tokens, format_str)
            model_name = model_info["name"]
            provider = model_info["provider"]
            
            # Get or create client for the selected provider
            client = await self._get_client(provider)
            
            # Prepare the request
            request = self._prepare_request(
                prompt=prompt,
                system_message=system_message,
                temperature=temperature,
                max_tokens=max_tokens,
                output_format=output_format,
                model=model_name,
                provider=provider
            )
            
            # Generate content using the appropriate client
            if provider == "openai":
                result = await self._generate_openai(client, request)
            elif provider == "anthropic":
                result = await self._generate_anthropic(client, request)
            else:
                logger.error(f"Unknown provider: {provider}")
                raise ValueError(f"Unsupported provider: {provider}")
            
            # Create LLMResponse
            response = LLMResponse(
                content=result.get("content", ""),
                structured_output=result.get("structured_output"),
                prompt_tokens=result.get("prompt_tokens"),
                completion_tokens=result.get("completion_tokens"),
                total_tokens=result.get("total_tokens"),
                model=model_name,
                metadata={
                    "processing_time": time.time() - start_time,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "provider": provider
                }
            )
            
            # Update usage statistics
            self._update_usage(request, result)
            
            logger.info(f"Content generated successfully in {time.time() - start_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"Error generating content: {str(e)}")
            
            # Create error response
            error_response = LLMResponse(
                content=f"Error generating content: {str(e)}",
                structured_output={"error": str(e)} if output_format else None,
                model=self.default_model,
                metadata={"error": str(e), "processing_time": time.time() - start_time}
            )
            
            return error_response
    
    async def summarize(self, 
                       text: str, 
                       max_length: int = 200,
                       focus: Optional[str] = None) -> LLMResponse:
        """
        Summarize text to a specified maximum length.
        
        Args:
            text: Text to summarize
            max_length: Target length in words
            focus: Optional focus for the summary
            
        Returns:
            LLMResponse with the summary
        """
        logger.info(f"Summarizing text: {text[:50]}... (max_length={max_length})")
        
        focus_text = f" focusing on {focus}" if focus else ""
        prompt = f"Summarize the following text in about {max_length} words or less{focus_text}:\n\n{text}"
        
        system_message = "You are a helpful assistant that creates clear, accurate summaries while preserving the key information and intent of the original text."
        
        return await self.generate_content(
            prompt=prompt,
            system_message=system_message,
            temperature=0.3,  # Lower temperature for factual summary
            max_tokens=max(100, int(max_length * 1.5))  # Allow some buffer
        )
    
    async def extract_key_points(self, text: str, num_points: int = 5) -> LLMResponse:
        """
        Extract key points from text.
        
        Args:
            text: Text to extract key points from
            num_points: Number of key points to extract
            
        Returns:
            LLMResponse with key points in structured output
        """
        logger.info(f"Extracting key points from text: {text[:50]}... (num_points={num_points})")
        
        prompt = f"Extract the {num_points} most important key points from the following text:\n\n{text}"
        
        system_message = "You are a helpful assistant that identifies and extracts the most important points from text."
        
        # Request JSON format for the output
        output_format = {"key_points": ["string"]}
        
        return await self.generate_content(
            prompt=prompt,
            system_message=system_message,
            temperature=0.3,  # Lower temperature for factual extraction
            max_tokens=int(len(text.split()) * 0.3),  # Roughly 30% of input length
            output_format=output_format
        )
    
    # -------- Private Helper Methods -------- #
    
    async def _apply_rate_limiting(self):
        """Apply rate limiting based on configuration."""
        # Check if rate limiting is enabled
        rate_limiting_enabled = self.config.get("llm", {}).get("rate_limiting_enabled", False)
        if not rate_limiting_enabled:
            return
        
        # Check if we're within the rate limits
        current_time = time.time()
        minute_window = self.usage_tracker["minute_window"]
        
        # Reset minute window if more than 60 seconds have passed
        if current_time - minute_window["start_time"] > 60:
            minute_window["tokens"] = 0
            minute_window["requests"] = 0
            minute_window["start_time"] = current_time
        
        # Check if we've exceeded the rate limits
        max_requests = self.config.get("llm", {}).get("max_requests_per_minute", 10)
        if minute_window["requests"] >= max_requests:
            # Wait until the minute window resets
            wait_time = 60 - (current_time - minute_window["start_time"])
            logger.warning(f"Rate limit reached. Waiting {wait_time:.2f}s before next request")
            await asyncio.sleep(wait_time)
            # Reset the window after waiting
            minute_window["tokens"] = 0
            minute_window["requests"] = 0
            minute_window["start_time"] = time.time()
    
    def _select_model(self, 
                    prompt: str, 
                    max_tokens: int, 
                    output_format: Optional[str]) -> Dict[str, str]:
        """
        Select appropriate model based on request characteristics.
        
        Args:
            prompt: User prompt
            max_tokens: Maximum output tokens
            output_format: Requested output format
            
        Returns:
            Dictionary with model name and provider
        """
        # For structured output, use a model that's good at following instructions
        if output_format:
            return {
                "name": self.structured_output_model,
                "provider": self.structured_output_provider
            }
        
        # For long outputs, use a model with higher context window
        prompt_tokens = len(prompt.split())
        if prompt_tokens + max_tokens > 4000:
            return {
                "name": self.long_context_model,
                "provider": self.long_context_provider
            }
        
        # Use default model for other cases
        return {
            "name": self.default_model,
            "provider": self.default_provider
        }
    
    async def _get_client(self, provider: str) -> Any:
        """
        Get or create client for the specified provider.
        
        Args:
            provider: LLM provider name
            
        Returns:
            Client instance
        """
        # Check if client already exists
        if provider in self._clients:
            return self._clients[provider]
        
        # Create new client based on provider
        if provider == "openai":
            # Initialize OpenAI client
            openai_api_key = os.environ.get("OPENAI_API_KEY", 
                                  self.config.get("llm", {}).get("openai_api_key", ""))
            
            if not openai_api_key:
                logger.error("OpenAI API key not found")
                raise ValueError("OpenAI API key not found in environment or config")
                
            client = AsyncOpenAI(
                api_key=openai_api_key,
                timeout=httpx.Timeout(30.0, connect=10.0)
            )
            self._clients[provider] = client
                
        elif provider == "anthropic":
            # Initialize Anthropic client
            anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", 
                                      self.config.get("llm", {}).get("anthropic_api_key", ""))
            
            if not anthropic_api_key:
                logger.error("Anthropic API key not found")
                raise ValueError("Anthropic API key not found in environment or config")
                
            client = AsyncAnthropic(
                api_key=anthropic_api_key,
                timeout=httpx.Timeout(30.0, connect=10.0)
            )
            self._clients[provider] = client
                
        else:
            # Unknown provider
            logger.error(f"Unknown provider {provider}")
            raise ValueError(f"Unsupported provider: {provider}")
        
        return self._clients[provider]
    
    def _prepare_request(self, 
                        prompt: str, 
                        system_message: Optional[str], 
                        temperature: float,
                        max_tokens: int,
                        output_format: Optional[Union[str, Dict[str, Any]]],
                        model: str,
                        provider: str) -> Dict[str, Any]:
        """
        Prepare request data for LLM API.
        
        Args:
            prompt: User prompt
            system_message: System message
            temperature: Temperature
            max_tokens: Maximum tokens
            output_format: Output format
            model: Model name
            provider: Provider name
            
        Returns:
            Request data
        """
        # Create format instructions if needed
        format_instructions = ""
        if output_format:
            if isinstance(output_format, dict):
                format_instructions = f"""
                Return a response in the following JSON format:
                {json.dumps(output_format, indent=2)}
                
                Your response should be valid JSON that conforms to this schema with no additional text.
                """
            elif isinstance(output_format, str) and "json" in output_format.lower():
                format_instructions = "Return your response as a valid JSON object with no additional text."
        
        # Combine system message with format instructions
        combined_system = system_message or "You are a helpful assistant."
        if format_instructions:
            combined_system += "\n\n" + format_instructions
        
        # Create request object based on provider
        if provider == "openai":
            return {
                "model": model,
                "messages": [
                    {"role": "system", "content": combined_system},
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "prompt": prompt,  # Keep original prompt for reference
                "system_message": system_message,  # Keep original system message for reference
                "output_format": output_format
            }
        elif provider == "anthropic":
            return {
                "model": model,
                "messages": [
                    {"role": "system", "content": combined_system},
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "original_prompt": prompt,  # Keep original prompt for reference
                "original_system": system_message,  # Keep original system message for reference
                "output_format": output_format
            }
        else:
            # Should never reach here due to check in _get_client
            raise ValueError(f"Unsupported provider: {provider}")
    
    async def _generate_openai(self, client, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate content using OpenAI API.
        
        Args:
            client: OpenAI client
            request: Request data
            
        Returns:
            Generated content and metadata
        """
        try:
            # Create completion using Chat API
            response = await client.chat.completions.create(
                model=request["model"],
                messages=request["messages"],
                temperature=request["temperature"],
                max_tokens=request["max_tokens"],
                response_format={"type": "json_object"} if request.get("output_format") else None
            )
            
            # Extract content from response
            content = response.choices[0].message.content
            
            # Try to parse structured output if needed
            structured_output = None
            if request.get("output_format"):
                try:
                    # Try to parse JSON from content
                    if content.strip().startswith("{") and content.strip().endswith("}"):
                        structured_output = json.loads(content)
                    else:
                        # Look for JSON code blocks
                        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
                        if json_match:
                            structured_output = json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    logger.warning("Failed to parse OpenAI response as JSON")
            
            # Prepare result
            return {
                "content": content,
                "structured_output": structured_output,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
        except Exception as e:
            logger.error(f"Error with OpenAI API: {str(e)}")
            raise
    
    async def _generate_anthropic(self, client, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate content using Anthropic API.
        
        Args:
            client: Anthropic client
            request: Request data
            
        Returns:
            Generated content and metadata
        """
        try:
            # Create completion
            response = await client.messages.create(
                model=request["model"],
                messages=request["messages"],
                temperature=request["temperature"],
                max_tokens=request["max_tokens"]
            )
            
            # Extract content
            content = response.content[0].text
            
            # Try to parse structured output if needed
            structured_output = None
            if request.get("output_format"):
                try:
                    # Try to parse JSON from content
                    if content.strip().startswith("{") and content.strip().endswith("}"):
                        structured_output = json.loads(content)
                    else:
                        # Look for JSON code blocks
                        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
                        if json_match:
                            structured_output = json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    logger.warning("Failed to parse Anthropic response as JSON")
            
            # Calculate approximate token counts (Anthropic v2 API doesn't provide token counts)
            # Rough approximation: 1 token ≈ 4 characters for English text
            prompt_tokens = sum(len(msg["content"]) // 4 for msg in request["messages"])
            completion_tokens = len(content) // 4
            total_tokens = prompt_tokens + completion_tokens
            
            # Prepare result with estimated token counts
            return {
                "content": content,
                "structured_output": structured_output,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }
            
        except Exception as e:
            logger.error(f"Error with Anthropic API: {str(e)}")
            raise
    
    def _update_usage(self, request: Dict[str, Any], response: Dict[str, Any]):
        """
        Update usage statistics.
        
        Args:
            request: Request data
            response: Response data
        """
        # Get token counts
        prompt_tokens = response.get("prompt_tokens", 0)
        completion_tokens = response.get("completion_tokens", 0)
        total_tokens = response.get("total_tokens", prompt_tokens + completion_tokens)
        
        # Update global stats
        self.usage_tracker["total_tokens"] += total_tokens
        self.usage_tracker["requests"] += 1
        self.usage_tracker["last_request_time"] = time.time()
        
        # Update minute window
        self.usage_tracker["minute_window"]["tokens"] += total_tokens
        self.usage_tracker["minute_window"]["requests"] += 1