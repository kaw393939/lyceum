"""
Media Generator
=============
Generator for media content (images, diagrams, charts, etc.) to accompany educational content.
"""

import logging
import base64
import uuid
import json
import os
import asyncio
from typing import Dict, List, Optional, Any, Union
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from pydantic import BaseModel, Field

from models.content import MediaType, MediaItem
from integrations.llm_service import LLMService
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class MediaRequest(BaseModel):
    """Model for a media generation request."""
    prompt: str = Field(..., description="Generation prompt")
    media_type: MediaType = Field(..., description="Type of media to generate")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    style: Optional[str] = Field(None, description="Style guidelines")
    size: Optional[str] = Field(None, description="Size/dimensions of the output")
    options: Dict[str, Any] = Field(default_factory=dict, description="Additional options")


class MediaResponse(BaseModel):
    """Model for a media generation response."""
    media_id: str = Field(..., description="Unique ID for the media item")
    media_type: MediaType = Field(..., description="Type of media generated")
    title: str = Field(..., description="Media title")
    description: str = Field(..., description="Media description")
    url: Optional[str] = Field(None, description="URL to the media (if external)")
    data: Optional[str] = Field(None, description="Base64 encoded data (if embedded)")
    alt_text: str = Field(..., description="Alt text for accessibility")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class MediaGenerator:
    """Generator for media content to accompany educational content."""
    
    def __init__(self):
        """Initialize the media generator."""
        self.llm_service = LLMService()
        self.client = httpx.AsyncClient(timeout=60.0)
        
        # Get API configuration from environment or config
        from config.settings import get_config
        self.config = get_config()
        
        # Get OpenAI API key from environment or config
        self.openai_api_key = os.environ.get("OPENAI_API_KEY", 
                                 self.config.get("openai", {}).get("api_key", ""))
        
        # Get image API configuration
        self.image_api_enabled = self.config.get("media_generator", {}).get("image_api_enabled", False)
        self.image_api_url = self.config.get("media_generator", {}).get("image_api_url", "https://api.openai.com/v1/images/generations")
        
        # Get chart configuration
        self.chart_service_url = self.config.get("media_generator", {}).get("chart_service_url", "")
        
        logger.info("Media generator initialized")
    
    async def generate_media(self, request: MediaRequest) -> MediaResponse:
        """
        Generate media based on a request.
        
        Args:
            request: Media generation request
            
        Returns:
            Generated media
        """
        logger.info(f"Generating {request.media_type.value} media: {request.prompt[:50]}...")
        
        # Generate a unique ID
        media_id = f"{request.media_type.value}-{uuid.uuid4()}"
        
        # Generate alt text
        alt_text = request.options.get("alt_text")
        if not alt_text:
            alt_text = await self._generate_alt_text(request)
        
        # Generate media based on type
        if request.media_type == MediaType.IMAGE:
            media_data = await self._generate_image(request)
        elif request.media_type == MediaType.DIAGRAM:
            media_data = await self._generate_diagram(request)
        elif request.media_type == MediaType.CHART:
            media_data = await self._generate_chart(request)
        elif request.media_type == MediaType.ILLUSTRATION:
            media_data = await self._generate_illustration(request)
        else:
            media_data = {}
            logger.warning(f"Unsupported media type: {request.media_type}")
        
        # Set defaults for missing data
        if not media_data:
            media_data = {}
        
        # Get or generate title
        title = request.options.get("title")
        if not title:
            title = f"{request.media_type.value.capitalize()} for {request.context.get('concept_name', 'concept')}"
        
        # Create response
        response = MediaResponse(
            media_id=media_id,
            media_type=request.media_type,
            title=title,
            description=request.prompt,
            url=media_data.get("url"),
            data=media_data.get("data"),
            alt_text=alt_text,
            metadata={
                "style": request.style,
                "size": request.size,
                "generation_prompt": request.prompt,
                "format": media_data.get("format", "unknown"),
                **request.options
            }
        )
        
        logger.info(f"Generated {request.media_type.value} media: {media_id}")
        return response
    
    async def _generate_alt_text(self, request: MediaRequest) -> str:
        """
        Generate alt text for a media item.
        
        Args:
            request: Media generation request
            
        Returns:
            Alt text for accessibility
        """
        try:
            # Use LLM to generate descriptive alt text
            prompt = f"""
            Create a concise but descriptive alt text for a {request.media_type.value} about the following concept:
            
            {request.prompt}
            
            The alt text should be informative for users with screen readers and about 1-2 sentences long.
            """
            
            response = await self.llm_service.generate_content(
                prompt=prompt,
                system_message="You are an assistant specialized in creating accessible alt text for educational content.",
                temperature=0.3,
                max_tokens=100
            )
            
            alt_text = response.content.strip()
            return alt_text
            
        except Exception as e:
            logger.error(f"Error generating alt text: {e}")
            # Fall back to a basic alt text
            return f"{request.media_type.value.capitalize()} related to {request.context.get('concept_name', 'the concept')}"
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def _generate_image(self, request: MediaRequest) -> Dict[str, Any]:
        """
        Generate an image using the OpenAI DALL-E API or similar service.
        
        Args:
            request: Media generation request
            
        Returns:
            Dictionary with image data or URL
        """
        try:
            # Check if image API is enabled
            if not self.image_api_enabled or not self.openai_api_key:
                logger.info("Image API disabled or API key missing, returning SVG placeholder")
                return await self._generate_placeholder_image(request)
            
            # Parse size
            size = request.size or "1024x1024"
            
            # Prepare prompt
            style_prompt = f" Style: {request.style}" if request.style else ""
            enhanced_prompt = f"{request.prompt}{style_prompt}"
            
            # Call image generation API
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.openai_api_key}"
            }
            
            payload = {
                "prompt": enhanced_prompt,
                "n": 1,
                "size": size,
                "response_format": "b64_json"
            }
            
            response = await self.client.post(
                self.image_api_url,
                headers=headers,
                json=payload
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Extract image data
            if "data" in result and len(result["data"]) > 0:
                if "b64_json" in result["data"][0]:
                    # Base64 encoded image
                    image_data = result["data"][0]["b64_json"]
                    return {
                        "data": image_data,
                        "format": "png",
                        "url": None
                    }
                elif "url" in result["data"][0]:
                    # URL to image
                    image_url = result["data"][0]["url"]
                    return {
                        "data": None,
                        "format": "png",
                        "url": image_url
                    }
            
            # If we get here, something went wrong
            logger.warning("Image API didn't return expected data format")
            return await self._generate_placeholder_image(request)
            
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            return await self._generate_placeholder_image(request)
    
    async def _generate_placeholder_image(self, request: MediaRequest) -> Dict[str, Any]:
        """
        Generate a placeholder SVG image when actual image generation is not available.
        
        Args:
            request: Media generation request
            
        Returns:
            Dictionary with SVG data
        """
        # Create a simple SVG placeholder with text from the prompt
        width = 800
        height = 600
        
        # Extract concept name or use first few words of prompt
        concept_name = request.context.get("concept_name", "")
        if not concept_name:
            # Take first 5 words of prompt
            words = request.prompt.split()
            concept_name = " ".join(words[:min(5, len(words))])
        
        # Create SVG with concept name and placeholder text
        svg = f"""
        <svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
            <rect width="100%" height="100%" fill="#f0f0f0"/>
            <text x="50%" y="45%" font-family="Arial" font-size="24" text-anchor="middle">
                {concept_name}
            </text>
            <text x="50%" y="55%" font-family="Arial" font-size="18" text-anchor="middle" fill="#666">
                {request.media_type.value.capitalize()} Placeholder
            </text>
        </svg>
        """
        
        # Convert to base64
        svg_bytes = svg.encode('utf-8')
        svg_base64 = base64.b64encode(svg_bytes).decode('utf-8')
        
        return {
            "data": svg_base64,
            "format": "svg+xml",
            "url": None
        }
    
    async def _generate_diagram(self, request: MediaRequest) -> Dict[str, Any]:
        """
        Generate a diagram using Mermaid or similar library.
        
        Args:
            request: Media generation request
            
        Returns:
            Dictionary with diagram data or URL
        """
        try:
            # Extract diagram type from options or default to flowchart
            diagram_type = request.options.get("diagram_type", "flowchart")
            
            # Extract subject domain for better context
            domain = request.context.get("domain", "")
            subject = request.context.get("subject", "")
            domain_context = f"Domain: {domain}" if domain else ""
            subject_context = f"Subject: {subject}" if subject else ""
            
            prompt = f"""
            Create a {diagram_type} diagram for the educational concept described below.
            Use Mermaid.js syntax. The diagram should be clear, educational, and illustrate the key relationships.
            
            {domain_context}
            {subject_context}
            Concept: {request.prompt}
            
            Your response should be ONLY valid Mermaid diagram code with no additional text.
            """
            
            response = await self.llm_service.generate_content(
                prompt=prompt,
                system_message="You are a diagram generation assistant that creates educational diagrams in Mermaid syntax. Return only the Mermaid diagram code with no explanations or additional text.",
                temperature=0.3,
                max_tokens=1000
            )
            
            # Extract mermaid code from response
            mermaid_code = response.content.strip()
            if "```mermaid" in mermaid_code:
                mermaid_code = mermaid_code.split("```mermaid")[1].split("```")[0].strip()
            elif "```" in mermaid_code:
                mermaid_code = mermaid_code.split("```")[1].strip()
            
            # Generate unique ID for the diagram
            diagram_id = f"diagram-{uuid.uuid4().hex[:8]}"
            
            return {
                "data": mermaid_code,
                "format": "mermaid",
                "diagram_type": diagram_type,
                "diagram_id": diagram_id
            }
            
        except Exception as e:
            logger.error(f"Error generating diagram: {e}")
            # Return a simple placeholder mermaid diagram
            return {
                "data": f"graph TD\n    A[{request.context.get('concept_name', 'Concept')}] --> B[Key Point 1]\n    A --> C[Key Point 2]\n    A --> D[Key Point 3]",
                "format": "mermaid",
                "diagram_type": "flowchart"
            }
    
    async def _generate_chart(self, request: MediaRequest) -> Dict[str, Any]:
        """
        Generate chart data for Chart.js or similar library.
        
        Args:
            request: Media generation request
            
        Returns:
            Dictionary with chart data
        """
        try:
            # Extract chart type from options or default to bar
            chart_type = request.options.get("chart_type", "bar")
            
            # Construct a detailed prompt based on the chart type
            chart_specific_instructions = ""
            if chart_type == "bar":
                chart_specific_instructions = """
                Create a bar chart with appropriate categories and values.
                The data should include 'labels' for the x-axis categories and 'datasets' with values.
                """
            elif chart_type == "line":
                chart_specific_instructions = """
                Create a line chart showing a trend or progression.
                The data should include 'labels' for the x-axis points and 'datasets' with values.
                Include at least 5-10 data points to show a meaningful trend.
                """
            elif chart_type == "pie":
                chart_specific_instructions = """
                Create a pie chart dividing a whole into categories.
                The data should include 'labels' for the segments and 'datasets' with values that sum to 100%.
                Limit to 3-7 segments for clarity.
                """
            elif chart_type == "radar":
                chart_specific_instructions = """
                Create a radar chart comparing multiple attributes.
                The data should include 'labels' for each attribute and 'datasets' with values for each comparison item.
                """
            
            prompt = f"""
            Create data for a {chart_type} chart for the educational concept described below.
            The data should be realistic, educational, and illustrate important quantitative aspects of the concept.
            {chart_specific_instructions}
            
            Concept: {request.prompt}
            
            Return the data as a valid JSON object with:
            - "labels": array of strings for categories/x-axis
            - "datasets": array of objects, each with "label" and "data" properties
            - "title": an informative title for the chart
            """
            
            response = await self.llm_service.generate_content(
                prompt=prompt,
                system_message="You are a data visualization assistant that creates educational chart data. Return only valid JSON that can be used with Chart.js.",
                temperature=0.3,
                max_tokens=1000,
                output_format={
                    "labels": ["string"],
                    "datasets": [{"label": "string", "data": [0]}],
                    "title": "string",
                    "options": {}
                }
            )
            
            # Extract chart data from response
            chart_data = None
            if response.structured_output:
                chart_data = response.structured_output
            else:
                try:
                    # Try to extract JSON from text
                    # Look for JSON block
                    content = response.content
                    if "```json" in content:
                        json_text = content.split("```json")[1].split("```")[0].strip()
                    elif "```" in content:
                        json_text = content.split("```")[1].strip()
                    else:
                        json_text = content.strip()
                    
                    chart_data = json.loads(json_text)
                except Exception as e:
                    logger.error(f"Error parsing chart data: {e}")
                    # Fallback to a simple structure
                    chart_data = {
                        "labels": ["Category 1", "Category 2", "Category 3"],
                        "datasets": [
                            {
                                "label": "Dataset 1",
                                "data": [10, 20, 30]
                            }
                        ],
                        "title": f"Chart for {request.context.get('concept_name', 'concept')}",
                        "options": {}
                    }
            
            # Add chart type to data
            chart_data["type"] = chart_type
            
            # Add colors if not present
            if "datasets" in chart_data and chart_data["datasets"]:
                colors = [
                    "#4285F4", "#34A853", "#FBBC05", "#EA4335",  # Google colors
                    "#3b5998", "#00acee", "#0e76a8", "#c4302b",  # Social media colors
                    "#6a0dad", "#3cb44b", "#ffe119", "#4363d8"   # More distinct colors
                ]
                
                for i, dataset in enumerate(chart_data["datasets"]):
                    color_index = i % len(colors)
                    if chart_type == "pie" or chart_type == "doughnut":
                        if "backgroundColor" not in dataset:
                            dataset["backgroundColor"] = colors[:len(dataset["data"])]
                    else:
                        if "backgroundColor" not in dataset:
                            dataset["backgroundColor"] = colors[color_index]
                        if "borderColor" not in dataset:
                            dataset["borderColor"] = colors[color_index]
            
            return {
                "data": json.dumps(chart_data),
                "format": "chart.js",
                "chart_type": chart_type,
                "title": chart_data.get("title")
            }
            
        except Exception as e:
            logger.error(f"Error generating chart: {e}")
            # Return simple fallback chart data
            fallback_data = {
                "type": "bar",
                "labels": ["Category 1", "Category 2", "Category 3"],
                "datasets": [
                    {
                        "label": "Data",
                        "data": [10, 20, 30],
                        "backgroundColor": "#4285F4"
                    }
                ],
                "title": f"Chart for {request.context.get('concept_name', 'concept')}"
            }
            
            return {
                "data": json.dumps(fallback_data),
                "format": "chart.js",
                "chart_type": "bar"
            }
    
    async def _generate_illustration(self, request: MediaRequest) -> Dict[str, Any]:
        """
        Generate an educational illustration.
        
        Args:
            request: Media generation request
            
        Returns:
            Dictionary with illustration data or URL
        """
        try:
            # Educational illustrations are a specialized type of image
            # Modify the prompt to emphasize the educational and illustrative aspects
            
            # Extract style from request or default to "clear, educational illustration"
            style = request.style or "clear, educational illustration with labeled parts"
            
            # Create enhanced prompt for educational illustration
            illustration_request = MediaRequest(
                prompt=f"Educational illustration of {request.prompt}. Make it clear, informative, and suitable for education. Include labels for important parts.",
                media_type=MediaType.IMAGE,
                context=request.context,
                style=style,
                size=request.size or "1024x1024",
                options=request.options
            )
            
            # Use the image generator with our enhanced request
            return await self._generate_image(illustration_request)
            
        except Exception as e:
            logger.error(f"Error generating illustration: {e}")
            # Fall back to a placeholder
            return await self._generate_placeholder_image(request)
            
    async def _generate_audio(self, request: MediaRequest) -> Dict[str, Any]:
        """
        Generate audio from text using a text-to-speech API.
        
        Args:
            request: Media generation request
            
        Returns:
            Dictionary with audio data or URL
        """
        try:
            # Check if audio API is enabled
            if not self.config.get("media_generator", {}).get("audio_api_enabled", False):
                logger.info("Audio API disabled, returning placeholder audio")
                return self._generate_placeholder_audio(request)
            
            # Extract voice parameters
            voice = request.options.get("voice", "default")
            speed = request.options.get("speed", 1.0)
            pitch = request.options.get("pitch", 0.0)
            
            # Use OpenAI TTS API if available
            if self.openai_api_key and self.config.get("media_generator", {}).get("use_openai_tts", False):
                logger.info(f"Generating audio with OpenAI TTS: {request.prompt[:50]}...")
                
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.openai_api_key}"
                }
                
                payload = {
                    "model": "tts-1",
                    "input": request.prompt,
                    "voice": voice if voice in ["alloy", "echo", "fable", "onyx", "nova", "shimmer"] else "nova",
                    "response_format": "mp3"
                }
                
                response = await self.client.post(
                    "https://api.openai.com/v1/audio/speech",
                    headers=headers,
                    json=payload
                )
                
                response.raise_for_status()
                
                # Get audio data
                audio_data = response.content
                
                # Convert to base64
                audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                
                return {
                    "data": audio_base64,
                    "format": "mp3",
                    "url": None
                }
            else:
                # Fall back to placeholder
                logger.info("OpenAI TTS not configured, returning placeholder audio")
                return self._generate_placeholder_audio(request)
                
        except Exception as e:
            logger.error(f"Error generating audio: {e}")
            return self._generate_placeholder_audio(request)
    
    def _generate_placeholder_audio(self, request: MediaRequest) -> Dict[str, Any]:
        """
        Generate a placeholder audio file with text description.
        
        Args:
            request: Media generation request
            
        Returns:
            Dictionary with placeholder audio data
        """
        # This is a very small mp3 file (1 second of silence)
        # In a real implementation, you might generate a tone or use a more 
        # sophisticated placeholder
        silent_mp3_base64 = "SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU4Ljc2LjEwMAAAAAAAAAAAAAAA//tQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAWGluZwAAAA8AAAACAAADpgCampqampqampqampqampqampqampqampqampqampqampqampqampqampqampqampqampqamp//////////////////////////////////////////////////////////////////8AAAAATGF2YzU4LjEzAAAAAAAAAAAAAAAAJAAAAAAAAAAAA6bd5iZgAAAAAAAAAAAAAAAAAAAAAP/7UEAAAAGkAIAALAMAAASAIAABHhAAACAAAjwAAASQAAABETEFNRTMuMTAwVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV//tQQAAACuQBHACwDAAAACPAAABFVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV"
        
        return {
            "data": silent_mp3_base64,
            "format": "mp3",
            "duration": 1.0,  # 1 second of silence
            "url": None
        }
    
    async def generate_from_template(self, template_spec: Dict[str, Any], context: Dict[str, Any]) -> Optional[MediaItem]:
        """
        Generate media from a template specification.
        
        Args:
            template_spec: Template specification
            context: Context data
            
        Returns:
            Generated media item or None if generation failed
        """
        try:
            # Replace placeholders in prompt
            prompt = template_spec.get("prompt", "")
            for key, value in context.items():
                placeholder = f"[{key}]"
                if isinstance(value, (dict, list)):
                    value_str = json.dumps(value)
                else:
                    value_str = str(value)
                prompt = prompt.replace(placeholder, value_str)
            
            # Create media request
            request = MediaRequest(
                prompt=prompt,
                media_type=MediaType(template_spec.get("media_type", "image")),
                context=context,
                style=template_spec.get("style"),
                size=template_spec.get("size"),
                options=template_spec.get("options", {})
            )
            
            # Generate media
            response = await self.generate_media(request)
            
            # Convert to MediaItem
            return MediaItem(
                media_id=response.media_id,
                media_type=response.media_type,
                title=response.title,
                description=response.description,
                url=response.url,
                data=response.data,
                alt_text=response.alt_text,
                metadata=response.metadata
            )
            
        except Exception as e:
            logger.error(f"Error generating media from template: {e}")
            return None