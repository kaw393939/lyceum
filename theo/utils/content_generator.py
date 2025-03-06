"""
Content generator utility for Lyceum using Anthropic Claude 3.7 API.
Generates educational content for the platform.
"""

import os
import json
import re
import time
import requests
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import logging
from datetime import datetime
from dotenv import load_dotenv
import anthropic

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class ContentGenerator:
    """Anthropic Claude content generator for Lyceum educational content."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the content generator.
        
        Args:
            api_key: Anthropic API key (optional, will use environment variable if not provided)
        """
        # Load API key from environment variable if not provided
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            logger.error("No Anthropic API key available")
            raise ValueError("No Anthropic API key provided. Set ANTHROPIC_API_KEY environment variable.")
        
        # Initialize with secure key handling - mask key in logs or error messages
        masked_key = self.api_key[:4] + "..." + self.api_key[-4:] if len(self.api_key) > 8 else "******"
        logger.info(f"Initializing content generator with API key: {masked_key}")
        
        try:
            self.client = anthropic.Anthropic(api_key=self.api_key)
            # Set the model to use
            self.model = "claude-3-7-sonnet-20250219"
        except Exception as e:
            logger.error("Error initializing Anthropic client")
            # Don't expose the API key in the error message
            raise RuntimeError(f"Error initializing Anthropic client: {str(e).replace(self.api_key, '[API_KEY]')}")
        
        # Create content directory if it doesn't exist
        self.content_dir = Path(__file__).parent.parent / "content"
        self.content_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_content(self, 
                       prompt: str, 
                       filename: str,
                       max_tokens: int = 4000,
                       temperature: float = 0.7) -> str:
        """
        Generate content using Claude 3.7.
        
        Args:
            prompt: Prompt for content generation
            filename: Base filename to save the content (without extension)
            max_tokens: Maximum tokens in the response
            temperature: Creativity of the response (0.0-1.0)
            
        Returns:
            Generated content text
        """
        # Input validation
        if not prompt or not filename:
            logger.error("Empty prompt or filename provided")
            raise ValueError("Prompt and filename must be provided")
            
        # Sanitize filename to prevent path traversal
        safe_filename = self._sanitize_filename(filename)
        if safe_filename != filename:
            logger.warning(f"Filename sanitized from '{filename}' to '{safe_filename}'")
            filename = safe_filename
            
        # Validate parameters
        if not (0.0 <= temperature <= 1.0):
            logger.warning(f"Invalid temperature value {temperature}, using default 0.7")
            temperature = 0.7
            
        if max_tokens < 1 or max_tokens > 100000:
            logger.warning(f"Invalid max_tokens value {max_tokens}, using default 4000")
            max_tokens = 4000
        
        logger.info(f"Generating content with prompt length: {len(prompt)} characters")
        
        # Rate limiting - basic implementation
        self._check_rate_limit()
        
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system="You are an expert educational content creator with deep knowledge of philosophy, pedagogy, and learning science. Create engaging, thoughtful, and intellectually rich content for the Lyceum educational platform.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Validate response
            if not message or not hasattr(message, 'content') or not message.content:
                logger.error("Empty or invalid response from API")
                raise ValueError("Failed to generate content: empty response")
                
            content = message.content[0].text
            
            # Save the content with timestamp
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            content_filename = f"{filename}_{timestamp}.md"
            content_path = self.content_dir / content_filename
            
            # Ensure directory exists
            content_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write to a temporary file first, then rename for atomic operation
            temp_path = content_path.with_suffix('.tmp')
            with open(temp_path, "w") as f:
                f.write(content)
                
            # Use os.replace for atomic operation
            os.replace(temp_path, content_path)
            
            logger.info(f"Content saved to {content_path}")
            
            # Save metadata with sanitized prompt (don't store full prompt in logs)
            metadata = {
                "prompt_length": len(prompt),
                "prompt_preview": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                "timestamp": datetime.now().isoformat(),
                "parameters": {
                    "model": self.model,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                },
                "path": str(content_path)
            }
            
            metadata_path = self.content_dir / f"{filename}_{timestamp}_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            
            return content
            
        except Exception as e:
            error_msg = str(e)
            # Ensure API key is not logged if it's in the error message
            if self.api_key and self.api_key in error_msg:
                error_msg = error_msg.replace(self.api_key, "[API_KEY]")
            logger.error(f"Error generating content: {error_msg}")
            # Propagate the error with sanitized message
            raise RuntimeError(f"Failed to generate content: {error_msg}")
            
    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename to prevent path traversal.
        
        Args:
            filename: Filename to sanitize
            
        Returns:
            Sanitized filename
        """
        # Remove path separators and dangerous characters
        return re.sub(r'[^\w\-\.]', '_', os.path.basename(filename))
        
    def _check_rate_limit(self) -> None:
        """
        Simple rate limiting to prevent API abuse.
        In a production system, this would be more sophisticated.
        """
        # This is a basic implementation - in production use a proper rate limiter
        current_time = time.time()
        if hasattr(self, '_last_request_time'):
            time_since_last = current_time - self._last_request_time
            if time_since_last < 1.0:  # Minimum 1 second between requests
                sleep_time = 1.0 - time_since_last
                logger.info(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
        self._last_request_time = current_time
    
    def generate_vision_content(self) -> str:
        """
        Generate vision content for the Lyceum homepage.
        
        Returns:
            Generated vision text
        """
        vision_prompt = """
        Create a visionary statement for the Lyceum educational platform that addresses the following:

        1. How Lyceum's approach to education differs from conventional platforms by emphasizing active learning, Socratic dialogue, and personalized mentorship.

        2. How this approach addresses known issues with educational technology, such as passive consumption, shallow engagement, and lack of critical thinking development.

        3. The potential impact of combining ancient philosophical methods with advanced AI to create a transformative learning experience.

        4. How Lyceum can help address current challenges in education related to motivation, deep understanding, and meaningful application of knowledge.

        5. A compelling vision of what education could look like in the future with this type of platform.

        The statement should be inspiring, intellectually rich, and forward-thinking. It should appeal to both educational philosophers and practical educators. Use vivid language that captures the transformative potential of this approach.
        
        Format the content with Markdown, using headers, bullet points, and other formatting to make it visually appealing and easy to read.
        """
        
        return self.generate_content(
            prompt=vision_prompt.strip(),
            filename="lyceum_vision",
            max_tokens=1500,
            temperature=0.7
        )
        
    def generate_technical_content(self) -> str:
        """
        Generate technical content for the technical architecture page.
        
        Returns:
            Generated technical text
        """
        technical_prompt = """
        Create comprehensive technical documentation for the Lyceum educational platform architecture. Include:

        1. High-level overview of the system architecture, explaining how different components interact
        
        2. Detailed description of each key component:
           - Knowledge Graph Service (using Neo4j and Qdrant)
           - Dialogue System (powered by Claude AI)
           - Content Engine (adaptive content generation)
           - Mentor Service (personalized learning guidance)
           - Learning Path Service (optimized educational journeys)
        
        3. Explanation of data flows between components
        
        4. Key technical innovations that enable the platform's unique capabilities
        
        5. Scalability, security, and performance considerations
        
        Make the documentation technically precise but also accessible to technical decision-makers who may not be AI specialists. Include references to specific technologies where appropriate.
        
        Format the content with Markdown, using headers, bullet points, code blocks, and other formatting to make it visually appealing and easy to read. Include diagrams descriptions that could be implemented as mermaid.js diagrams.
        """
        
        return self.generate_content(
            prompt=technical_prompt.strip(),
            filename="lyceum_technical",
            max_tokens=2500,
            temperature=0.7
        )
        
    def generate_business_content(self) -> str:
        """
        Generate business content for the business strategy page.
        
        Returns:
            Generated business text
        """
        business_prompt = """
        Create a compelling business strategy overview for the Lyceum educational platform, including:

        1. Market analysis - size of the educational technology market, growth trends, and unfilled needs
        
        2. Target segments - specific educational sectors and institutions that would benefit most from Lyceum
        
        3. Value proposition - key differentiators from existing educational technology platforms
        
        4. Revenue model - how Lyceum will generate income (subscription tiers, implementation services, etc.)
        
        5. Go-to-market strategy - how Lyceum will reach initial customers and scale
        
        6. Competitive analysis - how Lyceum compares to alternatives
        
        7. Key metrics - what success looks like for the platform
        
        Make the content strategic, forward-thinking, and grounded in market realities. Use a professional business tone while still conveying the transformative potential of the platform.
        
        Format the content with Markdown, using headers, bullet points, and other formatting to make it visually appealing and easy to read. Include descriptions of charts or graphics that could be implemented visually.
        """
        
        return self.generate_content(
            prompt=business_prompt.strip(),
            filename="lyceum_business",
            max_tokens=2000,
            temperature=0.7
        )
        
    def generate_agile_content(self) -> str:
        """
        Generate agile development content for the agile page.
        
        Returns:
            Generated agile text
        """
        agile_prompt = """
        Create a detailed agile development roadmap and methodology overview for the Lyceum educational platform, including:

        1. Development philosophy - principles guiding the agile approach to building Lyceum
        
        2. Roadmap phases - key milestones and features planned across multiple development phases:
           - Phase 1: Foundation (Knowledge Graph & Core Systems)
           - Phase 2: Dialogue Integration
           - Phase 3: Content Engine
           - Phase 4: Mentor Service
           - Phase 5: Learning Paths & Full Integration
        
        3. Iterative development approach - how feedback and testing inform continuous improvement
        
        4. Team structure - how cross-functional teams collaborate (engineers, educators, designers)
        
        5. Technical practices - CI/CD, testing strategy, quality assurance
        
        Make the content practical and informative, demonstrating a sophisticated understanding of modern software development while emphasizing the unique challenges of educational technology.
        
        Format the content with Markdown, using headers, bullet points, and other formatting to make it visually appealing and easy to read. Include descriptions of a Gantt chart or timeline visualization that could be implemented.
        """
        
        return self.generate_content(
            prompt=agile_prompt.strip(),
            filename="lyceum_agile",
            max_tokens=2000,
            temperature=0.7
        )
        
    def generate_contact_content(self) -> str:
        """
        Generate contact page content.
        
        Returns:
            Generated contact text
        """
        contact_prompt = """
        Create engaging content for the Lyceum contact page, including:

        1. A brief, compelling introduction explaining why organizations should connect with Lyceum
        
        2. 3-4 specific reasons to contact, such as:
           - Learning more about the platform
           - Scheduling a demonstration
           - Discussing implementation
           - Exploring partnerships
        
        3. Additional information about the team and company (2-3 short paragraphs)
        
        Keep the content warm, professional, and concise. Emphasize the team's expertise and passion for educational innovation.
        
        Format the content with Markdown, using headers and minimal formatting.
        """
        
        return self.generate_content(
            prompt=contact_prompt.strip(),
            filename="lyceum_contact",
            max_tokens=800,
            temperature=0.7
        )
        
    def generate_all_page_content(self) -> Dict[str, str]:
        """
        Generate content for all main pages.
        
        Returns:
            Dictionary mapping page names to content
        """
        content = {
            "vision": self.generate_vision_content(),
            "technical": self.generate_technical_content(),
            "business": self.generate_business_content(),
            "agile": self.generate_agile_content(),
            "contact": self.generate_contact_content()
        }
        
        return content


if __name__ == "__main__":
    # Example usage
    generator = ContentGenerator()
    content = generator.generate_vision_content()
    print(f"Generated content length: {len(content)} characters")