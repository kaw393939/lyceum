"""
Image generator utility for Lyceum using OpenAI API.
Generates images for the sage/magician archetype branding.
"""

import os
import json
import base64
import requests
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging
from datetime import datetime
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class ImageGenerator:
    """OpenAI image generator for Lyceum branding."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the image generator.
        
        Args:
            api_key: OpenAI API key (optional, will use environment variable if not provided)
        """
        # Load API key from environment variable if not provided
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.error("No OpenAI API key available")
            raise ValueError("No OpenAI API key provided. Set OPENAI_API_KEY environment variable.")
            
        # Initialize with secure key handling - mask key in logs or error messages
        masked_key = self.api_key[:4] + "..." + self.api_key[-4:] if len(self.api_key) > 8 else "******"
        logger.info(f"Initializing image generator with API key: {masked_key}")
        
        self.api_url = "https://api.openai.com/v1/images/generations"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # Create images directory if it doesn't exist
        self.images_dir = Path(__file__).parent.parent / "static" / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_image(self, 
                       prompt: str, 
                       filename: str, 
                       size: str = "1024x1024", 
                       quality: str = "standard", 
                       style: str = "vivid",
                       n: int = 1) -> List[str]:
        """
        Generate an image using OpenAI API.
        
        Args:
            prompt: Description of the image to generate
            filename: Base filename to save the image (without extension)
            size: Image size (256x256, 512x512, or 1024x1024)
            quality: Image quality (standard or hd)
            style: Image style (vivid or natural)
            n: Number of images to generate
            
        Returns:
            List of paths to the saved images
        """
        logger.info(f"Generating image with prompt: {prompt}")
        
        data = {
            "model": "dall-e-3",  # Using the latest DALL-E 3 model
            "prompt": prompt,
            "n": n,
            "size": size,
            "quality": quality,
            "style": style,
            "response_format": "url"
        }
        
        try:
            response = requests.post(self.api_url, headers=self.headers, json=data)
            response.raise_for_status()
            result = response.json()
            
            image_paths = []
            for i, image_data in enumerate(result.get("data", [])):
                image_url = image_data.get("url")
                if image_url:
                    # Download the image
                    image_response = requests.get(image_url)
                    image_response.raise_for_status()
                    
                    # Save the image with timestamp
                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                    image_filename = f"{filename}_{timestamp}_{i}.png"
                    image_path = self.images_dir / image_filename
                    
                    with open(image_path, "wb") as f:
                        f.write(image_response.content)
                    
                    logger.info(f"Image saved to {image_path}")
                    image_paths.append(str(image_path))
            
            # Save prompt and response metadata
            metadata = {
                "prompt": prompt,
                "timestamp": datetime.now().isoformat(),
                "parameters": {
                    "size": size,
                    "quality": quality,
                    "style": style,
                    "n": n
                },
                "paths": image_paths
            }
            
            metadata_path = self.images_dir / f"{filename}_{timestamp}_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            
            return image_paths
            
        except Exception as e:
            error_msg = str(e)
            # Ensure API key is not logged if it's in the error message
            if self.api_key and self.api_key in error_msg:
                error_msg = error_msg.replace(self.api_key, "[API_KEY]")
            logger.error(f"Error generating image: {error_msg}")
            # Propagate the error with sanitized message
            raise RuntimeError(f"Failed to generate image: {error_msg}")
    
    def generate_sage_images(self) -> List[str]:
        """
        Generate a set of sage/magician archetype images for Lyceum branding.
        
        Returns:
            List of paths to the generated images
        """
        prompts = [
            "A modern, stylized illustration of a sage/philosopher in a digital learning environment. The sage appears wise and enlightened, surrounded by floating holographic symbols of knowledge and wisdom. Use a palette of deep blues, purples, and gold accents. The style should be elegant, minimalist, and slightly mystical.",
            
            "A magical educational environment with ancient Greek architecture blending seamlessly with futuristic technology. A wise mentor figure guides students through interactive holograms of knowledge. The scene should evoke the feeling of an ancient Lyceum reimagined for the digital age. Use warm, golden lighting with accents of blue energy.",
            
            "An abstract, geometric representation of knowledge transfer showing the archetype of 'The Magician' as an AI educational guide. The design should be clean, modern, and sophisticated with glowing neural network patterns connecting various symbols of learning. Use a color scheme of deep indigo, teal, and gold highlights against a dark background.",
            
            "A minimalist logo design for an AI educational system named 'Lyceum'. The logo should subtly incorporate elements of both ancient wisdom (like an owl or Greek column) and modern AI (like neural networks or circuit patterns). The design should be elegant, balanced, and work well at various sizes. Use a sophisticated color palette of deep blue, gold, and white."
        ]
        
        all_paths = []
        for i, prompt in enumerate(prompts):
            image_paths = self.generate_image(
                prompt=prompt,
                filename=f"lyceum_brand_{i+1}",
                size="1024x1024",
                quality="standard",
                style="vivid"
            )
            all_paths.extend(image_paths)
        
        return all_paths
        
    def generate_page_images(self) -> Dict[str, List[str]]:
        """
        Generate a complete set of images for all website pages.
        
        Returns:
            Dictionary mapping page names to lists of image paths
        """
        page_prompts = {
            "vision": [
                "Create an inspiring image representing the vision of the Lyceum educational platform. Show a scene that blends ancient Greek learning environments with futuristic AI technology. Include subtle elements of dialogue, knowledge visualization, and mentorship. Use a palette of deep blues and gold, with a bright, hopeful atmosphere.",
                
                "A visual representation of educational transformation, showing the evolution from traditional learning to Lyceum's approach. On one side, show conventional education (passive, standardized); on the other side, show Lyceum's vision (interactive, personalized, dialogue-based). Use visual metaphors that convey depth vs. surface learning.",
                
                "A captivating architectural visualization blending ancient Greek academy design with futuristic learning spaces. Show an open, luminous environment with areas for dialogue, exploration, and reflection. Include subtle holographic knowledge visualizations and AI mentor presences. Use a warm, inviting color palette with gold accents."
            ],
            
            "technical": [
                "A sophisticated technical diagram visualizing the Lyceum educational platform architecture. Show the interconnected components: knowledge graph, dialogue system, content engine, mentor service, and learning paths. Use a clean, professional style with a blue and gold color scheme. Label components clearly and show data flows between them.",
                
                "A detailed visualization of the knowledge graph component of the Lyceum platform. Show an elegant network structure representing interconnected concepts, with nodes and edges forming a beautiful pattern. Use a deep blue background with glowing gold/white nodes and connections. Make it visually striking while conveying technical sophistication.",
                
                "A visualization of the dialogue system component of Lyceum. Show how the system facilitates Socratic questioning between an AI mentor and a learner. Include visual elements representing intent recognition, context awareness, and knowledge retrieval. Use a clean, modern style with blue, purple and gold accents."
            ],
            
            "business": [
                "Create a professional image representing the business potential of the Lyceum educational platform. Show elements of market growth, educational transformation, and technological innovation. Use charts or graphics that suggest expanding adoption and impact. Style should be clean, professional, with the Lyceum blue and gold palette.",
                
                "A visual representation of Lyceum's target markets and adoption strategy. Show different sectors (higher education, corporate training, lifelong learning) connected to the central Lyceum platform. Use infographic elements to convey market opportunity and strategic positioning. Style should be business-appropriate but with the distinctive Lyceum aesthetic.",
                
                "An elegant visualization of the Lyceum business model and value proposition. Show how the platform creates value through its innovative approach to education. Include elements representing monetization streams, competitive advantages, and customer benefits. Use a sophisticated design with the Lyceum color palette."
            ],
            
            "agile": [
                "A clean, professional visualization of the Lyceum development roadmap. Show key milestones and features planned across multiple phases. Use a modern, minimalist style with the Lyceum color palette. Include visual elements representing the iterative, agile approach to platform development.",
                
                "An image representing agile development methodologies used in building the Lyceum platform. Show sprint cycles, user stories, and continuous improvement processes. Use visual metaphors of growth and evolution. Style should be professional but dynamic, with the Lyceum brand colors.",
                
                "A visualization of the Lyceum team collaboration model. Show how different specialties (AI engineers, educational designers, content experts) work together in an agile framework. Use an elegant, professional style with subtle Greek-inspired elements and the Lyceum color palette."
            ]
        }
        
        results = {}
        
        for page, prompts in page_prompts.items():
            page_paths = []
            for i, prompt in enumerate(prompts):
                image_paths = self.generate_image(
                    prompt=prompt,
                    filename=f"lyceum_{page}_{i+1}",
                    size="1024x768",
                    quality="standard",
                    style="vivid"
                )
                page_paths.extend(image_paths)
            
            results[page] = page_paths
            
        return results


if __name__ == "__main__":
    # Example usage
    generator = ImageGenerator()
    paths = generator.generate_sage_images()
    print(f"Generated {len(paths)} images")
    for path in paths:
        print(f"- {path}")