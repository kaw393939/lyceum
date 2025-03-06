#!/usr/bin/env python3
"""
Generate media assets (images and audio) for the Lyceum platform.
This script calls the necessary generator utilities to create all needed assets.
"""

import os
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from utils.image_generator import ImageGenerator
from utils.audio_generator import AudioGenerator
from utils.content_generator import ContentGenerator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_all_media(openai_api_key: Optional[str] = None, anthropic_api_key: Optional[str] = None) -> Dict[str, Any]:
    """Generate all media assets for the Lyceum platform."""
    
    # Generate images
    logger.info("Generating images...")
    image_gen = ImageGenerator(api_key=openai_api_key)
    image_paths = image_gen.generate_sage_images()
    logger.info(f"Generated {len(image_paths)} images")
    
    # Generate new logo with enhanced prompt
    logo_prompt = """
    Create a modern, minimalist logo for 'Lyceum', an AI-powered educational platform. 
    The logo should symbolize the fusion of ancient wisdom and modern technology. 
    Include an elegant owl or Greek column combined with subtle neural network patterns. 
    Use a sophisticated color palette with deep blue and gold. 
    The design should be clean and work well at various sizes, conveying wisdom, intelligence, and enlightenment.
    Make it suitable for an innovative educational platform that uses artificial intelligence to transform learning.
    """
    
    logo_path = image_gen.generate_image(
        prompt=logo_prompt.strip(),
        filename="lyceum_logo",
        size="1024x1024",
        quality="hd",
        style="vivid"
    )
    logger.info(f"Generated new logo: {logo_path}")
    
    # Generate philosophical dialogue image
    dialogue_prompt = """
    Create a beautiful, artistic illustration of a Socratic dialogue in a futuristic learning environment. 
    Show a wise mentor figure engaging with students, with holographic displays of knowledge graphs and concepts floating around them. 
    The scene should blend classical Greek architectural elements with advanced technology.
    Use a color palette of deep blues, purples, and gold accents to create a sense of wisdom and enlightenment.
    The style should be detailed and elegant, evoking the atmosphere of ancient philosophical discussions enhanced by modern technology.
    """
    
    dialogue_path = image_gen.generate_image(
        prompt=dialogue_prompt.strip(),
        filename="lyceum_dialogue",
        size="1024x1024",
        quality="hd",
        style="vivid"
    )
    logger.info(f"Generated dialogue image: {dialogue_path}")
    
    # Generate knowledge graph visualization
    knowledge_prompt = """
    Create a detailed visualization of an educational knowledge graph for the Lyceum platform. 
    Show a network of interconnected concepts with glowing nodes and edges, representing how ideas relate to each other. 
    The visualization should be beautiful and abstract, with a deep blue background and gold/white nodes. 
    Include subtle labels on some of the nodes to suggest philosophical and educational concepts.
    Make it appear as if this is a holographic interface that someone could interact with.
    """
    
    knowledge_path = image_gen.generate_image(
        prompt=knowledge_prompt.strip(),
        filename="lyceum_knowledge",
        size="1024x1024",
        quality="hd",
        style="vivid"
    )
    logger.info(f"Generated knowledge graph image: {knowledge_path}")
    
    # Generate audio with Claude 3.7 script
    logger.info("Generating audio with Claude 3.7 script...")
    audio_gen = AudioGenerator(api_key=openai_api_key)
    audio_path = audio_gen.generate_introduction_audio(anthropic_api_key=anthropic_api_key)
    logger.info(f"Generated audio: {audio_path}")
    
    # Create symlink for latest audio file
    if audio_path:
        latest_link = Path(__file__).parent.parent / "static" / "audio" / "lyceum_introduction_latest.mp3"
        if os.path.exists(latest_link):
            os.remove(latest_link)
        os.symlink(audio_path, latest_link)
        logger.info(f"Created symlink from {latest_link} to {audio_path}")
        
        # Run a test to ensure the audio file is valid
        if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
            logger.info(f"Audio file successfully generated: {os.path.getsize(audio_path)} bytes")
        else:
            logger.error(f"Audio file generation failed or file is empty")
            
        # Test MP3 header
        try:
            with open(audio_path, 'rb') as f:
                header = f.read(10)
                # Check for MP3 header (ID3 or MPEG frame sync)
                if header.startswith(b'ID3') or header[0:2] in [b'\xFF\xFB', b'\xFF\xF3', b'\xFF\xF2', b'\xFF\xE3']:
                    logger.info("Audio file validated: Valid MP3 header found")
                else:
                    logger.warning("Audio file may not be a valid MP3")
        except Exception as e:
            logger.error(f"Error validating audio file: {e}")
    
    # Generate content with Claude 3.7 - wrap in try/except to handle errors properly
    content_result = None
    try:
        if anthropic_api_key:
            logger.info("Generating content with Claude 3.7...")
            content_gen = ContentGenerator(api_key=anthropic_api_key)
            vision_content = content_gen.generate_vision_content()
            
            # Validate content
            if vision_content and len(vision_content) > 100:  # Ensure we got meaningful content
                logger.info(f"Generated vision content of length: {len(vision_content)} characters")
                content_result = vision_content
            else:
                logger.warning("Generated content seems too short or empty")
        else:
            logger.warning("No Anthropic API key provided, skipping content generation")
    except Exception as e:
        logger.error(f"Error during content generation: {str(e)}")
        # Don't let content generation failure stop the whole process
    
    # Safely handle possible None values and normalize all paths
    result_images = []
    
    # Add the generic images
    if image_paths:
        result_images.extend([p for p in image_paths if p])
    
    # Add logo image
    if logo_path:
        if isinstance(logo_path, list):
            result_images.extend([p for p in logo_path if p])
        else:
            result_images.append(logo_path)
    
    # Add dialogue image
    if dialogue_path:
        if isinstance(dialogue_path, list):
            result_images.extend([p for p in dialogue_path if p])
        else:
            result_images.append(dialogue_path)
            
    # Add knowledge image
    if knowledge_path:
        if isinstance(knowledge_path, list):
            result_images.extend([p for p in knowledge_path if p])
        else:
            result_images.append(knowledge_path)
    
    return {
        "images": result_images,
        "audio": audio_path,
        "content": content_result
    }

def main() -> None:
    """Main entry point with proper error handling."""
    parser = argparse.ArgumentParser(description='Generate media assets for Lyceum')
    parser.add_argument('--openai-key', type=str, help='OpenAI API key (will use environment variable if not provided)')
    parser.add_argument('--anthropic-key', type=str, help='Anthropic API key (will use environment variable if not provided)')
    parser.add_argument('--skip-images', action='store_true', help='Skip image generation')
    parser.add_argument('--skip-audio', action='store_true', help='Skip audio generation')
    parser.add_argument('--skip-content', action='store_true', help='Skip content generation')
    args = parser.parse_args()
    
    try:
        logger.info("Starting media generation process...")
        
        # Validate API keys
        if not args.openai_key and not os.getenv("OPENAI_API_KEY") and not args.skip_images and not args.skip_audio:
            logger.warning("No OpenAI API key provided for image or audio generation")
            print("WARNING: No OpenAI API key provided. Set OPENAI_API_KEY environment variable or use --openai-key.")
        
        if not args.anthropic_key and not os.getenv("ANTHROPIC_API_KEY") and not args.skip_content:
            logger.warning("No Anthropic API key provided for content generation")
            print("WARNING: No Anthropic API key provided. Set ANTHROPIC_API_KEY environment variable or use --anthropic-key.")
        
        # Generate assets
        assets = generate_all_media(
            openai_api_key=args.openai_key,
            anthropic_api_key=args.anthropic_key
        )
        
        # Print results
        print("\nGenerated Assets:")
        
        if assets["images"]:
            print("Images:")
            for img in assets["images"]:
                print(f"  - {img}")
        else:
            print("No images were generated")
            
        if assets["audio"]:
            print(f"Audio: {assets['audio']}")
        else:
            print("No audio was generated")
            
        if assets["content"]:
            print(f"Content: Generated {len(assets['content'])} characters of content")
        else:
            print("No content was generated")
            
        logger.info("Media generation completed successfully")
        
    except Exception as e:
        logger.error(f"Error during media generation: {str(e)}")
        print(f"ERROR: Media generation failed: {str(e)}")
        import sys
        sys.exit(1)

if __name__ == "__main__":
    main()