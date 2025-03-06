#!/usr/bin/env python3
"""
Test script for audio generation.
Tests both the Claude script generation and the OpenAI TTS functionality.
"""

import os
import argparse
import logging
from dotenv import load_dotenv
from utils.audio_generator import AudioGenerator

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_audio_generation():
    """Test audio generation functionality."""
    # Get API keys from environment or arguments
    openai_api_key = os.getenv("OPENAI_API_KEY")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    
    if not openai_api_key:
        logger.error("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        return False
    
    # Test audio generation
    try:
        logger.info("Initializing AudioGenerator...")
        audio_gen = AudioGenerator(api_key=openai_api_key)
        
        # Test default script if no Anthropic key
        if not anthropic_api_key:
            logger.info("Testing audio generation with default script...")
            audio_path = audio_gen.generate_introduction_audio()
        else:
            logger.info("Testing audio generation with Claude 3.7 script...")
            audio_path = audio_gen.generate_introduction_audio(anthropic_api_key=anthropic_api_key)
        
        # Verify audio file was created
        if audio_path and os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
            logger.info(f"✓ Audio generated successfully: {audio_path} ({os.path.getsize(audio_path)} bytes)")
            
            # Create symlink to latest
            latest_path = os.path.join(os.path.dirname(audio_path), "lyceum_introduction_latest.mp3")
            if os.path.exists(latest_path):
                os.remove(latest_path)
            os.symlink(audio_path, latest_path)
            logger.info(f"✓ Created symlink to latest: {latest_path}")
            
            return True
        else:
            logger.error(f"× Audio generation failed or file is empty: {audio_path}")
            return False
    
    except Exception as e:
        logger.error(f"× Error during audio generation test: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test audio generation')
    parser.add_argument('--openai-key', type=str, help='OpenAI API key')
    parser.add_argument('--anthropic-key', type=str, help='Anthropic API key')
    args = parser.parse_args()
    
    # Override environment variables if args provided
    if args.openai_key:
        os.environ["OPENAI_API_KEY"] = args.openai_key
    if args.anthropic_key:
        os.environ["ANTHROPIC_API_KEY"] = args.anthropic_key
    
    # Run the test
    print("\n==== Testing Audio Generation ====\n")
    success = test_audio_generation()
    
    if success:
        print("\n✅ Audio generation test PASSED\n")
        exit(0)
    else:
        print("\n❌ Audio generation test FAILED\n")
        exit(1)