#!/usr/bin/env python
"""
Audio generator for Lyceum using OpenAI's TTS API.
Generates high-quality audio for all site sections using multiple voices:
alloy, ash, coral, echo, fable, onyx, nova, sage, shimmer
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('audio_generator')

# Import OpenAI client for TTS
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    logger.warning("OpenAI package not available. Install with: pip install openai")
    OPENAI_AVAILABLE = False

# Define paths
PROJECT_ROOT = Path(__file__).parent.parent
AUDIO_DIR = PROJECT_ROOT / "static" / "audio"
CONTENT_DIR = PROJECT_ROOT / "content"

# Create directories if they don't exist
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
CONTENT_DIR.mkdir(parents=True, exist_ok=True)

# All available OpenAI voices
AVAILABLE_VOICES = [
    "alloy",
    "ash", 
    "coral", 
    "echo", 
    "fable", 
    "onyx", 
    "nova", 
    "sage", 
    "shimmer"
]

# Script templates for different site sections
SECTION_SCRIPTS = {
    "introduction": """
    Welcome to Lyceum, a revolutionary approach to education that combines ancient philosophical wisdom with cutting-edge AI technology to create transformative learning experiences.

    Unlike conventional educational platforms, Lyceum employs a sophisticated multi-agent architecture powered by Graph Neural Networks (GNN), enabling personalized dialogue-based learning, historical character interactions, and financial incentives for educational achievement.

    Through our knowledge graph navigation, Socratic dialogue with historical figures, adaptive learning system, and Learn & Earn incentives, we create personalized educational journeys that adapt to your unique needs and abilities.

    Experience education reimagined—where dialogue with historical figures, immersive simulations, and intelligent learning paths transform how we learn, understand, and grow.
    """,
    
    "vision": """
    Lyceum represents a revolutionary approach to education, combining ancient philosophical wisdom with cutting-edge AI technology to create transformative learning experiences. 

    Our vision includes several key innovations: First, Socratic dialogue with historical figures, where students can engage with authentic representations of important historical personalities who explain their thinking and challenge assumptions.

    Second, knowledge graph navigation and personalized learning paths that adapt to your demonstrated understanding, adding remedial content where needed or accelerating past concepts you quickly master.

    Third, our Learn & Earn incentive system that creates tangible motivation for learning achievement by connecting educational progress to financial rewards.

    Fourth, adaptive learning following Bloom's taxonomy, ensuring complete cognitive development from basic recall to advanced creation.

    And finally, immersive simulations and AR/VR experiences that create memorable, embodied learning environments.

    Join us in reimagining education for the modern age.
    """,
    
    "technical": """
    Lyceum's transformative educational experiences are powered by a sophisticated technical architecture that combines modern AI techniques with robust distributed systems.

    At the core is our multi-agent system with specialized AI agents: Socrates manages conversations, Ptolemy maintains concept relationships, Gutenberg generates personalized materials, Galileo analyzes learning patterns, Alexandria interfaces with language models, Aristotle evaluates learning, Hypatia embodies historical figures, and Hermes manages incentive systems.

    Our intelligent core combines Galileo's Graph Neural Network with Kafka event streaming to create a distributed intelligence that learns from system interactions.

    For storage, we use specialized systems: Neo4j for knowledge relationships, Qdrant for semantic search, MinIO for educational content, and PostgreSQL for structured data.

    The entire infrastructure is defined as code and managed through Terraform, Kubernetes, Linkerd, and Airflow, with comprehensive monitoring through Prometheus, Grafana, Jaeger, and the ELK Stack.

    This architecture enables continuous learning, event-sourced interaction tracking, and knowledge graph-based exploration that gets smarter with every student interaction.
    """,
    
    "business": """
    Lyceum's architecture directly supports its business objectives through several key models:

    Our scalable pricing model leverages containerized microservices architecture for cost-effective deployment across different customer sizes, from individual users to educational institutions to enterprise learning environments.

    Our content marketplace is supported by modular storage architecture and content generation capabilities, enabling third-party content integration, creator tools, and revenue sharing models.

    Our API platform, powered by service mesh and gateway architecture, enables secure API access for developer integration, white-label solutions, and partner ecosystem expansion.

    And our Learn & Earn ecosystem, managed by the Hermes rewards agent, supports incentive-based business models including parental funding of educational achievements, enterprise learning rewards programs, and educational institution scholarship models.

    These business models are strengthened by our technical innovations including continuous learning that improves with usage, specialized agents with unified intelligence, intelligent resource allocation, extensible architecture, and a compounding data advantage through our knowledge graph structure.
    """,
    
    "agile": """
    Lyceum will be implemented in four strategic phases:

    In Foundation Phase during months 1-3, we'll establish core infrastructure with Terraform, implement basic agent services on Kubernetes, establish the Kafka event backbone, create our foundational knowledge graph, and develop basic dialogue capabilities. This will deliver basic dialogue-based learning and guided knowledge exploration.

    In Intelligence & Integration Phase during months 4-9, we'll deploy Galileo's GNN for pattern analysis, implement knowledge extraction through Alexandria, develop content generation through Gutenberg, create assessment capabilities with Aristotle, and integrate agent interactions through our service mesh. This will enable fully personalized learning paths and adaptive content.

    In Advanced Features Phase during months 10-15, we'll implement character embodiment through Hypatia, develop the Learn & Earn system through Hermes, create AR/VR interfaces, enhance our GNN with more sophisticated learning models, and develop comprehensive monitoring. This will deliver the complete Lyceum experience with historical characters and immersive content.

    Finally, in Scale & Optimization Phase during months 16-24, we'll optimize for large-scale deployment, improve agent efficiency through learned patterns, expand our content library, develop additional agent specializations, and create partner integration APIs, resulting in a mature platform with broader content offerings and third-party integration capabilities.
    """,
    
    "contact": """
    Thank you for your interest in Lyceum, the revolutionary educational platform that combines ancient philosophical wisdom with cutting-edge AI technology.

    For students, Lyceum offers Socratic dialogues with historical figures, personalized learning paths, educational achievement rewards, and immersive learning environments.

    For educators, we provide custom learning experiences, student progress analytics, motivational incentive systems, and rich knowledge graphs for curriculum development.

    For institutions, we enable cost-effective scaling of educational offerings, personalized learning at institutional scale, integration with existing tools, and GNN-powered learning analytics.

    We offer multiple partnership opportunities including content creation, technical integration, research collaboration, and pilot programs.

    To learn more about Lyceum and how it can transform your educational experience, please contact us at info@lyceum.education or call (555) 123-4567.
    """
}

def get_section_script(section: str) -> str:
    """
    Get the script for a specific section.
    
    Args:
        section (str): Section name (introduction, vision, technical, business, agile, contact)
        
    Returns:
        str: Script content for the section
    """
    if section not in SECTION_SCRIPTS:
        logger.warning(f"Unknown section: {section}, using introduction script")
        section = "introduction"
    
    script = SECTION_SCRIPTS[section].strip()
    
    # Save the script to a file if it's not already there
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    if section == "introduction":
        script_path = CONTENT_DIR / f"lyceum_{section}_{timestamp}.txt"
    else:
        script_path = CONTENT_DIR / f"lyceum_{section}_{timestamp}.md"
        
    with open(script_path, "w") as f:
        f.write(script)
        
    return script

def get_current_content(section: str) -> Optional[str]:
    """
    Get the current content for a section from the latest file.
    
    Args:
        section (str): Section name
        
    Returns:
        str: Current content or None if not found
    """
    # Always use the predefined scripts for audio generation
    if section in SECTION_SCRIPTS:
        return SECTION_SCRIPTS[section]
    
    # If not found in predefined scripts, try to read from file
    # Find the pattern based on section
    if section == "introduction":
        pattern = f"lyceum_{section}_latest.txt"
    else:
        pattern = f"lyceum_{section}_latest.md"
    
    # Find the latest file
    content_path = CONTENT_DIR / pattern
    if content_path.exists() and content_path.is_symlink():
        # If it's a symlink, get the target
        target_path = content_path.resolve()
        if target_path.exists():
            with open(target_path, "r") as f:
                return f.read().strip()
    
    # Also check direct file (non-symlink)
    if content_path.exists() and not content_path.is_symlink():
        with open(content_path, "r") as f:
            return f.read().strip()
    
    return None

def generate_audio(section: str, version: str = "latest", voice: str = "fable", 
                  api_key: Optional[str] = None, model: str = "tts-1-hd") -> Optional[str]:
    """
    Generate high-quality audio file from a section's content.
    
    Args:
        section (str): Section name (introduction, vision, technical, etc.)
        version (str): Version to use (latest or timestamp)
        voice (str): Voice to use for TTS
        api_key (str, optional): API key for OpenAI
        model (str): TTS model to use ('tts-1' or 'tts-1-hd' for higher quality)
        
    Returns:
        str: Path to generated audio file or None if failed
    """
    if voice not in AVAILABLE_VOICES:
        logger.warning(f"Invalid voice: {voice}, using fable instead")
        voice = "fable"
    
    # Get API key from parameter or environment variable
    openai_api_key = api_key or os.environ.get("OPENAI_API_KEY")
    
    if not OPENAI_AVAILABLE or not openai_api_key:
        logger.warning("OpenAI client not available or API key not provided.")
        return None
    
    # Get script content - either from the latest file or predefined
    if version == "latest":
        script = get_current_content(section)
        if not script:
            logger.info(f"No existing content found for {section}, using predefined script")
            script = get_section_script(section)
    else:
        # Use predefined script
        script = get_section_script(section)
    
    try:
        # Initialize the OpenAI client
        client = OpenAI(api_key=openai_api_key)
        
        # Generate timestamp for the audio file
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        audio_filename = f"lyceum_{section}_{voice}_{timestamp}.mp3"
        audio_path = AUDIO_DIR / audio_filename
        
        logger.info(f"Generating audio for {section} with voice '{voice}' using model '{model}'")
        
        # Process the script to remove extra whitespace
        processed_script = " ".join(script.split())
        
        # Generate audio from script with high quality settings
        response = client.audio.speech.create(
            model=model,  # Use high-definition model
            voice=voice,  # Use the specified voice
            input=processed_script,
            speed=0.9,  # Slightly slower for better clarity
            response_format="mp3"
        )
        
        # Save audio to file
        response.stream_to_file(str(audio_path))
        logger.info(f"Audio saved to {audio_path}")
        
        # Create a symlink to the latest audio for this section and voice
        latest_link = AUDIO_DIR / f"lyceum_{section}_{voice}_latest.mp3"
        if latest_link.exists():
            latest_link.unlink()
        try:
            os.symlink(audio_path, latest_link)
            logger.info(f"Created symlink: {latest_link} -> {audio_path}")
        except Exception as e:
            logger.error(f"Could not create symlink: {str(e)}")
            
        # Create metadata file
        metadata = {
            "timestamp": timestamp,
            "section": section,
            "voice": voice,
            "model": model,
            "script": processed_script,
            "file_path": str(audio_path)
        }
        
        metadata_path = AUDIO_DIR / f"lyceum_{section}_{voice}_{timestamp}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        return str(audio_path)
        
    except Exception as e:
        logger.error(f"Error generating audio: {str(e)}")
        return None

def generate_all_voices_for_section(section: str, api_key: Optional[str] = None) -> Dict[str, str]:
    """
    Generate audio for a section using all available voices.
    
    Args:
        section (str): Section name
        api_key (str, optional): API key for OpenAI
        
    Returns:
        dict: Dictionary mapping voice names to audio file paths
    """
    results = {}
    
    for voice in AVAILABLE_VOICES:
        logger.info(f"Generating {section} audio with voice: {voice}")
        audio_path = generate_audio(section, "latest", voice, api_key)
        if audio_path:
            results[voice] = audio_path
    
    return results

def generate_all_sections_with_voice(voice: str, api_key: Optional[str] = None) -> Dict[str, str]:
    """
    Generate audio for all sections using a specific voice.
    
    Args:
        voice (str): Voice to use
        api_key (str, optional): API key for OpenAI
        
    Returns:
        dict: Dictionary mapping section names to audio file paths
    """
    results = {}
    
    for section in SECTION_SCRIPTS.keys():
        logger.info(f"Generating audio for section '{section}' with voice '{voice}'")
        audio_path = generate_audio(section, "latest", voice, api_key)
        if audio_path:
            results[section] = audio_path
    
    return results

def generate_all_audio(api_key: Optional[str] = None) -> Dict[str, Dict[str, str]]:
    """
    Generate audio for all sections with all voices.
    
    Args:
        api_key (str, optional): API key for OpenAI
        
    Returns:
        dict: Nested dictionary mapping sections and voices to audio file paths
    """
    results = {}
    
    openai_api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not OPENAI_AVAILABLE or not openai_api_key:
        logger.warning("OpenAI client not available or API key not provided. Cannot generate audio.")
        return results
    
    for section in SECTION_SCRIPTS.keys():
        results[section] = {}
        for voice in AVAILABLE_VOICES:
            logger.info(f"Generating audio for section '{section}' with voice '{voice}'")
            audio_path = generate_audio(section, "latest", voice, openai_api_key)
            if audio_path:
                results[section][voice] = audio_path
    
    return results

def main():
    """
    Main function to generate audio introduction.
    """
    # Get API key from environment variable
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    
    if not openai_api_key:
        logger.warning("OPENAI_API_KEY environment variable not set")
        logger.warning("Cannot generate audio files")
        return
    
    # Generate audio for all sections with fable voice
    logger.info("Generating audio for all sections with fable voice")
    results = generate_all_sections_with_voice("fable", openai_api_key)
    
    logger.info(f"Generated {len(results)} audio files:")
    for section, path in results.items():
        logger.info(f"- {section}: {path}")
        
if __name__ == "__main__":
    main()