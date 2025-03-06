"""
Benchmark tests for Gutenberg components.

These tests measure performance of key components to identify bottlenecks.
They should be run manually, not as part of the regular test suite.
"""
import os
import sys
import time
import asyncio
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import uuid
import json
import logging
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import components to benchmark
from core.content_generator import ContentGenerator
from core.template_engine import TemplateEngine
from integrations.ptolemy_client import PtolemyClient
from integrations.llm_service import LLMService
from services.content_service import ContentService
from config.settings import get_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("benchmark")

# Test fixtures
@pytest.fixture
def template_data():
    """Load a template for testing."""
    template_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_data/templates/test_template.json"
    )
    with open(template_path, 'r') as f:
        return json.load(f)

@pytest.fixture
def content_parameters():
    """Parameters for content generation."""
    return {
        "concept_name": "photosynthesis",
        "age_range": "14-18",
        "difficulty": "intermediate",
        "length": "medium",
        "include_media": False
    }

class BenchmarkTimer:
    """Context manager for timing operations."""
    
    def __init__(self, name):
        self.name = name
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        logger.info(f"Benchmark - {self.name}: {self.duration:.4f} seconds")


@pytest.mark.asyncio
async def test_template_engine_performance(template_data, content_parameters):
    """Benchmark template engine performance."""
    # Create a mock LLM service that responds quickly
    mock_llm = MagicMock()
    mock_llm.generate_content = AsyncMock(return_value={"introduction": "Test content"})
    
    # Create the template engine
    engine = TemplateEngine(llm_service=mock_llm)
    
    # Measure template processing time
    logger.info("Benchmarking template engine...")
    
    # Warm-up run
    await engine.process_template(template_data, content_parameters)
    
    # Timed runs
    results = []
    for i in range(5):
        with BenchmarkTimer(f"Template processing run {i+1}"):
            result = await engine.process_template(template_data, content_parameters)
            results.append(result)
    
    # Calculate and log average time
    logger.info(f"Template engine benchmark complete.")
    
    return results


@pytest.mark.asyncio
async def test_placeholder_replacement_performance():
    """Benchmark placeholder replacement performance."""
    # Create the template engine
    engine = TemplateEngine()
    
    # Create a template with many placeholders
    template_text = "This is a [adj1] template with [adj2] placeholders that need to be [adj3] replaced. " \
                   "The goal is to [verb1] how [adv1] the system can [verb2] when faced with a [adj4] number " \
                   "of [noun1] to [verb3]."
    
    parameters = {
        "adj1": "complex",
        "adj2": "numerous",
        "adj3": "quickly",
        "adj4": "large",
        "verb1": "test",
        "verb2": "perform",
        "verb3": "process",
        "adv1": "efficiently",
        "noun1": "replacements"
    }
    
    # Measure placeholder replacement time
    logger.info("Benchmarking placeholder replacement...")
    
    # Warm-up run
    engine.replace_placeholders(template_text, parameters)
    
    # Timed runs
    iterations = 1000
    with BenchmarkTimer(f"Placeholder replacement ({iterations} iterations)"):
        for _ in range(iterations):
            result = engine.replace_placeholders(template_text, parameters)
    
    logger.info(f"Placeholder replacement benchmark complete.")


@pytest.mark.asyncio
async def test_content_generator_performance():
    """Benchmark content generator performance with mocked dependencies."""
    # Create mock services
    mock_ptolemy = MagicMock()
    mock_ptolemy.get_concept = AsyncMock(return_value={
        "id": "concept123",
        "name": "Photosynthesis", 
        "description": "Process by which plants convert light energy into chemical energy"
    })
    mock_ptolemy.get_relationships = AsyncMock(return_value=[
        {"source": "concept123", "target": "concept456", "type": "PREREQUISITE"}
    ])
    
    mock_template_engine = MagicMock()
    mock_template_engine.process_template = AsyncMock(return_value={
        "title": "Photosynthesis",
        "sections": [
            {"id": "intro", "title": "Introduction", "content": "Test content"}
        ]
    })
    
    # Create the content generator
    generator = ContentGenerator(
        ptolemy_client=mock_ptolemy,
        template_engine=mock_template_engine
    )
    
    # Test parameters
    request = {
        "topic": "photosynthesis",
        "template_id": "default",
        "parameters": {
            "age_range": "14-18",
            "difficulty": "intermediate"
        }
    }
    
    # Measure content generation time
    logger.info("Benchmarking content generator...")
    
    # Warm-up run
    await generator.generate_content(request)
    
    # Timed runs
    results = []
    for i in range(5):
        with BenchmarkTimer(f"Content generation run {i+1}"):
            result = await generator.generate_content(request)
            results.append(result)
    
    logger.info(f"Content generator benchmark complete.")
    
    return results


def main():
    """Run all benchmarks sequentially."""
    logger.info("Starting Gutenberg performance benchmarking")
    logger.info("=" * 80)
    
    # Get the event loop
    loop = asyncio.get_event_loop()
    
    # Run the benchmarks
    template_data = json.load(open(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_data/templates/test_template.json"
    )))
    
    content_parameters = {
        "concept_name": "photosynthesis",
        "age_range": "14-18",
        "difficulty": "intermediate",
        "length": "medium",
        "include_media": False
    }
    
    loop.run_until_complete(test_template_engine_performance(
        template_data, content_parameters
    ))
    logger.info("-" * 80)
    
    loop.run_until_complete(test_placeholder_replacement_performance())
    logger.info("-" * 80)
    
    loop.run_until_complete(test_content_generator_performance())
    logger.info("-" * 80)
    
    logger.info("Benchmark testing complete!")


if __name__ == "__main__":
    main()