"""
Tests for the template engine.
"""
import os
import json
import pytest
from pathlib import Path
import logging
from unittest.mock import Mock, patch, AsyncMock

# Import the class to test
from core.template_engine import TemplateEngine
from integrations.llm_service import LLMService

# Disable logging during tests
logging.basicConfig(level=logging.ERROR)

@pytest.fixture
def mock_llm_response():
    """Return a mock LLM response."""
    return {
        "introduction": "This is a test introduction about photosynthesis for 14-18 students."
    }

@pytest.fixture
def test_template_data():
    """Sample test template data."""
    return {
        "template_id": "test_template",
        "name": "Test Template",
        "sections": [
            {
                "id": "introduction",
                "title": "Introduction",
                "prompts": [
                    {
                        "id": "intro_prompt",
                        "prompt_text": "Write about [concept_name] for [age_range] students.",
                        "placeholders": [
                            {"name": "concept_name"},
                            {"name": "age_range"}
                        ],
                        "output_format": {"introduction": "string"}
                    }
                ]
            }
        ]
    }

@pytest.mark.asyncio
async def test_process_template(mock_llm_service, test_template_data, mock_llm_response):
    """Test processing a template."""
    # Mock the LLM service to return a predetermined response
    mock_llm_service.generate_content = AsyncMock(return_value=mock_llm_response)
    
    # Create engine with mocked services
    engine = TemplateEngine(llm_service=mock_llm_service)
    
    # Parameters to replace in the template
    parameters = {
        "concept_name": "photosynthesis",
        "age_range": "14-18"
    }
    
    # Process the template
    result = await engine.process_template(
        template=test_template_data,
        parameters=parameters
    )
    
    # Verify results
    assert result is not None
    assert "sections" in result
    assert len(result["sections"]) == 1
    assert result["sections"][0]["title"] == "Introduction"
    
    # Verify the LLM was called with the expected prompt
    expected_prompt = "Write about photosynthesis for 14-18 students."
    mock_llm_service.generate_content.assert_called_once()
    call_args = mock_llm_service.generate_content.call_args[1]
    assert "prompt" in call_args
    assert call_args["prompt"] == expected_prompt

@pytest.mark.asyncio
async def test_create_media_item(template_engine):
    """Test creating a media item."""
    # Test data
    media_spec = {
        "id": "test_image",
        "media_type": "image",
        "prompt": "Create an image about [concept_name]",
        "placeholders": [{"name": "concept_name"}]
    }
    parameters = {"concept_name": "photosynthesis"}
    
    # Mock the media generation function
    with patch.object(template_engine, "_generate_media", return_value={
        "url": "http://example.com/test.jpg",
        "alt_text": "An image about photosynthesis",
        "type": "image"
    }):
        # Process the media item
        result = await template_engine.create_media_item(media_spec, parameters)
        
        # Verify results
        assert result is not None
        assert result["url"] == "http://example.com/test.jpg"
        assert result["alt_text"] == "An image about photosynthesis"
        assert result["type"] == "image"

@pytest.mark.asyncio
async def test_replace_placeholders():
    """Test placeholder replacement in text."""
    engine = TemplateEngine()
    
    # Test text with placeholders
    text = "Learn about [concept_name] which is suitable for [age_range] students."
    parameters = {
        "concept_name": "photosynthesis",
        "age_range": "14-18"
    }
    
    # Replace placeholders
    result = engine.replace_placeholders(text, parameters)
    
    # Verify results
    assert result == "Learn about photosynthesis which is suitable for 14-18 students."
    
    # Test with missing placeholder
    text = "Learn about [concept_name] and [missing_value]."
    with pytest.raises(ValueError, match="Missing value for placeholder"):
        engine.replace_placeholders(text, parameters)
    
    # Test with default value for missing placeholder
    text = "Learn about [concept_name] and [missing_value]."
    placeholders = [
        {"name": "missing_value", "default_value": "default"}
    ]
    result = engine.replace_placeholders(text, parameters, placeholders)
    assert result == "Learn about photosynthesis and default."