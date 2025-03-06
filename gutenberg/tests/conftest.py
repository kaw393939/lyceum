"""
Pytest configuration for Gutenberg tests.
"""
import os
import sys
import asyncio
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
import logging
from pathlib import Path

# Add the parent directory to the Python path 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import app modules after path is set
from config.settings import get_config
from main import app as main_app
from integrations.mongodb_service import MongoDBService
from integrations.ptolemy_client import PtolemyClient
from integrations.llm_service import LLMService
from core.content_generator import ContentGenerator
from core.template_engine import TemplateEngine
from services.content_service import ContentService
from services.template_service import TemplateService

# Disable logging during tests
logging.basicConfig(level=logging.ERROR)

@pytest.fixture
def app() -> FastAPI:
    """Return the FastAPI app instance."""
    return main_app

@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Return a test client for the FastAPI app."""
    return TestClient(app)

@pytest.fixture
def config():
    """Return the application configuration."""
    return get_config()

@pytest.fixture
def test_template_path():
    """Return the path to test templates."""
    return Path("tests/test_data/templates/test_template.json")

@pytest.fixture
def mock_mongodb_service():
    """Return a mock MongoDB service."""
    mongodb = MongoDBService(use_mock=True)
    return mongodb

@pytest.fixture
def mock_ptolemy_client():
    """Return a mock Ptolemy client."""
    ptolemy = PtolemyClient(use_mock=True)
    return ptolemy

@pytest.fixture
def mock_llm_service():
    """Return a mock LLM service."""
    llm = LLMService(use_mock=True)
    return llm

@pytest.fixture
def template_engine(mock_llm_service):
    """Return a template engine instance with mock services."""
    engine = TemplateEngine(llm_service=mock_llm_service)
    return engine

@pytest.fixture
def content_generator(mock_ptolemy_client, template_engine):
    """Return a content generator instance with mock services."""
    generator = ContentGenerator(
        ptolemy_client=mock_ptolemy_client,
        template_engine=template_engine
    )
    return generator

@pytest.fixture
def content_service(mock_mongodb_service, content_generator):
    """Return a content service instance with mock services."""
    service = ContentService(
        mongodb=mock_mongodb_service,
        content_generator=content_generator
    )
    return service

@pytest.fixture
def template_service(mock_mongodb_service):
    """Return a template service instance with mock services."""
    service = TemplateService(
        mongodb=mock_mongodb_service
    )
    return service

@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for each test."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()