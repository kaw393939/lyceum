"""
Tests for the API routes.
"""
import os
import json
import pytest
from fastapi.testclient import TestClient
from fastapi import status
from unittest.mock import patch, MagicMock, AsyncMock
import uuid

# Test data
@pytest.fixture
def test_content_request():
    """Sample content generation request data."""
    return {
        "topic": "photosynthesis",
        "template_id": "default",
        "parameters": {
            "age_range": "14-18",
            "difficulty": "intermediate"
        }
    }

@pytest.fixture
def test_content_response():
    """Sample content generation response data."""
    return {
        "content_id": str(uuid.uuid4()),
        "request_id": str(uuid.uuid4()),
        "status": "processing"
    }

@pytest.fixture
def test_content_data():
    """Sample content data."""
    return {
        "_id": str(uuid.uuid4()),
        "title": "Photosynthesis",
        "template_id": "default",
        "status": "completed",
        "sections": [
            {
                "id": "introduction",
                "title": "Introduction",
                "content": "This is an introduction to photosynthesis."
            }
        ]
    }

def test_health_endpoint(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data

def test_api_routes(client):
    """Test that API routes are correctly registered."""
    response = client.get("/api/v1")
    assert response.status_code == 404  # No root handler for API
    
    # Should redirect to docs
    response = client.get("/")
    assert response.status_code == 200 or response.status_code == 307

@patch("api.content_routes.ContentService")
def test_generate_content(mock_content_service, client, test_content_request, test_content_response):
    """Test content generation endpoint."""
    # Set up the mock
    mock_instance = MagicMock()
    mock_instance.create_content.return_value = test_content_response
    mock_content_service.return_value = mock_instance
    
    # Make the request
    response = client.post("/api/v1/content/generate", json=test_content_request)
    
    # Check the response
    assert response.status_code == 202
    data = response.json()
    assert "content_id" in data
    assert "request_id" in data
    assert data["status"] == "processing"

@patch("api.content_routes.ContentService")
def test_get_content(mock_content_service, client, test_content_data):
    """Test retrieving content endpoint."""
    # Set up the mock
    mock_instance = MagicMock()
    mock_instance.get_content.return_value = test_content_data
    mock_content_service.return_value = mock_instance
    
    # Make the request
    response = client.get(f"/api/v1/content/{test_content_data['_id']}")
    
    # Check the response
    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "Photosynthesis"
    assert len(data["sections"]) == 1

@patch("api.content_routes.ContentService")
def test_get_content_not_found(mock_content_service, client):
    """Test retrieving non-existent content."""
    # Set up the mock
    mock_instance = MagicMock()
    mock_instance.get_content.return_value = None
    mock_content_service.return_value = mock_instance
    
    # Make the request
    response = client.get(f"/api/v1/content/{uuid.uuid4()}")
    
    # Check the response
    assert response.status_code == 404

@patch("api.template_routes.TemplateService")
def test_list_templates(mock_template_service, client):
    """Test listing templates endpoint."""
    # Set up the mock
    mock_instance = MagicMock()
    mock_instance.list_templates.return_value = [
        {"id": "default", "name": "Default Template"},
        {"id": "assessment", "name": "Assessment Template"}
    ]
    mock_template_service.return_value = mock_instance
    
    # Make the request
    response = client.get("/api/v1/templates")
    
    # Check the response
    assert response.status_code == 200
    data = response.json()
    assert "templates" in data
    assert len(data["templates"]) == 2
    assert data["templates"][0]["id"] == "default"