import os
import pytest
import httpx
import logging
from pymongo import MongoClient
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test requiring real services"
    )
    config.addinivalue_line(
        "markers", "services: mark test as requiring specific services to be running"
    )


@pytest.fixture(scope="session")
def ptolemy_url():
    """Get the Ptolemy service URL from environment or use default."""
    return os.environ.get("PTOLEMY_URL", "http://localhost:8000")


@pytest.fixture(scope="session")
def gutenberg_url():
    """Get the Gutenberg service URL from environment or use default."""
    return os.environ.get("GUTENBERG_URL", "http://localhost:8001")


@pytest.fixture(scope="session")
def mongodb_uri():
    """Get the MongoDB URI from environment or use default."""
    return os.environ.get("MONGODB_URI", "mongodb://localhost:27017")


@pytest.fixture(scope="session")
def api_key():
    """Get the API key from environment."""
    return os.environ.get("PLATO_API_KEY", "")


@pytest.fixture(scope="session")
def check_services_running(ptolemy_url, gutenberg_url):
    """Check if required services are running and skip tests if they're not."""
    services_status = {}
    
    # Check Ptolemy
    try:
        response = httpx.get(f"{ptolemy_url}/health", timeout=5.0)
        services_status["ptolemy"] = response.status_code == 200
    except Exception as e:
        logger.error(f"Error connecting to Ptolemy: {e}")
        services_status["ptolemy"] = False
    
    # Check Gutenberg
    try:
        response = httpx.get(f"{gutenberg_url}/health", timeout=5.0)
        services_status["gutenberg"] = response.status_code == 200
    except Exception as e:
        logger.error(f"Error connecting to Gutenberg: {e}")
        services_status["gutenberg"] = False
    
    # Skip if not all services are running
    if not all(services_status.values()):
        pytest.skip(f"Required services not available: {services_status}")
    
    return services_status


@pytest.fixture(scope="session")
def mongodb_client(mongodb_uri):
    """Create a MongoDB client for test database."""
    client = MongoClient(mongodb_uri)
    yield client
    client.close()


@pytest.fixture(scope="session")
def http_client():
    """Create a reusable HTTP client for tests."""
    with httpx.Client(timeout=10.0) as client:
        yield client


@pytest.fixture
def test_concept():
    """Create a test concept for testing."""
    return {
        "name": "Pytest Test Concept",
        "description": "A concept created for pytest integration testing",
        "domains": ["testing", "pytest", "integration"],
        "id": f"pytest_concept_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    }


@pytest.fixture
def create_test_concept(ptolemy_url, test_concept, api_key, http_client):
    """Create a test concept in Ptolemy and clean it up after test."""
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    # Create the concept
    response = http_client.post(
        f"{ptolemy_url}/api/v1/concepts",
        json=test_concept,
        headers=headers
    )
    
    if response.status_code != 200:
        pytest.skip(f"Could not create test concept: {response.text}")
    
    concept_data = response.json()
    concept_id = concept_data["id"]
    
    yield concept_id
    
    # Clean up - delete the concept
    try:
        http_client.delete(
            f"{ptolemy_url}/api/v1/concepts/{concept_id}",
            headers=headers
        )
    except Exception as e:
        logger.warning(f"Failed to delete test concept {concept_id}: {e}")


@pytest.fixture
def request_content_generation(gutenberg_url, create_test_concept, api_key, http_client):
    """Request content generation in Gutenberg and return the request ID."""
    concept_id = create_test_concept
    
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    response = http_client.post(
        f"{gutenberg_url}/api/v1/content/generate",
        json={
            "concept_id": concept_id,
            "format": "lesson",
            "style": "concise",
            "target_audience": "beginner"
        },
        headers=headers
    )
    
    if response.status_code != 200:
        pytest.skip(f"Could not request content generation: {response.text}")
    
    data = response.json()
    
    if "request_id" not in data:
        pytest.skip("No request_id in response")
        
    return data["request_id"]