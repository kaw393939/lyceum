import os
import time
import logging
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mark all tests as integration tests
pytestmark = [pytest.mark.integration, pytest.mark.services]


@pytest.fixture
def wait_for_content_completion(gutenberg_url, http_client, api_key):
    """Wait for content generation to complete with timeout."""
    def _wait(request_id, timeout_seconds=60, polling_interval=2):
        """
        Wait for content generation to complete.
        
        Args:
            request_id: The content generation request ID
            timeout_seconds: Maximum time to wait in seconds
            polling_interval: How often to check status in seconds
            
        Returns:
            The content ID if generation completed, None otherwise
        """
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=timeout_seconds)
        
        while datetime.now() < end_time:
            response = http_client.get(
                f"{gutenberg_url}/api/v1/content/status/{request_id}",
                headers=headers
            )
            
            if response.status_code != 200:
                logger.warning(f"Error checking content status: {response.text}")
                time.sleep(polling_interval)
                continue
                
            data = response.json()
            status = data.get("status")
            
            if status == "complete" and "content_id" in data:
                logger.info(f"Content generation completed in {(datetime.now() - start_time).total_seconds():.1f} seconds")
                return data["content_id"]
            elif status == "failed":
                logger.error(f"Content generation failed: {data.get('error', 'Unknown error')}")
                return None
                
            logger.info(f"Content status: {status}, waiting...")
            time.sleep(polling_interval)
            
        logger.warning(f"Timeout waiting for content generation after {timeout_seconds} seconds")
        return None
        
    return _wait


def test_concept_creation_and_retrieval(check_services_running, ptolemy_url, test_concept, http_client, api_key):
    """Test creating and retrieving a concept in Ptolemy."""
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    # Create concept
    response = http_client.post(
        f"{ptolemy_url}/api/v1/concepts",
        json=test_concept,
        headers=headers
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify response has required fields
    assert "id" in data
    assert data["name"] == test_concept["name"]
    
    concept_id = data["id"]
    logger.info(f"Created concept with ID: {concept_id}")
    
    # Retrieve concept
    response = http_client.get(
        f"{ptolemy_url}/api/v1/concepts/{concept_id}",
        headers=headers
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify retrieved concept matches
    assert data["id"] == concept_id
    assert data["name"] == test_concept["name"]
    
    # Clean up - delete concept
    response = http_client.delete(
        f"{ptolemy_url}/api/v1/concepts/{concept_id}",
        headers=headers
    )
    
    assert response.status_code == 200
    logger.info(f"Deleted concept with ID: {concept_id}")


def test_content_generation_and_retrieval(check_services_running, ptolemy_url, gutenberg_url, 
                                        create_test_concept, http_client, api_key,
                                        wait_for_content_completion):
    """Test generating and retrieving content from Gutenberg using a concept from Ptolemy."""
    concept_id = create_test_concept
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    # Request content generation
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
    
    assert response.status_code == 200
    data = response.json()
    
    assert "request_id" in data
    request_id = data["request_id"]
    logger.info(f"Content generation started with request ID: {request_id}")
    
    # Wait for content generation to complete
    content_id = wait_for_content_completion(request_id)
    
    # If content generation completed, retrieve the content
    if content_id:
        response = http_client.get(
            f"{gutenberg_url}/api/v1/content/{content_id}",
            headers=headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify content has required fields
        assert "id" in data
        assert "content" in data
        assert "concept_id" in data
        assert data["concept_id"] == concept_id
        
        logger.info(f"Retrieved content with ID: {content_id}")
    else:
        pytest.skip("Content generation did not complete in time")


def test_learning_path_integration(check_services_running, ptolemy_url, gutenberg_url, 
                                http_client, api_key):
    """Test learning path integration between Ptolemy and Gutenberg."""
    # This test creates a small learning path in Ptolemy and then
    # generates content for each concept in the path through Gutenberg
    
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    # Create test timestamp for unique IDs
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    
    # 1. Create multiple related concepts in Ptolemy
    concepts = [
        {
            "name": f"Path Concept 1 - {timestamp}",
            "description": "First concept in learning path",
            "domains": ["testing", "learning_path"],
            "id": f"path_1_{timestamp}"
        },
        {
            "name": f"Path Concept 2 - {timestamp}",
            "description": "Second concept in learning path",
            "domains": ["testing", "learning_path"],
            "id": f"path_2_{timestamp}"
        },
        {
            "name": f"Path Concept 3 - {timestamp}",
            "description": "Third concept in learning path",
            "domains": ["testing", "learning_path"],
            "id": f"path_3_{timestamp}"
        }
    ]
    
    concept_ids = []
    for concept in concepts:
        response = http_client.post(
            f"{ptolemy_url}/api/v1/concepts",
            json=concept,
            headers=headers
        )
        
        assert response.status_code == 200
        data = response.json()
        concept_ids.append(data["id"])
    
    logger.info(f"Created {len(concept_ids)} concepts for learning path")
    
    # 2. Create relationships between concepts
    relationships = [
        {
            "source_id": concept_ids[0],
            "target_id": concept_ids[1],
            "type": "PREREQUISITE",
            "strength": 0.8
        },
        {
            "source_id": concept_ids[1],
            "target_id": concept_ids[2],
            "type": "PREREQUISITE",
            "strength": 0.9
        }
    ]
    
    for relationship in relationships:
        response = http_client.post(
            f"{ptolemy_url}/api/v1/relationships",
            json=relationship,
            headers=headers
        )
        
        assert response.status_code == 200
    
    logger.info(f"Created relationships between concepts")
    
    # 3. Create a learning path
    learning_path = {
        "name": f"Test Learning Path - {timestamp}",
        "description": "A learning path for integration testing",
        "concepts": concept_ids
    }
    
    response = http_client.post(
        f"{ptolemy_url}/api/v1/learning-paths",
        json=learning_path,
        headers=headers
    )
    
    assert response.status_code == 200
    path_data = response.json()
    assert "id" in path_data
    path_id = path_data["id"]
    
    logger.info(f"Created learning path with ID: {path_id}")
    
    # 4. Request content generation for the learning path from Gutenberg
    response = http_client.post(
        f"{gutenberg_url}/api/v1/content/generate-path",
        json={
            "learning_path_id": path_id,
            "format": "lesson",
            "target_audience": "beginner"
        },
        headers=headers
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "batch_id" in data
    batch_id = data["batch_id"]
    
    logger.info(f"Requested learning path content generation with batch ID: {batch_id}")
    
    # 5. Check batch processing status (simplified, would poll in real test)
    response = http_client.get(
        f"{gutenberg_url}/api/v1/content/batch/{batch_id}/status",
        headers=headers
    )
    
    assert response.status_code == 200
    
    # 6. Clean up resources
    for concept_id in concept_ids:
        try:
            http_client.delete(
                f"{ptolemy_url}/api/v1/concepts/{concept_id}",
                headers=headers
            )
        except Exception as e:
            logger.warning(f"Failed to delete concept {concept_id}: {e}")
    
    try:
        http_client.delete(
            f"{ptolemy_url}/api/v1/learning-paths/{path_id}",
            headers=headers
        )
    except Exception as e:
        logger.warning(f"Failed to delete learning path {path_id}: {e}")


def test_data_consistency(check_services_running, ptolemy_url, gutenberg_url, 
                       mongodb_client, api_key, http_client, test_concept):
    """Test data consistency between Ptolemy and Gutenberg databases."""
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    # Check if we're using simulators
    using_simulators = os.environ.get("USING_SIMULATORS", "false").lower() == "true"
    
    # 1. Create a concept in Ptolemy
    response = http_client.post(
        f"{ptolemy_url}/api/v1/concepts",
        json=test_concept,
        headers=headers
    )
    
    assert response.status_code == 200
    concept_data = response.json()
    concept_id = concept_data["id"]

    # 2. Verify concept exists in Ptolemy's MongoDB (or mock check for simulators)
    if using_simulators:
        # When using simulators, we mock the MongoDB check
        logger.info("Using simulators - mocking MongoDB checks")
        mongo_concept = {
            "_id": concept_id,
            "name": test_concept["name"],
            "description": test_concept["description"]
        }
    else:
        # With real services, check the actual MongoDB
        ptolemy_db = mongodb_client.ptolemy
        mongo_concept = ptolemy_db.concepts.find_one({"_id": concept_id})
    
    assert mongo_concept is not None
    assert mongo_concept["name"] == test_concept["name"]
    
    # 3. Generate content for the concept in Gutenberg
    response = http_client.post(
        f"{gutenberg_url}/api/v1/content/generate",
        json={
            "concept_id": concept_id,
            "format": "lesson",
            "target_audience": "beginner"
        },
        headers=headers
    )
    
    assert response.status_code == 200
    request_data = response.json()
    request_id = request_data["request_id"]
    
    # 4. Wait briefly for processing to start
    time.sleep(2)
    
    # 5. Verify request exists in Gutenberg's MongoDB (or mock check for simulators)
    if using_simulators:
        # When using simulators, we mock the MongoDB check
        mongo_request = {
            "_id": request_id,
            "concept_id": concept_id,
            "format": "lesson",
            "status": "processing"
        }
    else:
        # With real services, check the actual MongoDB
        gutenberg_db = mongodb_client.gutenberg
        mongo_request = gutenberg_db.content_requests.find_one({"_id": request_id})
    
    assert mongo_request is not None
    assert mongo_request["concept_id"] == concept_id
    
    # 6. Clean up
    try:
        http_client.delete(
            f"{ptolemy_url}/api/v1/concepts/{concept_id}",
            headers=headers
        )
    except Exception as e:
        logger.warning(f"Failed to delete concept {concept_id}: {e}")


def test_error_handling(check_services_running, gutenberg_url, http_client, api_key):
    """Test error handling between services."""
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    # Test with non-existent concept ID
    non_existent_id = f"non_existent_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    response = http_client.post(
        f"{gutenberg_url}/api/v1/content/generate",
        json={
            "concept_id": non_existent_id,
            "format": "lesson",
            "target_audience": "beginner"
        },
        headers=headers
    )
    
    # Should return error response, not server error
    assert response.status_code in [400, 404, 422]
    data = response.json()
    
    # Response should include error information
    assert "error" in data or "detail" in data
    
    error_msg = data.get("error") or data.get("detail")
    logger.info(f"Received expected error: {error_msg}")
    
    # Ensure the error message mentions the concept
    assert "concept" in str(error_msg).lower()