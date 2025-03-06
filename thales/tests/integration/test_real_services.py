import os
import unittest
import json
import logging
import httpx
import pytest
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealServiceIntegrationTest(unittest.TestCase):
    """Integration tests using real Ptolemy and Gutenberg services."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test resources before running tests."""
        # Service URLs - can be overridden with environment variables
        cls.ptolemy_url = os.environ.get("PTOLEMY_URL", "http://localhost:8000")
        cls.gutenberg_url = os.environ.get("GUTENBERG_URL", "http://localhost:8001")
        
        # Optional auth credentials
        cls.api_key = os.environ.get("PLATO_API_KEY", "")
        
        # Check if services are running
        cls._check_services_running()
        
        # Test data that will be used and modified by tests
        cls.test_concept = {
            "name": "Test Integration Concept",
            "description": "A concept created for integration testing",
            "domains": ["testing", "integration"],
            "id": f"test_concept_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        }
    
    @classmethod
    def _check_services_running(cls):
        """Check if Ptolemy and Gutenberg services are running."""
        services_status = {}
        
        # Check Ptolemy
        try:
            response = httpx.get(f"{cls.ptolemy_url}/health", timeout=5.0)
            services_status["ptolemy"] = response.status_code == 200
        except Exception as e:
            logger.error(f"Error connecting to Ptolemy: {e}")
            services_status["ptolemy"] = False
        
        # Check Gutenberg
        try:
            response = httpx.get(f"{cls.gutenberg_url}/health", timeout=5.0)
            services_status["gutenberg"] = response.status_code == 200
        except Exception as e:
            logger.error(f"Error connecting to Gutenberg: {e}")
            services_status["gutenberg"] = False
        
        # Skip all tests if services aren't running
        if not all(services_status.values()):
            raise unittest.SkipTest(
                f"Services not available. Status: {json.dumps(services_status)}"
            )
        
        logger.info(f"Services are running: {json.dumps(services_status)}")
    
    def _make_request(self, service, endpoint, method="GET", data=None, params=None):
        """Helper method to make requests to services with proper error handling."""
        url = getattr(self, f"{service}_url") + endpoint
        headers = {}
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        try:
            if method.upper() == "GET":
                response = httpx.get(url, params=params, headers=headers, timeout=10.0)
            elif method.upper() == "POST":
                response = httpx.post(url, json=data, headers=headers, timeout=10.0)
            elif method.upper() == "PUT":
                response = httpx.put(url, json=data, headers=headers, timeout=10.0)
            elif method.upper() == "DELETE":
                response = httpx.delete(url, headers=headers, timeout=10.0)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
                
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Error making request to {url}: {str(e)}")
            raise
    
    def test_ptolemy_concept_creation(self):
        """Test creating a concept in Ptolemy."""
        # Create concept
        response = self._make_request(
            "ptolemy", 
            "/api/v1/concepts", 
            method="POST", 
            data=self.test_concept
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Verify response
        self.assertIn("id", data)
        self.assertEqual(data["name"], self.test_concept["name"])
        
        # Store concept ID for future tests
        self.__class__.concept_id = data["id"]
        logger.info(f"Created concept with ID: {self.concept_id}")
    
    def test_ptolemy_concept_retrieval(self):
        """Test retrieving a concept from Ptolemy."""
        # Skip if concept wasn't created
        if not hasattr(self, "concept_id"):
            self.skipTest("Concept creation test failed or was skipped")
        
        # Get concept
        response = self._make_request(
            "ptolemy", 
            f"/api/v1/concepts/{self.concept_id}"
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Verify response
        self.assertEqual(data["id"], self.concept_id)
        self.assertEqual(data["name"], self.test_concept["name"])
    
    def test_gutenberg_content_generation(self):
        """Test generating content in Gutenberg using a concept from Ptolemy."""
        # Skip if concept wasn't created
        if not hasattr(self, "concept_id"):
            self.skipTest("Concept creation test failed or was skipped")
        
        # Request content generation
        response = self._make_request(
            "gutenberg",
            "/api/v1/content/generate",
            method="POST",
            data={
                "concept_id": self.concept_id,
                "format": "lesson", 
                "style": "concise",
                "target_audience": "beginner"
            }
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Verify response has request ID
        self.assertIn("request_id", data)
        
        # Store request ID for content status check
        self.__class__.content_request_id = data["request_id"]
        logger.info(f"Content generation started with request ID: {self.content_request_id}")
    
    def test_gutenberg_content_status(self):
        """Test checking the status of content generation in Gutenberg."""
        # Skip if content generation wasn't started
        if not hasattr(self, "content_request_id"):
            self.skipTest("Content generation test failed or was skipped")
        
        # Check status
        response = self._make_request(
            "gutenberg",
            f"/api/v1/content/status/{self.content_request_id}"
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Verify response has status
        self.assertIn("status", data)
        logger.info(f"Content generation status: {data['status']}")
        
        # If content is complete, store the content ID
        if data["status"] == "complete" and "content_id" in data:
            self.__class__.content_id = data["content_id"]
            logger.info(f"Content generated with ID: {self.content_id}")
    
    def test_end_to_end_workflow(self):
        """Test the complete workflow from concept creation to content retrieval."""
        # Create new concept specifically for this test
        concept = {
            "name": "End-to-End Test Concept",
            "description": "A concept for testing the complete workflow",
            "domains": ["testing", "workflow"],
            "id": f"e2e_test_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        }
        
        # Step 1: Create concept in Ptolemy
        response = self._make_request(
            "ptolemy", 
            "/api/v1/concepts", 
            method="POST", 
            data=concept
        )
        self.assertEqual(response.status_code, 200)
        concept_data = response.json()
        concept_id = concept_data["id"]
        
        # Step 2: Generate content in Gutenberg
        response = self._make_request(
            "gutenberg",
            "/api/v1/content/generate",
            method="POST",
            data={
                "concept_id": concept_id,
                "format": "lesson", 
                "style": "concise",
                "target_audience": "beginner"
            }
        )
        self.assertEqual(response.status_code, 200)
        request_data = response.json()
        request_id = request_data["request_id"]
        
        # Step 3: Poll for content completion (simplified)
        # In a real test, you would implement polling with timeout
        response = self._make_request(
            "gutenberg",
            f"/api/v1/content/status/{request_id}"
        )
        self.assertEqual(response.status_code, 200)
        
        # Note: A real test would wait for completion
        # This is simplified for demonstration purposes
        
        logger.info("Completed end-to-end workflow test")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up resources after tests complete."""
        # Delete test concept if it was created
        if hasattr(cls, "concept_id"):
            try:
                response = httpx.delete(
                    f"{cls.ptolemy_url}/api/v1/concepts/{cls.concept_id}",
                    headers={"Authorization": f"Bearer {cls.api_key}"} if cls.api_key else {}
                )
                if response.status_code == 200:
                    logger.info(f"Deleted test concept: {cls.concept_id}")
                else:
                    logger.warning(f"Failed to delete test concept: {response.text}")
            except Exception as e:
                logger.error(f"Error deleting test concept: {str(e)}")


@pytest.mark.integration
def test_ptolemy_search_functionality():
    """Test Ptolemy's search functionality as a standalone test."""
    ptolemy_url = os.environ.get("PTOLEMY_URL", "http://localhost:8000")
    
    # Skip if Ptolemy is not running
    try:
        response = httpx.get(f"{ptolemy_url}/health", timeout=5.0)
        if response.status_code != 200:
            pytest.skip("Ptolemy service is not available")
    except Exception:
        pytest.skip("Ptolemy service is not available")
    
    # Test search endpoint
    search_query = "python programming"
    response = httpx.post(
        f"{ptolemy_url}/api/v1/search", 
        json={"query": search_query, "limit": 5}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify response structure
    assert "results" in data
    assert isinstance(data["results"], list)
    
    # Log results
    logger.info(f"Search results for '{search_query}': {len(data['results'])} results found")