import os
import time
import logging
import pytest
import asyncio
import httpx
import random
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mark all tests as integration tests
pytestmark = [pytest.mark.integration, pytest.mark.services]


@pytest.fixture(scope="module")
def test_concepts():
    """Create a batch of test concepts for load testing."""
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    
    concepts = []
    subjects = ["Python", "JavaScript", "Machine Learning", "Statistics", "Web Development"]
    
    for i in range(20):
        subject = random.choice(subjects)
        concepts.append({
            "name": f"Load Test Concept {i} - {subject} - {timestamp}",
            "description": f"Concept {i} for load testing with {subject}",
            "domains": ["testing", "load_testing", subject.lower().replace(" ", "_")],
            "id": f"load_test_{i}_{timestamp}"
        })
    
    return concepts


@pytest.fixture(scope="module")
def create_test_concepts(check_services_running, ptolemy_url, test_concepts, http_client, api_key):
    """Create multiple test concepts for load testing and clean up after."""
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    # Create concepts
    concept_ids = []
    for concept in test_concepts:
        try:
            response = http_client.post(
                f"{ptolemy_url}/api/v1/concepts",
                json=concept,
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                concept_ids.append(data["id"])
                logger.info(f"Created concept: {data['id']}")
            else:
                logger.warning(f"Failed to create concept: {response.text}")
        except Exception as e:
            logger.error(f"Error creating concept: {str(e)}")
    
    if not concept_ids:
        pytest.skip("Could not create any test concepts")
    
    logger.info(f"Created {len(concept_ids)} test concepts for load testing")
    
    # Return the IDs for use in tests
    yield concept_ids
    
    # Clean up - delete concepts
    for concept_id in concept_ids:
        try:
            http_client.delete(
                f"{ptolemy_url}/api/v1/concepts/{concept_id}",
                headers=headers
            )
            logger.info(f"Deleted concept: {concept_id}")
        except Exception as e:
            logger.warning(f"Failed to delete concept {concept_id}: {e}")


def test_concurrent_content_generation(check_services_running, ptolemy_url, gutenberg_url, 
                                     create_test_concepts, http_client, api_key):
    """Test concurrent content generation requests."""
    concept_ids = create_test_concepts
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    # Generate content for multiple concepts concurrently
    results = []
    start_time = time.time()
    
    # Use ThreadPoolExecutor for concurrent requests
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        
        for i, concept_id in enumerate(concept_ids):
            future = executor.submit(
                _generate_content,
                gutenberg_url=gutenberg_url,
                concept_id=concept_id,
                format="lesson" if i % 2 == 0 else "exercise",
                headers=headers
            )
            futures.append(future)
        
        # Collect results as they complete
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                logger.info(f"Completed request for concept: {result['concept_id']}")
            except Exception as e:
                logger.error(f"Request failed: {str(e)}")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate statistics
    successful = [r for r in results if r["success"]]
    success_rate = len(successful) / len(concept_ids) if concept_ids else 0
    
    logger.info(f"Concurrent content generation test completed:")
    logger.info(f"Total requests: {len(concept_ids)}")
    logger.info(f"Successful requests: {len(successful)}")
    logger.info(f"Success rate: {success_rate * 100:.1f}%")
    logger.info(f"Total time: {total_time:.2f} seconds")
    
    # Assert reasonable success rate
    assert success_rate >= 0.75, f"Success rate too low: {success_rate * 100:.1f}%"


def _generate_content(gutenberg_url, concept_id, format, headers):
    """Helper function to generate content for a concept."""
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                f"{gutenberg_url}/api/v1/content/generate",
                json={
                    "concept_id": concept_id,
                    "format": format,
                    "target_audience": "beginner"
                },
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "concept_id": concept_id,
                    "request_id": data["request_id"]
                }
            else:
                return {
                    "success": False,
                    "concept_id": concept_id,
                    "error": response.text
                }
    except Exception as e:
        return {
            "success": False,
            "concept_id": concept_id,
            "error": str(e)
        }


@pytest.mark.asyncio
async def test_service_resilience(check_services_running, ptolemy_url, gutenberg_url, api_key):
    """Test service resilience under load with occasional timeout simulation."""
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    # Configure test parameters
    test_duration = 30  # seconds
    request_delay = 1.0  # seconds between requests
    timeout_frequency = 0.2  # probability of simulating a timeout
    max_timeout = 10.0  # maximum simulated timeout in seconds
    
    # Create a test concept for the resilience test
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    test_concept = {
        "name": f"Resilience Test Concept - {timestamp}",
        "description": f"Concept for resilience testing",
        "domains": ["testing", "resilience"],
        "id": f"resilience_test_{timestamp}"
    }
    
    # Create the concept
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{ptolemy_url}/api/v1/concepts",
            json=test_concept,
            headers=headers
        )
        
        if response.status_code != 200:
            pytest.skip(f"Could not create test concept: {response.text}")
        
        concept_data = response.json()
        concept_id = concept_data["id"]
        logger.info(f"Created test concept: {concept_id}")
    
    # Run resilience test
    start_time = time.time()
    results = []
    request_count = 0
    
    try:
        # Continue sending requests for the specified duration
        while time.time() - start_time < test_duration:
            # Simulate occasional timeouts by using a very short timeout
            request_timeout = 0.5 if random.random() < timeout_frequency else 30.0
            
            logger.info(f"Sending request {request_count + 1} with timeout: {request_timeout}s")
            
            try:
                async with httpx.AsyncClient(timeout=request_timeout) as client:
                    search_query = random.choice(["Python", "Machine Learning", "Statistics", "Programming"])
                    response = await client.post(
                        f"{ptolemy_url}/api/v1/search",
                        json={"query": search_query, "limit": 5},
                        headers=headers
                    )
                    
                    results.append({
                        "success": response.status_code == 200,
                        "status_code": response.status_code,
                        "query": search_query
                    })
                    
                    logger.info(f"Request {request_count + 1} succeeded with status: {response.status_code}")
            except httpx.TimeoutException:
                results.append({
                    "success": False,
                    "status_code": None,
                    "query": search_query,
                    "error": "timeout"
                })
                logger.info(f"Request {request_count + 1} timed out as expected")
            except Exception as e:
                results.append({
                    "success": False,
                    "status_code": None,
                    "query": search_query,
                    "error": str(e)
                })
                logger.error(f"Request {request_count + 1} failed with error: {str(e)}")
            
            request_count += 1
            await asyncio.sleep(request_delay)
    
    finally:
        # Clean up - delete the test concept
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                await client.delete(
                    f"{ptolemy_url}/api/v1/concepts/{concept_id}",
                    headers=headers
                )
                logger.info(f"Deleted test concept: {concept_id}")
        except Exception as e:
            logger.warning(f"Failed to delete test concept {concept_id}: {e}")
    
    # Analyze results
    successful = [r for r in results if r["success"]]
    timeouts = [r for r in results if not r["success"] and r.get("error") == "timeout"]
    other_errors = [r for r in results if not r["success"] and r.get("error") != "timeout"]
    
    logger.info(f"Resilience test completed:")
    logger.info(f"Total requests: {len(results)}")
    logger.info(f"Successful requests: {len(successful)} ({len(successful)/len(results)*100:.1f}%)")
    logger.info(f"Timeouts: {len(timeouts)} ({len(timeouts)/len(results)*100:.1f}%)")
    logger.info(f"Other errors: {len(other_errors)} ({len(other_errors)/len(results)*100:.1f}%)")
    
    # Success rate should be reasonable given the simulated timeouts
    # Expected success rate: approximately (1 - timeout_frequency)
    expected_success_rate = 1 - timeout_frequency
    actual_success_rate = len(successful) / len(results) if results else 0
    
    # Allow for 20% deviation from expected success rate
    assert actual_success_rate >= expected_success_rate - 0.2, \
        f"Success rate too low: {actual_success_rate*100:.1f}% vs expected {expected_success_rate*100:.1f}%"
    

def test_search_performance(check_services_running, ptolemy_url, api_key):
    """Test search performance with multiple queries."""
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    search_queries = [
        "Python programming",
        "Machine learning algorithms",
        "Data structures",
        "Web development",
        "Software engineering",
        "Database design",
        "Artificial intelligence",
        "Statistics fundamentals",
        "Network protocols",
        "Security best practices"
    ]
    
    results = []
    
    for query in search_queries:
        start_time = time.time()
        
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.post(
                    f"{ptolemy_url}/api/v1/search",
                    json={"query": query, "limit": 10},
                    headers=headers
                )
                
                end_time = time.time()
                duration = end_time - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    result_count = len(data.get("results", []))
                    
                    results.append({
                        "query": query,
                        "success": True,
                        "duration": duration,
                        "result_count": result_count
                    })
                    
                    logger.info(f"Search for '{query}': {result_count} results in {duration:.2f}s")
                else:
                    results.append({
                        "query": query,
                        "success": False,
                        "duration": duration,
                        "error": response.text
                    })
                    
                    logger.warning(f"Search for '{query}' failed: {response.text}")
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            
            results.append({
                "query": query,
                "success": False,
                "duration": duration,
                "error": str(e)
            })
            
            logger.error(f"Search for '{query}' failed with error: {str(e)}")
    
    # Calculate statistics
    successful = [r for r in results if r["success"]]
    if not successful:
        pytest.skip("No successful searches to analyze")
    
    durations = [r["duration"] for r in successful]
    avg_duration = sum(durations) / len(durations)
    max_duration = max(durations)
    min_duration = min(durations)
    
    logger.info(f"Search performance test completed:")
    logger.info(f"Total queries: {len(search_queries)}")
    logger.info(f"Successful queries: {len(successful)}")
    logger.info(f"Average duration: {avg_duration:.2f}s")
    logger.info(f"Min duration: {min_duration:.2f}s")
    logger.info(f"Max duration: {max_duration:.2f}s")
    
    # Assert reasonable performance
    assert avg_duration < 5.0, f"Average search duration too high: {avg_duration:.2f}s"
    assert max_duration < 10.0, f"Maximum search duration too high: {max_duration:.2f}s"