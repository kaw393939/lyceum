#!/usr/bin/env python3
"""
Ptolemy Knowledge Map API - Enhanced Diagnostic Tool v1.1.0

A comprehensive diagnostic tool for testing and troubleshooting
the Ptolemy Knowledge Map API installation, with enhanced error detection
and targeted recommendations.

Usage:
  python ptolemy_diagnostics.py [--url URL] [--api-key KEY] [--verbose] 
                               [--report FILE] [--fix-suggestions]
                               [--no-cleanup] [--timeout SECONDS]
"""

import os
import sys
import json
import time
import uuid
import argparse
import requests
import logging
import datetime
import platform
import socket
import traceback
from typing import Dict, List, Any, Optional, Tuple, Set
from tabulate import tabulate
from colorama import init, Fore, Style

# Initialize colorama for cross-platform colored output
init()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ptolemy-diagnostics")

# Constants
DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_TIMEOUT = 15  # increased from 10 seconds
VERSION = "1.1.0"

class PtolemyDiagnostics:
    def __init__(self, url: str, api_key: Optional[str] = None, 
                verbose: bool = False, timeout: int = DEFAULT_TIMEOUT,
                fix_suggestions: bool = False):
        self.base_url = url.rstrip('/')
        self.api_key = api_key
        self.verbose = verbose
        self.timeout = timeout
        self.fix_suggestions = fix_suggestions
        self.results = {
            "summary": {
                "start_time": datetime.datetime.now().isoformat(),
                "environment": self._get_environment_info(),
                "tests_passed": 0,
                "tests_failed": 0,
                "tests_skipped": 0,
                "critical_issues": 0
            },
            "tests": [],
            "identified_issues": []
        }
        self.current_concept_ids = []
        self.current_relationship_ids = []
        self.current_learning_path_ids = []
        self.current_domain_ids = []
        self.has_openapi_doc = False
        self.openapi_endpoints = {}
        self.detected_issues = set()
        self.max_retries = 2

    def _get_environment_info(self) -> Dict[str, Any]:
        """Gather information about the environment"""
        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "hostname": socket.gethostname(),
            "timestamp": datetime.datetime.now().isoformat(),
            "requests_version": requests.__version__,
            "diagnostic_tool_version": VERSION
        }
        
    def _make_request(self, method: str, endpoint: str, data: Any = None, 
                     params: Dict[str, Any] = None, headers: Dict[str, str] = None,
                     expected_status: int = None, timeout: int = None,
                     retries: int = None) -> Tuple[Optional[requests.Response], str]:
        """Make an HTTP request to the API with detailed error handling and retries"""
        url = f"{self.base_url}{endpoint}"
        
        # Use instance timeout if none provided
        if timeout is None:
            timeout = self.timeout
        
        # Use instance max_retries if none provided
        if retries is None:
            retries = self.max_retries
            
        # Prepare headers
        request_headers = {}
        if self.api_key:
            request_headers["Authorization"] = f"Bearer {self.api_key}"
        
        if headers:
            request_headers.update(headers)
            
        # For JSON data
        if isinstance(data, (dict, list)) and 'Content-Type' not in request_headers:
            request_headers['Content-Type'] = 'application/json'
            
        error_msg = ""
        response = None
        retry_count = 0
        
        while retry_count <= retries:
            try:
                if self.verbose:
                    if retry_count > 0:
                        logger.info(f"Retry {retry_count} for {method} request to {url}")
                    else:
                        logger.info(f"Making {method} request to {url}")
                    if params:
                        logger.info(f"Parameters: {params}")
                    if data and self.verbose > 1:
                        logger.info(f"Data: {data}")
                        
                if method.upper() == "GET":
                    response = requests.get(url, headers=request_headers, params=params, timeout=timeout)
                elif method.upper() == "POST":
                    if isinstance(data, (dict, list)):
                        response = requests.post(url, headers=request_headers, json=data, params=params, timeout=timeout)
                    else:
                        response = requests.post(url, headers=request_headers, data=data, params=params, timeout=timeout)
                elif method.upper() == "PUT":
                    response = requests.put(url, headers=request_headers, json=data, params=params, timeout=timeout)
                elif method.upper() == "DELETE":
                    response = requests.delete(url, headers=request_headers, params=params, timeout=timeout)
                else:
                    error_msg = f"Unsupported HTTP method: {method}"
                    return None, error_msg
                    
                if self.verbose:
                    logger.info(f"Response status: {response.status_code}")
                    if self.verbose > 1:
                        try:
                            content_type = response.headers.get('Content-Type', '')
                            if 'json' in content_type:
                                logger.info(f"Response: {json.dumps(response.json(), indent=2)}")
                            elif len(response.content) < 1000:
                                logger.info(f"Response: {response.text}")
                            else:
                                logger.info(f"Response: [Content too large to display - {len(response.content)} bytes]")
                        except Exception as e:
                            logger.info(f"Response: [Error parsing response: {e}]")
                    
                # Check for specific error patterns to detect issues
                self._check_response_for_issues(response, endpoint)
                    
                if expected_status and response.status_code != expected_status:
                    error_msg = f"Expected status code {expected_status}, got {response.status_code}"
                    try:
                        content = response.json() if 'application/json' in response.headers.get('Content-Type', '') else response.text
                        error_msg += f"\nResponse: {content}"
                    except:
                        error_msg += f"\nResponse: {response.text[:200]}..."
                        
                    # For certain error types, don't retry
                    if response.status_code in [400, 404, 422]:
                        break
                    
                    # If we have an error but have retries left, try again
                    if retry_count < retries:
                        retry_count += 1
                        time.sleep(0.5 * retry_count)  # Exponential backoff
                        continue
                    else:
                        break
                else:
                    # Success, no need to retry
                    error_msg = ""
                    break
                    
            except requests.exceptions.ConnectionError as e:
                error_msg = f"Connection error: {e}"
                self.detected_issues.add("CONNECTION_ERROR")
            except requests.exceptions.Timeout as e:
                error_msg = f"Request timed out after {timeout}s: {e}"
                self.detected_issues.add("TIMEOUT")
            except requests.exceptions.RequestException as e:
                error_msg = f"Request error: {e}"
            except Exception as e:
                error_msg = f"Unexpected error: {e}"
                
            # If we have an error but have retries left, try again
            if retry_count < retries:
                retry_count += 1
                time.sleep(0.5 * retry_count)  # Exponential backoff
            else:
                break
                
        return response, error_msg
    
    def _check_response_for_issues(self, response, endpoint):
        """Check response for common issues and add to detected issues"""
        if not response:
            return
            
        try:
            # Check if response has JSON content
            if 'application/json' in response.headers.get('Content-Type', ''):
                data = response.json()
                
                # Check for specific errors in API responses
                if isinstance(data, dict):
                    detail = data.get('detail', '')
                    if isinstance(detail, str):
                        if "LLMConfig" in detail and "allow_gpt4" in detail:
                            self.detected_issues.add("LLM_CONFIG_MISSING_ATTRIBUTE")
                        elif "Generated learning path contains no steps" in detail:
                            self.detected_issues.add("LEARNING_PATH_GENERATION_ERROR")
                        elif "Limit value must be at least 1" in detail:
                            self.detected_issues.add("PAGINATION_LIMIT_ERROR")
                        elif "validation error" in detail.lower():
                            self.detected_issues.add("VALIDATION_ERROR")
                            if "difficulty" in detail:
                                self.detected_issues.add("DIFFICULTY_VALIDATION_ERROR")
                            if "embed" in detail.lower():
                                self.detected_issues.add("EMBEDDING_ERROR")
                                
                    # Check for auth issues
                    if response.status_code == 401:
                        self.detected_issues.add("AUTHENTICATION_ERROR")
                    elif response.status_code == 403:
                        self.detected_issues.add("AUTHORIZATION_ERROR")
                        
                    # Check for specific error status in domain generation
                    if endpoint.startswith("/domains/") and "status" in data:
                        status = data.get("status")
                        if status == "failed":
                            self.detected_issues.add("DOMAIN_GENERATION_FAILED")
                            if "error" in data:
                                error = data.get("error", "")
                                if "LLMConfig" in error:
                                    self.detected_issues.add("LLM_CONFIG_MISSING_ATTRIBUTE")
        except Exception as e:
            # Error during issue detection shouldn't stop the diagnostic flow
            if self.verbose:
                logger.warning(f"Error checking response for issues: {e}")

    def _record_test_result(self, test_name: str, category: str, passed: bool, 
                           details: str = "", error: str = "", critical: bool = False,
                           response: requests.Response = None, data: Dict[str, Any] = None) -> None:
        """Record a test result"""
        result = {
            "name": test_name,
            "category": category,
            "status": "passed" if passed else "failed",
            "timestamp": datetime.datetime.now().isoformat(),
            "details": details,
            "error": error,
            "critical": critical
        }
        
        if data:
            result["data"] = data
            
        if response:
            result["response"] = {
                "status_code": response.status_code,
                "elapsed_ms": int(response.elapsed.total_seconds() * 1000)
            }
            try:
                if 'application/json' in response.headers.get('Content-Type', ''):
                    result["response"]["content"] = response.json()
            except:
                pass
        
        self.results["tests"].append(result)
        
        if passed:
            self.results["summary"]["tests_passed"] += 1
            status_text = f"{Fore.GREEN}PASSED{Style.RESET_ALL}"
        else:
            self.results["summary"]["tests_failed"] += 1
            status_text = f"{Fore.RED}FAILED{Style.RESET_ALL}"
            if critical:
                self.results["summary"]["critical_issues"] += 1
        
        if self.verbose or not passed:
            print(f"{status_text} {category}: {test_name}")
            if details:
                print(f"  {details}")
            if error:
                print(f"  {Fore.RED}Error: {error}{Style.RESET_ALL}")

    def test_connectivity(self) -> bool:
        """Test basic connectivity to the API server"""
        print(f"\n{Fore.CYAN}Testing API Connectivity...{Style.RESET_ALL}")
        
        # Test 1: Root endpoint
        response, error = self._make_request("GET", "/")
        
        if response and response.status_code == 200:
            api_info = {}
            try:
                api_info = response.json()
                api_version = api_info.get("version", "unknown")
                api_name = api_info.get("name", "Ptolemy API")
                details = f"Connected to {api_name} version {api_version}"
            except:
                details = "Connected successfully but response was not valid JSON"
                
            self._record_test_result(
                "Root endpoint connection", "Connectivity", 
                passed=True, details=details, response=response,
                data=api_info
            )
            
            # Check if API requires authentication
            auth_required = False
            if "auth" in api_info:
                auth_required = api_info.get("auth", {}).get("required", False)
                if auth_required and not self.api_key:
                    print(f"{Fore.YELLOW}Warning: API requires authentication but no API key provided{Style.RESET_ALL}")
            
        else:
            self._record_test_result(
                "Root endpoint connection", "Connectivity", 
                passed=False, error=error or "Could not connect to API root endpoint",
                critical=True
            )
            return False

        # Test 2: Health endpoint
        response, error = self._make_request("GET", "/health")
        
        if response and response.status_code == 200:
            health_info = {}
            try:
                health_info = response.json()
                status = health_info.get("status", "unknown")
                details = f"Health endpoint status: {status}"
                
                # Check if all services are up
                services = health_info.get("services", {})
                if services:
                    unhealthy_services = [s for s, status in services.items() 
                                         if status.get("status", "").lower() != "up"]
                    if unhealthy_services:
                        details += f"\nUnhealthy services: {', '.join(unhealthy_services)}"
                        for service in unhealthy_services:
                            self.detected_issues.add(f"SERVICE_DOWN_{service.upper()}")
                        
            except:
                details = "Health endpoint returned non-JSON response"
                
            self._record_test_result(
                "Health endpoint check", "Connectivity", 
                passed=True, details=details, response=response,
                data=health_info
            )
        else:
            self._record_test_result(
                "Health endpoint check", "Connectivity", 
                passed=False, error=error or "Health endpoint not available",
                critical=True
            )
            return False

        # Test 3: OpenAPI documentation
        response, error = self._make_request("GET", "/openapi.json")
        
        if response and response.status_code == 200:
            try:
                openapi_spec = response.json()
                path_count = len(openapi_spec.get("paths", {}))
                self.has_openapi_doc = True
                self.openapi_endpoints = openapi_spec.get("paths", {})
                
                # Extract endpoint counts by tag
                tags = {}
                for path, methods in openapi_spec.get("paths", {}).items():
                    for method, info in methods.items():
                        for tag in info.get("tags", []):
                            tags[tag] = tags.get(tag, 0) + 1
                
                tag_info = ", ".join([f"{tag}: {count}" for tag, count in tags.items()])
                details = f"Found OpenAPI documentation with {path_count} endpoints ({tag_info})"
                
                self._record_test_result(
                    "OpenAPI documentation", "Connectivity", 
                    passed=True, details=details, response=response
                )
            except Exception as e:
                self._record_test_result(
                    "OpenAPI documentation", "Connectivity", 
                    passed=False, error=f"OpenAPI documentation is not valid JSON: {str(e)}"
                )
        else:
            self._record_test_result(
                "OpenAPI documentation", "Connectivity", 
                passed=False, error=error or "OpenAPI documentation not available"
            )

        return True

    def test_concepts_endpoints(self) -> bool:
        """Test concepts-related endpoints"""
        print(f"\n{Fore.CYAN}Testing Concept Endpoints...{Style.RESET_ALL}")
        
        # Test 1: Create concept
        concept_data = {
            "name": f"Diagnostic Test Concept {uuid.uuid4().hex[:8]}",
            "description": "A concept created by the diagnostic tool to test API functionality",
            "concept_type": "topic",
            "difficulty": "intermediate",
            "importance": 0.7,
            "complexity": 0.5,
            "keywords": ["diagnostic", "test", "concept"],
            "estimated_learning_time_minutes": 15
        }
        
        response, error = self._make_request(
            "POST", "/concepts/", data=concept_data, expected_status=201
        )
        
        if response and response.status_code == 201:
            try:
                concept = response.json()
                concept_id = concept.get("id")
                self.current_concept_ids.append(concept_id)
                
                details = f"Created concept with ID: {concept_id}"
                self._record_test_result(
                    "Create concept", "Concepts", 
                    passed=True, details=details, response=response
                )
            except Exception as e:
                self._record_test_result(
                    "Create concept", "Concepts", 
                    passed=False, error=f"Failed to parse response: {str(e)}"
                )
                return False
        else:
            self._record_test_result(
                "Create concept", "Concepts", 
                passed=False, error=error or "Failed to create concept",
                critical=True
            )
            return False

        # Now that we have a concept, let's test other endpoints
        
        # Test 2: Get concept by ID
        if self.current_concept_ids:
            concept_id = self.current_concept_ids[0]
            response, error = self._make_request(
                "GET", f"/concepts/{concept_id}", expected_status=200
            )
            
            if response and response.status_code == 200:
                self._record_test_result(
                    "Get concept by ID", "Concepts", 
                    passed=True, details=f"Retrieved concept {concept_id}",
                    response=response
                )
            else:
                self._record_test_result(
                    "Get concept by ID", "Concepts", 
                    passed=False, error=error or "Failed to retrieve concept",
                    critical=True
                )
                return False
                
        # Test 3: List concepts
        response, error = self._make_request(
            "GET", "/concepts/", params={"limit": 5}, expected_status=200
        )
        
        if response and response.status_code == 200:
            try:
                data = response.json()
                concepts = data.get("items", [])
                count = len(concepts)
                details = f"Listed {count} concepts"
                self._record_test_result(
                    "List concepts", "Concepts", 
                    passed=True, details=details, response=response
                )
            except Exception as e:
                self._record_test_result(
                    "List concepts", "Concepts", 
                    passed=False, error=f"Failed to parse response: {str(e)}"
                )
        else:
            self._record_test_result(
                "List concepts", "Concepts", 
                passed=False, error=error or "Failed to list concepts"
            )

        # Test 4: Update concept
        if self.current_concept_ids:
            concept_id = self.current_concept_ids[0]
            update_data = {
                "name": f"Updated Test Concept {uuid.uuid4().hex[:8]}",
                "keywords": ["diagnostic", "test", "updated"]
            }
            
            response, error = self._make_request(
                "PUT", f"/concepts/{concept_id}", data=update_data, expected_status=200
            )
            
            if response and response.status_code == 200:
                self._record_test_result(
                    "Update concept", "Concepts", 
                    passed=True, details=f"Updated concept {concept_id}",
                    response=response
                )
            else:
                self._record_test_result(
                    "Update concept", "Concepts", 
                    passed=False, error=error or "Failed to update concept"
                )

        # Test 5: Create additional concepts for relationship testing
        for i in range(2):
            concept_data = {
                "name": f"Related Test Concept {i} {uuid.uuid4().hex[:6]}",
                "description": f"Test concept {i} for relationship testing",
                "concept_type": "topic",
                "difficulty": "intermediate",
                "importance": 0.6,
                "keywords": ["diagnostic", "relationship", f"concept{i}"]
            }
            
            response, error = self._make_request(
                "POST", "/concepts/", data=concept_data, expected_status=201
            )
            
            if response and response.status_code == 201:
                try:
                    concept = response.json()
                    concept_id = concept.get("id")
                    self.current_concept_ids.append(concept_id)
                except:
                    pass

        # Test 6: Get concept relationships (if we have multiple concepts)
        if len(self.current_concept_ids) > 0:
            concept_id = self.current_concept_ids[0]
            response, error = self._make_request(
                "GET", f"/concepts/{concept_id}/relationships", expected_status=200
            )
            
            if response and response.status_code == 200:
                self._record_test_result(
                    "Get concept relationships", "Concepts", 
                    passed=True, details=f"Retrieved relationships for concept {concept_id}",
                    response=response
                )
            else:
                self._record_test_result(
                    "Get concept relationships", "Concepts", 
                    passed=False, error=error or "Failed to retrieve concept relationships"
                )

        # Test 7: Concept graph endpoint
        if self.current_concept_ids:
            concept_id = self.current_concept_ids[0]
            response, error = self._make_request(
                "GET", f"/concepts/{concept_id}/graph", params={"depth": 1}, expected_status=200
            )
            
            if response and response.status_code == 200:
                self._record_test_result(
                    "Get concept graph", "Concepts", 
                    passed=True, details=f"Retrieved graph for concept {concept_id}",
                    response=response
                )
            else:
                self._record_test_result(
                    "Get concept graph", "Concepts", 
                    passed=False, error=error or "Failed to retrieve concept graph"
                )

        return True

    def test_relationships_endpoints(self) -> bool:
        """Test relationship-related endpoints"""
        print(f"\n{Fore.CYAN}Testing Relationship Endpoints...{Style.RESET_ALL}")
        
        # Skip if we don't have enough concepts
        if len(self.current_concept_ids) < 2:
            self._record_test_result(
                "Relationships tests", "Relationships", 
                passed=False, error="Not enough concepts created to test relationships",
                critical=False
            )
            self.results["summary"]["tests_skipped"] += 1
            return False
            
        # Test 1: Create relationship
        relationship_data = {
            "source_id": self.current_concept_ids[0],
            "target_id": self.current_concept_ids[1],
            "relationship_type": "related_to",
            "strength": 0.8,
            "description": "A test relationship created by the diagnostic tool",
            "bidirectional": False
        }
        
        response, error = self._make_request(
            "POST", "/relationships/", data=relationship_data, expected_status=201
        )
        
        if response and response.status_code == 201:
            try:
                relationship = response.json()
                relationship_id = relationship.get("id")
                self.current_relationship_ids.append(relationship_id)
                
                details = f"Created relationship with ID: {relationship_id}"
                self._record_test_result(
                    "Create relationship", "Relationships", 
                    passed=True, details=details, response=response
                )
            except Exception as e:
                self._record_test_result(
                    "Create relationship", "Relationships", 
                    passed=False, error=f"Failed to parse response: {str(e)}"
                )
                return False
        else:
            self._record_test_result(
                "Create relationship", "Relationships", 
                passed=False, error=error or "Failed to create relationship",
                critical=True
            )
            return False

        # Test 2: Get relationship by ID
        if self.current_relationship_ids:
            relationship_id = self.current_relationship_ids[0]
            response, error = self._make_request(
                "GET", f"/relationships/{relationship_id}", expected_status=200
            )
            
            if response and response.status_code == 200:
                self._record_test_result(
                    "Get relationship by ID", "Relationships", 
                    passed=True, details=f"Retrieved relationship {relationship_id}",
                    response=response
                )
            else:
                self._record_test_result(
                    "Get relationship by ID", "Relationships", 
                    passed=False, error=error or "Failed to retrieve relationship"
                )

        # Test 3: Update relationship
        if self.current_relationship_ids:
            relationship_id = self.current_relationship_ids[0]
            update_data = {
                "strength": 0.9,
                "description": "Updated test relationship"
            }
            
            response, error = self._make_request(
                "PUT", f"/relationships/{relationship_id}", data=update_data, expected_status=200
            )
            
            if response and response.status_code == 200:
                self._record_test_result(
                    "Update relationship", "Relationships", 
                    passed=True, details=f"Updated relationship {relationship_id}",
                    response=response
                )
            else:
                self._record_test_result(
                    "Update relationship", "Relationships", 
                    passed=False, error=error or "Failed to update relationship"
                )

        # Test 4: Create another relationship with different type
        if len(self.current_concept_ids) >= 2:
            relationship_data = {
                "source_id": self.current_concept_ids[0],
                "target_id": self.current_concept_ids[-1],
                "relationship_type": "prerequisite",
                "strength": 0.7,
                "description": "Another test relationship"
            }
            
            response, error = self._make_request(
                "POST", "/relationships/", data=relationship_data, expected_status=201
            )
            
            if response and response.status_code == 201:
                try:
                    relationship = response.json()
                    relationship_id = relationship.get("id")
                    self.current_relationship_ids.append(relationship_id)
                    
                    details = f"Created second relationship with ID: {relationship_id}"
                    self._record_test_result(
                        "Create second relationship", "Relationships", 
                        passed=True, details=details, response=response
                    )
                except:
                    self._record_test_result(
                        "Create second relationship", "Relationships", 
                        passed=False, error="Failed to parse response"
                    )
            else:
                self._record_test_result(
                    "Create second relationship", "Relationships", 
                    passed=False, error=error or "Failed to create second relationship"
                )

        return True

    def test_search_endpoints(self) -> bool:
        """Test search endpoints"""
        print(f"\n{Fore.CYAN}Testing Search Endpoints...{Style.RESET_ALL}")
        
        # Test 1: Text search
        search_term = "diagnostic"  # Should match our test concepts
        response, error = self._make_request(
            "GET", "/search/text", params={"query": search_term}, expected_status=200
        )
        
        if response and response.status_code == 200:
            try:
                results = response.json()
                count = len(results)
                details = f"Found {count} concepts matching '{search_term}'"
                self._record_test_result(
                    "Text search", "Search", 
                    passed=True, details=details, response=response
                )
            except Exception as e:
                self._record_test_result(
                    "Text search", "Search", 
                    passed=False, error=f"Failed to parse response: {str(e)}"
                )
        else:
            self._record_test_result(
                "Text search", "Search", 
                passed=False, error=error or "Failed to perform text search"
            )
            
        # Test 2: Semantic search (if endpoint exists)
        if self.has_openapi_doc and "/search/semantic" in self.openapi_endpoints:
            response, error = self._make_request(
                "GET", "/search/semantic", 
                params={"query": "concept for testing relationships"}, 
                expected_status=200
            )
            
            if response and response.status_code == 200:
                try:
                    results = response.json()
                    count = len(results)
                    details = f"Found {count} semantically similar concepts"
                    self._record_test_result(
                        "Semantic search", "Search", 
                        passed=True, details=details, response=response
                    )
                except Exception as e:
                    self._record_test_result(
                        "Semantic search", "Search", 
                        passed=False, error=f"Failed to parse response: {str(e)}"
                    )
            else:
                self._record_test_result(
                    "Semantic search", "Search", 
                    passed=False, error=error or "Failed to perform semantic search"
                )
        else:
            self._record_test_result(
                "Semantic search", "Search", 
                passed=False, error="Semantic search endpoint not available",
                critical=False
            )
            self.results["summary"]["tests_skipped"] += 1

        return True

    def test_learning_paths_endpoints(self) -> bool:
        """Test learning paths endpoints"""
        print(f"\n{Fore.CYAN}Testing Learning Paths Endpoints...{Style.RESET_ALL}")
        
        # Skip if we don't have enough concepts
        if len(self.current_concept_ids) < 2:
            self._record_test_result(
                "Learning paths tests", "Learning Paths", 
                passed=False, error="Not enough concepts created to test learning paths",
                critical=False
            )
            self.results["summary"]["tests_skipped"] += 1
            return False
            
        # Test 1: Create learning path
        learning_path_data = {
            "goal": "Learn about test concepts with diagnostic tool",
            "learner_level": "beginner",
            "concept_ids": self.current_concept_ids,
            "max_time_minutes": 30,
            "include_assessments": True
        }
        
        # This endpoint might fail legitimately due to LLM config issues
        # Try with extended timeout and handle errors appropriately
        response, error = self._make_request(
            "POST", "/learning-paths/", data=learning_path_data, 
            expected_status=201, timeout=20  # Extended timeout
        )
        
        if response and response.status_code == 201:
            try:
                path = response.json()
                path_id = path.get("id")
                self.current_learning_path_ids.append(path_id)
                
                details = f"Created learning path with ID: {path_id}"
                self._record_test_result(
                    "Create learning path", "Learning Paths", 
                    passed=True, details=details, response=response
                )
                
                # If we successfully create a path, test getting it
                response, error = self._make_request(
                    "GET", f"/learning-paths/{path_id}", expected_status=200
                )
                
                if response and response.status_code == 200:
                    self._record_test_result(
                        "Get learning path", "Learning Paths", 
                        passed=True, details=f"Retrieved learning path {path_id}",
                        response=response
                    )
                else:
                    self._record_test_result(
                        "Get learning path", "Learning Paths", 
                        passed=False, error=error or "Failed to retrieve learning path"
                    )
                    
                # Also test listing learning paths
                response, error = self._make_request(
                    "GET", "/learning-paths/", params={"limit": 5}, expected_status=200
                )
                
                if response and response.status_code == 200:
                    try:
                        data = response.json()
                        paths = data.get("items", [])
                        count = len(paths)
                        details = f"Listed {count} learning paths"
                        self._record_test_result(
                            "List learning paths", "Learning Paths", 
                            passed=True, details=details, response=response
                        )
                    except Exception as e:
                        self._record_test_result(
                            "List learning paths", "Learning Paths", 
                            passed=False, error=f"Failed to parse response: {str(e)}"
                        )
                else:
                    self._record_test_result(
                        "List learning paths", "Learning Paths", 
                        passed=False, error=error or "Failed to list learning paths"
                    )
                
            except Exception as e:
                self._record_test_result(
                    "Create learning path", "Learning Paths", 
                    passed=False, error=f"Failed to parse response: {str(e)}"
                )
        else:
            # This might be a legitimate error if there are LLM configuration issues
            if "Generated learning path contains no steps" in error or "LLMConfig" in error:
                self.detected_issues.add("LEARNING_PATH_GENERATION_ERROR")
                self.detected_issues.add("LLM_CONFIG_MISSING_ATTRIBUTE")
                critical = False
            else:
                critical = True
                
            self._record_test_result(
                "Create learning path", "Learning Paths", 
                passed=False, error=error or "Failed to create learning path",
                critical=critical
            )
            
            # Even if creation fails, try listing to see if endpoint works
            response, error = self._make_request(
                "GET", "/learning-paths/", params={"limit": 5}, expected_status=200
            )
            
            if response and response.status_code == 200:
                self._record_test_result(
                    "List learning paths", "Learning Paths", 
                    passed=True, details="Listed learning paths (empty or existing)",
                    response=response
                )
            
        return True

    def test_domains_endpoints(self) -> bool:
        """Test domains endpoints"""
        print(f"\n{Fore.CYAN}Testing Domain Endpoints...{Style.RESET_ALL}")
        
        # Test 1: Create a small domain - this may timeout or fail due to LLM issues
        domain_data = {
            "domain_name": f"Diagnostic Test Domain {uuid.uuid4().hex[:6]}",
            "domain_description": "A small test domain created by the diagnostic tool",
            "depth": 1,
            "generate_relationships": True,
            "concept_count": 2,  # Keep it minimal
            "key_topics": ["testing", "diagnostics"],
            # Add any parameters that might help avoid known issues
            "difficulty_level": "intermediate"  # Explicitly setting to avoid validation errors
        }
        
        # Try with extended timeout for domain creation
        response, error = self._make_request(
            "POST", "/domains/", data=domain_data, 
            expected_status=201, timeout=30  # Extended timeout
        )
        
        domain_id = None
        
        if response and response.status_code == 201:
            try:
                result = response.json()
                # Domains might return different response formats
                if isinstance(result, dict):
                    domain_id = result.get("domain_id") or result.get("id")
                    if not domain_id and "domain" in result:
                        domain_id = result["domain"].get("id")
                
                if domain_id:
                    self.current_domain_ids.append(domain_id)
                    details = f"Created domain with ID: {domain_id}"
                    self._record_test_result(
                        "Create domain", "Domains", 
                        passed=True, details=details, response=response
                    )
                else:
                    self._record_test_result(
                        "Create domain", "Domains", 
                        passed=False, error="Created domain but could not extract domain ID"
                    )
                    return False
            except Exception as e:
                self._record_test_result(
                    "Create domain", "Domains", 
                    passed=False, error=f"Failed to parse response: {str(e)}"
                )
                return False
        else:
            # This might be a timeout or LLM config issue
            if "timeout" in error.lower():
                self.detected_issues.add("DOMAIN_GENERATION_TIMEOUT")
                critical = False
            elif "validation error" in error.lower() and "difficulty" in error.lower():
                self.detected_issues.add("DIFFICULTY_VALIDATION_ERROR")
                critical = False
            elif "LLMConfig" in error:
                self.detected_issues.add("LLM_CONFIG_MISSING_ATTRIBUTE")
                critical = False
            else:
                critical = True
                
            self._record_test_result(
                "Create domain", "Domains", 
                passed=False, error=error or "Failed to create domain",
                critical=critical
            )
            
            # If we can't create a domain, we might try to check if any existing domains
            # can be found - though this approach is less reliable
            try:
                response, error = self._make_request(
                    "GET", "/concepts/", 
                    params={"concept_type": "domain", "limit": 5}, 
                    expected_status=200
                )
                
                if response and response.status_code == 200:
                    data = response.json()
                    domains = data.get("items", [])
                    if domains:
                        existing_domain_id = domains[0].get("id")
                        if existing_domain_id:
                            domain_id = existing_domain_id
                            # Don't add to self.current_domain_ids since we didn't create it
                            print(f"{Fore.YELLOW}Using existing domain: {domain_id}{Style.RESET_ALL}")
            except:
                pass  # Ignore any errors in this fallback approach
                
            if not domain_id:
                return False
        
        # Test 2: Check domain generation status
        # This is an asynchronous operation in many implementations
        if domain_id:
            # Wait for a moment to let the domain generation start
            time.sleep(1)
            
            response, error = self._make_request(
                "GET", f"/domains/{domain_id}/status", expected_status=200
            )
            
            if response and response.status_code == 200:
                try:
                    status_data = response.json()
                    status = status_data.get("status", "unknown")
                    details = f"Domain generation status: {status}"
                    self._record_test_result(
                        "Domain generation status", "Domains", 
                        passed=True, details=details, response=response
                    )
                except Exception as e:
                    self._record_test_result(
                        "Domain generation status", "Domains", 
                        passed=False, error=f"Failed to parse response: {str(e)}"
                    )
            else:
                self._record_test_result(
                    "Domain generation status", "Domains", 
                    passed=False, error=error or "Failed to get domain generation status"
                )
        
        # Test 3: Get domain structure
        # Note: This might not be ready if generation is asynchronous
        if domain_id:
            response, error = self._make_request(
                "GET", f"/domains/{domain_id}/structure", expected_status=200
            )
            
            if response and response.status_code == 200:
                try:
                    structure = response.json()
                    concepts = structure.get("concepts", [])
                    concept_count = len(concepts)
                    details = f"Domain structure contains {concept_count} concepts"
                    self._record_test_result(
                        "Get domain structure", "Domains", 
                        passed=True, details=details, response=response
                    )
                except Exception as e:
                    self._record_test_result(
                        "Get domain structure", "Domains", 
                        passed=False, error=f"Failed to parse response: {str(e)}"
                    )
            else:
                # Don't mark as failed since domain generation might be in progress
                self._record_test_result(
                    "Get domain structure", "Domains", 
                    passed=False, 
                    error=error or "Failed to get domain structure (domain generation might be in progress)",
                    critical=False
                )
        
        return True

    def test_analytics_endpoints(self) -> bool:
        """Test analytics endpoints"""
        print(f"\n{Fore.CYAN}Testing Analytics Endpoints...{Style.RESET_ALL}")
        
        # Test 1: Concept counts
        # Fix for the issue with limit=0 parameter that was seen in logs
        response, error = self._make_request(
            "GET", "/analytics/concept-counts", expected_status=200
        )
        
        if response and response.status_code == 200:
            try:
                data = response.json()
                total = data.get("total", 0)
                details = f"System has {total} total concepts"
                self._record_test_result(
                    "Concept counts", "Analytics", 
                    passed=True, details=details, response=response
                )
            except Exception as e:
                self._record_test_result(
                    "Concept counts", "Analytics", 
                    passed=False, error=f"Failed to parse response: {str(e)}"
                )
        else:
            if "Limit value must be at least 1" in error:
                self.detected_issues.add("PAGINATION_LIMIT_ERROR")
                
            # Analytics endpoints are often non-critical
            self._record_test_result(
                "Concept counts", "Analytics", 
                passed=False, error=error or "Failed to get concept counts",
                critical=False
            )
            
        # Test 2: Relationship stats
        response, error = self._make_request(
            "GET", "/analytics/relationship-stats", expected_status=200
        )
        
        if response and response.status_code == 200:
            self._record_test_result(
                "Relationship stats", "Analytics", 
                passed=True, details="Retrieved relationship statistics",
                response=response
            )
        else:
            # Analytics endpoints are often non-critical
            self._record_test_result(
                "Relationship stats", "Analytics", 
                passed=False, error=error or "Failed to get relationship statistics",
                critical=False
            )
        
        return True

    def test_admin_endpoints(self) -> bool:
        """Test admin endpoints if API key is provided"""
        print(f"\n{Fore.CYAN}Testing Admin Endpoints...{Style.RESET_ALL}")
        
        if not self.api_key:
            print(f"{Fore.YELLOW}Skipping admin endpoint tests - no API key provided{Style.RESET_ALL}")
            self._record_test_result(
                "Admin endpoints", "Admin", 
                passed=False, error="No API key provided for admin endpoints",
                critical=False
            )
            self.results["summary"]["tests_skipped"] += 1
            return False
        
        # Test 1: System stats
        response, error = self._make_request(
            "GET", "/admin/stats", expected_status=200
        )
        
        if response and response.status_code == 200:
            self._record_test_result(
                "System stats", "Admin", 
                passed=True, details="Retrieved system statistics",
                response=response
            )
        else:
            # May require higher privileges or not be implemented
            self._record_test_result(
                "System stats", "Admin", 
                passed=False, error=error or "Failed to get system statistics",
                critical=False
            )
            
        # Test 2: Cache stats
        response, error = self._make_request(
            "GET", "/admin/cache/stats", expected_status=200
        )
        
        if response and response.status_code == 200:
            self._record_test_result(
                "Cache stats", "Admin", 
                passed=True, details="Retrieved cache statistics",
                response=response
            )
        else:
            # May require higher privileges or not be implemented
            self._record_test_result(
                "Cache stats", "Admin", 
                passed=False, error=error or "Failed to get cache statistics",
                critical=False
            )
        
        return True
        
    def _generate_fix_suggestions(self) -> List[Dict[str, str]]:
        """Generate specific fix suggestions based on detected issues"""
        suggestions = []
        
        # CONNECTION_ERROR
        if "CONNECTION_ERROR" in self.detected_issues:
            suggestions.append({
                "issue": "Connection Error",
                "description": "Unable to connect to the API server",
                "suggestion": "Verify the API server is running and accessible at the specified URL. Check network connectivity and firewall settings."
            })
            
        # TIMEOUT
        if "TIMEOUT" in self.detected_issues:
            suggestions.append({
                "issue": "Request Timeout",
                "description": "Requests to the API are timing out",
                "suggestion": "The server may be overloaded or slow to respond. Check server resources and increase the timeout parameter with --timeout."
            })
            
        # LEARNING_PATH_GENERATION_ERROR
        if "LEARNING_PATH_GENERATION_ERROR" in self.detected_issues:
            suggestions.append({
                "issue": "Learning Path Generation Error",
                "description": "Failed to generate learning paths - error: 'Generated learning path contains no steps'",
                "suggestion": "This is likely due to issues with the LLM configuration. Check the LLM settings in your config file and ensure the allow_gpt4 attribute is defined."
            })
            
        # LLM_CONFIG_MISSING_ATTRIBUTE
        if "LLM_CONFIG_MISSING_ATTRIBUTE" in self.detected_issues:
            suggestions.append({
                "issue": "LLM Configuration Error",
                "description": "The LLM configuration is missing the 'allow_gpt4' attribute",
                "suggestion": """
Update your LLM configuration in the config.py file:

class LLMConfig:
    def __init__(self):
        # ... existing code ...
        self.allow_gpt4 = True  # Add this line
                """
            })
            
        # PAGINATION_LIMIT_ERROR
        if "PAGINATION_LIMIT_ERROR" in self.detected_issues:
            suggestions.append({
                "issue": "Pagination Limit Error",
                "description": "API rejects zero value for limit parameter",
                "suggestion": """
The analytics endpoint(s) don't handle limit=0 parameter correctly. This requires a code fix in api.py.
Look for methods like get_concept_counts() or list_concepts() and modify to handle limit=0 properly:

1. In the list_concepts function, change:
   limit: int = Query(100, ge=1, le=1000, description="Maximum number of concepts to return")
   
   To:
   limit: int = Query(100, ge=0, le=1000, description="Maximum number of concepts to return")

2. Or in the implementation, add a check:
   if limit == 0:
       limit = sys.maxsize  # or some very large number
                """
            })
            
        # DIFFICULTY_VALIDATION_ERROR
        if "DIFFICULTY_VALIDATION_ERROR" in self.detected_issues:
            suggestions.append({
                "issue": "Difficulty Validation Error",
                "description": "Validation error related to difficulty field",
                "suggestion": """
LLM generation appears to be setting invalid difficulty values (numeric instead of string enum).
Check the code that parses LLM output in llm/generation.py to ensure difficulty values are properly validated:

1. Ensure difficulty values are mapped to valid enum values ('beginner', 'intermediate', 'advanced', 'expert')
2. Check for cases where difficulty might be returned as a float (0-1 range) instead of string enum
                """
            })
            
        # DOMAIN_GENERATION_TIMEOUT
        if "DOMAIN_GENERATION_TIMEOUT" in self.detected_issues:
            suggestions.append({
                "issue": "Domain Generation Timeout",
                "description": "Domain generation is taking too long and timing out",
                "suggestion": """
Domain generation is computationally intensive and may involve multiple LLM calls:
1. Reduce the concept_count parameter for domain generation (try 2-3 concepts)
2. Ensure your LLM service has adequate performance
3. Consider using background tasks for domain generation if not already implemented
4. The generate_relationships parameter can be set to False to speed up generation
                """
            })
            
        # SERVICE_DOWN issues
        for issue in self.detected_issues:
            if issue.startswith("SERVICE_DOWN_"):
                service = issue.replace("SERVICE_DOWN_", "").lower()
                suggestions.append({
                    "issue": f"{service.capitalize()} Service Unavailable",
                    "description": f"The {service} service is reported as down",
                    "suggestion": f"""
Check the {service} service:
1. Verify the service is running: 'docker-compose ps'
2. Check logs for errors: 'docker-compose logs {service}'
3. Restart the service: 'docker-compose restart {service}'
4. Verify network connectivity between the API and the {service} service
                    """
                })
                
        return suggestions

    def cleanup_resources(self) -> None:
        """Clean up all resources created during testing"""
        print(f"\n{Fore.CYAN}Cleaning up test resources...{Style.RESET_ALL}")
        
        # Clean up in reverse order of dependencies
        
        # 1. Delete learning paths
        for path_id in self.current_learning_path_ids:
            response, error = self._make_request(
                "DELETE", f"/learning-paths/{path_id}", expected_status=204
            )
            if response and response.status_code in [204, 200]:
                print(f"  Deleted learning path: {path_id}")
            else:
                print(f"  Failed to delete learning path {path_id}: {error}")
        
        # 2. Delete relationships
        for rel_id in self.current_relationship_ids:
            response, error = self._make_request(
                "DELETE", f"/relationships/{rel_id}", expected_status=204
            )
            if response and response.status_code in [204, 200]:
                print(f"  Deleted relationship: {rel_id}")
            else:
                print(f"  Failed to delete relationship {rel_id}: {error}")
        
        # 3. Delete concepts (except domains which might contain other resources)
        # Domains often have cascade deletion and will remove child concepts
        non_domain_concepts = [c for c in self.current_concept_ids if c not in self.current_domain_ids]
        for concept_id in non_domain_concepts:
            response, error = self._make_request(
                "DELETE", f"/concepts/{concept_id}", expected_status=204
            )
            if response and response.status_code in [204, 200]:
                print(f"  Deleted concept: {concept_id}")
            else:
                print(f"  Failed to delete concept {concept_id}: {error}")
                
        # 4. Delete domains (last)
        for domain_id in self.current_domain_ids:
            response, error = self._make_request(
                "DELETE", f"/concepts/{domain_id}", expected_status=204
            )
            if response and response.status_code in [204, 200]:
                print(f"  Deleted domain: {domain_id}")
            else:
                print(f"  Failed to delete domain {domain_id}: {error}")

    def run_all_tests(self, skip_cleanup: bool = False) -> Dict[str, Any]:
        """Run all diagnostic tests in sequence"""
        print(f"\n{Fore.GREEN}======== Ptolemy Knowledge Map API Diagnostic Tool v{VERSION} ========{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Target URL: {self.base_url}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}API Key: {'Provided' if self.api_key else 'Not provided'}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Starting diagnostics at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}")
        
        start_time = time.time()
        
        # First test connectivity
        connectivity_ok = self.test_connectivity()
        
        # Only proceed with other tests if connectivity is OK
        if connectivity_ok:
            # Core functionality tests
            self.test_concepts_endpoints()
            self.test_relationships_endpoints()
            
            # Additional functionality tests
            self.test_search_endpoints()
            self.test_learning_paths_endpoints()
            self.test_domains_endpoints()
            self.test_analytics_endpoints()
            self.test_admin_endpoints()
            
            # Clean up created resources unless explicitly skipped
            if not skip_cleanup:
                self.cleanup_resources()
        
        # Generate fix suggestions if requested
        if self.fix_suggestions:
            self.results["fix_suggestions"] = self._generate_fix_suggestions()
        
        # Finalize results
        end_time = time.time()
        self.results["summary"]["end_time"] = datetime.datetime.now().isoformat()
        self.results["summary"]["duration_seconds"] = round(end_time - start_time, 2)
        self.results["summary"]["detected_issues"] = list(self.detected_issues)
        
        # Print summary
        self.print_summary()
        
        return self.results
    
    def print_summary(self) -> None:
        """Print a summary of test results"""
        print(f"\n{Fore.GREEN}======== Test Results Summary ========{Style.RESET_ALL}")
        
        total_tests = (self.results["summary"]["tests_passed"] + 
                       self.results["summary"]["tests_failed"] +
                       self.results["summary"]["tests_skipped"])
                       
        pass_rate = (self.results["summary"]["tests_passed"] / total_tests * 100) if total_tests > 0 else 0
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {Fore.GREEN}{self.results['summary']['tests_passed']}{Style.RESET_ALL}")
        print(f"Failed: {Fore.RED}{self.results['summary']['tests_failed']}{Style.RESET_ALL}")
        print(f"Skipped: {Fore.YELLOW}{self.results['summary']['tests_skipped']}{Style.RESET_ALL}")
        print(f"Critical Issues: {Fore.RED}{self.results['summary']['critical_issues']}{Style.RESET_ALL}")
        print(f"Pass Rate: {Fore.GREEN}{pass_rate:.1f}%{Style.RESET_ALL}")
        print(f"Duration: {self.results['summary']['duration_seconds']} seconds")
        
        # Print detected issues
        if self.detected_issues:
            print(f"\n{Fore.YELLOW}Detected Issues:{Style.RESET_ALL}")
            for issue in sorted(self.detected_issues):
                print(f"• {issue.replace('_', ' ').title()}")
        
        # Print results by category
        print(f"\n{Fore.CYAN}Results by Category:{Style.RESET_ALL}")
        
        categories = {}
        for test in self.results["tests"]:
            category = test["category"]
            status = test["status"]
            if category not in categories:
                categories[category] = {"passed": 0, "failed": 0, "total": 0}
            
            categories[category]["total"] += 1
            if status == "passed":
                categories[category]["passed"] += 1
            else:
                categories[category]["failed"] += 1
        
        table_data = []
        for category, stats in sorted(categories.items()):
            pass_rate = (stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0
            status = "PASS" if stats["failed"] == 0 else "FAIL"
            color = Fore.GREEN if status == "PASS" else Fore.RED
            
            table_data.append([
                category,
                f"{stats['passed']}/{stats['total']}",
                f"{pass_rate:.1f}%",
                f"{color}{status}{Style.RESET_ALL}"
            ])
        
        print(tabulate(table_data, headers=["Category", "Passed/Total", "Pass Rate", "Status"], tablefmt="simple"))
        
        # Print critical failures if any
        critical_failures = [t for t in self.results["tests"] 
                           if t["status"] == "failed" and t["critical"]]
        
        if critical_failures:
            print(f"\n{Fore.RED}Critical Failures:{Style.RESET_ALL}")
            for failure in critical_failures:
                print(f"• {failure['category']}: {failure['name']}")
                print(f"  Error: {failure['error']}")
                
        # Print fix suggestions
        if self.fix_suggestions and "fix_suggestions" in self.results:
            print(f"\n{Fore.GREEN}Fix Suggestions:{Style.RESET_ALL}")
            
            if not self.results["fix_suggestions"]:
                print("No specific fix suggestions available for the detected issues.")
            
            for i, suggestion in enumerate(self.results["fix_suggestions"], 1):
                print(f"\n{Fore.CYAN}Issue {i}: {suggestion['issue']}{Style.RESET_ALL}")
                print(f"Description: {suggestion['description']}")
                print(f"Suggestion:\n{suggestion['suggestion'].strip()}")
        
        # Recommendations
        print(f"\n{Fore.CYAN}Recommendations:{Style.RESET_ALL}")
        
        if "LLM_CONFIG_MISSING_ATTRIBUTE" in self.detected_issues:
            print(f"{Fore.YELLOW}• Fix LLM configuration by adding 'allow_gpt4' attribute in config.py{Style.RESET_ALL}")
            
        if "PAGINATION_LIMIT_ERROR" in self.detected_issues:
            print(f"{Fore.YELLOW}• Fix pagination handling to support limit=0 in analytics endpoints{Style.RESET_ALL}")
            
        if "DIFFICULTY_VALIDATION_ERROR" in self.detected_issues:
            print(f"{Fore.YELLOW}• Fix difficulty validation in domain generation{Style.RESET_ALL}")
            
        if self.results["summary"]["critical_issues"] > 0:
            print(f"{Fore.RED}• Fix critical issues before proceeding{Style.RESET_ALL}")
            
        if not self.api_key and any(t["error"] and "unauthorized" in t["error"].lower() for t in self.results["tests"]):
            print("• Provide a valid API key to access protected endpoints")
            
        if not self.fix_suggestions and self.detected_issues:
            print(f"• Run with --fix-suggestions to get specific solutions for detected issues")
            
        if "TIMEOUT" in self.detected_issues:
            print(f"• Increase request timeout with --timeout parameter")
            
        if "DOMAIN_GENERATION_TIMEOUT" in self.detected_issues:
            print(f"• Use smaller concept counts for domain generation")
            
        connectivity_ok = any(t["category"] == "Connectivity" and t["status"] == "passed" for t in self.results["tests"])
        if connectivity_ok:
            if not any(t["category"] == "Concepts" and t["status"] == "passed" for t in self.results["tests"]):
                print("• Core Concepts functionality is not working correctly")
        else:
            print("• Check the server is running and accessible at the specified URL")
            
        print(f"\n{Fore.GREEN}Diagnostics complete at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}")

    def save_report(self, output_file: str) -> None:
        """Save test results to a file"""
        try:
            with open(output_file, 'w') as f:
                json.dump(self.results, f, indent=2)
            print(f"{Fore.GREEN}Report saved to: {output_file}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error saving report: {str(e)}{Style.RESET_ALL}")


def get_arguments():
    parser = argparse.ArgumentParser(description="Ptolemy Knowledge Map API Diagnostic Tool")
    parser.add_argument("--url", default=os.environ.get("PTOLEMY_API_URL", DEFAULT_API_URL),
                      help=f"API base URL (default: {DEFAULT_API_URL})")
    parser.add_argument("--api-key", default=os.environ.get("PTOLEMY_API_KEY"),
                      help="API key for authentication")
    parser.add_argument("--verbose", "-v", action="count", default=0,
                      help="Increase output verbosity (can be used multiple times)")
    parser.add_argument("--report", help="Save detailed report to specified file (JSON format)")
    parser.add_argument("--no-cleanup", action="store_true", 
                      help="Skip cleaning up created test resources")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                      help=f"Request timeout in seconds (default: {DEFAULT_TIMEOUT})")
    parser.add_argument("--fix-suggestions", action="store_true",
                      help="Generate detailed fix suggestions for detected issues")
    return parser.parse_args()

def main():
    args = get_arguments()
    
    diagnostics = PtolemyDiagnostics(
        url=args.url,
        api_key=args.api_key,
        verbose=(args.verbose > 0),
        timeout=args.timeout,
        fix_suggestions=args.fix_suggestions
    )
    
    try:
        # Run all tests
        results = diagnostics.run_all_tests(skip_cleanup=args.no_cleanup)
        
        # Save report if requested
        if args.report:
            diagnostics.save_report(args.report)
        
        # Return error code based on critical issues
        if results["summary"]["critical_issues"] > 0:
            sys.exit(1)
        else:
            sys.exit(0)
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Diagnostic interrupted by user.{Style.RESET_ALL}")
        try:
            # Try to clean up even when interrupted
            if not args.no_cleanup:
                diagnostics.cleanup_resources()
        except:
            pass
        sys.exit(130)  # 130 is the standard exit code for SIGINT
    except Exception as e:
        print(f"\n{Fore.RED}Diagnostic failed with error: {e}{Style.RESET_ALL}")
        print(traceback.format_exc())
        sys.exit(2)

if __name__ == "__main__":
    main()