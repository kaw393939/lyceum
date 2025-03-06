#!/usr/bin/env python
"""
Gutenberg Test Utility
======================
A test utility for the Gutenberg content generation system.
Generates test traffic with deliberate errors and analyzes the log file.
"""

import os
import sys
import time
import random
import json
import uuid
import argparse
import logging
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Optional, Tuple, Union

# Configure local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logging_utils import get_logger, create_log_analyzer

# Initialize logger
logger = get_logger("guttenberg_test")

class GutenbergTester:
    """Test utility for Gutenberg system, generating traffic and analyzing logs."""
    
    def __init__(
        self, 
        base_url: str = "http://localhost:8001", 
        log_file: str = "gutenberg.log",
        api_version: str = "v1",
        request_delay: float = 0.5,
        error_rate: float = 0.2,
        timeout: int = 30
    ):
        """
        Initialize the Gutenberg test utility.
        
        Args:
            base_url: Base URL of the Gutenberg API
            log_file: Path to log file for analysis
            api_version: API version to use
            request_delay: Delay between requests in seconds
            error_rate: Probability of generating erroneous requests (0-1)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.api_prefix = f"/api/{api_version}"
        self.api_url = f"{self.base_url}{self.api_prefix}"
        self.log_file = log_file
        self.request_delay = request_delay
        self.error_rate = min(max(error_rate, 0), 1)  # Clamp to 0-1
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "GutenbergTester/1.0",
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
        
        # Create log analyzer
        self.log_analyzer = create_log_analyzer(
            log_file=self.log_file,
            auto_analyze=False
        )
        
        # Track successful and failed requests
        self.success_count = 0
        self.error_count = 0
        self.endpoints_tested = set()
        self.test_start_time = None
        self.test_end_time = None
        
        logger.info(f"Initialized Gutenberg tester targeting {self.api_url}")
    
    def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        data: Optional[dict] = None, 
        params: Optional[dict] = None,
        headers: Optional[dict] = None,
        expected_error: bool = False
    ) -> Tuple[bool, requests.Response]:
        """
        Make an HTTP request to the API.
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint (without prefix)
            data: Request body data
            params: Query parameters
            headers: Additional headers
            expected_error: Whether this request is expected to fail
            
        Returns:
            Tuple of (success, response)
        """
        url = f"{self.api_url}{endpoint}"
        self.endpoints_tested.add(f"{method} {endpoint}")
        
        req_headers = {}
        if headers:
            req_headers.update(headers)
        
        # Add request ID for tracking
        request_id = str(uuid.uuid4())
        req_headers["X-Request-ID"] = request_id
        
        # Generate a deliberate error if needed
        if not expected_error and random.random() < self.error_rate:
            if data and isinstance(data, dict):
                # Create an invalid request by removing required fields
                if random.choice([True, False]) and data:
                    key_to_remove = random.choice(list(data.keys()))
                    data.pop(key_to_remove, None)
                    logger.debug(f"Removed '{key_to_remove}' to generate validation error")
                # Or add an invalid field value
                else:
                    data["_invalid_field"] = "Invalid value"
                    logger.debug("Added invalid field to generate error")
            expected_error = True
        
        try:
            # Log the request attempt
            logger.info(f"Sending {method} request to {endpoint}")
            
            # Make the request
            start_time = time.time()
            response = self.session.request(
                method=method,
                url=url,
                json=data if data else None,
                params=params,
                headers=req_headers,
                timeout=self.timeout
            )
            duration = time.time() - start_time
            
            # Introduce a random delay for some requests to create slow operations
            if random.random() < 0.1:  # 10% of requests will be slow
                slow_factor = random.uniform(1.5, 5)
                logger.debug(f"Simulating slow operation ({slow_factor:.2f}x)")
                time.sleep(duration * slow_factor)
            
            # Log the response details
            status_code = response.status_code
            is_success = 200 <= status_code < 300
            
            if is_success:
                logger.info(f"Request succeeded: {status_code}, took {duration:.2f}s")
                if not expected_error:
                    self.success_count += 1
                else:
                    logger.warning("Request unexpectedly succeeded")
            else:
                logger.warning(f"Request failed: {status_code}, took {duration:.2f}s")
                if not expected_error:
                    logger.error(f"Unexpected error: {response.text}")
                self.error_count += 1
            
            # Add a small delay between requests
            time.sleep(self.request_delay)
            
            return is_success, response
            
        except requests.RequestException as e:
            logger.error(f"Request exception: {str(e)}")
            self.error_count += 1
            return False, None
    
    def test_health_endpoint(self) -> bool:
        """Test the health check endpoint."""
        success, response = self._make_request("GET", "/health")
        if success:
            logger.info("Health check successful")
            logger.debug(f"Health data: {response.json()}")
            return True
        else:
            logger.error("Health check failed")
            return False
    
    def test_info_endpoint(self) -> bool:
        """Test the info endpoint."""
        success, response = self._make_request("GET", "/info")
        if success:
            logger.info("System info check successful")
            return True
        else:
            logger.error("System info check failed")
            return False
    
    def test_content_generation(self, count: int = 3) -> int:
        """Test content generation with various parameters."""
        success_count = 0
        
        for i in range(count):
            # Create a test content generation request
            request_data = {
                "topic": random.choice([
                    "Stoicism and resilience",
                    "Dichotomy of control",
                    "Epictetus teachings",
                    "Marcus Aurelius Meditations",
                    "Seneca's letters"
                ]),
                "template_id": "default",
                "parameters": {
                    "age_range": random.choice(["10-13", "14-18", "19-22", "adult"]),
                    "difficulty": random.choice(["beginner", "intermediate", "advanced"]),
                    "length": random.choice(["short", "medium", "long"]),
                    "include_media": random.choice([True, False]),
                },
                "metadata": {
                    "requester": "guttenberg_test",
                    "test_id": str(uuid.uuid4()),
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            # Randomly decide if this should be an erroneous request
            expected_error = random.random() < self.error_rate
            
            # Send request
            success, response = self._make_request(
                "POST", 
                "/content/generate", 
                data=request_data,
                expected_error=expected_error
            )
            
            if success:
                success_count += 1
                # Try to retrieve the content we just created
                try:
                    content_id = response.json().get("content_id")
                    if content_id:
                        time.sleep(0.5)  # Give the system time to process
                        self._make_request("GET", f"/content/{content_id}")
                except Exception as e:
                    logger.error(f"Error parsing response: {e}")
        
        return success_count
    
    def test_template_endpoints(self) -> int:
        """Test template-related endpoints."""
        success_count = 0
        
        # List templates
        success, response = self._make_request("GET", "/templates")
        if success:
            success_count += 1
            try:
                templates = response.json().get("templates", [])
                if templates:
                    # Get a specific template
                    template_id = templates[0].get("id", "default")
                    success, _ = self._make_request("GET", f"/templates/{template_id}")
                    if success:
                        success_count += 1
            except (ValueError, AttributeError) as e:
                logger.error(f"Error parsing templates: {e}")
        
        return success_count
    
    def test_error_cases(self, count: int = 5) -> None:
        """Deliberately test error cases."""
        for i in range(count):
            # Pick a random error case to test
            error_case = random.choice([
                # Invalid endpoint
                lambda: self._make_request("GET", f"/nonexistent_endpoint_{uuid.uuid4()}", expected_error=True),
                # Malformed JSON
                lambda: self._make_request("POST", "/content/generate", data="{invalid_json", expected_error=True),
                # Invalid content ID
                lambda: self._make_request("GET", f"/content/{uuid.uuid4()}", expected_error=True),
                # Invalid template ID
                lambda: self._make_request("GET", f"/templates/nonexistent_{uuid.uuid4()}", expected_error=True),
                # Invalid request method
                lambda: self._make_request("PUT", "/health", expected_error=True)
            ])
            error_case()
    
    def test_log_analysis(self) -> Dict[str, Any]:
        """Test the log analysis endpoints."""
        results = {}
        
        # Check if log_analyzer is available
        success, response = self._make_request("GET", "/logs/analysis")
        if success:
            logger.info("Log analysis endpoint available")
            results["analysis_available"] = True
            
            # Test manual analysis trigger
            success, response = self._make_request("POST", "/logs/analysis/run", params={"save": True})
            if success:
                results["manual_analysis"] = True
                logger.info("Manual log analysis triggered successfully")
            
            # Get list of reports
            success, response = self._make_request("GET", "/logs/analysis/reports")
            if success and response.json().get("reports"):
                reports = response.json().get("reports", [])
                results["report_count"] = len(reports)
                logger.info(f"Found {len(reports)} analysis reports")
                
                # Try to get the first report
                if reports:
                    report_filename = reports[0].get("filename")
                    if report_filename:
                        success, response = self._make_request(
                            "GET", f"/logs/analysis/reports/{report_filename}"
                        )
                        results["report_retrieval"] = success
        else:
            logger.warning("Log analysis endpoint not available")
            results["analysis_available"] = False
        
        return results
    
    def run_comprehensive_test(self, duration: int = 60, max_requests: int = 100) -> Dict[str, Any]:
        """
        Run a comprehensive test of the system for a specified duration.
        
        Args:
            duration: Test duration in seconds
            max_requests: Maximum number of requests to make
        
        Returns:
            Test results summary
        """
        logger.info(f"Starting comprehensive test (duration: {duration}s, max requests: {max_requests})")
        self.test_start_time = datetime.now()
        
        # Reset counters
        self.success_count = 0
        self.error_count = 0
        self.endpoints_tested = set()
        
        # Test health and info endpoints first
        self.test_health_endpoint()
        self.test_info_endpoint()
        
        # Define test actions with their weights
        test_actions = [
            (self.test_health_endpoint, 10),
            (self.test_info_endpoint, 10),
            (lambda: self.test_content_generation(1), 40),
            (self.test_template_endpoints, 20),
            (lambda: self.test_error_cases(1), 20)
        ]
        
        # Calculate total weight for weighted random selection
        total_weight = sum(weight for _, weight in test_actions)
        
        # Run tests until duration is exceeded or max requests reached
        end_time = time.time() + duration
        request_count = 0
        
        while time.time() < end_time and request_count < max_requests:
            # Select a random test action based on weights
            rand_val = random.randint(1, total_weight)
            cumulative_weight = 0
            
            for action, weight in test_actions:
                cumulative_weight += weight
                if rand_val <= cumulative_weight:
                    action()
                    break
            
            request_count += 1
        
        # Run log analysis tests at the end
        log_analysis_results = self.test_log_analysis()
        
        self.test_end_time = datetime.now()
        test_duration = (self.test_end_time - self.test_start_time).total_seconds()
        
        # Compile test results
        results = {
            "start_time": self.test_start_time.isoformat(),
            "end_time": self.test_end_time.isoformat(),
            "duration_seconds": test_duration,
            "requests_made": request_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "success_rate": self.success_count / max(1, (self.success_count + self.error_count)),
            "endpoints_tested": list(self.endpoints_tested),
            "log_analysis": log_analysis_results
        }
        
        logger.info(f"Test completed: {self.success_count} successes, {self.error_count} errors")
        return results
    
    def analyze_logs(self) -> Dict[str, Any]:
        """Analyze the log file and generate a report."""
        logger.info("Analyzing logs...")
        
        # Use the start time of our test if available
        since = self.test_start_time if self.test_start_time else datetime.now() - timedelta(hours=1)
        
        try:
            # Run the analysis
            analysis = self.log_analyzer.analyze(since=since)
            
            # Extract key findings
            key_findings = {
                "error_count": analysis["error_summary"]["total_errors"],
                "unique_error_types": analysis["error_summary"]["unique_error_types"],
                "most_common_errors": analysis["error_summary"]["most_common_errors"][:3],
                "slow_operations": analysis["performance_summary"]["slow_operations_count"],
                "slowest_operations": analysis["performance_summary"]["slowest_operations"][:3],
                "api_calls": analysis["api_summary"]["total_api_calls"],
                "most_used_endpoints": analysis["api_summary"]["most_used_endpoints"][:3],
                "recommendations": analysis["recommendations"][:5]
            }
            
            logger.info(f"Log analysis complete. Found {key_findings['error_count']} errors, "
                       f"{key_findings['slow_operations']} slow operations")
            
            return {
                "success": True,
                "key_findings": key_findings,
                "full_analysis": analysis,
                "analysis_time": analysis["analysis_time"]
            }
            
        except Exception as e:
            logger.error(f"Error analyzing logs: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def save_results(self, test_results: Dict, analysis_results: Dict, filename: str = None) -> str:
        """Save test and analysis results to a file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"gutenberg_test_results_{timestamp}.json"
        
        # Combine results
        combined_results = {
            "test_results": test_results,
            "analysis_results": analysis_results,
            "timestamp": datetime.now().isoformat(),
            "target_api": self.api_url,
            "log_file": self.log_file
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(combined_results, f, indent=2, default=str)  # Use default=str to handle non-serializable objects
            logger.info(f"Results saved to {filename}")
            return filename
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            return None
    
    def print_summary(self, test_results: Dict, analysis_results: Dict) -> None:
        """Print a summary of test and analysis results."""
        print("\n" + "="*80)
        print(f"GUTENBERG TEST SUMMARY")
        print("="*80)
        
        print(f"\nTest Duration: {test_results['duration_seconds']:.2f} seconds")
        print(f"Requests: {test_results['requests_made']} total, "
              f"{test_results['success_count']} successful, "
              f"{test_results['error_count']} errors")
        print(f"Success Rate: {test_results['success_rate']*100:.1f}%")
        
        print("\nEndpoints Tested:")
        for endpoint in sorted(test_results['endpoints_tested']):
            print(f"  - {endpoint}")
        
        if analysis_results["success"]:
            findings = analysis_results["key_findings"]
            print("\nLog Analysis Key Findings:")
            print(f"  - {findings['error_count']} total errors across {findings['unique_error_types']} error types")
            print(f"  - {findings['slow_operations']} slow operations detected")
            print(f"  - {findings['api_calls']} total API calls logged")
            
            print("\nMost Common Errors:")
            for error in findings["most_common_errors"]:
                print(f"  - {error['type']}: {error['count']} occurrences")
            
            print("\nSlowest Operations:")
            for op in findings["slowest_operations"]:
                print(f"  - {op['operation']}: {op['avg_time']:.2f}s average")
            
            print("\nTop Recommendations:")
            for rec in findings["recommendations"]:
                print(f"  - {rec}")
        else:
            print("\nLog Analysis Failed:", analysis_results["error"])
        
        print("\n" + "="*80 + "\n")


def main():
    """Main entry point for the test utility."""
    parser = argparse.ArgumentParser(description="Gutenberg System Test Utility")
    parser.add_argument("--url", default="http://localhost:8001", help="Base URL of the Gutenberg API")
    parser.add_argument("--log-file", default="gutenberg.log", help="Path to the log file to analyze")
    parser.add_argument("--duration", type=int, default=60, help="Test duration in seconds")
    parser.add_argument("--max-requests", type=int, default=100, help="Maximum number of requests to make")
    parser.add_argument("--error-rate", type=float, default=0.2, help="Probability of generating erroneous requests (0-1)")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between requests in seconds")
    parser.add_argument("--output", help="Output file for test results (default: auto-generated)")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze existing logs, don't generate test traffic")
    
    args = parser.parse_args()
    
    # Initialize the tester
    tester = GutenbergTester(
        base_url=args.url,
        log_file=args.log_file,
        error_rate=args.error_rate,
        request_delay=args.delay
    )
    
    if not args.analyze_only:
        # Run the comprehensive test
        print(f"Starting Gutenberg test against {args.url}")
        print(f"Test parameters: duration={args.duration}s, max_requests={args.max_requests}, error_rate={args.error_rate}")
        
        test_results = tester.run_comprehensive_test(
            duration=args.duration,
            max_requests=args.max_requests
        )
    else:
        # Skip testing, just set up minimal test results
        print("Skipping test generation, analyzing existing logs only")
        test_results = {
            "start_time": datetime.now().isoformat(),
            "end_time": datetime.now().isoformat(),
            "duration_seconds": 0,
            "requests_made": 0,
            "success_count": 0,
            "error_count": 0,
            "success_rate": 0,
            "endpoints_tested": [],
            "log_analysis": {"analysis_available": False}
        }
    
    # Analyze the logs
    print(f"Analyzing logs from {args.log_file}")
    analysis_results = tester.analyze_logs()
    
    # Save the results
    results_file = tester.save_results(test_results, analysis_results, args.output)
    if results_file:
        print(f"Results saved to {results_file}")
    
    # Print summary
    tester.print_summary(test_results, analysis_results)


if __name__ == "__main__":
    main()