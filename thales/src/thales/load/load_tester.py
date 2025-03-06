import os
import time
import json
import math
import statistics
import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

import aiohttp


class LoadTester:
    """Comprehensive load testing framework for microservices."""
    
    def __init__(
        self, 
        target_url: str, 
        concurrency: int = 10, 
        verbose: bool = False
    ):
        """Initialize the load tester.
        
        Args:
            target_url: Base URL of the target service
            concurrency: Maximum number of concurrent requests
            verbose: Whether to print progress information
        """
        self.target_url = target_url
        self.concurrency = concurrency
        self.verbose = verbose
        self.results = []
        
        # Set up logging
        self.logger = logging.getLogger("load_tester")
        self.logger.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(console_handler)
    
    async def run_test(
        self, 
        endpoint: str, 
        num_requests: int, 
        payload: Optional[Dict[str, Any]] = None, 
        method: str = "GET"
    ) -> Dict[str, Any]:
        """Runs a load test against the specified endpoint.
        
        Args:
            endpoint: API endpoint to test
            num_requests: Number of requests to send
            payload: Optional JSON payload for POST/PUT requests
            method: HTTP method to use
            
        Returns:
            Dictionary with test results
        """
        self.results = []
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for i in range(num_requests):
                task = self._timed_request(
                    session, 
                    f"{self.target_url}{endpoint}",
                    method=method,
                    payload=payload
                )
                tasks.append(task)
                
            if self.concurrency < num_requests:
                # Process in batches to control concurrency
                results = []
                for i in range(0, num_requests, self.concurrency):
                    batch = tasks[i:i+self.concurrency]
                    batch_results = await asyncio.gather(*batch)
                    results.extend(batch_results)
                    if self.verbose:
                        self._print_progress(i + len(batch), num_requests)
                self.results = results
            else:
                self.results = await asyncio.gather(*tasks)
                
        return self._analyze_results()
    
    async def _timed_request(
        self, 
        session: aiohttp.ClientSession, 
        url: str, 
        method: str = "GET", 
        payload: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Makes a request and times the response.
        
        Args:
            session: aiohttp session
            url: URL to request
            method: HTTP method to use
            payload: Optional JSON payload for POST/PUT requests
            
        Returns:
            Dictionary with request results
        """
        start_time = time.time()
        status = None
        response_size = 0
        error = None
        
        try:
            if method == "GET":
                async with session.get(url) as response:
                    status = response.status
                    body = await response.read()
                    response_size = len(body)
            elif method == "POST":
                async with session.post(url, json=payload) as response:
                    status = response.status
                    body = await response.read()
                    response_size = len(body)
            elif method == "PUT":
                async with session.put(url, json=payload) as response:
                    status = response.status
                    body = await response.read()
                    response_size = len(body)
            elif method == "DELETE":
                async with session.delete(url) as response:
                    status = response.status
                    body = await response.read()
                    response_size = len(body)
            else:
                raise ValueError(f"Unsupported method: {method}")
        except Exception as e:
            error = str(e)
            
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000
        
        return {
            "url": url,
            "method": method,
            "status": status,
            "duration_ms": duration_ms,
            "response_size": response_size,
            "error": error
        }
    
    def _print_progress(self, current: int, total: int):
        """Prints progress information during the test.
        
        Args:
            current: Current progress
            total: Total number of items
        """
        percent = 100 * current / total
        print(f"Progress: {current}/{total} requests ({percent:.1f}%)")
    
    def _analyze_results(self) -> Dict[str, Any]:
        """Analyzes test results and generates statistics.
        
        Returns:
            Dictionary with test statistics
        """
        if not self.results:
            return {"error": "No results to analyze"}
            
        # Calculate statistics
        durations = [r["duration_ms"] for r in self.results if r["error"] is None and r["status"] is not None]
        if not durations:
            return {"error": "No successful requests to analyze"}
            
        success_count = len(durations)
        error_count = len(self.results) - success_count
        
        stats = {
            "total_requests": len(self.results),
            "successful_requests": success_count,
            "failed_requests": error_count,
            "success_rate": success_count / len(self.results) if self.results else 0,
            "total_duration_ms": sum(durations),
            "average_response_time_ms": statistics.mean(durations) if durations else 0,
            "min_response_time_ms": min(durations) if durations else 0,
            "max_response_time_ms": max(durations) if durations else 0,
            "percentiles": {
                "50": statistics.median(durations) if durations else 0,
                "90": self._percentile(durations, 90) if durations else 0,
                "95": self._percentile(durations, 95) if durations else 0,
                "99": self._percentile(durations, 99) if durations else 0,
            },
            "requests_per_second": success_count / (sum(durations) / 1000) if durations else 0,
            "errors": self._analyze_errors()
        }
        
        return stats
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculates the specified percentile from the data.
        
        Args:
            data: List of values
            percentile: Percentile to calculate (0-100)
            
        Returns:
            Percentile value
        """
        size = len(data)
        if not size:
            return 0
            
        sorted_data = sorted(data)
        k = (size - 1) * percentile / 100
        f = math.floor(k)
        c = math.ceil(k)
        
        if f == c:
            return sorted_data[int(k)]
            
        d0 = sorted_data[int(f)] * (c - k)
        d1 = sorted_data[int(c)] * (k - f)
        return d0 + d1
    
    def _analyze_errors(self) -> Dict[str, Dict[str, int]]:
        """Analyzes error patterns in results.
        
        Returns:
            Dictionary with error analysis
        """
        error_results = [r for r in self.results if r["error"] is not None]
        status_counts = {}
        
        # Count status codes
        for result in self.results:
            if result["status"] is not None:
                status = result["status"]
                status_counts[status] = status_counts.get(status, 0) + 1
                
        # Group errors by type
        error_types = {}
        for result in error_results:
            error = result["error"]
            error_types[error] = error_types.get(error, 0) + 1
            
        return {
            "status_counts": status_counts,
            "error_types": error_types
        }
        
    def generate_report(
        self, 
        title: str = "Load Test Report", 
        output_file: Optional[str] = None
    ) -> str:
        """Generates a detailed HTML report of load test results.
        
        Args:
            title: Report title
            output_file: Optional path to save the report
            
        Returns:
            HTML report
        """
        analysis = self._analyze_results()
        
        # Generate HTML report
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                .stats {{ display: flex; flex-wrap: wrap; }}
                .stat-box {{ 
                    background: #f5f5f5; border-radius: 5px; padding: 15px;
                    margin: 10px; min-width: 200px; flex: 1;
                }}
                .chart {{ margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>{title}</h1>
            <p>Test run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Summary</h2>
            <div class="stats">
                <div class="stat-box">
                    <h3>Requests</h3>
                    <p>Total: {analysis['total_requests']}</p>
                    <p>Successful: {analysis['successful_requests']}</p>
                    <p>Failed: {analysis['failed_requests']}</p>
                    <p>Success Rate: {analysis['success_rate']*100:.2f}%</p>
                </div>
                
                <div class="stat-box">
                    <h3>Response Time (ms)</h3>
                    <p>Average: {analysis['average_response_time_ms']:.2f}</p>
                    <p>Minimum: {analysis['min_response_time_ms']:.2f}</p>
                    <p>Maximum: {analysis['max_response_time_ms']:.2f}</p>
                </div>
                
                <div class="stat-box">
                    <h3>Percentiles (ms)</h3>
                    <p>50%: {analysis['percentiles']['50']:.2f}</p>
                    <p>90%: {analysis['percentiles']['90']:.2f}</p>
                    <p>95%: {analysis['percentiles']['95']:.2f}</p>
                    <p>99%: {analysis['percentiles']['99']:.2f}</p>
                </div>
                
                <div class="stat-box">
                    <h3>Performance</h3>
                    <p>Throughput: {analysis['requests_per_second']:.2f} req/sec</p>
                    <p>Total Duration: {analysis['total_duration_ms']/1000:.2f} sec</p>
                </div>
            </div>
            
            <h2>HTTP Status Codes</h2>
            <table>
                <tr>
                    <th>Status Code</th>
                    <th>Count</th>
                    <th>Percentage</th>
                </tr>
        """
        
        for status, count in analysis["errors"]["status_counts"].items():
            percentage = 100 * count / analysis["total_requests"]
            html += f"""
                <tr>
                    <td>{status}</td>
                    <td>{count}</td>
                    <td>{percentage:.2f}%</td>
                </tr>
            """
            
        html += """
            </table>
            
            <h2>Error Types</h2>
            <table>
                <tr>
                    <th>Error Type</th>
                    <th>Count</th>
                    <th>Percentage</th>
                </tr>
        """
        
        for error, count in analysis["errors"]["error_types"].items():
            percentage = 100 * count / analysis["total_requests"]
            html += f"""
                <tr>
                    <td>{error}</td>
                    <td>{count}</td>
                    <td>{percentage:.2f}%</td>
                </tr>
            """
            
        html += """
            </table>
        </body>
        </html>
        """
        
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "w") as f:
                f.write(html)
                
        return html
