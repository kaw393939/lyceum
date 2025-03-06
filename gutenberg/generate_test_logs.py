#!/usr/bin/env python
"""
Generate Test Log Entries
========================
Creates test log entries to demonstrate log analysis functionality.
"""

import os
import time
import random
import uuid
import logging
from datetime import datetime
import sys
from utils.logging_utils import get_logger, configure_logging

# Configure logging to file
log_file = "gutenberg_test.log"
configure_logging(
    level="DEBUG",
    log_file=log_file, 
    console=True
)

# Get logger
logger = get_logger("test_generator")

def generate_api_request_log(method, path, status_code, duration):
    """Generate a log entry for an API request."""
    request_id = str(uuid.uuid4())
    logger.info(f"Request: {method} {path}")
    # Simulate processing time
    time.sleep(0.01)
    logger.info(f"Response: {status_code} - X-Process-Time: {duration}")

def generate_operation_log(operation, duration, success=True):
    """Generate a log entry for an operation with timing."""
    if success:
        logger.info(f"{operation} completed in {duration}s")
    else:
        logger.error(f"{operation} failed after {duration}s")

def generate_error_log(error_type, message):
    """Generate an error log entry."""
    logger.error(f"{error_type}: {message}")

def main():
    """Generate test log entries."""
    print(f"Generating test log entries in {log_file}")
    
    # Record start time
    start_time = time.time()
    
    # API endpoints to simulate
    api_endpoints = [
        ("GET", "/api/v1/health", 200),
        ("GET", "/api/v1/info", 200),
        ("GET", "/api/v1/templates", 200),
        ("GET", "/api/v1/templates/default", 200),
        ("GET", "/api/v1/templates/nonexistent", 404),
        ("POST", "/api/v1/content/generate", 201),
        ("GET", "/api/v1/content/123456", 200),
        ("GET", "/api/v1/content/nonexistent", 404),
        ("PUT", "/api/v1/content/123456", 200),
        ("DELETE", "/api/v1/content/123456", 204),
        ("GET", "/api/v1/logs/analysis", 200),
        ("POST", "/api/v1/logs/analysis/run", 200)
    ]
    
    # Operations to simulate
    operations = [
        "Content generation",
        "Database query",
        "Template rendering",
        "RAG processing",
        "Image generation",
        "Vector embedding",
        "Authentication",
        "File I/O",
        "API request to external service",
        "Log analysis"
    ]
    
    # Error types to simulate
    error_types = [
        ("ConnectionError", "Failed to connect to external service"),
        ("ValidationError", "Invalid content parameters"),
        ("TimeoutError", "Operation timed out after 30 seconds"),
        ("AuthenticationError", "Invalid API key"),
        ("PermissionError", "User does not have permission to access this resource"),
        ("NotFoundError", "Template not found"),
        ("ServerError", "Internal server error"),
        ("RateLimitExceeded", "Too many requests"),
        ("DatabaseError", "Failed to connect to database"),
        ("ConfigurationError", "Missing required configuration")
    ]
    
    # Generate a series of log entries
    entry_count = 1000  # Total log entries to generate
    
    for i in range(entry_count):
        # Determine what type of log entry to generate
        entry_type = random.choices(
            ["api", "operation", "error"],
            weights=[0.7, 0.2, 0.1],
            k=1
        )[0]
        
        if entry_type == "api":
            # Generate an API request/response log
            method, path, default_status = random.choice(api_endpoints)
            
            # Occasionally generate errors
            if random.random() < 0.1:  # 10% error rate
                status_code = random.choice([400, 401, 403, 404, 500])
            else:
                status_code = default_status
            
            # Generate a realistic duration
            if random.random() < 0.05:  # 5% slow requests
                duration = random.uniform(1.0, 5.0)  # Slow
            else:
                duration = random.uniform(0.01, 0.5)  # Normal
                
            generate_api_request_log(method, path, status_code, duration)
            
        elif entry_type == "operation":
            # Generate an operation log
            operation = random.choice(operations)
            success = random.random() < 0.9  # 90% success rate
            
            # Generate a realistic duration
            if random.random() < 0.05:  # 5% slow operations
                duration = random.uniform(1.0, 10.0)  # Very slow
            else:
                duration = random.uniform(0.05, 0.8)  # Normal
                
            generate_operation_log(operation, duration, success)
            
        elif entry_type == "error":
            # Generate an error log
            error_type, message_template = random.choice(error_types)
            # Add some randomness to the message
            message = f"{message_template} (ID: {uuid.uuid4().hex[:8]})"
            generate_error_log(error_type, message)
        
        # Add a small delay between entries to simulate real-world timing
        time.sleep(0.001)
        
        # Print progress every 100 entries
        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{entry_count} log entries...")
    
    # Record end time and print summary
    end_time = time.time()
    duration = end_time - start_time
    print(f"Finished generating {entry_count} log entries in {duration:.2f} seconds")
    print(f"Log file: {os.path.abspath(log_file)}")
    
    # Suggest next steps
    print("\nNext steps:")
    print(f"1. Run './guttenberg_test.py --log-file {log_file} --analyze-only' to analyze these logs")
    print("2. Run the main application and access /api/v1/logs/analysis to see the analysis via the API")

if __name__ == "__main__":
    main()