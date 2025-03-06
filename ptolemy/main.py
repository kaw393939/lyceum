#!/usr/bin/env python
"""
Ptolemy Knowledge Map System - Main Application
=============================================
Main entry point for the Ptolemy Knowledge Map API.
"""

import os
import sys
import time
import uuid
import argparse
import logging
from collections import defaultdict

import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Optional Prometheus metrics
try:
    from prometheus_client import start_http_server, Counter, Histogram, Gauge
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ptolemy")

# Import configuration and the main router (with its own middleware) from your API routes
from config import get_config
from api.routes import router, log_requests

# Get configuration
config = get_config()

# Create the FastAPI app with enhanced OpenAPI documentation
app = FastAPI(
    title="Ptolemy Knowledge Map API",
    description="""
    Ptolemy is the knowledge mapping service of the Goliath educational platform.
    
    ## Features
    
    * Create and manage educational concepts in a knowledge graph
    * Define relationships between concepts (prerequisites, related concepts, etc.)
    * Generate learning paths based on concept relationships
    * Search concepts by keyword or semantic similarity
    * Export and import knowledge maps in various formats
    
    ## Authentication
    
    Most endpoints require authentication using a Bearer token. Include the token in the Authorization header:
    `Authorization: Bearer {your_token}`
    
    ## Error Responses
    
    The API uses standard HTTP status codes and returns error details in the response body:
    ```json
    {
      "detail": "Error message",
      "status_code": 400,
      "error_type": "ValidationError"
    }
    ```
    
    ## Shared Resources
    
    This service uses a shared vector database (Qdrant) with the Gutenberg service. 
    All vectors include a `service` field with value "ptolemy" for namespacing.
    """,
    version=config.version,
    docs_url=config.api.docs_url,
    redoc_url=config.api.redoc_url,
    openapi_url="/openapi.json",
    terms_of_service="https://example.com/terms/",
    contact={
        "name": "Goliath Platform Team",
        "url": "https://example.com/contact/",
        "email": "support@example.com",
    },
    license_info={
        "name": "Proprietary",
        "url": "https://example.com/license/",
    },
    openapi_tags=[
        {
            "name": "concepts",
            "description": "Operations for managing educational concepts in the knowledge graph",
            "externalDocs": {
                "description": "Concept documentation",
                "url": "https://example.com/docs/concepts/",
            },
        },
        {
            "name": "relationships",
            "description": "Define and manage relationships between concepts",
        },
        {
            "name": "learning paths",
            "description": "Generate personalized learning paths based on concept relationships",
        },
        {
            "name": "domains",
            "description": "Manage top-level knowledge domains",
        },
        {
            "name": "search",
            "description": "Search concepts by keyword or semantic similarity",
        },
        {
            "name": "admin",
            "description": "Administrative operations for managing the system",
        },
        {
            "name": "analytics",
            "description": "Analytics and reporting on knowledge graph usage",
        },
    ]
)

# Add CORS middleware using the configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register request logging middleware (from the routes module)
app.middleware("http")(log_requests)

# Add rate limiting middleware if enabled in config
if config.api.rate_limit_enabled:
    async def rate_limiter(request: Request, call_next):
        if not hasattr(rate_limiter, "request_counts"):
            rate_limiter.request_counts = defaultdict(list)
        client_ip = request.client.host
        current_time = time.time()
        # Remove outdated request timestamps
        rate_limiter.request_counts[client_ip] = [
            t for t in rate_limiter.request_counts[client_ip]
            if current_time - t < config.api.rate_limit_period
        ]
        if len(rate_limiter.request_counts[client_ip]) >= config.api.rate_limit_requests:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Rate limit exceeded. Please try again later."}
            )
        rate_limiter.request_counts[client_ip].append(current_time)
        return await call_next(request)
    
    app.middleware("http")(rate_limiter)

# Additional middleware to add a unique request ID and processing time header
@app.middleware("http")
async def add_process_time_and_request_id(request: Request, call_next):
    start_time = time.time()
    request_id = str(uuid.uuid4())
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.3f}"
    response.headers["X-Request-ID"] = request_id
    return response

# Include the API routes
app.include_router(router)

# Startup and shutdown event handlers
@app.on_event("startup")
async def startup_event():
    logger.info(f"Starting Ptolemy Knowledge Map API v{config.version}")
    if PROMETHEUS_AVAILABLE and getattr(config, "metrics", None) and config.metrics.enabled:
        start_http_server(config.metrics.prometheus_port)
        logger.info(f"Prometheus metrics server started on port {config.metrics.prometheus_port}")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Ptolemy Knowledge Map API")
    # Place any resource cleanup here if needed

def main():
    parser = argparse.ArgumentParser(description="Ptolemy Knowledge Map System API")
    parser.add_argument("--host", type=str, default=config.api.host, help="API host")
    parser.add_argument("--port", type=int, default=config.api.port, help="API port")
    args = parser.parse_args()

    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        log_level=(config.logging.level.value.lower() if hasattr(config.logging, "level") 
                   else "info")
    )

if __name__ == "__main__":
    main()
