#!/usr/bin/env python
"""
Gutenberg - Educational Content Generator
========================================
Main application entry point for the Gutenberg content generation system.
Integrates with Ptolemy knowledge map generator to create rich educational content
with Stoic principles and adaptive learning paths.
"""

import os
import sys
import asyncio
import logging
import json
import time
import uuid
import glob
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

import uvicorn
import httpx
from fastapi import FastAPI, Request, Response, HTTPException, BackgroundTasks, Depends, Query, status
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import local modules
from config.settings import get_config
from utils.logging_utils import get_logger, configure_logging
from utils.error_handling import (
    AppError, NotFoundError, ValidationError, log_exception, 
    format_exception_response, exception_handler_factory
)
from api.routes import router as api_router
from core.content_generator import ContentGenerator
from core.template_engine import TemplateEngine
from core.rag_processor import RAGProcessor
from core.media_generator import MediaGenerator
from models.content import ContentRequest, ContentStatus
from services.content_service import ContentService
from services.template_service import TemplateService
from integrations.mongodb_service import MongoDBService
from integrations.llm_service import LLMService
from integrations.ptolemy_client import PtolemyClient

# Configure logging
logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan events for FastAPI application.
    This handles startup and shutdown events.
    """
    # Startup
    logger.info("Starting Gutenberg Content Generation System...")
    config = get_config()
    
    # Ensure required directories exist
    os.makedirs(config.general.storage_path, exist_ok=True)
    os.makedirs(config.general.tmp_path, exist_ok=True)
    os.makedirs(config.templates.templates_dir, exist_ok=True)
    
    # Create the reports directory if it doesn't exist
    reports_dir = os.path.join(config.general.storage_path, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    # Ensure generated content directory exists
    content_dir = "generated_content"
    os.makedirs(content_dir, exist_ok=True)

    # Initialize services
    try:
        # Initialize MongoDB service
        mongodb_service = MongoDBService()
        app.state.mongodb = mongodb_service
        logger.info("MongoDB service initialized")
        
        # Initialize LLM service
        llm_service = LLMService()
        app.state.llm = llm_service
        logger.info("LLM service initialized")
        
        # Initialize RAG processor
        rag_processor = RAGProcessor()
        app.state.rag_processor = rag_processor
        logger.info("RAG processor initialized")
        
        # Initialize Ptolemy client
        ptolemy_client = PtolemyClient()
        app.state.ptolemy = ptolemy_client
        logger.info("Ptolemy client initialized")
        
        # Initialize template engine
        template_engine = TemplateEngine()
        app.state.template_engine = template_engine
        logger.info("Template engine initialized")
        
        # Initialize template service
        template_service = TemplateService(mongodb_service)
        app.state.template_service = template_service
        logger.info("Template service initialized")
        
        # Initialize media generator
        media_generator = MediaGenerator()
        app.state.media_generator = media_generator
        logger.info("Media generator initialized")
        
        # Initialize content generator
        content_generator = ContentGenerator()
        app.state.content_generator = content_generator
        logger.info("Content generator initialized")
        
        # Initialize content service
        content_service = ContentService(
            mongodb=mongodb_service,
            content_generator=content_generator
        )
        app.state.content_service = content_service
        logger.info("Content service initialized")
        
        # Initialize HTTP client for external API calls
        app.state.http_client = httpx.AsyncClient(timeout=60.0)
        
        # Initialize log analyzer if enabled
        if config.logging.enable_log_analyzer:
            from utils.logging_utils import create_log_analyzer
            log_analyzer = create_log_analyzer(
                log_file=config.logging.file,
                analysis_interval=config.logging.log_analysis_interval,
                auto_analyze=True
            )
            app.state.log_analyzer = log_analyzer
            logger.info(f"Log analyzer initialized (analysis interval: {config.logging.log_analysis_interval}s)")
        
        logger.info("All services initialized successfully")
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        logger.exception("Startup exception details:")
    
    yield  # Application runs here
    
    # Shutdown
    logger.info("Shutting down Gutenberg Content Generation System...")
    
    # Cleanup resources
    try:
        # Close HTTP client
        if hasattr(app.state, 'http_client'):
            await app.state.http_client.aclose()
            logger.info("HTTP client closed")
        
        # Close MongoDB connections if needed
        if hasattr(app.state, 'mongodb') and hasattr(app.state.mongodb, 'close'):
            await app.state.mongodb.close()
            logger.info("MongoDB connections closed")
        
        # Stop the log analyzer if it was enabled
        if hasattr(app.state, 'log_analyzer'):
            app.state.log_analyzer.stop_auto_analysis()
            
            # Generate final report if configured to save reports
            if config.logging.log_analysis_save_reports:
                try:
                    report = app.state.log_analyzer.get_latest_report()
                    report_path = os.path.join(
                        config.general.storage_path, 
                        config.logging.log_analysis_reports_dir, 
                        f"log_analysis_{time.strftime('%Y%m%d_%H%M%S')}.json"
                    )
                    app.state.log_analyzer.save_report(report, report_path)
                    logger.info(f"Final log analysis report saved to {report_path}")
                except Exception as e:
                    logger.error(f"Error generating final log report: {e}")
            
        # Clean up old reports if retention period is set
        if config.logging.enable_log_analyzer and config.logging.log_analysis_retention_days > 0:
            try:
                reports_dir = os.path.join(
                    config.general.storage_path, 
                    config.logging.log_analysis_reports_dir
                )
                retention_time = time.time() - (config.logging.log_analysis_retention_days * 86400)
                
                report_files = glob.glob(os.path.join(reports_dir, "log_analysis_*.json"))
                for file_path in report_files:
                    if os.path.getmtime(file_path) < retention_time:
                        os.remove(file_path)
                        logger.debug(f"Removed old log analysis report: {file_path}")
            except Exception as e:
                logger.error(f"Error cleaning up old log reports: {e}")
                
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# Get configuration
config = get_config()

# Create FastAPI application with comprehensive OpenAPI documentation
app = FastAPI(
    title="Gutenberg Content Generation System",
    description="""
    Gutenberg is the content generation service of the Goliath educational platform.
    
    ## Features
    
    * Generate educational content based on concepts from Ptolemy knowledge graph
    * Create content using various templates and formats
    * Support for multiple content types: lessons, exercises, assessments, etc.
    * Generate supporting media (images, diagrams, audio)
    * Collect and process feedback on generated content
    
    ## Content Generation Process
    
    1. Submit a content generation request with the required parameters
    2. Receive a request ID for tracking progress
    3. Poll the status endpoint to check generation progress
    4. Retrieve the completed content when ready
    
    ## Templates
    
    Content is generated based on templates that define structure and style.
    Default templates are available, or you can create custom templates.
    
    ## Media Generation
    
    The API supports generating various media types to enhance content:
    - Images and illustrations
    - Diagrams and charts
    - Audio narration
    
    ## Shared Resources
    
    This service uses a shared vector database (Qdrant) with the Ptolemy service.
    All vectors include a `service` field with value "gutenberg" for namespacing.
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
            "name": "content",
            "description": "Operations for generating and managing educational content",
            "externalDocs": {
                "description": "Content generation documentation",
                "url": "https://example.com/docs/content/",
            },
        },
        {
            "name": "templates",
            "description": "Manage content generation templates",
        },
        {
            "name": "feedback",
            "description": "Collect and process feedback on generated content",
        },
        {
            "name": "media",
            "description": "Generate and manage supporting media (images, audio, etc.)",
        },
        {
            "name": "system",
            "description": "System information and health checks",
        },
    ],
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Add request ID middleware
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add request ID to each request for tracking."""
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    # Store request ID in request state
    request.state.request_id = request_id
    
    # Set request ID in logger
    if hasattr(logger, "set_request_id"):
        logger.set_request_id(request_id)
    
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# Add response timing middleware
@app.middleware("http")
async def add_response_time(request: Request, call_next):
    """Add processing time to response headers."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


# Add exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors."""
    error_details = {"field_errors": {}}
    
    for error in exc.errors():
        field_name = error["loc"][-1] if error["loc"] else "unknown"
        error_details["field_errors"][field_name] = error["msg"]
    
    app_error = ValidationError(
        message="Request validation error",
        field_errors=error_details["field_errors"]
    )
    log_exception(app_error)
    return format_exception_response(app_error)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    log_exception(exc)
    return format_exception_response(exc)


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError):
    """Handle application errors."""
    log_exception(exc)
    return format_exception_response(exc)


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    log_exception(exc)
    return format_exception_response(exc)


# Define routes
@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint that redirects to docs."""
    return RedirectResponse(url=config.api.docs_url)


# Include API routes
app.include_router(api_router, prefix=config.api.api_prefix)


# Health check endpoint
@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": config.version,
        "environment": getattr(config, "environment", "development"),
        "timestamp": time.time()
    }


# Mount static files if configured
static_dir = "generated_content"
if os.path.exists(static_dir):
    app.mount("/content", StaticFiles(directory=static_dir), name="content")
else:
    # Create the directory
    os.makedirs(static_dir, exist_ok=True)
    logger.info(f"Created static directory: {static_dir}")


def run_server():
    """Run the server using uvicorn."""
    # Configure logging first
    configure_logging()
    
    port = int(os.environ.get("PORT", config.api.port))
    
    uvicorn.run(
        "main:app",
        host=config.api.host,
        port=port,
        reload=config.api.debug,
        workers=config.api.workers
    )


if __name__ == "__main__":
    run_server()