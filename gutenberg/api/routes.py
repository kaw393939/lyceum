"""
Gutenberg Content Generation System - Main API Routes
====================================================
Contains the main API routes used by the application.
"""

from fastapi import APIRouter, Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from typing import Dict, Any, List

# Import sub-routers
from api.content_routes import router as content_router
from api.template_routes import router as template_router
from api.feedback_routes import router as feedback_router
from api.media_routes import router as media_router

# Create main router
router = APIRouter()

# Add sub-routers
router.include_router(content_router, prefix="/content", tags=["content"])
router.include_router(template_router, prefix="/templates", tags=["templates"])
router.include_router(feedback_router, prefix="/feedback", tags=["feedback"])
router.include_router(media_router, prefix="/media", tags=["media"])

# Define base routes
@router.get("/health", summary="Check system health", tags=["system"])
async def health_check() -> Dict[str, Any]:
    """Check the health status of the system."""
    return {
        "status": "ok",
        "version": "1.0.0",
        "services": {
            "api": {
                "status": "ok"
            },
            "llm": {
                "status": "ok"
            },
            "database": {
                "status": "ok"
            }
        }
    }

@router.get("/info", summary="Get system information", tags=["system"])
async def system_info() -> Dict[str, Any]:
    """Get information about the system."""
    return {
        "name": "Gutenberg Content Generation System",
        "version": "1.0.0",
        "api_version": "v1",
        "description": "Educational content generation with Stoic principles",
        "contact": {
            "name": "System Administrator",
            "email": "admin@example.com"
        },
        "license": {
            "name": "Proprietary",
            "url": "https://example.com/license"
        }
    }