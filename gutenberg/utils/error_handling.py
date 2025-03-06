"""
Gutenberg Content Generation System - Error Handling
===================================================
Utilities for error handling, validation, and exception management.
"""

import logging
import traceback
import sys
from typing import Dict, Any, List, Optional, Callable, Type, Union
from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse

# Get logger
logger = logging.getLogger("error_handler")

class AppError(Exception):
    """Base exception class for application errors."""
    
    def __init__(
        self, 
        message: str,
        code: str = "internal_error",
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the application error.
        
        Args:
            message: Human-readable error message
            code: Machine-readable error code
            status_code: HTTP status code
            details: Additional error details
        """
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the error to a dictionary for response."""
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details
            }
        }

class ValidationError(AppError):
    """Exception for validation errors."""
    
    def __init__(
        self, 
        message: str = "Validation error",
        field_errors: Optional[Dict[str, str]] = None,
        code: str = "validation_error",
        status_code: int = 400
    ):
        """
        Initialize the validation error.
        
        Args:
            message: Human-readable error message
            field_errors: Dictionary of field-specific errors
            code: Machine-readable error code
            status_code: HTTP status code
        """
        details = {"field_errors": field_errors or {}}
        super().__init__(message, code, status_code, details)

class NotFoundError(AppError):
    """Exception for resource not found errors."""
    
    def __init__(
        self, 
        resource_type: str,
        resource_id: str,
        message: Optional[str] = None,
        code: str = "not_found",
        status_code: int = 404
    ):
        """
        Initialize the not found error.
        
        Args:
            resource_type: Type of resource not found
            resource_id: ID of the resource not found
            message: Human-readable error message (generated if None)
            code: Machine-readable error code
            status_code: HTTP status code
        """
        if message is None:
            message = f"{resource_type} with ID {resource_id} not found"
        
        details = {
            "resource_type": resource_type,
            "resource_id": resource_id
        }
        
        super().__init__(message, code, status_code, details)

class AuthorizationError(AppError):
    """Exception for authorization errors."""
    
    def __init__(
        self, 
        message: str = "Not authorized to perform this action",
        code: str = "unauthorized",
        status_code: int = 403,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the authorization error.
        
        Args:
            message: Human-readable error message
            code: Machine-readable error code
            status_code: HTTP status code
            details: Additional error details
        """
        super().__init__(message, code, status_code, details)

class RateLimitError(AppError):
    """Exception for rate limit errors."""
    
    def __init__(
        self, 
        message: str = "Rate limit exceeded",
        limit: int = 0,
        reset_seconds: int = 0,
        code: str = "rate_limit_exceeded",
        status_code: int = 429
    ):
        """
        Initialize the rate limit error.
        
        Args:
            message: Human-readable error message
            limit: Rate limit
            reset_seconds: Seconds until the rate limit resets
            code: Machine-readable error code
            status_code: HTTP status code
        """
        details = {
            "limit": limit,
            "reset_seconds": reset_seconds
        }
        
        super().__init__(message, code, status_code, details)

def log_exception(exc: Exception) -> None:
    """
    Log an exception with appropriate level and details.
    
    Args:
        exc: The exception to log
    """
    if isinstance(exc, AppError):
        if exc.status_code >= 500:
            logger.error(f"Server error ({exc.code}): {exc.message}", exc_info=True)
        else:
            logger.warning(f"Client error ({exc.code}): {exc.message}")
    elif isinstance(exc, HTTPException):
        if exc.status_code >= 500:
            logger.error(f"HTTP error {exc.status_code}: {exc.detail}", exc_info=True)
        else:
            logger.warning(f"HTTP error {exc.status_code}: {exc.detail}")
    else:
        logger.error(f"Unexpected error: {str(exc)}", exc_info=True)

def format_exception_response(exc: Exception) -> JSONResponse:
    """
    Format an exception as a JSON response.
    
    Args:
        exc: The exception to format
        
    Returns:
        A FastAPI JSONResponse
    """
    if isinstance(exc, AppError):
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_dict()
        )
    elif isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": f"http_{exc.status_code}",
                    "message": exc.detail,
                    "details": {}
                }
            }
        )
    else:
        # For unexpected exceptions, return a generic 500 error
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "internal_error",
                    "message": "An unexpected error occurred",
                    "details": {
                        "error_type": type(exc).__name__
                    }
                }
            }
        )

def exception_handler_factory(exception_type: Type[Exception]) -> Callable:
    """
    Create an exception handler for a specific exception type.
    
    Args:
        exception_type: The type of exception to handle
        
    Returns:
        An async function to handle the exception
    """
    async def handler(request: Request, exc: exception_type) -> JSONResponse:
        log_exception(exc)
        return format_exception_response(exc)
    
    return handler