"""
Gutenberg Content Generation System - Template API Routes
========================================================
Contains routes related to content generation templates.
"""

from fastapi import APIRouter, Request, Response, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional
import os
import json
import time
import uuid

# Create router
router = APIRouter()

# Define routes
@router.get("/", summary="List available templates", tags=["templates"])
async def list_templates() -> Dict[str, Any]:
    """
    List all available content generation templates.
    """
    # In a real implementation, this would fetch from a database or file system
    # For now, we'll return mock data
    
    # Define mock templates
    templates = [
        {
            "id": "default",
            "name": "Default Content Template",
            "description": "Standard template for educational content generation with Stoic principles",
            "type": "concept",
            "version": "1.0",
            "created_at": int(time.time()) - 86400 * 30  # 30 days ago
        },
        {
            "id": "essay",
            "name": "Essay Template",
            "description": "Template for generating essay-style content",
            "type": "concept",
            "version": "1.0",
            "created_at": int(time.time()) - 86400 * 20  # 20 days ago
        },
        {
            "id": "quiz",
            "name": "Quiz Template",
            "description": "Template for generating quiz and assessment content",
            "type": "assessment",
            "version": "1.0",
            "created_at": int(time.time()) - 86400 * 10  # 10 days ago
        }
    ]
    
    return {
        "templates": templates,
        "count": len(templates)
    }

@router.get("/{template_id}", summary="Get template details", tags=["templates"])
async def get_template(template_id: str) -> Dict[str, Any]:
    """
    Get details of a specific template by ID.
    """
    # In a real implementation, this would fetch from a database or file system
    
    # Check if template exists
    if template_id == "default":
        # Try to read the default template file
        try:
            template_path = os.path.join("templates", "default.json")
            if os.path.exists(template_path):
                with open(template_path, 'r') as f:
                    template_data = json.load(f)
                    return {
                        "id": "default",
                        "name": template_data.get("name", "Default Content Template"),
                        "description": template_data.get("description", "Standard template for educational content generation"),
                        "type": template_data.get("template_type", "concept"),
                        "version": template_data.get("version", "1.0"),
                        "created_at": int(time.time()) - 86400 * 30,  # 30 days ago
                        "structure": template_data
                    }
        except Exception as e:
            # If there's an error reading the file, return mock data
            pass
        
        # Return mock data as fallback
        return {
            "id": "default",
            "name": "Default Content Template",
            "description": "Standard template for educational content generation with Stoic principles",
            "type": "concept",
            "version": "1.0",
            "created_at": int(time.time()) - 86400 * 30,  # 30 days ago
            "structure": {
                "sections": [
                    {"id": "introduction", "title": "Introduction"},
                    {"id": "overview", "title": "Concept Overview"},
                    {"id": "learning_objectives", "title": "Learning Objectives"},
                    {"id": "key_vocabulary", "title": "Key Vocabulary"},
                    {"id": "stoic_connection", "title": "Stoic Connection"},
                    {"id": "main_content", "title": "Main Content"},
                    {"id": "assessment", "title": "Assessment"},
                    {"id": "conclusion", "title": "Conclusion"},
                    {"id": "further_learning", "title": "Further Learning"}
                ]
            }
        }
    elif template_id == "essay":
        return {
            "id": "essay",
            "name": "Essay Template",
            "description": "Template for generating essay-style content",
            "type": "concept",
            "version": "1.0",
            "created_at": int(time.time()) - 86400 * 20,  # 20 days ago
            "structure": {
                "sections": [
                    {"id": "introduction", "title": "Introduction"},
                    {"id": "thesis", "title": "Thesis Statement"},
                    {"id": "body", "title": "Body Paragraphs"},
                    {"id": "conclusion", "title": "Conclusion"},
                    {"id": "references", "title": "References"}
                ]
            }
        }
    elif template_id == "quiz":
        return {
            "id": "quiz",
            "name": "Quiz Template",
            "description": "Template for generating quiz and assessment content",
            "type": "assessment",
            "version": "1.0",
            "created_at": int(time.time()) - 86400 * 10,  # 10 days ago
            "structure": {
                "sections": [
                    {"id": "multiple_choice", "title": "Multiple Choice Questions"},
                    {"id": "true_false", "title": "True/False Questions"},
                    {"id": "short_answer", "title": "Short Answer Questions"},
                    {"id": "essay_questions", "title": "Essay Questions"}
                ]
            }
        }
    else:
        raise HTTPException(
            status_code=404,
            detail=f"Template '{template_id}' not found"
        )

@router.post("/", summary="Create new template", tags=["templates"])
async def create_template(template: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a new content generation template.
    """
    # In a real implementation, this would save to a database or file system
    # For now, we'll return mock data
    
    # Generate a new template ID
    template_id = f"template-{str(uuid.uuid4())[:8]}"
    
    return {
        "id": template_id,
        "status": "created",
        "message": "Template created successfully"
    }

@router.put("/{template_id}", summary="Update template", tags=["templates"])
async def update_template(template_id: str, template: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update an existing content generation template.
    """
    # In a real implementation, this would update a database or file
    # For now, we'll return mock data
    
    if template_id not in ["default", "essay", "quiz"]:
        raise HTTPException(
            status_code=404,
            detail=f"Template '{template_id}' not found"
        )
    
    return {
        "id": template_id,
        "status": "updated",
        "message": "Template updated successfully"
    }

@router.delete("/{template_id}", summary="Delete template", tags=["templates"])
async def delete_template(template_id: str) -> Dict[str, Any]:
    """
    Delete a content generation template.
    """
    # In a real implementation, this would delete from a database or file system
    # For now, we'll return mock data
    
    if template_id not in ["default", "essay", "quiz"]:
        raise HTTPException(
            status_code=404,
            detail=f"Template '{template_id}' not found"
        )
    
    if template_id == "default":
        raise HTTPException(
            status_code=403,
            detail="Cannot delete the default template"
        )
    
    return {
        "id": template_id,
        "status": "deleted",
        "message": "Template deleted successfully"
    }