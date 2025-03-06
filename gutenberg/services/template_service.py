"""
Template Service
=============
Service for managing content templates.
"""

import logging
import time
import uuid
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple

from models.template import ContentTemplate, TemplateType
from integrations.mongodb_service import MongoDBService
from utils.error_handling import NotFoundError, ValidationError
from utils.logging_utils import get_logger
from config.settings import get_config

logger = get_logger(__name__)


class TemplateService:
    """Service for managing content templates."""
    
    def __init__(self, mongodb: Optional[MongoDBService] = None):
        """Initialize the template service."""
        self.config = get_config()
        self.mongodb = mongodb or MongoDBService()
        self.templates_dir = self.config.templates.templates_dir
        logger.info("TemplateService initialized")
    
    async def get_template(self, template_id: str) -> ContentTemplate:
        """
        Get a template by ID.
        
        Args:
            template_id: ID of the template
            
        Returns:
            Template object
            
        Raises:
            NotFoundError: If template not found
        """
        logger.info(f"Getting template: {template_id}")
        
        # First try to get from database
        template_data = await self.mongodb.get_template(template_id)
        
        # If not found in database, try to load from file
        if not template_data:
            template_data = await self._load_template_from_file(template_id)
        
        # If still not found, raise error
        if not template_data:
            logger.warning(f"Template not found: {template_id}")
            raise NotFoundError(
                message="Template not found",
                resource_type="template",
                resource_id=template_id
            )
        
        # Create ContentTemplate object
        return ContentTemplate(**template_data)
    
    async def list_templates(self, limit: int = 20, offset: int = 0,
                           filters: Dict[str, Any] = None) -> Tuple[List[Dict[str, Any]], int]:
        """
        List templates with pagination and filtering.
        
        Args:
            limit: Maximum number of templates to return
            offset: Number of templates to skip
            filters: Optional filters
            
        Returns:
            Tuple of (list of templates, total count)
        """
        logger.info(f"Listing templates: limit={limit}, offset={offset}, filters={filters}")
        
        # Get templates from database
        db_templates, db_total = await self.mongodb.list_templates(
            limit=limit,
            offset=offset,
            filters=filters
        )
        
        # Get templates from files
        file_templates = await self._list_file_templates(filters)
        
        # Combine results, avoiding duplicates
        db_template_ids = {t["template_id"] for t in db_templates}
        combined_templates = list(db_templates)
        
        # Add file templates that aren't in the database
        for template in file_templates:
            if template["template_id"] not in db_template_ids:
                combined_templates.append(template)
        
        # Sort by name
        combined_templates.sort(key=lambda t: t.get("name", ""))
        
        # Apply pagination
        paginated_templates = combined_templates[offset:offset + limit]
        total = len(combined_templates)
        
        # Convert results to template summary format
        templates = [self._template_to_summary(t) for t in paginated_templates]
        
        return templates, total
    
    async def create_template(self, template: ContentTemplate) -> Dict[str, Any]:
        """
        Create a new template.
        
        Args:
            template: Template to create
            
        Returns:
            Dictionary with template ID and status
            
        Raises:
            ValidationError: If template is invalid
        """
        logger.info(f"Creating template: {template.name}")
        
        # Validate template
        self._validate_template(template)
        
        # Generate template ID if not provided
        if not template.template_id:
            template.template_id = f"template_{str(uuid.uuid4())[:8]}"
        
        # Check if template already exists
        existing_template = await self.mongodb.get_template(template.template_id)
        if existing_template:
            logger.warning(f"Template already exists: {template.template_id}")
            raise ValidationError(
                message="Template already exists",
                field_errors={"template_id": "Template ID already exists"}
            )
        
        # Set created_by if not provided
        if not template.metadata.get("created_by"):
            template.metadata["created_by"] = "Gutenberg System"
        
        # Set last_modified
        template.metadata["last_modified"] = datetime.now().isoformat()
        
        # Store template in database
        await self.mongodb.create_template(template.dict())
        
        logger.info(f"Template created: {template.template_id}")
        
        return {
            "template_id": template.template_id,
            "name": template.name,
            "message": "Template created successfully"
        }
    
    async def update_template(self, template_id: str, template: ContentTemplate) -> Dict[str, Any]:
        """
        Update an existing template.
        
        Args:
            template_id: ID of the template to update
            template: New template data
            
        Returns:
            Dictionary with template ID and status
            
        Raises:
            NotFoundError: If template not found
            ValidationError: If template is invalid
        """
        logger.info(f"Updating template: {template_id}")
        
        # Check if template exists
        existing_template = await self.mongodb.get_template(template_id)
        if not existing_template:
            # Check if it exists as a file template
            existing_template = await self._load_template_from_file(template_id)
            
            if not existing_template:
                logger.warning(f"Template not found: {template_id}")
                raise NotFoundError(
                    message="Template not found",
                    resource_type="template",
                    resource_id=template_id
                )
            
            # If it's a file template, we need to create it in the database first
            logger.info(f"Converting file template to database template: {template_id}")
        
        # Validate template
        self._validate_template(template)
        
        # Keep original template ID
        template.template_id = template_id
        
        # Update last_modified
        if not template.metadata:
            template.metadata = {}
        template.metadata["last_modified"] = datetime.now().isoformat()
        
        # Store template in database
        await self.mongodb.update_template(template_id, template.dict())
        
        logger.info(f"Template updated: {template_id}")
        
        return {
            "template_id": template_id,
            "name": template.name,
            "message": "Template updated successfully"
        }
    
    async def delete_template(self, template_id: str) -> Dict[str, Any]:
        """
        Delete a template.
        
        Args:
            template_id: ID of the template to delete
            
        Returns:
            Dictionary with status message
            
        Raises:
            NotFoundError: If template not found
        """
        logger.info(f"Deleting template: {template_id}")
        
        # Check if template exists
        existing_template = await self.mongodb.get_template(template_id)
        if not existing_template:
            # Check if it exists as a file template
            existing_template = await self._load_template_from_file(template_id)
            
            if not existing_template:
                logger.warning(f"Template not found: {template_id}")
                raise NotFoundError(
                    message="Template not found",
                    resource_type="template",
                    resource_id=template_id
                )
            
            # We can't delete file templates
            logger.warning(f"Cannot delete file template: {template_id}")
            raise ValidationError(
                message="Cannot delete file template",
                field_errors={"template_id": "Cannot delete built-in file templates"}
            )
        
        # Delete from database
        await self.mongodb.delete_template(template_id)
        
        logger.info(f"Template deleted: {template_id}")
        
        return {
            "template_id": template_id,
            "message": "Template deleted successfully"
        }
    
    async def get_default_template(self, template_type: str) -> ContentTemplate:
        """
        Get the default template for a specific content type.
        
        Args:
            template_type: Type of template
            
        Returns:
            Default template for the given type
            
        Raises:
            NotFoundError: If no default template found
        """
        logger.info(f"Getting default template for type: {template_type}")
        
        # First try to find in database
        filters = {
            "template_type": template_type,
            "is_default": True
        }
        
        templates, _ = await self.mongodb.list_templates(limit=1, filters=filters)
        
        if templates:
            return ContentTemplate(**templates[0])
        
        # If not found in database, try to load from file
        if template_type == "concept" or not template_type:
            # Default template is default.json
            template_data = await self._load_template_from_file("default_content_template")
            if template_data:
                return ContentTemplate(**template_data)
        
        # Try to find a file template with naming convention
        template_id = f"default_{template_type}"
        template_data = await self._load_template_from_file(template_id)
        if template_data:
            return ContentTemplate(**template_data)
        
        # If still not found, raise error
        logger.warning(f"Default template not found for type: {template_type}")
        raise NotFoundError(
            message=f"Default template not found for type: {template_type}",
            resource_type="template",
            resource_id=f"default_{template_type}"
        )
    
    # -------- Private Helper Methods -------- #
    
    async def _load_template_from_file(self, template_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a template from a file.
        
        Args:
            template_id: ID of the template
            
        Returns:
            Template data or None if not found
        """
        try:
            # First try exact match
            template_path = os.path.join(self.templates_dir, f"{template_id}.json")
            if os.path.exists(template_path):
                with open(template_path, 'r', encoding='utf-8') as f:
                    template_data = json.load(f)
                    return template_data
            
            # Try default.json
            if template_id.startswith("default"):
                default_path = os.path.join(self.templates_dir, "default.json")
                if os.path.exists(default_path):
                    with open(default_path, 'r', encoding='utf-8') as f:
                        template_data = json.load(f)
                        return template_data
            
            # Try specialized directory
            specialized_path = os.path.join(self.templates_dir, "specialized", f"{template_id}.json")
            if os.path.exists(specialized_path):
                with open(specialized_path, 'r', encoding='utf-8') as f:
                    template_data = json.load(f)
                    return template_data
            
            # Not found
            return None
            
        except Exception as e:
            logger.error(f"Error loading template from file: {str(e)}")
            return None
    
    async def _list_file_templates(self, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        List templates from files.
        
        Args:
            filters: Optional filters
            
        Returns:
            List of template data
        """
        templates = []
        
        try:
            # Get all .json files in templates directory
            template_files = []
            for root, _, files in os.walk(self.templates_dir):
                for file in files:
                    if file.endswith(".json"):
                        template_files.append(os.path.join(root, file))
            
            # Load each template
            for file_path in template_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        template_data = json.load(f)
                        
                        # Apply filters if provided
                        if filters:
                            if "template_type" in filters and template_data.get("template_type") != filters["template_type"]:
                                continue
                            
                            if "search" in filters:
                                search_term = filters["search"].lower()
                                if (search_term not in template_data.get("name", "").lower() and
                                    search_term not in template_data.get("description", "").lower()):
                                    continue
                        
                        templates.append(template_data)
                        
                except Exception as e:
                    logger.warning(f"Error loading template file {file_path}: {str(e)}")
                    continue
                    
            return templates
            
        except Exception as e:
            logger.error(f"Error listing file templates: {str(e)}")
            return []
    
    def _template_to_summary(self, template: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert template data to summary format.
        
        Args:
            template: Template data
            
        Returns:
            Template summary
        """
        return {
            "template_id": template.get("template_id"),
            "name": template.get("name"),
            "description": template.get("description"),
            "template_type": template.get("template_type"),
            "target": template.get("target"),
            "version": template.get("version"),
            "section_count": len(template.get("sections", [])),
            "created_by": template.get("metadata", {}).get("created_by"),
            "last_modified": template.get("metadata", {}).get("last_modified")
        }
    
    def _validate_template(self, template: ContentTemplate) -> None:
        """
        Validate a template.
        
        Args:
            template: Template to validate
            
        Raises:
            ValidationError: If template is invalid
        """
        errors = {}
        
        # Validate basic fields
        if not template.name:
            errors["name"] = "Template name is required"
        
        if not template.sections:
            errors["sections"] = "Template must have at least one section"
        
        # Validate sections
        section_ids = set()
        for i, section in enumerate(template.sections):
            section_path = f"sections[{i}]"
            
            if not section.id:
                errors[f"{section_path}.id"] = "Section ID is required"
            elif section.id in section_ids:
                errors[f"{section_path}.id"] = f"Duplicate section ID: {section.id}"
            else:
                section_ids.add(section.id)
            
            if not section.title:
                errors[f"{section_path}.title"] = "Section title is required"
            
            # Check prompts
            if not section.prompts:
                errors[f"{section_path}.prompts"] = "Section must have at least one prompt"
        
        if errors:
            raise ValidationError(
                message="Invalid template",
                field_errors=errors
            )