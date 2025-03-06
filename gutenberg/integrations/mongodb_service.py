"""
MongoDB Service
==============
Service for interacting with MongoDB database, including GridFS for media storage.
"""

import os
import json
import logging
import time
import asyncio
import base64
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple, BinaryIO, IO
import uuid
import motor.motor_asyncio
from pymongo import ReturnDocument
from bson.objectid import ObjectId
import gridfs

from config.settings import get_config
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class MongoDBService:
    """Service for interacting with MongoDB."""
    
    def __init__(self):
        """Initialize the MongoDB service."""
        self.config = get_config()
        
        # Check if we should use mock mode
        self.mock = self.config.get("mongodb", {}).get("use_mock", False)
        
        # If no MongoDB URI is provided, fall back to mock mode
        if not os.environ.get("MONGODB_URI") and not self.config.get("mongodb", {}).get("uri"):
            logger.warning("No MongoDB URI found, falling back to mock mode")
            self.mock = True
        
        if not self.mock:
            # Initialize real MongoDB connection
            self._init_db_connection()
        else:
            # Initialize mock storage
            self.mock_data = {
                "content": {},
                "content_status": {},
                "content_requests": {},
                "templates": {},
                "feedback": {}
            }
            # Load default template
            self._load_default_template()
            logger.info("MongoDBService initialized in mock mode")
    
    def _init_db_connection(self):
        """Initialize the MongoDB connection."""
        try:
            # Get connection details from config, prioritizing environment variables
            mongodb_uri = os.environ.get("MONGODB_URI", 
                              self.config.get("mongodb", {}).get("uri", "mongodb://gutenberg_user:gutenberg_password@mongodb:27017/gutenberg"))
            
            # If the URI contains a database name at the end, extract it
            if "/" in mongodb_uri.split("://")[1] and mongodb_uri.count("/") > 2:
                host_part, db_part = mongodb_uri.rsplit("/", 1)
                if "?" in db_part:  # Handle query parameters
                    db_name = db_part.split("?")[0]
                else:
                    db_name = db_part
                logger.debug(f"Extracted database name from URI: {db_name}")
            else:
                # Otherwise get it from config or use default
                db_name = self.config.get("mongodb", {}).get("database", "gutenberg")
            
            # Create client
            self.client = motor.motor_asyncio.AsyncIOMotorClient(mongodb_uri)
            self.db = self.client[db_name]
            
            # Initialize collections
            self.content_collection = self.db["content"]
            self.template_collection = self.db["templates"]
            self.feedback_collection = self.db["feedback"]
            self.status_collection = self.db["content_status"]
            self.request_collection = self.db["content_requests"]
            
            # Initialize GridFS
            self.fs = gridfs.GridFSBucket(self.db, bucket_name="media")
            
            # Create indexes
            asyncio.create_task(self._create_indexes())
            
            logger.info(f"Connected to MongoDB database: {db_name}")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            self.mock = True  # Fall back to mock mode
            self.mock_data = {
                "content": {},
                "content_status": {},
                "content_requests": {},
                "templates": {},
                "feedback": {},
                "media": {}
            }
            self._load_default_template()
            logger.info("Falling back to mock mode due to connection error")
    
    async def _create_indexes(self):
        """Create necessary indexes on collections."""
        try:
            # Content collection indexes
            await self.content_collection.create_index("content_id", unique=True)
            await self.content_collection.create_index("created_at")
            await self.content_collection.create_index("metadata.content_type")
            
            # Template collection indexes
            await self.template_collection.create_index("template_id", unique=True)
            await self.template_collection.create_index("name")
            
            # Feedback collection indexes
            await self.feedback_collection.create_index("feedback_id", unique=True)
            await self.feedback_collection.create_index("content_id")
            
            # Status collection indexes
            await self.status_collection.create_index("request_id", unique=True)
            await self.status_collection.create_index("created_at")
            
            # Request collection indexes
            await self.request_collection.create_index("request_id", unique=True)
            
            logger.info("MongoDB indexes created successfully")
        except Exception as e:
            logger.error(f"Error creating MongoDB indexes: {str(e)}")
    
    def _load_default_template(self):
        """Load the default template into mock data."""
        try:
            with open('templates/default.json', 'r') as f:
                template = json.load(f)
                template_id = template.get('template_id', str(uuid.uuid4()))
                self.mock_data["templates"][template_id] = template
        except Exception as e:
            logger.error(f"Error loading default template: {e}")
            # Create a basic placeholder template
            self.mock_data["templates"]["default"] = {
                "template_id": "default",
                "name": "Default Template",
                "description": "Basic template for testing",
                "template_type": "concept",
                "version": "1.0",
                "sections": []
            }
    
    async def check_connection(self) -> bool:
        """
        Check database connection.
        
        Returns:
            True if database is accessible
        """
        if self.mock:
            return True
        
        try:
            # Ping database
            await self.client.admin.command('ping')
            return True
        except Exception as e:
            logger.error(f"Database connection check failed: {str(e)}")
            return False
    
    # Content Status Operations
    
    async def create_content_status(self, status_data: Dict[str, Any]) -> str:
        """
        Create a new content generation status.
        
        Args:
            status_data: Status data
            
        Returns:
            Request ID
        """
        request_id = status_data.get("request_id")
        if not request_id:
            request_id = str(uuid.uuid4())
            status_data["request_id"] = request_id
        
        # Add timestamps
        status_data["created_at"] = datetime.now()
        status_data["updated_at"] = datetime.now()
        
        if self.mock:
            # Convert datetimes to strings for JSON serialization in mock mode
            status_data["created_at"] = status_data["created_at"].isoformat()
            status_data["updated_at"] = status_data["updated_at"].isoformat()
            self.mock_data["content_status"][request_id] = status_data
            logger.info(f"Created content status with ID: {request_id} (mock)")
        else:
            try:
                await self.status_collection.insert_one(status_data)
                logger.info(f"Created content status with ID: {request_id}")
            except Exception as e:
                logger.error(f"Error creating content status: {str(e)}")
                raise
        
        return request_id
    
    async def update_content_status(self, request_id: str, update_data: Dict[str, Any]) -> bool:
        """
        Update content generation status.
        
        Args:
            request_id: Request ID
            update_data: Data to update
            
        Returns:
            True if successful
        """
        # Add updated timestamp
        update_data["updated_at"] = datetime.now()
        
        if self.mock:
            if request_id not in self.mock_data["content_status"]:
                logger.warning(f"Content status not found for update: {request_id} (mock)")
                return False
            
            # Convert datetime to string for JSON serialization in mock mode
            update_data["updated_at"] = update_data["updated_at"].isoformat()
            self.mock_data["content_status"][request_id].update(update_data)
            logger.info(f"Updated content status with ID: {request_id} (mock)")
            return True
        else:
            try:
                result = await self.status_collection.update_one(
                    {"request_id": request_id},
                    {"$set": update_data}
                )
                
                if result.modified_count > 0:
                    logger.info(f"Updated content status with ID: {request_id}")
                    return True
                else:
                    logger.warning(f"Content status not found for update: {request_id}")
                    return False
            except Exception as e:
                logger.error(f"Error updating content status: {str(e)}")
                return False
    
    async def get_content_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """
        Get content generation status.
        
        Args:
            request_id: Request ID
            
        Returns:
            Status data or None if not found
        """
        if self.mock:
            return self.mock_data["content_status"].get(request_id)
        else:
            try:
                result = await self.status_collection.find_one({"request_id": request_id})
                # Convert ObjectId to string for JSON serialization
                if result and "_id" in result:
                    result["_id"] = str(result["_id"])
                return result
            except Exception as e:
                logger.error(f"Error getting content status: {str(e)}")
                return None
    
    async def create_content_request(self, request_data: Dict[str, Any]) -> str:
        """
        Create a content generation request.
        
        Args:
            request_data: Request data
            
        Returns:
            Request ID
        """
        request_id = request_data.get("request_id")
        if not request_id:
            request_id = str(uuid.uuid4())
            request_data["request_id"] = request_id
        
        # Add timestamps
        request_data["created_at"] = datetime.now()
        request_data["updated_at"] = datetime.now()
        
        if self.mock:
            # Convert datetimes to strings for JSON serialization in mock mode
            request_data["created_at"] = request_data["created_at"].isoformat()
            request_data["updated_at"] = request_data["updated_at"].isoformat()
            self.mock_data["content_requests"][request_id] = request_data
            logger.info(f"Created content request with ID: {request_id} (mock)")
        else:
            try:
                await self.request_collection.insert_one(request_data)
                logger.info(f"Created content request with ID: {request_id}")
            except Exception as e:
                logger.error(f"Error creating content request: {str(e)}")
                raise
        
        return request_id
    
    async def get_content_request(self, request_id: str) -> Optional[Dict[str, Any]]:
        """
        Get content request by ID.
        
        Args:
            request_id: Request ID
            
        Returns:
            Request data or None if not found
        """
        if self.mock:
            return self.mock_data["content_requests"].get(request_id)
        else:
            try:
                result = await self.request_collection.find_one({"request_id": request_id})
                # Convert ObjectId to string for JSON serialization
                if result and "_id" in result:
                    result["_id"] = str(result["_id"])
                return result
            except Exception as e:
                logger.error(f"Error getting content request: {str(e)}")
                return None
    
    async def update_content_request(self, request_id: str, update_data: Dict[str, Any]) -> bool:
        """
        Update content request.
        
        Args:
            request_id: Request ID
            update_data: Data to update
            
        Returns:
            True if successful
        """
        # Add updated timestamp
        update_data["updated_at"] = datetime.now()
        
        if self.mock:
            if request_id not in self.mock_data["content_requests"]:
                logger.warning(f"Content request not found for update: {request_id} (mock)")
                return False
            
            # Convert datetime to string for JSON serialization in mock mode
            update_data["updated_at"] = update_data["updated_at"].isoformat()
            self.mock_data["content_requests"][request_id].update(update_data)
            logger.info(f"Updated content request with ID: {request_id} (mock)")
            return True
        else:
            try:
                result = await self.request_collection.update_one(
                    {"request_id": request_id},
                    {"$set": update_data}
                )
                
                if result.modified_count > 0:
                    logger.info(f"Updated content request with ID: {request_id}")
                    return True
                else:
                    logger.warning(f"Content request not found for update: {request_id}")
                    return False
            except Exception as e:
                logger.error(f"Error updating content request: {str(e)}")
                return False
        
    # Content Operations
    
    async def create_content(self, content: Dict[str, Any]) -> str:
        """
        Create a new content document.
        
        Args:
            content: Content data to insert
            
        Returns:
            ID of the created content
        """
        content_id = content.get("content_id", str(uuid.uuid4()))
        content["content_id"] = content_id
        content["created_at"] = datetime.now()
        content["updated_at"] = datetime.now()
        
        if self.mock:
            # Convert datetimes to strings for JSON serialization in mock mode
            content["created_at"] = content["created_at"].isoformat()
            content["updated_at"] = content["updated_at"].isoformat()
            self.mock_data["content"][content_id] = content
            logger.info(f"Created content with ID: {content_id} (mock)")
        else:
            try:
                await self.content_collection.insert_one(content)
                logger.info(f"Created content with ID: {content_id}")
            except Exception as e:
                logger.error(f"Error creating content: {str(e)}")
                raise
        
        return content_id
    
    async def get_content(self, content_id: str) -> Optional[Dict[str, Any]]:
        """
        Get content by ID.
        
        Args:
            content_id: ID of the content to retrieve
            
        Returns:
            Content document or None if not found
        """
        if self.mock:
            return self.mock_data["content"].get(content_id)
        else:
            try:
                result = await self.content_collection.find_one({"content_id": content_id})
                # Convert ObjectId to string for JSON serialization
                if result and "_id" in result:
                    result["_id"] = str(result["_id"])
                return result
            except Exception as e:
                logger.error(f"Error getting content: {str(e)}")
                return None
            
    async def get_content_by_id(self, content_id: str) -> Optional[Dict[str, Any]]:
        """
        Get content by ID (alias for compatibility).
        
        Args:
            content_id: ID of the content to retrieve
            
        Returns:
            Content document or None if not found
        """
        return await self.get_content(content_id)
    
    async def update_content(self, content_id: str, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Update content by ID.
        
        Args:
            content_id: ID of the content to update
            update_data: Data to update
            
        Returns:
            Updated content document or None if not found
        """
        # Add updated timestamp
        update_data["updated_at"] = datetime.now()
        
        if self.mock:
            if content_id not in self.mock_data["content"]:
                logger.warning(f"Content not found for update: {content_id} (mock)")
                return None
                
            # Convert datetime to string for JSON serialization in mock mode
            update_data["updated_at"] = update_data["updated_at"].isoformat()
            self.mock_data["content"][content_id].update(update_data)
            logger.info(f"Updated content with ID: {content_id} (mock)")
            return self.mock_data["content"][content_id]
        else:
            try:
                result = await self.content_collection.find_one_and_update(
                    {"content_id": content_id},
                    {"$set": update_data},
                    return_document=ReturnDocument.AFTER
                )
                
                if result:
                    # Convert ObjectId to string for JSON serialization
                    if "_id" in result:
                        result["_id"] = str(result["_id"])
                    logger.info(f"Updated content with ID: {content_id}")
                else:
                    logger.warning(f"Content not found for update: {content_id}")
                    
                return result
            except Exception as e:
                logger.error(f"Error updating content: {str(e)}")
                return None
    
    async def delete_content(self, content_id: str) -> bool:
        """
        Delete content by ID.
        
        Args:
            content_id: ID of the content to delete
            
        Returns:
            True if deleted, False if not found
        """
        if self.mock:
            if content_id not in self.mock_data["content"]:
                logger.warning(f"Content not found for deletion: {content_id} (mock)")
                return False
                
            del self.mock_data["content"][content_id]
            logger.info(f"Deleted content with ID: {content_id} (mock)")
            return True
        else:
            try:
                result = await self.content_collection.delete_one({"content_id": content_id})
                
                if result.deleted_count > 0:
                    logger.info(f"Deleted content with ID: {content_id}")
                    return True
                else:
                    logger.warning(f"Content not found for deletion: {content_id}")
                    return False
            except Exception as e:
                logger.error(f"Error deleting content: {str(e)}")
                return False
    
    async def list_content(self, limit: int = 10, offset: int = 0, 
                         filters: Dict[str, Any] = None) -> Tuple[List[Dict[str, Any]], int]:
        """
        List content with pagination and filtering.
        
        Args:
            limit: Maximum number of items
            offset: Number of items to skip
            filters: Optional filters
            
        Returns:
            Tuple of (items, total count)
        """
        filters = filters or {}
        
        if self.mock:
            filtered_items = []
            
            for content_id, content in self.mock_data["content"].items():
                # Apply filters if provided
                if filters:
                    matches = True
                    for key, value in filters.items():
                        # Handle nested fields (e.g., "metadata.content_type")
                        if "." in key:
                            parts = key.split(".")
                            item_value = content
                            for part in parts:
                                if isinstance(item_value, dict) and part in item_value:
                                    item_value = item_value[part]
                                else:
                                    item_value = None
                                    break
                        else:
                            item_value = content.get(key)
                        
                        if item_value != value:
                            matches = False
                            break
                    
                    if not matches:
                        continue
                
                # Add to filtered items
                filtered_items.append(content)
            
            # Sort by created_at if available
            filtered_items.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            
            # Apply pagination
            paginated_items = filtered_items[offset:offset + limit]
            
            return paginated_items, len(filtered_items)
        else:
            try:
                # Process nested field filters
                mongo_filters = {}
                for key, value in filters.items():
                    mongo_filters[key] = value
                
                # Get total count
                total = await self.content_collection.count_documents(mongo_filters)
                
                # Get paginated results
                cursor = self.content_collection.find(mongo_filters)
                cursor = cursor.sort("created_at", -1).skip(offset).limit(limit)
                
                results = await cursor.to_list(length=limit)
                
                # Convert ObjectId to string for JSON serialization
                for result in results:
                    if "_id" in result:
                        result["_id"] = str(result["_id"])
                
                return results, total
            except Exception as e:
                logger.error(f"Error listing content: {str(e)}")
                return [], 0
            
    async def check_content_exists(self, content_id: str) -> bool:
        """
        Check if content exists.
        
        Args:
            content_id: Content ID
            
        Returns:
            True if content exists
        """
        if self.mock:
            return content_id in self.mock_data["content"]
        else:
            try:
                count = await self.content_collection.count_documents({"content_id": content_id})
                return count > 0
            except Exception as e:
                logger.error(f"Error checking content existence: {str(e)}")
                return False
    
    # Template Operations
    
    async def create_template(self, template: Dict[str, Any]) -> str:
        """
        Create a new template.
        
        Args:
            template: Template data
            
        Returns:
            ID of the created template
        """
        template_id = template.get("template_id", str(uuid.uuid4()))
        template["template_id"] = template_id
        template["created_at"] = datetime.now()
        template["updated_at"] = datetime.now()
        
        if self.mock:
            # Convert datetimes to strings for JSON serialization in mock mode
            template["created_at"] = template["created_at"].isoformat()
            template["updated_at"] = template["updated_at"].isoformat()
            self.mock_data["templates"][template_id] = template
            logger.info(f"Created template with ID: {template_id} (mock)")
        else:
            try:
                await self.template_collection.insert_one(template)
                logger.info(f"Created template with ID: {template_id}")
            except Exception as e:
                logger.error(f"Error creating template: {str(e)}")
                raise
        
        return template_id
    
    async def get_template(self, template_id: str) -> Optional[Dict[str, Any]]:
        """
        Get template by ID.
        
        Args:
            template_id: ID of the template to retrieve
            
        Returns:
            Template document or None if not found
        """
        if self.mock:
            return self.mock_data["templates"].get(template_id)
        else:
            try:
                result = await self.template_collection.find_one({"template_id": template_id})
                # Convert ObjectId to string for JSON serialization
                if result and "_id" in result:
                    result["_id"] = str(result["_id"])
                return result
            except Exception as e:
                logger.error(f"Error getting template: {str(e)}")
                return None
    
    async def update_template(self, template_id: str, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Update template by ID.
        
        Args:
            template_id: ID of the template to update
            update_data: Data to update
            
        Returns:
            Updated template document or None if not found
        """
        # Add updated timestamp
        update_data["updated_at"] = datetime.now()
        
        if self.mock:
            if template_id not in self.mock_data["templates"]:
                logger.warning(f"Template not found for update: {template_id} (mock)")
                return None
                
            # Convert datetime to string for JSON serialization in mock mode
            update_data["updated_at"] = update_data["updated_at"].isoformat()
            self.mock_data["templates"][template_id].update(update_data)
            logger.info(f"Updated template with ID: {template_id} (mock)")
            return self.mock_data["templates"][template_id]
        else:
            try:
                result = await self.template_collection.find_one_and_update(
                    {"template_id": template_id},
                    {"$set": update_data},
                    return_document=ReturnDocument.AFTER
                )
                
                if result:
                    # Convert ObjectId to string for JSON serialization
                    if "_id" in result:
                        result["_id"] = str(result["_id"])
                    logger.info(f"Updated template with ID: {template_id}")
                else:
                    logger.warning(f"Template not found for update: {template_id}")
                    
                return result
            except Exception as e:
                logger.error(f"Error updating template: {str(e)}")
                return None
    
    async def delete_template(self, template_id: str) -> bool:
        """
        Delete template by ID.
        
        Args:
            template_id: ID of the template to delete
            
        Returns:
            True if deleted, False if not found
        """
        if self.mock:
            if template_id not in self.mock_data["templates"]:
                logger.warning(f"Template not found for deletion: {template_id} (mock)")
                return False
                
            del self.mock_data["templates"][template_id]
            logger.info(f"Deleted template with ID: {template_id} (mock)")
            return True
        else:
            try:
                result = await self.template_collection.delete_one({"template_id": template_id})
                
                if result.deleted_count > 0:
                    logger.info(f"Deleted template with ID: {template_id}")
                    return True
                else:
                    logger.warning(f"Template not found for deletion: {template_id}")
                    return False
            except Exception as e:
                logger.error(f"Error deleting template: {str(e)}")
                return False
    
    async def list_templates(self, limit: int = 10, offset: int = 0, 
                           filters: Dict[str, Any] = None) -> Tuple[List[Dict[str, Any]], int]:
        """
        List templates with optional filtering and pagination.
        
        Args:
            limit: Maximum number of results
            offset: Number of results to skip
            filters: Optional filters
            
        Returns:
            Tuple of (items, total count)
        """
        filters = filters or {}
        
        if self.mock:
            filtered_items = []
            
            for template_id, template in self.mock_data["templates"].items():
                # Apply filters if provided
                if filters:
                    matches = True
                    for key, value in filters.items():
                        if template.get(key) != value:
                            matches = False
                            break
                    
                    if not matches:
                        continue
                
                # Add to filtered items
                filtered_items.append(template)
            
            # Sort by name if available
            filtered_items.sort(key=lambda x: x.get("name", ""))
            
            # Apply pagination
            paginated_items = filtered_items[offset:offset + limit]
            
            return paginated_items, len(filtered_items)
        else:
            try:
                # Get total count
                total = await self.template_collection.count_documents(filters)
                
                # Get paginated results
                cursor = self.template_collection.find(filters)
                cursor = cursor.sort("name", 1).skip(offset).limit(limit)
                
                results = await cursor.to_list(length=limit)
                
                # Convert ObjectId to string for JSON serialization
                for result in results:
                    if "_id" in result:
                        result["_id"] = str(result["_id"])
                
                return results, total
            except Exception as e:
                logger.error(f"Error listing templates: {str(e)}")
                return [], 0
    
    # Feedback Operations
    
    async def create_feedback(self, feedback: Dict[str, Any]) -> str:
        """
        Create a new feedback entry.
        
        Args:
            feedback: Feedback data
            
        Returns:
            ID of the created feedback
        """
        feedback_id = feedback.get("feedback_id", str(uuid.uuid4()))
        feedback["feedback_id"] = feedback_id
        feedback["created_at"] = datetime.now()
        
        if self.mock:
            # Convert datetime to string for JSON serialization in mock mode
            feedback["created_at"] = feedback["created_at"].isoformat()
            self.mock_data["feedback"][feedback_id] = feedback
            logger.info(f"Created feedback with ID: {feedback_id} (mock)")
        else:
            try:
                await self.feedback_collection.insert_one(feedback)
                logger.info(f"Created feedback with ID: {feedback_id}")
            except Exception as e:
                logger.error(f"Error creating feedback: {str(e)}")
                raise
        
        return feedback_id
    
    async def get_feedback(self, feedback_id: str) -> Optional[Dict[str, Any]]:
        """
        Get feedback by ID.
        
        Args:
            feedback_id: ID of the feedback to retrieve
            
        Returns:
            Feedback document or None if not found
        """
        if self.mock:
            return self.mock_data["feedback"].get(feedback_id)
        else:
            try:
                result = await self.feedback_collection.find_one({"feedback_id": feedback_id})
                # Convert ObjectId to string for JSON serialization
                if result and "_id" in result:
                    result["_id"] = str(result["_id"])
                return result
            except Exception as e:
                logger.error(f"Error getting feedback: {str(e)}")
                return None
    
    async def get_content_feedback(self, content_id: str) -> List[Dict[str, Any]]:
        """
        Get all feedback for a content item.
        
        Args:
            content_id: ID of the content
            
        Returns:
            List of feedback documents
        """
        if self.mock:
            return [
                feedback for feedback in self.mock_data["feedback"].values()
                if feedback.get("content_id") == content_id
            ]
        else:
            try:
                cursor = self.feedback_collection.find({"content_id": content_id})
                results = await cursor.to_list(length=None)
                
                # Convert ObjectId to string for JSON serialization
                for result in results:
                    if "_id" in result:
                        result["_id"] = str(result["_id"])
                
                return results
            except Exception as e:
                logger.error(f"Error getting content feedback: {str(e)}")
                return []
    
    async def create_feedback_response(self, response_data: Dict[str, Any]) -> str:
        """
        Create a response to feedback.
        
        Args:
            response_data: Response data
            
        Returns:
            Response ID
        """
        response_id = response_data.get("response_id", str(uuid.uuid4()))
        response_data["response_id"] = response_id
        response_data["created_at"] = datetime.now()
        
        if self.mock:
            # Convert datetime to string for JSON serialization in mock mode
            response_data["created_at"] = response_data["created_at"].isoformat()
            
            # Create responses collection if it doesn't exist
            if "feedback_responses" not in self.mock_data:
                self.mock_data["feedback_responses"] = {}
                
            self.mock_data["feedback_responses"][response_id] = response_data
            logger.info(f"Created feedback response with ID: {response_id} (mock)")
        else:
            try:
                await self.db["feedback_responses"].insert_one(response_data)
                logger.info(f"Created feedback response with ID: {response_id}")
            except Exception as e:
                logger.error(f"Error creating feedback response: {str(e)}")
                raise
        
        return response_id
    
    async def get_feedback_responses(self, feedback_id: str) -> List[Dict[str, Any]]:
        """
        Get responses for a feedback item.
        
        Args:
            feedback_id: ID of the feedback
            
        Returns:
            List of response documents
        """
        if self.mock:
            if "feedback_responses" not in self.mock_data:
                return []
                
            return [
                response for response in self.mock_data["feedback_responses"].values()
                if response.get("feedback_id") == feedback_id
            ]
        else:
            try:
                cursor = self.db["feedback_responses"].find({"feedback_id": feedback_id})
                results = await cursor.to_list(length=None)
                
                # Convert ObjectId to string for JSON serialization
                for result in results:
                    if "_id" in result:
                        result["_id"] = str(result["_id"])
                
                return results
            except Exception as e:
                logger.error(f"Error getting feedback responses: {str(e)}")
                return []
    
    async def list_feedback(self, limit: int = 10, offset: int = 0, 
                          filters: Dict[str, Any] = None) -> Tuple[List[Dict[str, Any]], int]:
        """
        List feedback with pagination and filtering.
        
        Args:
            limit: Maximum number of results
            offset: Number of results to skip
            filters: Optional filters
            
        Returns:
            Tuple of (items, total count)
        """
        filters = filters or {}
        
        if self.mock:
            filtered_items = []
            
            for feedback_id, feedback in self.mock_data["feedback"].items():
                # Apply filters if provided
                if filters:
                    matches = True
                    for key, value in filters.items():
                        # Check if filter applies to nested feedback content
                        if key.startswith("feedback."):
                            nested_key = key.split(".", 1)[1]
                            if nested_key not in feedback.get("feedback", {}) or feedback["feedback"][nested_key] != value:
                                matches = False
                                break
                        elif key not in feedback or feedback[key] != value:
                            matches = False
                            break
                    
                    if not matches:
                        continue
                
                # Add to filtered items
                filtered_items.append(feedback)
            
            # Sort by created_at if available
            filtered_items.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            
            # Apply pagination
            paginated_items = filtered_items[offset:offset + limit]
            
            return paginated_items, len(filtered_items)
        else:
            try:
                # Build query from filters
                query = {}
                for key, value in filters.items():
                    query[key] = value
                
                # Get total count
                total = await self.feedback_collection.count_documents(query)
                
                # Get paginated results
                cursor = self.feedback_collection.find(query)
                cursor = cursor.sort("created_at", -1).skip(offset).limit(limit)
                
                results = await cursor.to_list(length=limit)
                
                # Convert ObjectId to string for JSON serialization
                for result in results:
                    if "_id" in result:
                        result["_id"] = str(result["_id"])
                
                return results, total
            except Exception as e:
                logger.error(f"Error listing feedback: {str(e)}")
                return [], 0
                
    async def update_feedback_status(self, feedback_id: str, status: str) -> Optional[Dict[str, Any]]:
        """
        Update the status of a feedback entry.
        
        Args:
            feedback_id: ID of the feedback
            status: New status
            
        Returns:
            Updated feedback document or None if not found
        """
        # Add updated timestamp
        updated_at = datetime.now()
        
        if self.mock:
            if feedback_id not in self.mock_data["feedback"]:
                logger.warning(f"Feedback not found for status update: {feedback_id} (mock)")
                return None
                
            self.mock_data["feedback"][feedback_id]["status"] = status
            self.mock_data["feedback"][feedback_id]["updated_at"] = updated_at.isoformat()
            logger.info(f"Updated feedback status to {status} for ID: {feedback_id} (mock)")
            return self.mock_data["feedback"][feedback_id]
        else:
            try:
                result = await self.feedback_collection.find_one_and_update(
                    {"feedback_id": feedback_id},
                    {"$set": {"status": status, "updated_at": updated_at}},
                    return_document=ReturnDocument.AFTER
                )
                
                if result:
                    # Convert ObjectId to string for JSON serialization
                    if "_id" in result:
                        result["_id"] = str(result["_id"])
                    logger.info(f"Updated feedback status to {status} for ID: {feedback_id}")
                else:
                    logger.warning(f"Feedback not found for status update: {feedback_id}")
                    
                return result
            except Exception as e:
                logger.error(f"Error updating feedback status: {str(e)}")
                return None
                
    # Media Storage Operations with GridFS
    
    async def store_media_file(self, file_data: Union[bytes, BinaryIO, str], 
                              filename: str, 
                              content_type: str,
                              metadata: Dict[str, Any] = None) -> str:
        """
        Store a media file in GridFS.
        
        Args:
            file_data: File data as bytes, file-like object, or base64 string
            filename: Name of the file
            content_type: MIME type of the file
            metadata: Additional metadata to store with the file
            
        Returns:
            ID of the stored file
        """
        # Process input if it's a base64 string
        if isinstance(file_data, str) and file_data.startswith(('data:', 'data:image', 'data:audio')):
            # Extract the base64 part from data URI
            if ',' in file_data:
                file_data = file_data.split(',', 1)[1]
            file_data = base64.b64decode(file_data)
        
        file_id = str(uuid.uuid4())
        metadata = metadata or {}
        metadata.update({
            "file_id": file_id,
            "filename": filename,
            "content_type": content_type,
            "created_at": datetime.now()
        })
        
        if self.mock:
            # In mock mode, just store a reference
            self.mock_data["media"][file_id] = {
                "file_id": file_id,
                "filename": filename,
                "content_type": content_type,
                "metadata": metadata,
                "length": len(file_data) if isinstance(file_data, bytes) else 0,
                "created_at": datetime.now().isoformat()
            }
            logger.info(f"Stored media file with ID: {file_id} (mock)")
        else:
            try:
                # Store file in GridFS
                grid_in = self.fs.open_upload_stream(
                    filename=filename,
                    metadata=metadata
                )
                
                # Write data
                if isinstance(file_data, bytes):
                    grid_in.write(file_data)
                else:
                    # Assume it's a file-like object
                    while True:
                        chunk = file_data.read(261120)  # 255 KB chunks
                        if not chunk:
                            break
                        grid_in.write(chunk)
                
                # Close and get the file ID
                grid_in.close()
                object_id = grid_in._id
                
                # Store a reference with our UUID
                await self.db["media_files"].insert_one({
                    "file_id": file_id,
                    "grid_id": object_id,
                    "filename": filename,
                    "content_type": content_type,
                    "metadata": metadata,
                    "created_at": datetime.now()
                })
                
                logger.info(f"Stored media file with ID: {file_id}")
            except Exception as e:
                logger.error(f"Error storing media file: {str(e)}")
                raise
        
        return file_id
    
    async def get_media_file(self, file_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a media file from GridFS.
        
        Args:
            file_id: ID of the file
            
        Returns:
            Dictionary with file data and metadata, or None if not found
        """
        if self.mock:
            if file_id not in self.mock_data["media"]:
                logger.warning(f"Media file not found: {file_id} (mock)")
                return None
            
            file_info = self.mock_data["media"][file_id]
            # In mock mode, we just return the metadata since we don't have the actual file
            return {
                "file_id": file_id,
                "filename": file_info["filename"],
                "content_type": file_info["content_type"],
                "metadata": file_info.get("metadata", {}),
                "created_at": file_info["created_at"],
                "data": None  # No actual data in mock mode
            }
        else:
            try:
                # Get file reference
                file_ref = await self.db["media_files"].find_one({"file_id": file_id})
                if not file_ref:
                    logger.warning(f"Media file reference not found: {file_id}")
                    return None
                
                # Get file from GridFS
                grid_out = await self.fs.open_download_stream(file_ref["grid_id"])
                
                # Read all data
                chunks = []
                while True:
                    chunk = await grid_out.readchunk()
                    if not chunk:
                        break
                    chunks.append(chunk)
                
                file_data = b''.join(chunks)
                
                return {
                    "file_id": file_id,
                    "filename": file_ref["filename"],
                    "content_type": file_ref["content_type"],
                    "metadata": file_ref.get("metadata", {}),
                    "created_at": file_ref["created_at"],
                    "data": file_data
                }
            except Exception as e:
                logger.error(f"Error getting media file: {str(e)}")
                return None
    
    async def delete_media_file(self, file_id: str) -> bool:
        """
        Delete a media file from GridFS.
        
        Args:
            file_id: ID of the file
            
        Returns:
            True if deleted, False if not found or error
        """
        if self.mock:
            if file_id not in self.mock_data["media"]:
                logger.warning(f"Media file not found for deletion: {file_id} (mock)")
                return False
            
            del self.mock_data["media"][file_id]
            logger.info(f"Deleted media file with ID: {file_id} (mock)")
            return True
        else:
            try:
                # Get file reference
                file_ref = await self.db["media_files"].find_one({"file_id": file_id})
                if not file_ref:
                    logger.warning(f"Media file reference not found for deletion: {file_id}")
                    return False
                
                # Delete file from GridFS
                await self.fs.delete(file_ref["grid_id"])
                
                # Delete reference
                result = await self.db["media_files"].delete_one({"file_id": file_id})
                
                if result.deleted_count > 0:
                    logger.info(f"Deleted media file with ID: {file_id}")
                    return True
                else:
                    logger.warning(f"Media file reference not found for deletion: {file_id}")
                    return False
            except Exception as e:
                logger.error(f"Error deleting media file: {str(e)}")
                return False
    
    async def list_media_files(self, filters: Dict[str, Any] = None, 
                             limit: int = 50, 
                             offset: int = 0) -> Tuple[List[Dict[str, Any]], int]:
        """
        List media files with pagination and filtering.
        
        Args:
            filters: Optional filters
            limit: Maximum number of items
            offset: Number of items to skip
            
        Returns:
            Tuple of (list of media files, total count)
        """
        filters = filters or {}
        
        if self.mock:
            filtered_items = []
            for file_id, file_info in self.mock_data["media"].items():
                # Apply filters
                if filters:
                    matches = True
                    for key, value in filters.items():
                        # Handle metadata fields
                        if key.startswith("metadata."):
                            meta_key = key.split(".", 1)[1]
                            if meta_key not in file_info.get("metadata", {}) or file_info["metadata"][meta_key] != value:
                                matches = False
                                break
                        elif key not in file_info or file_info[key] != value:
                            matches = False
                            break
                    
                    if not matches:
                        continue
                
                filtered_items.append(file_info)
            
            # Sort by created_at if available
            filtered_items.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            
            # Apply pagination
            paginated_items = filtered_items[offset:offset + limit]
            
            return paginated_items, len(filtered_items)
        else:
            try:
                # Build query from filters
                query = {}
                for key, value in filters.items():
                    query[key] = value
                
                # Get total count
                total = await self.db["media_files"].count_documents(query)
                
                # Get paginated results
                cursor = self.db["media_files"].find(query)
                cursor = cursor.sort("created_at", -1).skip(offset).limit(limit)
                
                results = await cursor.to_list(length=limit)
                
                # Convert ObjectId to string for JSON serialization
                for result in results:
                    if "_id" in result:
                        result["_id"] = str(result["_id"])
                    if "grid_id" in result:
                        result["grid_id"] = str(result["grid_id"])
                
                return results, total
            except Exception as e:
                logger.error(f"Error listing media files: {str(e)}")
                return [], 0