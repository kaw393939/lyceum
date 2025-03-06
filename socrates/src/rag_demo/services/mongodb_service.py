"""
MongoDB Service for Socrates
===========================
Provides database connectivity and operations for chat history and user sessions.
"""

import os
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import asyncio

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from bson.objectid import ObjectId

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mongodb_service")

class MongoDBService:
    """Service for MongoDB operations specifically for storing chat history."""
    
    def __init__(self, use_mock: bool = False):
        """
        Initialize MongoDB service.
        
        Args:
            use_mock: If True, use mock data instead of actual database
        """
        self.use_mock = use_mock or os.environ.get("MONGODB_USE_MOCK", "false").lower() in ("true", "1")
        
        if self.use_mock:
            logger.warning("Using mock MongoDB service")
            self.client = None
            self.db = None
            self._mock_data = {
                "chat_histories": {},
                "user_sessions": {}
            }
        else:
            mongodb_uri = os.environ.get("MONGODB_URI", "mongodb://socrates_user:socrates_password@mongodb:27017/socrates")
            database_name = os.environ.get("MONGODB_DATABASE", "socrates")
            
            try:
                # Initialize database client
                self.client = AsyncIOMotorClient(mongodb_uri)
                self.db = self.client[database_name]
                logger.info(f"Connected to MongoDB: {database_name}")
                
                # Test connection
                asyncio.create_task(self._test_connection())
            except Exception as e:
                logger.error(f"Failed to connect to MongoDB: {str(e)}")
                self.use_mock = True
                self.client = None
                self.db = None
                self._mock_data = {
                    "chat_histories": {},
                    "user_sessions": {}
                }
    
    async def _test_connection(self):
        """Test MongoDB connection."""
        try:
            await self.db.command("ping")
            logger.info("MongoDB connection test successful")
        except Exception as e:
            logger.error(f"MongoDB connection test failed: {str(e)}")
            self.use_mock = True
    
    async def create_chat_history(self, user_id: str, title: str = None) -> str:
        """
        Create a new chat history.
        
        Args:
            user_id: User identifier
            title: Optional title for the chat
            
        Returns:
            The ID of the created chat history
        """
        timestamp = datetime.now()
        
        if not title:
            title = f"Chat {timestamp.strftime('%Y-%m-%d %H:%M')}"
        
        chat_history = {
            "user_id": user_id,
            "title": title,
            "messages": [],
            "created_at": timestamp,
            "updated_at": timestamp,
            "metadata": {
                "source": "web",
                "version": "1.0",
                "tags": []
            }
        }
        
        if self.use_mock:
            chat_id = str(ObjectId())
            self._mock_data["chat_histories"][chat_id] = chat_history
            return chat_id
        
        result = await self.db.chat_histories.insert_one(chat_history)
        return str(result.inserted_id)
    
    async def get_chat_history(self, chat_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a chat history by ID.
        
        Args:
            chat_id: Chat history ID
            
        Returns:
            Chat history object or None if not found
        """
        if self.use_mock:
            return self._mock_data["chat_histories"].get(chat_id)
        
        return await self.db.chat_histories.find_one({"_id": ObjectId(chat_id)})
    
    async def update_chat_history(self, chat_id: str, update_data: Dict[str, Any]) -> bool:
        """
        Update a chat history.
        
        Args:
            chat_id: Chat history ID
            update_data: Data to update
            
        Returns:
            Success status
        """
        update_data["updated_at"] = datetime.now()
        
        if self.use_mock:
            if chat_id in self._mock_data["chat_histories"]:
                self._mock_data["chat_histories"][chat_id].update(update_data)
                return True
            return False
        
        result = await self.db.chat_histories.update_one(
            {"_id": ObjectId(chat_id)},
            {"$set": update_data}
        )
        
        return result.modified_count > 0
    
    async def add_message_to_chat(self, chat_id: str, message: Dict[str, Any]) -> bool:
        """
        Add a message to chat history.
        
        Args:
            chat_id: Chat history ID
            message: Message object with role, content, and timestamp
            
        Returns:
            Success status
        """
        if "timestamp" not in message:
            message["timestamp"] = datetime.now()
        
        if self.use_mock:
            if chat_id in self._mock_data["chat_histories"]:
                if "messages" not in self._mock_data["chat_histories"][chat_id]:
                    self._mock_data["chat_histories"][chat_id]["messages"] = []
                self._mock_data["chat_histories"][chat_id]["messages"].append(message)
                self._mock_data["chat_histories"][chat_id]["updated_at"] = datetime.now()
                return True
            return False
        
        result = await self.db.chat_histories.update_one(
            {"_id": ObjectId(chat_id)},
            {
                "$push": {"messages": message},
                "$set": {"updated_at": datetime.now()}
            }
        )
        
        return result.modified_count > 0
    
    async def get_user_chat_histories(self, user_id: str, limit: int = 10, skip: int = 0) -> List[Dict[str, Any]]:
        """
        Get all chat histories for a user.
        
        Args:
            user_id: User identifier
            limit: Maximum number of records to return
            skip: Number of records to skip
            
        Returns:
            List of chat histories
        """
        if self.use_mock:
            # Filter and sort mock data
            histories = [
                h for h in self._mock_data["chat_histories"].values()
                if h.get("user_id") == user_id
            ]
            histories.sort(key=lambda x: x.get("updated_at", datetime.min), reverse=True)
            return histories[skip:skip+limit]
        
        cursor = self.db.chat_histories.find({"user_id": user_id})
        cursor.sort("updated_at", -1).skip(skip).limit(limit)
        
        return await cursor.to_list(length=limit)
    
    async def create_or_update_user_session(self, user_id: str, session_data: Dict[str, Any]) -> bool:
        """
        Create or update a user session.
        
        Args:
            user_id: User identifier
            session_data: Session data to store
            
        Returns:
            Success status
        """
        session_data["updated_at"] = datetime.now()
        
        if self.use_mock:
            self._mock_data["user_sessions"][user_id] = session_data
            return True
        
        result = await self.db.user_sessions.update_one(
            {"user_id": user_id},
            {"$set": session_data},
            upsert=True
        )
        
        return result.modified_count > 0 or result.upserted_id is not None
    
    async def get_user_session(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a user session.
        
        Args:
            user_id: User identifier
            
        Returns:
            Session data or None if not found
        """
        if self.use_mock:
            return self._mock_data["user_sessions"].get(user_id)
        
        return await self.db.user_sessions.find_one({"user_id": user_id})
    
    async def delete_chat_history(self, chat_id: str) -> bool:
        """
        Delete a chat history.
        
        Args:
            chat_id: Chat history ID
            
        Returns:
            Success status
        """
        if self.use_mock:
            if chat_id in self._mock_data["chat_histories"]:
                del self._mock_data["chat_histories"][chat_id]
                return True
            return False
        
        result = await self.db.chat_histories.delete_one({"_id": ObjectId(chat_id)})
        return result.deleted_count > 0