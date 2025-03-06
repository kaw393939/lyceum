"""
Chat Service for Socrates
======================
Manages chat histories and sessions, providing persistence and retrieval.
"""

import logging
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime

from ..models.chat import ChatHistory, Message, UserSession
from .mongodb_service import MongoDBService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chat_service")

class ChatService:
    """Service for managing chat histories and sessions."""
    
    def __init__(self, use_mock: bool = False):
        """
        Initialize chat service.
        
        Args:
            use_mock: If True, use mock data instead of actual database
        """
        self.db = MongoDBService(use_mock=use_mock)
        self.anonymous_user_prefix = "anon_"
    
    def get_or_create_user_id(self) -> str:
        """
        Get or create a user ID for anonymous users.
        
        Returns:
            User ID
        """
        # In a real implementation, this would use authentication
        # For now, generate a random ID for anonymous users
        return f"{self.anonymous_user_prefix}{uuid.uuid4().hex[:8]}"
    
    async def create_chat(self, user_id: str, title: Optional[str] = None) -> ChatHistory:
        """
        Create a new chat history.
        
        Args:
            user_id: User identifier
            title: Optional title for the chat
            
        Returns:
            New chat history
        """
        if not title:
            title = f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        chat_id = await self.db.create_chat_history(user_id, title)
        
        # Update user session with new chat
        session = await self.get_or_create_user_session(user_id)
        session.active_chat_id = chat_id
        session.last_active = datetime.now()
        await self.db.create_or_update_user_session(user_id, session.to_dict())
        
        return ChatHistory(
            id=chat_id,
            user_id=user_id,
            title=title,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
    
    async def get_chat(self, chat_id: str) -> Optional[ChatHistory]:
        """
        Get a chat history by ID.
        
        Args:
            chat_id: Chat history ID
            
        Returns:
            Chat history or None if not found
        """
        chat_data = await self.db.get_chat_history(chat_id)
        
        if not chat_data:
            return None
        
        # Convert messages to Message objects
        messages = []
        for msg_data in chat_data.get("messages", []):
            messages.append(Message(
                role=msg_data["role"],
                content=msg_data["content"],
                timestamp=msg_data.get("timestamp", datetime.now())
            ))
        
        return ChatHistory(
            id=str(chat_data.get("_id", chat_id)),
            user_id=chat_data.get("user_id", "unknown"),
            title=chat_data.get("title", "Untitled Chat"),
            messages=messages,
            created_at=chat_data.get("created_at", datetime.now()),
            updated_at=chat_data.get("updated_at", datetime.now()),
            metadata=chat_data.get("metadata", {})
        )
    
    async def get_user_chats(self, user_id: str, limit: int = 10) -> List[ChatHistory]:
        """
        Get all chat histories for a user.
        
        Args:
            user_id: User identifier
            limit: Maximum number of records to return
            
        Returns:
            List of chat histories
        """
        chats_data = await self.db.get_user_chat_histories(user_id, limit)
        
        # Convert to ChatHistory objects (without full messages for performance)
        chat_histories = []
        for chat_data in chats_data:
            messages = []
            # Only include the most recent message as a preview
            if chat_data.get("messages"):
                last_message = chat_data["messages"][-1]
                messages.append(Message(
                    role=last_message["role"],
                    content=last_message["content"],
                    timestamp=last_message.get("timestamp", datetime.now())
                ))
            
            chat_histories.append(ChatHistory(
                id=str(chat_data.get("_id")),
                user_id=chat_data.get("user_id", "unknown"),
                title=chat_data.get("title", "Untitled Chat"),
                messages=messages,  # Just the last message
                created_at=chat_data.get("created_at", datetime.now()),
                updated_at=chat_data.get("updated_at", datetime.now()),
                metadata=chat_data.get("metadata", {})
            ))
        
        return chat_histories
    
    async def add_message(self, chat_id: str, role: str, content: str) -> bool:
        """
        Add a message to a chat history.
        
        Args:
            chat_id: Chat history ID
            role: Message role (user/assistant)
            content: Message content
            
        Returns:
            Success status
        """
        message = Message(role=role, content=content)
        
        return await self.db.add_message_to_chat(
            chat_id,
            message.to_dict()
        )
    
    async def update_chat_title(self, chat_id: str, title: str) -> bool:
        """
        Update chat title.
        
        Args:
            chat_id: Chat history ID
            title: New title
            
        Returns:
            Success status
        """
        return await self.db.update_chat_history(
            chat_id,
            {"title": title}
        )
    
    async def delete_chat(self, chat_id: str) -> bool:
        """
        Delete a chat history.
        
        Args:
            chat_id: Chat history ID
            
        Returns:
            Success status
        """
        return await self.db.delete_chat_history(chat_id)
    
    async def get_or_create_user_session(self, user_id: str) -> UserSession:
        """
        Get or create a user session.
        
        Args:
            user_id: User identifier
            
        Returns:
            User session
        """
        session_data = await self.db.get_user_session(user_id)
        
        if not session_data:
            # Create new session
            session = UserSession(
                user_id=user_id,
                last_active=datetime.now()
            )
            await self.db.create_or_update_user_session(user_id, session.to_dict())
            return session
        
        return UserSession(
            user_id=session_data.get("user_id", user_id),
            active_chat_id=session_data.get("active_chat_id"),
            last_active=session_data.get("last_active", datetime.now()),
            preferences=session_data.get("preferences", {})
        )
    
    async def set_active_chat(self, user_id: str, chat_id: str) -> bool:
        """
        Set the active chat for a user session.
        
        Args:
            user_id: User identifier
            chat_id: Chat history ID
            
        Returns:
            Success status
        """
        session = await self.get_or_create_user_session(user_id)
        session.active_chat_id = chat_id
        session.last_active = datetime.now()
        
        return await self.db.create_or_update_user_session(
            user_id,
            session.to_dict()
        )
    
    async def update_user_preferences(self, user_id: str, preferences: Dict[str, Any]) -> bool:
        """
        Update user preferences.
        
        Args:
            user_id: User identifier
            preferences: User preferences
            
        Returns:
            Success status
        """
        session = await self.get_or_create_user_session(user_id)
        session.preferences.update(preferences)
        session.last_active = datetime.now()
        
        return await self.db.create_or_update_user_session(
            user_id,
            session.to_dict()
        )