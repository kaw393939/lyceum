"""
Chat models for Socrates application
"""
from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class Message(BaseModel):
    """Chat message model."""
    role: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp
        }
    
    def to_llm_format(self) -> Dict[str, str]:
        """Convert to LLM API format."""
        return {
            "role": self.role,
            "content": self.content
        }


class ChatHistory(BaseModel):
    """Chat history model."""
    id: Optional[str] = None
    user_id: str
    title: str
    messages: List[Message] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def add_message(self, role: str, content: str) -> None:
        """Add a message to the chat history."""
        message = Message(role=role, content=content)
        self.messages.append(message)
        self.updated_at = datetime.now()
    
    def get_messages_for_llm(self) -> List[Dict[str, str]]:
        """Get messages in a format suitable for LLM API."""
        return [msg.to_llm_format() for msg in self.messages]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "title": self.title,
            "messages": [msg.to_dict() for msg in self.messages],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata
        }


class UserSession(BaseModel):
    """User session model."""
    user_id: str
    active_chat_id: Optional[str] = None
    last_active: datetime = Field(default_factory=datetime.now)
    preferences: Dict[str, Any] = Field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "user_id": self.user_id,
            "active_chat_id": self.active_chat_id,
            "last_active": self.last_active,
            "preferences": self.preferences
        }