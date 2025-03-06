"""
Ptolemy Client
=============
Client for interacting with the Ptolemy knowledge system API.
"""

import os
import json
import httpx
import asyncio
from typing import Dict, List, Optional, Any, Tuple

class PtolemyClient:
    """Client for interacting with the Ptolemy knowledge system API."""
    
    def __init__(self):
        """Initialize the Ptolemy client."""
        # Configure base URL from environment variables or use default
        self.base_url = os.environ.get("PTOLEMY_URL", "http://ptolemy:8000")
        self.api_prefix = "/api/v1"
        self.api_key = os.environ.get("PTOLEMY_API_KEY", "")
        
        # Check if we should use mock mode (for development/testing)
        self.use_mock = os.environ.get("PTOLEMY_USE_MOCK", "true").lower() in ("true", "1")
        if not self.api_key:
            self.use_mock = True
            print("No Ptolemy API key found, falling back to mock mode.")
            
        # Set up client session with appropriate headers
        self.headers = {
            "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        # Initialize cache
        self._cache = {}
        
    def _create_client(self):
        """Create an httpx client with proper headers."""
        return httpx.AsyncClient(headers=self.headers, timeout=30.0)
    
    async def search_concepts(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for concepts using semantic search.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            
        Returns:
            List of concept data
        """
        if self.use_mock:
            return self._mock_search_concepts(query, limit)
            
        try:
            async with httpx.AsyncClient(headers=self.headers, timeout=30.0) as client:
                response = await client.get(
                    f"{self.base_url}{self.api_prefix}/concepts/search",
                    params={"query": query, "limit": limit}
                )
                
                response.raise_for_status()
                return response.json().get("results", [])
                
        except Exception as e:
            print(f"Error searching concepts: {e}")
            # Fall back to mock data on error
            return self._mock_search_concepts(query, limit)
    
    async def get_concept(self, concept_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a concept.
        
        Args:
            concept_id: ID of the concept
            
        Returns:
            Concept data
        """
        if self.use_mock:
            return self._mock_get_concept(concept_id)
            
        try:
            async with httpx.AsyncClient(headers=self.headers, timeout=30.0) as client:
                response = await client.get(
                    f"{self.base_url}{self.api_prefix}/concepts/{concept_id}"
                )
                
                response.raise_for_status()
                return response.json()
                
        except Exception as e:
            print(f"Error getting concept: {e}")
            # Fall back to mock data on error
            return self._mock_get_concept(concept_id)
    
    async def get_concept_relationships(self, concept_id: str) -> Dict[str, Any]:
        """
        Get relationships for a concept.
        
        Args:
            concept_id: ID of the concept
            
        Returns:
            Dictionary with relationships by type
        """
        if self.use_mock:
            return self._mock_get_relationships(concept_id)
            
        try:
            async with httpx.AsyncClient(headers=self.headers, timeout=30.0) as client:
                response = await client.get(
                    f"{self.base_url}{self.api_prefix}/concepts/{concept_id}/relationships"
                )
                
                response.raise_for_status()
                return response.json()
                
        except Exception as e:
            print(f"Error getting concept relationships: {e}")
            # Fall back to mock data on error
            return self._mock_get_relationships(concept_id)
    
    async def search_content(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Convenience method to search both concepts and possibly other content.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            
        Returns:
            List of search results with content and score
        """
        concepts = await self.search_concepts(query, limit)
        
        # Format results for consistency with RAG pattern
        results = []
        for concept in concepts:
            results.append({
                "content": concept.get("description", ""),
                "metadata": {
                    "id": concept.get("id"),
                    "name": concept.get("name"),
                    "type": "concept",
                    "source": "ptolemy"
                },
                "score": concept.get("score", 0.0)
            })
            
        return results
    
    # -------- Mock Methods for Development/Testing -------- #
    
    def _mock_search_concepts(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Generate mock data for concept search."""
        mock_concepts = [
            {
                "id": "stoicism-virtues",
                "name": "Stoic Virtues",
                "description": "The four cardinal virtues of Stoicism are wisdom, courage, justice, and temperance.",
                "category": "philosophy",
                "score": 0.95
            },
            {
                "id": "dichotomy-of-control",
                "name": "Dichotomy of Control",
                "description": "The Stoic principle that we should focus only on what we can control and accept what we cannot.",
                "category": "philosophy",
                "score": 0.92
            },
            {
                "id": "cognitive-behavioral-therapy",
                "name": "Cognitive Behavioral Therapy",
                "description": "A psychotherapeutic approach influenced by Stoic philosophy that focuses on challenging cognitive distortions.",
                "category": "psychology",
                "score": 0.84
            },
            {
                "id": "marcus-aurelius",
                "name": "Marcus Aurelius",
                "description": "Roman Emperor and Stoic philosopher, author of 'Meditations'.",
                "category": "philosophy",
                "score": 0.80
            },
            {
                "id": "virtue-ethics",
                "name": "Virtue Ethics",
                "description": "An approach to ethics that emphasizes the development of character and virtues.",
                "category": "philosophy",
                "score": 0.78
            }
        ]
        
        # Filter based on query for more realistic results
        query_terms = query.lower().split()
        filtered_concepts = []
        for concept in mock_concepts:
            score = 0
            for term in query_terms:
                if term in concept["name"].lower() or term in concept["description"].lower():
                    score += 0.2
            if score > 0:
                concept_copy = concept.copy()
                concept_copy["score"] = min(0.99, concept["score"] * (1 + score))
                filtered_concepts.append(concept_copy)
        
        # Use original list if no matches
        results = filtered_concepts if filtered_concepts else mock_concepts
        return sorted(results, key=lambda x: x["score"], reverse=True)[:limit]
    
    def _mock_get_concept(self, concept_id: str) -> Dict[str, Any]:
        """Generate mock data for a specific concept."""
        # Sample mock concepts
        mock_concept_details = {
            "stoicism-virtues": {
                "id": "stoicism-virtues",
                "name": "Stoic Virtues",
                "description": "The four cardinal virtues of Stoicism are wisdom, courage, justice, and temperance.",
                "detailed_description": """
                The Stoics believed that cultivating these four virtues leads to eudaimonia (happiness or flourishing):
                
                1. Wisdom (Sophia) - Understanding what is good, bad, and indifferent
                2. Courage (Andreia) - Facing challenges and adversity with clear judgment
                3. Justice (Dikaiosyne) - Treating others with fairness and honesty
                4. Temperance (Sophrosyne) - Exercising moderation and self-discipline
                
                These virtues are considered the only true goods in Stoic philosophy, as they are entirely within our control.
                """,
                "category": "philosophy",
                "tags": ["stoicism", "ethics", "virtue", "philosophy"],
                "created_at": "2023-01-15T12:00:00Z",
                "updated_at": "2023-02-20T14:30:00Z"
            }
        }
        
        # Return mock data for the requested concept or a generic response
        return mock_concept_details.get(concept_id, {
            "id": concept_id,
            "name": f"Concept: {concept_id}",
            "description": f"Mock description for concept {concept_id}",
            "category": "unknown",
            "tags": ["mock", "data"],
            "created_at": "2023-01-01T00:00:00Z",
            "updated_at": "2023-01-01T00:00:00Z"
        })
    
    def _mock_get_relationships(self, concept_id: str) -> Dict[str, Any]:
        """Generate mock relationship data for a concept."""
        return {
            "prerequisites": [
                {"id": "basic-philosophy", "name": "Basic Philosophy Concepts", "relationship_strength": 0.9},
                {"id": "ancient-greece", "name": "Ancient Greek Philosophy", "relationship_strength": 0.8}
            ],
            "related": [
                {"id": "epicureanism", "name": "Epicureanism", "relationship_strength": 0.7},
                {"id": "cynicism", "name": "Cynicism", "relationship_strength": 0.75}
            ],
            "subtopics": [
                {"id": "stoic-wisdom", "name": "Stoic Wisdom (Sophia)", "relationship_strength": 0.95},
                {"id": "stoic-courage", "name": "Stoic Courage (Andreia)", "relationship_strength": 0.95},
                {"id": "stoic-justice", "name": "Stoic Justice (Dikaiosyne)", "relationship_strength": 0.95},
                {"id": "stoic-temperance", "name": "Stoic Temperance (Sophrosyne)", "relationship_strength": 0.95}
            ]
        }