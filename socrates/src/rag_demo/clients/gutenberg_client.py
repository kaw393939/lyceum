"""
Gutenberg Client
=============
Client for interacting with the Gutenberg content generation system API.
"""

import os
import json
import httpx
import asyncio
from typing import Dict, List, Optional, Any, Tuple

class GutenbergClient:
    """Client for interacting with the Gutenberg content generation system."""
    
    def __init__(self):
        """Initialize the Gutenberg client."""
        # Configure base URL from environment variables or use default
        self.base_url = os.environ.get("GUTENBERG_URL", "http://gutenberg:8001")
        self.api_prefix = "/api/v1"
        self.api_key = os.environ.get("GUTENBERG_API_KEY", "")
        
        # Check if we should use mock mode (for development/testing)
        self.use_mock = os.environ.get("GUTENBERG_USE_MOCK", "true").lower() in ("true", "1")
        if not self.api_key:
            self.use_mock = True
            print("No Gutenberg API key found, falling back to mock mode.")
            
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
        
    async def search_content(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for content using semantic search.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            
        Returns:
            List of search results with content and score
        """
        if self.use_mock:
            return self._mock_search_content(query, limit)
            
        try:
            async with httpx.AsyncClient(headers=self.headers, timeout=30.0) as client:
                response = await client.get(
                    f"{self.base_url}{self.api_prefix}/content/search",
                    params={"query": query, "limit": limit}
                )
                
                response.raise_for_status()
                results = response.json().get("results", [])
                
                # Format results for RAG
                formatted_results = []
                for item in results:
                    formatted_results.append({
                        "content": item.get("content", ""),
                        "metadata": {
                            "id": item.get("content_id"),
                            "title": item.get("title"),
                            "type": item.get("content_type"),
                            "source": "gutenberg"
                        },
                        "score": item.get("score", 0.0)
                    })
                
                return formatted_results
                
        except Exception as e:
            print(f"Error searching content: {e}")
            # Fall back to mock data on error
            return self._mock_search_content(query, limit)
    
    async def get_content(self, content_id: str) -> Dict[str, Any]:
        """
        Get content by ID.
        
        Args:
            content_id: ID of the content
            
        Returns:
            Content data
        """
        if self.use_mock:
            return self._mock_get_content(content_id)
            
        try:
            async with httpx.AsyncClient(headers=self.headers, timeout=30.0) as client:
                response = await client.get(
                    f"{self.base_url}{self.api_prefix}/content/{content_id}"
                )
                
                response.raise_for_status()
                return response.json()
                
        except Exception as e:
            print(f"Error getting content: {e}")
            # Fall back to mock data on error
            return self._mock_get_content(content_id)
    
    async def generate_content(self, 
                             concept_id: str, 
                             content_type: str = "concept_explanation",
                             difficulty: str = "intermediate") -> Dict[str, Any]:
        """
        Request generation of new content.
        
        Args:
            concept_id: ID of the concept to generate content for
            content_type: Type of content to generate
            difficulty: Difficulty level
            
        Returns:
            Generated content data
        """
        if self.use_mock:
            return self._mock_generate_content(concept_id, content_type, difficulty)
            
        try:
            request_data = {
                "concept_id": concept_id,
                "content_type": content_type,
                "difficulty": difficulty
            }
            
            async with httpx.AsyncClient(headers=self.headers, timeout=60.0) as client:
                # Create generation request
                response = await client.post(
                    f"{self.base_url}{self.api_prefix}/content/generate",
                    json=request_data
                )
                
                response.raise_for_status()
                request_info = response.json()
                
                # Wait for generation to complete (with timeout)
                request_id = request_info.get("request_id")
                max_retries = 10
                retry_count = 0
                
                while retry_count < max_retries:
                    # Check status
                    status_response = await client.get(
                        f"{self.base_url}{self.api_prefix}/content/status/{request_id}"
                    )
                    
                    status_info = status_response.json()
                    if status_info.get("status") == "completed":
                        # Get the generated content
                        content_id = status_info.get("content_id")
                        if content_id:
                            return await self.get_content(content_id)
                        else:
                            break
                    
                    if status_info.get("status") == "failed":
                        break
                        
                    # Wait before retrying
                    await asyncio.sleep(3)
                    retry_count += 1
                
                # If we reach here, generation failed or timed out
                return self._mock_generate_content(concept_id, content_type, difficulty)
                
        except Exception as e:
            print(f"Error generating content: {e}")
            # Fall back to mock data on error
            return self._mock_generate_content(concept_id, content_type, difficulty)
    
    # -------- Mock Methods for Development/Testing -------- #
    
    def _mock_search_content(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Generate mock data for content search."""
        mock_content = [
            {
                "content": "Stoicism teaches that virtue is the only true good. The four cardinal virtues are wisdom, courage, justice, and temperance. Stoics focus on distinguishing between what they can control and what they cannot.",
                "metadata": {
                    "id": "stoicism-intro",
                    "title": "Introduction to Stoicism",
                    "type": "lesson",
                    "source": "gutenberg"
                },
                "score": 0.95
            },
            {
                "content": "The dichotomy of control is a fundamental principle in Stoicism. It involves differentiating between things that are within our control (primarily our judgments, actions, and attitudes) and things that are not within our control (external events, others' opinions, etc.).",
                "metadata": {
                    "id": "dichotomy-control",
                    "title": "The Dichotomy of Control",
                    "type": "concept_explanation",
                    "source": "gutenberg"
                },
                "score": 0.92
            },
            {
                "content": "Marcus Aurelius was a Roman Emperor from 161 to 180 CE and a Stoic philosopher. His private journal, now known as 'Meditations', is considered one of the most significant works of Stoic philosophy.",
                "metadata": {
                    "id": "marcus-aurelius",
                    "title": "Marcus Aurelius",
                    "type": "biography",
                    "source": "gutenberg"
                },
                "score": 0.88
            },
            {
                "content": "Epictetus was born a slave and became one of the most influential Stoic philosophers. His teachings were recorded by his student Arrian in two books: the Discourses and the Enchiridion (or Handbook).",
                "metadata": {
                    "id": "epictetus",
                    "title": "Epictetus",
                    "type": "biography",
                    "source": "gutenberg"
                },
                "score": 0.85
            },
            {
                "content": "Practicing the dichotomy of control: 1. Identify a challenging situation. 2. List all aspects of the situation. 3. Sort these aspects into 'within my control' and 'outside my control'. 4. Focus attention and effort solely on things within your control.",
                "metadata": {
                    "id": "doc-exercise",
                    "title": "Dichotomy of Control Exercise",
                    "type": "exercise",
                    "source": "gutenberg"
                },
                "score": 0.82
            }
        ]
        
        # Filter based on query for more realistic results
        query_terms = query.lower().split()
        filtered_content = []
        for item in mock_content:
            score = 0
            for term in query_terms:
                if term in item["metadata"]["title"].lower() or term in item["content"].lower():
                    score += 0.2
            if score > 0:
                content_copy = item.copy()
                content_copy["score"] = min(0.99, item["score"] * (1 + score))
                filtered_content.append(content_copy)
        
        # Use original list if no matches
        results = filtered_content if filtered_content else mock_content
        return sorted(results, key=lambda x: x["score"], reverse=True)[:limit]
    
    def _mock_get_content(self, content_id: str) -> Dict[str, Any]:
        """Generate mock data for a specific content."""
        # Sample mock content
        mock_content_details = {
            "stoicism-intro": {
                "content_id": "stoicism-intro",
                "title": "Introduction to Stoicism",
                "content_type": "lesson",
                "difficulty": "intermediate",
                "content": """
                # Introduction to Stoicism
                
                Stoicism is a school of Hellenistic philosophy founded by Zeno of Citium in Athens in the early 3rd century BCE. 
                It is a philosophy of personal ethics informed by its system of logic and its views on the natural world.
                
                ## Core Principles
                
                Stoicism teaches the development of self-control and fortitude as a means of overcoming destructive emotions. 
                The philosophy holds that becoming a clear and unbiased thinker allows one to understand the universal reason (logos).
                
                The Stoics believed that:
                
                1. Virtue is the only true good
                2. External events are beyond our control
                3. Peace of mind comes from differentiating between what we can and cannot control
                4. We should live in accordance with nature and reason
                
                ## The Four Cardinal Virtues
                
                Stoicism emphasizes the development of four key virtues:
                
                1. **Wisdom** (Sophia): Understanding what is good, bad, and indifferent
                2. **Courage** (Andreia): Facing challenges and adversity with clear judgment
                3. **Justice** (Dikaiosyne): Treating others with fairness and honesty
                4. **Temperance** (Sophrosyne): Exercising moderation and self-discipline
                
                ## Historical Significance
                
                Stoicism has influenced many later philosophical traditions and continues to be relevant today.
                Its principles are often applied in modern contexts including cognitive behavioral therapy,
                leadership training, and personal development.
                """,
                "created_at": "2023-01-15T12:00:00Z",
                "updated_at": "2023-02-20T14:30:00Z",
                "metadata": {
                    "concept_id": "stoicism",
                    "author": "system",
                    "keywords": ["stoicism", "philosophy", "virtue", "ethics"],
                    "sources": ["Meditations", "Discourses", "Letters from a Stoic"]
                }
            },
            "dichotomy-control": {
                "content_id": "dichotomy-control",
                "title": "The Dichotomy of Control",
                "content_type": "concept_explanation",
                "difficulty": "beginner",
                "content": """
                # The Dichotomy of Control
                
                The dichotomy of control is one of the most fundamental principles in Stoic philosophy.
                It involves distinguishing between things that are within our control and things that are not.
                
                ## What Is Within Our Control
                
                According to Epictetus, the only things truly within our control are:
                
                - Our judgments and opinions
                - Our desires and aversions
                - Our own actions and decisions
                
                These are our "internal" domain and represent our true freedom.
                
                ## What Is Not Within Our Control
                
                Everything else falls outside our control:
                
                - Other people's opinions and actions
                - External events and circumstances
                - Our reputation and status
                - Our body and health (to a certain extent)
                - Wealth and material possessions
                
                ## Practical Application
                
                Understanding this distinction leads to greater peace of mind. When we focus only on what we can control:
                
                1. We avoid frustration from trying to control the uncontrollable
                2. We take full responsibility for our own choices
                3. We develop resilience against external circumstances
                4. We find tranquility by accepting what we cannot change
                
                ## Modern Relevance
                
                This principle is echoed in the Serenity Prayer and forms the basis of many modern psychological approaches,
                including Cognitive Behavioral Therapy.
                """,
                "created_at": "2023-03-10T09:15:00Z",
                "updated_at": "2023-03-12T16:20:00Z",
                "metadata": {
                    "concept_id": "dichotomy-of-control",
                    "author": "system",
                    "keywords": ["stoicism", "dichotomy of control", "epictetus", "agency"],
                    "sources": ["Enchiridion", "Discourses"]
                }
            }
        }
        
        # Return mock data for the requested content or a generic response
        return mock_content_details.get(content_id, {
            "content_id": content_id,
            "title": f"Content: {content_id}",
            "content_type": "lesson",
            "difficulty": "intermediate",
            "content": f"This is mock content for {content_id}. The requested content was not found.",
            "created_at": "2023-01-01T00:00:00Z",
            "updated_at": "2023-01-01T00:00:00Z",
            "metadata": {
                "concept_id": "unknown",
                "author": "system",
                "keywords": ["mock", "data"],
                "sources": []
            }
        })
    
    def _mock_generate_content(self, concept_id: str, content_type: str, difficulty: str) -> Dict[str, Any]:
        """Generate mock content dynamically."""
        content_id = f"{concept_id}-{content_type}"
        
        # Different content based on content type
        content_templates = {
            "lesson": f"""
            # Lesson on {concept_id.replace('-', ' ').title()}
            
            This is a comprehensive lesson on {concept_id.replace('-', ' ')}.
            
            ## Introduction
            
            {concept_id.replace('-', ' ').title()} is an important concept in Stoic philosophy.
            
            ## Main Principles
            
            1. First principle
            2. Second principle
            3. Third principle
            
            ## Practical Applications
            
            Here are some ways to apply these concepts in daily life...
            
            ## Conclusion
            
            By understanding {concept_id.replace('-', ' ')}, we can improve our lives and thinking.
            """,
            
            "concept_explanation": f"""
            # Understanding {concept_id.replace('-', ' ').title()}
            
            {concept_id.replace('-', ' ').title()} is a key concept that requires careful explanation.
            
            ## Definition
            
            {concept_id.replace('-', ' ').title()} refers to...
            
            ## Historical Context
            
            The idea was first developed by...
            
            ## Modern Understanding
            
            Today, we understand this concept as...
            """,
            
            "exercise": f"""
            # Exercise: Practicing {concept_id.replace('-', ' ').title()}
            
            This exercise will help you apply {concept_id.replace('-', ' ')} in your daily life.
            
            ## Steps
            
            1. First, reflect on...
            2. Then, practice...
            3. Finally, review your experience...
            
            ## Expected Outcomes
            
            Through this exercise, you should develop...
            """
        }
        
        # Default content if the type is not in our templates
        default_content = f"# {concept_id.replace('-', ' ').title()}\n\nThis is mock generated content about {concept_id.replace('-', ' ')}."
        
        return {
            "content_id": content_id,
            "title": f"{concept_id.replace('-', ' ').title()} - {content_type.replace('_', ' ').title()}",
            "content_type": content_type,
            "difficulty": difficulty,
            "content": content_templates.get(content_type, default_content),
            "created_at": "2023-01-01T00:00:00Z",
            "updated_at": "2023-01-01T00:00:00Z",
            "metadata": {
                "concept_id": concept_id,
                "generated": True,
                "author": "system",
                "keywords": [concept_id.replace('-', ' '), content_type.replace('_', ' ')],
                "sources": []
            }
        }