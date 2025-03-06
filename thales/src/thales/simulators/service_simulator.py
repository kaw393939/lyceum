import os
import yaml
import json
import random
import asyncio
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Union

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel


class ServiceSimulator:
    """Simulates a microservice with configurable response delays and errors."""
    
    def __init__(
        self, 
        service_name: str, 
        port: int,
        error_rate: float = 0.0,
        delay_ms: int = 0,
        config_file: Optional[str] = None
    ):
        """Initialize the service simulator.
        
        Args:
            service_name: Name of the service to simulate
            port: Port to run the service on
            error_rate: Probability of generating an error (0.0 to 1.0)
            delay_ms: Artificial delay in milliseconds
            config_file: Optional path to simulator configuration file
        """
        self.service_name = service_name
        self.port = port
        self.error_rate = error_rate
        self.delay_ms = delay_ms
        
        # Create FastAPI app
        self.app = FastAPI(title=f"{service_name} Simulator")
        
        # Load configuration if provided
        self.config = self._load_config(config_file)
        
        # Set up routes
        self.setup_routes()
        
    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file if provided.
        
        Args:
            config_file: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        if not config_file:
            return {}
            
        if not os.path.exists(config_file):
            return {}
            
        try:
            with open(config_file, "r") as f:
                if config_file.endswith(".yaml") or config_file.endswith(".yml"):
                    return yaml.safe_load(f)
                else:
                    return json.load(f)
        except Exception:
            return {}
            
    def setup_routes(self):
        """Set up default routes for the service simulator."""
        # Health check endpoint
        @self.app.get("/health")
        async def health_check():
            await self._simulate_conditions()
            return {"status": "ok", "service": self.service_name}
            
        # Add Ptolemy-specific routes if simulating Ptolemy
        if self.service_name.lower() == "ptolemy":
            self._setup_ptolemy_v1_routes()  # Add v1 API endpoints first
            self._setup_ptolemy_routes()
            
        # Add Gutenberg-specific routes if simulating Gutenberg
        elif self.service_name.lower() == "gutenberg":
            self._setup_gutenberg_v1_routes()  # Add v1 API endpoints first
            self._setup_gutenberg_routes()
            
        # Add catch-all route for any other endpoints - this must be last
        @self.app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
        async def catch_all(request: Request, path: str):
            await self._simulate_conditions()
            
            # Special case for v1 API endpoints with trailing slash
            if path.startswith("api/v1/") and not path.endswith("/"):
                # Redirect to the same path with trailing slash
                return JSONResponse(
                    status_code=404,
                    content={
                        "detail": f"Not Found: /{path}. You might need to use the correct API version."
                    }
                )
            
            # Get request body if any
            body = {}
            if request.method in ["POST", "PUT"]:
                try:
                    body = await request.json()
                except:
                    body = {}
            
            # Return a generic response
            return {
                "status": "success",
                "message": f"Simulated response for {request.method} /{path}",
                "received_body": body,
                "service": self.service_name
            }
            
    def _setup_ptolemy_routes(self):
        """Set up Ptolemy-specific routes."""
        # Get concept by ID
        @self.app.get("/api/concepts/{concept_id}")
        async def get_concept(concept_id: str):
            await self._simulate_conditions()
            return {
                "id": concept_id,
                "name": f"Concept {concept_id}",
                "description": f"Description for concept {concept_id}",
                "relationships": [
                    {
                        "target_id": f"related_{i}",
                        "type": random.choice(["PREREQUISITE", "RELATED", "PART_OF"]),
                        "weight": round(random.uniform(0.5, 1.0), 2)
                    }
                    for i in range(3)
                ]
            }
            
        # Get related concepts
        @self.app.get("/api/concepts/{concept_id}/related")
        async def get_related_concepts(concept_id: str, relationship_type: Optional[str] = None):
            await self._simulate_conditions()
            
            # Filter by relationship type if specified
            if relationship_type:
                relationship_types = [relationship_type]
            else:
                relationship_types = ["PREREQUISITE", "RELATED", "PART_OF"]
                
            # Generate random related concepts
            related_concepts = []
            for i in range(5):
                rel_type = random.choice(relationship_types)
                related_concepts.append({
                    "id": f"related_{i}",
                    "name": f"Related Concept {i}",
                    "relationship_type": rel_type,
                    "weight": round(random.uniform(0.5, 1.0), 2)
                })
                
            return {
                "concept_id": concept_id,
                "related_concepts": related_concepts
            }
            
        # Search concepts
        @self.app.post("/api/concepts/search")
        async def search_concepts(request: Request):
            await self._simulate_conditions()
            
            # Parse request body
            try:
                body = await request.json()
                query = body.get("query", "")
                limit = body.get("limit", 10)
            except:
                query = ""
                limit = 10
                
            # Generate random search results
            results = []
            for i in range(limit):
                results.append({
                    "id": f"result_{i}",
                    "name": f"Search Result {i} for '{query}'",
                    "description": f"Description for search result {i}",
                    "relevance": round(1.0 - (i * 0.1), 2)
                })
                
            return {
                "query": query,
                "results": results,
                "total": limit
            }
            
    def _setup_ptolemy_v1_routes(self):
        """Set up Ptolemy v1 API routes for integration testing."""
        # Store concept data for later retrieval
        concept_store = {}
        
        # Create concept
        @self.app.post("/api/v1/concepts")
        async def create_concept(request: Request):
            await self._simulate_conditions()
            
            # Parse request body
            try:
                body = await request.json()
                name = body.get("name", "")
                description = body.get("description", "")
                domains = body.get("domains", [])
                concept_id = body.get("id", f"concept_{int(time.time())}")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid request: {str(e)}")
                
            # Store the concept data for future GET requests
            concept_store[concept_id] = {
                "id": concept_id,
                "name": name,
                "description": description,
                "domains": domains,
                "relationships": [
                    {
                        "target_id": f"related_{i}",
                        "type": random.choice(["PREREQUISITE", "RELATED", "PART_OF"]),
                        "weight": round(random.uniform(0.5, 1.0), 2)
                    }
                    for i in range(3)
                ],
                "created_at": datetime.now().isoformat()
            }
                
            # Return concept with ID
            return {
                "id": concept_id,
                "name": name,
                "description": description,
                "domains": domains,
                "created_at": datetime.now().isoformat()
            }
            
        # Get concept by ID
        @self.app.get("/api/v1/concepts/{concept_id}")
        async def get_concept_v1(concept_id: str):
            await self._simulate_conditions()
            
            # If concept_id starts with "non_existent", return 404
            if concept_id.startswith("non_existent"):
                raise HTTPException(status_code=404, detail=f"Concept not found: {concept_id}")
                
            # Return the stored concept if it exists
            if concept_id in concept_store:
                return concept_store[concept_id]
                
            # Otherwise generate a generic concept
            return {
                "id": concept_id,
                "name": f"Concept {concept_id}",
                "description": f"Description for concept {concept_id}",
                "domains": ["sample", "test"],
                "created_at": datetime.now().isoformat(),
                "relationships": [
                    {
                        "target_id": f"related_{i}",
                        "type": random.choice(["PREREQUISITE", "RELATED", "PART_OF"]),
                        "weight": round(random.uniform(0.5, 1.0), 2)
                    }
                    for i in range(3)
                ]
            }
            
            # If concept_id starts with "non_existent", return 404
            if concept_id.startswith("non_existent"):
                raise HTTPException(status_code=404, detail=f"Concept not found: {concept_id}")
                
            # Return concept with its original data
            try:
                # Try to parse concept_id to extract a body if it was created with POST
                if "_" in concept_id:
                    # This is likely a concept we created via POST, so return its original data
                    name = concept_id.split("_")[-1] if "_" in concept_id else concept_id
                    return {
                        "id": concept_id,
                        "name": test_concept["name"] if "test_concept" in globals() else f"Concept {concept_id}",
                        "description": f"Description for concept {concept_id}",
                        "domains": ["sample", "test"],
                        "created_at": datetime.now().isoformat(),
                        "relationships": [
                            {
                                "target_id": f"related_{i}",
                                "type": random.choice(["PREREQUISITE", "RELATED", "PART_OF"]),
                                "weight": round(random.uniform(0.5, 1.0), 2)
                            }
                            for i in range(3)
                        ]
                    }
                else:
                    # Return generic concept
                    return {
                        "id": concept_id,
                        "name": f"Concept {concept_id}",
                        "description": f"Description for concept {concept_id}",
                        "domains": ["sample", "test"],
                        "created_at": datetime.now().isoformat(),
                        "relationships": [
                            {
                                "target_id": f"related_{i}",
                                "type": random.choice(["PREREQUISITE", "RELATED", "PART_OF"]),
                                "weight": round(random.uniform(0.5, 1.0), 2)
                            }
                            for i in range(3)
                        ]
                    }
            except Exception as e:
                # If any error, fall back to generic response
                return {
                    "id": concept_id,
                    "name": f"Concept {concept_id}",
                    "description": f"Description for concept {concept_id}",
                    "domains": ["sample", "test"],
                    "created_at": datetime.now().isoformat(),
                    "relationships": [
                        {
                            "target_id": f"related_{i}",
                            "type": random.choice(["PREREQUISITE", "RELATED", "PART_OF"]),
                            "weight": round(random.uniform(0.5, 1.0), 2)
                        }
                        for i in range(3)
                    ]
                }
            
        # Delete concept
        @self.app.delete("/api/v1/concepts/{concept_id}")
        async def delete_concept(concept_id: str):
            await self._simulate_conditions()
            return {"status": "success", "message": f"Deleted concept {concept_id}"}
            
        # Create relationship
        @self.app.post("/api/v1/relationships")
        async def create_relationship(request: Request):
            await self._simulate_conditions()
            
            # Parse request body
            try:
                body = await request.json()
                source_id = body.get("source_id", "")
                target_id = body.get("target_id", "")
                rel_type = body.get("type", "RELATED")
                strength = body.get("strength", 0.5)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid request: {str(e)}")
                
            # Return relationship
            return {
                "id": f"rel_{source_id}_{target_id}",
                "source_id": source_id,
                "target_id": target_id,
                "type": rel_type,
                "strength": strength,
                "created_at": datetime.now().isoformat()
            }
            
        # Create learning path
        @self.app.post("/api/v1/learning-paths")
        async def create_learning_path(request: Request):
            await self._simulate_conditions()
            
            # Parse request body
            try:
                body = await request.json()
                name = body.get("name", "")
                description = body.get("description", "")
                concepts = body.get("concepts", [])
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid request: {str(e)}")
                
            # Return learning path
            return {
                "id": f"path_{int(time.time())}",
                "name": name,
                "description": description,
                "concepts": concepts,
                "created_at": datetime.now().isoformat()
            }
            
        # Delete learning path
        @self.app.delete("/api/v1/learning-paths/{path_id}")
        async def delete_learning_path(path_id: str):
            await self._simulate_conditions()
            return {"status": "success", "message": f"Deleted learning path {path_id}"}
            
        # Search endpoint (v1)
        @self.app.post("/api/v1/search")
        async def search_v1(request: Request):
            await self._simulate_conditions()
            
            # Parse request body
            try:
                body = await request.json()
                query = body.get("query", "")
                limit = body.get("limit", 10)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid request: {str(e)}")
                
            # Generate search results
            results = []
            for i in range(limit):
                results.append({
                    "id": f"result_{i}",
                    "name": f"Search Result {i} for '{query}'",
                    "description": f"Description for search result {i}",
                    "relevance": round(1.0 - (i * 0.1), 2),
                    "domains": ["test", "sample"]
                })
                
            return {
                "query": query,
                "results": results,
                "total": limit
            }
            
    def _setup_gutenberg_routes(self):
        """Set up Gutenberg-specific routes."""
        # Generate content
        @self.app.post("/api/content/generate")
        async def generate_content(request: Request):
            await self._simulate_conditions()
            
            # Parse request body
            try:
                body = await request.json()
                concept_id = body.get("concept_id", "unknown")
                format = body.get("format", "lesson")
            except:
                concept_id = "unknown"
                format = "lesson"
                
            # Generate simulated content
            content = f"# Content for {concept_id}\n\n"
            
            if format == "lesson":
                content += "## Introduction\n\n"
                content += f"This is a simulated lesson about {concept_id}.\n\n"
                content += "## Main Content\n\n"
                content += "Lorem ipsum dolor sit amet, consectetur adipiscing elit.\n\n"
                content += "## Conclusion\n\n"
                content += "In conclusion, this is a simulated lesson."
            elif format == "assessment":
                content += "## Assessment Questions\n\n"
                content += "1. What is the main concept of this lesson?\n"
                content += "2. Explain the importance of this concept.\n"
                content += "3. How would you apply this concept in real life?"
            else:
                content += f"Generic content for format: {format}"
                
            return {
                "content_id": f"generated_{concept_id}_{format}",
                "concept_id": concept_id,
                "format": format,
                "content": content,
                "created_at": "2025-03-05T12:00:00Z"
            }
            
        # Get content by ID
        @self.app.get("/api/content/{content_id}")
        async def get_content(content_id: str):
            await self._simulate_conditions()
            
            # Parse content ID to extract concept and format
            parts = content_id.split("_")
            if len(parts) >= 3:
                concept_id = parts[1]
                format = parts[2]
            else:
                concept_id = "unknown"
                format = "unknown"
                
            # Generate simulated content
            content = f"# Content for {concept_id}\n\n"
            content += f"This is a simulated {format} about {concept_id}."
                
            return {
                "content_id": content_id,
                "concept_id": concept_id,
                "format": format,
                "content": content,
                "created_at": "2025-03-05T12:00:00Z"
            }
            
    def _setup_gutenberg_v1_routes(self):
        """Set up Gutenberg v1 API routes for integration testing."""
        # Store concept ID and content ID mappings
        concept_content_map = {}
        
        # Generate content
        @self.app.post("/api/v1/content/generate")
        async def generate_content_v1(request: Request):
            await self._simulate_conditions()
            
            # Parse request body
            try:
                body = await request.json()
                
                # Check for required field "content_type"
                if "content_type" not in body:
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "code": "validation_error",
                            "message": "Request validation error",
                            "details": {
                                "field_errors": {
                                    "content_type": "Field required"
                                }
                            }
                        }
                    )
                
                concept_id = body.get("concept_id", "")
                format = body.get("format", "lesson")
                style = body.get("style", "standard")
                target_audience = body.get("target_audience", "beginner")
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid request: {str(e)}")
                
            # If concept_id starts with "non_existent", return error with concept mention
            if concept_id.startswith("non_existent"):
                raise HTTPException(
                    status_code=404, 
                    detail={
                        "code": "not_found",
                        "message": f"Concept not found",
                        "details": {
                            "concept_id": concept_id,
                            "reason": "The requested concept does not exist in the knowledge base."
                        }
                    }
                )
                
            # Generate request ID and content ID
            timestamp = int(time.time())
            request_id = f"req_{concept_id}_{timestamp}"
            content_id = f"content_{request_id}"
            
            # Store in our request content map 
            self.content_store = getattr(self, 'content_store', {})
            self.content_store[content_id] = {
                "concept_id": concept_id,
                "format": format,
                "style": style,
                "target_audience": target_audience
            }
            
            return {
                "request_id": request_id,
                "status": "processing",
                "estimated_time_seconds": 5
            }
            
        # Get content generation status
        @self.app.get("/api/v1/content/status/{request_id}")
        async def get_content_status(request_id: str):
            await self._simulate_conditions()
            
            # Simulate completed status
            content_id = f"content_{request_id}"
            
            return {
                "request_id": request_id,
                "status": "complete",
                "content_id": content_id
            }
            
        # Get content by ID
        @self.app.get("/api/v1/content/{content_id}")
        async def get_content_v1(content_id: str):
            await self._simulate_conditions()
            
            # Get the content store
            self.content_store = getattr(self, 'content_store', {})
            
            # Check if we have stored the concept ID for this content
            if content_id in self.content_store:
                content_data = self.content_store[content_id]
                concept_id = content_data["concept_id"]
                format = content_data.get("format", "lesson")
            else:
                # Fallback: Try to parse from the content ID
                parts = content_id.split("_")
                if len(parts) >= 4 and parts[1] == "req":
                    # Get full concept ID (which might contain multiple parts)
                    concept_parts = parts[2:-1]  # exclude "content", "req" and the timestamp
                    concept_id = "_".join(concept_parts)
                else:
                    concept_id = "unknown"
                format = "lesson"
                
            # Generate simulated content
            content = f"# Content for {concept_id}\n\n"
            content += "## Introduction\n\n"
            content += f"This is a simulated lesson about {concept_id}.\n\n"
            content += "## Main Content\n\n"
            content += "Lorem ipsum dolor sit amet, consectetur adipiscing elit.\n\n"
            content += "## Conclusion\n\n"
            content += "In conclusion, this is a simulated lesson."
                
            return {
                "id": content_id,
                "concept_id": concept_id,
                "format": "lesson",
                "content": content,
                "created_at": datetime.now().isoformat()
            }
            
        # Generate content for learning path
        @self.app.post("/api/v1/content/generate-path")
        async def generate_path_content(request: Request):
            await self._simulate_conditions()
            
            # Parse request body
            try:
                body = await request.json()
                path_id = body.get("learning_path_id", "")
                format = body.get("format", "lesson")
                target_audience = body.get("target_audience", "beginner")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid request: {str(e)}")
                
            # Return batch ID for learning path content generation
            return {
                "batch_id": f"batch_{path_id}_{int(time.time())}",
                "status": "processing",
                "estimated_time_seconds": 15
            }
            
        # Get batch status
        @self.app.get("/api/v1/content/batch/{batch_id}/status")
        async def get_batch_status(batch_id: str):
            await self._simulate_conditions()
            
            # Return simulated batch status
            return {
                "batch_id": batch_id,
                "status": "processing",
                "completed": 2,
                "total": 5,
                "estimated_time_remaining_seconds": 10
            }
            
    async def _simulate_conditions(self):
        """Simulates real-world conditions like delays and errors."""
        # Simulate network delay
        if self.delay_ms > 0:
            await asyncio.sleep(self.delay_ms / 1000)
            
        # Simulate random errors
        if random.random() < self.error_rate:
            error_types = [
                (500, "Internal server error"),
                (502, "Bad gateway"),
                (503, "Service unavailable"),
                (504, "Gateway timeout"),
            ]
            status_code, detail = random.choice(error_types)
            raise HTTPException(status_code=status_code, detail=detail)