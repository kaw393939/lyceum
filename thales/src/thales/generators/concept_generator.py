import json
import random
import math
from typing import Dict, List, Any, Optional, Union

import pymongo
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client import models


class MockDataGenerator:
    """Generates mock data for testing and populates test databases."""
    
    def __init__(self, db_clients: Optional[Dict[str, Any]] = None):
        """Initialize the generator with optional database clients.
        
        Args:
            db_clients: Dictionary of database clients
        """
        self.db_clients = db_clients or {}
        
    def generate_mock_concepts(
        self, 
        count: int = 10, 
        output_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Generates mock concept data for testing purposes.
        
        Args:
            count: Number of concepts to generate
            output_path: Optional path to write the generated concepts to
            
        Returns:
            List of generated concept dictionaries
        """
        concepts = []
        for i in range(count):
            concept = {
                "id": f"concept_{i}",
                "name": f"Test Concept {i}",
                "description": f"Description for test concept {i}",
                "relationships": self._generate_mock_relationships(random.randint(1, 5))
            }
            concepts.append(concept)
            
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(concepts, f, indent=2)
                
        return concepts
        
    def _generate_mock_relationships(self, count: int) -> List[Dict[str, Any]]:
        """Generates mock relationship data for concepts.
        
        Args:
            count: Number of relationships to generate
            
        Returns:
            List of generated relationship dictionaries
        """
        relationship_types = ["PREREQUISITE", "RELATED", "PART_OF", "HAS_EXAMPLE"]
        relationships = []
        
        for _ in range(count):
            rel = {
                "target_id": f"concept_{random.randint(1, 100)}",
                "type": random.choice(relationship_types),
                "weight": round(random.uniform(0.1, 1.0), 2)
            }
            relationships.append(rel)
            
        return relationships
        
    def populate_mongodb(
        self, 
        concepts: List[Dict[str, Any]], 
        database: str = "ptolemy_test"
    ) -> int:
        """Populates MongoDB with mock concept data.
        
        Args:
            concepts: List of concept dictionaries
            database: Name of the database to populate
            
        Returns:
            Number of concepts inserted
        """
        if "mongodb" not in self.db_clients:
            raise ValueError("MongoDB client not provided")
            
        client = self.db_clients["mongodb"]
        collection = client[database]["concepts"]
        
        # Clear existing data
        collection.delete_many({})
        
        # Insert new documents
        result = collection.insert_many(concepts)
        return len(result.inserted_ids)
        
    def populate_neo4j(self, concepts: List[Dict[str, Any]]) -> int:
        """Populates Neo4j with mock concept graph.
        
        Args:
            concepts: List of concept dictionaries
            
        Returns:
            Number of concepts created
        """
        if "neo4j" not in self.db_clients:
            raise ValueError("Neo4j driver not provided")
            
        driver = self.db_clients["neo4j"]
        
        with driver.session() as session:
            # Clear existing data
            session.run("MATCH (n) DETACH DELETE n")
            
            # Create concept nodes
            for concept in concepts:
                session.run(
                    """
                    CREATE (c:Concept {
                        id: $id,
                        name: $name,
                        description: $description
                    })
                    """,
                    id=concept["id"],
                    name=concept["name"],
                    description=concept["description"]
                )
                
            # Create relationships
            for concept in concepts:
                for rel in concept.get("relationships", []):
                    # Only create relationships for targets that exist in our concepts
                    target_ids = [c["id"] for c in concepts]
                    if rel["target_id"] in target_ids:
                        session.run(
                            f"""
                            MATCH (a:Concept {{id: $source_id}})
                            MATCH (b:Concept {{id: $target_id}})
                            CREATE (a)-[r:{rel['type']} {{weight: $weight}}]->(b)
                            """,
                            source_id=concept["id"],
                            target_id=rel["target_id"],
                            weight=rel["weight"]
                        )
                    
        return len(concepts)
    
    def _generate_normalized_vector(self, size: int = 768) -> List[float]:
        """Generates a random normalized vector.
        
        Args:
            size: Dimensionality of the vector
            
        Returns:
            List of vector components
        """
        # Generate random vector
        vector = [random.uniform(-1, 1) for _ in range(size)]
        
        # Normalize the vector
        magnitude = math.sqrt(sum(x**2 for x in vector))
        return [x/magnitude for x in vector]
        
    def populate_qdrant(
        self, 
        concepts: List[Dict[str, Any]], 
        collection_name: str = "test_concepts",
        vector_size: int = 768
    ) -> int:
        """Populates Qdrant with mock vector embeddings for concepts.
        
        Args:
            concepts: List of concept dictionaries
            collection_name: Name of the Qdrant collection
            vector_size: Size of the embedding vectors
            
        Returns:
            Number of vectors inserted
        """
        if "qdrant" not in self.db_clients:
            raise ValueError("Qdrant client not provided")
            
        client = self.db_clients["qdrant"]
        
        # Clear existing collection
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
            
        # Create new collection
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE
            )
        )
        
        # Generate mock embeddings
        points = []
        for concept in concepts:
            # Generate random normalized vector
            vector = self._generate_normalized_vector(vector_size)
            
            points.append(
                models.PointStruct(
                    id=concept["id"],
                    vector=vector,
                    payload={"name": concept["name"]}
                )
            )
            
        # Upload in batches
        BATCH_SIZE = 100
        for i in range(0, len(points), BATCH_SIZE):
            batch = points[i:i+BATCH_SIZE]
            client.upsert(
                collection_name=collection_name,
                points=batch
            )
            
        return len(points)
        
    def generate_and_populate_all(
        self, 
        concept_count: int = 100, 
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generates mock data and populates all test databases.
        
        Args:
            concept_count: Number of concepts to generate
            output_path: Optional path to write the generated concepts to
            
        Returns:
            Dictionary with generation and population results
        """
        concepts = self.generate_mock_concepts(count=concept_count, output_path=output_path)
        
        results = {}
        if "mongodb" in self.db_clients:
            results["mongodb"] = self.populate_mongodb(concepts)
        if "neo4j" in self.db_clients:
            results["neo4j"] = self.populate_neo4j(concepts)
        if "qdrant" in self.db_clients:
            results["qdrant"] = self.populate_qdrant(concepts)
            
        return {
            "concepts_generated": len(concepts),
            "databases_populated": results
        }
