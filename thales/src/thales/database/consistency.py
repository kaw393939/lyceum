import json
from typing import Dict, List, Any, Set

from thales.database.inspector import DatabaseInspector


class ConsistencyVerifier:
    """Verifies data consistency between different data stores."""

    def __init__(self, db_inspector: DatabaseInspector):
        """Initialize with database inspector.
        
        Args:
            db_inspector: DatabaseInspector instance
        """
        self.inspector = db_inspector

    def verify_concepts(
        self, 
        source_db: str = "mongodb", 
        target_db: str = "neo4j",
        mongodb_database: str = "ptolemy",
        mongodb_collection: str = "concepts",
        neo4j_label: str = "Concept"
    ) -> Dict[str, Any]:
        """Verifies concept data is consistent between MongoDB and Neo4j.
        
        Args:
            source_db: Source database type (mongodb or neo4j)
            target_db: Target database type (mongodb or neo4j)
            mongodb_database: MongoDB database name
            mongodb_collection: MongoDB collection name
            neo4j_label: Neo4j node label
            
        Returns:
            Dictionary with verification results
        """
        # Get concepts from MongoDB
        mongodb_concepts = json.loads(self.inspector.inspect_mongodb(
            database=mongodb_database,
            collection=mongodb_collection,
            limit=1000
        ))
        
        # Get concepts from Neo4j
        neo4j_query = f"MATCH (c:{neo4j_label}) RETURN c"
        neo4j_results = self.inspector.inspect_neo4j(query=neo4j_query)
        neo4j_concepts = [record["c"] for record in neo4j_results]
        
        # Compare by ID
        mongodb_by_id = {c["_id"]: c for c in mongodb_concepts}
        neo4j_by_id = {c["id"]: c for c in neo4j_concepts}
        
        # Find inconsistencies
        inconsistencies = []
        
        # Check MongoDB concepts missing in Neo4j or with field mismatches
        for concept_id, mongo_concept in mongodb_by_id.items():
            if concept_id not in neo4j_by_id:
                inconsistencies.append({
                    "id": concept_id,
                    "error": "Missing in Neo4j",
                    "source_data": mongo_concept
                })
                continue
                
            # Check key fields match
            neo4j_concept = neo4j_by_id[concept_id]
            for field in ["name", "description"]:
                if mongo_concept.get(field) != neo4j_concept.get(field):
                    inconsistencies.append({
                        "id": concept_id,
                        "error": f"Field mismatch: {field}",
                        "mongodb_value": mongo_concept.get(field),
                        "neo4j_value": neo4j_concept.get(field)
                    })
        
        # Check Neo4j concepts missing in MongoDB
        for concept_id in neo4j_by_id:
            if concept_id not in mongodb_by_id:
                inconsistencies.append({
                    "id": concept_id,
                    "error": "Missing in MongoDB",
                    "source_data": neo4j_by_id[concept_id]
                })
                
        return {
            "total_mongodb_concepts": len(mongodb_concepts),
            "total_neo4j_concepts": len(neo4j_concepts),
            "inconsistencies": inconsistencies,
            "inconsistency_count": len(inconsistencies)
        }
        
    def verify_relationships(
        self,
        mongodb_database: str = "ptolemy",
        mongodb_collection: str = "concepts"
    ) -> Dict[str, Any]:
        """Verifies relationship data is consistent between MongoDB and Neo4j.
        
        Args:
            mongodb_database: MongoDB database name
            mongodb_collection: MongoDB collection name
            
        Returns:
            Dictionary with verification results
        """
        # Get relationships from MongoDB (assuming stored as subdocuments)
        mongodb_concepts = json.loads(self.inspector.inspect_mongodb(
            database=mongodb_database,
            collection=mongodb_collection,
            limit=1000
        ))
        
        # Extract relationships from MongoDB concepts
        mongodb_relationships = []
        for concept in mongodb_concepts:
            concept_id = concept["_id"]
            for rel in concept.get("relationships", []):
                mongodb_relationships.append({
                    "source_id": concept_id,
                    "target_id": rel["target_id"],
                    "type": rel["type"]
                })
                
        # Get relationships from Neo4j
        neo4j_relationships = self.inspector.inspect_neo4j(
            query="""
            MATCH (a:Concept)-[r]->(b:Concept) 
            RETURN a.id as source_id, type(r) as type, b.id as target_id
            """
        )
        
        # Create comparable format for Neo4j relationships
        neo4j_rel_set = {
            f"{r['source_id']}-{r['type']}-{r['target_id']}"
            for r in neo4j_relationships
        }
        mongodb_rel_set = {
            f"{r['source_id']}-{r['type']}-{r['target_id']}"
            for r in mongodb_relationships
        }
        
        # Find differences
        missing_in_neo4j = mongodb_rel_set - neo4j_rel_set
        missing_in_mongodb = neo4j_rel_set - mongodb_rel_set
        
        return {
            "total_mongodb_relationships": len(mongodb_relationships),
            "total_neo4j_relationships": len(neo4j_relationships),
            "missing_in_neo4j": list(missing_in_neo4j),
            "missing_in_mongodb": list(missing_in_mongodb)
        }
        
    def verify_vector_embeddings(
        self,
        mongodb_database: str = "ptolemy",
        mongodb_collection: str = "concepts",
        qdrant_collection: str = "concepts"
    ) -> Dict[str, Any]:
        """Verifies vector embeddings are consistent between Qdrant and concept data.
        
        Args:
            mongodb_database: MongoDB database name
            mongodb_collection: MongoDB collection name
            qdrant_collection: Qdrant collection name
            
        Returns:
            Dictionary with verification results
        """
        # Get concepts from MongoDB
        mongodb_concepts = json.loads(self.inspector.inspect_mongodb(
            database=mongodb_database,
            collection=mongodb_collection,
            limit=100  # Limit for practicality
        ))
        concept_ids = [c["_id"] for c in mongodb_concepts]
        
        # Get vectors from Qdrant
        qdrant_vectors = self.inspector.inspect_qdrant(
            collection=qdrant_collection,
            ids=concept_ids
        )
        
        # Check all concepts have vectors
        qdrant_by_id = {v.id: v for v in qdrant_vectors}
        
        missing_vectors = []
        for concept_id in concept_ids:
            if concept_id not in qdrant_by_id:
                missing_vectors.append(concept_id)
                
        # Calculate consistency percentage
        consistency_percentage = 0
        if concept_ids:
            consistency_percentage = 100 * (len(concept_ids) - len(missing_vectors)) / len(concept_ids)
                
        return {
            "total_concepts": len(concept_ids),
            "total_vectors": len(qdrant_vectors),
            "missing_vectors": missing_vectors,
            "consistency_percentage": consistency_percentage
        }
