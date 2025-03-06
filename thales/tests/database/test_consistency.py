import pytest
import json
from unittest.mock import patch, MagicMock

from thales.database.consistency import ConsistencyVerifier
from thales.database.inspector import DatabaseInspector


class TestConsistencyVerifier:
    @pytest.fixture
    def mock_inspector(self):
        inspector = MagicMock(spec=DatabaseInspector)
        return inspector

    @pytest.fixture
    def verifier(self, mock_inspector):
        return ConsistencyVerifier(mock_inspector)

    def test_initialization(self, verifier, mock_inspector):
        """Test that the verifier is initialized with the inspector"""
        assert verifier.inspector == mock_inspector

    def test_verify_concepts(self, verifier, mock_inspector):
        """Test concept verification between MongoDB and Neo4j"""
        # Setup mock responses
        mongodb_concepts = [
            {"_id": "concept1", "name": "Concept 1", "description": "Desc 1"},
            {"_id": "concept2", "name": "Concept 2", "description": "Desc 2"},
            {"_id": "concept3", "name": "Concept 3", "description": "Desc 3"}
        ]
        neo4j_concepts = [
            {"id": "concept1", "name": "Concept 1", "description": "Desc 1"},
            {"id": "concept2", "name": "Concept 2 Modified", "description": "Desc 2"},
            {"id": "concept4", "name": "Concept 4", "description": "Desc 4"}
        ]
        
        mock_inspector.inspect_mongodb.return_value = json.dumps(mongodb_concepts)
        mock_inspector.inspect_neo4j.return_value = [
            {"c": neo4j_concept} for neo4j_concept in neo4j_concepts
        ]
        
        # Call the method
        result = verifier.verify_concepts()
        
        # Verify the inspector was called with the right parameters
        mock_inspector.inspect_mongodb.assert_called_once()
        mock_inspector.inspect_neo4j.assert_called_once_with(
            query="MATCH (c:Concept) RETURN c"
        )
        
        # Verify the result
        assert result["total_mongodb_concepts"] == 3
        assert result["total_neo4j_concepts"] == 3
        assert result["inconsistency_count"] > 0
        
        # Check that inconsistencies were correctly identified
        inconsistencies = result["inconsistencies"]
        inconsistency_ids = [inc["id"] for inc in inconsistencies]
        
        # concept3 should be missing in Neo4j
        assert "concept3" in inconsistency_ids
        # concept4 should be missing in MongoDB
        assert "concept4" in inconsistency_ids
        # concept2 has a field mismatch
        concept2_issues = [inc for inc in inconsistencies if inc["id"] == "concept2"]
        assert concept2_issues
        assert concept2_issues[0]["error"] == "Field mismatch: name"

    def test_verify_relationships(self, verifier, mock_inspector):
        """Test relationship verification between MongoDB and Neo4j"""
        # Setup mock responses
        mongodb_concepts = [
            {
                "_id": "concept1",
                "relationships": [
                    {"target_id": "concept2", "type": "PREREQUISITE"},
                    {"target_id": "concept3", "type": "RELATED"}
                ]
            },
            {
                "_id": "concept2",
                "relationships": [
                    {"target_id": "concept3", "type": "PART_OF"}
                ]
            }
        ]
        
        neo4j_relationships = [
            {"source_id": "concept1", "type": "PREREQUISITE", "target_id": "concept2"},
            {"source_id": "concept1", "type": "RELATED", "target_id": "concept4"},  # Different from MongoDB
            {"source_id": "concept2", "type": "PART_OF", "target_id": "concept3"}
        ]
        
        mock_inspector.inspect_mongodb.return_value = json.dumps(mongodb_concepts)
        mock_inspector.inspect_neo4j.return_value = neo4j_relationships
        
        # Call the method
        result = verifier.verify_relationships()
        
        # Verify the inspector was called with the right parameters
        mock_inspector.inspect_mongodb.assert_called_once()
        mock_inspector.inspect_neo4j.assert_called_once()
        
        # Verify the result
        assert result["total_mongodb_relationships"] == 3
        assert result["total_neo4j_relationships"] == 3
        
        # Check that inconsistencies were correctly identified
        assert len(result["missing_in_neo4j"]) == 1
        assert "concept1-RELATED-concept3" in result["missing_in_neo4j"]
        
        assert len(result["missing_in_mongodb"]) == 1
        assert "concept1-RELATED-concept4" in result["missing_in_mongodb"]

    def test_verify_vector_embeddings(self, verifier, mock_inspector):
        """Test verification of vector embeddings in Qdrant"""
        # Setup mock responses
        mongodb_concepts = [
            {"_id": "concept1", "name": "Concept 1"},
            {"_id": "concept2", "name": "Concept 2"},
            {"_id": "concept3", "name": "Concept 3"}
        ]
        
        mock_point1 = MagicMock()
        mock_point1.id = "concept1"
        mock_point2 = MagicMock()
        mock_point2.id = "concept2"
        
        mock_inspector.inspect_mongodb.return_value = json.dumps(mongodb_concepts)
        mock_inspector.inspect_qdrant.return_value = [mock_point1, mock_point2]
        
        # Call the method
        result = verifier.verify_vector_embeddings()
        
        # Verify the inspector was called with the right parameters
        mock_inspector.inspect_mongodb.assert_called_once()
        mock_inspector.inspect_qdrant.assert_called_once()
        
        # Verify the result
        assert result["total_concepts"] == 3
        assert result["total_vectors"] == 2
        assert len(result["missing_vectors"]) == 1
        assert "concept3" in result["missing_vectors"]
        assert result["consistency_percentage"] == pytest.approx(66.67, rel=1e-2)