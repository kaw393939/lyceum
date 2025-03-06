import json
import os
import pytest
from unittest.mock import patch, MagicMock

from thales.generators.concept_generator import MockDataGenerator


class TestMockDataGenerator:
    @pytest.fixture
    def generator(self):
        return MockDataGenerator()

    @pytest.fixture
    def generator_with_mocks(self):
        mock_clients = {
            "mongodb": MagicMock(),
            "neo4j": MagicMock(),
            "qdrant": MagicMock()
        }
        return MockDataGenerator(db_clients=mock_clients), mock_clients

    def test_generate_mock_concepts(self, generator):
        """Test generating mock concepts"""
        # Test with default parameters
        concepts = generator.generate_mock_concepts(count=10)
        
        # Verify concepts were generated with the correct structure
        assert len(concepts) == 10
        for i, concept in enumerate(concepts):
            assert concept["id"] == f"concept_{i}"
            assert concept["name"] == f"Test Concept {i}"
            assert "description" in concept
            assert "relationships" in concept
    
    def test_generate_mock_concepts_with_output(self, generator, tmp_path):
        """Test generating mock concepts with file output"""
        # Create a temporary file path
        output_path = tmp_path / "test_concepts.json"
        
        # Generate concepts with output
        concepts = generator.generate_mock_concepts(count=5, output_path=str(output_path))
        
        # Verify file was created
        assert output_path.exists()
        
        # Verify file contains the correct data
        with open(output_path, 'r') as f:
            loaded_concepts = json.load(f)
        
        assert len(loaded_concepts) == 5
        assert loaded_concepts == concepts
        
    def test_generate_mock_relationships(self, generator):
        """Test generating mock relationships"""
        # Test with specific count
        relationships = generator._generate_mock_relationships(count=5)
        
        # Verify relationships were generated with the correct structure
        assert len(relationships) == 5
        for rel in relationships:
            assert "target_id" in rel
            assert "type" in rel
            assert "weight" in rel
            assert 0.1 <= rel["weight"] <= 1.0
    
    def test_populate_mongodb(self, generator_with_mocks):
        """Test populating MongoDB with mock concepts"""
        generator, mock_clients = generator_with_mocks
        mock_mongo = mock_clients["mongodb"]
        
        # Setup MongoDB mock
        mock_collection = MagicMock()
        mock_mongo.__getitem__.return_value.__getitem__.return_value = mock_collection
        mock_collection.insert_many.return_value.inserted_ids = ["id1", "id2", "id3"]
        
        # Generate mock concepts
        concepts = generator.generate_mock_concepts(count=3)
        
        # Populate MongoDB
        result = generator.populate_mongodb(concepts, database="test_db")
        
        # Verify MongoDB client was used correctly
        mock_mongo.__getitem__.assert_called_once_with("test_db")
        mock_mongo.__getitem__.return_value.__getitem__.assert_called_once_with("concepts")
        mock_collection.delete_many.assert_called_once_with({})
        mock_collection.insert_many.assert_called_once_with(concepts)
        
        # Verify the result
        assert result == 3
    
    def test_populate_neo4j(self, generator_with_mocks):
        """Test populating Neo4j with mock concepts"""
        generator, mock_clients = generator_with_mocks
        mock_neo4j = mock_clients["neo4j"]
        
        # Setup Neo4j mock
        mock_session = MagicMock()
        mock_neo4j.session.return_value.__enter__.return_value = mock_session
        
        # Generate mock concepts with relationships
        concepts = [
            {
                "id": "concept_1",
                "name": "Test Concept 1",
                "description": "Description 1",
                "relationships": [
                    {"target_id": "concept_2", "type": "PREREQUISITE", "weight": 0.8}
                ]
            },
            {
                "id": "concept_2",
                "name": "Test Concept 2",
                "description": "Description 2",
                "relationships": []
            }
        ]
        
        # Populate Neo4j
        result = generator.populate_neo4j(concepts)
        
        # Verify Neo4j session was used correctly
        mock_neo4j.session.assert_called()
        
        # Should call MATCH (n) DETACH DELETE n once
        assert mock_session.run.call_count >= 1
        mock_session.run.assert_any_call("MATCH (n) DETACH DELETE n")
        
        # Should create two concept nodes
        assert mock_session.run.call_count >= 3
        
        # Should create one relationship
        assert mock_session.run.call_count >= 4
        
        # Verify the result
        assert result == 2
    
    def test_populate_qdrant(self, generator_with_mocks):
        """Test populating Qdrant with mock vectors"""
        generator, mock_clients = generator_with_mocks
        mock_qdrant = mock_clients["qdrant"]
        
        # Generate mock concepts
        concepts = generator.generate_mock_concepts(count=5)
        
        # Populate Qdrant
        result = generator.populate_qdrant(concepts, collection_name="test_concepts")
        
        # Verify Qdrant client was used correctly
        mock_qdrant.delete_collection.assert_called_once_with("test_concepts")
        mock_qdrant.create_collection.assert_called_once()
        
        # Should upload points in batches
        assert mock_qdrant.upsert.call_count >= 1
        
        # Verify the result
        assert result == 5
    
    @patch('random.uniform')
    @patch('math.sqrt')
    def test_vector_generation(self, mock_sqrt, mock_uniform, generator):
        """Test generation of normalized vectors"""
        # Setup mocks
        mock_uniform.return_value = 0.5  # All vector components will be 0.5
        mock_sqrt.return_value = 1.0  # Simplify normalization
        
        # Generate a concept
        concept = {
            "id": "test_concept",
            "name": "Test"
        }
        
        # Create a point with a vector
        vector_size = 768
        vector = generator._generate_normalized_vector(vector_size)
        
        # Verify vector generation
        assert len(vector) == vector_size
        for component in vector:
            assert component == 0.5  # Because of our mocks
    
    def test_generate_and_populate_all(self, generator_with_mocks):
        """Test end-to-end data generation and population"""
        generator, mock_clients = generator_with_mocks
        
        # Setup mocks for each database
        mock_clients["mongodb"].__getitem__.return_value.__getitem__.return_value.insert_many.return_value.inserted_ids = ["id1", "id2"]
        
        # Mock helper methods
        with (
            patch.object(generator, 'populate_mongodb', return_value=2) as mock_mongo_populate,
            patch.object(generator, 'populate_neo4j', return_value=2) as mock_neo4j_populate,
            patch.object(generator, 'populate_qdrant', return_value=2) as mock_qdrant_populate,
        ):
            # Call the method
            result = generator.generate_and_populate_all(concept_count=2)
            
            # Verify the helper methods were called
            mock_mongo_populate.assert_called_once()
            mock_neo4j_populate.assert_called_once()
            mock_qdrant_populate.assert_called_once()
            
            # Verify the result
            assert result["concepts_generated"] == 2
            assert result["databases_populated"]["mongodb"] == 2
            assert result["databases_populated"]["neo4j"] == 2
            assert result["databases_populated"]["qdrant"] == 2