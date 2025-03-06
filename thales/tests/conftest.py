import os
import pytest
import json
from unittest.mock import MagicMock, patch

from thales.database.inspector import DatabaseInspector
from thales.database.consistency import ConsistencyVerifier
from thales.generators.concept_generator import MockDataGenerator


@pytest.fixture
def mock_mongodb_client():
    """Mock MongoDB client."""
    mock_client = MagicMock()
    return mock_client


@pytest.fixture
def mock_neo4j_driver():
    """Mock Neo4j driver."""
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session
    return mock_driver


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client."""
    mock_client = MagicMock()
    return mock_client


@pytest.fixture
def mock_config():
    """Sample config for testing."""
    return {
        "mongodb": {
            "uri": "mongodb://localhost:27017",
            "timeout_ms": 5000
        },
        "neo4j": {
            "uri": "bolt://localhost:7687",
            "user": "neo4j",
            "password": "password"
        },
        "qdrant": {
            "url": "http://localhost:6333"
        }
    }


@pytest.fixture
def mock_db_inspector(mock_mongodb_client, mock_neo4j_driver, mock_qdrant_client):
    """Create a DatabaseInspector with mocked clients."""
    with patch("thales.database.inspector.pymongo.MongoClient", return_value=mock_mongodb_client), \
         patch("thales.database.inspector.GraphDatabase.driver", return_value=mock_neo4j_driver), \
         patch("thales.database.inspector.QdrantClient", return_value=mock_qdrant_client), \
         patch("thales.database.inspector.DatabaseInspector._load_config", return_value={
                "mongodb": {"uri": "mock://uri"},
                "neo4j": {"uri": "mock://uri", "user": "mock", "password": "mock"},
                "qdrant": {"url": "mock://uri"}
         }):
        inspector = DatabaseInspector()
        return inspector


@pytest.fixture
def sample_concepts():
    """Sample concept data for testing."""
    return [
        {
            "id": "concept_1",
            "name": "Test Concept 1",
            "description": "Description for test concept 1",
            "relationships": [
                {"target_id": "concept_2", "type": "PREREQUISITE", "weight": 0.8},
                {"target_id": "concept_3", "type": "RELATED", "weight": 0.6}
            ]
        },
        {
            "id": "concept_2",
            "name": "Test Concept 2",
            "description": "Description for test concept 2",
            "relationships": [
                {"target_id": "concept_3", "type": "PART_OF", "weight": 0.9}
            ]
        },
        {
            "id": "concept_3",
            "name": "Test Concept 3",
            "description": "Description for test concept 3",
            "relationships": []
        }
    ]


@pytest.fixture
def mock_data_generator(mock_mongodb_client, mock_neo4j_driver, mock_qdrant_client, sample_concepts):
    """Create a MockDataGenerator with mocked clients and sample data."""
    db_clients = {
        "mongodb": mock_mongodb_client,
        "neo4j": mock_neo4j_driver,
        "qdrant": mock_qdrant_client
    }
    
    generator = MockDataGenerator(db_clients=db_clients)
    
    # Mock the generate_mock_concepts method
    generator.generate_mock_concepts = MagicMock(return_value=sample_concepts)
    
    return generator
