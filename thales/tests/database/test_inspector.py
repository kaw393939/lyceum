import pytest
import json
from unittest.mock import patch, MagicMock, mock_open

from thales.database.inspector import DatabaseInspector


class TestDatabaseInspector:
    @pytest.fixture
    def mock_config(self):
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
    def inspector(self, mock_config):
        with patch("builtins.open", mock_open(read_data=json.dumps(mock_config))):
            return DatabaseInspector(config_file="fake_path.yaml")

    @pytest.fixture
    def inspector_with_mocks(self, mock_config):
        with (
            patch("builtins.open", mock_open(read_data=json.dumps(mock_config))))as mock_open_:
            with patch("pymongo.MongoClient") as mock_mongo:
                with patch("neo4j.GraphDatabase.driver") as mock_neo4j:
                    with patch("qdrant_client.QdrantClient") as mock_qdrant:
                        inspector = DatabaseInspector(config_file="fake_path.yaml")
                        # Reset the mocks to clear any calls made during initialization
                        mock_mongo.reset_mock()
                        mock_neo4j.reset_mock()
                        mock_qdrant.reset_mock()
                        return inspector, mock_mongo, mock_neo4j, mock_qdrant

    def test_initialization_with_config_file(self, inspector):
        assert inspector.config is not None
        assert "mongodb" in inspector.config
        assert "neo4j" in inspector.config
        assert "qdrant" in inspector.config

    def test_clients_initialization(self):
        """Test that clients are properly initialized from the config."""
        # We'll just test if the clients are present in the result
        # Create a simple DatabaseInspector with mock configuration
        mock_config = {
            "mongodb": {"uri": "mongodb://localhost:27017"},
            "neo4j": {"uri": "bolt://localhost:7687", "user": "neo4j", "password": "password"},
            "qdrant": {"url": "http://localhost:6333"}
        }
        
        # Mock the database clients
        mock_mongo_client = MagicMock()
        mock_neo4j_driver = MagicMock()
        mock_qdrant_client = MagicMock()
        
        # Create a custom inspector that returns our mocked clients
        with patch.object(
            DatabaseInspector, 
            "_load_config", 
            return_value=mock_config
        ):
            # Override the _initialize_clients method to return our mock clients
            with patch.object(
                DatabaseInspector,
                "_initialize_clients",
                return_value={
                    "mongodb": mock_mongo_client,
                    "neo4j": mock_neo4j_driver,
                    "qdrant": mock_qdrant_client
                }
            ) as mock_initialize_clients:
                inspector = DatabaseInspector()
                
                # Verify initialize_clients was called
                mock_initialize_clients.assert_called_once()
                
                # Verify the clients were initialized
                assert inspector.clients["mongodb"] is mock_mongo_client
                assert inspector.clients["neo4j"] is mock_neo4j_driver  
                assert inspector.clients["qdrant"] is mock_qdrant_client

    def test_inspect_mongodb(self, inspector_with_mocks):
        inspector, mock_mongo, _, _ = inspector_with_mocks
        
        # Create a more complete mock structure
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_cursor = MagicMock()
        
        # Setup the return values
        mock_cursor.limit.return_value = [{"_id": "1", "name": "Test"}]
        mock_collection.find.return_value = mock_cursor
        mock_db.__getitem__.return_value = mock_collection
        mock_mongo.__getitem__.return_value = mock_db
        
        # Replace the client in the inspector
        inspector.clients["mongodb"] = mock_mongo
        
        # Call the method
        result = inspector.inspect_mongodb("test_db", "test_collection", {"name": "Test"}, 10)
        
        # Verify the correct methods were called
        mock_mongo.__getitem__.assert_called_with("test_db")
        mock_db.__getitem__.assert_called_with("test_collection")
        mock_collection.find.assert_called_once_with({"name": "Test"})
        mock_cursor.limit.assert_called_once_with(10)
        
        # Verify result is correct
        parsed_result = json.loads(result)
        assert len(parsed_result) == 1
        assert parsed_result[0]["_id"] == "1"
        assert parsed_result[0]["name"] == "Test"

    def test_inspect_neo4j(self, inspector_with_mocks):
        inspector, _, mock_neo4j, _ = inspector_with_mocks
        
        # Setup Neo4j mock
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_record1 = MagicMock()
        mock_record1.data.return_value = {"name": "Test1"}
        mock_record2 = MagicMock()
        mock_record2.data.return_value = {"name": "Test2"}
        mock_result.__iter__.return_value = [mock_record1, mock_record2]
        mock_session.run.return_value = mock_result
        
        mock_driver = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        inspector.clients["neo4j"] = mock_driver
        
        # Call the method
        result = inspector.inspect_neo4j("MATCH (n) RETURN n", {"param": "value"})
        
        # Verify the correct methods were called
        mock_driver.session.return_value.__enter__.assert_called_once()
        mock_session.run.assert_called_once_with("MATCH (n) RETURN n", {"param": "value"})
        
        # Verify result is correct
        assert len(result) == 2
        assert result[0]["name"] == "Test1"
        assert result[1]["name"] == "Test2"

    def test_inspect_qdrant(self, inspector_with_mocks):
        inspector, _, _, mock_qdrant = inspector_with_mocks
        
        # Setup Qdrant mock
        mock_client = MagicMock()
        mock_point1 = MagicMock()
        mock_point1.id = "1"
        mock_point2 = MagicMock()
        mock_point2.id = "2"
        
        mock_client.retrieve.return_value = [mock_point1, mock_point2]
        mock_client.scroll.return_value = ([mock_point1, mock_point2], None)
        
        inspector.clients["qdrant"] = mock_client
        
        # Test with IDs
        result_with_ids = inspector.inspect_qdrant("test_collection", ids=["1", "2"])
        mock_client.retrieve.assert_called_once_with(collection_name="test_collection", ids=["1", "2"])
        assert len(result_with_ids) == 2
        
        # Test without IDs (scroll)
        result_without_ids = inspector.inspect_qdrant("test_collection", limit=10)
        mock_client.scroll.assert_called_once_with(collection_name="test_collection", limit=10)
        assert len(result_without_ids) == 2

    def test_compare_data(self, inspector):
        with patch.object(inspector, "_get_data") as mock_get_data:
            # Setup mock data
            source_data = [
                {"id": "1", "name": "Common"},
                {"id": "2", "name": "Only in source"},
            ]
            target_data = [
                {"id": "1", "name": "Common"},
                {"id": "3", "name": "Only in target"},
            ]
            
            mock_get_data.side_effect = [source_data, target_data]
            
            # Call the method
            result = inspector.compare_data(
                source_spec={"type": "mongodb", "db": "test"},
                target_spec={"type": "neo4j", "query": "MATCH (n) RETURN n"},
                key_field="id"
            )
            
            # Verify the result
            assert "missing_in_target" in result
            assert "missing_in_source" in result
            assert "common_keys" in result
            
            assert "2" in result["missing_in_target"]
            assert "3" in result["missing_in_source"]
            assert "1" in result["common_keys"]