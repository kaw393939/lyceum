import pytest
from unittest.mock import patch, MagicMock
from click.testing import CliRunner

from thales.cli import main, verify_cmd, generate_cmd, inspect_cmd


class TestCLI:
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    def test_main_help(self, runner):
        """Test that the main CLI shows help message"""
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Usage:" in result.output
        assert "Options:" in result.output
        assert "Commands:" in result.output
    
    def test_verify_command(self, runner):
        """Test the verify command"""
        with patch("thales.cli.ConsistencyVerifier") as mock_verifier_class:
            mock_verifier = MagicMock()
            mock_verifier_class.return_value = mock_verifier
            mock_verifier.verify_concepts.return_value = {
                "total_mongodb_concepts": 10,
                "total_neo4j_concepts": 9,
                "inconsistency_count": 2,
                "inconsistencies": [
                    {"id": "concept1", "error": "Missing in Neo4j"},
                    {"id": "concept2", "error": "Field mismatch: name"}
                ]
            }
            
            # Run the command
            result = runner.invoke(verify_cmd, ["--source", "mongodb", "--target", "neo4j"])
            
            # Verify the command executed successfully
            assert result.exit_code == 0
            mock_verifier.verify_concepts.assert_called_once()
            assert "Concepts in MongoDB: 10" in result.output
            assert "Concepts in Neo4j: 9" in result.output
            assert "Inconsistencies: 2" in result.output
    
    def test_generate_command(self, runner):
        """Test the generate command"""
        with patch("thales.cli.MockDataGenerator") as mock_generator_class:
            mock_generator = MagicMock()
            mock_generator_class.return_value = mock_generator
            mock_generator.generate_mock_concepts.return_value = [{"id": "concept_1"}]
            mock_generator.generate_and_populate_all.return_value = {
                "concepts_generated": 5,
                "databases_populated": {
                    "mongodb": 5,
                    "neo4j": 5,
                    "qdrant": 5
                }
            }
            
            # Run the command to just generate concepts
            result = runner.invoke(generate_cmd, ["--concepts", "5"])
            
            # Verify the command executed successfully
            assert result.exit_code == 0
            mock_generator.generate_mock_concepts.assert_called_once_with(count=5, output_path=None)
            assert "Generated 1 concepts" in result.output
            
            # Reset mock
            mock_generator.generate_mock_concepts.reset_mock()
            
            # Run the command to generate and populate
            result = runner.invoke(generate_cmd, ["--concepts", "5", "--populate-all"])
            
            # Verify the command executed successfully
            assert result.exit_code == 0
            mock_generator.generate_and_populate_all.assert_called_once_with(concept_count=5, output_path=None)
            assert "Generated 5 concepts" in result.output
            assert "Populated MongoDB with 5 concepts" in result.output
            assert "Populated Neo4j with 5 concepts" in result.output
            assert "Populated Qdrant with 5 vectors" in result.output
    
    def test_inspect_command(self, runner):
        """Test the inspect command"""
        with patch("thales.cli.DatabaseInspector") as mock_inspector_class:
            mock_inspector = MagicMock()
            mock_inspector_class.return_value = mock_inspector
            mock_inspector.inspect_mongodb.return_value = '{"_id": "test", "name": "Test"}'
            
            # Run the command
            result = runner.invoke(inspect_cmd, ["mongodb", "test_db", "test_collection"])
            
            # Verify the command executed successfully
            assert result.exit_code == 0
            mock_inspector.inspect_mongodb.assert_called_once_with(
                database="test_db",
                collection="test_collection",
                query=None,
                limit=10
            )
            assert '{"_id": "test", "name": "Test"}' in result.output