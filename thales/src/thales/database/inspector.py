import json
import yaml
from typing import Dict, List, Any, Optional, Union, Set

import pymongo
from neo4j import GraphDatabase
from qdrant_client import QdrantClient


class DatabaseInspector:
    """Tool for inspecting database state during and after integration tests."""

    def __init__(self, config_file: str = "config/inspector_config.yaml"):
        """Initialize the DatabaseInspector with configuration.
        
        Args:
            config_file: Path to the configuration file with database connection details
        """
        self.config = self._load_config(config_file)
        self.clients = self._initialize_clients()

    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load configuration from file.
        
        Args:
            config_file: Path to the configuration file
            
        Returns:
            Dictionary with configuration parameters
        """
        try:
            with open(config_file, "r") as f:
                if config_file.endswith(".yaml") or config_file.endswith(".yml"):
                    return yaml.safe_load(f)
                else:
                    return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, yaml.YAMLError) as e:
            raise ValueError(f"Error loading configuration file {config_file}: {str(e)}")

    def _initialize_clients(self) -> Dict[str, Any]:
        """Initialize database clients based on configuration.
        
        Returns:
            Dictionary with initialized database clients
        """
        clients = {}
        
        # Initialize MongoDB client if configured
        if "mongodb" in self.config:
            clients["mongodb"] = pymongo.MongoClient(self.config["mongodb"]["uri"])
            
        # Initialize Neo4j driver if configured
        if "neo4j" in self.config:
            clients["neo4j"] = GraphDatabase.driver(
                self.config["neo4j"]["uri"],
                auth=(self.config["neo4j"]["user"], self.config["neo4j"]["password"])
            )
            
        # Initialize Qdrant client if configured
        if "qdrant" in self.config:
            clients["qdrant"] = QdrantClient(url=self.config["qdrant"]["url"])
            
        return clients

    def inspect_mongodb(
        self, 
        database: str, 
        collection: str, 
        query: Optional[Dict[str, Any]] = None, 
        limit: int = 10
    ) -> str:
        """Inspects documents in MongoDB matching the given query.
        
        Args:
            database: MongoDB database name
            collection: MongoDB collection name
            query: Filter query to match documents
            limit: Maximum number of documents to return
            
        Returns:
            JSON string with query results
        """
        query = query or {}
        client = self.clients["mongodb"]
        
        try:
            results = list(
                client[database][collection].find(query).limit(limit)
            )
            return json.dumps(results, default=str, indent=2)
        except Exception as e:
            raise RuntimeError(f"Error querying MongoDB: {str(e)}")

    def inspect_neo4j(
        self, 
        query: str, 
        params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Runs a Cypher query against Neo4j and returns results.
        
        Args:
            query: Cypher query to execute
            params: Query parameters
            
        Returns:
            List of query results as dictionaries
        """
        params = params or {}
        client = self.clients["neo4j"]
        
        try:
            with client.session() as session:
                result = session.run(query, params)
                return [record.data() for record in result]
        except Exception as e:
            raise RuntimeError(f"Error querying Neo4j: {str(e)}")

    def inspect_qdrant(
        self, 
        collection: str, 
        ids: Optional[List[str]] = None, 
        limit: int = 10
    ) -> List[Any]:
        """Retrieves vectors from Qdrant collection.
        
        Args:
            collection: Qdrant collection name
            ids: Optional list of point IDs to retrieve
            limit: Maximum number of points to return when not specifying IDs
            
        Returns:
            List of points from Qdrant
        """
        client = self.clients["qdrant"]
        
        try:
            if ids:
                return client.retrieve(
                    collection_name=collection,
                    ids=ids
                )
            else:
                return client.scroll(
                    collection_name=collection,
                    limit=limit
                )[0]
        except Exception as e:
            raise RuntimeError(f"Error querying Qdrant: {str(e)}")

    def _get_data(self, spec: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get data from a specified data source.
        
        Args:
            spec: Specification of where to get data from
            
        Returns:
            List of data items
        """
        data_type = spec.get("type")
        
        if data_type == "mongodb":
            return json.loads(self.inspect_mongodb(
                database=spec.get("database", ""),
                collection=spec.get("collection", ""),
                query=spec.get("query", {}),
                limit=spec.get("limit", 1000)
            ))
        elif data_type == "neo4j":
            return self.inspect_neo4j(
                query=spec.get("query", ""),
                params=spec.get("params", {})
            )
        elif data_type == "qdrant":
            return self.inspect_qdrant(
                collection=spec.get("collection", ""),
                ids=spec.get("ids"),
                limit=spec.get("limit", 100)
            )
        else:
            raise ValueError(f"Unsupported data source type: {data_type}")

    def compare_data(
        self, 
        source_spec: Dict[str, Any], 
        target_spec: Dict[str, Any], 
        key_field: str = "id"
    ) -> Dict[str, Union[List[str], Set[str]]]:
        """Compares data between source and target data stores.
        
        Args:
            source_spec: Specification for source data
            target_spec: Specification for target data
            key_field: Field to use as key for comparison
            
        Returns:
            Dictionary with comparison results
        """
        # Get data from source and target
        source_data = self._get_data(source_spec)
        target_data = self._get_data(target_spec)
        
        # Organize by key field for comparison
        source_map = {str(item.get(key_field)): item for item in source_data if item.get(key_field) is not None}
        target_map = {str(item.get(key_field)): item for item in target_data if item.get(key_field) is not None}
        
        # Find missing and mismatched items
        missing_in_target = [k for k in source_map if k not in target_map]
        missing_in_source = [k for k in target_map if k not in source_map]
        common_keys = set(source_map.keys()) & set(target_map.keys())
        
        return {
            "missing_in_target": missing_in_target,
            "missing_in_source": missing_in_source,
            "common_keys": list(common_keys)
        }
