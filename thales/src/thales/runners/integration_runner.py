import os
import yaml
import json
import importlib
import logging
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime

import httpx

from thales.database.inspector import DatabaseInspector
from thales.database.consistency import ConsistencyVerifier
from thales.generators.concept_generator import MockDataGenerator


class IntegrationTestRunner:
    """Runs integration tests between specified services with DB verification."""
    
    def __init__(
        self, 
        config_path: str = "config/test_scenarios.yaml",
        db_config_path: str = "config/database_config.yaml",
        output_dir: str = "reports"
    ):
        """Initialize with configuration.
        
        Args:
            config_path: Path to test scenario configuration file
            db_config_path: Path to database configuration file
            output_dir: Directory to store test reports
        """
        self.config = self._load_config(config_path)
        self.db_config = self._load_config(db_config_path)
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up logging
        self.logger = self._setup_logger()
        
        # Initialize database inspector
        self.db_inspector = DatabaseInspector(config_file=db_config_path)
        
        # Initialize consistency verifier
        self.consistency_verifier = ConsistencyVerifier(self.db_inspector)
        
        # Initialize action registry - maps action names to handler methods
        self.action_registry = {
            "generate_concepts": self._handle_generate_concepts,
            "request": self._handle_request,
            "verify_db": self._handle_verify_db,
            "verify_consistency": self._handle_verify_consistency,
            "start_simulator": self._handle_start_simulator,
            "verify_logs": self._handle_verify_logs,
            "verify_response": self._handle_verify_response,
        }
        
        # Store test results
        self.results = {}
        
        # Store responses for later verification
        self.last_response = None
        
        # Store service simulators
        self.simulators = {}
    
    def _setup_logger(self) -> logging.Logger:
        """Set up a logger for the test runner.
        
        Returns:
            Configured logger instance
        """
        logger = logging.getLogger("integration_runner")
        logger.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(console_handler)
        
        # Create file handler
        log_file = os.path.join(
            self.output_dir, 
            f"integration_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(file_handler)
        
        return logger
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    
    def run(self, scenario_name: Optional[str] = None) -> Dict[str, Any]:
        """Run integration tests.
        
        Args:
            scenario_name: Optional name of scenario to run. If None, run all scenarios.
            
        Returns:
            Dictionary with test results
        """
        start_time = datetime.now()
        
        scenarios = self.config.get("scenarios", {})
        if not scenarios:
            self.logger.error("No scenarios defined in configuration")
            return {"status": "error", "message": "No scenarios defined"}
        
        # If a specific scenario is requested, filter the scenarios
        if scenario_name:
            if scenario_name not in scenarios:
                self.logger.error(f"Scenario '{scenario_name}' not found")
                return {"status": "error", "message": f"Scenario '{scenario_name}' not found"}
            
            # Create a new dictionary with only the requested scenario
            scenarios = {scenario_name: scenarios[scenario_name]}
        
        # Run each scenario
        self.results = {
            "scenarios": {},
            "start_time": start_time.isoformat(),
            "end_time": None,
            "status": "success"
        }
        
        for name, scenario in scenarios.items():
            self.logger.info(f"Running scenario: {name}")
            scenario_result = self._run_scenario(name, scenario)
            self.results["scenarios"][name] = scenario_result
            
            # If any scenario fails, mark the overall test as failed
            if scenario_result.get("status") == "error":
                self.results["status"] = "error"
        
        # Record end time
        end_time = datetime.now()
        self.results["end_time"] = end_time.isoformat()
        self.results["duration_seconds"] = (end_time - start_time).total_seconds()
        
        # Generate and save report
        self._generate_report()
        
        return self.results
    
    def _run_scenario(self, name: str, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single test scenario.
        
        Args:
            name: Name of the scenario
            scenario: Scenario configuration
            
        Returns:
            Dictionary with scenario results
        """
        start_time = datetime.now()
        steps = scenario.get("steps", [])
        
        result = {
            "name": name,
            "description": scenario.get("description", ""),
            "steps": [],
            "start_time": start_time.isoformat(),
            "end_time": None,
            "status": "success"
        }
        
        for step in steps:
            step_name = step.get("name", "Unnamed step")
            action = step.get("action")
            params = step.get("params", {})
            
            self.logger.info(f"Running step: {step_name}")
            
            if not action:
                error_message = f"No action specified for step: {step_name}"
                self.logger.error(error_message)
                step_result = {
                    "name": step_name,
                    "status": "error",
                    "message": error_message
                }
                result["steps"].append(step_result)
                result["status"] = "error"
                break
            
            if action not in self.action_registry:
                error_message = f"Unknown action '{action}' for step: {step_name}"
                self.logger.error(error_message)
                step_result = {
                    "name": step_name,
                    "action": action,
                    "status": "error",
                    "message": error_message
                }
                result["steps"].append(step_result)
                result["status"] = "error"
                break
            
            try:
                # Execute the action
                action_handler = self.action_registry[action]
                action_result = action_handler(params)
                
                step_result = {
                    "name": step_name,
                    "action": action,
                    "params": params,
                    "result": action_result,
                    "status": "success"
                }
                
                result["steps"].append(step_result)
                
                # If the action failed, mark the scenario as failed and stop execution
                if action_result.get("status") == "error":
                    result["status"] = "error"
                    break
                    
            except Exception as e:
                error_message = f"Error executing action '{action}': {str(e)}"
                self.logger.error(error_message, exc_info=True)
                step_result = {
                    "name": step_name,
                    "action": action,
                    "params": params,
                    "status": "error",
                    "message": error_message
                }
                result["steps"].append(step_result)
                result["status"] = "error"
                break
        
        # Record end time
        end_time = datetime.now()
        result["end_time"] = end_time.isoformat()
        result["duration_seconds"] = (end_time - start_time).total_seconds()
        
        return result
    
    def _generate_report(self) -> str:
        """Generate a test report and save it to a file.
        
        Returns:
            Path to the report file
        """
        report_file = os.path.join(
            self.output_dir, 
            f"integration_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(report_file, "w") as f:
            json.dump(self.results, f, indent=2)
        
        self.logger.info(f"Test report saved to: {report_file}")
        return report_file
    
    #
    # Action handlers
    #
    
    def _handle_generate_concepts(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle the 'generate_concepts' action.
        
        Args:
            params: Action parameters
            
        Returns:
            Action result
        """
        count = params.get("count", 10)
        output_path = params.get("output_path")
        populate_mongodb = params.get("populate_mongodb", False)
        populate_neo4j = params.get("populate_neo4j", False)
        populate_qdrant = params.get("populate_qdrant", False)
        
        self.logger.info(f"Generating {count} test concepts")
        
        # If we need to populate any database, create the necessary clients
        db_clients = {}
        
        if populate_mongodb:
            db_clients["mongodb"] = self.db_inspector.clients.get("mongodb")
            
        if populate_neo4j:
            db_clients["neo4j"] = self.db_inspector.clients.get("neo4j")
            
        if populate_qdrant:
            db_clients["qdrant"] = self.db_inspector.clients.get("qdrant")
        
        # Create generator and generate concepts
        generator = MockDataGenerator(db_clients=db_clients)
        
        # If we need to populate any database, use generate_and_populate_all
        if populate_mongodb or populate_neo4j or populate_qdrant:
            result = generator.generate_and_populate_all(
                concept_count=count,
                output_path=output_path
            )
            return {
                "status": "success",
                "message": f"Generated and populated {count} concepts",
                "concepts_generated": result["concepts_generated"],
                "databases_populated": result["databases_populated"]
            }
        else:
            # Otherwise, just generate concepts
            concepts = generator.generate_mock_concepts(
                count=count,
                output_path=output_path
            )
            return {
                "status": "success",
                "message": f"Generated {len(concepts)} concepts",
                "concepts_generated": len(concepts)
            }
    
    def _handle_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle the 'request' action, making HTTP requests to services.
        
        Args:
            params: Action parameters
            
        Returns:
            Action result
        """
        service = params.get("service")
        endpoint = params.get("endpoint")
        method = params.get("method", "GET")
        body = params.get("body")
        headers = params.get("headers", {})
        expect_success = params.get("expect_success", True)
        
        if not service:
            return {
                "status": "error",
                "message": "No service specified for request"
            }
            
        if not endpoint:
            return {
                "status": "error",
                "message": "No endpoint specified for request"
            }
        
        # Get service URL from environment variables or use default
        service_urls = {
            "ptolemy": os.environ.get("PTOLEMY_URL", "http://localhost:8000"),
            "gutenberg": os.environ.get("GUTENBERG_URL", "http://localhost:8001"),
            "galileo": os.environ.get("GALILEO_URL", "http://localhost:8502"),
            "socrates": os.environ.get("SOCRATES_URL", "http://localhost:8501")
        }
        
        if service not in service_urls:
            return {
                "status": "error",
                "message": f"Unknown service: {service}"
            }
            
        url = f"{service_urls[service]}{endpoint}"
        
        self.logger.info(f"Making {method} request to {url}")
        
        # Make the request
        try:
            with httpx.Client() as client:
                if method == "GET":
                    response = client.get(url, headers=headers)
                elif method == "POST":
                    response = client.post(url, json=body, headers=headers)
                elif method == "PUT":
                    response = client.put(url, json=body, headers=headers)
                elif method == "DELETE":
                    response = client.delete(url, headers=headers)
                else:
                    return {
                        "status": "error",
                        "message": f"Unsupported HTTP method: {method}"
                    }
                
                # Store the response for later verification
                self.last_response = {
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "body": response.json() if response.headers.get("content-type") == "application/json" else response.text
                }
                
                # Check if the response was successful
                is_success = 200 <= response.status_code < 300
                
                if is_success != expect_success:
                    status = "error"
                    if expect_success:
                        message = f"Expected success but got status code {response.status_code}"
                    else:
                        message = f"Expected failure but got status code {response.status_code}"
                else:
                    status = "success"
                    message = f"Request completed with status code {response.status_code}"
                
                return {
                    "status": status,
                    "message": message,
                    "response": self.last_response
                }
                
        except Exception as e:
            self.logger.error(f"Error making request: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": f"Error making request: {str(e)}"
            }
    
    def _handle_verify_db(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle the 'verify_db' action, checking database state.
        
        Args:
            params: Action parameters
            
        Returns:
            Action result
        """
        verifications = params.get("verifications", [])
        
        if not verifications:
            return {
                "status": "error",
                "message": "No verifications specified"
            }
            
        results = []
        all_passed = True
        
        for verification in verifications:
            db_type = verification.get("type")
            
            if db_type == "mongodb":
                result = self._verify_mongodb(verification)
            elif db_type == "neo4j":
                result = self._verify_neo4j(verification)
            elif db_type == "qdrant":
                result = self._verify_qdrant(verification)
            else:
                result = {
                    "status": "error",
                    "message": f"Unknown database type: {db_type}"
                }
                
            results.append(result)
            
            if result.get("status") == "error":
                all_passed = False
        
        return {
            "status": "success" if all_passed else "error",
            "message": "All verifications passed" if all_passed else "Some verifications failed",
            "results": results
        }
    
    def _verify_mongodb(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Verify MongoDB state.
        
        Args:
            params: Verification parameters
            
        Returns:
            Verification result
        """
        database = params.get("database")
        collection = params.get("collection")
        query = params.get("query", {})
        expected_count = params.get("expected_count")
        expected_fields = params.get("expected_fields", [])
        
        if not database or not collection:
            return {
                "status": "error",
                "message": "Database and collection required for MongoDB verification"
            }
            
        try:
            result_json = self.db_inspector.inspect_mongodb(
                database=database,
                collection=collection,
                query=query
            )
            results = json.loads(result_json)
            
            # Check count if expected_count is specified
            if expected_count is not None:
                actual_count = len(results)
                if actual_count != expected_count:
                    return {
                        "status": "error",
                        "message": f"Expected {expected_count} documents, found {actual_count}",
                        "query": query,
                        "results": results
                    }
                    
            # Check fields if expected_fields is specified
            if expected_fields and results:
                for result in results:
                    for field in expected_fields:
                        if field not in result:
                            return {
                                "status": "error",
                                "message": f"Expected field '{field}' not found in result",
                                "query": query,
                                "result": result
                            }
            
            return {
                "status": "success",
                "message": f"Found {len(results)} documents matching query",
                "query": query,
                "results": results
            }
            
        except Exception as e:
            self.logger.error(f"Error verifying MongoDB: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": f"Error verifying MongoDB: {str(e)}"
            }
    
    def _verify_neo4j(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Verify Neo4j state.
        
        Args:
            params: Verification parameters
            
        Returns:
            Verification result
        """
        query = params.get("query")
        params_dict = params.get("params", {})
        expected_count = params.get("expected_count")
        
        if not query:
            return {
                "status": "error",
                "message": "Query required for Neo4j verification"
            }
            
        try:
            results = self.db_inspector.inspect_neo4j(
                query=query,
                params=params_dict
            )
            
            # Check count if expected_count is specified
            if expected_count is not None:
                actual_count = len(results)
                if actual_count != expected_count:
                    return {
                        "status": "error",
                        "message": f"Expected {expected_count} records, found {actual_count}",
                        "query": query,
                        "results": results
                    }
            
            return {
                "status": "success",
                "message": f"Found {len(results)} records matching query",
                "query": query,
                "results": results
            }
            
        except Exception as e:
            self.logger.error(f"Error verifying Neo4j: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": f"Error verifying Neo4j: {str(e)}"
            }
    
    def _verify_qdrant(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Verify Qdrant state.
        
        Args:
            params: Verification parameters
            
        Returns:
            Verification result
        """
        collection = params.get("collection")
        ids = params.get("ids")
        expected_count = params.get("expected_count")
        
        if not collection:
            return {
                "status": "error",
                "message": "Collection required for Qdrant verification"
            }
            
        try:
            results = self.db_inspector.inspect_qdrant(
                collection=collection,
                ids=ids
            )
            
            # Check count if expected_count is specified
            if expected_count is not None:
                actual_count = len(results)
                if actual_count != expected_count:
                    return {
                        "status": "error",
                        "message": f"Expected {expected_count} points, found {actual_count}",
                        "collection": collection,
                        "ids": ids
                    }
            
            return {
                "status": "success",
                "message": f"Found {len(results)} points in collection",
                "collection": collection,
                "ids": ids
            }
            
        except Exception as e:
            self.logger.error(f"Error verifying Qdrant: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": f"Error verifying Qdrant: {str(e)}"
            }
    
    def _handle_verify_consistency(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle the 'verify_consistency' action.
        
        Args:
            params: Action parameters
            
        Returns:
            Action result
        """
        source = params.get("source")
        target = params.get("target")
        entity_type = params.get("entity_type")
        
        if not source or not target:
            return {
                "status": "error",
                "message": "Source and target required for consistency verification"
            }
            
        if not entity_type:
            return {
                "status": "error",
                "message": "Entity type required for consistency verification"
            }
            
        try:
            if entity_type == "concepts":
                result = self.consistency_verifier.verify_concepts(
                    source_db=source,
                    target_db=target
                )
                
                if result["inconsistency_count"] > 0:
                    return {
                        "status": "error",
                        "message": f"Found {result['inconsistency_count']} inconsistent concepts",
                        "result": result
                    }
                else:
                    return {
                        "status": "success",
                        "message": "All concepts are consistent",
                        "result": result
                    }
                    
            elif entity_type == "relationships":
                result = self.consistency_verifier.verify_relationships()
                
                if len(result["missing_in_neo4j"]) > 0 or len(result["missing_in_mongodb"]) > 0:
                    return {
                        "status": "error",
                        "message": f"Found {len(result['missing_in_neo4j'])} relationships missing in Neo4j and {len(result['missing_in_mongodb'])} missing in MongoDB",
                        "result": result
                    }
                else:
                    return {
                        "status": "success",
                        "message": "All relationships are consistent",
                        "result": result
                    }
                    
            elif entity_type == "embeddings":
                result = self.consistency_verifier.verify_vector_embeddings()
                
                if len(result["missing_vectors"]) > 0:
                    return {
                        "status": "error",
                        "message": f"Found {len(result['missing_vectors'])} concepts without vector embeddings",
                        "result": result
                    }
                else:
                    return {
                        "status": "success",
                        "message": "All concepts have vector embeddings",
                        "result": result
                    }
            else:
                return {
                    "status": "error",
                    "message": f"Unknown entity type: {entity_type}"
                }
                
        except Exception as e:
            self.logger.error(f"Error verifying consistency: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": f"Error verifying consistency: {str(e)}"
            }
    
    def _handle_start_simulator(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle the 'start_simulator' action.
        
        Args:
            params: Action parameters
            
        Returns:
            Action result
        """
        service = params.get("service")
        port = params.get("port")
        error_rate = params.get("error_rate", 0.0)
        delay_ms = params.get("delay_ms", 0)
        
        if not service:
            return {
                "status": "error",
                "message": "Service name required for simulator"
            }
            
        if not port:
            return {
                "status": "error",
                "message": "Port required for simulator"
            }
            
        # Import here to avoid circular imports
        try:
            from thales.simulators.service_simulator import ServiceSimulator
            
            # Create simulator
            simulator = ServiceSimulator(
                service_name=service,
                port=port,
                error_rate=error_rate,
                delay_ms=delay_ms
            )
            
            # Store simulator for later use
            self.simulators[service] = simulator
            
            # Start simulator in a separate thread
            import threading
            import uvicorn
            
            # Define a function to run the simulator
            def run_simulator():
                uvicorn.run(simulator.app, host="0.0.0.0", port=port)
                
            # Start the simulator in a separate thread
            thread = threading.Thread(target=run_simulator)
            thread.daemon = True
            thread.start()
            
            return {
                "status": "success",
                "message": f"Started {service} simulator on port {port}",
                "service": service,
                "port": port,
                "error_rate": error_rate,
                "delay_ms": delay_ms
            }
            
        except ImportError:
            return {
                "status": "error",
                "message": "Service simulator not available"
            }
        except Exception as e:
            self.logger.error(f"Error starting simulator: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": f"Error starting simulator: {str(e)}"
            }
    
    def _handle_verify_logs(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle the 'verify_logs' action.
        
        Args:
            params: Action parameters
            
        Returns:
            Action result
        """
        service = params.get("service")
        should_contain = params.get("should_contain", [])
        should_not_contain = params.get("should_not_contain", [])
        days = params.get("days", 1)
        
        if not service:
            return {
                "status": "error",
                "message": "Service name required for log verification"
            }
            
        try:
            # Import here to avoid circular imports
            from thales.diagnostics.error_collector import ErrorCollector
            
            collector = ErrorCollector()
            logs = collector.collect_service_logs(service, days=days)
            
            # Check if logs contain required patterns
            missing_patterns = []
            for pattern in should_contain:
                found = False
                for log in logs:
                    if pattern in log.get("message", ""):
                        found = True
                        break
                        
                if not found:
                    missing_patterns.append(pattern)
                    
            # Check if logs do not contain prohibited patterns
            found_prohibited = []
            for pattern in should_not_contain:
                for log in logs:
                    if pattern in log.get("message", ""):
                        found_prohibited.append(pattern)
                        break
            
            if missing_patterns or found_prohibited:
                message_parts = []
                if missing_patterns:
                    message_parts.append(f"Missing patterns: {', '.join(missing_patterns)}")
                if found_prohibited:
                    message_parts.append(f"Found prohibited patterns: {', '.join(found_prohibited)}")
                    
                return {
                    "status": "error",
                    "message": "; ".join(message_parts),
                    "missing_patterns": missing_patterns,
                    "found_prohibited": found_prohibited,
                    "logs": logs
                }
            else:
                return {
                    "status": "success",
                    "message": "Logs contain all required patterns and no prohibited patterns",
                    "logs": logs
                }
                
        except Exception as e:
            self.logger.error(f"Error verifying logs: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": f"Error verifying logs: {str(e)}"
            }
    
    def _handle_verify_response(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle the 'verify_response' action, checking the last response.
        
        Args:
            params: Action parameters
            
        Returns:
            Action result
        """
        expected_status = params.get("expected_status", 200)
        expected_fields = params.get("expected_fields", [])
        expected_count = params.get("expected_count")
        source = params.get("source")
        query = params.get("query")
        response_field = params.get("response_field")
        
        if not self.last_response:
            return {
                "status": "error",
                "message": "No response to verify"
            }
            
        # Check status code
        if self.last_response["status_code"] != expected_status:
            return {
                "status": "error",
                "message": f"Expected status code {expected_status}, got {self.last_response['status_code']}"
            }
            
        # If response is not JSON, we can't verify fields
        if not isinstance(self.last_response["body"], dict):
            if expected_fields or response_field:
                return {
                    "status": "error",
                    "message": "Response is not JSON, can't verify fields"
                }
            else:
                return {
                    "status": "success",
                    "message": "Response status code matches expected value"
                }
                
        # Check expected fields
        if expected_fields:
            missing_fields = []
            for field in expected_fields:
                if field not in self.last_response["body"]:
                    missing_fields.append(field)
                    
            if missing_fields:
                return {
                    "status": "error",
                    "message": f"Missing fields in response: {', '.join(missing_fields)}"
                }
                
        # Check expected count if applicable
        if expected_count is not None:
            # If response_field is specified, check count of that field
            if response_field:
                if response_field not in self.last_response["body"]:
                    return {
                        "status": "error",
                        "message": f"Field '{response_field}' not found in response"
                    }
                    
                actual_count = len(self.last_response["body"][response_field])
                if actual_count != expected_count:
                    return {
                        "status": "error",
                        "message": f"Expected {expected_count} items in '{response_field}', found {actual_count}"
                    }
            else:
                # Otherwise, check count of top-level items
                actual_count = len(self.last_response["body"])
                if actual_count != expected_count:
                    return {
                        "status": "error",
                        "message": f"Expected {expected_count} items in response, found {actual_count}"
                    }
                    
        # Check response against database if source and query are specified
        if source and query:
            if source == "neo4j":
                db_results = self.db_inspector.inspect_neo4j(query=query)
            elif source == "mongodb":
                db_results = json.loads(self.db_inspector.inspect_mongodb(
                    database=params.get("database", ""),
                    collection=params.get("collection", ""),
                    query=params.get("mongodb_query", {})
                ))
            else:
                return {
                    "status": "error",
                    "message": f"Unsupported source: {source}"
                }
                
            # Get response data to compare
            response_data = self.last_response["body"]
            if response_field:
                if response_field not in response_data:
                    return {
                        "status": "error",
                        "message": f"Field '{response_field}' not found in response"
                    }
                response_data = response_data[response_field]
                
            # Now we need to compare response_data with db_results
            # This depends on the specific format of both, so we'll just
            # check that counts match for now
            if len(response_data) != len(db_results):
                return {
                    "status": "error",
                    "message": f"Response data count ({len(response_data)}) doesn't match database results count ({len(db_results)})"
                }
                
        return {
            "status": "success",
            "message": "Response verification passed"
        }