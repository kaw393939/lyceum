#!/usr/bin/env python3
"""
Gutenberg System Testing and Debugging Tool
==========================================
A comprehensive tool for testing and debugging the Gutenberg content generation system.
Performs system-wide validation, integration testing, and identifies issues.
"""

import os
import sys
import json
import time
import uuid
import logging
import asyncio
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set
import traceback

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import Gutenberg components
try:
    from config.settings import get_config
    from integrations.ptolemy_client import PtolemyClient
    from integrations.mongodb_service import MongoDBService
    from integrations.llm_service import LLMService
    from integrations.vector_store import VectorStore
    from core.content_generator import ContentGenerator
    from core.rag_processor import RAGProcessor
    from core.template_engine import TemplateEngine
    from models.content import ContentRequest, ContentType, ContentDifficulty
    from utils.logging_utils import get_logger, configure_logging, create_log_analyzer
except ImportError as e:
    print(f"Warning: Could not import all required modules: {e}")
    print("Some functionality may be limited.")

# Configure logging
log_file = "gutenberg_tests.log"
configure_logging(level="DEBUG", log_file=log_file, console=True)
logger = get_logger("gutenberg_test")

# Define test status constants
TEST_PASS = "✅ PASS"
TEST_FAIL = "❌ FAIL"
TEST_WARN = "⚠️ WARN"
TEST_INFO = "ℹ️ INFO"

class GutenbergTester:
    """Comprehensive testing and debugging for the Gutenberg system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the tester.
        
        Args:
            config_path: Optional path to config file
        """
        self.start_time = datetime.now()
        self.test_results = []
        self.issues_found = []
        self.warnings = []
        self.recommendations = []
        
        # Load config
        if config_path:
            # This would need to be implemented in settings.py
            # self.config = load_config(config_path)
            self.config = get_config()
        else:
            self.config = get_config()
        
        logger.info(f"Gutenberg test tool initialized with config")
        
        # Initialize test data
        self.test_data = {
            "concept_ids": [
                "stoicism", "dichotomy_of_control", "virtue_ethics", 
                "stoic_virtues", "negative_visualization"
            ],
            "path_ids": ["stoicism_intro", "stoic_practices"],
            "generated_content_ids": set(),
            "template_ids": set(),
            "request_ids": set()
        }
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """
        Run all tests and generate a comprehensive report.
        
        Returns:
            Test results report
        """
        logger.info("Starting comprehensive system tests")
        
        try:
            # System connection tests
            await self.test_system_connections()
            
            # Individual component tests
            await self.test_ptolemy_client()
            await self.test_mongodb_service()
            await self.test_llm_service()
            await self.test_vector_store()
            
            # Integrated functionality tests
            await self.test_template_engine()
            await self.test_rag_processor()
            await self.test_content_generator()
            
            # API endpoint tests (would require running server)
            # await self.test_api_endpoints()
            
            # Generate report
            report = self.generate_report()
            
            # Analyze logs
            log_analysis = await self.analyze_logs()
            report["log_analysis"] = log_analysis
            
            # Save report
            self.save_report(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Error during tests: {str(e)}")
            logger.debug(traceback.format_exc())
            
            # Generate partial report with error
            report = self.generate_report()
            report["error"] = str(e)
            report["traceback"] = traceback.format_exc()
            
            # Save report even if tests didn't complete
            self.save_report(report)
            
            return report
    
    async def test_system_connections(self):
        """Test all system connections."""
        self.log_test_section("System Connection Tests")
        
        # Test MongoDB connection
        mongodb = MongoDBService()
        mongo_result = await mongodb.check_connection()
        self.log_test_result(
            "MongoDB Connection", 
            mongo_result, 
            "Connected to MongoDB" if mongo_result else "Failed to connect to MongoDB",
            failure_recommendation="Check MongoDB configuration and ensure the service is running"
        )
        
        # Vector store connection check
        try:
            vector_store = VectorStore()
            vector_result = await vector_store.health_check()
            self.log_test_result(
                "Vector Store Connection", 
                vector_result, 
                "Connected to vector store" if vector_result else "Failed to connect to vector store",
                failure_recommendation="Check vector store configuration and ensure the service is running"
            )
        except Exception as e:
            self.log_test_result(
                "Vector Store Connection", 
                False, 
                f"Error connecting to vector store: {str(e)}",
                failure_recommendation="Check vector store configuration and ensure the service is running"
            )
        
        # LLM service connection check
        try:
            llm_service = LLMService()
            llm_result = await llm_service.check_connection()
            self.log_test_result(
                "LLM Service Connection", 
                llm_result, 
                "Connected to LLM service" if llm_result else "Failed to connect to LLM service",
                failure_recommendation="Check LLM service API keys and configuration"
            )
        except Exception as e:
            self.log_test_result(
                "LLM Service Connection", 
                False, 
                f"Error connecting to LLM service: {str(e)}",
                failure_recommendation="Check LLM service API keys and configuration"
            )
    
    async def test_ptolemy_client(self):
        """Test Ptolemy client functionality."""
        self.log_test_section("Ptolemy Client Tests")
        
        ptolemy = PtolemyClient()
        
        # Test getting a concept
        concept_id = self.test_data["concept_ids"][0]
        concept = await ptolemy.get_concept(concept_id)
        self.log_test_result(
            "Get Concept", 
            concept is not None, 
            f"Retrieved concept: {concept_id}" if concept else f"Failed to retrieve concept: {concept_id}",
            failure_recommendation="Check Ptolemy connection and ensure concept exists"
        )
        
        # Check concept data structure
        if concept:
            concept_valid = self._validate_concept_structure(concept)
            self.log_test_result(
                "Concept Data Structure", 
                concept_valid, 
                "Concept has valid structure" if concept_valid else "Concept has invalid structure",
                failure_recommendation="Review Ptolemy concept schema"
            )
        
        # Test getting concept relationships
        relationships = await ptolemy.get_concept_relationships(concept_id)
        self.log_test_result(
            "Get Concept Relationships", 
            relationships is not None and len(relationships) > 0, 
            f"Retrieved {len(relationships)} relationships" if relationships else "Failed to retrieve relationships",
            failure_recommendation="Check Ptolemy relationship data"
        )
        
        # Test getting concept graph
        graph = await ptolemy.get_concept_graph(concept_id, depth=1)
        graph_valid = graph and "nodes" in graph and "edges" in graph
        self.log_test_result(
            "Get Concept Graph", 
            graph_valid, 
            f"Retrieved graph with {len(graph.get('nodes', []))} nodes and {len(graph.get('edges', []))} edges" if graph_valid else "Failed to retrieve concept graph",
            failure_recommendation="Check Ptolemy graph functionality"
        )
        
        # Test getting learning path
        path_id = self.test_data["path_ids"][0] if self.test_data["path_ids"] else None
        if path_id:
            path = await ptolemy.get_learning_path(path_id)
            self.log_test_result(
                "Get Learning Path", 
                path is not None, 
                f"Retrieved learning path: {path_id}" if path else f"Failed to retrieve learning path: {path_id}",
                failure_recommendation="Check Ptolemy learning path data"
            )
    
    async def test_mongodb_service(self):
        """Test MongoDB service functionality."""
        self.log_test_section("MongoDB Service Tests")
        
        mongodb = MongoDBService()
        
        # Test creating and retrieving content
        test_content = {
            "content_id": f"test_{uuid.uuid4()}",
            "title": "Test Content",
            "content": "This is test content for MongoDB service validation.",
            "metadata": {
                "test": True,
                "content_type": "test"
            }
        }
        
        try:
            # Create content
            content_id = await mongodb.create_content(test_content)
            self.test_data["generated_content_ids"].add(content_id)
            
            self.log_test_result(
                "Create Content", 
                content_id is not None, 
                f"Created content with ID: {content_id}" if content_id else "Failed to create content",
                failure_recommendation="Check MongoDB write permissions"
            )
            
            # Retrieve content
            retrieved_content = await mongodb.get_content(content_id)
            content_match = retrieved_content and retrieved_content.get("title") == test_content["title"]
            
            self.log_test_result(
                "Retrieve Content", 
                content_match, 
                "Retrieved content matches original" if content_match else "Retrieved content does not match original",
                failure_recommendation="Check MongoDB read consistency"
            )
            
            # Update content
            update_result = await mongodb.update_content(content_id, {"title": "Updated Test Content"})
            update_success = update_result and update_result.get("title") == "Updated Test Content"
            
            self.log_test_result(
                "Update Content", 
                update_success, 
                "Updated content successfully" if update_success else "Failed to update content",
                failure_recommendation="Check MongoDB update functionality"
            )
            
            # Clean up test data
            delete_result = await mongodb.delete_content(content_id)
            self.log_test_result(
                "Delete Content", 
                delete_result, 
                "Deleted content successfully" if delete_result else "Failed to delete content",
                test_type=TEST_INFO  # Informational only, not critical
            )
            
        except Exception as e:
            self.log_test_result(
                "MongoDB Content Operations", 
                False, 
                f"Error during MongoDB content operations: {str(e)}",
                failure_recommendation="Check MongoDB service configuration and permissions"
            )
        
        # Test template operations
        test_template = {
            "template_id": f"test_template_{uuid.uuid4()}",
            "name": "Test Template",
            "description": "Test template for MongoDB service validation",
            "template_type": "test",
            "sections": [{"id": "section1", "name": "Test Section", "content": "Test content"}]
        }
        
        try:
            # Create template
            template_id = await mongodb.create_template(test_template)
            self.test_data["template_ids"].add(template_id)
            
            self.log_test_result(
                "Create Template", 
                template_id is not None, 
                f"Created template with ID: {template_id}" if template_id else "Failed to create template",
                failure_recommendation="Check MongoDB template collection configuration"
            )
            
            # Retrieve template
            retrieved_template = await mongodb.get_template(template_id)
            template_match = retrieved_template and retrieved_template.get("name") == test_template["name"]
            
            self.log_test_result(
                "Retrieve Template", 
                template_match, 
                "Retrieved template matches original" if template_match else "Retrieved template does not match original",
                failure_recommendation="Check MongoDB template retrieval functionality"
            )
            
            # List templates
            templates, count = await mongodb.list_templates(limit=10)
            self.log_test_result(
                "List Templates", 
                templates is not None, 
                f"Listed {len(templates)} templates" if templates else "Failed to list templates",
                test_type=TEST_INFO
            )
            
            # Clean up test data
            await mongodb.delete_template(template_id)
            
        except Exception as e:
            self.log_test_result(
                "MongoDB Template Operations", 
                False, 
                f"Error during MongoDB template operations: {str(e)}",
                failure_recommendation="Check MongoDB template collection configuration"
            )
    
    async def test_llm_service(self):
        """Test LLM service functionality."""
        self.log_test_section("LLM Service Tests")
        
        llm_service = LLMService()
        
        # Test text generation
        test_prompt = "Explain the concept of Stoicism in one paragraph."
        system_message = "You are a helpful assistant explaining philosophical concepts."
        
        try:
            response = await llm_service.generate_content(
                prompt=test_prompt,
                system_message=system_message
            )
            
            # Check response
            generation_success = response and response.content and len(response.content) > 10
            self.log_test_result(
                "LLM Text Generation", 
                generation_success, 
                f"Generated content of length {len(response.content) if response and response.content else 0}" if generation_success else "Failed to generate content",
                failure_recommendation="Check LLM service API key and rate limits"
            )
            
            # Check for expected content (basic check)
            if generation_success:
                content_relevance = "stoic" in response.content.lower() or "stoicism" in response.content.lower()
                self.log_test_result(
                    "LLM Content Relevance", 
                    content_relevance, 
                    "Generated content is relevant to the prompt" if content_relevance else "Generated content may not be relevant to the prompt",
                    test_type=TEST_WARN if not content_relevance else TEST_PASS
                )
            
        except Exception as e:
            self.log_test_result(
                "LLM Service Operations", 
                False, 
                f"Error during LLM service operations: {str(e)}",
                failure_recommendation="Check LLM service configuration and API key"
            )
    
    async def test_vector_store(self):
        """Test vector store functionality."""
        self.log_test_section("Vector Store Tests")
        
        vector_store = VectorStore()
        
        # Test embedding
        test_text = "Stoicism is a school of Greek philosophy that emphasizes the development of self-control and fortitude as a means of overcoming destructive emotions."
        
        try:
            # Generate embedding
            embedding = await vector_store.get_embedding(test_text)
            embedding_success = embedding is not None and len(embedding) > 0
            
            self.log_test_result(
                "Vector Embedding Generation", 
                embedding_success, 
                f"Generated embedding of dimension {len(embedding) if embedding else 0}" if embedding_success else "Failed to generate embedding",
                failure_recommendation="Check vector store and embedding model configuration"
            )
            
            # Test vector storage
            if embedding_success:
                test_doc_id = f"test_{uuid.uuid4()}"
                storage_result = await vector_store.store_embedding(
                    doc_id=test_doc_id,
                    text=test_text,
                    embedding=embedding,
                    metadata={"source": "test", "category": "philosophy"}
                )
                
                self.log_test_result(
                    "Vector Storage", 
                    storage_result, 
                    "Stored vector embedding successfully" if storage_result else "Failed to store vector embedding",
                    failure_recommendation="Check vector store write permissions"
                )
                
                # Test vector search
                search_query = "philosophy of self-control"
                search_results = await vector_store.search(search_query, limit=1)
                search_success = search_results and len(search_results) > 0
                
                self.log_test_result(
                    "Vector Search", 
                    search_success, 
                    f"Search returned {len(search_results) if search_results else 0} results" if search_success else "Search returned no results",
                    failure_recommendation="Check vector search functionality"
                )
                
                # Check if search result matches our test document
                if search_success:
                    result_matches = any(res.get("id") == test_doc_id for res in search_results)
                    self.log_test_result(
                        "Vector Search Accuracy", 
                        result_matches, 
                        "Search correctly returned test document" if result_matches else "Search did not return test document",
                        test_type=TEST_WARN if not result_matches else TEST_PASS
                    )
                
                # Clean up test data
                await vector_store.delete_document(test_doc_id)
            
        except Exception as e:
            self.log_test_result(
                "Vector Store Operations", 
                False, 
                f"Error during vector store operations: {str(e)}",
                failure_recommendation="Check vector store configuration and API access"
            )
    
    async def test_template_engine(self):
        """Test template engine functionality."""
        self.log_test_section("Template Engine Tests")
        
        template_engine = TemplateEngine()
        
        # Get a template to test
        mongodb = MongoDBService()
        templates, _ = await mongodb.list_templates(limit=1)
        
        if not templates:
            self.log_test_result(
                "Template Availability", 
                False, 
                "No templates available for testing",
                failure_recommendation="Add at least one template to the system"
            )
            return
        
        template = templates[0]
        
        # Basic context for testing
        test_context = {
            "concept": {
                "id": "stoicism",
                "name": "Stoicism",
                "description": "A philosophy of personal ethics informed by its system of logic and views on the natural world."
            },
            "concepts": [
                {
                    "id": "stoicism",
                    "name": "Stoicism",
                    "description": "A philosophy of personal ethics informed by its system of logic and views on the natural world."
                }
            ],
            "difficulty": "intermediate",
            "age_range": "14-18",
            "content_type": "lesson"
        }
        
        try:
            # Process template
            from models.content import ContentType, ContentDifficulty
            from models.template import ContentTemplate
            
            # Convert dict to ContentTemplate
            if isinstance(template, dict):
                template = ContentTemplate(**template)
                
            processed_template = await template_engine.process_template(
                template=template,
                context=test_context,
                content_type=ContentType.LESSON,
                difficulty=ContentDifficulty.INTERMEDIATE,
                age_range="14-18"
            )
            
            processing_success = processed_template and processed_template.content and len(processed_template.content) > 0
            
            self.log_test_result(
                "Template Processing", 
                processing_success, 
                f"Processed template with {len(processed_template.content) if processed_template and processed_template.content else 0} characters" if processing_success else "Failed to process template",
                failure_recommendation="Check template engine configuration and template structure"
            )
            
            # Check for variable substitution
            if processing_success and "stoicism" in test_context["concept"]["name"].lower():
                variable_substitution = "stoicism" in processed_template.content.lower()
                self.log_test_result(
                    "Variable Substitution", 
                    variable_substitution, 
                    "Template variables were substituted correctly" if variable_substitution else "Template variables may not have been substituted correctly",
                    test_type=TEST_WARN if not variable_substitution else TEST_PASS
                )
            
        except Exception as e:
            self.log_test_result(
                "Template Engine Operations", 
                False, 
                f"Error during template engine operations: {str(e)}",
                failure_recommendation="Check template engine implementation and template format"
            )
    
    async def test_rag_processor(self):
        """Test RAG processor functionality."""
        self.log_test_section("RAG Processor Tests")
        
        rag_processor = RAGProcessor()
        
        # Test RAG query
        test_query = "What are the main principles of Stoicism?"
        test_context = {"concept_id": "stoicism"}
        
        try:
            # Process query
            rag_result = await rag_processor.process_query(
                query=test_query,
                context=test_context
            )
            
            processing_success = rag_result and rag_result.content and len(rag_result.content) > 0
            
            self.log_test_result(
                "RAG Query Processing", 
                processing_success, 
                f"Processed RAG query with {len(rag_result.content) if rag_result and rag_result.content else 0} characters of response" if processing_success else "Failed to process RAG query",
                failure_recommendation="Check RAG processor, LLM service, and vector store configuration"
            )
            
            # Check retrieval results
            if processing_success:
                retrieval_success = len(rag_result.retrieval_results) > 0
                self.log_test_result(
                    "RAG Retrieval", 
                    retrieval_success, 
                    f"RAG retrieved {len(rag_result.retrieval_results)} results" if retrieval_success else "RAG did not retrieve any results",
                    test_type=TEST_WARN if not retrieval_success else TEST_PASS
                )
                
                # Check citations
                citation_success = len(rag_result.citations) > 0
                self.log_test_result(
                    "RAG Citations", 
                    citation_success, 
                    f"RAG generated {len(rag_result.citations)} citations" if citation_success else "RAG did not generate any citations",
                    test_type=TEST_INFO  # Informational only, not critical
                )
            
        except Exception as e:
            self.log_test_result(
                "RAG Processor Operations", 
                False, 
                f"Error during RAG processor operations: {str(e)}",
                failure_recommendation="Check RAG processor implementation and dependencies"
            )
    
    async def test_content_generator(self):
        """Test content generator functionality."""
        self.log_test_section("Content Generator Tests")
        
        content_generator = ContentGenerator()
        
        # Create a test content request
        from models.content import ContentRequest, ContentType, ContentDifficulty
        
        test_concept_id = self.test_data["concept_ids"][0]
        test_request = ContentRequest(
            concept_id=test_concept_id,
            content_type=ContentType.LESSON,
            difficulty=ContentDifficulty.INTERMEDIATE,
            age_range="14-18",
            template_id=None  # Use default template
        )
        
        try:
            # Generate content
            content_response = await content_generator.generate_content(test_request)
            
            generation_success = content_response and content_response.content and len(content_response.content) > 0
            
            self.log_test_result(
                "Content Generation", 
                generation_success, 
                f"Generated content with ID {content_response.content_id if content_response else 'N/A'}" if generation_success else "Failed to generate content",
                failure_recommendation="Check content generator, template engine, and LLM service configuration"
            )
            
            # Save content ID for further tests
            if generation_success:
                self.test_data["generated_content_ids"].add(content_response.content_id)
                
                # Check content structure
                structure_valid = (
                    content_response.sections and 
                    len(content_response.sections) > 0 and
                    all(hasattr(section, "content") and section.content for section in content_response.sections)
                )
                
                self.log_test_result(
                    "Content Structure", 
                    structure_valid, 
                    f"Generated content has {len(content_response.sections) if content_response.sections else 0} sections" if structure_valid else "Generated content has invalid structure",
                    test_type=TEST_WARN if not structure_valid else TEST_PASS
                )
                
                # Check if content contains concept information
                concept_relevant = test_concept_id.lower() in content_response.content.lower()
                self.log_test_result(
                    "Content Relevance", 
                    concept_relevant, 
                    "Generated content references the requested concept" if concept_relevant else "Generated content may not reference the requested concept",
                    test_type=TEST_WARN if not concept_relevant else TEST_PASS
                )
                
                # Check metadata
                metadata_valid = (
                    content_response.metadata and
                    "content_type" in content_response.metadata and
                    "difficulty" in content_response.metadata
                )
                
                self.log_test_result(
                    "Content Metadata", 
                    metadata_valid, 
                    "Generated content has valid metadata" if metadata_valid else "Generated content has invalid metadata",
                    test_type=TEST_WARN if not metadata_valid else TEST_PASS
                )
            
        except Exception as e:
            self.log_test_result(
                "Content Generator Operations", 
                False, 
                f"Error during content generation: {str(e)}",
                failure_recommendation="Check content generator implementation and dependencies"
            )
    
    async def analyze_logs(self) -> Dict[str, Any]:
        """Analyze test logs to find patterns and issues."""
        logger.info("Analyzing test logs")
        
        try:
            # Create log analyzer
            analyzer = create_log_analyzer(
                log_file=log_file,
                auto_analyze=False
            )
            
            # Run analysis on logs generated during tests
            analysis_result = analyzer.analyze(since=self.start_time)
            
            # Extract key findings
            if "error_summary" in analysis_result and "performance_summary" in analysis_result:
                # Log key findings
                error_count = analysis_result["error_summary"]["total_errors"]
                slow_ops = analysis_result["performance_summary"]["slow_operations_count"]
                
                logger.info(f"Log analysis complete: found {error_count} errors and {slow_ops} slow operations")
                
                # Add recommendations from log analysis to our recommendations
                for recommendation in analysis_result.get("recommendations", []):
                    self.recommendations.append(recommendation)
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error analyzing logs: {str(e)}")
            return {
                "error": str(e),
                "success": False
            }
    
    def log_test_result(self, test_name: str, result: bool, message: str, 
                      failure_recommendation: Optional[str] = None, test_type: Optional[str] = None):
        """
        Log a test result.
        
        Args:
            test_name: Name of the test
            result: Test result (True=Pass, False=Fail)
            message: Test message
            failure_recommendation: Recommendation if test failed
            test_type: Override test type (PASS, FAIL, WARN, INFO)
        """
        if test_type is None:
            test_type = TEST_PASS if result else TEST_FAIL
        
        # Determine log level based on test type
        if test_type == TEST_FAIL:
            logger.error(f"{test_name}: {message}")
            self.issues_found.append({"test": test_name, "message": message})
            if failure_recommendation:
                self.recommendations.append(failure_recommendation)
        elif test_type == TEST_WARN:
            logger.warning(f"{test_name}: {message}")
            self.warnings.append({"test": test_name, "message": message})
        else:
            logger.info(f"{test_name}: {message}")
        
        # Record test result
        self.test_results.append({
            "test": test_name,
            "result": test_type,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "recommendation": failure_recommendation
        })
    
    def log_test_section(self, section_name: str):
        """Log the start of a test section."""
        logger.info(f"\n----- {section_name} -----")
        self.test_results.append({
            "section": section_name,
            "timestamp": datetime.now().isoformat()
        })
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive test report.
        
        Returns:
            Test report dictionary
        """
        # Calculate summary statistics
        pass_count = sum(1 for r in self.test_results if "result" in r and r["result"] == TEST_PASS)
        fail_count = sum(1 for r in self.test_results if "result" in r and r["result"] == TEST_FAIL)
        warn_count = sum(1 for r in self.test_results if "result" in r and r["result"] == TEST_WARN)
        info_count = sum(1 for r in self.test_results if "result" in r and r["result"] == TEST_INFO)
        
        # Create report
        report = {
            "timestamp": datetime.now().isoformat(),
            "test_duration_seconds": (datetime.now() - self.start_time).total_seconds(),
            "summary": {
                "total_tests": pass_count + fail_count + warn_count + info_count,
                "pass": pass_count,
                "fail": fail_count,
                "warnings": warn_count,
                "info": info_count,
                "success_rate": pass_count / max(1, (pass_count + fail_count)) * 100
            },
            "issues": self.issues_found,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
            "test_results": self.test_results,
            "test_data": {
                "generated_content_ids": list(self.test_data["generated_content_ids"]),
                "template_ids": list(self.test_data["template_ids"]),
                "request_ids": list(self.test_data["request_ids"])
            }
        }
        
        return report
    
    def save_report(self, report: Dict[str, Any], filename: Optional[str] = None) -> str:
        """
        Save the test report to a file.
        
        Args:
            report: Test report
            filename: Optional filename
            
        Returns:
            Path to saved report
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"gutenberg_test_report_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Test report saved to {filename}")
            return filename
        except Exception as e:
            logger.error(f"Error saving test report: {e}")
            return ""
    
    def _validate_concept_structure(self, concept: Dict[str, Any]) -> bool:
        """
        Validate concept data structure.
        
        Args:
            concept: Concept data
            
        Returns:
            True if valid
        """
        required_fields = ["id", "name", "description"]
        return all(field in concept for field in required_fields)

async def run_tests(args):
    """Run the specified tests."""
    tester = GutenbergTester(config_path=args.config)
    report = await tester.run_all_tests()
    
    # Print summary to console
    summary = report["summary"]
    
    print("\n========== GUTENBERG TEST SUMMARY ==========")
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Pass: {summary['pass']}")
    print(f"Fail: {summary['fail']}")
    print(f"Warnings: {summary['warnings']}")
    print(f"Success Rate: {summary['success_rate']:.1f}%")
    print("============================================\n")
    
    # Print issues if any
    if report["issues"]:
        print("\nISSUES FOUND:")
        for issue in report["issues"]:
            print(f"- {issue['test']}: {issue['message']}")
    
    # Print recommendations
    if report["recommendations"]:
        print("\nRECOMMENDATIONS:")
        for rec in report["recommendations"]:
            print(f"- {rec}")
    
    print(f"\nDetailed report saved to: {report.get('report_file', 'unknown')}")
    
    return report

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Gutenberg System Testing Tool")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--output", help="Output file for test report")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze logs, don't run tests")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run_tests(args))