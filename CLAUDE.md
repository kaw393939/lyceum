# CLAUDE.md - AI Assistant Context and Task Tracking

This file helps Claude maintain context between sessions and track progress on tasks for the Goliath Educational Platform.

## Project Overview
The Goliath Educational Platform is a microservices-based educational system with four main components:
- **Ptolemy**: Knowledge mapping service (Neo4j + Qdrant)
- **Gutenberg**: Content generation service (FastAPI + MongoDB)
- **Galileo**: Learning path recommendations (GNN-based)
- **Socrates**: Learner interaction system (Streamlit)

## Common Commands

### Starting Services
```bash
# Start all services
./start-services.sh

# Start individual services
cd gutenberg && python main.py
cd ptolemy && python main.py
cd socrates && python -m src.rag_demo.main
cd galileo && python app.py
```

### Testing
```bash
# Run all tests
./scripts/run-tests.sh

# Run service-specific tests
cd gutenberg && pytest
cd ptolemy && pytest
cd socrates && pytest
```

### Database Management
```bash
# Restart databases
./restart-database.sh
```

## Task Tracking

### Completed Tasks
- [x] Created CLAUDE.md for context maintenance (2025-03-04)
- [x] Added .flake8 configuration file (2025-03-04)
- [x] Created DOCSTRING_TEMPLATE.md with standardized docstring format (2025-03-04)
- [x] Created DEVELOPMENT_GUIDE.md with workflow documentation (2025-03-04)
- [x] Added pyproject.toml with Black, isort, and mypy configuration (2025-03-04)
- [x] Created ARCHITECTURE.md with text-based architecture diagrams (2025-03-04)
- [x] Created API_DOCUMENTATION.md with API standards and endpoints (2025-03-04)
- [x] Created CONTRIBUTING.md with contribution guidelines (2025-03-04)
- [x] Created comprehensive test plan for Ptolemy-Gutenberg integration (2025-03-05)

### Pending Tasks
- [ ] Implement integration tests based on test plan
- [ ] Create testing tools and utilities outlined in the test plan:
  - [ ] Database inspection tools for validating data consistency
  - [ ] Error collection and aggregation tools for debugging
  - [ ] Data consistency verification tools across MongoDB, Neo4j, and Qdrant
  - [ ] Service simulators with configurable error rates and latency
  - [ ] Load testing framework with detailed reporting
  - [ ] Mock data generators with database population capabilities
- [ ] Set up continuous integration for database validations
- [ ] Create test data fixtures for all database types
- [ ] Implement end-to-end test workflows
- [ ] Expand test coverage for individual services

## Style Guide

### Python Standards
- Use type hints for all function parameters and return values
- Document all functions with docstrings
- Follow PEP 8 conventions

### Project-Specific Patterns
- Use service-specific prefixes for database collections
- Implement consistent error handling with utils.error_handling decorators
- Follow existing naming conventions for each service

## Notes
- Pending fixes documented in GUTENBERG_FIXES.md and DATABASE_FIXES.md
- Microservices communicate via HTTP APIs
- MongoDB, Neo4j, and Qdrant are shared across services

## Ptolemy-Gutenberg Testing Toolkit

### Quick Start

The Ptolemy-Gutenberg Testing Toolkit provides comprehensive tools for testing the integration between these two critical services. This toolkit helps with identifying integration issues, validating data consistency, and ensuring robust error handling.

```bash
# Install testing toolkit
cd /home/kwilliams/projects/plato
pip install -e ./tools

# Run a quick integration test
python -m pgtest run --scenario basic_concept_retrieval

# Inspect database consistency
python -m pgtest verify --source mongodb --target neo4j

# Generate test data
python -m pgtest generate --concepts 50 --populate-all

# Run load test
python -m pgtest loadtest --target ptolemy --endpoint /concept/search --requests 100

# Collect and analyze errors
python -m pgtest diagnose --services ptolemy gutenberg --days 1
```

### Directory Structure

```
/tools/
  /pgtest/
    __init__.py
    cli.py                    # Command-line interface
    /fixtures/                # Test data and fixtures
    /database/                # Database inspection tools
      inspector.py            # DB inspection utilities
      consistency.py          # Data consistency verification
    /generators/              # Mock data generators
      concept_generator.py    # Mock concept generation
      relationship_generator.py # Mock relationship generation
    /simulators/              # Service simulators
      service_simulator.py    # Configurable service simulator
    /runners/                 # Test runners
      integration_runner.py   # Integration test runner
    /diagnostics/             # Diagnostic tools
      error_collector.py      # Log collection and analysis
    /load/                    # Load testing
      load_tester.py          # Load testing framework
    /config/                  # Configuration files
      inspector_config.yaml   # DB inspector configuration
      simulator_config.yaml   # Simulator configuration
      test_scenarios.yaml     # Test scenarios
```

### Using Database Inspector

The Database Inspector allows examining data across MongoDB, Neo4j, and Qdrant:

```python
from pgtest.database.inspector import DatabaseInspector

# Initialize with configuration
inspector = DatabaseInspector(config_file="config/inspector_config.yaml")

# Examine MongoDB data
concepts = inspector.inspect_mongodb(
    database="ptolemy", 
    collection="concepts",
    query={"name": {"$regex": "Python"}}
)
print(f"Found {len(concepts)} Python-related concepts")

# Run Cypher query against Neo4j
relationships = inspector.inspect_neo4j(
    query="""
    MATCH (a:Concept)-[r:PREREQUISITE]->(b:Concept) 
    RETURN a.name as source, b.name as target, r.weight as weight
    """
)
print(f"Found {len(relationships)} prerequisite relationships")

# Compare data between MongoDB and Neo4j
differences = inspector.compare_data(
    source_spec={"type": "mongodb", "database": "ptolemy", "collection": "concepts"},
    target_spec={"type": "neo4j", "query": "MATCH (c:Concept) RETURN c"},
    key_field="id"
)
print(f"Missing in Neo4j: {len(differences['missing_in_target'])}")
print(f"Missing in MongoDB: {len(differences['missing_in_source'])}")
```

### Data Consistency Verification

Verify data consistency between different data stores:

```python
from pgtest.database.consistency import ConsistencyVerifier
from pgtest.database.inspector import DatabaseInspector

# Initialize components
inspector = DatabaseInspector()
verifier = ConsistencyVerifier(inspector)

# Verify concepts across MongoDB and Neo4j
concept_results = verifier.verify_concepts()
print(f"MongoDB concepts: {concept_results['total_mongodb_concepts']}")
print(f"Neo4j concepts: {concept_results['total_neo4j_concepts']}")
print(f"Inconsistencies: {concept_results['inconsistency_count']}")

# Verify relationships
relationship_results = verifier.verify_relationships()
print(f"Relationships missing in Neo4j: {len(relationship_results['missing_in_neo4j'])}")

# Verify vector embeddings in Qdrant
embedding_results = verifier.verify_vector_embeddings()
print(f"Vector consistency: {embedding_results['consistency_percentage']:.2f}%")

# Generate a full consistency report
with open("consistency_report.json", "w") as f:
    json.dump({
        "concepts": concept_results,
        "relationships": relationship_results,
        "embeddings": embedding_results
    }, f, indent=2, default=str)
```

### Error Collection and Diagnosis

Collect and analyze error patterns across services:

```python
from pgtest.diagnostics.error_collector import ErrorCollector

# Initialize error collector
collector = ErrorCollector(log_dir="/var/log", report_dir="reports")

# Collect errors from service logs
ptolemy_errors = collector.collect_service_logs("ptolemy", days=2)
gutenberg_errors = collector.collect_service_logs("gutenberg", days=2)

print(f"Found {len(ptolemy_errors)} errors in Ptolemy logs")
print(f"Found {len(gutenberg_errors)} errors in Gutenberg logs")

# Generate comprehensive error report for multiple services
report_file = collector.generate_error_report(
    service_names=["ptolemy", "gutenberg", "socrates"],
    days=2
)
print(f"Error report generated: {report_file}")

# Analyze error patterns to identify common issues
from pgtest.diagnostics.error_analyzer import ErrorAnalyzer

analyzer = ErrorAnalyzer()
patterns = analyzer.find_correlated_errors(ptolemy_errors, gutenberg_errors)
print(f"Found {len(patterns)} correlated error patterns")
```

### Mock Data Generation

Generate test data and populate databases:

```python
from pgtest.generators.concept_generator import MockDataGenerator
from pymongo import MongoClient
from neo4j import GraphDatabase
from qdrant_client import QdrantClient

# Set up database clients
db_clients = {
    "mongodb": MongoClient("mongodb://localhost:27017"),
    "neo4j": GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password")),
    "qdrant": QdrantClient(url="http://localhost:6333")
}

# Initialize generator with database clients
generator = MockDataGenerator(db_clients=db_clients)

# Generate 50 mock concepts
concepts = generator.generate_mock_concepts(count=50, output_path="fixtures/concepts.json")
print(f"Generated {len(concepts)} mock concepts")

# Populate all databases
result = generator.generate_and_populate_all(concept_count=50)
print(f"MongoDB documents: {result['databases_populated'].get('mongodb', 0)}")
print(f"Neo4j nodes: {result['databases_populated'].get('neo4j', 0)}")
print(f"Qdrant vectors: {result['databases_populated'].get('qdrant', 0)}")
```

### Service Simulation

Simulate services with controlled failure modes:

```python
import asyncio
from pgtest.simulators.service_simulator import ServiceSimulator
import uvicorn

# Create a Ptolemy service simulator with 10% error rate and 50ms delay
ptolemy_sim = ServiceSimulator(
    service_name="ptolemy",
    port=8001,
    error_rate=0.1,  # 10% of requests will fail
    delay_ms=50      # 50ms artificial delay
)

# Add custom route with specific behavior
@ptolemy_sim.app.get("/related-concepts/{concept_id}")
async def get_related_concepts(concept_id: str, limit: int = 10):
    await ptolemy_sim._simulate_conditions()
    # Custom logic to return related concepts
    return [
        {"id": f"related_{i}", "name": f"Related Concept {i}", "relevance": 0.9 - (i * 0.1)}
        for i in range(limit)
    ]

# Run the simulator
uvicorn.run(ptolemy_sim.app, host="0.0.0.0", port=8001)
```

### Load Testing

Perform load testing on endpoints:

```python
import asyncio
from pgtest.load.load_tester import LoadTester

async def run_load_test():
    # Create load tester with concurrency of 20
    tester = LoadTester(
        target_url="http://localhost:8000",
        concurrency=20,
        verbose=True
    )
    
    # Run test against concept search endpoint
    results = await tester.run_test(
        endpoint="/api/concept/search",
        num_requests=500,
        method="POST",
        payload={"query": "machine learning", "limit": 5}
    )
    
    # Generate HTML report
    tester.generate_report(
        title="Concept Search Load Test",
        output_file="reports/load_test_report.html"
    )
    
    print(f"Average response time: {results['average_response_time_ms']:.2f}ms")
    print(f"Requests per second: {results['requests_per_second']:.2f}")
    print(f"Success rate: {results['success_rate'] * 100:.2f}%")

# Run the load test
asyncio.run(run_load_test())
```

### Test Scenarios

The toolkit includes predefined test scenarios:

```yaml
# Example from /tools/pgtest/config/test_scenarios.yaml
scenarios:
  basic_concept_retrieval:
    description: "Tests basic concept retrieval from Ptolemy to Gutenberg"
    steps:
      - name: "Generate test concepts"
        action: "generate_concepts"
        params:
          count: 10
          populate: true
      - name: "Request concept from Gutenberg"
        action: "request"
        params:
          service: "gutenberg"
          endpoint: "/api/content/generate"
          method: "POST"
          body: {"concept_id": "concept_1", "format": "lesson"}
      - name: "Verify database state"
        action: "verify_db"
        params:
          verifications:
            - type: "mongodb"
              database: "gutenberg"
              collection: "content"
              query: {"concept_id": "concept_1"}
              expected_count: 1
  
  error_resilience:
    description: "Tests Gutenberg resilience to Ptolemy service failures"
    steps:
      - name: "Start Ptolemy simulator with high error rate"
        action: "start_simulator"
        params:
          service: "ptolemy"
          port: 8001
          error_rate: 0.7
          delay_ms: 200
      - name: "Request content generation from Gutenberg"
        action: "request"
        params:
          service: "gutenberg"
          endpoint: "/api/content/generate"
          method: "POST"
          body: {"concept_id": "resilience_test", "format": "lesson"}
          expect_success: false
      - name: "Verify error logging"
        action: "verify_logs"
        params:
          service: "gutenberg"
          should_contain: ["Connection error", "Ptolemy service"]
```

### Running the Toolkit from CLI

```bash
# Run a specific test scenario
python -m pgtest run --scenario basic_concept_retrieval

# Generate test data and populate databases
python -m pgtest generate --concepts 50 --relationships 100 --populate-all

# Verify data consistency between services
python -m pgtest verify --all

# Run load test against service
python -m pgtest loadtest --target gutenberg --endpoint /api/content/generate --method POST --data '{"concept_id":"test_concept","format":"lesson"}' --requests 100 --concurrency 10

# Generate integration test report
python -m pgtest report --services ptolemy gutenberg --days 7 --output integration_report.html
```

## Ptolemy-Gutenberg Integration Test Plan

### 1. API Connectivity and Authentication Tests
- Basic connectivity between services
- Authentication with API keys
- Service discovery through Docker networking

### 2. Concept Retrieval Tests
- Single and batch concept retrieval
- Related concept retrieval with relationship preservation
- Concept graph traversal at various depths
- Semantic search functionality

### 3. Learning Path Integration Tests
- Learning path retrieval and content generation
- Step sequencing and relationship handling
- Path navigation (previous/next)

### 4. Error Handling and Resilience Tests
- Service unavailability handling
- Invalid data processing
- Retry mechanisms and backoff behavior

### 5. Caching Tests
- Cache functionality and hit/miss behavior
- TTL-based expiration and invalidation

### 6. Content Generation Integration Tests
- Content generation using Ptolemy knowledge
- RAG processing with retrieved data
- Template processing with concept information

### 7. End-to-End Workflow Tests
- Complete content generation pipeline
- Full learning path content creation

### 8. Performance and Load Tests
- Batch vs. individual operation performance
- Concurrent request handling
- Large learning path performance

### 9. Configuration Tests
- Environment variable configurations
- Connection settings optimization

### 10. Data Consistency Tests
- Concept data format consistency
- Learning path data integrity

## Testing Tools and Utilities

### Test Fixtures
```python
# pytest fixtures for service mocking
@pytest.fixture
def mock_ptolemy_service():
    """Returns a mocked Ptolemy service that returns predefined responses"""
    with patch("gutenberg.integrations.ptolemy_client.PtolemyClient") as mock:
        client = mock.return_value
        client.get_concept.return_value = sample_concept_data
        client.get_related_concepts.return_value = sample_related_concepts
        yield client

@pytest.fixture
def mock_gutenberg_service():
    """Returns a mocked Gutenberg service with predefined content generation"""
    with patch("socrates.clients.gutenberg_client.GutenbergClient") as mock:
        client = mock.return_value
        client.generate_content.return_value = sample_content
        yield client

@pytest.fixture
def mongodb_test_client():
    """Returns a MongoDB client connected to test database instances"""
    client = pymongo.MongoClient(
        os.getenv("MONGODB_TEST_URI", "mongodb://localhost:27017/test")
    )
    # Clear test collections before each test
    client.ptolemy_test.concepts.delete_many({})
    client.gutenberg_test.templates.delete_many({})
    client.gutenberg_test.content.delete_many({})
    
    yield client
    # Optional: clear data after test completes
    
@pytest.fixture
def neo4j_test_client():
    """Returns a Neo4j client connected to test database instance"""
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_TEST_URI", "bolt://localhost:7687"),
        auth=(
            os.getenv("NEO4J_TEST_USER", "neo4j"),
            os.getenv("NEO4J_TEST_PASSWORD", "password")
        )
    )
    # Clear test database before each test
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    
    yield driver
    driver.close()

@pytest.fixture
def qdrant_test_client():
    """Returns a Qdrant client connected to test vector store"""
    client = QdrantClient(
        url=os.getenv("QDRANT_TEST_URL", "http://localhost:6333")
    )
    # Clear test collections
    try:
        client.delete_collection("test_concepts")
    except Exception:
        pass
    
    client.create_collection(
        collection_name="test_concepts",
        vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE)
    )
    
    yield client
```

### Database Inspection Tools
```python
# Run with: python -m tools.db_inspector --db mongodb --collection concepts
class DatabaseInspector:
    """Tool for inspecting database state during and after integration tests"""
    
    def __init__(self, config_file="config/db_inspector.yaml"):
        self.config = self._load_config(config_file)
        self.clients = self._initialize_clients()
        
    def _load_config(self, config_file):
        with open(config_file, "r") as f:
            return yaml.safe_load(f)
    
    def _initialize_clients(self):
        clients = {}
        if "mongodb" in self.config:
            clients["mongodb"] = pymongo.MongoClient(self.config["mongodb"]["uri"])
        if "neo4j" in self.config:
            clients["neo4j"] = GraphDatabase.driver(
                self.config["neo4j"]["uri"],
                auth=(self.config["neo4j"]["user"], self.config["neo4j"]["password"])
            )
        if "qdrant" in self.config:
            clients["qdrant"] = QdrantClient(url=self.config["qdrant"]["url"])
        return clients
    
    def inspect_mongodb(self, database, collection, query=None, limit=10):
        """Inspects documents in MongoDB matching the given query"""
        query = query or {}
        client = self.clients["mongodb"]
        results = list(client[database][collection].find(query).limit(limit))
        return json.dumps(results, default=str, indent=2)
    
    def inspect_neo4j(self, query, params=None):
        """Runs a Cypher query against Neo4j and returns results"""
        params = params or {}
        client = self.clients["neo4j"]
        with client.session() as session:
            result = session.run(query, params)
            return [record.data() for record in result]
    
    def inspect_qdrant(self, collection, ids=None, limit=10):
        """Retrieves vectors from Qdrant collection"""
        client = self.clients["qdrant"]
        if ids:
            results = client.retrieve(collection_name=collection, ids=ids)
        else:
            results = client.scroll(
                collection_name=collection,
                limit=limit
            )[0]
        return results
    
    def compare_data(self, source_spec, target_spec, key_field="id"):
        """Compares data between source and target data stores"""
        # Source could be MongoDB, target could be Neo4j for example
        source_data = self._get_data(source_spec)
        target_data = self._get_data(target_spec)
        
        # Organize by key field for comparison
        source_map = {item.get(key_field): item for item in source_data}
        target_map = {item.get(key_field): item for item in target_data}
        
        # Find missing and mismatched items
        missing_in_target = [k for k in source_map if k not in target_map]
        missing_in_source = [k for k in target_map if k not in source_map]
        
        return {
            "missing_in_target": missing_in_target,
            "missing_in_source": missing_in_source,
            "common_keys": list(set(source_map.keys()) & set(target_map.keys()))
        }
```

### Diagnostic and Error Collection Tools
```python
# Run with: python -m tools.error_collector --service gutenberg --days 1
class ErrorCollector:
    """Collects and aggregates errors from service logs and databases"""
    
    def __init__(self, log_dir="logs", report_dir="reports"):
        self.log_dir = log_dir
        self.report_dir = report_dir
        os.makedirs(report_dir, exist_ok=True)
        
    def collect_service_logs(self, service_name, days=1, level="ERROR"):
        """Collects error logs from service log files"""
        service_log_path = os.path.join(self.log_dir, service_name)
        logs = []
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Find relevant log files
        log_files = []
        if os.path.exists(service_log_path):
            for file in os.listdir(service_log_path):
                if file.endswith(".log"):
                    log_files.append(os.path.join(service_log_path, file))
        
        # Extract errors from log files
        for log_file in log_files:
            file_logs = self._parse_log_file(log_file, start_date, end_date, level)
            logs.extend(file_logs)
            
        return logs
    
    def _parse_log_file(self, log_file, start_date, end_date, level):
        """Parses a log file and extracts relevant error entries"""
        entries = []
        with open(log_file, "r") as f:
            for line in f:
                try:
                    # Assuming log format like: 2025-03-05 14:22:15,123 - ERROR - Message
                    parts = line.split(" - ", 2)
                    if len(parts) >= 3:
                        timestamp_str = parts[0].strip()
                        log_level = parts[1].strip()
                        message = parts[2].strip()
                        
                        if level != "ALL" and log_level != level:
                            continue
                            
                        try:
                            timestamp = datetime.strptime(
                                timestamp_str, "%Y-%m-%d %H:%M:%S,%f"
                            )
                            if start_date <= timestamp <= end_date:
                                entries.append({
                                    "timestamp": timestamp,
                                    "level": log_level,
                                    "message": message
                                })
                        except ValueError:
                            # Skip lines with invalid timestamp format
                            pass
                except Exception:
                    # Skip malformed lines
                    pass
        return entries
    
    def collect_database_errors(self, db_inspector, days=1):
        """Collects error records from database error logs collection"""
        query = {
            "timestamp": {
                "$gte": datetime.now() - timedelta(days=days)
            },
            "level": "ERROR"
        }
        return json.loads(db_inspector.inspect_mongodb(
            database="logs", 
            collection="errors",
            query=query,
            limit=1000
        ))
    
    def generate_error_report(self, service_names, days=1):
        """Generates a comprehensive error report for specified services"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "period_days": days,
            "services": {}
        }
        
        for service in service_names:
            service_logs = self.collect_service_logs(service, days)
            report["services"][service] = {
                "log_errors": service_logs,
                "error_count": len(service_logs),
                "most_frequent": self._get_most_frequent_errors(service_logs)
            }
            
        # Write report to file
        report_file = os.path.join(
            self.report_dir, 
            f"error_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(report_file, "w") as f:
            json.dump(report, f, default=str, indent=2)
            
        return report_file
    
    def _get_most_frequent_errors(self, logs, top_n=10):
        """Identifies most frequent error patterns in logs"""
        # Simple approach: count occurrences of each message
        message_counts = {}
        for log in logs:
            msg = log.get("message", "")
            # Remove specific IDs or timestamps from message to group similar errors
            # This is a simple example - a more sophisticated approach might use regex
            message_counts[msg] = message_counts.get(msg, 0) + 1
            
        # Sort by count and return top N
        return sorted(
            [{"message": k, "count": v} for k, v in message_counts.items()],
            key=lambda x: x["count"],
            reverse=True
        )[:top_n]
```

### Service Simulators
```python
# Run with: python -m tools.service_simulator --service ptolemy --port 8001
class ServiceSimulator:
    """Simulates a microservice with configurable response delays and errors"""
    def __init__(self, service_name, port, error_rate=0.0, delay_ms=0):
        self.app = FastAPI(title=f"{service_name} Simulator")
        self.error_rate = error_rate
        self.delay_ms = delay_ms
        self.setup_routes()
        
    def setup_routes(self):
        @self.app.get("/health")
        async def health_check():
            return {"status": "ok"}
            
        @self.app.get("/concept/{concept_id}")
        async def get_concept(concept_id: str):
            await self._simulate_conditions()
            return test_data.get(concept_id, {"error": "not found"})
            
    async def _simulate_conditions(self):
        """Simulates real-world conditions like delays and errors"""
        # Simulate network delay
        if self.delay_ms > 0:
            await asyncio.sleep(self.delay_ms / 1000)
            
        # Simulate random errors
        if random.random() < self.error_rate:
            error_types = [
                HTTPException(status_code=500, detail="Internal server error"),
                HTTPException(status_code=503, detail="Service unavailable"),
                HTTPException(status_code=504, detail="Gateway timeout"),
            ]
            raise random.choice(error_types)
```

### Data Consistency Verification Tool
```python
# Run with: python -m tools.verify_consistency --source-db mongodb --target-db neo4j
class ConsistencyVerifier:
    """Verifies data consistency between different data stores"""
    
    def __init__(self, db_inspector):
        self.db_inspector = db_inspector
        
    def verify_concepts(self, source_db="mongodb", target_db="neo4j"):
        """Verifies concept data is consistent between MongoDB and Neo4j"""
        # Get concepts from MongoDB
        mongodb_concepts = json.loads(self.db_inspector.inspect_mongodb(
            database="ptolemy",
            collection="concepts",
            limit=1000
        ))
        
        # Get concepts from Neo4j
        neo4j_concepts = self.db_inspector.inspect_neo4j(
            query="MATCH (c:Concept) RETURN c"
        )
        neo4j_concepts = [record["c"] for record in neo4j_concepts]
        
        # Compare by ID
        mongodb_by_id = {c["_id"]: c for c in mongodb_concepts}
        neo4j_by_id = {c["id"]: c for c in neo4j_concepts}
        
        # Find inconsistencies
        inconsistencies = []
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
        
        # Check for Neo4j concepts missing in MongoDB
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
        
    def verify_relationships(self):
        """Verifies relationship data is consistent between MongoDB and Neo4j"""
        # Get relationships from MongoDB (assuming stored as subdocuments)
        mongodb_concepts = json.loads(self.db_inspector.inspect_mongodb(
            database="ptolemy",
            collection="concepts",
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
        neo4j_relationships = self.db_inspector.inspect_neo4j(
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
        
    def verify_vector_embeddings(self):
        """Verifies vector embeddings are consistent between Qdrant and concept data"""
        # Get concepts from MongoDB
        mongodb_concepts = json.loads(self.db_inspector.inspect_mongodb(
            database="ptolemy",
            collection="concepts",
            limit=100  # Limit for practicality
        ))
        concept_ids = [c["_id"] for c in mongodb_concepts]
        
        # Get vectors from Qdrant
        qdrant_vectors = self.db_inspector.inspect_qdrant(
            collection="concepts",
            ids=concept_ids
        )
        
        # Check all concepts have vectors
        qdrant_by_id = {v.id: v for v in qdrant_vectors}
        
        missing_vectors = []
        for concept_id in concept_ids:
            if concept_id not in qdrant_by_id:
                missing_vectors.append(concept_id)
                
        return {
            "total_concepts": len(concept_ids),
            "total_vectors": len(qdrant_vectors),
            "missing_vectors": missing_vectors,
            "consistency_percentage": 
                100 * (len(concept_ids) - len(missing_vectors)) / len(concept_ids)
                if concept_ids else 100
        }
```

### Integration Test Runner with Database Verification
```python
# Run with: python -m tools.integration_test_runner --config tests/configs/ptolemy_gutenberg.yaml
class IntegrationTestRunner:
    """Runs integration tests between specified services with DB verification"""
    def __init__(self, config_path):
        self.config = self._load_config(config_path)
        self.results = []
        self.db_inspector = DatabaseInspector()
        self.consistency_verifier = ConsistencyVerifier(self.db_inspector)
        
    def _load_config(self, path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)
            
    def run(self):
        """Executes all integration tests defined in the config"""
        for test_case in self.config["test_cases"]:
            # Capture database state before test
            before_state = self._capture_db_state(test_case.get("db_verification", []))
            
            # Run the test case
            result = self._run_test_case(test_case)
            
            # Capture database state after test
            after_state = self._capture_db_state(test_case.get("db_verification", []))
            
            # Verify changes match expectations
            db_verification = self._verify_db_changes(
                before_state, 
                after_state,
                test_case.get("expected_changes", {})
            )
            
            # Add verification results to test results
            result["db_verification"] = db_verification
            self.results.append(result)
            
        return self._generate_report()
        
    def _capture_db_state(self, verification_specs):
        """Captures database state for specified collections/tables"""
        state = {}
        for spec in verification_specs:
            db_type = spec["type"]  # mongodb, neo4j, qdrant
            if db_type == "mongodb":
                state[f"{db_type}:{spec['database']}:{spec['collection']}"] = (
                    json.loads(self.db_inspector.inspect_mongodb(
                        database=spec["database"],
                        collection=spec["collection"],
                        query=spec.get("query", {})
                    ))
                )
            elif db_type == "neo4j":
                state[f"{db_type}:{spec['query']}"] = self.db_inspector.inspect_neo4j(
                    query=spec["query"],
                    params=spec.get("params", {})
                )
            elif db_type == "qdrant":
                state[f"{db_type}:{spec['collection']}"] = self.db_inspector.inspect_qdrant(
                    collection=spec["collection"],
                    ids=spec.get("ids")
                )
        return state
        
    def _verify_db_changes(self, before, after, expected_changes):
        """Verifies database changes match expectations"""
        results = {}
        
        for key in set(before.keys()) | set(after.keys()):
            before_data = before.get(key, [])
            after_data = after.get(key, [])
            
            # Compare document counts
            before_count = len(before_data)
            after_count = len(after_data)
            count_diff = after_count - before_count
            
            # Check if count change matches expectation
            expected_diff = expected_changes.get(key, {}).get("count_diff", 0)
            results[key] = {
                "before_count": before_count,
                "after_count": after_count,
                "actual_diff": count_diff,
                "expected_diff": expected_diff,
                "matches_expectation": count_diff == expected_diff
            }
            
            # For more detailed verification we could add custom field comparisons here
            
        return results
```

### Mock Data Generator with Database Population
```python
# Run with: python -m tools.mock_data_generator --output tests/fixtures/concepts.json --populate-db
class MockDataGenerator:
    """Generates mock data for testing and populates test databases"""
    
    def __init__(self, db_clients=None):
        self.db_clients = db_clients or {}
        
    def generate_mock_concepts(self, count=10, output_path=None):
        """Generates mock concept data for testing purposes"""
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
        
    def _generate_mock_relationships(self, count):
        """Generates mock relationship data for concepts"""
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
        
    def populate_mongodb(self, concepts, database="ptolemy_test"):
        """Populates MongoDB with mock concept data"""
        if "mongodb" not in self.db_clients:
            raise ValueError("MongoDB client not provided")
            
        client = self.db_clients["mongodb"]
        collection = client[database]["concepts"]
        
        # Clear existing data
        collection.delete_many({})
        
        # Insert new documents
        result = collection.insert_many(concepts)
        return len(result.inserted_ids)
        
    def populate_neo4j(self, concepts, database="neo4j"):
        """Populates Neo4j with mock concept graph"""
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
        
    def populate_qdrant(self, concepts, collection_name="test_concepts"):
        """Populates Qdrant with mock vector embeddings for concepts"""
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
                size=768,  # Common embedding size
                distance=models.Distance.COSINE
            )
        )
        
        # Generate mock embeddings
        points = []
        for concept in concepts:
            # Generate random embedding vector
            vector = [random.uniform(-1, 1) for _ in range(768)]
            # Normalize the vector
            magnitude = math.sqrt(sum(x**2 for x in vector))
            vector = [x/magnitude for x in vector]
            
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
        
    def generate_and_populate_all(self, concept_count=100, output_path=None):
        """Generates mock data and populates all test databases"""
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
```

### Load Testing Tools
```python
# Run with: python -m tools.load_tester --target ptolemy --endpoint /concept --requests 100
class LoadTester:
    """Comprehensive load testing framework for microservices"""
    
    def __init__(self, target_url, concurrency=10, verbose=False):
        self.target_url = target_url
        self.concurrency = concurrency
        self.verbose = verbose
        self.results = []
        
    async def run_test(self, endpoint, num_requests, payload=None, method="GET"):
        """Runs a load test against the specified endpoint"""
        self.results = []
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for i in range(num_requests):
                task = self._timed_request(
                    session, 
                    f"{self.target_url}{endpoint}",
                    method=method,
                    payload=payload
                )
                tasks.append(task)
                
            if self.concurrency < num_requests:
                # Process in batches to control concurrency
                results = []
                for i in range(0, num_requests, self.concurrency):
                    batch = tasks[i:i+self.concurrency]
                    batch_results = await asyncio.gather(*batch)
                    results.extend(batch_results)
                    if self.verbose:
                        self._print_progress(i + len(batch), num_requests)
                self.results = results
            else:
                self.results = await asyncio.gather(*tasks)
                
        return self._analyze_results()
    
    async def _timed_request(self, session, url, method="GET", payload=None):
        """Makes a request and times the response"""
        start_time = time.time()
        status = None
        response_size = 0
        error = None
        
        try:
            if method == "GET":
                async with session.get(url) as response:
                    status = response.status
                    body = await response.read()
                    response_size = len(body)
            elif method == "POST":
                async with session.post(url, json=payload) as response:
                    status = response.status
                    body = await response.read()
                    response_size = len(body)
            else:
                raise ValueError(f"Unsupported method: {method}")
        except Exception as e:
            error = str(e)
            
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000
        
        return {
            "url": url,
            "method": method,
            "status": status,
            "duration_ms": duration_ms,
            "response_size": response_size,
            "error": error
        }
    
    def _print_progress(self, current, total):
        """Prints progress information during the test"""
        percent = 100 * current / total
        print(f"Progress: {current}/{total} requests ({percent:.1f}%)")
    
    def _analyze_results(self):
        """Analyzes test results and generates statistics"""
        if not self.results:
            return {"error": "No results to analyze"}
            
        # Calculate statistics
        durations = [r["duration_ms"] for r in self.results if r["error"] is None]
        success_count = len(durations)
        error_count = len(self.results) - success_count
        
        stats = {
            "total_requests": len(self.results),
            "successful_requests": success_count,
            "failed_requests": error_count,
            "success_rate": success_count / len(self.results) if self.results else 0,
            "total_duration_ms": sum(durations) if durations else 0,
            "average_response_time_ms": statistics.mean(durations) if durations else 0,
            "min_response_time_ms": min(durations) if durations else 0,
            "max_response_time_ms": max(durations) if durations else 0,
            "percentiles": {
                "50": statistics.median(durations) if durations else 0,
                "90": self._percentile(durations, 90) if durations else 0,
                "95": self._percentile(durations, 95) if durations else 0,
                "99": self._percentile(durations, 99) if durations else 0,
            },
            "requests_per_second": success_count / (sum(durations) / 1000) if durations else 0,
            "errors": self._analyze_errors()
        }
        
        return stats
    
    def _percentile(self, data, percentile):
        """Calculates the specified percentile from the data"""
        size = len(data)
        if not size:
            return 0
            
        sorted_data = sorted(data)
        k = (size - 1) * percentile / 100
        f = math.floor(k)
        c = math.ceil(k)
        
        if f == c:
            return sorted_data[int(k)]
            
        d0 = sorted_data[int(f)] * (c - k)
        d1 = sorted_data[int(c)] * (k - f)
        return d0 + d1
    
    def _analyze_errors(self):
        """Analyzes error patterns in results"""
        error_results = [r for r in self.results if r["error"] is not None]
        status_counts = {}
        
        # Count status codes
        for result in self.results:
            if result["status"] is not None:
                status = result["status"]
                status_counts[status] = status_counts.get(status, 0) + 1
                
        # Group errors by type
        error_types = {}
        for result in error_results:
            error = result["error"]
            error_types[error] = error_types.get(error, 0) + 1
            
        return {
            "status_counts": status_counts,
            "error_types": error_types
        }
        
    def generate_report(self, title="Load Test Report", output_file=None):
        """Generates a detailed HTML report of load test results"""
        analysis = self._analyze_results()
        
        # Generate HTML report
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                .stats {{ display: flex; flex-wrap: wrap; }}
                .stat-box {{ 
                    background: #f5f5f5; border-radius: 5px; padding: 15px;
                    margin: 10px; min-width: 200px; flex: 1;
                }}
                .chart {{ margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>{title}</h1>
            <p>Test run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Summary</h2>
            <div class="stats">
                <div class="stat-box">
                    <h3>Requests</h3>
                    <p>Total: {analysis['total_requests']}</p>
                    <p>Successful: {analysis['successful_requests']}</p>
                    <p>Failed: {analysis['failed_requests']}</p>
                    <p>Success Rate: {analysis['success_rate']*100:.2f}%</p>
                </div>
                
                <div class="stat-box">
                    <h3>Response Time (ms)</h3>
                    <p>Average: {analysis['average_response_time_ms']:.2f}</p>
                    <p>Minimum: {analysis['min_response_time_ms']:.2f}</p>
                    <p>Maximum: {analysis['max_response_time_ms']:.2f}</p>
                </div>
                
                <div class="stat-box">
                    <h3>Percentiles (ms)</h3>
                    <p>50%: {analysis['percentiles']['50']:.2f}</p>
                    <p>90%: {analysis['percentiles']['90']:.2f}</p>
                    <p>95%: {analysis['percentiles']['95']:.2f}</p>
                    <p>99%: {analysis['percentiles']['99']:.2f}</p>
                </div>
                
                <div class="stat-box">
                    <h3>Performance</h3>
                    <p>Throughput: {analysis['requests_per_second']:.2f} req/sec</p>
                    <p>Total Duration: {analysis['total_duration_ms']/1000:.2f} sec</p>
                </div>
            </div>
            
            <h2>HTTP Status Codes</h2>
            <table>
                <tr>
                    <th>Status Code</th>
                    <th>Count</th>
                    <th>Percentage</th>
                </tr>
        """
        
        for status, count in analysis["errors"]["status_counts"].items():
            percentage = 100 * count / analysis["total_requests"]
            html += f"""
                <tr>
                    <td>{status}</td>
                    <td>{count}</td>
                    <td>{percentage:.2f}%</td>
                </tr>
            """
            
        html += """
            </table>
            
            <h2>Error Types</h2>
            <table>
                <tr>
                    <th>Error Type</th>
                    <th>Count</th>
                    <th>Percentage</th>
                </tr>
        """
        
        for error, count in analysis["errors"]["error_types"].items():
            percentage = 100 * count / analysis["total_requests"]
            html += f"""
                <tr>
                    <td>{error}</td>
                    <td>{count}</td>
                    <td>{percentage:.2f}%</td>
                </tr>
            """
            
        html += """
            </table>
        </body>
        </html>
        """
        
        if output_file:
            with open(output_file, "w") as f:
                f.write(html)
                
        return html
```