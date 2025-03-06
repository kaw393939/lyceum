# Goliath Shared Resources Architecture

This document describes how database resources are shared between microservices in the Goliath educational platform, along with implementation details and best practices.

## 1. System Architecture Overview

The Goliath platform consists of multiple microservices, each responsible for specific functions in the educational ecosystem:

- **Ptolemy**: Knowledge mapping and concept graph management
- **Gutenberg**: Educational content generation and templating
- **Galileo**: Learning path recommendations and personalization (planned)
- **Socrates**: Learner interaction and feedback collection (planned)

These services collaborate through shared databases and APIs to create a comprehensive educational system. Our shared infrastructure approach improves efficiency, reduces duplication, and ensures consistency across the platform.

## 2. Shared Resources

The following resources are shared across services:

- **Qdrant Vector Database**: Used for semantic search and vector storage
- **MongoDB**: Used for document storage with separate databases per service
- **Neo4j**: Used by Ptolemy for knowledge graph relationships

### 2.1 Qdrant Vector Database

#### Configuration Details

- **Collection name**: `goliath_vectors`
- **Vector size**: 1536 dimensions (matching OpenAI's embedding dimensions)
- **Distance metric**: Cosine similarity
- **Indexing threshold**: 20,000 vectors
- **On-disk payload**: Enabled for persistence and scalability

#### Namespacing Implementation

Each service adds a metadata field to vectors to maintain logical separation:
- **Field name**: `service`
- **Values**:
  - `ptolemy` for knowledge concepts
  - `gutenberg` for educational content

Services also prefix vector IDs with service-specific identifiers:
- `ptolemy_` for Ptolemy vectors
- `gutenberg_` for Gutenberg vectors

#### Search Filtering

When performing semantic search, services include a filter condition to limit results to their namespace:

```python
# Example filter in Qdrant query (pseudo-code)
filter_condition = FieldCondition(
    key="service",
    match=MatchValue(value="service_name")
)
```

This prevents cross-service contamination while maintaining the advantages of a shared collection.

#### Cross-Service Search

For cases where cross-service search is needed (e.g., Ptolemy finding related content from Gutenberg), the namespace filter can be omitted or modified to include multiple services.

### 2.2 MongoDB

MongoDB provides document storage with logical separation between services:

#### Database Organization

Each service uses its own database within the shared MongoDB instance:

- **ptolemy**: Stores concept metadata, relationships, and analytics
  - Collections: concepts, domains, relationships, learning_paths, etc.
  
- **gutenberg**: Stores content templates, generated content, and feedback
  - Collections: templates, content, generations, feedback, etc.

#### Authentication and Security

- Each service connects with dedicated service-specific credentials
- Database initialization creates appropriate users with scoped permissions
- Authentication is enforced for all connections
- Proper indexes are created on startup for optimal performance

### 2.3 Neo4j Graph Database

Neo4j is used exclusively by Ptolemy to manage concept relationships in a graph structure.

- **Authentication**: Standard username/password
- **Graph isolation**: All services use the same database but different node types
- **Connection pooling**: Configured for optimal performance

## 3. Service Integration Patterns

### 3.1 Ptolemy Client in Gutenberg

Gutenberg integrates with Ptolemy to access concept data for content generation. This integration uses several optimization patterns:

#### Batch Operations

```python
async def get_concepts_batch(self, concept_ids: List[str]) -> List[Dict[str, Any]]:
    """Fetch multiple concepts in a single API call."""
    # Implementation limits batch size and handles chunking
```

#### Caching Layer

```python
# Cache configuration
self._cache = {}  # In-memory cache
self._cache_ttl = 300  # 5 minutes expiration
self._max_cache_size = 1000  # Maximum entries
```

#### Automatic Retry with Backoff

```python
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((httpx.RequestError, httpx.TimeoutException))
)
async def _make_request(self, endpoint: str, method: str = "GET", 
                      data: Optional[Dict] = None) -> Dict[str, Any]:
    # Implementation handles retries with exponential backoff
```

#### Parallel Request Processing

```python
async def get_multiple_concepts(self, concept_ids: List[str]) -> List[Dict[str, Any]]:
    """Fetch multiple concepts in parallel with concurrency control."""
    # Creates multiple concurrent requests with throttling
```

### 3.2 Performance Considerations

To maintain optimal performance across shared resources:

- **Connection Pooling**: All database clients use connection pools
- **Health Checks**: All services implement health endpoints
- **Circuit Breaking**: API clients fail fast when services are down
- **Graceful Degradation**: Services handle unavailable dependencies
- **Memory Management**: Caches expire old data to limit memory usage

## 4. Operational Aspects

### 4.1 Startup Order and Dependencies

Services are started with appropriate dependencies:

1. **Databases**: MongoDB, Qdrant, Neo4j start first
2. **Core Services**: Ptolemy starts next as it provides foundational data
3. **Dependent Services**: Gutenberg, Galileo, and Socrates start last

### 4.2 Health Checking

All services expose health endpoints that:
- Check database connectivity
- Verify dependent service availability
- Report service status

These endpoints are used by Docker for orchestration and monitoring.

### 4.3 Error Handling and Resilience

Services implement robust error handling:
- Retry logic for transient failures
- Circuit breakers for persistent failures
- Fallback mechanisms (e.g., mock mode)
- Comprehensive logging

## 5. Adding New Services

When adding a new service to the Goliath platform:

1. Use the shared `goliath_vectors` collection with a new service namespace
2. Create a dedicated database in MongoDB with appropriate authentication
3. Add required healthchecks in docker-compose.yml
4. Define clear API contracts for communication with other services
5. Implement caching, batching, and resilience patterns
6. Add appropriate logging and monitoring

### 5.1 Service Template

New services should follow the established pattern:
- Docker containerization
- Health check endpoints
- Documented API contracts
- Shared database access with namespacing
- Comprehensive logging

## 6. Future Enhancements

Planned improvements to the shared architecture:

- **Event Bus**: Implement RabbitMQ/Kafka for asynchronous communication
- **API Gateway**: Create unified access point for all services
- **Distributed Tracing**: Add OpenTelemetry for cross-service request tracing
- **Full-Text Search**: Add Elasticsearch to complement vector search
- **Service Mesh**: Implement for advanced service discovery and circuit breaking
- **Zero-Downtime Deployment**: Enhance container orchestration for continuous operation

## 7. Monitoring and Observability

The current implementation includes:
- Structured logging across all services
- Prometheus metrics endpoints
- Log aggregation
- Health check dashboards

Future plans include:
- Enhanced metrics collection
- Centralized observability platform
- Automated alerting
- Performance trend analysis