# Thales - Ptolemy-Gutenberg Integration Testing Toolkit

Thales is a comprehensive testing toolkit for the Goliath Educational Platform, focusing on integration testing between Ptolemy (knowledge mapping service) and Gutenberg (content generation service).

Named after the Greek philosopher Thales of Miletus, who was known for unifying knowledge across different domains, this toolkit aims to ensure that different services in our educational platform work seamlessly together.

## Features

- **Real Service Testing**: Test against actual running services rather than simulations
- **End-to-End Integration**: Test the complete workflow from concept creation to content generation
- **Database Inspection**: Examine and compare data across MongoDB, Neo4j, and Qdrant
- **Data Consistency Verification**: Ensure data integrity between different data stores
- **Error Collection and Analysis**: Identify and diagnose integration issues
- **Mock Data Generation**: Create consistent test data across all database types
- **Service Simulation**: Test resilience with configurable failure modes and latency
- **Load Testing**: Evaluate performance under various conditions

## Quick Start

### Prerequisites

- Python 3.9+
- Running instances of Ptolemy and Gutenberg services (or use the simulators)
- Access to MongoDB for data verification (optional with simulators)

### Installation

```bash
# Clone the repository
git clone https://github.com/goliath-educational/thales.git
cd thales

# Using pip
pip install -e .
```

### Running Integration Tests

You can run integration tests either against real services or simulated services.

#### Using Real Services

```bash
# Run basic integration tests against real services
./scripts/run_integration_tests.sh

# Run with custom service URLs
./scripts/run_integration_tests.sh --ptolemy-url http://custom-ptolemy:8000 --gutenberg-url http://custom-gutenberg:8001

# Run all tests including load tests
./scripts/run_integration_tests.sh --all

# Run a specific test
./scripts/run_integration_tests.sh --test tests/integration/test_ptolemy_gutenberg.py::test_concept_creation_and_retrieval
```

#### Using Simulators

```bash
# Run basic integration tests with simulators
./scripts/run_simulator_tests.sh

# Run all tests including load tests with simulators
./scripts/run_simulator_tests.sh --all

# Run with error simulation
./scripts/run_simulator_tests.sh --error-rate 0.2

# Run with delay simulation
./scripts/run_simulator_tests.sh --delay-ms 100

# Test error handling and service resilience
./scripts/test_resilience.sh
```

## Test Types

### Basic Integration Tests

Tests basic functionality and integration between services:

- Creating and retrieving concepts in Ptolemy
- Generating content in Gutenberg based on Ptolemy concepts
- Testing learning path integration
- Verifying data consistency between services
- Testing error handling

### Load and Resilience Tests

Tests behavior under load and with failures:

- Concurrent content generation
- Service resilience with simulated timeouts
- Search performance testing

## Configuration

Environment variables can be used to configure the tests:

- `PTOLEMY_URL`: URL of the Ptolemy service (default: http://localhost:8000)
- `GUTENBERG_URL`: URL of the Gutenberg service (default: http://localhost:8001) 
- `MONGODB_URI`: URI for the MongoDB connection (default: mongodb://localhost:27017)
- `PLATO_API_KEY`: API key for authentication (if required)

## Development

Thales follows test-driven development practices:

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=src/thales
```

## Project Structure

```
/thales/
  /tests/
    /integration/
      conftest.py                  # Shared test fixtures
      test_real_services.py        # Basic service tests
      test_ptolemy_gutenberg.py    # Service integration tests
      test_load_resilience.py      # Load & resilience tests
  /scripts/
    run_integration_tests.sh       # Test runner script
  /src/
    /thales/                       # Core package
      /database/                   # Database inspection tools
      /simulators/                 # Service simulators
      /generators/                 # Mock data generators
  setup.py                         # Project setup
  README.md                        # Documentation
```

## Service Simulators

Thales includes service simulators for both Ptolemy and Gutenberg services, making it possible to run integration tests without requiring real service instances:

### Features

- **Mock API endpoints**: Simulators implement the same API endpoints as the real services
- **Configurable error rates**: Test error handling by simulating random service failures
- **Configurable delays**: Test timeout handling by adding artificial delays
- **In-memory data persistence**: Simulators maintain state during test execution
- **No database requirements**: Run tests without MongoDB, Neo4j, or Qdrant

### Using Simulators

```bash
# Run individual simulators
python -m src.thales.simulators.service_simulator --service ptolemy --port 8000
python -m src.thales.simulators.service_simulator --service gutenberg --port 8001

# Start both simulators with the helper script
./scripts/simulators/start_simulators.py

# Run integration tests against simulators
./scripts/run_simulator_tests.sh
```

## FAQ

**Q: Do I need all services running to run the tests?**  
A: You can either run tests against real services or use the built-in simulators. When using simulators, you don't need real services or databases.

**Q: Can I run the tests against production services?**  
A: We recommend running tests against development or staging environments to avoid any impact on production data. For most testing scenarios, the simulators are sufficient.

**Q: How do I debug test failures?**  
A: Run the tests with the `--verbose` flag for more detailed output. Check the test logs for specific error messages. When using simulators, you can add `--error-rate` and `--delay-ms` flags to test resilience.

## License

MIT