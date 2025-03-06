#!/bin/bash

# Script to run integration tests against real Ptolemy and Gutenberg services

# Determine script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default values
PTOLEMY_URL="http://localhost:8000"
GUTENBERG_URL="http://localhost:8001"
MONGODB_URI="mongodb://localhost:27017"
API_KEY=""
VERBOSE=0
RUN_ALL=0
SPECIFIC_TEST=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --ptolemy-url)
      PTOLEMY_URL="$2"
      shift 2
      ;;
    --gutenberg-url)
      GUTENBERG_URL="$2"
      shift 2
      ;;
    --mongodb-uri)
      MONGODB_URI="$2"
      shift 2
      ;;
    --api-key)
      API_KEY="$2"
      shift 2
      ;;
    --verbose|-v)
      VERBOSE=1
      shift
      ;;
    --all|-a)
      RUN_ALL=1
      shift
      ;;
    --test|-t)
      SPECIFIC_TEST="$2"
      shift 2
      ;;
    --help|-h)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --ptolemy-url URL     URL of the Ptolemy service (default: http://localhost:8000)"
      echo "  --gutenberg-url URL   URL of the Gutenberg service (default: http://localhost:8001)"
      echo "  --mongodb-uri URI     MongoDB connection URI (default: mongodb://localhost:27017)"
      echo "  --api-key KEY         API key for authentication"
      echo "  --verbose, -v         Enable verbose output"
      echo "  --all, -a             Run all tests, including load and stress tests"
      echo "  --test, -t TEST       Run a specific test module or test case"
      echo "  --help, -h            Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Check if services are running
echo "Checking if services are running..."
PTOLEMY_RUNNING=0
GUTENBERG_RUNNING=0

if curl -s -o /dev/null -w "%{http_code}" "${PTOLEMY_URL}/health" | grep -q "200"; then
    PTOLEMY_RUNNING=1
    echo "✓ Ptolemy service is running at ${PTOLEMY_URL}"
else
    echo "✗ Ptolemy service is not running at ${PTOLEMY_URL}"
fi

if curl -s -o /dev/null -w "%{http_code}" "${GUTENBERG_URL}/health" | grep -q "200"; then
    GUTENBERG_RUNNING=1
    echo "✓ Gutenberg service is running at ${GUTENBERG_URL}"
else
    echo "✗ Gutenberg service is not running at ${GUTENBERG_URL}"
fi

if [[ $PTOLEMY_RUNNING -eq 0 || $GUTENBERG_RUNNING -eq 0 ]]; then
    echo "Error: Both services must be running to perform integration tests."
    echo "Start the services with ./start-services.sh and try again."
    exit 1
fi

# Set environment variables for the tests
export PTOLEMY_URL="${PTOLEMY_URL}"
export GUTENBERG_URL="${GUTENBERG_URL}"
export MONGODB_URI="${MONGODB_URI}"
export PLATO_API_KEY="${API_KEY}"

# Set Python path to include the project root
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Print test configuration
echo "Integration Test Configuration:"
echo "- Ptolemy URL: ${PTOLEMY_URL}"
echo "- Gutenberg URL: ${GUTENBERG_URL}"
echo "- MongoDB URI: ${MONGODB_URI}"
echo "- API Key: ${API_KEY:+[HIDDEN]}"
echo "- Test Mode: ${RUN_ALL -eq 1 && echo 'All tests' || echo 'Basic tests only'}"
echo "- Specific Test: ${SPECIFIC_TEST:-None}"

# Build the pytest command
PYTEST_CMD="pytest -v -m integration"

# Add filter for load tests if not running all tests
if [[ $RUN_ALL -eq 0 ]]; then
    PYTEST_CMD="${PYTEST_CMD} -k 'not (test_concurrent or test_service_resilience or test_search_performance)'"
fi

# Add specific test if provided
if [[ -n "$SPECIFIC_TEST" ]]; then
    PYTEST_CMD="${PYTEST_CMD} ${SPECIFIC_TEST}"
else
    PYTEST_CMD="${PYTEST_CMD} ${PROJECT_ROOT}/tests/integration/"
fi

# Add verbosity
if [[ $VERBOSE -eq 1 ]]; then
    PYTEST_CMD="${PYTEST_CMD} -v"
fi

# Run the tests
echo "Running integration tests..."
echo "Command: ${PYTEST_CMD}"
eval ${PYTEST_CMD}

# Save the exit code
EXIT_CODE=$?

# Print summary
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "✓ All integration tests passed!"
else
    echo "✗ Some integration tests failed."
fi

exit $EXIT_CODE