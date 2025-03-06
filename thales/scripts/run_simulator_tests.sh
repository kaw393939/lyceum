#!/bin/bash

# Script to run integration tests against simulated Ptolemy and Gutenberg services

# Determine script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default values
PTOLEMY_PORT=8000
GUTENBERG_PORT=8001
PTOLEMY_URL="http://localhost:${PTOLEMY_PORT}"
GUTENBERG_URL="http://localhost:${GUTENBERG_PORT}"
MONGODB_URI="mongodb://localhost:27017"
API_KEY=""
VERBOSE=0
RUN_ALL=0
SPECIFIC_TEST=""
ERROR_RATE=0.0
DELAY_MS=0
SIMULATOR_PROCESS_IDS=()

# Function to kill simulators on exit
cleanup() {
    echo "Shutting down simulators..."
    for pid in "${SIMULATOR_PROCESS_IDS[@]}"; do
        if ps -p $pid > /dev/null; then
            kill $pid
        fi
    done
}

# Register cleanup function to run on script exit
trap cleanup EXIT

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --ptolemy-port)
      PTOLEMY_PORT="$2"
      PTOLEMY_URL="http://localhost:${PTOLEMY_PORT}"
      shift 2
      ;;
    --gutenberg-port)
      GUTENBERG_PORT="$2"
      GUTENBERG_URL="http://localhost:${GUTENBERG_PORT}"
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
    --error-rate)
      ERROR_RATE="$2"
      shift 2
      ;;
    --delay-ms)
      DELAY_MS="$2"
      shift 2
      ;;
    --help|-h)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --ptolemy-port PORT     Port for Ptolemy simulator (default: 8000)"
      echo "  --gutenberg-port PORT   Port for Gutenberg simulator (default: 8001)"
      echo "  --mongodb-uri URI       MongoDB connection URI (default: mongodb://localhost:27017)"
      echo "  --api-key KEY           API key for authentication"
      echo "  --verbose, -v           Enable verbose output"
      echo "  --all, -a               Run all tests, including load and stress tests"
      echo "  --test, -t TEST         Run a specific test module or test case"
      echo "  --error-rate RATE       Error rate for simulators (0.0 to 1.0, default: 0.0)"
      echo "  --delay-ms MS           Artificial delay in milliseconds (default: 0)"
      echo "  --help, -h              Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Start simulators
echo "Starting service simulators..."

# Start Ptolemy simulator
echo "Starting Ptolemy simulator on port ${PTOLEMY_PORT}..."
python ${PROJECT_ROOT}/scripts/simulators/start_simulators.py --only ptolemy --ptolemy-port ${PTOLEMY_PORT} --error-rate ${ERROR_RATE} --delay-ms ${DELAY_MS} &
SIMULATOR_PROCESS_IDS+=($!)

# Wait for Ptolemy simulator to start
echo "Waiting for Ptolemy simulator to start..."
for i in {1..10}; do
    if curl -s -o /dev/null -w "%{http_code}" "${PTOLEMY_URL}/health" | grep -q "200"; then
        echo "✓ Ptolemy simulator is running at ${PTOLEMY_URL}"
        break
    fi
    if [ $i -eq 10 ]; then
        echo "✗ Ptolemy simulator failed to start"
        exit 1
    fi
    sleep 1
done

# Start Gutenberg simulator
echo "Starting Gutenberg simulator on port ${GUTENBERG_PORT}..."
python ${PROJECT_ROOT}/scripts/simulators/start_simulators.py --only gutenberg --gutenberg-port ${GUTENBERG_PORT} --error-rate ${ERROR_RATE} --delay-ms ${DELAY_MS} &
SIMULATOR_PROCESS_IDS+=($!)

# Wait for Gutenberg simulator to start
echo "Waiting for Gutenberg simulator to start..."
for i in {1..10}; do
    if curl -s -o /dev/null -w "%{http_code}" "${GUTENBERG_URL}/health" | grep -q "200"; then
        echo "✓ Gutenberg simulator is running at ${GUTENBERG_URL}"
        break
    fi
    if [ $i -eq 10 ]; then
        echo "✗ Gutenberg simulator failed to start"
        exit 1
    fi
    sleep 1
done

# Set environment variables for the tests
export PTOLEMY_URL="${PTOLEMY_URL}"
export GUTENBERG_URL="${GUTENBERG_URL}"
export MONGODB_URI="${MONGODB_URI}"
export PLATO_API_KEY="${API_KEY}"
export USING_SIMULATORS="true"

# Set Python path to include the project root
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Print test configuration
echo "Integration Test Configuration (with Simulators):"
echo "- Ptolemy Simulator URL: ${PTOLEMY_URL}"
echo "- Gutenberg Simulator URL: ${GUTENBERG_URL}"
echo "- MongoDB URI: ${MONGODB_URI}"
echo "- API Key: ${API_KEY:+[HIDDEN]}"
if [ $RUN_ALL -eq 1 ]; then
    echo "- Test Mode: All tests"
else
    echo "- Test Mode: Basic tests only"
fi
echo "- Specific Test: ${SPECIFIC_TEST:-None}"
echo "- Simulator Error Rate: ${ERROR_RATE}"
echo "- Simulator Delay: ${DELAY_MS}ms"

# Build the pytest command
PYTEST_CMD="pytest -v -m integration"

# Add filter for load tests if not running all tests
if [[ $RUN_ALL -eq 0 && -z "$SPECIFIC_TEST" ]]; then
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
echo "Running integration tests against simulators..."
echo "Command: ${PYTEST_CMD}"
eval ${PYTEST_CMD}

# Save the exit code
EXIT_CODE=$?

# Print summary
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "✓ All integration tests passed against simulators!"
else
    echo "✗ Some integration tests failed against simulators."
fi

exit $EXIT_CODE