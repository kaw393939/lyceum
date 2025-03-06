#!/bin/bash

# Script to run service simulators for integration testing

# Determine script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default values
PTOLEMY_PORT=8000
GUTENBERG_PORT=8001
ERROR_RATE=0.0
DELAY_MS=0
BACKGROUND=1

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --ptolemy-port)
      PTOLEMY_PORT="$2"
      shift 2
      ;;
    --gutenberg-port)
      GUTENBERG_PORT="$2"
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
    --foreground)
      BACKGROUND=0
      shift
      ;;
    --help|-h)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --ptolemy-port PORT   Port for Ptolemy simulator (default: 8000)"
      echo "  --gutenberg-port PORT Port for Gutenberg simulator (default: 8001)"
      echo "  --error-rate RATE     Error rate between 0.0 and 1.0 (default: 0.0)"
      echo "  --delay-ms MS         Response delay in milliseconds (default: 0)"
      echo "  --foreground          Run in foreground instead of background"
      echo "  --help, -h            Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Activate virtual environment if it exists
if [ -d "${PROJECT_ROOT}/.venv" ]; then
    source "${PROJECT_ROOT}/.venv/bin/activate"
    echo "Activated virtual environment at ${PROJECT_ROOT}/.venv"
else
    echo "Warning: Virtual environment not found at ${PROJECT_ROOT}/.venv"
    echo "Make sure dependencies are installed globally."
fi

# Set Python path to include the project root
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Launch Ptolemy simulator
launch_ptolemy() {
    echo "Starting Ptolemy simulator on port ${PTOLEMY_PORT}..."
    python -m uvicorn thales.simulators.service_simulator:app \
        --port ${PTOLEMY_PORT} \
        --host 0.0.0.0 \
        --app-factory "from thales.simulators.service_simulator import ServiceSimulator; app = ServiceSimulator('ptolemy', ${PTOLEMY_PORT}, ${ERROR_RATE}, ${DELAY_MS}).app"
}

# Launch Gutenberg simulator
launch_gutenberg() {
    echo "Starting Gutenberg simulator on port ${GUTENBERG_PORT}..."
    python -m uvicorn thales.simulators.service_simulator:app \
        --port ${GUTENBERG_PORT} \
        --host 0.0.0.0 \
        --app-factory "from thales.simulators.service_simulator import ServiceSimulator; app = ServiceSimulator('gutenberg', ${GUTENBERG_PORT}, ${ERROR_RATE}, ${DELAY_MS}).app"
}

# Set environment variable to indicate we're using simulators
export USING_SIMULATORS=true

# Launch simulators
if [ ${BACKGROUND} -eq 1 ]; then
    # Run in background
    launch_ptolemy > "${PROJECT_ROOT}/logs/ptolemy_simulator.log" 2>&1 &
    PTOLEMY_PID=$!
    echo "Ptolemy simulator started with PID ${PTOLEMY_PID}"
    echo ${PTOLEMY_PID} > "${PROJECT_ROOT}/logs/ptolemy_simulator.pid"
    
    launch_gutenberg > "${PROJECT_ROOT}/logs/gutenberg_simulator.log" 2>&1 &
    GUTENBERG_PID=$!
    echo "Gutenberg simulator started with PID ${GUTENBERG_PID}"
    echo ${GUTENBERG_PID} > "${PROJECT_ROOT}/logs/gutenberg_simulator.pid"
    
    echo "Simulators are running in the background."
    echo "Use ./stop_simulator.sh to stop them."
else
    # Run Ptolemy in foreground, Gutenberg in background
    launch_gutenberg > "${PROJECT_ROOT}/logs/gutenberg_simulator.log" 2>&1 &
    GUTENBERG_PID=$!
    echo "Gutenberg simulator started with PID ${GUTENBERG_PID}"
    echo ${GUTENBERG_PID} > "${PROJECT_ROOT}/logs/gutenberg_simulator.pid"
    
    echo "Ptolemy simulator running in foreground. Press Ctrl+C to stop."
    launch_ptolemy
fi