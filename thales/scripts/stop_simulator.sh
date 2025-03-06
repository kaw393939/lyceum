#!/bin/bash

# Script to stop service simulators

# Determine script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Check if PID files exist
PTOLEMY_PID_FILE="${PROJECT_ROOT}/logs/ptolemy_simulator.pid"
GUTENBERG_PID_FILE="${PROJECT_ROOT}/logs/gutenberg_simulator.pid"

if [ -f "$PTOLEMY_PID_FILE" ]; then
    PTOLEMY_PID=$(cat "$PTOLEMY_PID_FILE")
    echo "Stopping Ptolemy simulator (PID: $PTOLEMY_PID)..."
    if kill -0 $PTOLEMY_PID 2>/dev/null; then
        kill $PTOLEMY_PID
        echo "Ptolemy simulator stopped."
    else
        echo "Ptolemy simulator not running."
    fi
    rm "$PTOLEMY_PID_FILE"
else
    echo "Ptolemy PID file not found."
fi

if [ -f "$GUTENBERG_PID_FILE" ]; then
    GUTENBERG_PID=$(cat "$GUTENBERG_PID_FILE")
    echo "Stopping Gutenberg simulator (PID: $GUTENBERG_PID)..."
    if kill -0 $GUTENBERG_PID 2>/dev/null; then
        kill $GUTENBERG_PID
        echo "Gutenberg simulator stopped."
    else
        echo "Gutenberg simulator not running."
    fi
    rm "$GUTENBERG_PID_FILE"
else
    echo "Gutenberg PID file not found."
fi

echo "All simulators stopped."