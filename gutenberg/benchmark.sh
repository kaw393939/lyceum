#!/bin/bash
# Run Gutenberg benchmark tests

# Ensure we're in the gutenberg directory
cd "$(dirname "$0")"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

echo "Running Gutenberg benchmark tests..."
echo "====================================="
python -m tests.benchmark_tests

# Deactivate virtual environment if it was activated
if [ -n "$VIRTUAL_ENV" ]; then
    deactivate
fi