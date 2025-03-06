#!/bin/bash
# Test script for Goliath project

echo "Running tests for Goliath project..."

# Set environment variables from .env file if it exists
if [ -f .env ]; then
    echo "Loading environment variables from .env file..."
    export $(cat .env | grep -v '^#' | xargs)
fi

# Function to run tests for a specific component
run_component_tests() {
    local component=$1
    echo "============================================="
    echo "Testing $component..."
    echo "============================================="
    
    if [ ! -d "$component" ]; then
        echo "Error: $component directory does not exist"
        return 1
    fi
    
    cd $component
    
    # Check if pytest is installed in the container
    if docker-compose exec $component pip list | grep -q pytest; then
        docker-compose exec $component pytest -xvs tests/
    else
        # Run tests locally if container doesn't have pytest
        if [ -f "requirements-dev.txt" ]; then
            echo "Installing development dependencies..."
            pip install -r requirements-dev.txt
        fi
        pytest -xvs tests/
    fi
    
    cd ..
    return $?
}

# Run all tests if no argument is provided, otherwise run tests for the specified component
if [ $# -eq 0 ]; then
    # Test each component
    run_component_tests ptolemy
    ptolemy_status=$?
    
    run_component_tests gutenberg
    gutenberg_status=$?
    
    # Check for failures
    if [ $ptolemy_status -ne 0 ] || [ $gutenberg_status -ne 0 ]; then
        echo "Some tests failed!"
        exit 1
    fi
else
    # Run tests for the specified component
    run_component_tests $1
    if [ $? -ne 0 ]; then
        echo "Tests for $1 failed!"
        exit 1
    fi
fi

echo "All tests completed successfully!"