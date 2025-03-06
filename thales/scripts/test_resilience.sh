#!/bin/bash

# Script to test resilience of integrations by introducing errors with simulators

# Default values
ERROR_RATES=(0.0 0.1 0.2 0.3 0.4 0.5)
DELAY_VALUES=(0 50 100 200 500)
TEST_NAME="test_error_handling"

echo "Testing resilience with simulated errors and delays"
echo "===================================================="

# Test increasing error rates
echo "Testing with increasing error rates (no delay):"
for rate in "${ERROR_RATES[@]}"; do
    echo -n "  Error rate ${rate}: "
    ./scripts/run_simulator_tests.sh --error-rate ${rate} --test "tests/integration/test_ptolemy_gutenberg.py::${TEST_NAME}" > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "✓ Passed"
    else
        echo "✗ Failed"
    fi
done

# Test increasing delays
echo "Testing with increasing delays (no errors):"
for delay in "${DELAY_VALUES[@]}"; do
    echo -n "  Delay ${delay}ms: "
    ./scripts/run_simulator_tests.sh --delay-ms ${delay} --test "tests/integration/test_ptolemy_gutenberg.py::${TEST_NAME}" > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "✓ Passed"
    else
        echo "✗ Failed"
    fi
done

# Test combined error rates and delays
echo "Testing with combined error rates and delays:"
for rate in 0.1 0.3; do
    for delay in 50 200; do
        echo -n "  Error rate ${rate}, delay ${delay}ms: "
        ./scripts/run_simulator_tests.sh --error-rate ${rate} --delay-ms ${delay} --test "tests/integration/test_ptolemy_gutenberg.py::${TEST_NAME}" > /dev/null 2>&1
        if [ $? -eq 0 ]; then
            echo "✓ Passed"
        else
            echo "✗ Failed"
        fi
    done
done

echo "Resilience testing complete!"