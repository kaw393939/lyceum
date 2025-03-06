#!/bin/bash

echo "Installing required test dependencies..."
pip install selenium requests webdriver-manager

echo "Running website tests..."
python test_website.py

echo "Test results complete!"