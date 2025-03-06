#!/bin/bash

echo "Starting the Lyceum Educational Platform website"
echo "==============================================="

# Check if Python is installed
if ! command -v python >/dev/null 2>&1; then
  echo "Error: Python is not installed or not in PATH"
  exit 1
fi

# Create media assets if needed
echo "Step 1: Checking for assets and creating if necessary..."
python regenerate_assets.py --fallback --update-templates

# Create the latest asset links
echo "Step 2: Creating links to the latest assets..."
python create_latest_links.py

# Start the server
echo "Step 3: Starting the web server..."
echo "The website will be available at http://localhost:8081"
echo "Press Ctrl+C to stop the server when done"
echo "==============================================="

python serve.py