#!/bin/bash

echo "Installing dependencies for Lyceum development..."

# Make sure pip is installed
if ! command -v pip >/dev/null 2>&1; then
    echo "Error: pip is not installed"
    echo "Please install pip first, then run this script again"
    exit 1
fi

# Install Python packages
echo "Installing Python packages..."
pip install watchdog[watchmedo] pytest pillow flask flask-cors

# Check if Docker is installed
if command -v docker >/dev/null 2>&1; then
    echo "Docker is installed, checking Docker Compose..."
    
    # Check if Docker Compose is installed
    if command -v docker-compose >/dev/null 2>&1 || docker compose version >/dev/null 2>&1; then
        echo "Docker Compose is installed, you're all set for containerized development!"
    else
        echo "Warning: Docker is installed but Docker Compose was not found"
        echo "Please install Docker Compose to use the containerized development environment"
    fi
else
    echo "Docker is not installed"
    echo "For the best development experience, we recommend installing Docker and Docker Compose"
    echo "See https://docs.docker.com/get-docker/ for installation instructions"
fi

echo "Dependencies installation complete!"
echo "You can now run the application using Docker Compose:"
echo "   docker-compose up"
echo "Or run it directly with live reloading:"
echo "   ./start_website.sh"