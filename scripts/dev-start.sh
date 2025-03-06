#!/bin/bash
# Development startup script for Goliath project

echo "Starting Goliath development environment..."

# Create required directories
echo "Creating required directories..."
mkdir -p gutenberg/media
mkdir -p gutenberg/storage/reports
mkdir -p ptolemy/data

# Set environment variables from .env file if it exists
if [ -f .env ]; then
    echo "Loading environment variables from .env file..."
    export $(cat .env | grep -v '^#' | xargs)
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running or not installed."
    echo "Please start Docker and try again."
    exit 1
fi

# Start the services using docker-compose
echo "Starting services..."
docker-compose down
docker-compose up -d mongodb neo4j qdrant

# Wait for databases to be ready
echo "Waiting for databases to start..."
sleep 5

# Start development services with logs
echo "Starting Ptolemy and Gutenberg services..."
docker-compose up -d ptolemy gutenberg

# Show logs
echo "Development environment is starting..."
echo ""
echo "To view logs, run: docker-compose logs -f"
echo "To access Ptolemy: http://localhost:8000/docs"
echo "To access Gutenberg: http://localhost:8001/docs"
echo ""
echo "Database interfaces:"
echo "- MongoDB: mongodb://localhost:27017"
echo "- Neo4j: http://localhost:7474 (neo4j/password)"
echo "- Qdrant: http://localhost:6333/dashboard"
echo ""
echo "Development environment ready!"