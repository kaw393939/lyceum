#!/bin/bash
# Service startup script for Goliath platform
# This script starts the database services first, then the application services

set -e

echo "Starting Goliath Platform Services..."

# Function to check if a container is healthy
check_health() {
  local service=$1
  local max_attempts=$2
  local attempt=1
  
  echo "Waiting for $service to be healthy..."
  
  while [ $attempt -le $max_attempts ]; do
    health_status=$(docker inspect --format='{{.State.Health.Status}}' plato-$service-1 2>/dev/null || echo "notfound")
    
    if [ "$health_status" = "healthy" ]; then
      echo "✅ $service is healthy!"
      return 0
    elif [ "$health_status" = "notfound" ]; then
      echo "❌ $service container not found. Check if the service name is correct."
      return 1
    fi
    
    echo "Attempt $attempt/$max_attempts: $service status is $health_status. Waiting 10 seconds..."
    sleep 10
    ((attempt++))
  done
  
  echo "❌ $service failed to become healthy after $max_attempts attempts."
  return 1
}

# Special function for Qdrant which doesn't have a proper healthcheck
check_qdrant() {
  local max_attempts=$1
  local attempt=1
  
  echo "Waiting for Qdrant to be available..."
  
  # Wait 30 seconds initially for Qdrant to start
  sleep 30
  
  while [ $attempt -le $max_attempts ]; do
    if curl -s http://localhost:6333 > /dev/null; then
      echo "✅ Qdrant is available!"
      return 0
    fi
    
    echo "Attempt $attempt/$max_attempts: Qdrant not ready yet. Waiting 10 seconds..."
    sleep 10
    ((attempt++))
  done
  
  echo "❌ Qdrant failed to become available after $max_attempts attempts."
  return 1
}

# Step 1: Start essential database services first
echo "Starting database services..."
docker-compose up -d mongodb neo4j
docker-compose up -d qdrant

# Step 2: Check health of database services
echo "Checking database health..."
check_health "mongodb" 12 # 2 minute timeout
check_health "neo4j" 12 # 2 minute timeout 
check_qdrant 12 # 2 minute timeout

# Step 3: Start application services
echo "Starting application services..."
docker-compose up -d ptolemy gutenberg

# Step 4: Check health of core services
echo "Checking core services health..."
check_health "ptolemy" 10 # ~1.5 minute timeout
check_health "gutenberg" 10 # ~1.5 minute timeout

# Step 5: Start the frontend
echo "Starting Socrates frontend..."
docker-compose up -d socrates

# Step 6: Final check
echo "Checking final service health..."
check_health "socrates" 6 # ~1 minute timeout

echo "All services started successfully!"
echo
echo "Services available at:"
echo "- Socrates: http://localhost:8501"
echo "- Ptolemy API: http://localhost:8000/docs"
echo "- Gutenberg API: http://localhost:8001/docs"
echo "- Neo4j Browser: http://localhost:7474 (neo4j/password)"
echo "- Qdrant Dashboard: http://localhost:6333/dashboard"
echo
echo "To view logs: docker-compose logs -f"