#!/bin/bash
# Database services maintenance script for Goliath platform
# This script safely restarts database services and rebuilds if needed

set -e

echo "🔄 Goliath Platform Database Maintenance"
echo "----------------------------------------"

# Check if rebuild is requested
REBUILD=false
if [ "$1" == "--rebuild" ]; then
  REBUILD=true
  echo "Rebuild mode enabled - containers will be rebuilt"
fi

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

# Step 1: Stop application services that depend on databases
echo "Stopping application services..."
docker-compose stop socrates ptolemy gutenberg
echo "✅ Application services stopped"

# Step 2: Restart database services
echo "Restarting database services..."

if [ "$REBUILD" = true ]; then
  # Rebuild and start database services
  docker-compose up -d --build --force-recreate mongodb neo4j qdrant
else
  # Just restart database services
  docker-compose restart mongodb neo4j qdrant
fi

# Step 3: Check database health
echo "Checking database health..."
check_health "mongodb" 12 # 2 minute timeout
check_health "neo4j" 12 # 2 minute timeout 
check_qdrant 12 # 2 minute timeout

# Step 4: Restart application services
echo "Restarting application services..."
docker-compose up -d socrates ptolemy gutenberg

# Step 5: Check application health
echo "Checking application services health..."
check_health "ptolemy" 10 # ~1.5 minute timeout
check_health "gutenberg" 10 # ~1.5 minute timeout
check_health "socrates" 6 # ~1 minute timeout

echo "✅ All services restarted and healthy!"
echo
echo "Services available at:"
echo "- Socrates: http://localhost:8501"
echo "- Ptolemy API: http://localhost:8000/docs"
echo "- Gutenberg API: http://localhost:8001/docs"
echo "- Neo4j Browser: http://localhost:7474 (neo4j/password)"
echo "- Qdrant Dashboard: http://localhost:6333/dashboard"