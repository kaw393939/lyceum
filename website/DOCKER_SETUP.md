# Docker Development Setup for Lyceum

This document explains the Docker Compose setup created for Lyceum development with live-reloading.

## Key Components

### 1. Docker Compose Configuration (`docker-compose.yml`)

```yaml
version: '3.8'

services:
  lyceum:
    build:
      context: .
      dockerfile: Dockerfile.dev
    container_name: lyceum_web
    ports:
      - "8080:8080"
    volumes:
      - .:/app
      - node_modules:/app/node_modules
    environment:
      - PYTHONDONTWRITEBYTECODE=1
      - PYTHONUNBUFFERED=1
      - WATCHDOG_TIMEOUT=5
    tty: true
    stdin_open: true
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s
    restart: unless-stopped

volumes:
  node_modules:
```

This configuration:
- Builds from the `Dockerfile.dev` development-specific Dockerfile
- Maps port 8080 from the container to the host
- Mounts the current directory to `/app` in the container for live code updates
- Sets up a health check to ensure the server is running
- Configures restart policy and environment variables

### 2. Development Dockerfile (`Dockerfile.dev`)

```Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages needed for development
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install watchdog[watchmedo] pytest pillow

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Expose the port the server runs on
EXPOSE 8080

# Copy the entrypoint script
COPY docker-entrypoint.sh /app/
RUN chmod +x /app/docker-entrypoint.sh

# Use an entrypoint script to handle initialization and live reloading
ENTRYPOINT ["/app/docker-entrypoint.sh"]
```

This Dockerfile:
- Uses Python 3.12 slim as the base image
- Installs required dependencies including curl for health checks
- Sets up Python packages for development, including watchdog for file monitoring
- Exposes port 8080 for the web server
- Uses a custom entrypoint script for initialization and monitoring

### 3. Entrypoint Script (`docker-entrypoint.sh`)

```bash
#!/bin/bash
set -e

echo "Starting Lyceum Development Environment"
echo "======================================="

# Generate fallback assets if needed
echo "Step 1: Creating necessary assets..."
python regenerate_assets.py --fallback --update-templates

# Create latest links
echo "Step 2: Creating latest asset links..."
python create_latest_links.py

echo "Step 3: Starting development server with live reloading..."
echo "Website will be available at http://localhost:8080"
echo "Any file changes will automatically reload the server"
echo "======================================="

# Use watchdog to monitor for changes and restart the server
# This setup allows for live reloading when files change
exec watchmedo auto-restart \
    --directory=/app \
    --pattern="*.py;*.html;*.css;*.js" \
    --recursive \
    --kill-after=${WATCHDOG_TIMEOUT:-5} \
    -- python serve.py --host=0.0.0.0 --port=8080
```

This script:
- Initializes the development environment
- Generates fallback assets and creates latest links
- Starts the server with watchdog for file monitoring
- Automatically restarts the server when files change

### 4. Server Modifications for Live Reloading

We modified the serve.py file to:
- Add proper Last-Modified headers to all responses
- Include cache control headers to ensure fresh content
- Calculate modification times based on file timestamps
- Serve static files with proper MIME types and headers

### 5. Client-Side Hot Reload

We added JavaScript to the base template that:
- Checks for server changes every 2 seconds
- Reloads the page when changes are detected
- Shows a notification when updates occur
- Works with all modern browsers

### 6. Dependencies Installation Script

The `install_dependencies.sh` script helps users set up the required dependencies:
- Installs Python packages needed for development
- Checks for Docker and Docker Compose installation
- Provides guidance on how to run the application

## Using the Development Environment

### Starting the Environment

```bash
docker-compose up
```

### Stopping the Environment

```bash
docker-compose down
```

### Rebuilding After Dependency Changes

```bash
docker-compose build --no-cache
docker-compose up
```

## Development Workflow

1. Make changes to source files (Python, HTML, CSS, JavaScript)
2. The server automatically detects changes and restarts
3. The browser detects the server restart and refreshes
4. You see your changes immediately

This workflow significantly speeds up development by eliminating manual restarts and refreshes.

## Troubleshooting

If live reloading isn't working:

1. Check container logs: `docker-compose logs -f`
2. Ensure file permissions allow read/write from the container
3. Verify that the watchdog service is running
4. Try manually restarting with: `docker-compose restart`