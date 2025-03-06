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