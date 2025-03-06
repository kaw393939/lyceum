#!/bin/bash
# Run the Streamlit application with proper environment setup

# Set environment variables for services - they default to mock mode
export OPENAI_API_KEY=${OPENAI_API_KEY:-"sk-your-api-key"}
export MODEL=${MODEL:-"gpt-4o"}
export PTOLEMY_URL=${PTOLEMY_URL:-"http://ptolemy:8000"}
export GUTENBERG_URL=${GUTENBERG_URL:-"http://gutenberg:8001"}
export PTOLEMY_USE_MOCK=${PTOLEMY_USE_MOCK:-"true"}
export GUTENBERG_USE_MOCK=${GUTENBERG_USE_MOCK:-"true"}

# Use Docker service discovery in Docker environments
if [ -n "$DOCKER_ENV" ]; then
  export PTOLEMY_URL="http://ptolemy:8000"
  export GUTENBERG_URL="http://gutenberg:8001"
fi

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
  source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
  source venv/Scripts/activate
fi

# Ensure src package is in PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# Run streamlit application
streamlit run app.py "$@"
