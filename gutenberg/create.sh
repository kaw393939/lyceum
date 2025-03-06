#!/bin/bash
# Gutenberg Project Setup Script
# This script will set up the Gutenberg project with all necessary files and directories

# Ensure script is run from the gutenberg directory
if [[ $(basename "$PWD") != "gutenberg" ]]; then
  echo "Please run this script from the gutenberg directory"
  exit 1
fi

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p concept_imports
mkdir -p generated_content
mkdir -p storage/reports
mkdir -p templates/specialized
mkdir -p tmp

# Check if .env file exists, create it if not
if [ ! -f .env ]; then
  echo "Creating .env file..."
  cat > .env << EOF
# Gutenberg Content Generation System Environment Variables
# API Keys and Secrets (Replace with your actual keys)
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
PTOLEMY_API_KEY=your_ptolemy_key_here

# Database connections
MONGODB_URI=mongodb://mongodb:27017/gutenberg
QDRANT_URL=http://qdrant:6333

# Services
PTOLEMY_URL=http://ptolemy:8000
EOF
  echo ".env file created. Please edit it with your actual API keys."
else
  echo ".env file already exists."
fi

echo "Setting up environment..."

# Check if Python virtual environment exists, create if not
if [ ! -d "venv" ]; then
  echo "Creating Python virtual environment..."
  python -m venv venv
  source venv/bin/activate
  echo "Installing dependencies..."
  pip install -r requirements.txt
else
  echo "Virtual environment already exists."
  source venv/bin/activate
  echo "Updating dependencies..."
  pip install -r requirements.txt
fi

echo "Gutenberg setup complete!"
echo "To activate the virtual environment: source venv/bin/activate"
echo "To run the server: python main.py"
echo "To run tests: pytest"

# Make script executable
chmod +x "$0"