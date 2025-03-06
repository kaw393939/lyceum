# Goliath Educational Platform

A comprehensive, integrated system for educational content creation, knowledge mapping, and interactive learning with AI assistance.

## Project Overview

The Goliath platform consists of four main microservices working together:

1. **Ptolemy**: Knowledge mapping service that organizes educational concepts and their relationships.
2. **Gutenberg**: Content generation service that produces rich educational materials using templates and LLMs.
3. **Galileo**: Graph neural network-based recommendation system for personalized learning paths (in development).
4. **Socrates**: Learner interaction system for content delivery and progress tracking.

## System Architecture

The system is built with a microservices architecture, with each component having its own dedicated responsibility:

- **Ptolemy**: Neo4j graph database + Qdrant vector database for concept storage and retrieval
- **Gutenberg**: FastAPI + MongoDB for content generation and storage with GridFS for media
- **Galileo**: PyTorch + PyG for machine learning recommendation engine (under development)
- **Socrates**: Streamlit app for learner interaction and feedback

## Database Infrastructure

The platform uses multiple specialized databases:

- **MongoDB**: Document storage used by all services for structured data
- **Neo4j**: Graph database for knowledge mapping and concept relationships
- **Qdrant**: Vector database for semantic search and embeddings
- **PostgreSQL**: Relational database for structured data storage (for Galileo)
- **Redis**: Caching layer for improved performance

## Development Setup

### Prerequisites

- Docker and Docker Compose
- Python 3.11+
- Access to OpenAI API (for content generation, optional when using mock mode)

### Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/your-org/goliath.git
   cd goliath
   ```

2. Create `.env` files for each service:
   ```bash
   cp socrates/.env.example socrates/.env
   cp ptolemy/.env.example ptolemy/.env
   cp gutenberg/.env.example gutenberg/.env
   ```

3. Start the platform with ordered services:
   ```bash
   # Use the convenience script for ordered startup
   ./start-services.sh
   
   # Or start everything at once (less reliable)
   docker-compose up -d
   ```

4. Access services:
   - Socrates (Learning Interface): http://localhost:8501
   - Ptolemy API: http://localhost:8000/docs
   - Gutenberg API: http://localhost:8001/docs
   - Neo4j Browser: http://localhost:7474 (neo4j/password)
   - Qdrant Dashboard: http://localhost:6333/dashboard

### Running Tests

Run tests for all components:
```bash
docker-compose run socrates pytest
docker-compose run ptolemy pytest
docker-compose run gutenberg pytest
```

Or test a specific component:
```bash
docker-compose run gutenberg python -m pytest tests/
```

### Troubleshooting Database Connections

If you encounter database health check issues:

1. Use the database restart script:
   ```bash
   # Restart databases without rebuilding
   ./restart-database.sh
   
   # Force rebuild of all database containers
   ./restart-database.sh --rebuild
   ```

2. Check container logs for specific issues:
   ```bash
   docker-compose logs mongodb
   docker-compose logs qdrant
   docker-compose logs neo4j
   ```

3. Verify container status:
   ```bash
   docker-compose ps
   ```

4. For persistent storage issues, you may need to reset volumes:
   ```bash
   # WARNING: This will delete all data!
   docker-compose down -v
   ./start-services.sh
   ```

## Docker Services

The project includes a `docker-compose.yml` file that sets up all required services:

- **ptolemy**: Knowledge mapping API service
- **gutenberg**: Content generation service
- **neo4j**: Graph database for concept relationships
- **mongodb**: Document store for content
- **qdrant**: Vector database for semantic search
- **redis**: Caching service
- **postgres**: Relational database for user data (future use)

## Debugging

For detailed logs:
```bash
docker-compose logs -f
```

To access a specific service's logs:
```bash
docker-compose logs -f gutenberg
```

## Directory Structure

- `/ptolemy`: Knowledge mapping service
- `/gutenberg`: Content generation service
- `/galileo`: Recommendation engine (GNN)
- `/socrates`: Learner interaction system
- `/scripts`: Development and utility scripts

## API Usage Examples

### Ptolemy - Create a Concept

```bash
curl -X POST "http://localhost:8000/api/v1/concepts" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Pythagorean Theorem",
    "description": "Fundamental relation in Euclidean geometry",
    "category": "mathematics",
    "difficulty": "intermediate"
  }'
```

### Gutenberg - Generate Content

```bash
curl -X POST "http://localhost:8001/api/v1/content/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "concept_id": "pythagorean_theorem",
    "content_type": "lesson",
    "difficulty": "intermediate",
    "age_range": "14-18"
  }'
```

## Gutenberg Features

Gutenberg provides comprehensive content generation capabilities:

- **Dynamic Concept Generation:**  
  Uses OpenAI’s models (e.g., GPT-3.5 and GPT-4) to create a structured set of educational concepts with clear definitions, prerequisites, and examples.

- **AI-Powered Analysis:**  
  Determines the ideal number of concepts and estimates topic difficulty. Also performs graph analysis (e.g., prerequisite depths, circular dependencies) to ensure a logical learning progression.

- **Flexible Configuration:**  
  Loads settings from a YAML configuration file (`config.yaml`) with sensible defaults and supports environment variable overrides.

- **Multiple Operation Modes:**  
  - **Single Mode:** Generate concepts for a single domain.
  - **Batch Mode:** Process multiple domains from a JSON file.
  - **Continuous Mode:** Monitor an input directory for new domain files and process them automatically.
  - **API Mode:** Expose endpoints via FastAPI for interactive or integrated usage.

- **Gutenberg Integration:**  
  Optionally export concepts to a Gutenberg-compatible format and trigger content generation callbacks.

- **Extensible CLI:**  
  Offers a rich command-line interface with subcommands and flags to customize output files, generate content automatically, and update integration status.

---

## Architecture and Code Structure

The program is organized into several key sections:

1. **Configuration Classes:**  
   Uses [Pydantic](https://pydantic-docs.helpmanual.io/) models to define configurations for:
   - OpenAI API settings (`OpenAIConfig`)
   - Input/output directories (`InputConfig` and `OutputConfig`)
   - Application server settings (`AppConfig`)
   - Callback and integration parameters (`CallbackConfig` and `IntegrationConfig`)
   - Logging (`LoggingConfig`)

2. **Models:**  
   Defines request and response models for domains, batches, concepts, and analysis using Pydantic. This ensures validation and structured data exchange.

3. **Utility Functions:**  
   Functions for:
   - **Slugification & Filename Generation:** Creating file-friendly names based on the domain and timestamp.
   - **JSON Repair:** Correcting common formatting issues.
   - **Mapping Difficulty Levels:** Converting numeric difficulty to labels (Beginner, Intermediate, Advanced, Expert).

4. **OpenAI Service Functions:**  
   Functions that:
   - Obtain an OpenAI client using an API key.
   - Determine the ideal number of concepts.
   - Extract topics from domain descriptions.
   - Generate a structured set of concepts with prerequisites and enriched descriptions.

5. **Analysis Service Functions:**  
   Includes:
   - Graph analysis to compute average prerequisites, depth, and detection of isolated or circular dependencies.
   - A function to calculate an optimal learning path through the concepts.

6. **Callback and Integration Functions:**  
   Manage communication with the Gutenberg content generator by:
   - Triggering content generation requests with retry logic.
   - Exporting concepts into Gutenberg’s required JSON format.
   - Importing content generation statuses back into the concept data.

7. **CLI Functions:**  
   Supports different operation modes:
   - **Scanning an input directory** for new domain files.
   - **Batch processing** of multiple domain requests.
   - **Continuous mode** to repeatedly monitor and process incoming files.

8. **FastAPI Application:**  
   Exposes endpoints for:
   - Concept generation (`/concepts` and `/concepts/batch`)
   - Domain analysis (`/concepts/analyze-count`, `/analysis/concept-graph`)
   - Content generation triggering and status checks (`/content/generate`, `/content/status/{content_id}`)
   - Gutenberg export (`/export/gutenberg`)
   - Retrieving configuration (`/config`)

9. **Main Function and Argument Parsing:**  
   Uses Python’s `argparse` to handle different CLI modes and options, providing backward compatibility for older command-line invocations.

---

## Configuration

Configuration is managed via a YAML file (`config.yaml`) and environment variables:

- **OpenAI Settings:**  
  - `api_key`: The OpenAI API key (can be set via the environment variable `OPENAI_API_KEY`).
  - `default_model`: Default model (e.g., `"gpt-3.5-turbo-1106"`).
  - `max_retries`: Maximum API call retries.

- **Input/Output Settings:**  
  - `input_dir`: Directory to monitor for new JSON files.
  - `default_output_dir`: Directory where generated concept files are saved.
  - `processed_log`: A log file to track which input files have been processed.

- **App Server Settings:**  
  - `debug`, `host`, `port`, and `reload` options for running the API server.

- **Callback Configuration:**  
  - Options to enable content generation callbacks.
  - `url` for the Gutenberg content generator.
  - `auto_generate_content`, target `age_range`, `content_format`, and retry parameters.

- **Logging:**  
  - Log level, output file, and console logging preferences.

- **Integration Settings:**  
  - Gutenberg-related settings such as `gutenberg_url` and paths to content templates.

The configuration is loaded at startup, and if the file does not exist, a default configuration is created and saved.

---

## Installation and Dependencies

### Requirements

- **Python 3.8+**
- **Dependencies:**
  - `fastapi`
  - `uvicorn`
  - `pydantic`
  - `httpx`
  - `PyYAML`
  - `openai`
  - Additional standard libraries such as `os`, `json`, `argparse`, `datetime`, etc.

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/aristotle.git
   cd aristotle
   ```

2. **Create a virtual environment and install dependencies:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows

   pip install -r requirements.txt
   ```

3. **Set the OpenAI API key (either in the environment or in `config.yaml`):**

   ```bash
   export OPENAI_API_KEY="your_openai_api_key_here"
   ```

---

## Usage

### CLI Modes

The program supports several CLI modes through subcommands:

#### Single Domain Mode

Generate concepts for a single domain:

```bash
python aristotle.py single --domain "Python Programming" \
  --description "An introduction to Python covering basics and fundamentals." \
  --num-concepts 10 --topics "Variables" "Functions" "Loops" \
  --age-range "Adult (25+)" --difficulty "Beginner" --content-format "Mixed media" \
  --auto-name --export-gutenberg --generate-content
```

- **--domain / --description:** Provide the domain name and a brief description.
- **--num-concepts:** Optionally specify the number of concepts (if omitted, AI determines an optimal count).
- **--topics:** Optional list of key topics; if not provided, topics are extracted from the description.
- **--age-range, --difficulty, --content-format:** Specify target audience and content style.
- **--auto-name:** Automatically generate an output filename based on the domain and timestamp.
- **--export-gutenberg:** Export the generated concepts in a Gutenberg-compatible format.
- **--generate-content:** Trigger content generation for the created concepts.

#### Batch Mode

Process multiple domains using a JSON batch file:

```bash
python aristotle.py batch --batch-file domains_batch.json \
  --output-dir ./output --generate-content --export-gutenberg
```

- The batch file should contain a JSON structure with a list of domains and their descriptions.

#### Continuous Mode

Monitor an input directory for new domain files and process them automatically:

```bash
python aristotle.py continuous --input-dir ./input --output-dir ./output \
  --processed-log processed_files.json --interval 60 \
  --generate-content --export-gutenberg
```

- The tool scans the specified input directory every 60 seconds for new JSON files.

#### API Mode

Run the application as a FastAPI web service:

```bash
python aristotle.py api --host 0.0.0.0 --port 8000 --reload \
  --enable-callback --callback-url "http://localhost:8000/generate" \
  --auto-generate --gutenberg-url "http://localhost:8000"
```

- Use this mode to interact with endpoints via HTTP (e.g., using Postman or a frontend application).

#### Integration Mode

Perform Gutenberg-specific integration tasks such as exporting a concept file or updating content status:

```bash
python aristotle.py integration --export concepts.json --output export_gutenberg.json
```

or

```bash
python aristotle.py integration --import-status gutenberg_status.json --update concepts.json
```

---

### API Endpoints

Once running in API mode, the following endpoints are available:

- **GET `/`**  
  Returns a welcome message with the API version.

- **POST `/concepts`**  
  Accepts a domain request (JSON) and returns generated concepts.  
  *Example JSON:*
  ```json
  {
    "domain": "Python Programming",
    "description": "A beginner's introduction to Python.",
    "num_concepts": 10,
    "topics": ["Variables", "Functions"],
    "age_range": "Adult (25+)",
    "difficulty": "Beginner",
    "content_format": "Mixed media"
  }
  ```

- **POST `/concepts/batch`**  
  Processes a batch of domain requests.

- **GET `/concepts/analyze-count`**  
  Analyzes a domain description to recommend an appropriate number of concepts.

- **GET `/concepts/extract-topics`**  
  Extracts key topics from a domain description.

- **POST `/analysis/concept-graph`**  
  Analyzes the generated concept graph for quality, structure, and learning path metrics.

- **POST `/content/generate`**  
  Triggers content generation for specified concept IDs.

- **GET `/content/status/{content_id}`**  
  Checks the status of a content generation job.

- **POST `/export/gutenberg`**  
  Exports concepts to a Gutenberg-compatible JSON format.

- **GET `/config`**  
  Returns the current configuration (with the API key masked).

---

## Integration with Gutenberg

Aristotle supports integration with the Gutenberg content generator:

- **Exporting Concepts:**  
  Concepts can be exported in a format compatible with Gutenberg. The export includes metadata such as the learning path, domain description, and formatted concepts.

- **Content Generation Callback:**  
  When enabled, the system automatically triggers content generation for each concept by sending a POST request to the configured Gutenberg API endpoint. Retry logic is built in to handle transient errors.

- **Status Updates:**  
  The application can also import status updates from Gutenberg to keep track of which concepts have been successfully processed.

---

## Logging and Error Handling

- **Logging:**  
  Configurable logging is set up to output to both a file (default: `aristotle.log`) and the console. The log level is configurable (default: INFO).

- **Error Handling:**  
  - API calls include retry logic and JSON repair functions.
  - When generating or analyzing concepts, the code logs errors and falls back to default values if necessary.
  - In CLI modes, files that fail to process are logged and skipped.

---

## Extending and Customizing

- **Custom Prompts and Models:**  
  The prompts used for concept generation, topic extraction, and analysis can be customized. The application supports switching between GPT models based on domain complexity.

- **Configuration Overrides:**  
  Users can override configuration settings via command-line arguments or environment variables without modifying the source code.

- **Integration Hooks:**  
  The callback functions can be extended to integrate with other content generation engines or learning management systems.

