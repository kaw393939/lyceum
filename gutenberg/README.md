# Gutenberg Content Generation System

Gutenberg is an advanced educational content generation system that creates rich, structured educational materials using knowledge maps from the Ptolemy system. It uses modern LLM techniques and template systems to transform knowledge structures into coherent, pedagogically sound educational content.

## Key Features

- **Template-Based Content Generation**: Create consistent educational content using customizable templates
- **Stoic Principles Integration**: Content includes Stoic philosophical principles to promote resilience and critical thinking
- **Rich Media Support**: Generate and serve images, diagrams, audio, and interactive elements
- **Educational Assessment Creation**: Automatically create assessments aligned with learning objectives
- **Integration with Knowledge Graphs**: Connect to Ptolemy system for concept mapping
- **Personalization Options**: Adapt content for different age groups, difficulty levels, and learning paths
- **Learner Feedback System**: Collect, analyze, and respond to detailed user feedback
- **Complete BREAD Operations**: Browse, Read, Edit, Add, Delete functionality for all content
- **Media Storage with GridFS**: Store and serve images and audio using MongoDB GridFS
- **Advanced Analytics**: Analyze feedback to improve content quality

## System Architecture

Gutenberg is built using a modern, modular architecture:

- **FastAPI Backend**: High-performance API with async support
- **Template Engine**: Processes structured templates with dynamic content generation
- **LLM Integration**: Uses OpenAI and Anthropic models for content creation
- **Content Storage**: MongoDB for storing generated content
- **Media Management**: GridFS for storing and serving media files (images, audio)
- **Vector Search**: Qdrant for semantic search capabilities
- **Ptolemy Integration**: Client for connecting to the knowledge graph system
- **Feedback System**: Collection and analysis of learner feedback
- **Media Generation**: Create images, diagrams, charts and audio for educational content
- **BREAD Operations**: Complete set of RESTful APIs for content management
- **Feedback Analytics**: Process and analyze feedback data for content improvement

## Getting Started

### Prerequisites

- Python 3.12+
- MongoDB
- Docker (optional, for containerized deployment)
- Ptolemy Knowledge Graph System (for full functionality)

### Installation

1. Clone the repository
2. Run the setup script to create required directories and environment:
   ```
   ./create.sh
   ```

3. Configure the environment variables in the `.env` file:
   ```
   # API Keys and Secrets
   OPENAI_API_KEY=your_key_here
   PTOLEMY_API_KEY=your_key_here
   
   # Database connections
   MONGODB_URI=mongodb://mongodb:27017/gutenberg
   ```

4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

5. Start the service:
   ```
   python main.py
   ```

### Docker Deployment

Gutenberg includes a multi-stage Dockerfile for both development and production environments:

```bash
# Development mode
docker build -t gutenberg:dev --target development .
docker run -p 8001:8001 -v $(pwd):/app gutenberg:dev

# Production mode
docker build -t gutenberg:latest --target production .
docker run -p 8001:8001 gutenberg:latest
```

Alternatively, use docker-compose to start the entire educational system:

```bash
docker-compose up -d
```

## Content Generation Process

The content generation follows these steps:

1. **Request Submission**: Client sends topic, template ID, and parameters
2. **Knowledge Retrieval**: System fetches concept data from Ptolemy
3. **Template Selection**: The appropriate content template is selected
4. **Content Generation**: LLM processes generate structured content using the template
5. **Media Generation**: Images, diagrams, and interactive elements are created
6. **Assessment Creation**: Learning assessments are generated
7. **Content Storage**: Final content is stored and delivered to the client

## API Reference

Gutenberg exposes a RESTful API:

### Content Management (BREAD)
- **GET /api/v1/content**: Browse all content with pagination and filtering
- **GET /api/v1/content/{content_id}**: Read specific content
- **POST /api/v1/content/generate**: Add new content
- **PATCH /api/v1/content/{content_id}**: Edit content properties
- **DELETE /api/v1/content/{content_id}**: Delete content

### Templates
- **GET /api/v1/templates**: List available templates
- **GET /api/v1/templates/{template_id}**: Get specific template
- **POST /api/v1/templates**: Create new template
- **PUT /api/v1/templates/{template_id}**: Update template
- **DELETE /api/v1/templates/{template_id}**: Delete template

### Feedback
- **POST /api/v1/feedback**: Submit feedback on content
- **GET /api/v1/feedback/{feedback_id}**: Get specific feedback
- **GET /api/v1/feedback/content/{content_id}**: Get all feedback for content
- **GET /api/v1/feedback/summary/{content_id}**: Get feedback summary for content
- **PATCH /api/v1/feedback/{feedback_id}/status**: Update feedback status

### Media
- **POST /api/v1/media/generate**: Generate media (images, audio, etc.)
- **POST /api/v1/media/upload**: Upload media files
- **GET /api/v1/media/{media_id}**: Retrieve media
- **GET /api/v1/media**: List all media
- **DELETE /api/v1/media/{media_id}**: Delete media

### System
- **GET /health**: System health check
- **GET /api/v1/info**: System information

## Customizing Templates

Gutenberg uses a flexible template system defined in JSON. Templates include:

- Structured sections (introduction, explanation, assessment, etc.)
- Prompts for LLM generation
- Media specifications
- Interactive element definitions
- Placeholder definitions for content personalization

Example template structure:
```json
{
  "template_id": "default",
  "name": "Default Content Template",
  "sections": [
    {
      "id": "introduction",
      "title": "Introduction",
      "prompts": [
        {
          "prompt_text": "Write an introduction about [concept_name]...",
          "placeholders": [{"name": "concept_name"}]
        }
      ]
    }
  ]
}
```

## Testing and Performance

Run the test suite to verify functionality:
```
pytest
```

Run benchmark tests to measure performance:
```
./benchmark.sh
```

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b new-feature`
3. Commit changes: `git commit -am 'Add new feature'`
4. Push to branch: `git push origin new-feature`
5. Submit a pull request