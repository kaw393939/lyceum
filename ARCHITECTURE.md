# Goliath Platform Architecture

This document provides a textual description of the Goliath Educational Platform architecture. This can be used to create proper diagrams using tools like draw.io, Mermaid, or PlantUML.

## System Overview

```
┌───────────────┐     ┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│               │     │               │     │               │     │               │
│    Ptolemy    │◄────►   Gutenberg   │◄────►    Galileo    │◄────►    Socrates   │
│ (Knowledge Map)│     │(Content Gen)  │     │  (Learning    │     │  (Learner     │
│               │     │               │     │   Paths)      │     │  Interaction) │
└───────┬───────┘     └───────┬───────┘     └───────┬───────┘     └───────┬───────┘
        │                     │                     │                     │
        │                     │                     │                     │
┌───────┴───────┐     ┌───────┴───────┐     ┌───────┴───────┐     ┌───────┴───────┐
│     Neo4j     │     │    MongoDB    │     │  PostgreSQL   │     │     Redis     │
│  (Graph DB)   │     │  (Document DB)│     │ (Relational DB)│     │   (Cache)    │
└───────────────┘     └───────────────┘     └───────────────┘     └───────────────┘
        ▲                     ▲
        │                     │
        │                     │
┌───────┴───────┐     ┌───────┴───────┐
│    Qdrant     │     │   MongoDB     │
│  (Vector DB)  │     │   GridFS      │
└───────────────┘     └───────────────┘
```

## Component Details

### Ptolemy (Knowledge Mapping Service)
- **Purpose**: Manages knowledge graphs and concept relationships
- **Key Components**:
  - Neo4j Service: Graph database interactions
  - Knowledge Manager: Manages concept relationships
  - Embedding Service: Creates vector embeddings for concepts
  - LLM Integration: Enriches knowledge graph
- **APIs**:
  - Concept management
  - Relationship creation
  - Knowledge search
  - Vector similarity search

### Gutenberg (Content Generation Service)
- **Purpose**: Generates educational content from templates
- **Key Components**:
  - Template Engine: Processes educational templates
  - Content Generator: Creates educational materials
  - Media Generator: Creates accompanying media
  - RAG Processor: Retrieval-augmented generation
- **APIs**:
  - Template management
  - Content generation
  - Media generation
  - Feedback collection

### Galileo (Learning Path Recommendation)
- **Purpose**: Creates personalized learning paths
- **Key Components**:
  - GNN Education System: Graph neural network model
  - Learning Path Generator: Creates tailored paths
  - Recommendation Engine: Suggests next concepts
  - Learner Model: Tracks learner progress
- **APIs**:
  - Learner profile management
  - Path recommendations
  - Progress tracking

### Socrates (Learner Interaction)
- **Purpose**: User-facing interface for learners
- **Key Components**:
  - Chat Service: Interactive learning
  - RAG Service: Enhanced responses
  - Streamlit UI: Web interface
  - Diagnostic Tools: System monitoring
- **APIs**:
  - Chat interactions
  - Progress visualization
  - Content presentation

## Data Flow

1. **Knowledge Creation**:
   - Ptolemy creates a knowledge graph of educational concepts
   - Concepts are enriched with LLM-generated metadata
   - Vector embeddings enable semantic search

2. **Content Generation**:
   - Gutenberg takes templates and concepts from Ptolemy
   - Templates are processed with LLM to create content
   - Media generation adds visual elements
   - Content is stored in MongoDB with GridFS

3. **Learning Path Creation**:
   - Galileo analyzes the knowledge graph from Ptolemy
   - Learner profiles inform path recommendations
   - GNN creates personalized learning sequences
   - Recommendations are updated based on learner progress

4. **Learner Interaction**:
   - Socrates presents content from Gutenberg
   - Learner follows paths recommended by Galileo
   - Interactive elements use RAG for enhanced responses
   - Progress is tracked and fed back to other services

## Deployment Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Docker Compose                          │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │   Ptolemy   │  │  Gutenberg  │  │   Galileo   │  │   Socrates  │ │
│  │  Container  │  │  Container  │  │  Container  │  │  Container  │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │    Neo4j    │  │   MongoDB   │  │ PostgreSQL  │  │    Redis    │ │
│  │  Container  │  │  Container  │  │  Container  │  │  Container  │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │
│                                                             │
│  ┌─────────────┐                                            │
│  │   Qdrant    │                                            │
│  │  Container  │                                            │
│  └─────────────┘                                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Communication Patterns

- **Service Discovery**: Direct container addressing
- **Communication Protocol**: REST APIs with JSON
- **Authentication**: API keys for service-to-service
- **Error Handling**: Consistent error responses with codes
- **Circuit Breaking**: Retry mechanisms with exponential backoff

## Scaling Considerations

- **Stateless Services**: Ptolemy, Gutenberg can scale horizontally
- **Database Scaling**: 
  - MongoDB: Sharding for document storage
  - Neo4j: Read replicas for knowledge graph
  - Qdrant: Distributed vector search
- **Cache Layer**: Redis for performance optimization