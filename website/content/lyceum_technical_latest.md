# Lyceum Technical Architecture

## Core Architecture Components

```
                    ┌───────────────┐
                    │ User Interface│
                    │ Web, Mobile,  │
                    │ AR/VR         │
                    └───────┬───────┘
                            │
┌───────────────────────────┴──────────────────────────┐
│                 API Gateway / Service Mesh            │
│                      (Linkerd)                        │
└───────────────────────────┬──────────────────────────┘
                            │
┌───────────────────────────┴──────────────────────────┐
│         Agent Services on Kubernetes (K3s)            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐ │
│  │ Socrates │  │ Ptolemy  │  │Gutenberg │  │Hypatia │ │
│  │ Dialogue │  │Knowledge │  │ Content  │  │Character│ │
│  └──────────┘  └──────────┘  └──────────┘  └────────┘ │
│                                                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐ │
│  │Aristotle │  │Alexandria│  │  Hermes  │  │ Other  │ │
│  │Assessment│  │Knowledge │  │ Rewards  │  │ Agents │ │
│  └──────────┘  └──────────┘  └──────────┘  └────────┘ │
└───────────────────────────┬──────────────────────────┘
                            │
┌───────────────────────────┴──────────────────────────┐
│           Intelligent Core & Event Streaming          │
│      ┌─────────────────┐      ┌──────────────┐       │
│      │  Galileo GNN    │◄────►│  Kafka       │       │
│      │  Intelligence   │      │  Streaming   │       │
│      └─────────────────┘      └──────────────┘       │
└───────────────────────────┬──────────────────────────┘
                            │
┌───────────────────────────┴──────────────────────────┐
│                  Storage Layer                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐ │
│  │  Neo4j   │  │  Qdrant  │  │  MinIO   │  │Postgres│ │
│  │  Graph   │  │  Vector  │  │  Object  │  │Relation│ │
│  └──────────┘  └──────────┘  └──────────┘  └────────┘ │
└───────────────────────────┬──────────────────────────┘
                            │
┌───────────────────────────┴──────────────────────────┐
│                Workflow Orchestration                 │
│                      (Airflow)                        │
└───────────────────────────┬──────────────────────────┘
                            │
┌───────────────────────────┴──────────────────────────┐
│               LLM Inference Infrastructure            │
│           (Claude API + Open Source Models)           │
└────────────────────────────────────────────────────┘
```

## Key Technical Components

### 1. Multi-Agent System

Lyceum employs specialized AI agents, each with distinct responsibilities:

- **Socrates**: Dialogue and user interface agent that manages conversations
- **Ptolemy**: Knowledge graph agent that maintains concept relationships
- **Gutenberg**: Content creation agent that generates personalized materials
- **Galileo**: Analytics agent using GNN to understand learning patterns
- **Alexandria**: Knowledge extraction agent that interfaces with LLMs
- **Aristotle**: Assessment agent that evaluates learning
- **Hypatia**: Character simulation agent that embodies historical figures
- **Hermes**: Rewards agent that manages incentive systems

These agents operate as containerized microservices orchestrated by Kubernetes, allowing independent scaling based on demand.

### 2. Intelligent Core: Galileo GNN + Kafka

The system's "neural network" consists of:

- **Galileo's GNN (Graph Neural Network)**: A heterogeneous graph neural network that models complex relationships between users, content, concepts, and interactions. This network learns patterns of effective learning and provides intelligent routing and recommendations.

- **Kafka Event Streaming**: A distributed event streaming platform that connects all components, providing reliable message delivery and event sourcing. Kafka topics are organized by agent and event type, creating a comprehensive activity log.

This combination creates a distributed intelligence that learns from system interactions and continuously improves performance.

### 3. Storage Layer

Diverse data requirements are met through specialized storage systems:

- **Neo4j**: Graph database for storing knowledge relationships and concept connections
- **Qdrant**: Vector database for semantic search and similarity matching
- **MinIO**: S3-compatible object storage for educational content and media
- **PostgreSQL**: Relational database for structured data and transactional records

### 4. Infrastructure Automation

The entire infrastructure is defined as code and managed through:

- **Terraform**: Infrastructure-as-code tool that defines all cloud resources
- **Kubernetes (K3s)**: Lightweight container orchestration for agent deployment
- **Linkerd**: Service mesh providing observability and communication control
- **Airflow**: Workflow orchestration for complex educational processes

### 5. Monitoring & Analytics

Comprehensive observability is provided by:

- **Prometheus**: Metrics collection and alerting
- **Grafana**: Visualization dashboards for system and educational analytics
- **Jaeger**: Distributed tracing for transaction flows
- **ELK Stack**: Log aggregation and analysis

## Technical Differentiators

1. **Distributed Intelligence Model**: Rather than a centralized AI, Lyceum uses a distributed network of specialized agents coordinated through an intelligent GNN core.

2. **Event-Sourced Architecture**: All system activities are captured as events in Kafka, enabling complete reconstruction of state and robust analytics.

3. **Knowledge Graph Foundation**: Unlike flat content structures, Lyceum's knowledge is represented as a rich graph of interconnected concepts.

4. **Heterogeneous GNN**: The system uses a specialized graph neural network that can model diverse entity types and relationships to learn complex educational patterns.

5. **Open Source Scale**: Built primarily with open source components, enabling cost-effective deployment that can scale from small pilots to large implementations.