# Enhanced GNN-Based Educational Achievement and Recommendation System

## Table of Contents
1. [Introduction](#introduction)
2. [Key Features](#key-features)
3. [System Architecture](#system-architecture)
4. [Technical Overview](#technical-overview)
5. [Installation](#installation)
6. [Configuration](#configuration)
7. [Usage Guide](#usage-guide)
8. [Command Reference](#command-reference)
9. [Troubleshooting](#troubleshooting)
10. [Additional Resources](#additional-resources)

## Introduction

This system uses Heterogeneous Graph Neural Networks (GNNs) to create an intelligent educational platform that tracks learner progress, provides personalized content recommendations, and facilitates adaptive learning experiences. By modeling the complex relationships between concepts, learners, questions, and resources as a heterogeneous graph, the system can make sophisticated predictions about learner mastery, identify optimal learning paths, and deliver personalized educational experiences.

The system incorporates principles from educational psychology, including spaced repetition, adaptive difficulty, and personalized learning styles, while leveraging modern AI techniques like graph neural networks and large language models.

## Key Features

- **Concept Mastery Tracking**: Track learner progress through interconnected concepts with sophisticated mastery metrics
- **Personalized Recommendations**: Recommend resources, questions, and next concepts based on learner profiles
- **Achievement System**: Award points and achievements based on demonstrated mastery and learning behaviors
- **Interactive Learning Sessions**: Facilitate Socratic-style questioning sessions with AI-generated follow-ups
- **Automated Content Generation**: Generate educational content including questions, resources, and learning objectives
- **Adaptive Learning Paths**: Create personalized learning paths toward mastery of target concepts
- **Weekly Study Planning**: Generate structured study plans with time estimates and daily activities
- **LLM Integration**: Leverage large language models for questioning, content creation, and response evaluation

## System Architecture

The system consists of several integrated components:

- **Knowledge Graph**: Core data structure representing concepts, questions, resources, and learners
- **Heterogeneous GNN Model**: Neural network that learns from the graph structure to make predictions
- **Recommendation System**: Provides personalized content recommendations using the GNN embeddings
- **Achievement System**: Tracks achievements and awards points for demonstrated mastery
- **LLM Service**: Interfaces with OpenAI for content generation and evaluation
- **Content Generator**: Creates educational content using the LLM
- **Command Line Interface**: User interface for interacting with the system

## Technical Overview

### Heterogeneous Graph Neural Network

The system uses a heterogeneous graph with multiple node types:
- **Concepts**: Educational topics with relationships to other concepts
- **Questions**: Assessment items that test understanding of concepts
- **Resources**: Learning materials that teach concepts
- **Learners**: Users who interact with the system

The heterogeneous GNN architecture enables the model to:
1. Learn different embeddings for each node type
2. Handle different feature dimensions for each node and edge type
3. Process specialized relationships between different entity types
4. Integrate relation-specific attention mechanisms

Key technical aspects:
- Utilizes PyTorch Geometric's heterogeneous graph capabilities
- Implements custom message passing for heterogeneous graphs
- Employs multi-headed attention for relationship weighting
- Handles complex edge features for representing relationship attributes
- Maintains incremental graph updates to efficiently handle changes

### Recommendation System

The recommendation system balances exploration vs. exploitation through:
- Thompson sampling for exploration-exploitation trade-offs
- Personalization based on learning styles and preferences
- Media type matching to learner preferences
- Difficulty-appropriate content selection
- Diversity promotion in recommendations
- Concept prerequisite awareness
- Spaced repetition principles for review suggestions

### Achievement Mechanism

The system includes a sophisticated achievement mechanism that:
- Awards achievements for concept mastery
- Recognizes connected concept understanding
- Rewards deep reasoning abilities
- Acknowledges learning persistence
- Adjusts point values based on difficulty
- Provides streak bonuses for consistent engagement

## Installation

### Prerequisites

- Python 3.8+ (3.10 or 3.11 recommended)
- PyTorch 1.12+
- CUDA drivers (optional, for GPU acceleration)

### Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/yourusername/gnn-education-system.git
cd gnn-education-system
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
# Install PyTorch (adjust based on your CUDA requirements)
pip install torch torchvision torchaudio

# Install PyTorch Geometric and its dependencies
pip install torch-geometric
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv

# Install other dependencies
pip install scikit-learn python-dotenv tqdm networkx openai pyyaml
```

4. Set up environment variables:
Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Configuration

The system uses a YAML configuration file (`config.yaml`) for customization. Key configuration sections include:

```yaml
knowledge_graph:
  concepts_file: "data/concepts.json"
  questions_file: "data/questions.json"
  resources_file: "data/resources.json"
  learners_file: "data/learners.json"
  cache_dir: "data/cache"

gnn:
  model_type: "hetero_gat"  # Options: hetero_gat, hetero_sage, hetero_gcn
  hidden_channels: 64
  num_layers: 2
  num_heads: 4
  dropout: 0.2
  learning_rate: 0.001

achievements:
  mastery_threshold: 0.75
  points_multiplier: 1.0
  difficulty_bonus: 0.5
  streak_bonus: 0.2

recommendation:
  exploration_weight: 0.3
  recency_decay: 0.9
  diversity_factor: 0.2
  personalization_weight: 0.7

llm:
  model: "gpt-3.5-turbo"
  temperature: 0.7
  max_tokens: 200
  cache_responses: true
```

## Usage Guide

### Initial Setup

1. Initialize the system:
```bash
python gnn_education_system.py init
```

2. Add example data for testing:
```bash
python gnn_education_system.py init --example-data
```

3. Import concepts from a JSON file:
```bash
python gnn_education_system.py import concepts.json
```

### Training the Model

Train the GNN model on the knowledge graph:
```bash
python gnn_education_system.py model train --epochs 100
```

### Working with Learners

1. Add a new learner:
```bash
python gnn_education_system.py add-learner --name "John Doe" --email "john@example.com" --learning-style visual
```

2. Get learner statistics:
```bash
python gnn_education_system.py learner stats learner123
```

3. Generate recommendations for a learner:
```bash
python gnn_education_system.py learner recommend learner123 --type concepts
```

4. Create a learning path:
```bash
python gnn_education_system.py learner path learner123 --target concept456
```

5. Generate a weekly study plan:
```bash
python gnn_education_system.py learner plan learner123
```

6. Start an interactive learning session:
```bash
python gnn_education_system.py learner session learner123 --concept-id concept789
```

### Managing Content

1. Add a new concept:
```bash
python gnn_education_system.py add-concept --name "Calculus" --description "Study of continuous change" --difficulty 0.7
```

2. Add a concept with auto-generated content:
```bash
python gnn_education_system.py add-concept --name "Python Basics" --description "Fundamental programming concepts in Python" --generate-content
```

3. Generate content for existing concept:
```bash
python gnn_education_system.py concept generate-content concept123
```

4. View concept statistics:
```bash
python gnn_education_system.py concept stats concept123
```

## Command Reference

### Entity Management
- `init`: Initialize the system
- `add-concept`: Add a new concept
- `add-learner`: Add a new learner
- `import`: Import concepts from a JSON file

### Listing and Searching
- `list`: List entities of a specified type
- `search`: Search for entities

### Learner Operations
- `learner stats`: Show learner statistics
- `learner recommend`: Get recommendations for a learner
- `learner path`: Generate a learning path
- `learner plan`: Generate a weekly study plan
- `learner session`: Start interactive learning session
- `learner update`: Update learner attributes

### Concept Operations
- `concept stats`: Show concept statistics
- `concept update`: Update concept attributes
- `concept generate-content`: Generate content for a concept

### Model Operations
- `model train`: Train the GNN model
- `model save`: Save the trained model
- `model load`: Load a saved model

### Achievement Operations
- `achievement check`: Check for new achievements

### Cache Operations
- `cache clear`: Clear the node embedding cache

## Example Learning Session

The system supports interactive learning sessions with AI-generated questions and feedback:

```bash
python gnn_education_system.py learner session learner123 --concept-id python101
```

This will start a session like:

```
=== Interactive Session for John Doe ===
Concept: Python Basics
Current Mastery: 35.0%

Learning Objectives:
  1. Explain the fundamental data types in Python including integers, floats, strings, and booleans
  2. Write basic Python programs using variables, conditionals, and loops
  3. Implement simple functions with parameters and return values

Recommended Resources:
  1. Python for Beginners: A Complete Guide
     URL: https://www.python.org/about/gettingstarted/

=== Starting Interactive Q&A ===
================================================================================
Question: How would you explain the difference between mutable and immutable data types in Python, and why is this distinction important?
================================================================================
Type your answer below, or type 'quit' to end the session.

Your answer: Mutable types like lists can be changed after creation, while immutable types like strings and tuples cannot be modified once created. This is important because it affects how data is stored and passed to functions.

--------------------------------------------------------------------------------
Evaluation:
  Correctness: 90/100
  Reasoning: 85/100
  Overall Score: 87.5/100

Strengths:
You've correctly identified the core distinction between mutable and immutable types and provided good examples of each category.

Feedback:
Your answer shows a solid understanding of mutability in Python. To make your answer more complete, you could mention how this affects the behavior in function calls (pass by reference vs pass by value behavior) and perhaps touch on how this relates to memory management and performance.

🏆 Achievement Unlocked! 🏆
  +75 points: Earned Deep Reasoning: Demonstrated sophisticated reasoning skills (Python Basics)

Mastery increased by 5.2%!
New mastery level: 40.2%

================================================================================
Next Question: Can you provide an example of a situation where you might choose to use a tuple instead of a list in Python, and explain your reasoning?
================================================================================
```

## Data Structure

### Knowledge Graph Format

The system stores data in JSON files:

- **concepts.json**: Educational concepts and their relationships
- **questions.json**: Assessment questions linked to concepts
- **resources.json**: Learning resources tied to concepts
- **learners.json**: User data including mastery levels and achievements

### Concept Schema Example

```json
{
  "concepts": [
    {
      "id": "concept123",
      "name": "Python Variables",
      "description": "Variables in Python and how they store data.",
      "difficulty": 0.3,
      "complexity": 0.4,
      "importance": 0.9,
      "prerequisites": ["concept111", "concept112"]
    }
  ]
}
```

## Troubleshooting

### Common Issues

1. **Embedding Cache Issues**
   - Symptom: Recommendations aren't updating properly
   - Solution: Clear the cache with `python gnn_education_system.py cache clear`

2. **Model Training Issues**
   - Symptom: Training fails with GPU errors
   - Solution: Try CPU-only training by setting `device: cpu` in config.yaml

3. **OpenAI API Issues**
   - Symptom: Content generation fails
   - Solution: Check your API key in .env file and ensure proper API access

4. **Missing Node Types**
   - Symptom: "Missing node type" errors during model training
   - Solution: Ensure your knowledge graph has all required node types (concepts, questions, resources, learners)

## Additional Resources

- The system builds on research from:
  - Graph Neural Networks for educational recommendation
  - Heterogeneous graph representation learning
  - Educational psychology principles for adaptive learning
  - Spaced repetition and mastery learning models

## Using the GNN System in Research

This system can be used as a research platform for:
- Evaluating personalized learning approaches
- Testing achievement and motivation frameworks
- Developing new graph-based recommendation algorithms
- Studying knowledge graph representations for education

---
