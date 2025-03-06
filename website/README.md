# Lyceum Educational System

## Overview

The Lyceum is a modern reinterpretation of ancient Greek learning centers, designed to facilitate deep learning through dialogue, exploration, and structured knowledge paths. This system extends the existing Goliath Educational Platform with specialized components focused on dialectic learning, mentorship, and philosophical inquiry.

## Documentation

This folder contains comprehensive design and planning documents for the Lyceum system:

- [System Design](SYSTEM_DESIGN.md): Detailed architecture and component design
- [Implementation Plan](IMPLEMENTATION_PLAN.md): Phased implementation approach and timeline

## Interactive Design System

This folder also contains an interactive design system to visualize the Lyceum architecture and components:

```
/lyceum
├── index.html                   # Main visualization interface
├── static/                      # Static assets
│   ├── css/                     # Stylesheets
│   │   └── main.css             # Main CSS file
│   ├── js/                      # JavaScript
│   │   ├── main.js              # Main application code
│   │   └── visualizer.js        # Visualization engine
│   └── images/                  # Images and icons
├── templates/                   # HTML templates
└── diagrams/                    # Source diagrams
```

## Visualization Features

The interactive visualization system provides:

1. **System Architecture Diagrams**: Flow charts showing system components and their relationships
2. **Data Flow Visualizations**: Sequence diagrams illustrating data flow between components
3. **Component Visualizations**: Interactive exploration of individual components
4. **Roadmap Timeline**: Visual representation of the development roadmap

## Key System Components

The Lyceum system consists of five primary components:

1. **Core System**: Central orchestration layer
2. **Knowledge Graph**: Connected network of concepts
3. **Dialogue System**: Socratic conversation facilitation
4. **Content Engine**: Adaptive learning material generation
5. **Mentor Service**: Personalized guidance and intervention

## Design Principles

The system is built on these core philosophical principles:

1. **Dialectic Learning**: Knowledge through structured dialogue
2. **Interconnected Knowledge**: Web of understanding vs isolated facts
3. **Guided Discovery**: Exploration with mentorship
4. **Virtue-Oriented Education**: Character development alongside knowledge
5. **Community of Practice**: Learning in communities of shared inquiry

## Technical Implementation

The Lyceum system integrates with the existing Goliath Educational Platform:

| Lyceum Component | Integrates With | Integration Point |
|-----------------|-----------------|-------------------|
| Knowledge Graph | Ptolemy | Extends Neo4j + Qdrant knowledge service |
| Content Engine | Gutenberg | Enhances content generation with dialectic patterns |
| Mentor Service | Galileo | Extends learning path recommendations with mentorship |
| Dialogue System | Socrates | Deepens conversational capabilities |
| Core System | All Services | Orchestrates the integrated experience |

## Development Roadmap

The system will be developed in five phases:

1. **Foundation** (Months 1-3): Core architecture and basic integrations
2. **Knowledge Integration** (Months 4-6): Enhanced concept relationships
3. **Dialogue Systems** (Months 6-9): Advanced conversational capabilities
4. **Mentor AI** (Months 9-12): Personalized mentorship frameworks
5. **Full Integration** (Months 12-15): Complete system integration and optimization

## Getting Started

### Quick Start with Docker Compose (Recommended)

1. **Install dependencies:**
   ```bash
   ./install_dependencies.sh
   ```

2. **Start the development server with live reloading:**
   ```bash
   docker-compose up
   ```

3. **Access the website:**
   Open your browser and go to `http://localhost:8080`

4. **Live Reloading:**
   Any changes to Python, HTML, CSS, or JavaScript files will automatically reload the server and refresh your browser.

### Manual Setup

If you prefer not to use Docker:

1. **Install dependencies:**
   ```bash
   ./install_dependencies.sh
   ```

2. **Start the development server:**
   ```bash
   ./start_website.sh
   ```

3. **Access the website:**
   Open your browser and go to `http://localhost:8081`

4. Navigate through the different sections using the top navigation
5. Interact with the visualizations to explore system components
6. Review the detailed documentation for implementation specifics

## AI-Generated Media Assets

The Lyceum platform features AI-generated assets created using state-of-the-art models:

1. **DALL-E 3 Generated Images**: Logo, concept illustrations, and visualizations
2. **OpenAI TTS Audio Introduction**: An audio introduction to the platform's vision
3. **Claude 3.7 Generated Content**: Vision statement and other textual content

### Regenerating Assets

To regenerate these assets, you'll need API keys for OpenAI and Anthropic:

1. Set environment variables or use command-line arguments:
   ```
   export OPENAI_API_KEY=your_openai_key
   export ANTHROPIC_API_KEY=your_anthropic_key
   ```

2. Run the regeneration script:
   ```
   ./regenerate_assets.py
   ```

3. View the updated assets in the web interface:
   ```
   python serve.py
   ```

### Media Utilities

The system includes utilities for generating different types of media:

- `utils/image_generator.py`: Generates images using DALL-E 3
- `utils/audio_generator.py`: Creates audio files using OpenAI TTS
- `utils/content_generator.py`: Produces textual content using Claude 3.7
- `utils/generate_media.py`: Orchestrates generation of all media types

## Contributing

Refer to the [Implementation Plan](IMPLEMENTATION_PLAN.md) for detailed task breakdowns and development guidelines.

## Recent Improvements

The Lyceum platform has undergone significant improvements:

1. **Enhanced Audio System**
   - Robust audio player with error handling and fallbacks
   - Multiple voice support (Fable, Alloy, Echo, Nova)
   - Dynamic voice selector that adapts to available voices
   - Visual audio wave animation during playback

2. **Visualizations**
   - D3.js powered knowledge graph visualization 
   - Mermaid.js diagrams for system architecture
   - Fallback visualizations for environments without JavaScript
   - Interactive component diagrams with hover effects

3. **Team Showcase**
   - New Team page featuring leadership, technical, and education teams
   - Profile cards with smooth hover animations
   - Career opportunities section

4. **Website Structure**
   - Improved base template with responsive navigation
   - Golden ratio design system for harmonious proportions
   - Hot-reload capability for faster development
   - Better error handling and fallbacks

5. **Performance Optimizations**
   - Threaded HTTP server for better concurrency
   - Efficient asset loading with preconnect hints
   - Lazy-loaded visualizations for faster initial page load
   - Proper caching headers for static assets

See [FIXES_SUMMARY.md](FIXES_SUMMARY.md) for a detailed changelog of recent updates.