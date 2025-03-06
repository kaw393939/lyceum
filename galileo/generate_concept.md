# Enhanced OpenAI Content Generation Script

This Python script is designed to generate a structured set of educational concepts (a knowledge graph) for any given domain using the OpenAI API. It leverages AI to determine the optimal number of concepts and builds a comprehensive, logically sequenced learning path complete with detailed descriptions, prerequisites, and seed content ideas. The script supports multiple modes (single, batch, and continuous) to suit different workflows.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Command-Line Modes](#command-line-modes)
  - [Examples](#examples)
- [How It Works](#how-it-works)
  - [Determining Concept Count](#determining-concept-count)
  - [Generating and Enriching Concepts](#generating-and-enriching-concepts)
  - [Analyzing the Concept Graph](#analyzing-the-concept-graph)
  - [Directory & Batch Processing](#directory--batch-processing)
- [Configuration](#configuration)
- [License](#license)

---

## Overview

This script generates educational content by:
- Analyzing a domain's description and key topics.
- Using OpenAI's models (GPT-4 or GPT-3.5) to determine an optimal number of learning concepts.
- Generating a knowledge graph where each concept includes a unique ID, name, detailed description, difficulty, complexity, importance, and prerequisites.
- Enriching each concept with additional seed content ideas to support future content generation.
- Providing functionality to analyze the quality and structure of the generated concept graph.

---

## Features

- **Automatic Concept Count Determination:** Leverages AI analysis to decide the number of concepts based on domain complexity.
- **Robust JSON Handling:** Includes mechanisms to repair and validate JSON responses.
- **Multiple Operation Modes:** Supports single-domain generation, batch processing from a file, and continuous directory monitoring.
- **Concept Enrichment:** Augments concept descriptions with teaching approach suggestions.
- **Concept Graph Analysis:** Provides statistics and warnings on concept distribution, prerequisite structure, and description quality.
- **Flexible Filename Generation:** Creates unique output filenames based on the domain name and timestamp.

---

## Requirements

- Python 3.7 or later
- OpenAI Python package (ensure you have a compatible client, e.g., `openai`)
- A valid OpenAI API key set in the `OPENAI_API_KEY` environment variable

---

## Installation

1. **Clone or Download the Repository:**

   ```bash
   git clone https://github.com/yourusername/yourrepository.git
   cd yourrepository
   ```

2. **Install Dependencies:**

   Ensure you have the required Python packages. You can install them via pip:

   ```bash
   pip install openai
   ```

3. **Set Up Your Environment:**

   Export your OpenAI API key in your shell:

   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```

---

## Usage

The script provides several modes of operation. You can run it directly from the command line.

### Command-Line Modes

- **Single Mode:** Generate concepts for a single domain.
- **Batch Mode:** Process a JSON file containing multiple domain requests.
- **Continuous Mode:** Monitor a directory for new domain files and process them continuously.

The script also supports backward compatibility by accepting legacy command-line arguments.

### Examples

#### Single Mode

Generate concepts for a single domain:

```bash
python script.py single \
  --domain "Python Programming" \
  --description "A comprehensive course covering basics to advanced Python topics." \
  --output "python_concepts.json" \
  --topics "syntax" "data structures" "OOP" "modules" \
  --analyze \
  --auto-name
```

#### Batch Mode

Process a batch file (`domains.json`) with multiple domains:

```bash
python script.py batch \
  --batch-file "domains.json" \
  --output-dir "output_concepts"
```

#### Continuous Mode

Monitor an input directory for new domain JSON files and process them:

```bash
python script.py continuous \
  --input-dir "new_domains" \
  --output-dir "processed_concepts" \
  --processed-log "processed_files.json" \
  --interval 60
```

---

## How It Works

### Determining Concept Count

- **Function:** `determine_concept_count`
- **Purpose:** Analyzes the domain description and key topics using OpenAI’s API (using GPT-4-turbo-preview by default) to decide the ideal number of learning concepts.
- **Fallback:** In case of failure, `safe_determine_concept_count` defaults to a preset number (default is 10).

### Generating and Enriching Concepts

- **Function:** `generate_concepts`
- **Purpose:** Constructs a detailed knowledge graph by prompting the OpenAI API to generate a JSON-formatted array of concepts. Each concept includes fields like `id`, `name`, `description`, `difficulty`, `complexity`, `importance`, and `prerequisites`.
- **Enrichment:** The `enrich_concepts_with_seed_content` function adds seed ideas (teaching approaches and examples) to each concept’s description.

### Analyzing the Concept Graph

- **Function:** `analyze_concept_graph`
- **Purpose:** After generating concepts, this function computes statistics such as:
  - Total number of concepts.
  - Distribution of beginner, intermediate, and advanced concepts.
  - Average prerequisites per concept.
  - Maximum prerequisite depth.
  - Description length analysis.
  - Identification of isolated concepts and circular dependencies.
- **Output:** Prints a detailed analysis to help assess the quality and readiness of the generated content.

### Directory & Batch Processing

- **Directory Processing:**  
  The `scan_input_directory` function monitors a specified folder for new JSON files containing domain details, processes them, and logs the processed files.
  
- **Batch Processing:**  
  The `process_batch_file` function reads a batch file with multiple domain requests and processes each one sequentially.

- **Continuous Mode:**  
  The `run_continuous_mode` function periodically scans an input directory at specified intervals (e.g., every 60 seconds) to process new domain files automatically.

---

## Configuration

- **OpenAI API Key:**  
  The script requires an API key which must be set in the environment variable `OPENAI_API_KEY`.

- **Output Settings:**  
  You can specify output directories and filenames using command-line options like `--output`, `--output-dir`, and `--auto-name` (which generates a filename based on the domain and current timestamp).

- **Model Selection:**  
  The script intelligently selects an appropriate OpenAI model based on the domain complexity. You can override this using the `--model` argument if desired.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

This README provides a comprehensive overview of the script's functionality, usage examples, and configuration options to help you quickly get started with generating educational content using OpenAI's API.