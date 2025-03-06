#!/usr/bin/env python
"""
Enhanced OpenAI content generation script for educational concepts
that intelligently determines appropriate concept count and structure
"""

import os
import json
import argparse
import time
import datetime
import re
import uuid
from pathlib import Path
from openai import OpenAI

def slugify(text):
    """Convert text to a filename-friendly slug"""
    # Convert to lowercase
    text = text.lower()
    # Replace spaces with underscores
    text = re.sub(r'\s+', '_', text)
    # Remove any non-word (alphanumeric + underscore) characters
    text = re.sub(r'[^\w\-]', '', text)
    # Limit length
    return text[:50]  # Limit to 50 chars

def generate_filename(domain, output_dir=None, extension="json"):
    """Generate a unique filename based on domain and timestamp"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    domain_slug = slugify(domain)
    filename = f"{domain_slug}_{timestamp}.{extension}"
    
    if output_dir:
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, filename)
    
    return filename

def repair_json(json_str):
    """Attempt to repair common JSON formatting issues"""
    # Look for unterminated strings
    try:
        # First try standard parsing
        json.loads(json_str)
        return json_str  # If it parses correctly, return as is
    except json.JSONDecodeError as e:
        print(f"Attempting to repair JSON: {e}")
        
        # Handle case of unterminated quotes
        if "Unterminated string" in str(e):
            # Simple approach: Add a closing quote if the error is at the end
            if "at the end of the string" in str(e) or e.pos >= len(json_str) - 10:
                json_str += '"'
            
            # Try to find and fix unterminated strings more generally
            lines = json_str.split('\n')
            for i, line in enumerate(lines):
                # Look for lines with odd number of quotes
                quotes_count = line.count('"')
                if quotes_count % 2 == 1:
                    # Add closing quote at end of line
                    lines[i] = line + '"'
                    print(f"Added missing quote at line {i+1}")
            
            json_str = '\n'.join(lines)
        
        # Handle common trailing comma issues
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        # Try to parse again
        try:
            json.loads(json_str)
            print("JSON repair appears successful")
            return json_str
        except json.JSONDecodeError:
            print("JSON repair attempt failed")
            return json_str  # Return original even though it's invalid for debugging

def safe_determine_concept_count(domain_name, description, key_topics=None, default_count=10):
    """Safely determine concept count with fallback in case of failure"""
    try:
        analysis = determine_concept_count(domain_name, description, key_topics)
        return analysis
    except Exception as e:
        print(f"Error determining concept count: {e}")
        print(f"Falling back to default count of {default_count}")
        return {
            "concept_count": default_count,
            "model": "gpt-3.5-turbo-1106" 
        }

def determine_concept_count(domain_name, description, key_topics=None):
    """Determine the appropriate number of concepts for the given domain using AI"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return None
        
    client = OpenAI(api_key=api_key)
    
    # Format the key topics for the prompt
    topics_text = ""
    if key_topics and len(key_topics) > 0:
        topics_text = "Key topics to consider:\n" + "\n".join(f"- {topic}" for topic in key_topics)
    
    analysis_prompt = f"""
    Analyze the educational domain "{domain_name}" with this description: 
    "{description}"
    
    {topics_text}
    
    Determine the appropriate number of distinct concepts that should be taught to provide:
    1. Comprehensive coverage of the domain
    2. Appropriate granularity (not too broad or too specific)
    3. A logical learning progression
    
    Respond with a JSON object containing:
    1. recommended_concept_count: the ideal number of concepts (as an integer)
    2. justification: brief explanation of your reasoning
    3. suggested_model: either "gpt-3.5-turbo" or "gpt-4-turbo" based on domain complexity
    
    Example format:
    {{"recommended_concept_count": 15, "justification": "This domain has moderate complexity...", "suggested_model": "gpt-3.5-turbo"}}
    """
    
    print("Analyzing domain complexity to determine optimal concept count...")
    
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",  # Using GPT-4 for this analytical task
            messages=[
                {"role": "system", "content": "You are an expert curriculum designer who understands how to structure knowledge domains optimally for learning."},
                {"role": "user", "content": analysis_prompt}
            ],
            temperature=0.2,
            max_tokens=1000,
            response_format={"type": "json_object"}
        )
        
        analysis_json = response.choices[0].message.content
        analysis = json.loads(analysis_json)
        
        # Extract and validate the recommended concept count
        concept_count = int(analysis.get("recommended_concept_count", 10))
        suggested_model = analysis.get("suggested_model", "gpt-3.5-turbo-1106")
        
        # Ensure we have a reasonable number
        if concept_count < 3:
            concept_count = 3
            print("Warning: Adjusted concept count to minimum of 3")
        elif concept_count > 30:
            concept_count = 30
            print("Warning: Adjusted concept count to maximum of 30")
            
        print(f"Analysis complete. Recommended number of concepts: {concept_count}")
        print(f"Justification: {analysis.get('justification', 'No justification provided')}")
        print(f"Suggested model: {suggested_model}")
        
        return {
            "concept_count": concept_count,
            "model": suggested_model
        }
        
    except Exception as e:
        print(f"Error analyzing domain: {e}")
        print("Defaulting to 10 concepts")
        return {
            "concept_count": 10,
            "model": "gpt-3.5-turbo-1106"
        }

def enrich_concepts_with_seed_content(concepts_data):
    """Add seed content ideas for each concept to help with future content generation"""
    concepts = concepts_data.get("concepts", [])
    if not concepts:
        return concepts_data
        
    # Make a copy to avoid modifying the original
    enriched_data = {"concepts": []}
    
    import random
    
    for concept in concepts:
        # Add seed content ideas in the description without changing structure
        description = concept.get("description", "")
        name = concept.get("name", "")
        
        # Add teaching approach suggestions in the description if not already present
        if "could be taught" not in description.lower() and "can be learned" not in description.lower():
            approach_templates = [
                "This concept can be effectively taught through {}.",
                "Learners typically grasp this concept best through {}.",
                "A recommended approach for teaching this concept is through {}."
            ]
            approach_types = [
                "hands-on exercises and practice problems",
                "real-world case studies and examples",
                "visual demonstrations and diagrams",
                "analogies to familiar concepts",
                "guided discovery and exploration",
                "step-by-step tutorials",
                "interactive simulations",
                "peer discussion and collaborative learning"
            ]
            
            # First choose a random approach type
            selected_approach_type = random.choice(approach_types)
            # Then format the template with the selected type
            approach_sentence = random.choice(approach_templates).format(selected_approach_type)
            
            # Only add if description isn't already too long
            if len(description) < 400:  # Keep descriptions reasonable
                description += f" {approach_sentence}"
        
        # Create enriched concept
        enriched_concept = concept.copy()
        enriched_concept["description"] = description
        enriched_data["concepts"].append(enriched_concept)
    
    return enriched_data

def generate_concepts(domain_name, description, num_concepts=None, key_topics=None, model=None, max_retries=3):
    """Generate concepts using OpenAI API with JSON response format"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return None
        
    client = OpenAI(api_key=api_key)
    
    # If num_concepts is None, determine it
    if num_concepts is None:
        analysis = safe_determine_concept_count(domain_name, description, key_topics)
        num_concepts = analysis["concept_count"]
        if not model:
            model = analysis["model"]
    
    # Default model if not specified
    if not model:
        model = "gpt-3.5-turbo-1106"
        
    # Cap the number of concepts to a reasonable maximum
    if num_concepts > 15 and model.startswith("gpt-3.5"):
        print(f"Warning: Reducing concept count from {num_concepts} to 15 for GPT-3.5 model")
        num_concepts = 15
    elif num_concepts > 25:
        print(f"Warning: Reducing concept count from {num_concepts} to 25 for better reliability")
        num_concepts = 25
    
    # Format the key topics for the prompt
    topics_text = ""
    if key_topics and len(key_topics) > 0:
        topics_text = "Key topics to include:\n" + "\n".join(f"- {topic}" for topic in key_topics)
    
    # Build prompt
    prompt = f"""
    Create a knowledge graph with {num_concepts} concepts for teaching {domain_name}.
    
    Domain description: {description}
    
    {topics_text}
    
    Important guidelines:
    1. Ensure concepts have appropriate granularity - not too broad or too specific
    2. Cover both fundamental and advanced aspects of the domain
    3. Create a logical progression from basic to advanced concepts
    4. Assign prerequisites thoughtfully to create a meaningful learning path
    5. Ensure a mix of theoretical and practical concepts where appropriate
    
    Return a JSON object with an array of {num_concepts} concepts. Each concept must have:
    - id: a unique number as a string
    - name: a clear, concise name
    - description: a comprehensive explanation that includes:
      * Core definition (what it is)
      * Why it matters (relevance and application)
      * How it connects to other concepts
      * A brief example or illustration if applicable
      Keep all of this within a single description field as a paragraph of 3-5 sentences.
    - difficulty: a number from 0.0 to 1.0 (easiest to hardest)
    - complexity: a number from 0.0 to 1.0 (simple to complex)
    - importance: a number from 0.0 to 1.0 (less to more important)
    - prerequisites: an array of concept IDs that should be learned first
    
    Make sure descriptions are rich and informative while maintaining a consistent structure:
    1. Start with a clear definition
    2. Explain practical relevance or applications
    3. Include key characteristics or components
    4. Add a brief concrete example or case where helpful
    5. Mention common misconceptions or important nuances when relevant
    
    IMPORTANT: Ensure all JSON syntax is valid with proper quoting and escaping of special characters.
    
    Example format:
    {{"concepts": [
      {{"id": "1", "name": "Example Concept", "description": "Example Concept refers to the fundamental principle of X that establishes how Y works in practice. It serves as the foundation for understanding more complex topics like Z and is widely applied in real-world scenarios such as A and B. The concept involves three key components: first, understanding the core mechanism; second, recognizing pattern variations; and third, applying appropriate constraints. For instance, when implementing this concept in C, developers typically start by defining the scope before proceeding to implementation.", "difficulty": 0.5, "complexity": 0.6, "importance": 0.8, "prerequisites": []}}
    ]}}
    """
    
    print(f"Generating {num_concepts} concepts using {model}, please wait...")
    
    # Add retry loop for robustness
    for attempt in range(max_retries):
        try:
            # Adjust max tokens based on number of concepts
            max_token_count = 4000 if num_concepts > 15 else 3000
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a curriculum design expert who creates well-structured learning paths with precisely the right number of concepts to cover the domain effectively. You always produce valid, well-formed JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=max_token_count,
                response_format={"type": "json_object"}
            )
            
            # Get the raw response content
            json_str = response.choices[0].message.content
            
            # Print first bit for debugging
            print(f"Received response from OpenAI. First 100 chars: {json_str[:100]}...")
            
            try:
                # Attempt to repair JSON if needed
                repaired_json = repair_json(json_str)
                
                # Parse JSON
                data = json.loads(repaired_json)
                
                # Validate structure
                if "concepts" in data and isinstance(data["concepts"], list) and len(data["concepts"]) > 0:
                    print(f"Successfully generated {len(data['concepts'])} concepts")
                    
                    # Save raw response for inspection
                    with open("last_successful_response.json", "w") as f:
                        f.write(repaired_json)
                        
                    return data
                else:
                    print("Response was valid JSON but missing expected 'concepts' array")
                    if attempt < max_retries - 1:
                        print(f"Retrying... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(2)  # Brief delay before retry
                    else:
                        # Write debug file on last attempt
                        with open("debug_response.txt", "w") as f:
                            f.write(json_str)
                        return None
                    
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                print("Writing raw response to debug_response.txt for inspection")
                with open("debug_response.txt", "w") as f:
                    f.write(json_str)
                    
                if attempt < max_retries - 1:
                    print(f"Retrying... (attempt {attempt + 1}/{max_retries})")
                    
                    # On retry, adjust the prompt to emphasize valid JSON with fewer concepts
                    if num_concepts > 10 and attempt == max_retries - 2:
                        print("Reducing number of concepts for final attempt...")
                        num_concepts = 10  # Reduce for last attempt
                    
                    time.sleep(2)  # Brief delay before retry
                else:
                    # Try fallback to gpt-3.5-turbo on last attempt if using gpt-4
                    if model.startswith("gpt-4") and attempt == max_retries - 1:
                        print("Trying fallback to gpt-3.5-turbo...")
                        fallback_model = "gpt-3.5-turbo-1106"
                        
                        fallback_response = client.chat.completions.create(
                            model=fallback_model,
                            messages=[
                                {"role": "system", "content": "You are a curriculum design expert who creates well-structured learning paths. You always produce valid, well-formed JSON."},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.2,
                            max_tokens=2500,
                            response_format={"type": "json_object"}
                        )
                        
                        fallback_json = fallback_response.choices[0].message.content
                        
                        try:
                            # Try to parse fallback response
                            fallback_data = json.loads(fallback_json)
                            if "concepts" in fallback_data and isinstance(fallback_data["concepts"], list) and len(fallback_data["concepts"]) > 0:
                                print(f"Fallback successful. Generated {len(fallback_data['concepts'])} concepts with {fallback_model}")
                                return fallback_data
                        except:
                            print("Fallback attempt also failed")
                    
                    return None
                
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying... (attempt {attempt + 1}/{max_retries})")
                time.sleep(5)  # Longer delay for API errors
            else:
                return None

def analyze_concept_graph(concepts_data):
    """Analyze the generated concept graph for quality and structure"""
    concepts = concepts_data.get("concepts", [])
    total = len(concepts)
    
    if total == 0:
        return
    
    # Collect statistics
    prereq_counts = [len(c.get("prerequisites", [])) for c in concepts]
    avg_prereqs = sum(prereq_counts) / total if total > 0 else 0
    difficulty_values = [c.get("difficulty", 0) for c in concepts]
    avg_difficulty = sum(difficulty_values) / total if total > 0 else 0
    
    # Check for isolated concepts (no prerequisites and not a prerequisite for others)
    concept_ids = [c.get("id") for c in concepts]
    all_prereqs = []
    for c in concepts:
        all_prereqs.extend(c.get("prerequisites", []))
    
    isolated_count = 0
    isolated_concepts = []
    for c in concepts:
        c_id = c.get("id")
        if c_id not in all_prereqs and not c.get("prerequisites", []):
            isolated_count += 1
            isolated_concepts.append(c.get("name", f"Concept {c_id}"))
    
    # Analyze description quality
    description_lengths = [len(c.get("description", "")) for c in concepts]
    avg_desc_length = sum(description_lengths) / total if total > 0 else 0
    short_descriptions = [c.get("name", f"Concept {c.get('id')}") for c in concepts if len(c.get("description", "")) < 100]
    
    # Analyze concept spread
    difficulties = [c.get("difficulty", 0) for c in concepts]
    beginner_concepts = sum(1 for d in difficulties if d < 0.33)
    intermediate_concepts = sum(1 for d in difficulties if 0.33 <= d < 0.66)
    advanced_concepts = sum(1 for d in difficulties if d >= 0.66)
    
    # Analyze prereq depth
    max_depth = 0
    concept_depths = {}
    
    def calculate_depth(concept_id, visited=None):
        if visited is None:
            visited = set()
        if concept_id in visited:
            return 0  # Avoid circular dependencies
        visited.add(concept_id)
        
        concept = next((c for c in concepts if c.get("id") == concept_id), None)
        if not concept or not concept.get("prerequisites", []):
            return 0
            
        prereq_depths = [calculate_depth(prereq_id, visited.copy()) for prereq_id in concept.get("prerequisites", [])]
        return 1 + max(prereq_depths) if prereq_depths else 0
    
    for concept in concepts:
        depth = calculate_depth(concept.get("id"))
        concept_depths[concept.get("id")] = depth
        max_depth = max(max_depth, depth)
    
    # Print analysis
    print("\n----- Concept Graph Analysis -----")
    print(f"Total concepts: {total}")
    print(f"Concept distribution: {beginner_concepts} beginner, {intermediate_concepts} intermediate, {advanced_concepts} advanced")
    print(f"Average prerequisites per concept: {avg_prereqs:.2f}")
    print(f"Maximum prerequisite depth: {max_depth}")
    print(f"Average description length: {avg_desc_length:.1f} characters")
    
    if short_descriptions:
        print(f"Concepts with short descriptions ({len(short_descriptions)}):")
        for name in short_descriptions[:5]:  # Show just first 5
            print(f"  - {name}")
        if len(short_descriptions) > 5:
            print(f"  - ...and {len(short_descriptions) - 5} more")
    
    if isolated_concepts:
        print(f"Isolated concepts ({isolated_count}):")
        for name in isolated_concepts[:5]:  # Show just first 5
            print(f"  - {name}")
        if len(isolated_concepts) > 5:
            print(f"  - ...and {len(isolated_concepts) - 5} more")
    
    # Check for circular dependencies
    circular = []
    for c in concepts:
        c_id = c.get("id")
        for prereq_id in c.get("prerequisites", []):
            for other in concepts:
                if other.get("id") == prereq_id and c_id in other.get("prerequisites", []):
                    circular.append((c_id, prereq_id))
    
    if circular:
        print("Warning: Found circular dependencies:")
        for a, b in circular:
            print(f"  Concept {a} ↔ Concept {b}")
    else:
        print("No circular dependencies found (good!)")
    
    # Content generation readiness
    description_quality = "High" if avg_desc_length > 300 else "Medium" if avg_desc_length > 150 else "Low"
    structure_quality = "Good" if not circular and isolated_count < total * 0.1 else "Fair" if not circular else "Poor"
    
    print(f"\nContent Generation Readiness:")
    print(f"  Description Quality: {description_quality}")
    print(f"  Knowledge Structure: {structure_quality}")
    print(f"  Learning Path Depth: {max_depth} levels")
    
    print("--------------------------------\n")

def extract_topics_from_description(description):
    """Use AI to extract key topics from a domain description"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return []
        
    client = OpenAI(api_key=api_key)
    
    prompt = f"""
    Based on this description of an educational domain:
    "{description}"
    
    Extract 5-8 key topics that should be covered when teaching this domain.
    Return a JSON array of strings representing these topics.
    
    Example format:
    ["Topic 1", "Topic 2", "Topic 3"]
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=[
                {"role": "system", "content": "You are a curriculum design expert who identifies the most important topics in educational domains."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=500,
            response_format={"type": "json_object"}
        )
        
        # Get the raw response content and parse
        try:
            json_str = response.choices[0].message.content
            data = json.loads(json_str)
            
            # Extract the topics - handle both array directly or nested in a property
            if isinstance(data, list):
                topics = data
            else:
                # Try common key names
                for key in ["topics", "key_topics", "results"]:
                    if key in data and isinstance(data[key], list):
                        topics = data[key]
                        break
                else:
                    # If no known keys, try the first list value found
                    topics = next((v for v in data.values() if isinstance(v, list)), [])
            
            if topics and len(topics) > 0:
                print(f"Extracted {len(topics)} key topics from description")
                return topics
            else:
                print("Could not extract topics from description")
                return []
                
        except (json.JSONDecodeError, AttributeError) as e:
            print(f"Error parsing topics: {e}")
            return []
            
    except Exception as e:
        print(f"Error extracting topics: {e}")
        return []

def scan_input_directory(input_dir, processed_log, output_dir=None):
    """Scan input directory for new domain files and process them"""
    # Create directory if it doesn't exist
    os.makedirs(input_dir, exist_ok=True)
    
    # Create a log of processed files if it doesn't exist
    if not os.path.exists(processed_log):
        with open(processed_log, 'w') as f:
            json.dump({"processed_files": []}, f)
    
    # Load processed files log
    with open(processed_log, 'r') as f:
        processed = json.load(f)
        processed_files = set(processed.get("processed_files", []))
    
    # Scan directory for new files
    new_files = []
    for file in os.listdir(input_dir):
        filepath = os.path.join(input_dir, file)
        if os.path.isfile(filepath) and filepath.endswith('.json') and filepath not in processed_files:
            new_files.append(filepath)
    
    if not new_files:
        print(f"No new domain files found in {input_dir}")
        return 0
    
    print(f"Found {len(new_files)} new domain files to process")
    success_count = 0
    
    # Process each file
    for filepath in new_files:
        try:
            with open(filepath, 'r') as f:
                request = json.load(f)
            
            domain = request.get("domain")
            description = request.get("description")
            num_concepts = request.get("num_concepts")
            topics = request.get("topics", [])
            model = request.get("model")
            
            if not domain or not description:
                print(f"Skipping {filepath}: Missing required domain or description")
                continue
            
            # Generate output filename
            if output_dir:
                output_file = generate_filename(domain, output_dir)
            else:
                output_file = generate_filename(domain)
            
            print(f"Processing {filepath} -> {output_file}")
            
            # Generate concepts
            data = generate_concepts(domain, description, num_concepts, topics, model)
            
            if data:
                # Enrich and save
                enriched_data = enrich_concepts_with_seed_content(data)
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(enriched_data, f, indent=2)
                print(f"Generated concepts for '{domain}' saved to {output_file}")
                success_count += 1
                
                # Mark as processed
                processed_files.add(filepath)
                
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
    
    # Update processed files log
    with open(processed_log, 'w') as f:
        json.dump({"processed_files": list(processed_files)}, f)
    
    print(f"Successfully processed {success_count} out of {len(new_files)} files")
    return success_count

def process_batch_file(batch_file, output_dir=None):
    """Process a batch file containing multiple domain requests"""
    try:
        with open(batch_file, 'r') as f:
            batch = json.load(f)
        
        domains = batch.get("domains", [])
        if not domains:
            print(f"No domains found in batch file {batch_file}")
            return 0
        
        print(f"Processing {len(domains)} domains from batch file")
        success_count = 0
        
        for domain_info in domains:
            domain = domain_info.get("domain")
            description = domain_info.get("description")
            
            if not domain or not description:
                print(f"Skipping domain: Missing required domain or description")
                continue
            
            # Get optional parameters
            num_concepts = domain_info.get("num_concepts")
            topics = domain_info.get("topics", [])
            model = domain_info.get("model")
            
            # Generate output filename
            if output_dir:
                output_file = generate_filename(domain, output_dir)
            else:
                output_file = generate_filename(domain)
            
            print(f"Processing domain '{domain}' -> {output_file}")
            
            # Generate concepts
            data = generate_concepts(domain, description, num_concepts, topics, model)
            
            if data:
                # Enrich and save
                enriched_data = enrich_concepts_with_seed_content(data)
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(enriched_data, f, indent=2)
                print(f"Generated concepts for '{domain}' saved to {output_file}")
                success_count += 1
        
        print(f"Successfully processed {success_count} out of {len(domains)} domains")
        return success_count
        
    except Exception as e:
        print(f"Error processing batch file {batch_file}: {e}")
        return 0

def run_continuous_mode(input_dir, output_dir, processed_log, interval=60):
    """Run in continuous mode, monitoring input directory for new domain files"""
    print(f"Starting continuous mode. Monitoring {input_dir} every {interval} seconds")
    print(f"Generated concepts will be saved to {output_dir}")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            scan_input_directory(input_dir, processed_log, output_dir)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopping continuous mode")
    
    return 0

def main():
    parser = argparse.ArgumentParser(description="Generate educational concepts with OpenAI")
    
    # Create subparsers for different modes
    subparsers = parser.add_subparsers(dest="mode", help="Operation mode")
    
    # Single domain mode
    single_parser = subparsers.add_parser("single", help="Generate concepts for a single domain")
    single_parser.add_argument("--domain", type=str, required=True, help="Domain name (e.g., 'Python Programming')")
    single_parser.add_argument("--description", type=str, required=True, help="Brief domain description")
    single_parser.add_argument("--output", type=str, help="Output JSON file")
    single_parser.add_argument("--output-dir", type=str, help="Directory to store output files")
    single_parser.add_argument("--num-concepts", type=int, help="Number of concepts to generate (if not specified, AI will determine)")
    single_parser.add_argument("--topics", nargs="+", help="Key topics to include (if not provided, will be extracted from description)")
    single_parser.add_argument("--model", type=str, help="OpenAI model to use (if not specified, will be selected based on domain complexity)")
    single_parser.add_argument("--analyze", action="store_true", help="Analyze the concept graph after generation")
    single_parser.add_argument("--auto-name", action="store_true", help="Automatically generate filename based on domain and timestamp")
    
    # Batch mode
    batch_parser = subparsers.add_parser("batch", help="Process a batch file with multiple domains")
    batch_parser.add_argument("--batch-file", type=str, required=True, help="Path to JSON batch file")
    batch_parser.add_argument("--output-dir", type=str, required=True, help="Directory to store output files")
    
    # Continuous mode
    continuous_parser = subparsers.add_parser("continuous", help="Run in continuous mode, monitoring an input directory")
    continuous_parser.add_argument("--input-dir", type=str, required=True, help="Directory to monitor for new domain files")
    continuous_parser.add_argument("--output-dir", type=str, required=True, help="Directory to store output files")
    continuous_parser.add_argument("--processed-log", type=str, default="processed_files.json", help="Path to log of processed files")
    continuous_parser.add_argument("--interval", type=int, default=60, help="Check interval in seconds (default: 60)")
    
    # Default mode (for backward compatibility)
    parser.add_argument("--domain", type=str, help="Domain name (for backward compatibility)")
    parser.add_argument("--description", type=str, help="Brief domain description (for backward compatibility)")
    parser.add_argument("--output", type=str, default="concepts.json", help="Output JSON file (for backward compatibility)")
    parser.add_argument("--output-dir", type=str, help="Directory to store output files (for backward compatibility)")
    parser.add_argument("--num-concepts", type=int, help="Number of concepts (for backward compatibility)")
    parser.add_argument("--topics", nargs="+", help="Key topics (for backward compatibility)")
    parser.add_argument("--model", type=str, help="OpenAI model (for backward compatibility)")
    parser.add_argument("--analyze", action="store_true", help="Analyze the concept graph (for backward compatibility)")
    parser.add_argument("--auto-name", action="store_true", help="Auto-generate filename (for backward compatibility)")
    parser.add_argument("--batch-file", type=str, help="Path to batch file (for backward compatibility)")
    
    args = parser.parse_args()
    
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        return 1
    
    # Determine mode
    if args.mode == "continuous":
        return run_continuous_mode(
            args.input_dir,
            args.output_dir,
            args.processed_log,
            args.interval
        )
    elif args.mode == "batch":
        return process_batch_file(
            args.batch_file,
            args.output_dir
        )
    elif args.mode == "single":
        # Extract topics if not provided
        key_topics = args.topics
        if not key_topics:
            print("No topics provided, extracting from description...")
            key_topics = extract_topics_from_description(args.description)
        
        # Determine output file
        output_file = args.output
        if args.auto_name or not output_file:
            output_file = generate_filename(args.domain, args.output_dir)
            
        # Generate concepts
        data = generate_concepts(
            args.domain,
            args.description,
            args.num_concepts,
            key_topics,
            args.model
        )
        
        if data:
            # Enrich concepts with seed content ideas
            print("Enriching concepts with seed content ideas...")
            enriched_data = enrich_concepts_with_seed_content(data)
            
            # Analyze the concept graph if requested
            if args.analyze:
                analyze_concept_graph(enriched_data)
                
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(enriched_data, f, indent=2)
            print(f"Saved enriched concepts to {output_file}")
            return 0
        else:
            print("Failed to generate valid concept data")
            return 1
    else:
        # Backward compatibility mode - determine what to do based on arguments
        if args.batch_file:
            output_dir = args.output_dir or os.path.dirname(args.output) or "."
            return process_batch_file(args.batch_file, output_dir)
        elif args.domain and args.description:
            # Extract topics if not provided
            key_topics = args.topics
            if not key_topics:
                print("No topics provided, extracting from description...")
                key_topics = extract_topics_from_description(args.description)
            
            # Determine output file
            output_file = args.output
            if args.auto_name:
                output_file = generate_filename(args.domain, args.output_dir)
                
            # Generate concepts
            data = generate_concepts(
                args.domain,
                args.description,
                args.num_concepts,
                key_topics,
                args.model
            )
            
            if data:
                # Enrich concepts with seed content ideas
                print("Enriching concepts with seed content ideas...")
                enriched_data = enrich_concepts_with_seed_content(data)
                
                # Analyze the concept graph if requested
                if args.analyze:
                    analyze_concept_graph(enriched_data)
                    
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(enriched_data, f, indent=2)
                print(f"Saved enriched concepts to {output_file}")
                
                # Generate a sample content idea for one concept
                if enriched_data["concepts"]:
                    example_concept = enriched_data["concepts"][0]
                    print("\n----- Sample Content Generation From Concept -----")
                    print(f"Concept: {example_concept.get('name')}")
                    print(f"Description: {example_concept.get('description')}")
                    print("This enriched description can be used to generate:")
                    print("1. Quiz questions about key components")
                    print("2. Lesson plans focusing on the practical applications")
                    print("3. Interactive exercises based on the examples")
                    print("4. Discussion prompts from the nuances or misconceptions")
                    print("5. Visual aids illustrating the relationships")
                    print("-----------------------------------------------")
                
                print("You can now import this file into your educational system")
                return 0
            else:
                print("Failed to generate valid concept data")
                return 1
        else:
            parser.print_help()
            return 1

if __name__ == "__main__":
    exit(main())