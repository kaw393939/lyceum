"""
Ptolemy Knowledge Map System - LLM Enrichment Module
==================================================
Specialized functions for enriching and improving knowledge graph quality.
"""

import json
import logging
import time
import uuid
import traceback
from typing import Dict, List, Optional, Any, Union, Tuple, Set, Callable
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps

from generation import LLMService

# Configure module-level logger
logger = logging.getLogger("llm.enrichment")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Function decorator for timing and logging
def log_execution_time(func):
    """Decorator to log execution time of functions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        logger.debug(f"{func.__name__} executed in {execution_time:.2f} seconds")
        return result
    return wrapper

class KnowledgeEnrichment:
    """Service for enhancing the knowledge graph using LLM capabilities."""
    
    def __init__(self, llm_service: LLMService):
        """Initialize the enrichment service with an LLM service."""
        self.llm = llm_service
        
        # Cache to store results temporarily to avoid duplicate work in a session
        self._cache = {}
        self._cache_ttl = 3600  # 1 hour cache TTL
        
        # Initialize thread pool for parallel processing
        self._executor = ThreadPoolExecutor(max_workers=5)
        
        logger.info("Knowledge Enrichment service initialized")
    
    def close(self):
        """Clean up resources."""
        if self._executor:
            self._executor.shutdown(wait=False)
        self._cache.clear()
        logger.info("Knowledge Enrichment service closed")
    
    def _get_from_cache(self, key):
        """Get a value from the cache if it exists and is fresh."""
        if key in self._cache:
            entry = self._cache[key]
            if time.time() - entry["timestamp"] < self._cache_ttl:
                logger.debug(f"Cache hit for {key}")
                return entry["data"]
            else:
                del self._cache[key]  # Clean expired entry
        return None
    
    def _add_to_cache(self, key, data):
        """Add a value to the cache."""
        self._cache[key] = {
            "data": data,
            "timestamp": time.time()
        }
        
        # Clean old cache entries if cache is getting large
        if len(self._cache) > 100:
            self._clean_cache()
    
    def _clean_cache(self):
        """Remove expired cache entries."""
        now = time.time()
        expired_keys = [k for k, v in self._cache.items() if now - v["timestamp"] > self._cache_ttl]
        for key in expired_keys:
            del self._cache[key]
    
    def _process_in_batches(self, items: List[Any], processor: Callable, 
                           batch_size: int = 10, parallel: bool = True) -> List[Any]:
        """Process a list of items in batches, optionally in parallel."""
        if not items:
            return []
        
        # Split into batches
        batches = [items[i:i+batch_size] for i in range(0, len(items), batch_size)]
        logger.info(f"Processing {len(items)} items in {len(batches)} batches")
        
        results = []
        
        if parallel and len(batches) > 1:
            # Process in parallel
            futures = []
            with ThreadPoolExecutor(max_workers=min(len(batches), 5)) as executor:
                for batch in batches:
                    future = executor.submit(processor, batch)
                    futures.append(future)
                
                # Collect results as they complete
                for future in as_completed(futures):
                    try:
                        batch_result = future.result()
                        if batch_result:
                            results.extend(batch_result)
                    except Exception as e:
                        logger.error(f"Error in batch processing: {e}")
                        logger.debug(traceback.format_exc())
        else:
            # Process sequentially
            for batch in batches:
                try:
                    batch_result = processor(batch)
                    if batch_result:
                        results.extend(batch_result)
                except Exception as e:
                    logger.error(f"Error in batch processing: {e}")
                    logger.debug(traceback.format_exc())
        
        return results
    
    def _merge_with_originals(self, original_items: List[Dict[str, Any]], 
                             enhanced_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge enhanced items with originals, preserving original data where needed."""
        if not enhanced_items:
            return original_items
        
        if not original_items:
            return enhanced_items
        
        # Create a map of original items by ID
        original_map = {item["id"]: item for item in original_items if "id" in item}
        
        # Build merged list
        merged = []
        
        # First, process all enhanced items
        for enhanced in enhanced_items:
            if "id" in enhanced and enhanced["id"] in original_map:
                # Get the original
                original = original_map[enhanced["id"]]
                
                # Create a merged item starting with original data
                merged_item = original.copy()
                
                # Update with enhanced values that are not empty
                for key, value in enhanced.items():
                    if value is not None and value != "":
                        # Special handling for dictionaries to do a deep merge
                        if isinstance(value, dict) and key in original and isinstance(original[key], dict):
                            merged_item[key] = {**original[key], **value}
                        else:
                            merged_item[key] = value
                
                merged.append(merged_item)
            else:
                # If no matching original, use the enhanced directly
                merged.append(enhanced)
        
        # Add any originals that weren't enhanced
        enhanced_ids = {item["id"] for item in enhanced_items if "id" in item}
        for item_id, item in original_map.items():
            if item_id not in enhanced_ids:
                merged.append(item)
        
        return merged
    
    @log_execution_time
    def enhance_concept_descriptions(self, concepts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhance concept descriptions with more detailed explanations."""
        if not concepts:
            return []
        
        logger.info(f"Enhancing descriptions for {len(concepts)} concepts")
        
        # Check cache first
        cache_key = f"enhance_descriptions_{hash(str(concepts))}"
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            return cached_result
        
        try:
            # Use the built-in enrichment function
            enhanced = self.llm.enrich_concepts(concepts)
            
            # Ensure we don't lose data
            result = self._merge_with_originals(concepts, enhanced)
            
            # Cache the result
            self._add_to_cache(cache_key, result)
            
            return result
        except Exception as e:
            logger.error(f"Error enhancing concept descriptions: {e}")
            logger.debug(traceback.format_exc())
            return concepts  # Return original concepts on error
    
    @log_execution_time
    def add_teaching_approaches(self, concepts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add specific teaching approaches to concepts."""
        if not concepts:
            return []
        
        logger.info(f"Adding teaching approaches for {len(concepts)} concepts")
        
        # Check cache first
        cache_key = f"teaching_approaches_{hash(str(concepts))}"
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            return cached_result
        
        # Create a custom prompt for teaching approaches
        custom_prompt = (
            "For each of the following educational concepts, add a section about "
            "effective teaching approaches. Include at least three specific teaching "
            "methods that would work well for each concept, along with a brief "
            "explanation of why that method is effective for teaching this particular concept.\n\n"
            "Focus on active learning techniques, varied approaches for different learning styles, "
            "and assessment strategies to check understanding."
        )
        
        # Define the batch processor function
        def process_batch(batch):
            return self._add_teaching_approaches_batch(batch, custom_prompt)
        
        # Process in batches
        result = self._process_in_batches(concepts, process_batch, batch_size=10, parallel=True)
        
        # Ensure we merge with originals to preserve data
        merged_result = self._merge_with_originals(concepts, result)
        
        # Cache the result
        self._add_to_cache(cache_key, merged_result)
        
        return merged_result
    
    def _add_teaching_approaches_batch(self, concepts_batch: List[Dict[str, Any]], prompt: str) -> List[Dict[str, Any]]:
        """Process a batch of concepts to add teaching approaches."""
        system_prompt = "You are an expert in educational pedagogy and instructional design."
        
        batch_json = json.dumps({"concepts": concepts_batch}, indent=2)
        prompt_text = f"{prompt}\n\n{batch_json}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_text}
        ]
        
        try:
            # Use GPT-4 when available for better pedagogical insights
            model = "gpt-4-turbo" if getattr(self.llm.config, 'allow_gpt4', False) else "gpt-3.5-turbo"
            
            json_str = self.llm._call_openai_with_retry(
                messages=messages,
                model=model,
                temperature=0.4,  # Lower temperature for more focused responses
                max_tokens=4000,
                json_mode=True
            )
            
            if json_str:
                repaired_json = self.llm._repair_json(json_str)
                data = json.loads(repaired_json)
                
                if "concepts" in data and isinstance(data["concepts"], list):
                    logger.info(f"Successfully added teaching approaches for {len(data['concepts'])} concepts")
                    return data["concepts"]
                else:
                    logger.warning("Invalid response format for teaching approaches. Using original batch.")
            else:
                logger.warning("Failed to add teaching approaches. Using original batch.")
            
            return concepts_batch  # Return original batch on error
        except Exception as e:
            logger.error(f"Error processing teaching approaches: {e}")
            logger.debug(traceback.format_exc())
            return concepts_batch
    
    @log_execution_time
    def add_assessment_strategies(self, concepts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add assessment strategies to concepts."""
        if not concepts:
            return []
        
        logger.info(f"Adding assessment strategies for {len(concepts)} concepts")
        
        # Check cache first
        cache_key = f"assessment_strategies_{hash(str(concepts))}"
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            return cached_result
        
        # Create a custom prompt for assessment strategies
        custom_prompt = (
            "For each of the following educational concepts, add appropriate assessment strategies. "
            "Include at least three different assessment approaches that effectively measure "
            "understanding of the concept. For each assessment strategy, explain:\n"
            "1. What type of assessment it is (formative, summative, diagnostic, etc.)\n"
            "2. How it specifically tests understanding of this concept\n"
            "3. What student outcomes or behaviors would demonstrate mastery\n\n"
            "Ensure the assessment strategies are appropriate for the concept's difficulty level."
        )
        
        # Define the batch processor function
        def process_batch(batch):
            return self._add_assessment_strategies_batch(batch, custom_prompt)
        
        # Process in batches
        result = self._process_in_batches(concepts, process_batch, batch_size=10, parallel=True)
        
        # Ensure we merge with originals to preserve data
        merged_result = self._merge_with_originals(concepts, result)
        
        # Cache the result
        self._add_to_cache(cache_key, merged_result)
        
        return merged_result
    
    def _add_assessment_strategies_batch(self, concepts_batch: List[Dict[str, Any]], prompt: str) -> List[Dict[str, Any]]:
        """Process a batch of concepts to add assessment strategies."""
        system_prompt = "You are an expert in educational assessment and evaluation."
        
        batch_json = json.dumps({"concepts": concepts_batch}, indent=2)
        prompt_text = f"{prompt}\n\n{batch_json}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_text}
        ]
        
        try:
            # Use GPT-4 when available for better assessment designs
            model = "gpt-4-turbo" if getattr(self.llm.config, 'allow_gpt4', False) else "gpt-3.5-turbo"
            
            json_str = self.llm._call_openai_with_retry(
                messages=messages,
                model=model,
                temperature=0.4,
                max_tokens=4000,
                json_mode=True
            )
            
            if json_str:
                repaired_json = self.llm._repair_json(json_str)
                data = json.loads(repaired_json)
                
                if "concepts" in data and isinstance(data["concepts"], list):
                    logger.info(f"Successfully added assessment strategies for {len(data['concepts'])} concepts")
                    return data["concepts"]
                else:
                    logger.warning("Invalid response format for assessment strategies. Using original batch.")
            else:
                logger.warning("Failed to add assessment strategies. Using original batch.")
            
            return concepts_batch  # Return original batch on error
        except Exception as e:
            logger.error(f"Error processing assessment strategies: {e}")
            logger.debug(traceback.format_exc())
            return concepts_batch
    
    @log_execution_time
    def identify_missing_concepts(self, 
                               domain: str, 
                               existing_concepts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify potentially missing concepts in a knowledge domain."""
        if not existing_concepts:
            return []
        
        logger.info(f"Identifying missing concepts for domain '{domain}' with {len(existing_concepts)} existing concepts")
        
        # Check cache first
        cache_key = f"missing_concepts_{domain}_{hash(str(existing_concepts))}"
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            return cached_result
        
        # For large concept sets, use a sampling approach
        if len(existing_concepts) > 50:
            logger.info("Using sampling approach for large concept set")
            return self._identify_missing_concepts_sampling(domain, existing_concepts)
        
        # Create a list of concept names and descriptions
        concept_summaries = []
        for concept in existing_concepts:
            if not isinstance(concept, dict) or "name" not in concept:
                continue
                
            description = concept.get("description", "")
            if description:
                summary = f"{concept['name']}: {description[:100]}..."
                concept_summaries.append(summary)
            else:
                summary = f"{concept['name']}"
                concept_summaries.append(summary)
        
        existing_concepts_text = "\n".join(concept_summaries)
        
        custom_prompt = (
            f"Analyze this existing knowledge graph for the domain '{domain}':\n\n"
            f"{existing_concepts_text}\n\n"
            "Identify important concepts that are missing from this knowledge graph. "
            "Consider what gaps exist in the current structure. A good knowledge graph should "
            "have appropriate breadth and depth, covering both foundational and advanced topics.\n\n"
            "For each missing concept you identify:\n"
            "1. Provide a clear name\n"
            "2. Write a detailed description\n"
            "3. Explain why it's an important addition\n"
            "4. Suggest which existing concepts it relates to\n"
            "5. Indicate its appropriate difficulty level\n\n"
            "Return the results as a JSON array of missing concept objects."
        )
        
        system_prompt = "You are an expert in curriculum design and knowledge organization."
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": custom_prompt}
        ]
        
        try:
            # Use GPT-4 for better analysis when available
            model = "gpt-4-turbo" if getattr(self.llm.config, 'allow_gpt4', False) else "gpt-3.5-turbo"
            
            json_str = self.llm._call_openai_with_retry(
                messages=messages,
                model=model,
                temperature=0.5,
                max_tokens=3000,
                json_mode=True
            )
            
            if not json_str:
                logger.error("Failed to identify missing concepts")
                return []
                
            repaired_json = self.llm._repair_json(json_str)
            data = json.loads(repaired_json)
            
            missing_concepts = []
            if isinstance(data, list):
                missing_concepts = data
            elif isinstance(data, dict) and "missing_concepts" in data:
                missing_concepts = data["missing_concepts"]
            elif isinstance(data, dict) and "concepts" in data:
                missing_concepts = data["concepts"]
            
            # Add IDs and timestamps to the missing concepts
            processed_concepts = []
            for concept in missing_concepts:
                if not isinstance(concept, dict) or "name" not in concept:
                    continue
                    
                # Add required fields
                concept["id"] = str(uuid.uuid4())
                concept["created_at"] = datetime.now().isoformat()
                concept["concept_type"] = concept.get("concept_type", "topic")
                concept["suggested"] = True
                
                # Ensure at least a minimal description
                if "description" not in concept or not concept["description"]:
                    concept["description"] = f"Suggested concept: {concept['name']}"
                
                processed_concepts.append(concept)
            
            logger.info(f"Identified {len(processed_concepts)} missing concepts")
            
            # Cache the result
            self._add_to_cache(cache_key, processed_concepts)
            
            return processed_concepts
        except Exception as e:
            logger.error(f"Error identifying missing concepts: {e}")
            logger.debug(traceback.format_exc())
            return []
    
    def _identify_missing_concepts_sampling(self, domain: str, existing_concepts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Use sampling approach to identify missing concepts for large concept sets."""
        # Get overall domain statistics
        concept_types = {}
        difficulties = {}
        for concept in existing_concepts:
            concept_type = concept.get("concept_type", "topic")
            difficulty = concept.get("difficulty", "intermediate")
            
            concept_types[concept_type] = concept_types.get(concept_type, 0) + 1
            difficulties[difficulty] = difficulties.get(difficulty, 0) + 1
        
        # Sample concepts for each type and difficulty
        sampled_concepts = []
        
        # Sample by type
        for concept_type, count in concept_types.items():
            # Sample up to 10 concepts of each type
            sample_count = min(10, count)
            type_concepts = [c for c in existing_concepts if c.get("concept_type", "topic") == concept_type]
            sampled_concepts.extend(type_concepts[:sample_count])
        
        # Sample by difficulty
        for difficulty, count in difficulties.items():
            # Sample up to 10 concepts of each difficulty
            sample_count = min(10, count)
            difficulty_concepts = [c for c in existing_concepts if c.get("difficulty", "intermediate") == difficulty]
            sampled_concepts.extend(difficulty_concepts[:sample_count])
        
        # Remove duplicates
        unique_ids = set()
        unique_sampled = []
        for concept in sampled_concepts:
            if concept["id"] not in unique_ids:
                unique_ids.add(concept["id"])
                unique_sampled.append(concept)
        
        # Ensure we have at most 50 concepts total
        if len(unique_sampled) > 50:
            unique_sampled = unique_sampled[:50]
        
        logger.info(f"Sampled {len(unique_sampled)} concepts for analysis")
        
        # Get missing concepts based on the sample
        missing_concepts = self.identify_missing_concepts(domain, unique_sampled)
        
        # Enhance the results by adding information about the full set
        for concept in missing_concepts:
            concept["metadata"] = concept.get("metadata", {})
            concept["metadata"]["identified_from_sample"] = True
            concept["metadata"]["total_concepts_in_domain"] = len(existing_concepts)
        
        return missing_concepts
    
    @log_execution_time
    def improve_relationships(self, 
                           concepts: List[Dict[str, Any]], 
                           existing_relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Improve existing relationships and suggest new ones."""
        if not concepts or not existing_relationships:
            return []
        
        logger.info(f"Improving relationships for {len(concepts)} concepts with {len(existing_relationships)} existing relationships")
        
        # Check cache first
        cache_key = f"improve_relationships_{hash(str(concepts))}_{hash(str(existing_relationships))}"
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            return cached_result
        
        # For large graphs, use a focused approach
        if len(concepts) > 30 or len(existing_relationships) > 50:
            logger.info("Using focused approach for large relationship set")
            return self._improve_relationships_focused(concepts, existing_relationships)
        
        # Create concept ID to name mapping
        concept_map = {c["id"]: c["name"] for c in concepts if "id" in c and "name" in c}
        
        # Summarize existing relationships
        relationship_summaries = []
        for rel in existing_relationships:
            if "source_id" not in rel or "target_id" not in rel:
                continue
                
            source_name = concept_map.get(rel["source_id"], "Unknown")
            target_name = concept_map.get(rel["target_id"], "Unknown")
            rel_type = rel.get("relationship_type", "related_to")
            summary = f"{source_name} -> {rel_type} -> {target_name}"
            relationship_summaries.append(summary)
        
        relationships_text = "\n".join(relationship_summaries)
        
        # Prepare concept summaries - limit to reduce token usage
        concept_summaries = []
        for concept in concepts:
            if "id" not in concept or "name" not in concept:
                continue
                
            description = concept.get("description", "")
            description_snippet = description[:100] + "..." if description and len(description) > 100 else description
            
            summary = (
                f"ID: {concept['id']}\n"
                f"Name: {concept['name']}\n"
                f"Type: {concept.get('concept_type', 'topic')}\n"
                f"Description: {description_snippet}\n"
            )
            concept_summaries.append(summary)
        
        concepts_text = "\n\n".join(concept_summaries)
        
        custom_prompt = (
            "Analyze these educational concepts and their existing relationships:\n\n"
            f"=== CONCEPTS ===\n{concepts_text}\n\n"
            f"=== EXISTING RELATIONSHIPS ===\n{relationships_text}\n\n"
            "Identify improvements to the knowledge graph by:\n"
            "1. Suggesting new relationships that should exist but are missing\n"
            "2. Identifying relationships that may be incorrect or have the wrong type\n"
            "3. Recommending relationship strength adjustments\n\n"
            "For each new or modified relationship, specify:\n"
            "- source_id: ID of the source concept\n"
            "- target_id: ID of the target concept\n"
            "- relationship_type: one of 'prerequisite', 'builds_on', 'related_to', 'part_of', 'example_of', 'contrasts_with'\n"
            "- strength: number from 0.0 to 1.0 indicating relationship strength\n"
            "- description: explanation of the relationship\n"
            "- action: 'add' for new relationships, 'modify' for changing existing ones\n\n"
            "Return a JSON object with an array 'relationships' containing these suggestions."
        )
        
        system_prompt = "You are an expert in knowledge graph design and educational relationships."
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": custom_prompt}
        ]
        
        try:
            # Use GPT-4 for better relationship analysis when available
            model = "gpt-4-turbo" if getattr(self.llm.config, 'allow_gpt4', False) else "gpt-3.5-turbo"
            
            json_str = self.llm._call_openai_with_retry(
                messages=messages,
                model=model,
                temperature=0.4,
                max_tokens=4000,
                json_mode=True
            )
            
            if not json_str:
                logger.error("Failed to improve relationships")
                return []
                
            repaired_json = self.llm._repair_json(json_str)
            data = json.loads(repaired_json)
            
            suggested_relationships = []
            if "relationships" in data and isinstance(data["relationships"], list):
                suggested_relationships = data["relationships"]
            
            # Process the suggested relationships
            new_relationships = []
            modifications = []
            
            # Get existing relationship pairs for duplicate checking
            existing_pairs = set()
            for rel in existing_relationships:
                if "source_id" in rel and "target_id" in rel and "relationship_type" in rel:
                    existing_pairs.add((rel["source_id"], rel["target_id"], rel["relationship_type"]))
            
            for rel in suggested_relationships:
                if not isinstance(rel, dict) or "source_id" not in rel or "target_id" not in rel:
                    continue
                    
                action = rel.get("action", "add")
                rel_type = rel.get("relationship_type", "related_to")
                
                # Skip if it's a duplicate of an existing relationship
                if (rel["source_id"], rel["target_id"], rel_type) in existing_pairs:
                    continue
                
                # Validate that both concepts exist
                if rel["source_id"] not in concept_map or rel["target_id"] not in concept_map:
                    continue
                    
                # Prevent self-relationships
                if rel["source_id"] == rel["target_id"]:
                    continue
                
                # Create new relationship object
                relationship = {
                    "id": str(uuid.uuid4()),
                    "source_id": rel["source_id"],
                    "target_id": rel["target_id"],
                    "relationship_type": rel_type,
                    "strength": rel.get("strength", 0.5),
                    "description": rel.get("description", ""),
                    "created_at": datetime.now().isoformat(),
                    "suggested": True
                }
                
                if action == "add":
                    new_relationships.append(relationship)
                elif action == "modify":
                    modifications.append(relationship)
            
            logger.info(f"Suggested {len(new_relationships)} new relationships and {len(modifications)} modifications")
            
            # Cache the result (just the new relationships)
            self._add_to_cache(cache_key, new_relationships)
            
            # Return the new relationships - modifications would be handled separately
            return new_relationships
        except Exception as e:
            logger.error(f"Error improving relationships: {e}")
            logger.debug(traceback.format_exc())
            return []
    
    def _improve_relationships_focused(self, concepts: List[Dict[str, Any]], 
                                     existing_relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Use a focused approach for large relationship sets."""
        # Identify concepts with few or no relationships
        concept_ids = set(c["id"] for c in concepts if "id" in c)
        
        # Count relationships per concept
        relationship_counts = {concept_id: 0 for concept_id in concept_ids}
        for rel in existing_relationships:
            if "source_id" in rel and rel["source_id"] in concept_ids:
                relationship_counts[rel["source_id"]] = relationship_counts.get(rel["source_id"], 0) + 1
            if "target_id" in rel and rel["target_id"] in concept_ids:
                relationship_counts[rel["target_id"]] = relationship_counts.get(rel["target_id"], 0) + 1
        
        # Find concepts with few relationships (less than average)
        total_rels = sum(relationship_counts.values())
        avg_rels = total_rels / len(concept_ids) if concept_ids else 0
        threshold = max(1, avg_rels / 2)
        
        focus_concepts = [c for c in concepts if c.get("id") in concept_ids and 
                         relationship_counts.get(c.get("id"), 0) < threshold]
        
        # Limit to at most 20 focus concepts
        if len(focus_concepts) > 20:
            focus_concepts = focus_concepts[:20]
        
        logger.info(f"Focusing on {len(focus_concepts)} concepts with few relationships")
        
        # Find relationships involving these focus concepts
        focus_ids = set(c.get("id") for c in focus_concepts)
        focus_relationships = [r for r in existing_relationships if 
                             r.get("source_id") in focus_ids or r.get("target_id") in focus_ids]
        
        # Generate additional relationships focusing on these concepts
        added_relationships = self.improve_relationships(focus_concepts, focus_relationships)
        
        # Mark these as from focused analysis
        for rel in added_relationships:
            rel["metadata"] = rel.get("metadata", {})
            rel["metadata"]["from_focused_analysis"] = True
        
        return added_relationships
    
    @log_execution_time
    def suggest_learning_materials(self, concept: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest learning materials and resources for a specific concept."""
        if not concept or "id" not in concept or "name" not in concept:
            return concept
        
        logger.info(f"Suggesting learning materials for concept: {concept.get('name')}")
        
        # Check cache first
        cache_key = f"learning_materials_{concept.get('id')}_{hash(str(concept))}"
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            return cached_result
        
        custom_prompt = (
            f"Provide learning materials and resources for teaching this educational concept:\n\n"
            f"Name: {concept['name']}\n"
            f"Description: {concept.get('description', 'No description provided')}\n"
            f"Type: {concept.get('concept_type', 'topic')}\n"
            f"Difficulty: {concept.get('difficulty', 'intermediate')}\n\n"
            "Suggest the following resources:\n"
            "1. 2-3 types of instructional materials (videos, readings, interactive simulations, etc.)\n"
            "2. 1-2 practice activities or exercises\n"
            "3. 1 assessment approach suitable for this concept\n"
            "4. Key terminology or vocabulary students should know\n\n"
            "Format your response as a JSON object with these sections."
        )
        
        system_prompt = "You are an expert educational content curator."
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": custom_prompt}
        ]
        
        try:
            # GPT-3.5 is sufficient for this task
            json_str = self.llm._call_openai_with_retry(
                messages=messages,
                model="gpt-3.5-turbo",
                temperature=0.4,
                max_tokens=1500,
                json_mode=True
            )
            
            if not json_str:
                logger.warning(f"Failed to generate learning materials for {concept['name']}")
                return concept
                
            repaired_json = self.llm._repair_json(json_str)
            data = json.loads(repaired_json)
            
            # Add suggested materials to the concept
            enhanced_concept = concept.copy()
            
            # Add learning materials to metadata
            if not enhanced_concept.get("metadata"):
                enhanced_concept["metadata"] = {}
            
            enhanced_concept["metadata"]["learning_materials"] = data
            enhanced_concept["metadata"]["learning_materials_generated_at"] = datetime.now().isoformat()
            
            logger.info(f"Successfully generated learning materials for {concept['name']}")
            
            # Cache the result
            self._add_to_cache(cache_key, enhanced_concept)
            
            return enhanced_concept
        except Exception as e:
            logger.error(f"Error suggesting learning materials: {e}")
            logger.debug(traceback.format_exc())
            return concept
    
    @log_execution_time
    def create_hierarchical_structure(self, 
                                   domain_name: str, 
                                   concepts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a hierarchical structure from flat concepts."""
        if not concepts:
            return {"hierarchy": []}
        
        logger.info(f"Creating hierarchical structure for domain '{domain_name}' with {len(concepts)} concepts")
        
        # Check cache first
        cache_key = f"hierarchy_{domain_name}_{hash(str(concepts))}"
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            return cached_result
        
        # For large concept sets, use a chunked approach
        if len(concepts) > 40:
            logger.info("Using chunked approach for large concept set")
            return self._create_hierarchy_chunked(domain_name, concepts)
        
        # Create concept summaries with indices
        concept_text = []
        for i, concept in enumerate(concepts):
            if "name" not in concept:
                continue
                
            description = concept.get("description", "")
            description_snippet = description[:100] + "..." if description and len(description) > 100 else description
            
            summary = f"{i+1}. {concept['name']}: {description_snippet}"
            concept_text.append(summary)
        
        concepts_text = "\n".join(concept_text)
        
        custom_prompt = (
            f"Organize these concepts from the domain '{domain_name}' into a hierarchical structure:\n\n"
            f"{concepts_text}\n\n"
            "Create a meaningful hierarchy with at most 4 levels:\n"
            "1. Domain (top level - there should be just one, the domain itself)\n"
            "2. Subject areas (major subdivisions of the domain)\n"
            "3. Topics (specific areas within subject areas)\n"
            "4. Subtopics/concepts (specific components of topics)\n\n"
            "For each item in the hierarchy, include:\n"
            "- id: Use the index number from the list above (e.g., the concept at position 3 would have id '3')\n"
            "- name: The name of the concept\n"
            "- level: The level in the hierarchy (1=domain, 2=subject, 3=topic, 4=subtopic)\n"
            "- parent_id: The id of the parent item (null for the domain at the top level)\n\n"
            "Return a JSON object with a 'hierarchy' array containing all these items organized properly."
        )
        
        system_prompt = "You are an expert in taxonomy and knowledge organization."
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": custom_prompt}
        ]
        
        try:
            # Use GPT-4 for better hierarchical organization when available
            model = "gpt-4-turbo" if getattr(self.llm.config, 'allow_gpt4', False) else "gpt-3.5-turbo"
            
            json_str = self.llm._call_openai_with_retry(
                messages=messages,
                model=model,
                temperature=0.3,
                max_tokens=3000,
                json_mode=True
            )
            
            if not json_str:
                logger.error("Failed to create hierarchical structure")
                return {"hierarchy": []}
                
            repaired_json = self.llm._repair_json(json_str)
            data = json.loads(repaired_json)
            
            hierarchy = []
            if "hierarchy" in data and isinstance(data["hierarchy"], list):
                hierarchy = data["hierarchy"]
            
            # Map the hierarchy items back to the actual concept objects
            result = {"hierarchy": [], "domain_name": domain_name}
            
            for item in hierarchy:
                item_id = item.get("id")
                if not item_id:
                    continue
                    
                # Handle both string and integer IDs
                try:
                    if isinstance(item_id, str) and item_id.isdigit():
                        index = int(item_id) - 1
                    elif isinstance(item_id, int):
                        index = item_id - 1
                    else:
                        continue
                        
                    if 0 <= index < len(concepts):
                        # Create new object with hierarchy information
                        hierarchical_concept = concepts[index].copy()
                        hierarchical_concept["hierarchy_level"] = item.get("level")
                        hierarchical_concept["parent_id"] = item.get("parent_id")
                        result["hierarchy"].append(hierarchical_concept)
                except (ValueError, TypeError):
                    continue
            
            logger.info(f"Created hierarchical structure with {len(result['hierarchy'])} organized concepts")
            
            # Cache the result
            self._add_to_cache(cache_key, result)
            
            return result
        except Exception as e:
            logger.error(f"Error creating hierarchical structure: {e}")
            logger.debug(traceback.format_exc())
            return {"hierarchy": []}
    
    def _create_hierarchy_chunked(self, domain_name: str, concepts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a hierarchical structure for large concept sets using chunking."""
        # First, identify high-level categories
        category_prompt = (
            f"For the domain '{domain_name}', suggest 5-8 major categories that would effectively organize "
            "the knowledge area. These will become the top-level divisions of our hierarchy.\n\n"
            "For each category, provide:\n"
            "1. A clear name\n"
            "2. A brief description\n"
            "3. Keywords that would help identify concepts belonging to this category\n\n"
            "Return your response as a JSON array of category objects."
        )
        
        system_prompt = "You are an expert in knowledge organization and taxonomy."
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": category_prompt}
        ]
        
        try:
            # Get categories from LLM
            json_str = self.llm._call_openai_with_retry(
                messages=messages,
                model="gpt-4-turbo" if getattr(self.llm.config, 'allow_gpt4', False) else "gpt-3.5-turbo",
                temperature=0.3,
                max_tokens=2000,
                json_mode=True
            )
            
            if not json_str:
                logger.error("Failed to generate categories for hierarchical structure")
                return {"hierarchy": []}
                
            repaired_json = self.llm._repair_json(json_str)
            data = json.loads(repaired_json)
            
            categories = []
            if isinstance(data, list):
                categories = data
            elif isinstance(data, dict) and "categories" in data:
                categories = data["categories"]
            
            if not categories:
                logger.error("No categories were generated")
                return {"hierarchy": []}
            
            logger.info(f"Generated {len(categories)} categories for domain structure")
            
            # Create a domain concept
            domain_concept = {
                "id": str(uuid.uuid4()),
                "name": domain_name,
                "description": f"The domain of {domain_name}",
                "concept_type": "domain",
                "hierarchy_level": 1,
                "parent_id": None
            }
            
            # Create category concepts
            category_concepts = []
            for i, category in enumerate(categories):
                if not isinstance(category, dict) or "name" not in category:
                    continue
                    
                category_concept = {
                    "id": str(uuid.uuid4()),
                    "name": category["name"],
                    "description": category.get("description", f"Category: {category['name']}"),
                    "concept_type": "subject_area",
                    "hierarchy_level": 2,
                    "parent_id": domain_concept["id"],
                    "keywords": category.get("keywords", [])
                }
                category_concepts.append(category_concept)
            
            # Now assign each concept to a category
            assigned_concepts = []
            for concept in concepts:
                if "name" not in concept or "description" not in concept:
                    continue
                
                best_category = None
                best_score = -1
                
                # Simple keyword matching to find the best category
                for category in category_concepts:
                    score = 0
                    
                    # Check name and description for category name
                    if category["name"].lower() in concept["name"].lower():
                        score += 3
                        
                    if category["name"].lower() in concept["description"].lower():
                        score += 2
                    
                    # Check for keywords
                    for keyword in category.get("keywords", []):
                        if keyword.lower() in concept["name"].lower():
                            score += 2
                        if keyword.lower() in concept["description"].lower():
                            score += 1
                    
                    if score > best_score:
                        best_score = score
                        best_category = category
                
                # If no good match, assign to first category
                if best_score <= 0 and category_concepts:
                    best_category = category_concepts[0]
                
                if best_category:
                    concept_copy = concept.copy()
                    concept_copy["hierarchy_level"] = 3
                    concept_copy["parent_id"] = best_category["id"]
                    assigned_concepts.append(concept_copy)
                else:
                    # Fallback: assign directly to domain
                    concept_copy = concept.copy()
                    concept_copy["hierarchy_level"] = 2
                    concept_copy["parent_id"] = domain_concept["id"]
                    assigned_concepts.append(concept_copy)
            
            # Combine all hierarchy elements
            hierarchy = [domain_concept] + category_concepts + assigned_concepts
            
            result = {"hierarchy": hierarchy, "domain_name": domain_name}
            logger.info(f"Created chunked hierarchical structure with {len(hierarchy)} elements")
            return result
            
        except Exception as e:
            logger.error(f"Error in chunked hierarchy creation: {e}")
            logger.debug(traceback.format_exc())
            return {"hierarchy": []}