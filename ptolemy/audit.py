#!/usr/bin/env python
"""
Ptolemy Knowledge Graph Audit Tool
==================================
Performs comprehensive analysis and validation of Ptolemy knowledge graphs.
Identifies quality issues, structural problems, and potential improvements.
"""

import os
import sys
import json
import time
import logging
import asyncio
import argparse
from typing import Dict, List, Set, Any, Optional, Tuple, Counter
from collections import defaultdict
import aiohttp
import networkx as nx
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("knowledge_graph_audit.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("knowledge_graph_audit")

# Load environment variables
load_dotenv()

# Configuration
API_BASE_URL = os.getenv("PTOLEMY_API_URL", "http://localhost:8000/api/v1")
BEARER_TOKEN = os.getenv("PTOLEMY_BEARER_TOKEN")

# Constants
CONCEPT_TYPES = ["domain", "subject", "topic", "subtopic", "term", "skill"]
DIFFICULTY_LEVELS = ["beginner", "intermediate", "advanced", "expert"]
RELATIONSHIP_TYPES = [
    "prerequisite", "builds_on", "related_to", "part_of", "example_of", "contrasts_with"
]
DIFFICULTY_RANK = {
    "beginner": 1,
    "intermediate": 2,
    "advanced": 3,
    "expert": 4
}

class PtolemyClient:
    """Client for interacting with the Ptolemy Knowledge Map API."""
    
    def __init__(self, base_url: str, bearer_token: Optional[str] = None):
        self.base_url = base_url
        self.headers = {"Content-Type": "application/json"}
        if bearer_token:
            self.headers["Authorization"] = f"Bearer {bearer_token}"
        
    async def health_check(self, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Check the health of the Ptolemy API."""
        try:
            url = f"{self.base_url}/health"
            async with session.get(url, headers=self.headers) as response:
                if response.status == 200:
                    return await response.json()
                logger.error(f"Health check failed with status: {response.status}")
                return {"status": "error", "details": f"HTTP {response.status}"}
        except Exception as e:
            logger.error(f"Error during health check: {str(e)}")
            return {"status": "error", "details": str(e)}
    
    async def list_concepts(
        self,
        session: aiohttp.ClientSession,
        skip: int = 0,
        limit: int = 100,
        concept_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List concepts with pagination and optional filtering."""
        try:
            url = f"{self.base_url}/concepts/?skip={skip}&limit={limit}"
            if concept_type:
                url += f"&concept_type={concept_type}"
                
            async with session.get(url, headers=self.headers) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to list concepts: HTTP {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error listing concepts: {str(e)}")
            return []
    
    async def get_concept(
        self,
        session: aiohttp.ClientSession,
        concept_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get a specific concept by ID."""
        try:
            url = f"{self.base_url}/concepts/{concept_id}"
            async with session.get(url, headers=self.headers) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to get concept {concept_id}: HTTP {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error getting concept {concept_id}: {str(e)}")
            return None
    
    async def get_concept_relationships(
        self,
        session: aiohttp.ClientSession,
        concept_id: str,
        direction: str = "both"
    ) -> List[Dict[str, Any]]:
        """Get relationships for a concept."""
        try:
            url = f"{self.base_url}/concepts/{concept_id}/relationships?direction={direction}"
            async with session.get(url, headers=self.headers) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to get relationships for {concept_id}: HTTP {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error getting relationships for {concept_id}: {str(e)}")
            return []
    
    async def get_domain_structure(
        self,
        session: aiohttp.ClientSession,
        domain_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get the structure of a domain."""
        try:
            url = f"{self.base_url}/domains/{domain_id}/structure"
            async with session.get(url, headers=self.headers) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to get domain structure for {domain_id}: HTTP {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error getting domain structure for {domain_id}: {str(e)}")
            return None

class KnowledgeGraphAuditor:
    """Performs comprehensive audit of a Ptolemy knowledge graph."""
    
    def __init__(self, client: PtolemyClient):
        self.client = client
        self.concepts = {}  # ID -> concept data
        self.relationships = []  # List of all relationships
        self.graph = nx.DiGraph()  # NetworkX graph representation
        self.issues = []  # List of detected issues
        self.metrics = {}  # Computed metrics about the graph
        self.domain_ids = []  # List of domain concept IDs
        self.concepts_loaded = False
        self.relationships_loaded = False
        
    async def load_all_concepts(self, session: aiohttp.ClientSession) -> bool:
        """Load all concepts from the API."""
        print("Loading concepts...")
        all_concepts = []
        skip = 0
        limit = 100
        
        while True:
            batch = await self.client.list_concepts(session, skip=skip, limit=limit)
            if not batch:
                if skip == 0:
                    logger.error("Failed to load any concepts")
                    return False
                break
                
            all_concepts.extend(batch)
            print(f"  Loaded {len(all_concepts)} concepts so far...")
            
            if len(batch) < limit:
                break
                
            skip += limit
            
        print(f"Loaded {len(all_concepts)} concepts total")
        
        # Process and store concepts
        self.concepts = {}
        for concept in all_concepts:
            self.concepts[concept["id"]] = concept
            
            # Track domain concepts for later use
            if concept.get("concept_type") == "domain":
                self.domain_ids.append(concept["id"])
                
        self.concepts_loaded = True
        return True
    
    async def load_all_relationships(self, session: aiohttp.ClientSession) -> bool:
        """Load relationships for all concepts."""
        if not self.concepts_loaded:
            logger.error("Cannot load relationships before loading concepts")
            return False
            
        print(f"Loading relationships for {len(self.concepts)} concepts...")
        self.relationships = []
        relationship_map = {}  # Track unique relationships

        concept_ids = list(self.concepts.keys())
        progress_interval = max(1, len(concept_ids) // 10)
        
        for i, concept_id in enumerate(concept_ids):
            if i % progress_interval == 0 or i == 0:
                print(f"  Progress: {i}/{len(concept_ids)} concepts processed")
                
            # Get relationships for this concept
            relationships = await self.client.get_concept_relationships(session, concept_id)
            
            # Debug the first relationship to see its structure
            if i == 0 and relationships:
                logger.info(f"First relationship structure: {relationships[0]}")
                
            # Process each relationship
            for rel in relationships:
                try:
                    # Handle different possible response formats
                    if "relationship" in rel:
                        # Extract from nested structure
                        rel_data = rel["relationship"]
                        source_id = rel_data.get("source_id", rel.get("source_id"))
                        target_id = rel_data.get("target_id", rel.get("target_id"))
                        rel_type = rel_data.get("relationship_type", rel_data.get("type", "unknown"))
                    else:
                        # Direct properties
                        source_id = rel.get("source_id")
                        target_id = rel.get("target_id") 
                        rel_type = rel.get("relationship_type", rel.get("type", "unknown"))
                        
                    # If still missing key fields, try guessing from context
                    if not source_id and "from" in rel:
                        source_id = rel.get("from", {}).get("id", rel.get("from_id"))
                    if not target_id and "to" in rel:
                        target_id = rel.get("to", {}).get("id", rel.get("to_id"))
                    
                    # For nested relationship data
                    if not source_id and "from_concept" in rel:
                        source_id = rel.get("from_concept", {}).get("id", rel.get("from_id"))
                    if not target_id and "to_concept" in rel:
                        target_id = rel.get("to_concept", {}).get("id", rel.get("to_id"))
                        
                    # Last resort - use current concept ID as source
                    if not source_id:
                        source_id = concept_id
                        
                    # Skip if missing critical data
                    if not source_id or not target_id or not rel_type:
                        # Try to extract more info for debugging
                        logger.warning(f"Incomplete relationship data: {rel}")
                        continue
                        
                    # Add to our internal representation
                    rel_key = (source_id, target_id, rel_type)
                    if rel_key not in relationship_map:
                        # Add minimal required data for our processing
                        clean_rel = {
                            "source_id": source_id,
                            "target_id": target_id,
                            "relationship_type": rel_type,
                            "strength": rel.get("strength", 0.5),
                            "id": rel.get("id", f"{source_id}-{target_id}")
                        }
                        relationship_map[rel_key] = clean_rel
                        self.relationships.append(clean_rel)
                        
                except Exception as e:
                    logger.warning(f"Error processing relationship: {e} - Data: {rel}")
                    continue
        
        print(f"Loaded {len(self.relationships)} unique relationships")
        self.relationships_loaded = True
        return True
    
    def build_graph(self) -> bool:
        """Build a NetworkX graph from the loaded concepts and relationships."""
        if not self.concepts_loaded or not self.relationships_loaded:
            logger.error("Cannot build graph before loading concepts and relationships")
            return False
            
        print("Building graph representation...")
        self.graph = nx.DiGraph()
        
        # Add nodes
        for concept_id, concept in self.concepts.items():
            self.graph.add_node(
                concept_id,
                name=concept["name"],
                type=concept.get("concept_type"),
                difficulty=concept.get("difficulty"),
                importance=concept.get("importance", 0.5)
            )
        
        # Add edges
        edge_count = 0
        for rel in self.relationships:
            source_id = rel["source_id"]
            target_id = rel["target_id"]
            
            # Skip if either source or target doesn't exist
            if source_id not in self.concepts or target_id not in self.concepts:
                continue
                
            self.graph.add_edge(
                source_id,
                target_id,
                id=rel.get("id"),
                type=rel["relationship_type"],
                strength=rel.get("strength", 0.5)
            )
            edge_count += 1
            
        print(f"Built graph with {self.graph.number_of_nodes()} nodes and {edge_count} edges")
        return True
    
    def audit_graph(self) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Perform a comprehensive audit of the knowledge graph."""
        print("Running audit checks...")
        self.issues = []
        
        # Check for orphaned concepts
        self._check_orphaned_concepts()
        
        # Check for circular prerequisites
        self._check_circular_prerequisites()
        
        # Check for inconsistent difficulty levels
        self._check_difficulty_consistency()
        
        # Check for contradictory relationships
        self._check_contradictory_relationships()
        
        # Check concept quality
        self._check_concept_quality()
        
        # Calculate graph metrics
        self._calculate_metrics()
        
        print(f"Audit complete. Found {len(self.issues)} issues.")
        return self.issues, self.metrics
    
    def _check_orphaned_concepts(self):
        """Check for concepts with no relationships."""
        orphaned_count = 0
        
        for node in self.graph.nodes():
            if self.graph.degree(node) == 0:
                concept = self.concepts[node]
                self.issues.append({
                    "issue_type": "orphaned_concept",
                    "severity": "warning",
                    "concept_id": node,
                    "concept_name": concept["name"],
                    "concept_type": concept.get("concept_type"),
                    "description": f"Concept '{concept['name']}' has no relationships",
                    "recommendation": "Connect this concept to the knowledge graph with appropriate relationships"
                })
                orphaned_count += 1
                
        if orphaned_count > 0:
            print(f"  Found {orphaned_count} orphaned concepts")
    
    def _check_circular_prerequisites(self):
        """Check for circular prerequisite chains."""
        prereq_subgraph = nx.DiGraph()
        
        # Create a subgraph of only prerequisite relationships
        for u, v, data in self.graph.edges(data=True):
            if data.get("type") == "prerequisite":
                prereq_subgraph.add_edge(u, v)
        
        try:
            cycles = list(nx.simple_cycles(prereq_subgraph))
            
            for cycle in cycles:
                # Get concept names for the cycle
                cycle_names = [self.concepts[node]["name"] for node in cycle]
                cycle_str = " → ".join(cycle_names)
                
                self.issues.append({
                    "issue_type": "circular_prerequisites",
                    "severity": "error",
                    "concept_ids": cycle,
                    "concept_names": cycle_names,
                    "description": f"Circular prerequisite chain detected: {cycle_str}",
                    "recommendation": "Break the circular dependency by removing or changing at least one relationship"
                })
                
            if cycles:
                print(f"  Found {len(cycles)} circular prerequisite chains")
                
        except nx.NetworkXNoCycle:
            # No cycles found
            pass
        except Exception as e:
            logger.error(f"Error checking for cycles: {e}")
    
    def _check_difficulty_consistency(self):
        """Check for inconsistent difficulty levels between prerequisites."""
        inconsistent_count = 0
        
        for u, v, data in self.graph.edges(data=True):
            if data.get("type") == "prerequisite" and u in self.concepts and v in self.concepts:
                source_concept = self.concepts[u]
                target_concept = self.concepts[v]
                
                source_difficulty = source_concept.get("difficulty")
                target_difficulty = target_concept.get("difficulty")
                
                if (source_difficulty and target_difficulty and 
                    DIFFICULTY_RANK.get(source_difficulty, 0) > DIFFICULTY_RANK.get(target_difficulty, 0)):
                    
                    self.issues.append({
                        "issue_type": "inconsistent_difficulty",
                        "severity": "warning",
                        "source_id": u,
                        "source_name": source_concept["name"],
                        "source_difficulty": source_difficulty,
                        "target_id": v,
                        "target_name": target_concept["name"],
                        "target_difficulty": target_difficulty,
                        "description": f"Prerequisite '{source_concept['name']}' ({source_difficulty}) "
                                       f"is more difficult than '{target_concept['name']}' ({target_difficulty})",
                        "recommendation": "Adjust difficulty levels or review the prerequisite relationship"
                    })
                    inconsistent_count += 1
                    
        if inconsistent_count > 0:
            print(f"  Found {inconsistent_count} inconsistent difficulty relationships")
    
    def _check_contradictory_relationships(self):
        """Check for contradictory or redundant relationships between concepts."""
        contradictions = {
            ("prerequisite", "builds_on"),
            ("part_of", "example_of")
        }
        
        contradiction_count = 0
        bidirectional_prereq_count = 0
        
        # Check for bidirectional prerequisites (usually a problem)
        for u, v in self.graph.edges():
            if v in self.graph.predecessors(u):
                # There's an edge in both directions, check relationship types
                forward_type = self.graph.edges[u, v].get("type")
                backward_type = self.graph.edges[v, u].get("type")
                
                # Both are prerequisites - this is definitely wrong
                if forward_type == "prerequisite" and backward_type == "prerequisite":
                    source_name = self.concepts[u]["name"]
                    target_name = self.concepts[v]["name"]
                    
                    self.issues.append({
                        "issue_type": "bidirectional_prerequisites",
                        "severity": "error",
                        "concept1_id": u,
                        "concept1_name": source_name,
                        "concept2_id": v,
                        "concept2_name": target_name,
                        "description": f"Bidirectional prerequisites between '{source_name}' and '{target_name}'"
                                       f" (both can't be prerequisites of each other)",
                        "recommendation": "Remove one of the prerequisite relationships or change its type"
                    })
                    bidirectional_prereq_count += 1
                
                # Check for other contradictory relationship pairs
                if (forward_type, backward_type) in contradictions or (backward_type, forward_type) in contradictions:
                    source_name = self.concepts[u]["name"]
                    target_name = self.concepts[v]["name"]
                    
                    self.issues.append({
                        "issue_type": "contradictory_relationships",
                        "severity": "warning",
                        "concept1_id": u,
                        "concept1_name": source_name,
                        "concept2_id": v,
                        "concept2_name": target_name,
                        "relationship1": forward_type,
                        "relationship2": backward_type,
                        "description": f"Contradictory relationships between '{source_name}' and '{target_name}': "
                                      f"'{forward_type}' and '{backward_type}'",
                        "recommendation": "Review these relationships and resolve the contradiction"
                    })
                    contradiction_count += 1
        
        if bidirectional_prereq_count > 0:
            print(f"  Found {bidirectional_prereq_count} bidirectional prerequisite issues")
        if contradiction_count > 0:
            print(f"  Found {contradiction_count} contradictory relationships")
    
    def _check_concept_quality(self):
        """Check the quality of concept data."""
        short_descriptions = 0
        missing_keywords = 0
        
        for concept_id, concept in self.concepts.items():
            # Check description length
            description = concept.get("description", "")
            if len(description) < 50:  # Arbitrary threshold for a too-short description
                self.issues.append({
                    "issue_type": "short_description",
                    "severity": "info",
                    "concept_id": concept_id,
                    "concept_name": concept["name"],
                    "description": f"Concept '{concept['name']}' has a very short description ({len(description)} chars)",
                    "recommendation": "Expand the description to provide more comprehensive information"
                })
                short_descriptions += 1
            
            # Check keywords
            keywords = concept.get("keywords", [])
            if not keywords or len(keywords) < 3:  # Arbitrary threshold for too few keywords
                self.issues.append({
                    "issue_type": "missing_keywords",
                    "severity": "info",
                    "concept_id": concept_id,
                    "concept_name": concept["name"],
                    "description": f"Concept '{concept['name']}' has few or no keywords",
                    "recommendation": "Add relevant keywords to improve searchability and context"
                })
                missing_keywords += 1
        
        if short_descriptions > 0:
            print(f"  Found {short_descriptions} concepts with short descriptions")
        if missing_keywords > 0:
            print(f"  Found {missing_keywords} concepts with missing or few keywords")
    
    def _calculate_metrics(self):
        """Calculate various metrics about the knowledge graph."""
        metrics = {}
        
        # Basic counts
        metrics["total_concepts"] = len(self.concepts)
        metrics["total_relationships"] = len(self.relationships)
        metrics["total_domains"] = len(self.domain_ids)
        
        # Graph metrics
        metrics["graph_density"] = nx.density(self.graph)
        metrics["avg_degree"] = sum(dict(self.graph.degree()).values()) / max(1, len(self.graph))
        
        # Count by concept type
        concept_types = defaultdict(int)
        for concept in self.concepts.values():
            concept_type = concept.get("concept_type")
            if concept_type:
                concept_types[concept_type] += 1
        metrics["concept_types"] = dict(concept_types)
        
        # Count by difficulty level
        difficulty_levels = defaultdict(int)
        for concept in self.concepts.values():
            difficulty = concept.get("difficulty")
            if difficulty:
                difficulty_levels[difficulty] += 1
        metrics["difficulty_levels"] = dict(difficulty_levels)
        
        # Count by relationship type
        relationship_types = defaultdict(int)
        for rel in self.relationships:
            relationship_types[rel["relationship_type"]] += 1
        metrics["relationship_types"] = dict(relationship_types)
        
        # Connected components
        try:
            metrics["weakly_connected_components"] = nx.number_weakly_connected_components(self.graph)
        except Exception as e:
            logger.error(f"Error calculating connected components: {e}")
            metrics["weakly_connected_components"] = "error"
        
        # Average clustering coefficient (using undirected version of graph)
        try:
            undirected_graph = self.graph.to_undirected()
            metrics["avg_clustering"] = nx.average_clustering(undirected_graph)
        except Exception as e:
            logger.error(f"Error calculating clustering: {e}")
            metrics["avg_clustering"] = None
            
        # Try to calculate betweenness centrality for a sample of nodes if graph is large
        try:
            if len(self.graph) > 1000:
                # Sample 1000 nodes for large graphs
                sample_nodes = list(self.graph.nodes())[:1000]
                centrality = nx.betweenness_centrality(self.graph, k=min(100, len(self.graph)))
            else:
                centrality = nx.betweenness_centrality(self.graph)
                
            # Find top nodes by centrality
            top_centrality = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:20]
            top_central_concepts = []
            
            for node_id, score in top_centrality:
                if node_id in self.concepts:
                    top_central_concepts.append({
                        "id": node_id,
                        "name": self.concepts[node_id]["name"],
                        "type": self.concepts[node_id].get("concept_type"),
                        "centrality_score": score
                    })
            
            metrics["top_central_concepts"] = top_central_concepts
        except Exception as e:
            logger.error(f"Error calculating centrality: {str(e)}")
            metrics["top_central_concepts"] = []
            
        self.metrics = metrics
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """Generate a comprehensive audit report."""
        report_lines = []
        
        # Header
        report_lines.append("# Ptolemy Knowledge Graph Audit Report")
        report_lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Summary
        report_lines.append("## Summary")
        report_lines.append(f"- **Total concepts:** {self.metrics.get('total_concepts', 0)}")
        report_lines.append(f"- **Total relationships:** {self.metrics.get('total_relationships', 0)}")
        report_lines.append(f"- **Total domains:** {self.metrics.get('total_domains', 0)}")
        report_lines.append(f"- **Graph density:** {self.metrics.get('graph_density', 0):.4f}")
        report_lines.append(f"- **Average degree:** {self.metrics.get('avg_degree', 0):.2f}")
        report_lines.append(f"- **Weakly connected components:** {self.metrics.get('weakly_connected_components', 0)}")
        report_lines.append("")
        
        # Issues summary
        severity_counts = Counter(issue.get("severity", "unknown") for issue in self.issues)
        report_lines.append("## Issues Overview")
        report_lines.append(f"- **Total issues found:** {len(self.issues)}")
        report_lines.append(f"  - Critical: {severity_counts.get('critical', 0)}")
        report_lines.append(f"  - Error: {severity_counts.get('error', 0)}")
        report_lines.append(f"  - Warning: {severity_counts.get('warning', 0)}")
        report_lines.append(f"  - Info: {severity_counts.get('info', 0)}")
        report_lines.append("")
        
        # Distribution by concept type
        report_lines.append("## Concept Type Distribution")
        concept_types = self.metrics.get("concept_types", {})
        total_concepts = self.metrics.get("total_concepts", 1)  # Avoid division by zero
        
        for concept_type in CONCEPT_TYPES:
            count = concept_types.get(concept_type, 0)
            percentage = (count / total_concepts) * 100
            report_lines.append(f"- **{concept_type}:** {count} ({percentage:.1f}%)")
        report_lines.append("")
        
        # Distribution by difficulty level
        report_lines.append("## Difficulty Level Distribution")
        difficulty_levels = self.metrics.get("difficulty_levels", {})
        
        for difficulty in DIFFICULTY_LEVELS:
            count = difficulty_levels.get(difficulty, 0)
            percentage = (count / total_concepts) * 100
            report_lines.append(f"- **{difficulty}:** {count} ({percentage:.1f}%)")
        report_lines.append("")
        
        # Distribution by relationship type
        report_lines.append("## Relationship Type Distribution")
        relationship_types = self.metrics.get("relationship_types", {})
        total_relationships = self.metrics.get("total_relationships", 1)  # Avoid division by zero
        
        for rel_type in RELATIONSHIP_TYPES:
            count = relationship_types.get(rel_type, 0)
            percentage = (count / total_relationships) * 100
            report_lines.append(f"- **{rel_type}:** {count} ({percentage:.1f}%)")
        report_lines.append("")
        
        # Top central concepts
        report_lines.append("## Top Central Concepts")
        top_concepts = self.metrics.get("top_central_concepts", [])
        
        for concept in top_concepts:
            report_lines.append(f"- **{concept['name']}** ({concept.get('type', 'unknown')}) - "
                               f"Centrality Score: {concept['centrality_score']:.4f}")
        report_lines.append("")
        
        # Detailed issue listing
        report_lines.append("## Detailed Issues")
        
        # Group issues by type for better organization
        issues_by_type = defaultdict(list)
        for issue in self.issues:
            issues_by_type[issue["issue_type"]].append(issue)
            
        for issue_type, issues in issues_by_type.items():
            report_lines.append(f"### {issue_type.replace('_', ' ').title()} ({len(issues)})")
            
            # Sort issues by severity
            severity_order = {"critical": 0, "error": 1, "warning": 2, "info": 3}
            sorted_issues = sorted(issues, key=lambda x: severity_order.get(x.get("severity", "info"), 4))
            
            # If there are many issues of this type, just show the first few
            if len(sorted_issues) > 10:
                report_lines.append(f"Showing 10 of {len(sorted_issues)} issues:")
                sorted_issues = sorted_issues[:10]
                
            # Add each issue
            for i, issue in enumerate(sorted_issues, 1):
                severity = issue.get("severity", "").upper()
                description = issue.get("description", "No description")
                recommendation = issue.get("recommendation", "No recommendation")
                
                report_lines.append(f"**{i}. {severity}: {description}**")
                
                # Add context based on issue type
                if "concept_name" in issue:
                    report_lines.append(f"   - Concept: {issue['concept_name']}")
                    
                if "concept_names" in issue:
                    report_lines.append(f"   - Concepts: {', '.join(issue['concept_names'])}")
                    
                if "source_name" in issue and "target_name" in issue:
                    source_name = issue["source_name"]
                    target_name = issue["target_name"]
                    report_lines.append(f"   - Relationship: {source_name} → {target_name}")
                    
                report_lines.append(f"   - Recommendation: {recommendation}")
                report_lines.append("")
            
            report_lines.append("")
        
        # Write to file if specified
        report_text = "\n".join(report_lines)
        if output_file:
            try:
                with open(output_file, "w") as f:
                    f.write(report_text)
                print(f"Report written to: {output_file}")
            except Exception as e:
                logger.error(f"Error writing report to {output_file}: {str(e)}")
                
        return report_text
    
    def generate_visualizations(self, output_dir: str):
        """Generate visualizations of the knowledge graph structure."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        print(f"Generating visualizations in {output_dir}...")
        
        # 1. Concept Type Distribution
        try:
            plt.figure(figsize=(10, 6))
            concept_types = self.metrics.get("concept_types", {})
            counts = [concept_types.get(ct, 0) for ct in CONCEPT_TYPES]
            
            plt.bar(CONCEPT_TYPES, counts)
            plt.title("Concept Type Distribution")
            plt.xlabel("Concept Type")
            plt.ylabel("Count")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "concept_types.png"))
            plt.close()
            
            print("  Generated concept type distribution chart")
        except Exception as e:
            logger.error(f"Error generating concept type visualization: {str(e)}")
        
        # 2. Relationship Type Distribution
        try:
            plt.figure(figsize=(10, 6))
            relationship_types = self.metrics.get("relationship_types", {})
            counts = [relationship_types.get(rt, 0) for rt in RELATIONSHIP_TYPES]
            
            plt.bar(RELATIONSHIP_TYPES, counts)
            plt.title("Relationship Type Distribution")
            plt.xlabel("Relationship Type")
            plt.ylabel("Count")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "relationship_types.png"))
            plt.close()
            
            print("  Generated relationship type distribution chart")
        except Exception as e:
            logger.error(f"Error generating relationship type visualization: {str(e)}")
        
        # 3. Difficulty Level Distribution
        try:
            plt.figure(figsize=(10, 6))
            difficulty_levels = self.metrics.get("difficulty_levels", {})
            counts = [difficulty_levels.get(dl, 0) for dl in DIFFICULTY_LEVELS]
            
            plt.bar(DIFFICULTY_LEVELS, counts)
            plt.title("Difficulty Level Distribution")
            plt.xlabel("Difficulty Level")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "difficulty_levels.png"))
            plt.close()
            
            print("  Generated difficulty level distribution chart")
        except Exception as e:
            logger.error(f"Error generating difficulty level visualization: {str(e)}")
        
        # 4. Graph Visualization (if not too large)
        if len(self.graph) <= 150:  # Only visualize small graphs
            try:
                plt.figure(figsize=(12, 12))
                
                # Use different colors for different concept types
                node_colors = []
                for node in self.graph.nodes():
                    if node in self.concepts:
                        concept_type = self.concepts[node].get("concept_type", "unknown")
                        color_idx = CONCEPT_TYPES.index(concept_type) if concept_type in CONCEPT_TYPES else -1
                        node_colors.append(plt.cm.tab10(color_idx % 10))
                    else:
                        node_colors.append("gray")
                
                # Use spring layout for nice visualization
                pos = nx.spring_layout(self.graph, seed=42)
                
                # Draw the graph
                nx.draw(
                    self.graph,
                    pos,
                    with_labels=False,
                    node_color=node_colors,
                    node_size=100,
                    alpha=0.8,
                    edge_color="gray",
                    arrows=True
                )
                
                # Add labels for important nodes
                high_degree_nodes = [n for n, d in self.graph.degree() if d > 5]
                labels = {n: self.concepts[n]["name"] for n in high_degree_nodes if n in self.concepts}
                nx.draw_networkx_labels(
                    self.graph,
                    pos,
                    labels=labels,
                    font_size=8,
                    font_color="black"
                )
                
                plt.axis("off")
                plt.savefig(os.path.join(output_dir, "knowledge_graph.png"), dpi=300, bbox_inches="tight")
                plt.close()
                
                print("  Generated knowledge graph visualization")
            except Exception as e:
                logger.error(f"Error generating graph visualization: {str(e)}")
        else:
            print("  Graph too large for visualization, skipping")

def print_colored(text: str, color: str = ""):
    """Print text with optional ANSI color codes."""
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "reset": "\033[0m"
    }
    
    color_code = colors.get(color.lower(), "")
    reset_code = colors["reset"]
    
    print(f"{color_code}{text}{reset_code}")

def display_issues_summary(issues: List[Dict[str, Any]]):
    """Display a summary of the issues found."""
    if not issues:
        print_colored("No issues found!", "green")
        return
        
    # Group by severity and type
    by_severity = defaultdict(int)
    by_type = defaultdict(int)
    
    for issue in issues:
        severity = issue.get("severity", "unknown")
        issue_type = issue.get("issue_type", "unknown")
        
        by_severity[severity] += 1
        by_type[issue_type] += 1
    
    # Display summary by severity
    print_colored("\nIssues by Severity:", "cyan")
    for severity in ["critical", "error", "warning", "info"]:
        count = by_severity.get(severity, 0)
        if count > 0:
            color = {
                "critical": "red",
                "error": "red",
                "warning": "yellow",
                "info": "blue"
            }.get(severity, "white")
            
            print_colored(f"  {severity.upper()}: {count}", color)
    
    # Display summary by type
    print_colored("\nIssues by Type:", "cyan")
    for issue_type, count in sorted(by_type.items(), key=lambda x: x[1], reverse=True):
        print(f"  {issue_type.replace('_', ' ').title()}: {count}")
    
    # Show some examples
    print_colored("\nExample Issues:", "cyan")
    severity_order = {"critical": 0, "error": 1, "warning": 2, "info": 3}
    sorted_issues = sorted(issues, key=lambda x: severity_order.get(x.get("severity", "info"), 4))
    
    for i, issue in enumerate(sorted_issues[:5], 1):
        severity = issue.get("severity", "unknown").upper()
        severity_color = {
            "CRITICAL": "red",
            "ERROR": "red",
            "WARNING": "yellow",
            "INFO": "blue"
        }.get(severity, "white")
        
        print_colored(f"{i}. [{severity}] {issue.get('description', 'No description')}", severity_color)

def display_metrics_summary(metrics: Dict[str, Any]):
    """Display a summary of the graph metrics."""
    print_colored("\nKnowledge Graph Metrics:", "cyan")
    
    # Basic stats
    print_colored("\nBasic Statistics:", "cyan")
    print(f"  Total Concepts: {metrics.get('total_concepts', 0)}")
    print(f"  Total Relationships: {metrics.get('total_relationships', 0)}")
    print(f"  Total Domains: {metrics.get('total_domains', 0)}")
    
    # Graph properties
    print_colored("\nGraph Properties:", "cyan")
    print(f"  Graph Density: {metrics.get('graph_density', 0):.4f}")
    print(f"  Average Degree: {metrics.get('avg_degree', 0):.2f}")
    print(f"  Weakly Connected Components: {metrics.get('weakly_connected_components', 0)}")
    
    if metrics.get("avg_clustering") is not None:
        print(f"  Average Clustering Coefficient: {metrics.get('avg_clustering', 0):.4f}")
    
    # Show top central concepts
    top_concepts = metrics.get("top_central_concepts", [])
    if top_concepts:
        print_colored("\nTop Central Concepts:", "cyan")
        for i, concept in enumerate(top_concepts[:5], 1):
            print(f"  {i}. {concept['name']} - {concept.get('type', 'unknown')}")

async def main():
    """Main function to run the audit tool."""
    parser = argparse.ArgumentParser(description="Ptolemy Knowledge Graph Audit Tool")
    parser.add_argument("--api-url", type=str, default=API_BASE_URL, help="Base URL for the Ptolemy API")
    parser.add_argument("--token", type=str, default=BEARER_TOKEN, help="Bearer token for authentication")
    parser.add_argument("--report", type=str, help="Output file for the audit report (markdown format)")
    parser.add_argument("--visualizations", type=str, help="Output directory for visualizations")
    parser.add_argument("--domain", type=str, help="Specific domain ID to audit (not implemented yet)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with more verbose output")
    
    args = parser.parse_args()
    
    # Set debug level if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        print_colored("Debug mode enabled", "yellow")
    
    # Initialize client
    client = PtolemyClient(args.api_url, args.token)
    
    # Initialize auditor
    auditor = KnowledgeGraphAuditor(client)
    
    # Run the audit
    try:
        print_colored("Starting Ptolemy Knowledge Graph Audit", "cyan")
        
        async with aiohttp.ClientSession() as session:
            # Check API health
            health = await client.health_check(session)
            if health.get("status") not in ["ok", "healthy"]:
                print_colored(f"API health check failed: {health}", "red")
                return 1
                
            print_colored(f"API health check passed: {health.get('status')}", "green")
            
            # Load concepts
            if not await auditor.load_all_concepts(session):
                print_colored("Failed to load concepts. Aborting audit.", "red")
                return 1
                
            # Load relationships
            if not await auditor.load_all_relationships(session):
                print_colored("Failed to load relationships. Aborting audit.", "red")
                return 1
                
            # Build the graph
            if not auditor.build_graph():
                print_colored("Failed to build graph. Aborting audit.", "red")
                return 1
                
            # Run the audit
            issues, metrics = auditor.audit_graph()
            
            # Display results
            print_colored("\nAudit Results", "cyan")
            display_issues_summary(issues)
            display_metrics_summary(metrics)
            
            # Generate report if requested
            if args.report:
                auditor.generate_report(args.report)
                
            # Generate visualizations if requested
            if args.visualizations:
                auditor.generate_visualizations(args.visualizations)
                
            print_colored("\nAudit completed successfully!", "green")
            
    except Exception as e:
        print_colored(f"Error during audit: {str(e)}", "red")
        logger.exception("Error during audit")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))