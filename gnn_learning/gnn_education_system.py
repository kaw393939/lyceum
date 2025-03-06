#!/usr/bin/env python
"""
Enhanced GNN-Based Educational Achievement and Recommendation System

This system uses Heterogeneous Graph Neural Networks to:
1. Track learner progress through interconnected concepts with improved relationship modeling
2. Award points based on demonstrated mastery with adaptive difficulty
3. Provide personalized content recommendations using educational psychology principles
4. Integrate with a Socratic/Stoic LLM for questioning

Major improvements over previous version:
- Heterogeneous graph representation (HeteroData) for different node and edge types
- Advanced GNN with relation-specific transformations and attention
- Task-specific training objectives instead of generic reconstruction loss
- Exploration-exploitation balance in recommendations using Thompson sampling
- Incremental graph updates instead of full rebuilds
- Proper evaluation metrics and validation
- Robust error handling with graceful degradation
- Enhanced personalization using learner feedback and adaptive paths
- Data scalability with lazy loading and caching
- Educational psychology principles incorporated in the recommendation system
"""

import argparse
import datetime
import json
import os
import random
import logging
import time
import uuid
import re
import math
import copy
from collections import defaultdict, Counter, deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Any, Union, DefaultDict, Iterator, TypeVar, Generic, Callable

import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler
from torch_geometric.nn import GATConv, Linear, HeteroConv, GCNConv, SAGEConv, MessagePassing
from torch_geometric.data import HeteroData, Data, Dataset, Batch
from torch_geometric.loader import DataLoader, NeighborLoader
from torch_geometric.transforms import ToUndirected, RemoveIsolatedNodes
from torch_geometric.utils import to_networkx, from_networkx
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
from dotenv import load_dotenv
from tqdm import tqdm
import networkx as nx
import hashlib

# Custom scatter implementations to replace torch_scatter
def scatter_mean(src, index, dim=0, dim_size=None):
    if dim_size is None:
        dim_size = index.max().item() + 1
    output = torch.zeros(dim_size, src.size(1), device=src.device)
    count = torch.zeros(dim_size, device=src.device)
    for i in range(src.size(0)):
        output[index[i]] += src[i]
        count[index[i]] += 1
    count = count.view(-1, 1).clamp(min=1)
    return output / count

def scatter_sum(src, index, dim=0, dim_size=None):
    if dim_size is None:
        dim_size = index.max().item() + 1
    output = torch.zeros(dim_size, src.size(1), device=src.device)
    for i in range(src.size(0)):
        output[index[i]] += src[i]
    return output

def scatter_max(src, index, dim=0, dim_size=None):
    if dim_size is None:
        dim_size = index.max().item() + 1
    output = torch.full((dim_size, src.size(1)), float('-inf'), device=src.device)
    for i in range(src.size(0)):
        output[index[i]] = torch.max(output[index[i]], src[i])
    return output



# Try to import openai, but don't fail if it's not installed
try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI package not found. LLM functionality will be limited.")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("education_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables (API keys, etc.)
load_dotenv()

# -----------------------------
# CONSTANTS AND CONFIGURATION
# -----------------------------

# Node types
NODE_TYPES = {
    'concept': 0,
    'question': 1,
    'resource': 2,
    'learner': 3
}

# Edge types as tuples of (source_type, relation, target_type)
# Add these reverse edge definitions
EDGE_TYPES = [
    ('question', 'tests', 'concept'),
    ('resource', 'teaches', 'concept'),
    ('learner', 'studies', 'concept'),
    ('concept', 'requires', 'concept'),
    ('learner', 'answered', 'question'),
    ('learner', 'used', 'resource'),
    # Add these reverse edges:
    ('concept', 'studied_by', 'learner'),
    ('concept', 'taught_by', 'resource'),
    ('question', 'answered_by', 'learner'),
    ('resource', 'used_by', 'learner')
]

# Learning style spectrum
LEARNING_STYLES = {
    'visual': 0.0,       # Learns best through visual media
    'balanced': 0.5,     # Equal effectiveness with different media
    'textual': 1.0       # Learns best through text
}

# Question types spectrum
QUESTION_TYPES = {
    'recall': 0.0,           # Basic knowledge recall
    'application': 0.5,      # Apply concepts to situations
    'analysis': 1.0          # Deep analysis and synthesis
}

# Media types spectrum
MEDIA_TYPES = {
    'text': 0.0,             # Text-based resources
    'video': 0.5,            # Video resources
    'interactive': 1.0       # Interactive simulations/exercises
}

# Achievement types
ACHIEVEMENT_TYPES = {
    'concept_mastery': {
        'name': 'Concept Mastery',
        'description': 'Mastered a concept with high proficiency',
        'base_points': 100
    },
    'connected_concepts': {
        'name': 'Knowledge Connections',
        'description': 'Made connections between related concepts',
        'base_points': 50
    },
    'deep_reasoning': {
        'name': 'Deep Reasoning',
        'description': 'Demonstrated sophisticated reasoning skills',
        'base_points': 75
    },
    'persistence': {
        'name': 'Persistent Scholar',
        'description': 'Showed persistence in mastering difficult concepts',
        'base_points': 25
    },
    'fast_learner': {
        'name': 'Fast Learner',
        'description': 'Quickly mastered new concepts',
        'base_points': 30
    },
    'knowledge_explorer': {
        'name': 'Knowledge Explorer',
        'description': 'Explored diverse topics beyond requirements',
        'base_points': 40
    }
}

# Spaced repetition intervals (in days)
SPACED_REPETITION_INTERVALS = [1, 3, 7, 14, 30, 90]

# Default configuration
DEFAULT_CONFIG = {
    "knowledge_graph": {
        "concepts_file": "data/concepts.json",
        "questions_file": "data/questions.json",
        "resources_file": "data/resources.json",
        "learners_file": "data/learners.json",
        "cache_dir": "data/cache",
        "max_cache_size_mb": 100
    },
    "gnn": {
        "model_type": "hetero_gat",  # Options: hetero_gat, hetero_sage, hetero_gcn
        "hidden_channels": 64,
        "num_layers": 2,
        "num_heads": 4,  # For attention-based models
        "dropout": 0.2,
        "learning_rate": 0.001,
        "weight_decay": 5e-4,
        "batch_size": 32,
        "patience": 10,  # For early stopping
        "validation_ratio": 0.2
    },
    "training": {
        "mastery_prediction_weight": 1.0,
        "resource_recommendation_weight": 0.8, 
        "question_selection_weight": 0.8,
        "path_prediction_weight": 0.5
    },
    "achievements": {
        "mastery_threshold": 0.75,
        "points_multiplier": 1.0,
        "difficulty_bonus": 0.5,  # Additional multiplier for difficult concepts
        "streak_bonus": 0.2  # Bonus for consecutive achievements
    },
    "recommendation": {
        "exploration_weight": 0.3,  # Balance between exploration and exploitation
        "recency_decay": 0.9,  # How quickly recent interactions lose importance
        "diversity_factor": 0.2,  # Encourages recommending diverse content
        "max_recommendations": 5,
        "personalization_weight": 0.7  # How much to personalize vs. general popularity
    },
    "llm": {
        "model": "gpt-3.5-turbo",
        "temperature": 0.7,
        "max_tokens": 200,
        "cache_responses": True,
        "cache_ttl_hours": 24,
        "socratic_template": "Ask a Socratic question that guides the learner to discover {concept} without directly stating the answer. Use questioning techniques that encourage critical thinking.",
        "retry_attempts": 3,
        "retry_delay_seconds": 2
    },
    "evaluation": {
        "metrics": ["auc", "precision", "recall", "ndcg"],
        "test_interval_epochs": 5,
        "save_best_model": True
    },
    "system": {
        "log_level": "INFO",
        "auto_save_interval_minutes": 10,
        "backup_count": 3,
        "thread_pool_size": 4
    }
}

# -----------------------------
# UTILITY FUNCTIONS
# -----------------------------

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from file or use defaults with validation
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing configuration settings
    """
    config = DEFAULT_CONFIG.copy()
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f)
                
            # Recursively update config with user settings
            def update_config(base, updates):
                for key, value in updates.items():
                    if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                        update_config(base[key], value)
                    else:
                        base[key] = value
            
            update_config(config, user_config)
            
            # Validate numeric settings
            for section, params in [
                ("gnn", ["hidden_channels", "num_layers", "num_heads", "dropout", 
                         "learning_rate", "weight_decay", "batch_size", "patience"]),
                ("achievements", ["mastery_threshold", "points_multiplier", 
                                  "difficulty_bonus", "streak_bonus"]),
                ("recommendation", ["exploration_weight", "recency_decay", 
                                   "diversity_factor", "personalization_weight"]),
                ("training", ["mastery_prediction_weight", "resource_recommendation_weight",
                             "question_selection_weight", "path_prediction_weight"])
            ]:
                for param in params:
                    if param in config[section]:
                        # Ensure numeric values are properly typed
                        if param in ["hidden_channels", "num_layers", "num_heads", 
                                     "batch_size", "patience"]:
                            config[section][param] = int(config[section][param])
                        else:
                            config[section][param] = float(config[section][param])
            
            logger.info(f"Loaded configuration from {config_path}")
        except (yaml.YAMLError, IOError) as e:
            logger.error(f"Error reading config file: {e}. Using default configuration.")
            return DEFAULT_CONFIG
    else:
        logger.warning(f"Config file not found at {config_path}. Using default configuration.")
    
    return config

def create_directory_if_not_exists(directory_path: str) -> None:
    """Create a directory if it doesn't exist"""
    try:
        os.makedirs(directory_path, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create directory {directory_path}: {e}")
        raise

def calculate_concept_similarity(concept1: Dict[str, Any], concept2: Dict[str, Any]) -> float:
    """
    Calculate similarity between two concepts based on their attributes
    
    Args:
        concept1: First concept
        concept2: Second concept
        
    Returns:
        Similarity score between 0 and 1
    """
    # Extract features
    features1 = [float(concept1.get("difficulty", 0.5)), 
                float(concept1.get("complexity", 0.5)),
                float(concept1.get("importance", 0.5))]
    
    features2 = [float(concept2.get("difficulty", 0.5)),
                float(concept2.get("complexity", 0.5)),
                float(concept2.get("importance", 0.5))]
    
    # Calculate cosine similarity
    dot_product = sum(a * b for a, b in zip(features1, features2))
    magnitude1 = math.sqrt(sum(a * a for a in features1))
    magnitude2 = math.sqrt(sum(b * b for b in features2))
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    return dot_product / (magnitude1 * magnitude2)

def safe_request(func: Callable, *args, max_retries: int = 3, 
                base_delay: float = 1.0, **kwargs) -> Any:
    """
    Execute a request with exponential backoff for retries
    
    Args:
        func: Function to execute
        max_retries: Maximum number of retry attempts
        base_delay: Base delay for exponential backoff
        args, kwargs: Arguments to pass to the function
        
    Returns:
        Result from the function or None if all retries failed
    """
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries:
                logger.error(f"Request failed after {max_retries} attempts: {e}")
                return None
            
            # Calculate delay with exponential backoff and jitter
            delay = base_delay * (2 ** attempt) * (0.5 + random.random())
            logger.warning(f"Request failed, retrying in {delay:.2f}s: {e}")
            time.sleep(delay)

# -----------------------------
# KNOWLEDGE GRAPH
# -----------------------------

class KnowledgeGraph:
    """
    Enhanced knowledge graph with heterogeneous graph support and incremental updates
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.graph_config = config["knowledge_graph"]
        
        # Create cache directory if needed
        create_directory_if_not_exists(self.graph_config["cache_dir"])
        
        # Load data
        self.concepts = self._load_or_create_json(self.graph_config["concepts_file"])
        self.questions = self._load_or_create_json(self.graph_config["questions_file"])
        self.resources = self._load_or_create_json(self.graph_config["resources_file"])
        self.learners = self._load_or_create_json(self.graph_config["learners_file"])
        
        # Create lookup dictionaries for faster access
        self._concept_lookup = {}
        self._question_lookup = {}
        self._resource_lookup = {}
        self._learner_lookup = {}
        self._update_lookups()
        
        # Create heterogeneous graph structure for PyTorch Geometric
        self.hetero_data = None
        self.node_indices = {}  # Maps node types to {id: index} mappings
        
        # Track updated nodes and edges for incremental updates
        self._updates = {
            'added_nodes': defaultdict(list),  # type -> [(id, features)]
            'added_edges': defaultdict(list),  # (src_type, edge_type, dst_type) -> [(src_id, dst_id, features)]
            'updated_nodes': defaultdict(list),  # type -> [(id, features)]
            'updated_edges': defaultdict(list),  # (src_type, edge_type, dst_type) -> [(src_id, dst_id, features)]
            'deleted_nodes': defaultdict(list),  # type -> [id]
            'deleted_edges': defaultdict(list),  # (src_type, edge_type, dst_type) -> [(src_id, dst_id)]
        }
        
        # Initialize the graph
        self._build_hetero_graph()
        
        # Set up auto-save timer if enabled
        self._last_save_time = time.time()
        self._auto_save_interval = config["system"]["auto_save_interval_minutes"] * 60
        
        logger.info("Knowledge graph initialized successfully")
    
    def _update_lookups(self) -> None:
        """Update lookup dictionaries for faster access to entities"""
        # Create concept lookup
        self._concept_lookup = {
            c["id"]: c for c in self.concepts.get("concepts", [])
        }
        
        # Create question lookup
        self._question_lookup = {
            q["id"]: q for q in self.questions.get("questions", [])
        }
        
        # Create resource lookup
        self._resource_lookup = {
            r["id"]: r for r in self.resources.get("resources", [])
        }
        
        # Create learner lookup
        self._learner_lookup = {
            l["id"]: l for l in self.learners.get("learners", [])
        }
    
    def _load_or_create_json(self, file_path: str) -> Dict[str, Any]:
        """
        Load JSON data or create empty structure if file doesn't exist
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Dictionary containing the loaded or created data
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON from {file_path}: {e}")
                # Backup the corrupted file
                backup_path = f"{file_path}.bak.{int(time.time())}"
                try:
                    import shutil
                    shutil.copy2(file_path, backup_path)
                    logger.info(f"Backed up corrupted file to {backup_path}")
                except Exception as backup_err:
                    logger.error(f"Failed to backup corrupted file: {backup_err}")
            except IOError as e:
                logger.error(f"Error reading file {file_path}: {e}")
        
        # Create empty structure based on file type
        if "concepts" in file_path:
            data = {"concepts": []}
        elif "questions" in file_path:
            data = {"questions": []}
        elif "resources" in file_path:
            data = {"resources": []}
        elif "learners" in file_path:
            data = {"learners": []}
        else:
            data = {}
        
        # Save the empty structure
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except IOError as e:
            logger.error(f"Error creating file {file_path}: {e}")
        
        return data
    
    def _get_node_features(self, node_type: str, node_data: Dict[str, Any]) -> torch.Tensor:
        """
        Get features for a specific node
        
        Args:
            node_type: Type of the node (concept, question, resource, learner)
            node_data: Node data dictionary
            
        Returns:
            Tensor of node features
        """
        if node_type == 'concept':
            # [difficulty, complexity, importance, has_prerequisites]
            return torch.tensor([
                float(node_data.get("difficulty", 0.5)),
                float(node_data.get("complexity", 0.5)),
                float(node_data.get("importance", 0.5)),
                1.0 if node_data.get("prerequisites") else 0.0
            ], dtype=torch.float)
            
        elif node_type == 'question':
            # [difficulty, type_value, discriminator_power]
            return torch.tensor([
                float(node_data.get("difficulty", 0.5)),
                float(node_data.get("type_value", QUESTION_TYPES["application"])),
                float(node_data.get("discriminator_power", 0.5))
            ], dtype=torch.float)
            
        elif node_type == 'resource':
            # [quality, complexity, media_type_value]
            return torch.tensor([
                float(node_data.get("quality", 0.5)),
                float(node_data.get("complexity", 0.5)),
                float(node_data.get("media_type_value", MEDIA_TYPES["text"]))
            ], dtype=torch.float)
            
        elif node_type == 'learner':
            # [overall_progress, learning_style_value, persistence, experience]
            return torch.tensor([
                float(node_data.get("overall_progress", 0.0)),
                float(node_data.get("learning_style_value", LEARNING_STYLES["balanced"])),
                float(node_data.get("persistence", 0.5)),
                min(1.0, len(node_data.get("answered_questions", [])) / 100.0)  # Experience normalized
            ], dtype=torch.float)
        
        else:
            logger.warning(f"Unknown node type: {node_type}, using empty features")
            return torch.tensor([], dtype=torch.float)
    
    def _get_edge_features(self, src_type: str, edge_type: str, dst_type: str, 
                          src_id: str, dst_id: str) -> torch.Tensor:
        """
        Get features for a specific edge
        
        Args:
            src_type: Source node type
            edge_type: Type of the edge
            dst_type: Destination node type
            src_id: Source node ID
            dst_id: Destination node ID
            
        Returns:
            Tensor of edge features
        """
        # Handle learner to concept edges (mastery/study relationship)
        if src_type == 'learner' and edge_type == 'studies' and dst_type == 'concept':
            # Find mastery record
            learner = self.get_learner_by_id(src_id)
            if learner:
                for mastery in learner.get("concept_mastery", []):
                    if mastery["concept_id"] == dst_id:
                        # [mastery_level, last_interaction_recency]
                        days_since = (datetime.datetime.now() - 
                                     datetime.datetime.fromisoformat(mastery.get("timestamp", 
                                                                               datetime.datetime.now().isoformat())))
                        recency = max(0.0, min(1.0, 1.0 - (days_since.total_seconds() / (30*86400))))
                        return torch.tensor([float(mastery["level"]), recency], dtype=torch.float)
            
            # Default for new relationships
            return torch.tensor([0.0, 1.0], dtype=torch.float)
        
        # Handle learner to question edges (answered relationship)
        elif src_type == 'learner' and edge_type == 'answered' and dst_type == 'question':
            learner = self.get_learner_by_id(src_id)
            if learner:
                # Find the most recent answer to this question
                latest_answer = None
                for answer in learner.get("answered_questions", []):
                    if answer["question_id"] == dst_id:
                        if not latest_answer or datetime.datetime.fromisoformat(answer["timestamp"]) > \
                           datetime.datetime.fromisoformat(latest_answer["timestamp"]):
                            latest_answer = answer
                
                if latest_answer:
                    # [correctness, reasoning_quality, attempts]
                    attempts = sum(1 for a in learner.get("answered_questions", []) 
                                  if a["question_id"] == dst_id)
                    return torch.tensor([
                        1.0 if latest_answer.get("correct", False) else 0.0,
                        float(latest_answer.get("reasoning_quality", 0.5)),
                        min(1.0, attempts / 5.0)  # Normalize attempts
                    ], dtype=torch.float)
            
            # Default for new relationships
            return torch.tensor([0.0, 0.0, 0.0], dtype=torch.float)
        
        # Handle learner to resource edges (used relationship)
        elif src_type == 'learner' and edge_type == 'used' and dst_type == 'resource':
            learner = self.get_learner_by_id(src_id)
            if learner:
                # Find resource usage data
                for usage in learner.get("resource_usage", []):
                    if usage["resource_id"] == dst_id:
                        # [engagement_level, usefulness_rating, times_accessed]
                        times_accessed = min(1.0, usage.get("times_accessed", 1) / 10.0)
                        return torch.tensor([
                            float(usage.get("engagement_level", 0.5)),
                            float(usage.get("usefulness_rating", 0.5)),
                            times_accessed
                        ], dtype=torch.float)
            
            # Default for new relationships
            return torch.tensor([0.0, 0.0, 0.0], dtype=torch.float)
        
        # Handle concept to concept edges (prerequisite relationship)
        elif src_type == 'concept' and edge_type == 'requires' and dst_type == 'concept':
            # [strength, importance]
            return torch.tensor([0.8, 0.7], dtype=torch.float)
        
        # Handle question to concept edges (tests relationship)
        elif src_type == 'question' and edge_type == 'tests' and dst_type == 'concept':
            # [relevance]
            return torch.tensor([0.9], dtype=torch.float)
        
        # Handle resource to concept edges (teaches relationship)
        elif src_type == 'resource' and edge_type == 'teaches' and dst_type == 'concept':
            # [relevance, comprehensiveness]
            return torch.tensor([0.8, 0.7], dtype=torch.float)
            
        else:
            # Default edge features
            return torch.tensor([0.5], dtype=torch.float)

    def _build_hetero_graph(self) -> None:
        """
        Build the heterogeneous graph structure from loaded data with bidirectional edges
        to enable proper message passing for all node types
        """
        hetero_data = HeteroData()
        
        # Reset node indices
        self.node_indices = {node_type: {} for node_type in NODE_TYPES.keys()}
        
        # Add concept nodes
        concept_features = []
        for i, concept in enumerate(self.concepts.get("concepts", [])):
            concept_id = concept["id"]
            self.node_indices['concept'][concept_id] = i
            concept_features.append(self._get_node_features('concept', concept))
        
        if concept_features:
            hetero_data['concept'].x = torch.stack(concept_features)
        else:
            hetero_data['concept'].x = torch.zeros((0, 4), dtype=torch.float)
        
        # Add question nodes
        question_features = []
        for i, question in enumerate(self.questions.get("questions", [])):
            question_id = question["id"]
            self.node_indices['question'][question_id] = i
            question_features.append(self._get_node_features('question', question))
        
        if question_features:
            hetero_data['question'].x = torch.stack(question_features)
        else:
            hetero_data['question'].x = torch.zeros((0, 3), dtype=torch.float)
        
        # Add resource nodes
        resource_features = []
        for i, resource in enumerate(self.resources.get("resources", [])):
            resource_id = resource["id"]
            self.node_indices['resource'][resource_id] = i
            resource_features.append(self._get_node_features('resource', resource))
        
        if resource_features:
            hetero_data['resource'].x = torch.stack(resource_features)
        else:
            hetero_data['resource'].x = torch.zeros((0, 3), dtype=torch.float)
        
        # Add learner nodes
        learner_features = []
        for i, learner in enumerate(self.learners.get("learners", [])):
            learner_id = learner["id"]
            self.node_indices['learner'][learner_id] = i
            learner_features.append(self._get_node_features('learner', learner))
        
        if learner_features:
            hetero_data['learner'].x = torch.stack(learner_features)
        else:
            hetero_data['learner'].x = torch.zeros((0, 4), dtype=torch.float)
        
        # Add all forward edges first
        edge_data = {}  # Store edge data to use for creating reverse edges later
        
        # 1. Question -> tests -> Concept edges
        src_indices = []
        dst_indices = []
        edge_features = []
        
        for question in self.questions.get("questions", []):
            q_idx = self.node_indices['question'].get(question["id"])
            if q_idx is None:
                continue
                
            for concept_id in question.get("related_concepts", []):
                c_idx = self.node_indices['concept'].get(concept_id)
                if c_idx is not None:
                    src_indices.append(q_idx)
                    dst_indices.append(c_idx)
                    edge_features.append(
                        self._get_edge_features('question', 'tests', 'concept', 
                                            question["id"], concept_id)
                    )
        
        if src_indices:
            hetero_data['question', 'tests', 'concept'].edge_index = torch.tensor([src_indices, dst_indices], 
                                                                            dtype=torch.long)
            hetero_data['question', 'tests', 'concept'].edge_attr = torch.stack(edge_features)
            # Store for reverse edge creation
            edge_data[('question', 'tests', 'concept')] = {
                'edge_index': [src_indices, dst_indices], 
                'edge_attr': edge_features
            }
        
        # 2. Resource -> teaches -> Concept edges
        src_indices = []
        dst_indices = []
        edge_features = []
        
        for resource in self.resources.get("resources", []):
            r_idx = self.node_indices['resource'].get(resource["id"])
            if r_idx is None:
                continue
                
            for concept_id in resource.get("related_concepts", []):
                c_idx = self.node_indices['concept'].get(concept_id)
                if c_idx is not None:
                    src_indices.append(r_idx)
                    dst_indices.append(c_idx)
                    edge_features.append(
                        self._get_edge_features('resource', 'teaches', 'concept', 
                                            resource["id"], concept_id)
                    )
        
        if src_indices:
            hetero_data['resource', 'teaches', 'concept'].edge_index = torch.tensor([src_indices, dst_indices], 
                                                                            dtype=torch.long)
            hetero_data['resource', 'teaches', 'concept'].edge_attr = torch.stack(edge_features)
            # Store for reverse edge creation
            edge_data[('resource', 'teaches', 'concept')] = {
                'edge_index': [src_indices, dst_indices], 
                'edge_attr': edge_features
            }
        
        # 3. Concept -> requires -> Concept edges (prerequisites)
        src_indices = []
        dst_indices = []
        edge_features = []
        
        for concept in self.concepts.get("concepts", []):
            c_idx = self.node_indices['concept'].get(concept["id"])
            if c_idx is None:
                continue
                
            for prereq_id in concept.get("prerequisites", []):
                p_idx = self.node_indices['concept'].get(prereq_id)
                if p_idx is not None:
                    src_indices.append(c_idx)
                    dst_indices.append(p_idx)
                    edge_features.append(
                        self._get_edge_features('concept', 'requires', 'concept', 
                                            concept["id"], prereq_id)
                    )
        
        if src_indices:
            hetero_data['concept', 'requires', 'concept'].edge_index = torch.tensor([src_indices, dst_indices], 
                                                                            dtype=torch.long)
            hetero_data['concept', 'requires', 'concept'].edge_attr = torch.stack(edge_features)
            # Store for reverse edge creation
            edge_data[('concept', 'requires', 'concept')] = {
                'edge_index': [src_indices, dst_indices], 
                'edge_attr': edge_features
            }
        
        # 4. Learner -> studies -> Concept edges (mastery)
        src_indices = []
        dst_indices = []
        edge_features = []
        
        for learner in self.learners.get("learners", []):
            l_idx = self.node_indices['learner'].get(learner["id"])
            if l_idx is None:
                continue
                
            for mastery in learner.get("concept_mastery", []):
                c_idx = self.node_indices['concept'].get(mastery["concept_id"])
                if c_idx is not None:
                    src_indices.append(l_idx)
                    dst_indices.append(c_idx)
                    edge_features.append(
                        self._get_edge_features('learner', 'studies', 'concept', 
                                            learner["id"], mastery["concept_id"])
                    )
        
        if src_indices:
            hetero_data['learner', 'studies', 'concept'].edge_index = torch.tensor([src_indices, dst_indices], 
                                                                            dtype=torch.long)
            hetero_data['learner', 'studies', 'concept'].edge_attr = torch.stack(edge_features)
            # Store for reverse edge creation
            edge_data[('learner', 'studies', 'concept')] = {
                'edge_index': [src_indices, dst_indices], 
                'edge_attr': edge_features
            }
        
        # 5. Learner -> answered -> Question edges
        src_indices = []
        dst_indices = []
        edge_features = []
        
        for learner in self.learners.get("learners", []):
            l_idx = self.node_indices['learner'].get(learner["id"])
            if l_idx is None:
                continue
            
            # Track question IDs that have been processed to avoid duplicates
            processed_questions = set()
            
            for answer in learner.get("answered_questions", []):
                question_id = answer["question_id"]
                
                # Skip if we've already processed this learner-question pair
                learner_question_pair = (learner["id"], question_id)
                if learner_question_pair in processed_questions:
                    continue
                    
                processed_questions.add(learner_question_pair)
                
                q_idx = self.node_indices['question'].get(question_id)
                if q_idx is not None:
                    src_indices.append(l_idx)
                    dst_indices.append(q_idx)
                    edge_features.append(
                        self._get_edge_features('learner', 'answered', 'question', 
                                            learner["id"], question_id)
                    )
        
        if src_indices:
            hetero_data['learner', 'answered', 'question'].edge_index = torch.tensor([src_indices, dst_indices], 
                                                                            dtype=torch.long)
            hetero_data['learner', 'answered', 'question'].edge_attr = torch.stack(edge_features)
            # Store for reverse edge creation
            edge_data[('learner', 'answered', 'question')] = {
                'edge_index': [src_indices, dst_indices], 
                'edge_attr': edge_features
            }
        
        # 6. Learner -> used -> Resource edges
        src_indices = []
        dst_indices = []
        edge_features = []
        
        for learner in self.learners.get("learners", []):
            l_idx = self.node_indices['learner'].get(learner["id"])
            if l_idx is None:
                continue
                
            for usage in learner.get("resource_usage", []):
                r_idx = self.node_indices['resource'].get(usage["resource_id"])
                if r_idx is not None:
                    src_indices.append(l_idx)
                    dst_indices.append(r_idx)
                    edge_features.append(
                        self._get_edge_features('learner', 'used', 'resource', 
                                            learner["id"], usage["resource_id"])
                    )
        
        if src_indices:
            hetero_data['learner', 'used', 'resource'].edge_index = torch.tensor([src_indices, dst_indices], 
                                                                        dtype=torch.long)
            hetero_data['learner', 'used', 'resource'].edge_attr = torch.stack(edge_features)
            # Store for reverse edge creation
            edge_data[('learner', 'used', 'resource')] = {
                'edge_index': [src_indices, dst_indices], 
                'edge_attr': edge_features
            }
        
        # CREATE REVERSE EDGES FOR PROPER MESSAGE PASSING
        # Concept -> studied_by -> Learner (reverse of Learner -> studies -> Concept)
        if ('learner', 'studies', 'concept') in edge_data:
            data = edge_data[('learner', 'studies', 'concept')]
            src_indices, dst_indices = data['edge_index']
            
            # For reverse edge, swap source and destination
            hetero_data['concept', 'studied_by', 'learner'].edge_index = torch.tensor(
                [dst_indices, src_indices], dtype=torch.long
            )
            
            # Use same edge features but potentially transform them
            # (in a real implementation, you might want to customize these)
            if data['edge_attr']:
                hetero_data['concept', 'studied_by', 'learner'].edge_attr = torch.stack(data['edge_attr'])
        
        # Concept -> taught_by -> Resource (reverse of Resource -> teaches -> Concept)
        if ('resource', 'teaches', 'concept') in edge_data:
            data = edge_data[('resource', 'teaches', 'concept')]
            src_indices, dst_indices = data['edge_index']
            
            hetero_data['concept', 'taught_by', 'resource'].edge_index = torch.tensor(
                [dst_indices, src_indices], dtype=torch.long
            )
            
            if data['edge_attr']:
                hetero_data['concept', 'taught_by', 'resource'].edge_attr = torch.stack(data['edge_attr'])
        
        # Question -> answered_by -> Learner (reverse of Learner -> answered -> Question)
        if ('learner', 'answered', 'question') in edge_data:
            data = edge_data[('learner', 'answered', 'question')]
            src_indices, dst_indices = data['edge_index']
            
            hetero_data['question', 'answered_by', 'learner'].edge_index = torch.tensor(
                [dst_indices, src_indices], dtype=torch.long
            )
            
            if data['edge_attr']:
                hetero_data['question', 'answered_by', 'learner'].edge_attr = torch.stack(data['edge_attr'])
        
        # Resource -> used_by -> Learner (reverse of Learner -> used -> Resource)
        if ('learner', 'used', 'resource') in edge_data:
            data = edge_data[('learner', 'used', 'resource')]
            src_indices, dst_indices = data['edge_index']
            
            hetero_data['resource', 'used_by', 'learner'].edge_index = torch.tensor(
                [dst_indices, src_indices], dtype=torch.long
            )
            
            if data['edge_attr']:
                hetero_data['resource', 'used_by', 'learner'].edge_attr = torch.stack(data['edge_attr'])
        
        # Concept -> required_for -> Concept (reverse of Concept -> requires -> Concept)
        # This one is special because it's between the same node types
        if ('concept', 'requires', 'concept') in edge_data:
            data = edge_data[('concept', 'requires', 'concept')]
            src_indices, dst_indices = data['edge_index']
            
            hetero_data['concept', 'required_for', 'concept'].edge_index = torch.tensor(
                [dst_indices, src_indices], dtype=torch.long
            )
            
            if data['edge_attr']:
                hetero_data['concept', 'required_for', 'concept'].edge_attr = torch.stack(data['edge_attr'])
        
        # Concept -> tested_by -> Question (reverse of Question -> tests -> Concept)
        if ('question', 'tests', 'concept') in edge_data:
            data = edge_data[('question', 'tests', 'concept')]
            src_indices, dst_indices = data['edge_index']
            
            hetero_data['concept', 'tested_by', 'question'].edge_index = torch.tensor(
                [dst_indices, src_indices], dtype=torch.long
            )
            
            if data['edge_attr']:
                hetero_data['concept', 'tested_by', 'question'].edge_attr = torch.stack(data['edge_attr'])
        
        # Store the heterogeneous graph data
        self.hetero_data = hetero_data
        
        # Clear the updates since we've rebuilt the graph
        self._clear_updates()
        
        # Log graph statistics
        node_counts = {node_type: data.num_nodes for node_type, data in hetero_data.node_items()}
        edge_counts = {edge_type: data.num_edges for edge_type, data in hetero_data.edge_items()}
        
        logger.info(f"Built heterogeneous graph with node counts: {node_counts}")
        logger.info(f"Edge counts: {edge_counts}")
        
        # Verify no warnings about unconnected node types
        node_types_as_dst = set()
        for edge_type in hetero_data.edge_types:
            node_types_as_dst.add(edge_type[2])  # Destination node type
        
        missing_dst_types = set(NODE_TYPES.keys()) - node_types_as_dst
        if missing_dst_types:
            logger.warning(f"Node types not occurring as destinations (this may cause embedding issues): {missing_dst_types}")

    def _update_hetero_graph(self) -> None:
        """
        Update the heterogeneous graph incrementally based on tracked changes
        
        This is more efficient than rebuilding the entire graph when only a few nodes
        or edges have changed.
        """
        if self.hetero_data is None:
            # No existing graph, build from scratch
            self._build_hetero_graph()
            return
        
        # Check if there are any updates at all
        has_updates = False
        for update_type, updates in self._updates.items():
            if any(updates.values()):
                has_updates = True
                break
        
        if not has_updates:
            # No updates to apply
            return
        
        # Handle added/updated nodes
        for node_type in NODE_TYPES.keys():
            # First handle deleted nodes by rebuilding that node type
            if self._updates['deleted_nodes'][node_type]:
                logger.info(f"Rebuilding {node_type} nodes due to deletions")
                self._rebuild_node_type(node_type)
                continue
            
            # Handle added nodes
            if self._updates['added_nodes'][node_type]:
                self._add_nodes_to_graph(node_type, self._updates['added_nodes'][node_type])
            
            # Handle updated nodes
            if self._updates['updated_nodes'][node_type]:
                self._update_nodes_in_graph(node_type, self._updates['updated_nodes'][node_type])
        
        # Handle edge updates
        for edge_type in EDGE_TYPES:
            edge_key = (edge_type[0], edge_type[1], edge_type[2])
            
            # If there are deleted edges, we need to rebuild the entire edge type
            if self._updates['deleted_edges'][edge_key]:
                logger.info(f"Rebuilding {edge_key} edges due to deletions")
                self._rebuild_edge_type(edge_key)
                continue
            
            # Handle added edges
            if self._updates['added_edges'][edge_key]:
                self._add_edges_to_graph(edge_key, self._updates['added_edges'][edge_key])
            
            # Handle updated edges
            if self._updates['updated_edges'][edge_key]:
                self._update_edges_in_graph(edge_key, self._updates['updated_edges'][edge_key])
        
        # Clear the updates
        self._clear_updates()
    
    def _rebuild_node_type(self, node_type: str) -> None:
        """Rebuild all nodes of a specific type"""
        data_sources = {
            'concept': self.concepts.get("concepts", []),
            'question': self.questions.get("questions", []),
            'resource': self.resources.get("resources", []),
            'learner': self.learners.get("learners", [])
        }
        
        # Get the data for this node type
        nodes = data_sources.get(node_type, [])
        
        # Rebuild node indices for this type
        self.node_indices[node_type] = {}
        features = []
        
        for i, node in enumerate(nodes):
            node_id = node["id"]
            self.node_indices[node_type][node_id] = i
            features.append(self._get_node_features(node_type, node))
        
        # Update the features in the heterogeneous graph
        if features:
            self.hetero_data[node_type].x = torch.stack(features)
        else:
            # Create empty feature tensor with the right number of columns
            feature_dims = {
                'concept': 4,
                'question': 3,
                'resource': 3,
                'learner': 4
            }
            self.hetero_data[node_type].x = torch.zeros((0, feature_dims[node_type]), dtype=torch.float)
    
    def _rebuild_edge_type(self, edge_key: Tuple[str, str, str]) -> None:
        """Rebuild all edges of a specific type"""
        src_type, edge_type, dst_type = edge_key
        
        # Collect edge data based on edge type
        src_indices = []
        dst_indices = []
        edge_features = []
        
        # Question -> tests -> Concept
        if edge_key == ('question', 'tests', 'concept'):
            for question in self.questions.get("questions", []):
                q_idx = self.node_indices['question'].get(question["id"])
                if q_idx is None:
                    continue
                    
                for concept_id in question.get("related_concepts", []):
                    c_idx = self.node_indices['concept'].get(concept_id)
                    if c_idx is not None:
                        src_indices.append(q_idx)
                        dst_indices.append(c_idx)
                        edge_features.append(
                            self._get_edge_features('question', 'tests', 'concept', 
                                                 question["id"], concept_id)
                        )
        
        # Resource -> teaches -> Concept
        elif edge_key == ('resource', 'teaches', 'concept'):
            for resource in self.resources.get("resources", []):
                r_idx = self.node_indices['resource'].get(resource["id"])
                if r_idx is None:
                    continue
                    
                for concept_id in resource.get("related_concepts", []):
                    c_idx = self.node_indices['concept'].get(concept_id)
                    if c_idx is not None:
                        src_indices.append(r_idx)
                        dst_indices.append(c_idx)
                        edge_features.append(
                            self._get_edge_features('resource', 'teaches', 'concept', 
                                                 resource["id"], concept_id)
                        )
        
        # Concept -> requires -> Concept (prerequisites)
        elif edge_key == ('concept', 'requires', 'concept'):
            for concept in self.concepts.get("concepts", []):
                c_idx = self.node_indices['concept'].get(concept["id"])
                if c_idx is None:
                    continue
                    
                for prereq_id in concept.get("prerequisites", []):
                    p_idx = self.node_indices['concept'].get(prereq_id)
                    if p_idx is not None:
                        src_indices.append(c_idx)
                        dst_indices.append(p_idx)
                        edge_features.append(
                            self._get_edge_features('concept', 'requires', 'concept', 
                                                 concept["id"], prereq_id)
                        )
        
        # Learner -> studies -> Concept (mastery)
        elif edge_key == ('learner', 'studies', 'concept'):
            for learner in self.learners.get("learners", []):
                l_idx = self.node_indices['learner'].get(learner["id"])
                if l_idx is None:
                    continue
                    
                for mastery in learner.get("concept_mastery", []):
                    c_idx = self.node_indices['concept'].get(mastery["concept_id"])
                    if c_idx is not None:
                        src_indices.append(l_idx)
                        dst_indices.append(c_idx)
                        edge_features.append(
                            self._get_edge_features('learner', 'studies', 'concept', 
                                                 learner["id"], mastery["concept_id"])
                        )
        
        # Learner -> answered -> Question
        elif edge_key == ('learner', 'answered', 'question'):
            # Track processed pairs to avoid duplicates
            processed_pairs = set()
            
            for learner in self.learners.get("learners", []):
                l_idx = self.node_indices['learner'].get(learner["id"])
                if l_idx is None:
                    continue
                
                for answer in learner.get("answered_questions", []):
                    q_id = answer["question_id"]
                    
                    # Skip if already processed this pair
                    pair = (learner["id"], q_id)
                    if pair in processed_pairs:
                        continue
                    
                    processed_pairs.add(pair)
                    q_idx = self.node_indices['question'].get(q_id)
                    
                    if q_idx is not None:
                        src_indices.append(l_idx)
                        dst_indices.append(q_idx)
                        edge_features.append(
                            self._get_edge_features('learner', 'answered', 'question', 
                                                 learner["id"], q_id)
                        )
        
        # Learner -> used -> Resource
        elif edge_key == ('learner', 'used', 'resource'):
            for learner in self.learners.get("learners", []):
                l_idx = self.node_indices['learner'].get(learner["id"])
                if l_idx is None:
                    continue
                    
                for usage in learner.get("resource_usage", []):
                    r_idx = self.node_indices['resource'].get(usage["resource_id"])
                    if r_idx is not None:
                        src_indices.append(l_idx)
                        dst_indices.append(r_idx)
                        edge_features.append(
                            self._get_edge_features('learner', 'used', 'resource', 
                                                 learner["id"], usage["resource_id"])
                        )
        
        # Update the edge data in the heterogeneous graph
        if src_indices:
            self.hetero_data[edge_key].edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
            self.hetero_data[edge_key].edge_attr = torch.stack(edge_features)
        else:
            # Create empty tensors
            self.hetero_data[edge_key].edge_index = torch.zeros((2, 0), dtype=torch.long)
            
            # Determine feature dimension based on edge type
            if edge_key == ('learner', 'studies', 'concept'):
                feature_dim = 2
            elif edge_key in [('learner', 'answered', 'question'), ('learner', 'used', 'resource')]:
                feature_dim = 3
            elif edge_key == ('concept', 'requires', 'concept'):
                feature_dim = 2
            elif edge_key in [('question', 'tests', 'concept'), ('resource', 'teaches', 'concept')]:
                feature_dim = 1 if edge_key == ('question', 'tests', 'concept') else 2
            else:
                feature_dim = 1
                
            self.hetero_data[edge_key].edge_attr = torch.zeros((0, feature_dim), dtype=torch.float)
    
    def _add_nodes_to_graph(self, node_type: str, nodes: List[Tuple[str, torch.Tensor]]) -> None:
        """Add new nodes to the graph"""
        if not nodes:
            return
            
        # Get existing feature tensor
        existing_features = self.hetero_data[node_type].x
        
        # Get new features
        new_features = [features for _, features in nodes]
        
        # Add new node indices
        for node_id, _ in nodes:
            self.node_indices[node_type][node_id] = len(self.node_indices[node_type])
        
        # Update the feature tensor
        self.hetero_data[node_type].x = torch.cat([existing_features, torch.stack(new_features)])
    
    def _update_nodes_in_graph(self, node_type: str, nodes: List[Tuple[str, torch.Tensor]]) -> None:
        """Update existing nodes in the graph"""
        if not nodes:
            return
            
        for node_id, features in nodes:
            idx = self.node_indices[node_type].get(node_id)
            if idx is not None:
                self.hetero_data[node_type].x[idx] = features
    
    def _add_edges_to_graph(self, edge_key: Tuple[str, str, str], 
                           edges: List[Tuple[str, str, torch.Tensor]]) -> None:
        """Add new edges to the graph"""
        if not edges:
            return
            
        src_type, _, dst_type = edge_key
        
        # Get existing edge data
        if edge_key in self.hetero_data:
            existing_edge_index = self.hetero_data[edge_key].edge_index
            existing_edge_attr = self.hetero_data[edge_key].edge_attr
        else:
            # Create empty tensors if this edge type doesn't exist yet
            existing_edge_index = torch.zeros((2, 0), dtype=torch.long)
            
            # Determine feature dimension based on edge type
            if edge_key == ('learner', 'studies', 'concept'):
                feature_dim = 2
            elif edge_key in [('learner', 'answered', 'question'), ('learner', 'used', 'resource')]:
                feature_dim = 3
            elif edge_key == ('concept', 'requires', 'concept'):
                feature_dim = 2
            else:
                feature_dim = 1
                
            existing_edge_attr = torch.zeros((0, feature_dim), dtype=torch.float)
        
        # Collect new edge data
        new_src_indices = []
        new_dst_indices = []
        new_edge_attr = []
        
        for src_id, dst_id, features in edges:
            src_idx = self.node_indices[src_type].get(src_id)
            dst_idx = self.node_indices[dst_type].get(dst_id)
            
            if src_idx is not None and dst_idx is not None:
                new_src_indices.append(src_idx)
                new_dst_indices.append(dst_idx)
                new_edge_attr.append(features)
        
        if not new_src_indices:
            return
            
        # Create new edge tensors
        new_edge_index = torch.tensor([new_src_indices, new_dst_indices], dtype=torch.long)
        new_edge_attr_tensor = torch.stack(new_edge_attr)
        
        # Concatenate with existing data
        self.hetero_data[edge_key].edge_index = torch.cat([existing_edge_index, new_edge_index], dim=1)
        self.hetero_data[edge_key].edge_attr = torch.cat([existing_edge_attr, new_edge_attr_tensor])
    
    def _update_edges_in_graph(self, edge_key: Tuple[str, str, str], 
                              edges: List[Tuple[str, str, torch.Tensor]]) -> None:
        """Update existing edges in the graph"""
        if not edges or edge_key not in self.hetero_data:
            return
            
        src_type, _, dst_type = edge_key
        
        # Get existing edge data
        edge_index = self.hetero_data[edge_key].edge_index
        edge_attr = self.hetero_data[edge_key].edge_attr
        
        for src_id, dst_id, features in edges:
            src_idx = self.node_indices[src_type].get(src_id)
            dst_idx = self.node_indices[dst_type].get(dst_id)
            
            if src_idx is None or dst_idx is None:
                continue
                
            # Find the edge in the edge index
            for i in range(edge_index.size(1)):
                if edge_index[0, i] == src_idx and edge_index[1, i] == dst_idx:
                    # Update the edge attributes
                    edge_attr[i] = features
                    break
    
    def _clear_updates(self) -> None:
        """Clear tracked updates after applying them to the graph"""
        for update_type in self._updates:
            self._updates[update_type] = defaultdict(list)
    
    def save(self) -> bool:
        """
        Save all knowledge graph data to files
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create backup of existing files
            backup_count = self.config["system"]["backup_count"]
            file_paths = [
                self.graph_config["concepts_file"],
                self.graph_config["questions_file"],
                self.graph_config["resources_file"],
                self.graph_config["learners_file"]
            ]
            
            # Create backups if requested
            if backup_count > 0:
                for path in file_paths:
                    if os.path.exists(path):
                        # Create a timestamp-based backup
                        backup_path = f"{path}.bak.{int(time.time())}"
                        try:
                            import shutil
                            shutil.copy2(path, backup_path)
                            
                            # Remove old backups if we have too many
                            backup_files = sorted([
                                f for f in os.listdir(os.path.dirname(path)) 
                                if f.startswith(os.path.basename(path) + ".bak")
                            ])
                            
                            if len(backup_files) > backup_count:
                                for old_backup in backup_files[:-backup_count]:
                                    old_path = os.path.join(os.path.dirname(path), old_backup)
                                    os.remove(old_path)
                        except Exception as e:
                            logger.warning(f"Could not create backup of {path}: {e}")
            
            # Save the data
            with open(self.graph_config["concepts_file"], 'w', encoding='utf-8') as f:
                json.dump(self.concepts, f, indent=2)
                
            with open(self.graph_config["questions_file"], 'w', encoding='utf-8') as f:
                json.dump(self.questions, f, indent=2)
                
            with open(self.graph_config["resources_file"], 'w', encoding='utf-8') as f:
                json.dump(self.resources, f, indent=2)
                
            with open(self.graph_config["learners_file"], 'w', encoding='utf-8') as f:
                json.dump(self.learners, f, indent=2)
            
            self._last_save_time = time.time()
            logger.info("Knowledge graph data saved successfully")
            return True
        except Exception as e:
            logger.error(f"Error saving knowledge graph data: {e}")
            return False
    
    def check_auto_save(self) -> None:
        """Check if it's time to auto-save and save if needed"""
        if time.time() - self._last_save_time > self._auto_save_interval:
            logger.info("Auto-saving knowledge graph data...")
            self.save()
    
    def add_concept(self, name: str, description: str = "", difficulty: float = 0.5, 
                  complexity: float = 0.5, importance: float = 0.5, 
                  prerequisites: List[str] = None) -> str:
        """
        Add a new concept to the knowledge graph
        
        Args:
            name: Name of the concept
            description: Description of the concept
            difficulty: Difficulty level (0.0 to 1.0)
            complexity: Complexity level (0.0 to 1.0)
            importance: Importance level (0.0 to 1.0)
            prerequisites: List of prerequisite concept IDs
            
        Returns:
            ID of the newly created concept
        """
        # Validate inputs
        difficulty = max(0.0, min(1.0, float(difficulty)))
        complexity = max(0.0, min(1.0, float(complexity)))
        importance = max(0.0, min(1.0, float(importance)))
        
        # Generate new ID
        concept_id = str(uuid.uuid4())[:8]
        
        # Create concept
        concept = {
            "id": concept_id,
            "name": name,
            "description": description,
            "difficulty": difficulty,
            "complexity": complexity,
            "importance": importance,
            "prerequisites": prerequisites or []
        }
        
        # Add to concepts list
        if "concepts" not in self.concepts:
            self.concepts["concepts"] = []
        
        self.concepts["concepts"].append(concept)
        
        # Update lookup and track the addition
        self._concept_lookup[concept_id] = concept
        self._updates["added_nodes"]["concept"].append(
            (concept_id, self._get_node_features("concept", concept))
        )
        
        # If the concept has prerequisites, add those edges
        for prereq_id in concept.get("prerequisites", []):
            if prereq_id in self._concept_lookup:
                self._updates["added_edges"][("concept", "requires", "concept")].append(
                    (concept_id, prereq_id, self._get_edge_features(
                        "concept", "requires", "concept", concept_id, prereq_id)
                    )
                )
        
        # Check for auto-save
        self.check_auto_save()
        
        return concept_id
    
    def update_concept(self, concept_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update an existing concept
        
        Args:
            concept_id: ID of the concept to update
            updates: Dictionary of fields to update
            
        Returns:
            True if successful, False if concept not found
        """
        concept = self.get_concept_by_id(concept_id)
        if not concept:
            logger.warning(f"Concept with ID {concept_id} not found")
            return False
        
        # Track if prerequisites changed for edge updates
        old_prereqs = set(concept.get("prerequisites", []))
        new_prereqs = set(updates.get("prerequisites", old_prereqs))
        prereqs_changed = old_prereqs != new_prereqs
        
        # Update the concept
        concept.update(updates)
        
        # Ensure numeric fields are in valid range
        for field in ["difficulty", "complexity", "importance"]:
            if field in concept:
                concept[field] = max(0.0, min(1.0, float(concept[field])))
        
        # Track updated node
        self._updates["updated_nodes"]["concept"].append(
            (concept_id, self._get_node_features("concept", concept))
        )
        
        # Handle edge updates if prerequisites changed
        if prereqs_changed:
            # Remove deleted prerequisites
            for removed_prereq in old_prereqs - new_prereqs:
                self._updates["deleted_edges"][("concept", "requires", "concept")].append(
                    (concept_id, removed_prereq)
                )
            
            # Add new prerequisites
            for added_prereq in new_prereqs - old_prereqs:
                self._updates["added_edges"][("concept", "requires", "concept")].append(
                    (concept_id, added_prereq, self._get_edge_features(
                        "concept", "requires", "concept", concept_id, added_prereq)
                    )
                )
        
        # Check for auto-save
        self.check_auto_save()
        
        return True
    
    def delete_concept(self, concept_id: str) -> bool:
        """
        Delete a concept from the knowledge graph
        
        Args:
            concept_id: ID of the concept to delete
            
        Returns:
            True if successful, False if concept not found
        """
        concept = self.get_concept_by_id(concept_id)
        if not concept:
            logger.warning(f"Concept with ID {concept_id} not found")
            return False
        
        # Remove the concept from the list
        self.concepts["concepts"] = [c for c in self.concepts.get("concepts", []) 
                                    if c["id"] != concept_id]
        
        # Remove from lookup
        if concept_id in self._concept_lookup:
            del self._concept_lookup[concept_id]
        
        # Track the deletion
        self._updates["deleted_nodes"]["concept"].append(concept_id)
        
        # Update prerequisites in other concepts
        for c in self.concepts.get("concepts", []):
            if concept_id in c.get("prerequisites", []):
                c["prerequisites"] = [p for p in c["prerequisites"] if p != concept_id]
                # Track the edge deletion
                self._updates["deleted_edges"][("concept", "requires", "concept")].append(
                    (c["id"], concept_id)
                )
        
        # Check for auto-save
        self.check_auto_save()
        
        return True
    
    def add_question(self, text: str, answer: str, difficulty: float = 0.5,
                    question_type: str = "application", discriminator_power: float = 0.5,
                    related_concepts: List[str] = None) -> str:
        """
        Add a new question to the knowledge graph
        
        Args:
            text: Question text
            answer: Question answer
            difficulty: Difficulty level (0.0 to 1.0)
            question_type: Type of question (recall, application, analysis)
            discriminator_power: How well the question distinguishes mastery (0.0 to 1.0)
            related_concepts: List of related concept IDs
            
        Returns:
            ID of the newly created question
        """
        # Map question type to value
        type_value = QUESTION_TYPES.get(question_type.lower(), QUESTION_TYPES["application"])
        
        # Validate inputs
        difficulty = max(0.0, min(1.0, float(difficulty)))
        discriminator_power = max(0.0, min(1.0, float(discriminator_power)))
        
        # Generate new ID
        question_id = str(uuid.uuid4())[:8]
        
        # Create question
        question = {
            "id": question_id,
            "text": text,
            "answer": answer,
            "difficulty": difficulty,
            "type": question_type,
            "type_value": type_value,
            "discriminator_power": discriminator_power,
            "related_concepts": related_concepts or []
        }
        
        # Add to questions list
        if "questions" not in self.questions:
            self.questions["questions"] = []
        
        self.questions["questions"].append(question)
        
        # Update lookup and track the addition
        self._question_lookup[question_id] = question
        self._updates["added_nodes"]["question"].append(
            (question_id, self._get_node_features("question", question))
        )
        
        # Add edges to related concepts
        for concept_id in question.get("related_concepts", []):
            if concept_id in self._concept_lookup:
                self._updates["added_edges"][("question", "tests", "concept")].append(
                    (question_id, concept_id, self._get_edge_features(
                        "question", "tests", "concept", question_id, concept_id)
                    )
                )
        
        # Check for auto-save
        self.check_auto_save()
        
        return question_id
    
    def update_question(self, question_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update an existing question
        
        Args:
            question_id: ID of the question to update
            updates: Dictionary of fields to update
            
        Returns:
            True if successful, False if question not found
        """
        question = self.get_question_by_id(question_id)
        if not question:
            logger.warning(f"Question with ID {question_id} not found")
            return False
        
        # Track if related concepts changed for edge updates
        old_concepts = set(question.get("related_concepts", []))
        new_concepts = set(updates.get("related_concepts", old_concepts))
        concepts_changed = old_concepts != new_concepts
        
        # Update the question
        question.update(updates)
        
        # Update type_value if type was updated
        if "type" in updates:
            question["type_value"] = QUESTION_TYPES.get(updates["type"].lower(), 
                                                     QUESTION_TYPES["application"])
        
        # Ensure numeric fields are in valid range
        for field in ["difficulty", "discriminator_power"]:
            if field in question:
                question[field] = max(0.0, min(1.0, float(question[field])))
        
        # Track updated node
        self._updates["updated_nodes"]["question"].append(
            (question_id, self._get_node_features("question", question))
        )
        
        # Handle edge updates if related concepts changed
        if concepts_changed:
            # Remove deleted concept relationships
            for removed_concept in old_concepts - new_concepts:
                self._updates["deleted_edges"][("question", "tests", "concept")].append(
                    (question_id, removed_concept)
                )
            
            # Add new concept relationships
            for added_concept in new_concepts - old_concepts:
                self._updates["added_edges"][("question", "tests", "concept")].append(
                    (question_id, added_concept, self._get_edge_features(
                        "question", "tests", "concept", question_id, added_concept)
                    )
                )
        
        # Check for auto-save
        self.check_auto_save()
        
        return True
    
    def delete_question(self, question_id: str) -> bool:
        """
        Delete a question from the knowledge graph
        
        Args:
            question_id: ID of the question to delete
            
        Returns:
            True if successful, False if question not found
        """
        question = self.get_question_by_id(question_id)
        if not question:
            logger.warning(f"Question with ID {question_id} not found")
            return False
        
        # Remove the question from the list
        self.questions["questions"] = [q for q in self.questions.get("questions", []) 
                                      if q["id"] != question_id]
        
        # Remove from lookup
        if question_id in self._question_lookup:
            del self._question_lookup[question_id]
        
        # Track the deletion
        self._updates["deleted_nodes"]["question"].append(question_id)
        
        # Track edge deletions for related concepts
        for concept_id in question.get("related_concepts", []):
            self._updates["deleted_edges"][("question", "tests", "concept")].append(
                (question_id, concept_id)
            )
        
        # Also need to handle learner answered questions relationships
        for learner in self.learners.get("learners", []):
            answers_removed = False
            
            # Filter out answers for this question
            if "answered_questions" in learner:
                original_len = len(learner["answered_questions"])
                learner["answered_questions"] = [
                    a for a in learner["answered_questions"] 
                    if a["question_id"] != question_id
                ]
                answers_removed = len(learner["answered_questions"]) < original_len
            
            # Track edge deletions if needed
            if answers_removed:
                self._updates["deleted_edges"][("learner", "answered", "question")].append(
                    (learner["id"], question_id)
                )
        
        # Check for auto-save
        self.check_auto_save()
        
        return True
    
    def add_resource(self, title: str, url: str, description: str = "", 
                    quality: float = 0.5, complexity: float = 0.5, 
                    media_type: str = "text", related_concepts: List[str] = None) -> str:
        """
        Add a new learning resource to the knowledge graph
        
        Args:
            title: Resource title
            url: Resource URL
            description: Resource description
            quality: Quality rating (0.0 to 1.0)
            complexity: Complexity level (0.0 to 1.0)
            media_type: Type of media (text, video, interactive)
            related_concepts: List of related concept IDs
            
        Returns:
            ID of the newly created resource
        """
        # Map media type to value
        media_value = MEDIA_TYPES.get(media_type.lower(), MEDIA_TYPES["text"])
        
        # Validate inputs
        quality = max(0.0, min(1.0, float(quality)))
        complexity = max(0.0, min(1.0, float(complexity)))
        
        # Generate new ID
        resource_id = str(uuid.uuid4())[:8]
        
        # Create resource
        resource = {
            "id": resource_id,
            "title": title,
            "url": url,
            "description": description,
            "quality": quality,
            "complexity": complexity,
            "media_type": media_type,
            "media_type_value": media_value,
            "related_concepts": related_concepts or []
        }
        
        # Add to resources list
        if "resources" not in self.resources:
            self.resources["resources"] = []
        
        self.resources["resources"].append(resource)
        
        # Update lookup and track the addition
        self._resource_lookup[resource_id] = resource
        self._updates["added_nodes"]["resource"].append(
            (resource_id, self._get_node_features("resource", resource))
        )
        
        # Add edges to related concepts
        for concept_id in resource.get("related_concepts", []):
            if concept_id in self._concept_lookup:
                self._updates["added_edges"][("resource", "teaches", "concept")].append(
                    (resource_id, concept_id, self._get_edge_features(
                        "resource", "teaches", "concept", resource_id, concept_id)
                    )
                )
        
        # Check for auto-save
        self.check_auto_save()
        
        return resource_id
    
    def update_resource(self, resource_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update an existing resource
        
        Args:
            resource_id: ID of the resource to update
            updates: Dictionary of fields to update
            
        Returns:
            True if successful, False if resource not found
        """
        resource = self.get_resource_by_id(resource_id)
        if not resource:
            logger.warning(f"Resource with ID {resource_id} not found")
            return False
        
        # Track if related concepts changed for edge updates
        old_concepts = set(resource.get("related_concepts", []))
        new_concepts = set(updates.get("related_concepts", old_concepts))
        concepts_changed = old_concepts != new_concepts
        
        # Update the resource
        resource.update(updates)
        
        # Update media_type_value if media_type was updated
        if "media_type" in updates:
            resource["media_type_value"] = MEDIA_TYPES.get(updates["media_type"].lower(), 
                                                        MEDIA_TYPES["text"])
        
        # Ensure numeric fields are in valid range
        for field in ["quality", "complexity"]:
            if field in resource:
                resource[field] = max(0.0, min(1.0, float(resource[field])))
        
        # Track updated node
        self._updates["updated_nodes"]["resource"].append(
            (resource_id, self._get_node_features("resource", resource))
        )
        
        # Handle edge updates if related concepts changed
        if concepts_changed:
            # Remove deleted concept relationships
            for removed_concept in old_concepts - new_concepts:
                self._updates["deleted_edges"][("resource", "teaches", "concept")].append(
                    (resource_id, removed_concept)
                )
            
            # Add new concept relationships
            for added_concept in new_concepts - old_concepts:
                self._updates["added_edges"][("resource", "teaches", "concept")].append(
                    (resource_id, added_concept, self._get_edge_features(
                        "resource", "teaches", "concept", resource_id, added_concept)
                    )
                )
        
        # Check for auto-save
        self.check_auto_save()
        
        return True
    
    def delete_resource(self, resource_id: str) -> bool:
        """
        Delete a resource from the knowledge graph
        
        Args:
            resource_id: ID of the resource to delete
            
        Returns:
            True if successful, False if resource not found
        """
        resource = self.get_resource_by_id(resource_id)
        if not resource:
            logger.warning(f"Resource with ID {resource_id} not found")
            return False
        
        # Remove the resource from the list
        self.resources["resources"] = [r for r in self.resources.get("resources", []) 
                                      if r["id"] != resource_id]
        
        # Remove from lookup
        if resource_id in self._resource_lookup:
            del self._resource_lookup[resource_id]
        
        # Track the deletion
        self._updates["deleted_nodes"]["resource"].append(resource_id)
        
        # Track edge deletions for related concepts
        for concept_id in resource.get("related_concepts", []):
            self._updates["deleted_edges"][("resource", "teaches", "concept")].append(
                (resource_id, concept_id)
            )
        
        # Also need to handle learner used resources relationships
        for learner in self.learners.get("learners", []):
            usage_removed = False
            
            # Filter out usage for this resource
            if "resource_usage" in learner:
                original_len = len(learner["resource_usage"])
                learner["resource_usage"] = [
                    u for u in learner["resource_usage"] 
                    if u["resource_id"] != resource_id
                ]
                usage_removed = len(learner["resource_usage"]) < original_len
            
            # Track edge deletions if needed
            if usage_removed:
                self._updates["deleted_edges"][("learner", "used", "resource")].append(
                    (learner["id"], resource_id)
                )
        
        # Check for auto-save
        self.check_auto_save()
        
        return True
    
    def add_learner(self, name: str, email: str = "", 
                   learning_style: str = "balanced",
                   persistence: float = 0.5) -> str:
        """
        Add a new learner to the knowledge graph
        
        Args:
            name: Learner name
            email: Learner email
            learning_style: Learning style (visual, balanced, textual)
            persistence: Persistence level (0.0 to 1.0)
            
        Returns:
            ID of the newly created learner
        """
        # Map learning style to value
        style_value = LEARNING_STYLES.get(learning_style.lower(), LEARNING_STYLES["balanced"])
        
        # Validate inputs
        persistence = max(0.0, min(1.0, float(persistence)))
        
        # Generate new ID
        learner_id = str(uuid.uuid4())[:8]
        
        # Create learner
        learner = {
            "id": learner_id,
            "name": name,
            "email": email,
            "learning_style": learning_style,
            "learning_style_value": style_value,
            "persistence": persistence,
            "overall_progress": 0.0,
            "points": 0,
            "achievements": [],
            "concept_mastery": [],
            "answered_questions": [],
            "resource_usage": [],
            "learning_path": [],
            "study_streak": 0,
            "last_activity": datetime.datetime.now().isoformat(),
            "preferences": {
                "difficulty_preference": 0.5,  # 0 = easy, 1 = hard
                "content_diversity": 0.5       # 0 = focused, 1 = diverse
            }
        }
        
        # Add to learners list
        if "learners" not in self.learners:
            self.learners["learners"] = []
        
        self.learners["learners"].append(learner)
        
        # Update lookup and track the addition
        self._learner_lookup[learner_id] = learner
        self._updates["added_nodes"]["learner"].append(
            (learner_id, self._get_node_features("learner", learner))
        )
        
        # Check for auto-save
        self.check_auto_save()
        
        return learner_id
    
    def update_learner(self, learner_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update an existing learner
        
        Args:
            learner_id: ID of the learner to update
            updates: Dictionary of fields to update
            
        Returns:
            True if successful, False if learner not found
        """
        learner = self.get_learner_by_id(learner_id)
        if not learner:
            logger.warning(f"Learner with ID {learner_id} not found")
            return False
        
        # Update the learner
        for key, value in updates.items():
            # Special handling for nested preferences
            if key == "preferences" and isinstance(value, dict):
                if "preferences" not in learner:
                    learner["preferences"] = {}
                learner["preferences"].update(value)
            else:
                learner[key] = value
        
        # Update learning_style_value if learning_style was updated
        if "learning_style" in updates:
            learner["learning_style_value"] = LEARNING_STYLES.get(updates["learning_style"].lower(), 
                                                               LEARNING_STYLES["balanced"])
        
        # Ensure numeric fields are in valid range
        for field in ["persistence", "overall_progress"]:
            if field in learner:
                learner[field] = max(0.0, min(1.0, float(learner[field])))
        
        # Track updated node
        self._updates["updated_nodes"]["learner"].append(
            (learner_id, self._get_node_features("learner", learner))
        )
        
        # Check for auto-save
        self.check_auto_save()
        
        return True
    
    def delete_learner(self, learner_id: str) -> bool:
        """
        Delete a learner from the knowledge graph
        
        Args:
            learner_id: ID of the learner to delete
            
        Returns:
            True if successful, False if learner not found
        """
        learner = self.get_learner_by_id(learner_id)
        if not learner:
            logger.warning(f"Learner with ID {learner_id} not found")
            return False
        
        # Remove the learner from the list
        self.learners["learners"] = [l for l in self.learners.get("learners", []) 
                                    if l["id"] != learner_id]
        
        # Remove from lookup
        if learner_id in self._learner_lookup:
            del self._learner_lookup[learner_id]
        
        # Track the deletion
        self._updates["deleted_nodes"]["learner"].append(learner_id)
        
        # Track edge deletions for all learner relationships
        # Concept mastery
        for mastery in learner.get("concept_mastery", []):
            self._updates["deleted_edges"][("learner", "studies", "concept")].append(
                (learner_id, mastery["concept_id"])
            )
        
        # Answered questions
        answered_questions = set()
        for answer in learner.get("answered_questions", []):
            question_id = answer["question_id"]
            if question_id not in answered_questions:
                answered_questions.add(question_id)
                self._updates["deleted_edges"][("learner", "answered", "question")].append(
                    (learner_id, question_id)
                )
        
        # Resource usage
        for usage in learner.get("resource_usage", []):
            self._updates["deleted_edges"][("learner", "used", "resource")].append(
                (learner_id, usage["resource_id"])
            )
        
        # Check for auto-save
        self.check_auto_save()
        
        return True
        
    def update_mastery(self, learner_id: str, concept_id: str, mastery_level: float,
                        update_streak: bool = True) -> bool:
            """
            Update a learner's mastery level for a specific concept.
            
            Args:
                learner_id: ID of the learner
                concept_id: ID of the concept
                mastery_level: New mastery level (0.0 to 1.0)
                update_streak: Whether to update the study streak
                
            Returns:
                True if successful, False otherwise.
            """
            # Get learner and concept
            learner = self.get_learner_by_id(learner_id)
            concept = self.get_concept_by_id(concept_id)
            if not learner or not concept:
                return False
            
            # Validate mastery level
            mastery_level = max(0.0, min(1.0, float(mastery_level)))
            
            # Update or add mastery record
            mastery_updated = False
            for mastery in learner.get("concept_mastery", []):
                if mastery["concept_id"] == concept_id:
                    mastery["level"] = mastery_level
                    mastery["timestamp"] = datetime.datetime.now().isoformat()
                    mastery_updated = True
                    break
            
            if not mastery_updated:
                if "concept_mastery" not in learner:
                    learner["concept_mastery"] = []
                learner["concept_mastery"].append({
                    "concept_id": concept_id,
                    "level": mastery_level,
                    "timestamp": datetime.datetime.now().isoformat()
                })
            
            # Update study streak if requested
            if update_streak:
                today = datetime.datetime.now().date()
                if "last_activity" in learner:
                    last_date = datetime.datetime.fromisoformat(learner["last_activity"]).date()
                    days_diff = (today - last_date).days
                    if days_diff == 1:
                        learner["study_streak"] = learner.get("study_streak", 0) + 1
                    elif days_diff > 1:
                        learner["study_streak"] = 1
                    # If days_diff == 0, no change to the streak.
                else:
                    learner["study_streak"] = 1
                # Now update last_activity after computing the streak
                learner["last_activity"] = datetime.datetime.now().isoformat()
            
            # Recalculate overall progress
            self._recalculate_learner_progress(learner_id)
            
            # Update the edge between learner and concept and update learner node features
            self._updates["updated_edges"][("learner", "studies", "concept")].append(
                (learner_id, concept_id, self._get_edge_features("learner", "studies", "concept", learner_id, concept_id))
            )
            self._updates["updated_nodes"]["learner"].append(
                (learner_id, self._get_node_features("learner", learner))
            )
            self.check_auto_save()
            
            return True

    def _recalculate_learner_progress(self, learner_id: str) -> None:
        """
        Recalculate a learner's overall progress
        
        Args:
            learner_id: ID of the learner
        """
        learner = self.get_learner_by_id(learner_id)
        if not learner:
            return
            
        total_concepts = len(self.concepts.get("concepts", []))
        if total_concepts == 0:
            learner["overall_progress"] = 0.0
            return
            
        # Get mastery threshold from config
        threshold = float(self.config.get("achievements", {}).get("mastery_threshold", 0.75))
        
        # Count mastered concepts
        mastered_concepts = sum(1 for m in learner.get("concept_mastery", []) 
                               if float(m["level"]) >= threshold)
        
        # Update overall progress
        learner["overall_progress"] = mastered_concepts / total_concepts
    
    def log_resource_usage(self, learner_id: str, resource_id: str, 
                          engagement_level: float = 0.5,
                          usefulness_rating: float = 0.5) -> bool:
        """
        Log a learner's usage of a resource
        
        Args:
            learner_id: ID of the learner
            resource_id: ID of the resource
            engagement_level: Level of engagement (0.0 to 1.0)
            usefulness_rating: Rating of usefulness (0.0 to 1.0)
            
        Returns:
            True if successful, False otherwise
        """
        # Get learner and resource
        learner = self.get_learner_by_id(learner_id)
        resource = self.get_resource_by_id(resource_id)
        
        if not learner or not resource:
            return False
        
        # Validate inputs
        engagement_level = max(0.0, min(1.0, float(engagement_level)))
        usefulness_rating = max(0.0, min(1.0, float(usefulness_rating)))
        
        # Update or add resource usage record
        usage_updated = False
        if "resource_usage" not in learner:
            learner["resource_usage"] = []
            
        for usage in learner.get("resource_usage", []):
            if usage["resource_id"] == resource_id:
                usage["engagement_level"] = engagement_level
                usage["usefulness_rating"] = usefulness_rating
                usage["timestamp"] = datetime.datetime.now().isoformat()
                usage["times_accessed"] = usage.get("times_accessed", 1) + 1
                usage_updated = True
                break
        
        if not usage_updated:
            learner["resource_usage"].append({
                "resource_id": resource_id,
                "engagement_level": engagement_level,
                "usefulness_rating": usefulness_rating,
                "timestamp": datetime.datetime.now().isoformat(),
                "times_accessed": 1
            })
        
        # Update last activity timestamp
        learner["last_activity"] = datetime.datetime.now().isoformat()
        
        # Update the edge between learner and resource
        edge_exists = any(usage["resource_id"] == resource_id for usage in learner.get("resource_usage", []))
        
        if edge_exists:
            self._updates["updated_edges"][("learner", "used", "resource")].append(
                (learner_id, resource_id, self._get_edge_features(
                    "learner", "used", "resource", learner_id, resource_id)
                )
            )
        else:
            self._updates["added_edges"][("learner", "used", "resource")].append(
                (learner_id, resource_id, self._get_edge_features(
                    "learner", "used", "resource", learner_id, resource_id)
                )
            )
        
        # Update learner node features
        self._updates["updated_nodes"]["learner"].append(
            (learner_id, self._get_node_features("learner", learner))
        )
        
        # Check for auto-save
        self.check_auto_save()
        
        return True
    
    def log_question_answer(self, learner_id: str, question_id: str, 
                           correct: bool, reasoning_quality: float = 0.0,
                           response_text: str = "") -> bool:
        """
        Log a learner's response to a question
        
        Args:
            learner_id: ID of the learner
            question_id: ID of the question
            correct: Whether the answer was correct
            reasoning_quality: Quality of reasoning (0.0 to 1.0)
            response_text: Text of the learner's response
            
        Returns:
            True if successful, False otherwise
        """
        # Get learner and question
        learner = self.get_learner_by_id(learner_id)
        question = self.get_question_by_id(question_id)
        
        if not learner or not question:
            return False
        
        # Validate reasoning quality
        reasoning_quality = max(0.0, min(1.0, float(reasoning_quality)))
        
        # Log the answer
        if "answered_questions" not in learner:
            learner["answered_questions"] = []
        
        answer_record = {
            "question_id": question_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "correct": correct,
            "reasoning_quality": reasoning_quality,
            "response_text": response_text[:1000] if response_text else ""  # Limit length
        }
        
        learner["answered_questions"].append(answer_record)
        
        # Update last activity timestamp
        learner["last_activity"] = datetime.datetime.now().isoformat()
        
        # If correct, update mastery for all related concepts
        if correct:
            for concept_id in question.get("related_concepts", []):
                # Find current mastery level
                current_mastery = 0.0
                for mastery in learner.get("concept_mastery", []):
                    if mastery["concept_id"] == concept_id:
                        current_mastery = float(mastery["level"])
                        break
                
                # Calculate mastery gain using factors like:
                # - Question difficulty
                # - Diminishing returns based on current mastery
                # - Reasoning quality
                question_difficulty = float(question.get("difficulty", 0.5))
                difficulty_factor = 0.5 + question_difficulty  # Higher difficulty = more gain
                
                # Diminishing returns curve
                diminishing_factor = 1.0 - (current_mastery ** 1.5)
                
                # Reasoning quality bonus
                reasoning_bonus = reasoning_quality * 0.2
                
                # Calculate gain (max 0.3 for a single question)
                mastery_gain = min(0.3, 0.15 * difficulty_factor * diminishing_factor + reasoning_bonus)
                
                # Update mastery
                new_mastery = min(1.0, current_mastery + mastery_gain)
                self.update_mastery(learner_id, concept_id, new_mastery, update_streak=False)
        
        # Update the edge between learner and question
        # Check if this is the first answer to this question
        first_answer = len([a for a in learner.get("answered_questions", []) 
                         if a["question_id"] == question_id]) <= 1
        
        if first_answer:
            self._updates["added_edges"][("learner", "answered", "question")].append(
                (learner_id, question_id, self._get_edge_features(
                    "learner", "answered", "question", learner_id, question_id)
                )
            )
        else:
            self._updates["updated_edges"][("learner", "answered", "question")].append(
                (learner_id, question_id, self._get_edge_features(
                    "learner", "answered", "question", learner_id, question_id)
                )
            )
        
        # Update learner node features
        self._updates["updated_nodes"]["learner"].append(
            (learner_id, self._get_node_features("learner", learner))
        )
        
        # Check for auto-save
        self.check_auto_save()
        
        return True
    
    def get_hetero_data(self) -> HeteroData:
        """
        Get the heterogeneous graph data
        
        Returns:
            PyTorch Geometric HeteroData object
        """
        # Apply any pending updates before returning the graph
        self._update_hetero_graph()
        return self.hetero_data
    
    def get_concept_by_id(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a concept by its ID
        
        Args:
            concept_id: ID of the concept
            
        Returns:
            Concept dictionary or None if not found
        """
        return self._concept_lookup.get(concept_id)
    
    def get_question_by_id(self, question_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a question by its ID
        
        Args:
            question_id: ID of the question
            
        Returns:
            Question dictionary or None if not found
        """
        return self._question_lookup.get(question_id)
    
    def get_resource_by_id(self, resource_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a resource by its ID
        
        Args:
            resource_id: ID of the resource
            
        Returns:
            Resource dictionary or None if not found
        """
        return self._resource_lookup.get(resource_id)
    
    def get_learner_by_id(self, learner_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a learner by its ID
        
        Args:
            learner_id: ID of the learner
            
        Returns:
            Learner dictionary or None if not found
        """
        return self._learner_lookup.get(learner_id)
    
    def get_node_index(self, node_type: str, node_id: str) -> Optional[int]:
        """
        Get the index of a node in the graph
        
        Args:
            node_type: Type of the node (concept, question, resource, learner)
            node_id: ID of the node
            
        Returns:
            Node index or None if not found
        """
        return self.node_indices.get(node_type, {}).get(node_id)
    
    def search_concepts(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for concepts by name or description
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of matching concepts
        """
        query = query.lower()
        results = []
        
        for concept in self.concepts.get("concepts", []):
            name = concept.get("name", "").lower()
            description = concept.get("description", "").lower()
            
            if query in name or query in description:
                results.append(concept)
                
                if len(results) >= limit:
                    break
        
        return results
    
    def search_resources(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for resources by title or description
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of matching resources
        """
        query = query.lower()
        results = []
        
        for resource in self.resources.get("resources", []):
            title = resource.get("title", "").lower()
            description = resource.get("description", "").lower()
            
            if query in title or query in description:
                results.append(resource)
                
                if len(results) >= limit:
                    break
        
        return results
    
    def get_concept_graph(self) -> nx.DiGraph:
        """
        Get a NetworkX directed graph of concepts and their prerequisites
        
        Returns:
            NetworkX DiGraph object
        """
        G = nx.DiGraph()
        
        # Add concept nodes
        for concept in self.concepts.get("concepts", []):
            concept_id = concept["id"]
            G.add_node(concept_id, **{
                "name": concept.get("name", ""),
                "difficulty": concept.get("difficulty", 0.5),
                "type": "concept"
            })
        
        # Add prerequisite edges
        for concept in self.concepts.get("concepts", []):
            concept_id = concept["id"]
            for prereq_id in concept.get("prerequisites", []):
                if G.has_node(prereq_id):
                    G.add_edge(concept_id, prereq_id, type="requires")
        
        return G

# -----------------------------
# HETEROGENEOUS GNN MODEL
# -----------------------------

class RelationAttention(nn.Module):
    """
    Attention mechanism for relation-specific transformations in heterogeneous GNNs
    """
    
    def __init__(self, in_channels: int, out_channels: int, num_heads: int = 1, dropout: float = 0.0):
        super(RelationAttention, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Linear transformation for queries, keys, and values
        self.query = nn.Linear(in_channels, out_channels * num_heads)
        self.key = nn.Linear(in_channels, out_channels * num_heads)
        self.value = nn.Linear(in_channels, out_channels * num_heads)
        
        # Output projection
        self.output_proj = nn.Linear(out_channels * num_heads, out_channels)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        edge_attr_dict: Dict[Tuple[str, str, str], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass that produces updated embeddings for each node type.
        Ensures that all node types present in the input appear in the output,
        even if they didn't receive any messages during propagation.
        """
        # 1) Project node features into hidden_channels for each node type
        initial_embeddings = {}
        h_dict = {}
        for node_type, x in x_dict.items():
            if x is None or x.size(0) == 0:
                device = x.device if x is not None else torch.device("cpu")
                initial_embeddings[node_type] = torch.zeros((0, self.hidden_channels), device=device)
            else:
                initial_embeddings[node_type] = self.node_embeddings[node_type](x)
            # Initialize h_dict with the projected features
            h_dict[node_type] = initial_embeddings[node_type]

        # 2) Iterate over each hetero conv layer
        for layer_idx, hetero_conv in enumerate(self.convs):
            # Build filtered edge dicts to skip problematic edges
            filtered_edge_index_dict = {}
            filtered_edge_attr_dict = {}

            for (src_type, rel, dst_type), edge_index in edge_index_dict.items():
                if src_type not in h_dict or dst_type not in h_dict:
                    continue
                if edge_index is None or edge_index.size(1) == 0:
                    continue
                filtered_edge_index_dict[(src_type, rel, dst_type)] = edge_index
                if (src_type, rel, dst_type) in edge_attr_dict:
                    filtered_edge_attr_dict[(src_type, rel, dst_type)] = edge_attr_dict[(src_type, rel, dst_type)]
                else:
                    filtered_edge_attr_dict[(src_type, rel, dst_type)] = None

            # Apply the conv layer
            h_dict = hetero_conv(h_dict, filtered_edge_index_dict, filtered_edge_attr_dict)

            # If not the last layer, apply ReLU and dropout
            if layer_idx < self.num_layers - 1:
                for ntype, h in h_dict.items():
                    if h.size(0) > 0:
                        h = F.relu(h)
                        h = F.dropout(h, p=self.dropout, training=self.training)
                    h_dict[ntype] = h

        # 3) Ensure all original node types are present.
        # If any node type was dropped during message passing, use its initial embedding.
        for node_type, initial in initial_embeddings.items():
            if node_type not in h_dict:
                h_dict[node_type] = initial

        return h_dict

class EdgeFeatureConv(MessagePassing):
    """
    GNN convolution that explicitly uses edge features
    """
    
    def __init__(self, in_channels: int, out_channels: int, edge_dim: int, 
                aggr: str = "mean", **kwargs):
        super(EdgeFeatureConv, self).__init__(aggr=aggr, **kwargs)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        
        # Node feature transformation
        self.lin_node = nn.Linear(in_channels, out_channels)
        
        # Edge feature transformation
        self.lin_edge = nn.Linear(edge_dim, out_channels)
        
        # Final transformation after aggregation
        self.lin_out = nn.Linear(out_channels + in_channels, out_channels)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
               edge_attr: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Transform node features
        x_transformed = self.lin_node(x)
        
        # Propagate messages
        out = self.propagate(edge_index, x=x_transformed, edge_attr=edge_attr)
        
        # Combine with original features (residual connection)
        out = torch.cat([out, x], dim=1)
        out = self.lin_out(out)
        
        return out
    
    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """Message function incorporating edge features"""
        # Transform edge features
        edge_features = self.lin_edge(edge_attr)
        
        # Combine node and edge features
        return x_j + edge_features
    
    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        """Update function"""
        return aggr_out

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv, GCNConv
from typing import Dict, Tuple
from torch_geometric.nn.conv import MessagePassing
from torch import Tensor
from torch_geometric.nn import Linear

# If you have your custom EdgeFeatureConv somewhere else, import it.
# from my_module import EdgeFeatureConv  # Example placeholder

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from torch_geometric.nn import HeteroConv, GATConv, GCNConv
# from your_code import EdgeFeatureConv  # if you have a custom edge-based SAGE

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from torch_geometric.nn import HeteroConv, GATConv, GCNConv
# from your_code import EdgeFeatureConv  # If you have a custom SAGE or edge-based conv

class HeteroGNN(nn.Module):
    """
    Heterogeneous Graph Neural Network with improved handling of all node types
    """
    
    def __init__(
        self,
        metadata: Tuple,
        feature_dims: Dict[str, int],
        edge_feature_dims: Dict[Tuple[str, str, str], int],
        hidden_channels: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.2,
        model_type: str = "hetero_gat",
    ):
        super().__init__()
        
        self.metadata = metadata
        self.feature_dims = feature_dims
        self.edge_feature_dims = edge_feature_dims
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.model_type = model_type
        
        # Node type embeddings
        self.node_embeddings = nn.ModuleDict()
        for node_type, feat_dim in feature_dims.items():
            self.node_embeddings[node_type] = nn.Linear(feat_dim, hidden_channels)
        
        # Convolution layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {}
            
            # Create conv layers for all edge types
            for edge_type in metadata[1]:
                src_type, rel, dst_type = edge_type
                edge_dim = edge_feature_dims.get(edge_type, 1)
                
                if model_type == "hetero_gat":
                    conv_dict[edge_type] = GATConv(
                        (hidden_channels, hidden_channels),
                        hidden_channels,
                        heads=num_heads,
                        dropout=dropout,
                        add_self_loops=False,
                        edge_dim=edge_dim,
                        concat=False,
                    )
                elif model_type == "hetero_gcn":
                    conv_dict[edge_type] = GCNConv(
                        in_channels=hidden_channels,
                        out_channels=hidden_channels,
                        improved=True,
                        add_self_loops=True,
                    )
                else:  # Default to EdgeFeatureConv
                    conv_dict[edge_type] = EdgeFeatureConv(
                        in_channels=hidden_channels,
                        out_channels=hidden_channels,
                        edge_dim=edge_dim,
                        aggr="mean",
                    )
            
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))
        
        # Task-specific prediction heads
        self.mastery_predictor = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid(),
        )
        self.resource_predictor = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid(),
        )
        self.question_predictor = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid(),
        )
        self.path_predictor = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        """
        Forward pass with explicit preservation of all node types
        """
        # Store initial node types to ensure they remain in output
        original_node_types = set(x_dict.keys())
        
        # Initial node embeddings
        h_dict = {}
        for node_type, x in x_dict.items():
            if x is None or x.size(0) == 0:
                device = x.device if x is not None else torch.device("cpu")
                h_dict[node_type] = torch.zeros((0, self.hidden_channels), device=device)
            else:
                h_dict[node_type] = self.node_embeddings[node_type](x)
        
        # Save initial embeddings to preserve all node types
        initial_h_dict = {k: v.clone() for k, v in h_dict.items()}
        
        # Process through convolution layers
        for layer_idx, hetero_conv in enumerate(self.convs):
            # Filter edges to avoid errors
            filtered_edge_index_dict = {}
            filtered_edge_attr_dict = {}
            
            for edge_type, edge_index in edge_index_dict.items():
                src_type, _, dst_type = edge_type
                
                # Skip edges if source or destination nodes don't exist
                if src_type not in h_dict or dst_type not in h_dict:
                    continue
                
                # Skip empty edge indices
                if edge_index is None or edge_index.size(1) == 0:
                    continue
                
                # Skip if either node type has no nodes
                if h_dict[src_type].size(0) == 0 or h_dict[dst_type].size(0) == 0:
                    continue
                
                # Add the edge to filtered dictionaries
                filtered_edge_index_dict[edge_type] = edge_index
                
                # Add corresponding edge attributes if available
                if edge_type in edge_attr_dict and edge_attr_dict[edge_type] is not None:
                    filtered_edge_attr_dict[edge_type] = edge_attr_dict[edge_type]
                else:
                    # Default: One-dimensional attributes
                    edge_dim = self.edge_feature_dims.get(edge_type, 1)
                    filtered_edge_attr_dict[edge_type] = torch.ones(
                        edge_index.size(1), edge_dim, device=next(self.parameters()).device
                    )
            
            # Apply convolution if there are any edges to process
            if filtered_edge_index_dict:
                updated_h_dict = hetero_conv(h_dict, filtered_edge_index_dict, filtered_edge_attr_dict)
                
                # Apply activation and dropout except for last layer
                if layer_idx < self.num_layers - 1:
                    for node_type, h in updated_h_dict.items():
                        if h.size(0) > 0:
                            updated_h_dict[node_type] = F.dropout(F.relu(h), p=self.dropout, training=self.training)
                
                # Merge with previous embeddings - this ensures that node types that weren't
                # updated in this layer will still have their embeddings from the previous layer
                new_h_dict = {}
                for node_type in original_node_types:
                    if node_type in updated_h_dict:
                        new_h_dict[node_type] = updated_h_dict[node_type]
                    elif node_type in h_dict:
                        new_h_dict[node_type] = h_dict[node_type]
                
                h_dict = new_h_dict
            
            # If no edges could be processed in this layer, keep embeddings the same
        
        # Final check: make sure all original node types are included in output
        for node_type in original_node_types:
            if node_type not in h_dict:
                h_dict[node_type] = initial_h_dict[node_type]
        
        return h_dict
    # ------------------------------------------------------------------
    # Prediction heads: each expects embeddings of shape (batch, hidden)
    # ------------------------------------------------------------------
    def predict_mastery(self, learner_emb: torch.Tensor, concept_emb: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([learner_emb, concept_emb], dim=-1)
        return self.mastery_predictor(combined)

    def predict_resource_relevance(self, learner_emb: torch.Tensor, resource_emb: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([learner_emb, resource_emb], dim=-1)
        return self.resource_predictor(combined)

    def predict_question_difficulty(self, learner_emb: torch.Tensor, question_emb: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([learner_emb, question_emb], dim=-1)
        return self.question_predictor(combined)

    def predict_next_concept(self, learner_emb: torch.Tensor, concept_emb: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([learner_emb, concept_emb], dim=-1)
        return self.path_predictor(combined)


# ------------------------------------------
# Updated ModelTrainer with attribute fix
# ------------------------------------------
class ModelTrainer:
    """
    Handles training and evaluation of the GNN model
    """
    
    def __init__(self, model: HeteroGNN, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Training parameters
        self.learning_rate = config["gnn"]["learning_rate"]
        self.weight_decay = config["gnn"]["weight_decay"]
        self.batch_size = config["gnn"]["batch_size"]
        self.patience = config["gnn"]["patience"]
        
        # Task weights
        training_config = config["training"]
        self.task_weights = {
            "mastery": training_config["mastery_prediction_weight"],
            "resource": training_config["resource_recommendation_weight"],
            "question": training_config["question_selection_weight"],
            "path": training_config["path_prediction_weight"]
        }
        
        # Optimizer and scheduler
        self.optimizer = Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
        
        # Track best validation loss
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.epochs_without_improvement = 0
        
        logger.info(f"Model trainer initialized. Using device: {self.device}")
    
    def prepare_batch(self, batch: HeteroData) -> Tuple[Dict, Dict, Dict, Dict]:
        """
        Prepare a batch for training.

        Returns:
            (x_dict, edge_index_dict, edge_attr_dict, task_data)
        """
        # Get node features, edge indices, and edge features
        x_dict = {}
        for node_type, x in batch.x_dict.items():
            x_dict[node_type] = x.to(self.device)
        
        edge_index_dict = {}
        edge_attr_dict = {}
        
        for edge_type, edge_index in batch.edge_index_dict.items():
            # Move edge_index to device
            edge_index_dict[edge_type] = edge_index.to(self.device)

            # Build the attribute name: src__rel__dst
            attr_name = f"{edge_type[0]}__{edge_type[1]}__{edge_type[2]}"

            # Check if batch actually has edge_attr for this relation
            if hasattr(batch, attr_name) and hasattr(getattr(batch, attr_name), "edge_attr"):
                edge_attr = getattr(batch, attr_name).edge_attr
                edge_attr_dict[edge_type] = edge_attr.to(self.device)
            else:
                # Create default edge attributes if missing
                # Use the dimension from self.model.edge_feature_dims
                dim = self.model.edge_feature_dims.get(edge_type, 1)
                edge_attr_dict[edge_type] = torch.ones(edge_index.size(1), dim, device=self.device)
        
        # Prepare task data (target labels for the different tasks)
        task_data = {}
        
        # Mastery prediction task
        if ('learner', 'studies', 'concept') in edge_index_dict:
            edge_type = ('learner', 'studies', 'concept')
            mask = (batch[edge_type].train_mask 
                    if hasattr(batch[edge_type], 'train_mask') 
                    else torch.ones(edge_index_dict[edge_type].size(1), dtype=torch.bool))
            
            src_idx = edge_index_dict[edge_type][0][mask]
            dst_idx = edge_index_dict[edge_type][1][mask]
            
            # First feature is mastery level
            if edge_type in edge_attr_dict:
                labels = edge_attr_dict[edge_type][mask, 0]
                task_data['mastery'] = (src_idx, dst_idx, labels)
        
        # Resource relevance task
        if ('learner', 'used', 'resource') in edge_index_dict:
            edge_type = ('learner', 'used', 'resource')
            mask = (batch[edge_type].train_mask
                    if hasattr(batch[edge_type], 'train_mask')
                    else torch.ones(edge_index_dict[edge_type].size(1), dtype=torch.bool))
            
            src_idx = edge_index_dict[edge_type][0][mask]
            dst_idx = edge_index_dict[edge_type][1][mask]
            
            if edge_type in edge_attr_dict:
                # Typically second feature is 'usefulness'
                if edge_attr_dict[edge_type].size(1) > 1:
                    labels = edge_attr_dict[edge_type][mask, 1]
                else:
                    labels = edge_attr_dict[edge_type][mask, 0]
                task_data['resource'] = (src_idx, dst_idx, labels)
        
        # Question difficulty (correctness) task
        if ('learner', 'answered', 'question') in edge_index_dict:
            edge_type = ('learner', 'answered', 'question')
            mask = (batch[edge_type].train_mask
                    if hasattr(batch[edge_type], 'train_mask')
                    else torch.ones(edge_index_dict[edge_type].size(1), dtype=torch.bool))
            
            src_idx = edge_index_dict[edge_type][0][mask]
            dst_idx = edge_index_dict[edge_type][1][mask]
            
            if edge_type in edge_attr_dict:
                # First feature is correctness (binary)
                labels = edge_attr_dict[edge_type][mask, 0]
                task_data['question'] = (src_idx, dst_idx, labels)
        
        # Path prediction task (concept->requires->concept)
        if ('concept', 'requires', 'concept') in edge_index_dict:
            edge_type = ('concept', 'requires', 'concept')
            mask = (batch[edge_type].train_mask
                    if hasattr(batch[edge_type], 'train_mask')
                    else torch.ones(edge_index_dict[edge_type].size(1), dtype=torch.bool))
            
            src_idx = edge_index_dict[edge_type][0][mask]
            dst_idx = edge_index_dict[edge_type][1][mask]
            
            labels = torch.ones(src_idx.size(0), device=self.device)
            task_data['path'] = (src_idx, dst_idx, labels)
        
        return x_dict, edge_index_dict, edge_attr_dict, task_data

    def compute_loss(self, node_embeds: Dict[str, torch.Tensor],
                    task_data: Dict[str, Tuple]) -> Tuple[torch.Tensor, Dict[str, float]]:
        task_losses = {}
        device = next(self.model.parameters()).device
        # Initialize the total loss as a tensor that requires grad
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        # Mastery prediction task (regression with MSE)
        if 'mastery' in task_data and 'learner' in node_embeds and 'concept' in node_embeds:
            src_idx, dst_idx, labels = task_data['mastery']
            learner_emb = node_embeds['learner'][src_idx]
            concept_emb = node_embeds['concept'][dst_idx]

            preds = self.model.predict_mastery(learner_emb, concept_emb).squeeze()
            loss = F.mse_loss(preds, labels)
            task_losses['mastery'] = loss.item()
            total_loss = total_loss + self.task_weights['mastery'] * loss

        # Resource recommendation task (MSE)
        if 'resource' in task_data and 'learner' in node_embeds and 'resource' in node_embeds:
            src_idx, dst_idx, labels = task_data['resource']
            learner_emb = node_embeds['learner'][src_idx]
            resource_emb = node_embeds['resource'][dst_idx]

            preds = self.model.predict_resource_relevance(learner_emb, resource_emb).squeeze()
            loss = F.mse_loss(preds, labels)
            task_losses['resource'] = loss.item()
            total_loss = total_loss + self.task_weights['resource'] * loss

        # Question correctness task (binary cross-entropy)
        if 'question' in task_data and 'learner' in node_embeds and 'question' in node_embeds:
            src_idx, dst_idx, labels = task_data['question']
            learner_emb = node_embeds['learner'][src_idx]
            question_emb = node_embeds['question'][dst_idx]

            preds = self.model.predict_question_difficulty(learner_emb, question_emb).squeeze()
            loss = F.binary_cross_entropy(preds, labels)
            task_losses['question'] = loss.item()
            total_loss = total_loss + self.task_weights['question'] * loss

        # Path prediction task (binary cross-entropy)
        if 'path' in task_data and 'concept' in node_embeds:
            src_idx, dst_idx, labels = task_data['path']
            concept_src_emb = node_embeds['concept'][src_idx]
            concept_dst_emb = node_embeds['concept'][dst_idx]

            preds = self.model.predict_next_concept(concept_src_emb, concept_dst_emb).squeeze()
            loss = F.binary_cross_entropy(preds, labels)
            task_losses['path'] = loss.item()
            total_loss = total_loss + self.task_weights['path'] * loss

        return total_loss, task_losses

    def train_epoch(self, data_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        """
        self.model.train()
        total_loss = 0
        task_losses = defaultdict(float)
        num_batches = 0
        
        for batch in data_loader:
            self.optimizer.zero_grad()
            
            # Prepare batch
            x_dict, edge_index_dict, edge_attr_dict, task_data = self.prepare_batch(batch)
            
            # If no edges labeled for any task, skip the batch.
            if not task_data:
                continue
            
            # Forward pass
            node_embeds = self.model(x_dict, edge_index_dict, edge_attr_dict)
            # Compute loss for the tasks present in this batch
            loss, batch_task_losses = self.compute_loss(node_embeds, task_data)
            
            # Backprop & update
            loss.backward()
            self.optimizer.step()
            
            # Track losses
            total_loss += loss.item()
            for t, val in batch_task_losses.items():
                task_losses[t] += val
            
            num_batches += 1
        
        avg_loss = total_loss / max(1, num_batches)
        avg_task_losses = {
            t: (val / max(1, num_batches)) for t, val in task_losses.items()
        }
        metrics = {"loss": avg_loss}
        for t, val in avg_task_losses.items():
            metrics[f"{t}_loss"] = val
        
        return metrics

    def validate(self, data_loader: DataLoader) -> Dict[str, float]:
        """
        Validate the model on a validation set.
        """
        self.model.eval()
        total_loss = 0
        task_losses = defaultdict(float)
        task_metrics = defaultdict(list)
        num_batches = 0
        
        with torch.no_grad():
            for batch in data_loader:
                x_dict, edge_index_dict, edge_attr_dict, task_data = self.prepare_batch(batch)
                if not task_data:
                    continue
                
                node_embeds = self.model(x_dict, edge_index_dict, edge_attr_dict)
                loss, batch_task_losses = self.compute_loss(node_embeds, task_data)
                
                total_loss += loss.item()
                for t, val in batch_task_losses.items():
                    task_losses[t] += val
                
                # Optional additional metrics
                for t, (src_idx, dst_idx, labels) in task_data.items():
                    if t == 'mastery' and 'learner' in node_embeds and 'concept' in node_embeds:
                        learner_emb = node_embeds['learner'][src_idx]
                        concept_emb = node_embeds['concept'][dst_idx]
                        preds = self.model.predict_mastery(learner_emb, concept_emb).squeeze()
                        mse_val = F.mse_loss(preds, labels).item()
                        task_metrics[f"{t}_mse"].append(mse_val)

                    elif t == 'resource' and 'learner' in node_embeds and 'resource' in node_embeds:
                        learner_emb = node_embeds['learner'][src_idx]
                        resource_emb = node_embeds['resource'][dst_idx]
                        preds = self.model.predict_resource_relevance(learner_emb, resource_emb).squeeze()
                        mse_val = F.mse_loss(preds, labels).item()
                        task_metrics[f"{t}_mse"].append(mse_val)

                    elif t == 'question' and 'learner' in node_embeds and 'question' in node_embeds:
                        learner_emb = node_embeds['learner'][src_idx]
                        question_emb = node_embeds['question'][dst_idx]
                        preds = self.model.predict_question_difficulty(learner_emb, question_emb).squeeze()
                        if labels.min() < labels.max():
                            # We can try to compute AUC only if there's more than one class present.
                            try:
                                auc = roc_auc_score(labels.cpu().numpy(), preds.cpu().numpy())
                                task_metrics[f"{t}_auc"].append(auc)
                            except:
                                pass
                
                num_batches += 1
        
        avg_loss = total_loss / max(1, num_batches)
        avg_task_losses = {
            t: (val / max(1, num_batches)) for t, val in task_losses.items()
        }
        avg_task_metrics = {
            m: (sum(vals)/len(vals)) for m, vals in task_metrics.items() if len(vals) > 0
        }
        
        metrics = {'val_loss': avg_loss}
        for t, val in avg_task_losses.items():
            metrics[f"val_{t}_loss"] = val
        for m, val in avg_task_metrics.items():
            metrics[f"val_{m}"] = val
        
        return metrics

    def train(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int = 100) -> Dict[str, List[float]]:
        """
        Main training loop across epochs.
        """
        history = defaultdict(list)
        
        for epoch in range(num_epochs):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            
            # Adjust LR based on val loss
            self.scheduler.step(val_metrics['val_loss'])
            
            # Record metrics in history
            combined = {**train_metrics, **val_metrics}
            for k, v in combined.items():
                history[k].append(v)
            
            # Logging
            log_str = " - ".join([f"{k}: {v:.4f}" for k, v in combined.items()])
            logger.info(f"Epoch {epoch+1}/{num_epochs} - {log_str}")
            
            # Check for best model & early stopping
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                self.epochs_without_improvement = 0
                logger.info(f"New best model saved with val_loss: {self.best_val_loss:.4f}")
            else:
                self.epochs_without_improvement += 1
            
            if self.epochs_without_improvement >= self.patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Restore best model if we have one
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            logger.info(f"Restored best model with val_loss: {self.best_val_loss:.4f}")
        
        return dict(history)

    def save_model(self, path: str) -> None:
        """
        Save model, optimizer, scheduler states.
        """
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'best_val_loss': self.best_val_loss,
                'config': self.config
            }, path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")

    def load_model(self, path: str) -> None:
        """
        Load model, optimizer, and scheduler states.
        """
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.best_val_loss = checkpoint['best_val_loss']
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")


# -----------------------------
# ACHIEVEMENT & POINT SYSTEM
# -----------------------------

class AchievementSystem:
    """
    Enhanced achievement system with dynamic scoring and educational psychology principles
    """
    
    def __init__(self, knowledge_graph: KnowledgeGraph, config: Dict[str, Any]):
        self.kg = knowledge_graph
        self.config = config
        
        # Extract achievement configuration
        achievements_config = config.get("achievements", {})
        self.mastery_threshold = float(achievements_config.get("mastery_threshold", 0.75))
        self.points_multiplier = float(achievements_config.get("points_multiplier", 1.0))
        self.difficulty_bonus = float(achievements_config.get("difficulty_bonus", 0.5))
        self.streak_bonus = float(achievements_config.get("streak_bonus", 0.2))
        
        # Spaced repetition tracking (for persistent learning)
        self.review_intervals = SPACED_REPETITION_INTERVALS
        
        logger.info("Achievement system initialized")
    
    def get_achievement_info(self, achievement_type: str) -> Dict[str, Any]:
        """
        Get information about an achievement type
        
        Args:
            achievement_type: Type of achievement
            
        Returns:
            Dictionary of achievement information
        """
        return ACHIEVEMENT_TYPES.get(achievement_type, {
            "name": "Unknown Achievement",
            "description": "An unknown achievement type",
            "base_points": 50
        })
    
    def calculate_points(self, achievement_type: str, difficulty: float = 0.5, 
                       streak: int = 0) -> int:
        """
        Calculate points for an achievement with dynamic scoring
        
        Args:
            achievement_type: Type of achievement
            difficulty: Difficulty factor (0.0 to 1.0)
            streak: Current achievement streak (for bonuses)
            
        Returns:
            Points for the achievement
        """
        # Get base points for this achievement type
        base_points = self.get_achievement_info(achievement_type).get("base_points", 50)
        
        # Apply global multiplier
        points = base_points * self.points_multiplier
        
        # Apply difficulty bonus
        difficulty_bonus = 1.0 + (difficulty * self.difficulty_bonus)
        points *= difficulty_bonus
        
        # Apply streak bonus
        if streak > 0:
            streak_bonus = 1.0 + (min(streak, 5) * self.streak_bonus / 5)
            points *= streak_bonus
        
        return int(points)
    
    def add_achievement(self, learner_id: str, achievement_type: str, 
                       metadata: Dict[str, Any] = None) -> Tuple[bool, int, str]:
        """
        Add an achievement for a learner
        
        Args:
            learner_id: ID of the learner
            achievement_type: Type of achievement
            metadata: Additional metadata for the achievement
            
        Returns:
            Tuple of (success, points, message)
        """
        learner = self.kg.get_learner_by_id(learner_id)
        if not learner:
            logger.warning(f"Learner with ID {learner_id} not found")
            return False, 0, "Learner not found"
        
        # Initialize metadata if not provided
        metadata = metadata or {}
        
        # Get achievement info
        achievement_info = self.get_achievement_info(achievement_type)
        
        # Calculate points based on difficulty and streak
        difficulty = metadata.get("difficulty", 0.5)
        
        # Calculate streak (consecutive achievements of the same type)
        achievement_streak = 0
        if "achievements" in learner:
            recent_achievements = sorted(
                [a for a in learner["achievements"] if a.get("type") == achievement_type],
                key=lambda a: a.get("timestamp", ""), 
                reverse=True
            )
            
            if recent_achievements:
                # Count consecutive daily achievements
                today = datetime.datetime.now().date()
                streak_dates = set()
                
                for achievement in recent_achievements:
                    if "timestamp" in achievement:
                        date = datetime.datetime.fromisoformat(achievement["timestamp"]).date()
                        days_diff = (today - date).days
                        
                        if 0 <= days_diff < len(self.review_intervals):
                            streak_dates.add(date)
                        else:
                            break
                
                achievement_streak = len(streak_dates)
        
        # Calculate points
        points = self.calculate_points(achievement_type, difficulty, achievement_streak)
        
        # Create the achievement record
        achievement = {
            "id": str(uuid.uuid4())[:8],
            "type": achievement_type,
            "name": achievement_info["name"],
            "description": achievement_info["description"],
            "points": points,
            "timestamp": datetime.datetime.now().isoformat(),
            "metadata": metadata
        }
        
        # Add to learner's achievements
        if "achievements" not in learner:
            learner["achievements"] = []
        
        learner["achievements"].append(achievement)
        
        # Update total points
        learner["points"] = learner.get("points", 0) + points
        
        # Prepare message
        message = f"Earned {achievement_info['name']}: {achievement_info['description']}"
        if metadata.get("concept_name"):
            message += f" ({metadata['concept_name']})"
        
        logger.info(f"Learner {learner['name']} earned achievement: {message}")
        
        return True, points, message
    
    def award_concept_mastery(self, learner_id: str, concept_id: str) -> Tuple[bool, int, str]:
        """
        Award achievement for mastering a concept
        
        Args:
            learner_id: ID of the learner
            concept_id: ID of the concept
            
        Returns:
            Tuple of (success, points, message)
        """
        learner = self.kg.get_learner_by_id(learner_id)
        concept = self.kg.get_concept_by_id(concept_id)
        
        if not learner or not concept:
            logger.warning(f"Learner or concept not found")
            return False, 0, "Learner or concept not found"
        
        # Check if this achievement already exists
        if "achievements" in learner:
            for achievement in learner["achievements"]:
                if (achievement.get("type") == "concept_mastery" and 
                    achievement.get("metadata", {}).get("concept_id") == concept_id):
                    return False, 0, f"Already mastered concept: {concept['name']}"
        
        # Check mastery level
        mastery_level = 0.0
        for mastery in learner.get("concept_mastery", []):
            if mastery["concept_id"] == concept_id:
                mastery_level = float(mastery["level"])
                break
        
        if mastery_level < self.mastery_threshold:
            return False, 0, "Concept not mastered yet"
        
        # Award achievement
        metadata = {
            "concept_id": concept_id,
            "concept_name": concept.get("name", ""),
            "difficulty": float(concept.get("difficulty", 0.5)),
            "mastery_level": mastery_level
        }
        
        return self.add_achievement(learner_id, "concept_mastery", metadata)
    
    def award_connected_concepts(self, learner_id: str) -> List[Tuple[bool, int, str]]:
        """
        Award achievements for mastering connected concepts
        
        Args:
            learner_id: ID of the learner
            
        Returns:
            List of (success, points, message) tuples
        """
        learner = self.kg.get_learner_by_id(learner_id)
        if not learner:
            return [(False, 0, "Learner not found")]
        
        results = []
        
        # Get mastered concepts
        mastered_concepts = set()
        mastery_levels = {}
        
        for mastery in learner.get("concept_mastery", []):
            concept_id = mastery["concept_id"]
            level = float(mastery["level"])
            
            if level >= self.mastery_threshold:
                mastered_concepts.add(concept_id)
                mastery_levels[concept_id] = level
        
        # Build a graph of concepts and prerequisites
        concept_graph = self.kg.get_concept_graph()
        
        # Find completed chains of concepts
        for concept_id in mastered_concepts:
            concept = self.kg.get_concept_by_id(concept_id)
            if not concept:
                continue
            
            # Skip if no prerequisites or not all prerequisites are mastered
            prerequisites = concept.get("prerequisites", [])
            if not prerequisites or not all(prereq in mastered_concepts for prereq in prerequisites):
                continue
            
            # Create a unique identifier for this concept chain
            chain_id = concept_id + "_" + "_".join(sorted(prerequisites))
            
            # Check if this chain has already been awarded
            chain_awarded = False
            if "achievements" in learner:
                for achievement in learner["achievements"]:
                    if (achievement.get("type") == "connected_concepts" and 
                        achievement.get("metadata", {}).get("chain_id") == chain_id):
                        chain_awarded = True
                        break
            
            if chain_awarded:
                continue
            
            # Calculate chain attributes
            prereq_concepts = [self.kg.get_concept_by_id(p) for p in prerequisites]
            prereq_names = [p.get("name", "") for p in prereq_concepts if p]
            
            # Calculate average difficulty of the chain
            difficulties = [float(concept.get("difficulty", 0.5))]
            difficulties.extend([float(p.get("difficulty", 0.5)) for p in prereq_concepts if p])
            avg_difficulty = sum(difficulties) / len(difficulties)
            
            # Award achievement
            metadata = {
                "chain_id": chain_id,
                "concept_id": concept_id,
                "concept_name": concept.get("name", ""),
                "prerequisite_ids": prerequisites,
                "prerequisite_names": prereq_names,
                "difficulty": avg_difficulty,
                "chain_length": len(prerequisites) + 1
            }
            
            success, points, message = self.add_achievement(learner_id, "connected_concepts", metadata)
            results.append((success, points, message))
        
        return results
    
    def award_deep_reasoning(self, learner_id: str, question_id: str, 
                           reasoning_quality: float) -> Tuple[bool, int, str]:
        """
        Award achievement for demonstrating deep reasoning skills
        
        Args:
            learner_id: ID of the learner
            question_id: ID of the question
            reasoning_quality: Quality of reasoning (0.0 to 1.0)
            
        Returns:
            Tuple of (success, points, message)
        """
        learner = self.kg.get_learner_by_id(learner_id)
        question = self.kg.get_question_by_id(question_id)
        
        if not learner or not question:
            return False, 0, "Learner or question not found"
        
        # Only award for high-quality reasoning
        if reasoning_quality < 0.8:
            return False, 0, "Reasoning quality not sufficient for deep reasoning award"
        
        # Check if this question already awarded deep reasoning
        if "achievements" in learner:
            for achievement in learner["achievements"]:
                if (achievement.get("type") == "deep_reasoning" and 
                    achievement.get("metadata", {}).get("question_id") == question_id):
                    return False, 0, "Deep reasoning already awarded for this question"
        
        # Get concept information
        concept_names = []
        max_difficulty = 0.0
        
        for concept_id in question.get("related_concepts", []):
            concept = self.kg.get_concept_by_id(concept_id)
            if concept:
                concept_names.append(concept.get("name", ""))
                max_difficulty = max(max_difficulty, float(concept.get("difficulty", 0.5)))
        
        # Award achievement
        metadata = {
            "question_id": question_id,
            "question_text": question.get("text", "")[:100],
            "reasoning_quality": reasoning_quality,
            "difficulty": max(question.get("difficulty", 0.5), max_difficulty),
            "related_concepts": concept_names
        }
        
        return self.add_achievement(learner_id, "deep_reasoning", metadata)
    
    def award_persistence(self, learner_id: str, concept_id: str = None) -> Tuple[bool, int, str]:
        """
        Award achievement for persistent learning efforts
        
        Args:
            learner_id: ID of the learner
            concept_id: ID of a specific concept (optional)
            
        Returns:
            Tuple of (success, points, message)
        """
        learner = self.kg.get_learner_by_id(learner_id)
        if not learner:
            return False, 0, "Learner not found"
        
        # Check streak
        streak = learner.get("study_streak", 0)
        if streak < 3:  # Require at least 3 consecutive days
            return False, 0, "Study streak not long enough for persistence award"
        
        # If specific concept provided, check if already awarded
        if concept_id:
            concept = self.kg.get_concept_by_id(concept_id)
            if not concept:
                return False, 0, "Concept not found"
            
            # Check if already awarded for this concept
            if "achievements" in learner:
                for achievement in learner["achievements"]:
                    if (achievement.get("type") == "persistence" and 
                        achievement.get("metadata", {}).get("concept_id") == concept_id):
                        return False, 0, f"Persistence already awarded for concept: {concept['name']}"
            
            # Award for concept-specific persistence
            metadata = {
                "concept_id": concept_id,
                "concept_name": concept.get("name", ""),
                "difficulty": float(concept.get("difficulty", 0.5)),
                "streak": streak
            }
        else:
            # General persistence award (check if awarded in the last 7 days)
            if "achievements" in learner:
                recent_persistence = False
                for achievement in learner["achievements"]:
                    if achievement.get("type") == "persistence" and not achievement.get("metadata", {}).get("concept_id"):
                        timestamp = achievement.get("timestamp", "")
                        if timestamp:
                            date = datetime.datetime.fromisoformat(timestamp).date()
                            days_diff = (datetime.datetime.now().date() - date).days
                            if days_diff < 7:
                                recent_persistence = True
                                break
                
                if recent_persistence:
                    return False, 0, "General persistence already awarded recently"
            
            # Award for general persistence
            metadata = {
                "streak": streak,
                "difficulty": learner.get("persistence", 0.5)  # Use learner's persistence as difficulty
            }
        
        return self.add_achievement(learner_id, "persistence", metadata)
    
    def award_fast_learner(self, learner_id: str, concept_id: str) -> Tuple[bool, int, str]:
        """
        Award achievement for quickly mastering a concept
        
        Args:
            learner_id: ID of the learner
            concept_id: ID of the concept
            
        Returns:
            Tuple of (success, points, message)
        """
        learner = self.kg.get_learner_by_id(learner_id)
        concept = self.kg.get_concept_by_id(concept_id)
        
        if not learner or not concept:
            return False, 0, "Learner or concept not found"
        
        # Check mastery level
        mastery_level = 0.0
        mastery_record = None
        
        for mastery in learner.get("concept_mastery", []):
            if mastery["concept_id"] == concept_id:
                mastery_level = float(mastery["level"])
                mastery_record = mastery
                break
        
        if not mastery_record or mastery_level < self.mastery_threshold:
            return False, 0, "Concept not mastered yet"
        
        # Check if already awarded
        if "achievements" in learner:
            for achievement in learner["achievements"]:
                if (achievement.get("type") == "fast_learner" and 
                    achievement.get("metadata", {}).get("concept_id") == concept_id):
                    return False, 0, f"Fast learner already awarded for concept: {concept['name']}"
        
        # Count related question answers
        concept_questions = []
        for question in self.kg.questions.get("questions", []):
            if concept_id in question.get("related_concepts", []):
                concept_questions.append(question["id"])
        
        # Get number of attempts for related questions
        question_attempts = sum(1 for answer in learner.get("answered_questions", [])
                              if answer["question_id"] in concept_questions)
        
        # Award if mastered with few attempts
        # Scale based on concept difficulty (harder concepts allow more attempts)
        difficulty = float(concept.get("difficulty", 0.5))
        max_attempts = 3 + int(difficulty * 7)  # 3 for easiest, 10 for hardest
        
        if question_attempts <= max_attempts and question_attempts > 0:
            metadata = {
                "concept_id": concept_id,
                "concept_name": concept.get("name", ""),
                "difficulty": difficulty,
                "attempts": question_attempts,
                "mastery_level": mastery_level
            }
            
            return self.add_achievement(learner_id, "fast_learner", metadata)
        else:
            return False, 0, "Too many attempts or no attempts"
    
    def award_knowledge_explorer(self, learner_id: str) -> Tuple[bool, int, str]:
        """
        Award achievement for exploring diverse topics
        
        Args:
            learner_id: ID of the learner
            
        Returns:
            Tuple of (success, points, message)
        """
        learner = self.kg.get_learner_by_id(learner_id)
        if not learner:
            return False, 0, "Learner not found"
        
        # Check if sufficient diverse concepts have been studied
        unique_concepts = set()
        
        for mastery in learner.get("concept_mastery", []):
            concept_id = mastery["concept_id"]
            level = float(mastery["level"])
            
            # Only count concepts with at least some meaningful level of mastery
            if level >= 0.3:
                unique_concepts.add(concept_id)
        
        # Check for concept diversity (at least 5 unique concepts)
        if len(unique_concepts) < 5:
            return False, 0, "Not enough diverse concepts studied"
        
        # Check if already awarded in the last 30 days
        if "achievements" in learner:
            recent_explorer = False
            for achievement in learner["achievements"]:
                if achievement.get("type") == "knowledge_explorer":
                    timestamp = achievement.get("timestamp", "")
                    if timestamp:
                        date = datetime.datetime.fromisoformat(timestamp).date()
                        days_diff = (datetime.datetime.now().date() - date).days
                        if days_diff < 30:
                            recent_explorer = True
                            break
            
            if recent_explorer:
                return False, 0, "Knowledge explorer already awarded recently"
        
        # Collect concept names for the message
        concept_names = []
        for concept_id in unique_concepts:
            concept = self.kg.get_concept_by_id(concept_id)
            if concept:
                concept_names.append(concept.get("name", ""))
        
        # Award achievement
        metadata = {
            "concept_count": len(unique_concepts),
            "concept_names": concept_names[:5],  # Just include first 5 for brevity
            "difficulty": 0.5  # Use medium difficulty for this achievement
        }
        
        return self.add_achievement(learner_id, "knowledge_explorer", metadata)
    
    def check_for_achievements(self, learner_id: str) -> List[Tuple[bool, int, str]]:
        """
        Check for and award all possible achievements for a learner
        
        Args:
            learner_id: ID of the learner
            
        Returns:
            List of (success, points, message) tuples
        """
        results = []
        
        # Check concept mastery achievements
        learner = self.kg.get_learner_by_id(learner_id)
        if not learner:
            return [(False, 0, "Learner not found")]
        
        # Find concepts to check
        concepts_to_check = set()
        
        for mastery in learner.get("concept_mastery", []):
            if float(mastery["level"]) >= self.mastery_threshold:
                concepts_to_check.add(mastery["concept_id"])
        
        # Award concept mastery achievements
        for concept_id in concepts_to_check:
            result = self.award_concept_mastery(learner_id, concept_id)
            if result[0]:  # If successful
                results.append(result)
                
                # Also check for fast learner achievement
                fast_result = self.award_fast_learner(learner_id, concept_id)
                if fast_result[0]:
                    results.append(fast_result)
        
        # Award connected concepts achievements
        chain_results = self.award_connected_concepts(learner_id)
        results.extend([r for r in chain_results if r[0]])
        
        # Check persistence
        persistence_result = self.award_persistence(learner_id)
        if persistence_result[0]:
            results.append(persistence_result)
        
        # Check knowledge explorer
        explorer_result = self.award_knowledge_explorer(learner_id)
        if explorer_result[0]:
            results.append(explorer_result)
        
        return results

# -----------------------------
# RECOMMENDATION SYSTEM
# -----------------------------
class RecommendationSystem:
    """
    Enhanced recommendation system with personalization, exploration-exploitation balance,
    and educational psychology principles
    """
    
    def __init__(
        self,
        knowledge_graph: KnowledgeGraph,
        model: HeteroGNN,
        config: Dict[str, Any],
        device: torch.device
    ):
        self.kg = knowledge_graph
        self.model = model
        self.config = config
        self.device = device  # <-- Now we store the device here
        
        # Recommendation parameters
        rec_config = config.get("recommendation", {})
        self.exploration_weight = float(rec_config.get("exploration_weight", 0.3))
        self.recency_decay = float(rec_config.get("recency_decay", 0.9))
        self.diversity_factor = float(rec_config.get("diversity_factor", 0.2))
        self.max_recommendations = int(rec_config.get("max_recommendations", 5))
        self.personalization_weight = float(rec_config.get("personalization_weight", 0.7))
        
        # Cached embeddings
        self.node_embeddings = None
        self.last_embedding_time = 0
        
        # Maximum embedding age in seconds before refresh (5 minutes)
        self.embedding_ttl = 300
        
        logger.info("Recommendation system initialized")

    def clear_cache(self):
        self.node_embeddings = None
        self.last_embedding_time = 0
        logger.info("Node embedding cache cleared")

    def diagnose_recommendation_system(self, learner_id: str, concept_id: str = None) -> Dict[str, Any]:
        """
        Diagnose issues with the recommendation system
        
        Args:
            learner_id: ID of the learner
            concept_id: ID of a specific concept (optional)
            
        Returns:
            Dictionary with diagnostic information
        """
        diagnosis = {
            "learner_exists": False,
            "concept_exists": False if concept_id else None,
            "model_initialized": False,
            "embeddings_available": False,
            "node_indices_found": {},
            "recommendation_possible": False,
            "issues": [],
            "suggestions": []
        }
        
        # 1. Check if learner exists
        learner = self.kg.get_learner_by_id(learner_id)
        if not learner:
            diagnosis["issues"].append(f"Learner with ID '{learner_id}' not found")
            diagnosis["suggestions"].append("Add this learner or use an existing learner ID")
            return diagnosis
        
        diagnosis["learner_exists"] = True
        diagnosis["learner_info"] = {
            "name": learner.get("name", "Unknown"),
            "mastery_count": len(learner.get("concept_mastery", [])),
            "questions_answered": len(learner.get("answered_questions", [])),
            "resources_used": len(learner.get("resource_usage", []))
        }
        
        # 2. Check if concept exists (if specified)
        if concept_id:
            concept = self.kg.get_concept_by_id(concept_id)
            if not concept:
                diagnosis["issues"].append(f"Concept with ID '{concept_id}' not found")
                diagnosis["suggestions"].append("Provide a valid concept ID or leave it empty for general recommendations")
                return diagnosis
            
            diagnosis["concept_exists"] = True
            diagnosis["concept_info"] = {
                "name": concept.get("name", "Unknown"),
                "related_resources": sum(1 for r in self.kg.resources.get("resources", []) 
                                if concept_id in r.get("related_concepts", [])),
                "related_questions": sum(1 for q in self.kg.questions.get("questions", [])
                                if concept_id in q.get("related_concepts", []))
            }
        
        # 3. Check if model is initialized
        try:
            if not hasattr(self, "model") or self.model is None:
                diagnosis["issues"].append("Recommendation model is not initialized")
                diagnosis["suggestions"].append("Initialize the model or train it if not already done")
                return diagnosis
            
            # Verify model has parameters
            next(self.model.parameters())
            diagnosis["model_initialized"] = True
        except StopIteration:
            diagnosis["issues"].append("Model has no parameters")
            diagnosis["suggestions"].append("Train the model or initialize it with parameters")
            return diagnosis
        except Exception as e:
            diagnosis["issues"].append(f"Error accessing model: {str(e)}")
            diagnosis["suggestions"].append("Reinitialize the model")
            return diagnosis
        
        # 4. Check embeddings
        try:
            embeddings = self.get_node_embeddings()
            if embeddings is None:
                diagnosis["issues"].append("Node embeddings are None")
                diagnosis["suggestions"].append("Update embeddings and ensure model forward pass works correctly")
                return diagnosis
            
            diagnosis["embeddings_available"] = True
            diagnosis["embedding_info"] = {
                "node_types": list(embeddings.keys()),
                "sizes": {k: v.size() for k, v in embeddings.items()}
            }
            
            # Check if embeddings contain expected node types
            expected_types = ["learner", "concept", "resource", "question"]
            missing_types = [t for t in expected_types if t not in embeddings]
            if missing_types:
                diagnosis["issues"].append(f"Missing embeddings for node types: {missing_types}")
                diagnosis["suggestions"].append("Ensure all node types are properly processed in the model forward pass")
        except Exception as e:
            diagnosis["issues"].append(f"Error retrieving embeddings: {str(e)}")
            diagnosis["suggestions"].append("Debug get_node_embeddings method")
            return diagnosis
        
        # 5. Check node indices
        learner_idx = self.kg.get_node_index("learner", learner_id)
        diagnosis["node_indices_found"]["learner"] = learner_idx is not None
        
        if not diagnosis["node_indices_found"]["learner"]:
            diagnosis["issues"].append(f"Learner index not found for ID '{learner_id}'")
            diagnosis["suggestions"].append("Rebuild the knowledge graph or ensure learner is properly added")
        
        if concept_id:
            concept_idx = self.kg.get_node_index("concept", concept_id)
            diagnosis["node_indices_found"]["concept"] = concept_idx is not None
            
            if not diagnosis["node_indices_found"]["concept"]:
                diagnosis["issues"].append(f"Concept index not found for ID '{concept_id}'")
                diagnosis["suggestions"].append("Rebuild the knowledge graph or ensure concept is properly added")
        
        # 6. Check for recommendation possibility
        if (diagnosis["embeddings_available"] and 
            diagnosis["model_initialized"] and
            diagnosis["learner_exists"] and
            (not concept_id or diagnosis["concept_exists"]) and
            diagnosis["node_indices_found"].get("learner", False) and
            (not concept_id or diagnosis["node_indices_found"].get("concept", False))):
            diagnosis["recommendation_possible"] = True
        
        # 7. Check content availability for recommendations
        if concept_id and diagnosis["concept_exists"]:
            resources = [r for r in self.kg.resources.get("resources", []) 
                        if concept_id in r.get("related_concepts", [])]
            
            if not resources:
                diagnosis["issues"].append(f"No resources found for concept '{concept_id}'")
                diagnosis["suggestions"].append("Add resources related to this concept")
        else:
            resources = self.kg.resources.get("resources", [])
            if not resources:
                diagnosis["issues"].append("No resources available in the knowledge graph")
                diagnosis["suggestions"].append("Add resources to the knowledge graph")
        
        diagnosis["content_info"] = {
            "total_concepts": len(self.kg.concepts.get("concepts", [])),
            "total_resources": len(self.kg.resources.get("resources", [])),
            "total_questions": len(self.kg.questions.get("questions", [])),
            "total_learners": len(self.kg.learners.get("learners", []))
        }
        
        # 8. Test sample recommendation
        if diagnosis["recommendation_possible"]:
            try:
                if concept_id:
                    sample_recs = self.recommend_resources(learner_id, concept_id, top_n=1)
                else:
                    sample_recs = self.recommend_next_concepts(learner_id, top_n=1)
                
                diagnosis["sample_recommendation"] = {
                    "success": len(sample_recs) > 0,
                    "count": len(sample_recs)
                }
                
                if not sample_recs:
                    diagnosis["issues"].append("Recommendation algorithm returned no results despite passing all checks")
                    diagnosis["suggestions"].append("Debug the specific recommendation method being used")
            except Exception as e:
                diagnosis["issues"].append(f"Error during test recommendation: {str(e)}")
                diagnosis["suggestions"].append("Debug the specific recommendation method")
        
        return diagnosis

    def get_node_embeddings(self, force_refresh: bool = False) -> Dict[str, torch.Tensor]:
        current_time = time.time()
        
        # Check if embeddings need refresh
        if (
            force_refresh 
            or self.node_embeddings is None 
            or (current_time - self.last_embedding_time) > self.embedding_ttl
        ):
            logging.info("Refreshing node embeddings")
            self.update_embeddings()
        
        return self.node_embeddings

    def update_embeddings(self) -> Dict[str, torch.Tensor]:
        """Get embeddings for all node types, using fallbacks for missing ones"""
        try:
            # Get current graph data and model device
            hetero_data = self.kg.get_hetero_data()
            model_device = next(self.model.parameters()).device
            hetero_data = hetero_data.to(model_device)
            
            x_dict = hetero_data.x_dict
            edge_index_dict = hetero_data.edge_index_dict
            
            # Filter edges to only use types the model was trained with
            filtered_edge_index_dict = {}
            filtered_edge_attr_dict = {}
            
            # Define correct dimensions for each edge type
            expected_edge_dims = {
                ('question', 'tests', 'concept'): 1,
                ('resource', 'teaches', 'concept'): 2,
                ('concept', 'requires', 'concept'): 2,
                ('learner', 'studies', 'concept'): 2,
                ('learner', 'answered', 'question'): 3,
                ('learner', 'used', 'resource'): 3
            }
            
            # Process edges with correct dimensions
            for edge_type in expected_edge_dims.keys():
                if edge_type in edge_index_dict:
                    edge_index = edge_index_dict[edge_type]
                    if edge_index.size(1) > 0:
                        filtered_edge_index_dict[edge_type] = edge_index
                        
                        expected_dim = expected_edge_dims.get(edge_type, 1)
                        
                        # Adjust edge features to match expected dimensions
                        if hasattr(hetero_data[edge_type], "edge_attr") and hetero_data[edge_type].edge_attr is not None:
                            edge_attr = hetero_data[edge_type].edge_attr
                            
                            if edge_attr.size(1) != expected_dim:
                                if edge_attr.size(1) > expected_dim:
                                    filtered_edge_attr_dict[edge_type] = edge_attr[:, :expected_dim]
                                else:
                                    padding = torch.zeros(edge_attr.size(0), expected_dim - edge_attr.size(1), device=model_device)
                                    filtered_edge_attr_dict[edge_type] = torch.cat([edge_attr, padding], dim=1)
                            else:
                                filtered_edge_attr_dict[edge_type] = edge_attr
                        else:
                            filtered_edge_attr_dict[edge_type] = torch.ones(edge_index.size(1), expected_dim, device=model_device)
            
            # Run model forward pass
            self.model.eval()
            hidden_channels = getattr(self.model, 'hidden_channels', 64)
            
            with torch.no_grad():
                # Get embeddings from model
                model_output = self.model(x_dict, filtered_edge_index_dict, filtered_edge_attr_dict)
                
                # Ensure we have embeddings for all node types
                complete_embeddings = {}
                
                for node_type in ['concept', 'question', 'resource', 'learner']:
                    if node_type in model_output:
                        # Use model's output for this node type
                        complete_embeddings[node_type] = model_output[node_type]
                    elif node_type in x_dict:
                        # Create fallback zeros for missing node types
                        num_nodes = x_dict[node_type].size(0)
                        complete_embeddings[node_type] = torch.zeros((num_nodes, hidden_channels), device=model_device)
                        logger.warning(f"Created fallback embeddings for {node_type}")
                
                self.node_embeddings = complete_embeddings
            
            self.last_embedding_time = time.time()
            return self.node_embeddings
            
        except Exception as e:
            # Complete fallback if anything fails
            logger.error(f"Error during embedding update: {e}")
            
            fallback_embeddings = {}
            hidden_channels = getattr(self.model, 'hidden_channels', 64)
            
            for node_type in ['concept', 'question', 'resource', 'learner']:
                if node_type in hetero_data.x_dict:
                    x = hetero_data.x_dict[node_type]
                    if x.size(0) > 0:
                        fallback_embeddings[node_type] = torch.zeros((x.size(0), hidden_channels), device=model_device)
            
            self.node_embeddings = fallback_embeddings
            self.last_embedding_time = time.time()
            return fallback_embeddings

    def thompson_sampling(self, scores: List[float], uncertainties: List[float]) -> List[float]:
        """
        Apply Thompson sampling for exploration-exploitation balance
        
        Args:
            scores: Base recommendation scores
            uncertainties: Uncertainty estimates for each score
            
        Returns:
            Adjusted scores with exploration
        """
        # Scale uncertainties by exploration weight
        scaled_uncertainties = [u * self.exploration_weight for u in uncertainties]
        
        # Generate samples from beta distributions
        samples = []
        for score, uncertainty in zip(scores, scaled_uncertainties):
            # Use uncertainty as the variance notion in a Beta distribution
            alpha = max(0.01, score * (1 - uncertainty) / max(uncertainty, 1e-6))
            beta = max(0.01, (1 - score) * (1 - uncertainty) / max(uncertainty, 1e-6))
            sample = np.random.beta(alpha, beta)
            samples.append(sample)
        
        return samples
    
    def get_mastered_concepts(self, learner_id: str) -> Set[str]:
        """
        Get the set of mastered concepts for a learner
        
        Args:
            learner_id: ID of the learner
            
        Returns:
            Set of mastered concept IDs
        """
        learner = self.kg.get_learner_by_id(learner_id)
        if not learner:
            return set()
        
        # Get threshold from config
        threshold = float(self.config.get("achievements", {}).get("mastery_threshold", 0.75))
        
        # Return set of mastered concepts
        return {
            mastery["concept_id"] 
            for mastery in learner.get("concept_mastery", [])
            if float(mastery["level"]) >= threshold
        }
    
    def get_concept_mastery_level(self, learner_id: str, concept_id: str) -> float:
        """
        Get a learner's mastery level for a specific concept
        
        Args:
            learner_id: ID of the learner
            concept_id: ID of the concept
            
        Returns:
            Mastery level (0.0 to 1.0)
        """
        learner = self.kg.get_learner_by_id(learner_id)
        if not learner:
            return 0.0
        
        for mastery in learner.get("concept_mastery", []):
            if mastery["concept_id"] == concept_id:
                return float(mastery["level"])
        
        return 0.0
    
    def calculate_learner_concept_similarity(self, learner_id: str, concept_id: str) -> float:
        """
        Calculate similarity between a learner and concept using embeddings
        
        Args:
            learner_id: ID of the learner
            concept_id: ID of the concept
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Get embeddings
        embeddings = self.get_node_embeddings()
        
        # Get node indices
        learner_idx = self.kg.get_node_index("learner", learner_id)
        concept_idx = self.kg.get_node_index("concept", concept_id)
        
        if learner_idx is None or concept_idx is None:
            return 0.0
            
        if "learner" not in embeddings or "concept" not in embeddings:
            return 0.0
            
        learner_emb = embeddings["learner"][learner_idx]
        concept_emb = embeddings["concept"][concept_idx]
        
        # Calculate cosine similarity
        similarity = F.cosine_similarity(learner_emb.unsqueeze(0), concept_emb.unsqueeze(0)).item()
        
        # Ensure it's in range [0, 1]
        return max(0.0, min(1.0, (similarity + 1) / 2))
    
    def predict_mastery(self, learner_id: str, concept_id: str) -> float:
        """
        Predict mastery level for a learner-concept pair
        
        Args:
            learner_id: ID of the learner
            concept_id: ID of the concept
            
        Returns:
            Predicted mastery level (0.0 to 1.0)
        """
        # Get embeddings
        embeddings = self.get_node_embeddings()
        
        # Get node indices
        learner_idx = self.kg.get_node_index("learner", learner_id)
        concept_idx = self.kg.get_node_index("concept", concept_id)
        
        if learner_idx is None or concept_idx is None:
            return 0.0
            
        if "learner" not in embeddings or "concept" not in embeddings:
            return 0.0
            
        # Get embeddings for the nodes
        learner_emb = embeddings["learner"][learner_idx]
        concept_emb = embeddings["concept"][concept_idx]
        
        # Use the model to predict mastery
        self.model.eval()
        with torch.no_grad():
            mastery = self.model.predict_mastery(
                learner_emb.unsqueeze(0), concept_emb.unsqueeze(0)
            ).item()
        
        return mastery
    
    def calculate_concept_difficulty_fit(self, learner_id: str, concept_id: str) -> float:
        """
        Calculate how well a concept's difficulty matches the learner's preference
        
        Args:
            learner_id: ID of the learner
            concept_id: ID of the concept
            
        Returns:
            Difficulty fit score (0.0 to 1.0, higher is better fit)
        """
        learner = self.kg.get_learner_by_id(learner_id)
        concept = self.kg.get_concept_by_id(concept_id)
        
        if not learner or not concept:
            return 0.0
        
        # Get learner's difficulty preference
        difficulty_pref = learner.get("preferences", {}).get("difficulty_preference", 0.5)
        
        # Get concept difficulty
        concept_difficulty = float(concept.get("difficulty", 0.5))
        
        # Calculate how well they match (1.0 = perfect match, 0.0 = maximum mismatch)
        return 1.0 - abs(difficulty_pref - concept_difficulty)
    
    def calculate_media_type_fit(self, learner_id: str, resource_id: str) -> float:
        """
        Calculate how well a resource's media type matches the learner's style.
        The ideal media type for a learner is defined as the complement of their learning style.
        
        For example, if a learner's learning style is 0.0 (visual), then their ideal media type is 1.0 
        (interactive), and if the resource's media type is 1.0, then the fit is perfect (1.0).
        Conversely, if a learner's learning style is 1.0 (textual), their ideal media type would be 0.0 
        (text), so a text resource scores 1.0.
        
        Returns a fit score between 0.0 and 1.0, where 1.0 indicates a perfect match.
        """
        learner = self.kg.get_learner_by_id(learner_id)
        resource = self.kg.get_resource_by_id(resource_id)
        
        if not learner or not resource:
            return 0.0
        
        # Get the learner's learning style (0.0 for visual, 0.5 for balanced, 1.0 for textual)
        learning_style = float(learner.get("learning_style_value", LEARNING_STYLES["balanced"]))
        
        # Get the resource's media type (0.0 for text, 0.5 for video, 1.0 for interactive)
        media_type = float(resource.get("media_type_value", MEDIA_TYPES["text"]))
        
        # Define the ideal media type as the complement of the learner's learning style.
        # This means a visual learner (0.0) ideally prefers an interactive resource (1.0),
        # and a textual learner (1.0) ideally prefers a text resource (0.0).
        ideal_media = 1.0 - learning_style
        
        # Compute the fit as one minus the absolute difference between the resource's media type and the ideal.
        fit = 1.0 - abs(media_type - ideal_media)
        
        # Clamp the result to the range [0.0, 1.0] for safety.
        return max(0.0, min(1.0, fit))


    def recommend_next_concepts(self, learner_id: str, top_n: int = None) -> List[Dict[str, Any]]:
        """
        Recommend next concepts to learn
        """
        if top_n is None:
            top_n = self.max_recommendations
            
        learner = self.kg.get_learner_by_id(learner_id)
        if not learner:
            logger.warning(f"Learner with ID {learner_id} not found")
            return []
        
        # Get node embeddings
        embeddings = self.get_node_embeddings()
        
        # Get mastered concepts
        mastered_concepts = self.get_mastered_concepts(learner_id)
        
        # Get learner node index
        learner_idx = self.kg.get_node_index("learner", learner_id)
        if learner_idx is None or "learner" not in embeddings:
            return []
            
        learner_emb = embeddings["learner"][learner_idx]
        
        candidates = []
        
        # Loop through all concepts
        for concept in self.kg.concepts.get("concepts", []):
            concept_id = concept["id"]
            
            # Skip already-mastered
            if concept_id in mastered_concepts:
                continue
            
            # Check prerequisites are met
            prerequisites_met = all(
                prereq_id in mastered_concepts
                for prereq_id in concept.get("prerequisites", [])
            )
            if not prerequisites_met:
                continue
            
            concept_idx = self.kg.get_node_index("concept", concept_id)
            if concept_idx is None or "concept" not in embeddings:
                continue
            
            concept_emb = embeddings["concept"][concept_idx]
            
            # Get path_score & mastery prediction
            self.model.eval()
            with torch.no_grad():
                path_score = self.model.predict_next_concept(
                    learner_emb.unsqueeze(0), concept_emb.unsqueeze(0)
                ).item()
                mastery = self.model.predict_mastery(
                    learner_emb.unsqueeze(0), concept_emb.unsqueeze(0)
                ).item()
            
            # Weighted combo + difficulty fit
            difficulty_fit = self.calculate_concept_difficulty_fit(learner_id, concept_id)
            base_score = 0.7 * path_score + 0.3 * (1.0 - mastery)
            adjusted_score = base_score * (0.8 + 0.2 * difficulty_fit)
            
            # Estimate uncertainty for exploration
            uncertainty = 0.2
            
            candidates.append({
                "id": concept_id,
                "name": concept.get("name", ""),
                "score": adjusted_score,
                "uncertainty": uncertainty,
                "difficulty": float(concept.get("difficulty", 0.5)),
                "complexity": float(concept.get("complexity", 0.5)),
                "current_mastery": mastery,
                "importance": float(concept.get("importance", 0.5))
            })
        
        if candidates:
            scores = [c["score"] for c in candidates]
            uncertainties = [c["uncertainty"] for c in candidates]
            sampled_scores = self.thompson_sampling(scores, uncertainties)
            
            for i, sscore in enumerate(sampled_scores):
                candidates[i]["sampled_score"] = sscore
            
            candidates.sort(key=lambda x: x["sampled_score"], reverse=True)
        
        return candidates[:top_n]

    def recommend_resources(self, learner_id: str, concept_id: str, top_n: int = None) -> List[Dict[str, Any]]:
        """
        Recommend resources for learning a specific concept with improved ID handling
        """
        if top_n is None:
            top_n = self.max_recommendations
            
        # Use direct lookups instead of normalizing IDs
        learner = self.kg.get_learner_by_id(learner_id)
        concept = self.kg.get_concept_by_id(concept_id)
        if not learner or not concept:
            logger.error(f"Learner ID {learner_id} or concept ID {concept_id} not found.")
            return []

        # Get embeddings and check indices exist
        try:
            embeddings = self.get_node_embeddings()
            learner_idx = self.kg.get_node_index("learner", learner_id)
            concept_idx = self.kg.get_node_index("concept", concept_id)
            
            if learner_idx is None:
                logger.error(f"Learner index not found for ID: {learner_id}")
                return []
            if concept_idx is None:
                logger.error(f"Concept index not found for ID: {concept_id}")
                return []
            if "resource" not in embeddings:
                logger.error("Resource embeddings missing from model output")
                return []
                
            learner_emb = embeddings["learner"][learner_idx]
            
            # Track used resources without normalization
            used_resources = set(r["resource_id"] for r in learner.get("resource_usage", []))
            
            # Process each resource
            candidates = []
            for resource in self.kg.resources.get("resources", []):
                resource_id = resource.get("id")
                if not resource_id:
                    continue
                    
                # Check if resource is related to concept (direct match, no normalization)
                if concept_id not in resource.get("related_concepts", []):
                    continue
                    
                resource_idx = self.kg.get_node_index("resource", resource_id)
                if resource_idx is None:
                    continue
                    
                resource_emb = embeddings["resource"][resource_idx]
                
                # Compute relevance score
                self.model.eval()
                with torch.no_grad():
                    relevance = self.model.predict_resource_relevance(
                        learner_emb.unsqueeze(0), resource_emb.unsqueeze(0)
                    ).item()
                    
                # Calculate media type fit
                media_fit = self.calculate_media_type_fit(learner_id, resource_id)
                
                # Apply usage penalty
                usage_penalty = 0.7 if resource_id in used_resources else 1.0
                
                # Compute final score
                score = relevance * (0.7 + 0.3 * media_fit) * usage_penalty
                uncertainty = 0.1 if resource_id in used_resources else 0.3
                
                candidates.append({
                    "id": resource_id,
                    "title": resource.get("title", ""),
                    "url": resource.get("url", ""),
                    "description": resource.get("description", ""),
                    "media_type": resource.get("media_type", "text"),
                    "score": score,
                    "uncertainty": uncertainty,
                    "quality": float(resource.get("quality", 0.5)),
                    "complexity": float(resource.get("complexity", 0.5))
                })
            
            # Apply Thompson sampling for exploration-exploitation balance
            if candidates:
                scores = [c["score"] for c in candidates]
                uncertainties = [c["uncertainty"] for c in candidates]
                sampled_scores = self.thompson_sampling(scores, uncertainties)
                
                for i, sscore in enumerate(sampled_scores):
                    candidates[i]["sampled_score"] = sscore
                    
                candidates.sort(key=lambda x: x["sampled_score"], reverse=True)
                
            return candidates[:top_n]
        
        except Exception as e:
            logger.error(f"Error in recommend_resources: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    def recommend_questions(
        self,
        learner_id: str,
        concept_id: str = None,
        question_type: str = None,
        top_n: int = None
    ) -> List[Dict[str, Any]]:
        """
        Recommend questions for a learner
        """
        if top_n is None:
            top_n = self.max_recommendations
        
        learner = self.kg.get_learner_by_id(learner_id)
        if not learner:
            return []
        
        embeddings = self.get_node_embeddings()
        learner_idx = self.kg.get_node_index("learner", learner_id)
        if learner_idx is None or "learner" not in embeddings:
            return []
        learner_emb = embeddings["learner"][learner_idx]
        
        correctly_answered = set()
        incorrectly_answered = set()
        for ans in learner.get("answered_questions", []):
            if ans.get("correct", False):
                correctly_answered.add(ans["question_id"])
            else:
                incorrectly_answered.add(ans["question_id"])
        
        # For spaced repetition
        concept_masteries = {}
        concept_timestamps = {}
        for mast in learner.get("concept_mastery", []):
            concept_masteries[mast["concept_id"]] = float(mast["level"])
            if "timestamp" in mast:
                concept_timestamps[mast["concept_id"]] = datetime.datetime.fromisoformat(mast["timestamp"])
        
        candidates = []
        
        for question in self.kg.questions.get("questions", []):
            qid = question["id"]
            # Filter by concept if needed
            if concept_id and concept_id not in question.get("related_concepts", []):
                continue
            # Filter by question type
            if question_type and question.get("type", "") != question_type:
                continue
            
            q_idx = self.kg.get_node_index("question", qid)
            if q_idx is None or "question" not in embeddings:
                continue
            
            q_emb = embeddings["question"][q_idx]
            
            self.model.eval()
            with torch.no_grad():
                difficulty_for_learner = self.model.predict_question_difficulty(
                    learner_emb.unsqueeze(0), q_emb.unsqueeze(0)
                ).item()
            
            spaced_repetition_bonus = 0.0
            for c_id in question.get("related_concepts", []):
                if c_id in concept_masteries and c_id in concept_timestamps:
                    mastery_val = concept_masteries[c_id]
                    ts = concept_timestamps[c_id]
                    days_since = (datetime.datetime.now() - ts).days
                    
                    if mastery_val > 0.8:
                        interval_idx = 5
                    elif mastery_val > 0.6:
                        interval_idx = 4
                    elif mastery_val > 0.4:
                        interval_idx = 3
                    elif mastery_val > 0.2:
                        interval_idx = 2
                    else:
                        interval_idx = 1
                    interval = self.review_intervals[interval_idx]
                    
                    if days_since >= interval:
                        spaced_repetition_bonus = 0.3
                        break
            
            if qid in correctly_answered:
                score = 0.2 + spaced_repetition_bonus
                uncertainty = 0.1
            elif qid in incorrectly_answered:
                score = 0.7 * (1.0 - difficulty_for_learner)
                uncertainty = 0.2
            else:
                target_difficulty = 0.6
                score = 1.0 - abs(difficulty_for_learner - target_difficulty)
                uncertainty = 0.3
            
            candidates.append({
                "id": qid,
                "text": question.get("text", ""),
                "type": question.get("type", "application"),
                "score": score,
                "uncertainty": uncertainty,
                "difficulty": float(question.get("difficulty", 0.5)),
                "difficulty_for_learner": difficulty_for_learner,
                "related_concepts": question.get("related_concepts", [])
            })
        
        if candidates:
            scores = [c["score"] for c in candidates]
            uncertainties = [c["uncertainty"] for c in candidates]
            sampled_scores = self.thompson_sampling(scores, uncertainties)
            
            for i, sscore in enumerate(sampled_scores):
                candidates[i]["sampled_score"] = sscore
            
            candidates.sort(key=lambda x: x["sampled_score"], reverse=True)
        
        # Attempt to enforce question-type diversity
        if len(candidates) > top_n:
            question_types_seen = set()
            diverse_candidates = []
            for candidate in candidates:
                qtype = candidate.get("type", "application")
                if len(diverse_candidates) < top_n - 1:
                    if qtype not in question_types_seen or len(question_types_seen) < 3:
                        diverse_candidates.append(candidate)
                        question_types_seen.add(qtype)
                else:
                    # Add the highest scoring remaining one
                    diverse_candidates.append(candidates[len(diverse_candidates)])
                    break
            if diverse_candidates:
                return diverse_candidates
        
        return candidates[:top_n]
    
    def generate_learning_path(self, learner_id: str, target_concept_id: str = None) -> List[Dict[str, Any]]:
        """
        Generate a personalized learning path for a learner
        """
        learner = self.kg.get_learner_by_id(learner_id)
        if not learner:
            return []
        
        mastered_concepts = self.get_mastered_concepts(learner_id)
        
        # If a target concept is specified, build a path to that
        if target_concept_id:
            concept = self.kg.get_concept_by_id(target_concept_id)
            if not concept:
                return []
            
            G = self.kg.get_concept_graph()
            if target_concept_id in mastered_concepts:
                return []
            
            all_prereqs = set()
            queue = deque([target_concept_id])
            while queue:
                cur = queue.popleft()
                if cur in all_prereqs:
                    continue
                if cur != target_concept_id:
                    all_prereqs.add(cur)
                
                c = self.kg.get_concept_by_id(cur)
                if c:
                    for p in c.get("prerequisites", []):
                        queue.append(p)
            unmastered_prereqs = all_prereqs - mastered_concepts
            if target_concept_id not in mastered_concepts:
                unmastered_prereqs.add(target_concept_id)
            
            subgraph = G.subgraph(unmastered_prereqs | {target_concept_id})
            
            try:
                ordered_concepts = list(nx.topological_sort(subgraph))
                ordered_concepts.reverse()
            except nx.NetworkXUnfeasible:
                ordered_concepts = sorted(unmastered_prereqs | {target_concept_id})
            
            path = []
            for cid in ordered_concepts:
                cobj = self.kg.get_concept_by_id(cid)
                if not cobj:
                    continue
                
                prereqs_met = all(
                    pre in mastered_concepts or pre in [st["concept_id"] for st in path]
                    for pre in cobj.get("prerequisites", [])
                )
                if not prereqs_met:
                    continue
                
                resources = self.recommend_resources(learner_id, cid, top_n=2)
                questions = self.recommend_questions(learner_id, cid, top_n=2)
                mastery_level = self.get_concept_mastery_level(learner_id, cid)
                
                path.append({
                    "concept_id": cid,
                    "name": cobj.get("name", ""),
                    "description": cobj.get("description", ""),
                    "difficulty": float(cobj.get("difficulty", 0.5)),
                    "current_mastery": mastery_level,
                    "resources": resources,
                    "questions": questions,
                    "estimated_study_time_minutes": int(30 + 30 * cobj.get("complexity", 0.5))
                })
            return path
        else:
            recommended_concepts = self.recommend_next_concepts(learner_id, top_n=5)
            path = []
            for cdata in recommended_concepts:
                cid = cdata["id"]
                resources = self.recommend_resources(learner_id, cid, top_n=2)
                questions = self.recommend_questions(learner_id, cid, top_n=2)
                cobj = self.kg.get_concept_by_id(cid)
                path.append({
                    "concept_id": cid,
                    "name": cobj.get("name", ""),
                    "description": cobj.get("description", ""),
                    "difficulty": float(cobj.get("difficulty", 0.5)),
                    "current_mastery": cdata.get("current_mastery", 0.0),
                    "resources": resources,
                    "questions": questions,
                    "estimated_study_time_minutes": int(30 + 30 * cobj.get("complexity", 0.5))
                })
            return path
    
    def generate_weekly_plan(self, learner_id: str) -> Dict[str, Any]:
        """
        Generate a weekly study plan for a learner
        """
        learner = self.kg.get_learner_by_id(learner_id)
        if not learner:
            return {}
        
        learning_path = self.generate_learning_path(learner_id)
        total_minutes = sum(step.get("estimated_study_time_minutes", 30) for step in learning_path)
        daily_minutes = min(120, max(30, total_minutes // 7))
        
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        schedule = []
        
        path_index = 0
        for day in days:
            day_schedule = {
                "day": day,
                "study_minutes": daily_minutes,
                "activities": []
            }
            remaining = daily_minutes
            while remaining > 0 and path_index < len(learning_path):
                concept = learning_path[path_index]
                ctime = min(remaining, concept.get("estimated_study_time_minutes", 30))
                activity = {
                    "concept_id": concept["concept_id"],
                    "concept_name": concept["name"],
                    "minutes": ctime,
                    "resources": concept.get("resources", [])[:1],
                    "questions": concept.get("questions", [])[:1]
                }
                day_schedule["activities"].append(activity)
                remaining -= ctime
                if ctime >= concept.get("estimated_study_time_minutes", 30):
                    path_index += 1
                else:
                    learning_path[path_index]["estimated_study_time_minutes"] -= ctime
            
            schedule.append(day_schedule)
        
        return {
            "learner_id": learner_id,
            "learner_name": learner.get("name", ""),
            "total_study_time_minutes": daily_minutes * 7,
            "daily_study_time_minutes": daily_minutes,
            "schedule": schedule,
            "start_date": datetime.datetime.now().strftime("%Y-%m-%d"),
            "end_date": (datetime.datetime.now() + datetime.timedelta(days=6)).strftime("%Y-%m-%d")
        }

# -----------------------------
# LLM INTEGRATION
# -----------------------------

class LLMService:
    """
    Enhanced LLM integration with caching, fallbacks, and Socratic dialog support
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm_config = config.get("llm", {})
        self.client = None
        
        # Initialize the OpenAI client if available
        if OPENAI_AVAILABLE:
            try:
                self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                logger.info("OpenAI client initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing OpenAI client: {e}")
        
        # Response caching
        self.cache_enabled = self.llm_config.get("cache_responses", True)
        self.cache_ttl_hours = self.llm_config.get("cache_ttl_hours", 24)
        self.response_cache = {}
        
        # System message templates
        self.system_messages = {
            "socratic": (
                "You are a wise teacher who uses the Socratic method to guide learners to discover knowledge "
                "through critical thinking. Ask questions that lead the student to their own insights rather "
                "than providing direct answers. Your questions should challenge assumptions, examine implications, "
                "and help the student connect ideas."
            ),
            "stoic": (
                "You are a Stoic philosopher and teacher. Guide the learner with principles of Stoicism: "
                "focus on what's within their control, accept challenges as opportunities for growth, "
                "maintain equanimity in the face of difficulty, and pursue virtue through rational thought. "
                "Frame learning as a process that requires patience and persistence."
            ),
            "combined": (
                "You are a wise teacher who combines Socratic questioning with Stoic philosophy. "
                "Guide the learner to discover knowledge through thoughtful questions while emphasizing "
                "Stoic principles: focus on what you can control, view challenges as opportunities, "
                "maintain emotional equilibrium, and pursue knowledge as a virtue. "
                "Your questions should lead the student to insights while encouraging resilience and rational thought."
            ),
            "evaluator": (
                "You are an educational evaluator who assesses student responses for conceptual understanding "
                "and quality of reasoning. Provide accurate, fair assessments that consider both factual "
                "correctness and depth of reasoning. Be specific about strengths and areas for improvement."
            ),
            "content_creator": (
                "You are an expert educational content creator specializing in clear, engaging materials "
                "that facilitate effective learning. Create content that is accurate, well-structured, "
                "and appropriate for the target audience. Include examples, analogies, and appropriate "
                "challenges to promote deep understanding."
            ),
            "metacognition": (
                "You are an expert in metacognitive strategies for learning. Help students reflect on "
                "their own thinking processes, identify strengths and weaknesses in their understanding, "
                "and develop strategies for more effective learning. Encourage self-questioning, "
                "knowledge monitoring, and strategic planning."
            )
        }
        
        logger.info("LLM service initialized")
    
    def _get_cache_key(self, system_message: str, prompt: str) -> str:
        """Generate a cache key from system message and prompt"""
        combined = f"{system_message}|{prompt}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _check_cache(self, cache_key: str) -> Optional[str]:
        """Check if a response is in the cache and not expired"""
        if not self.cache_enabled:
            return None
            
        if cache_key in self.response_cache:
            timestamp, response = self.response_cache[cache_key]
            hours_old = (time.time() - timestamp) / 3600
            
            if hours_old < self.cache_ttl_hours:
                return response
        
        return None
    
    def _store_in_cache(self, cache_key: str, response: str) -> None:
        """Store a response in the cache"""
        if self.cache_enabled:
            self.response_cache[cache_key] = (time.time(), response)
    
    def generate_response(self, system_message: str, prompt: str, 
                        temperature: float = None, max_tokens: int = None) -> str:
        """
        Generate a response using the LLM
        
        Args:
            system_message: System message for the LLM
            prompt: User prompt
            temperature: Temperature for generation (default from config)
            max_tokens: Maximum tokens in response (default from config)
            
        Returns:
            Generated text response
        """
        # Default parameters from config
        temperature = temperature if temperature is not None else float(self.llm_config.get("temperature", 0.7))
        max_tokens = max_tokens if max_tokens is not None else int(self.llm_config.get("max_tokens", 200))
        
        # Check cache
        cache_key = self._get_cache_key(system_message, prompt)
        cached_response = self._check_cache(cache_key)
        
        if cached_response:
            return cached_response
        
        # Generate response with retries
        retry_attempts = int(self.llm_config.get("retry_attempts", 3))
        retry_delay = float(self.llm_config.get("retry_delay_seconds", 2))
        
        if self.client:
            # Use OpenAI API
            response_text = safe_request(
                self._call_openai_api,
                system_message, prompt, temperature, max_tokens,
                max_retries=retry_attempts,
                base_delay=retry_delay
            )
            
            if response_text:
                # Store in cache
                self._store_in_cache(cache_key, response_text)
                return response_text
        
        # Fallback response if API call fails or client not available
        fallback_response = self._generate_fallback_response(prompt, system_message)
        
        # Store fallback in cache with shorter TTL
        if self.cache_enabled:
            # Store with 1 hour TTL for fallbacks
            self.response_cache[cache_key] = (time.time() - (self.cache_ttl_hours - 1) * 3600, fallback_response)
        
        return fallback_response
    
    def _call_openai_api(self, system_message: str, prompt: str, 
                       temperature: float, max_tokens: int) -> str:
        """
        Call the OpenAI API
        
        Args:
            system_message: System message
            prompt: User prompt
            temperature: Temperature for generation
            max_tokens: Maximum tokens in response
            
        Returns:
            Generated text response
        """
        model = self.llm_config.get("model", "gpt-3.5-turbo")
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            raise
    
    def _generate_fallback_response(self, prompt: str, system_type: str) -> str:
        """
        Generate a fallback response when API is unavailable
        
        Args:
            prompt: User prompt
            system_type: Type of system message
            
        Returns:
            Fallback response text
        """
        # Extract keywords from prompt
        keywords = [word.lower() for word in re.findall(r'\b\w+\b', prompt) 
                  if word.lower() not in {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at'}]
        
        if "socratic" in system_type.lower():
            questions = [
                "What aspects of this topic do you find most challenging?",
                "How would you explain this concept in your own words?",
                "What evidence supports your understanding?",
                "How does this relate to what you already know?",
                "What might be an alternative perspective?",
                "Have you considered the implications?",
                "What questions do you still have about this topic?"
            ]
            return random.choice(questions)
        
        elif "stoic" in system_type.lower():
            messages = [
                "Remember that progress comes through consistent effort, not immediate mastery.",
                "Focus on what's within your control: your attention, effort, and persistence.",
                "View challenges as opportunities to strengthen your understanding.",
                "How might you approach this with both reason and resilience?",
                "What virtue can you practice as you engage with this material?"
            ]
            return random.choice(messages)
        
        elif "evaluator" in system_type.lower():
            return "Your response shows some understanding of the concepts. Consider exploring the relationships between ideas more deeply and providing specific examples to demonstrate your points."
        
        else:
            return "I encourage you to explore this topic further. What specific aspects would you like to understand better?"
    
    def generate_socratic_question(self, concept_name: str, concept_description: str = "",
                                style: str = "combined") -> str:
        """
        Generate a Socratic question about a concept
        
        Args:
            concept_name: Name of the concept
            concept_description: Description of the concept
            style: Style of questioning (socratic, stoic, combined)
            
        Returns:
            Generated question
        """
        system_message = self.system_messages.get(style, self.system_messages["combined"])
        
        if not concept_description:
            concept_description = concept_name
        
        prompt = (
            f"Generate a thought-provoking Socratic question about the concept of '{concept_name}'.\n\n"
            f"Concept description: {concept_description}\n\n"
            f"The question should guide the learner to discover key insights about this concept "
            f"through their own reasoning rather than directly stating the answer. "
            f"Use questioning techniques that encourage critical thinking and conceptual connections."
        )
        
        return self.generate_response(system_message, prompt, temperature=0.7)
    
    def generate_followup_question(self, concept_name: str, previous_question: str,
                                 learner_response: str, style: str = "combined") -> str:
        """
        Generate a follow-up question based on a learner's response
        
        Args:
            concept_name: Name of the concept
            previous_question: Previous question asked
            learner_response: Learner's response to the question
            style: Style of questioning (socratic, stoic, combined)
            
        Returns:
            Follow-up question
        """
        system_message = self.system_messages.get(style, self.system_messages["combined"])
        
        prompt = (
            f"The learner is studying the concept of '{concept_name}'.\n\n"
            f"Previous question: {previous_question}\n\n"
            f"Learner's response: \"{learner_response}\"\n\n"
            f"Generate a thoughtful follow-up question that:\n"
            f"1. Acknowledges the strengths in their current understanding\n"
            f"2. Addresses any misconceptions or gaps gently\n"
            f"3. Probes deeper into their understanding\n"
            f"4. Guides them toward discovering important insights they haven't yet expressed\n\n"
            f"The question should continue the Socratic dialogue without directly providing answers."
        )
        
        return self.generate_response(system_message, prompt, temperature=0.7)

# Replace the evaluate_response method with this fixed version:

    def evaluate_response(self, concept_name: str, question: str, 
                        learner_response: str, detailed: bool = False) -> Dict[str, Any]:
        """
        Evaluate a learner's response to a question
        
        Args:
            concept_name: Name of the concept
            question: Question asked
            learner_response: Learner's response
            detailed: Whether to provide detailed evaluation
            
        Returns:
            Dictionary with evaluation results
        """
        system_message = self.system_messages["evaluator"]
        
        prompt = (
            f"Evaluate the learner's response about the concept of '{concept_name}'.\n\n"
            f"Question: {question}\n\n"
            f"Learner's Response: \"{learner_response}\"\n\n"
            f"Provide a complete evaluation with the following components:\n"
            f"1. CORRECTNESS (0-100): How factually accurate is the response?\n"
            f"2. REASONING (0-100): How well does the response demonstrate depth of reasoning?\n"
            f"3. CONCEPTUAL (0-100): How well does the response show conceptual understanding?\n"
            f"4. MISCONCEPTIONS: Identify any misconceptions (if none, state 'None detected')\n"
            f"5. STRENGTHS: Key strengths of the response\n"
            f"6. FEEDBACK: Constructive feedback for improvement\n\n"
            f"Format your response exactly as:\n"
            f"CORRECTNESS: [0-100]\n"
            f"REASONING: [0-100]\n"
            f"CONCEPTUAL: [0-100]\n"
            f"MISCONCEPTIONS: [description]\n"
            f"STRENGTHS: [description]\n"
            f"FEEDBACK: [description]"
        )
        
        response = self.generate_response(system_message, prompt, temperature=0.3, max_tokens=500)
        
        # Parse the evaluation
        evaluation = {
            "correctness": 0,
            "reasoning": 0,
            "conceptual": 0,
            "misconceptions": "Unable to evaluate",
            "strengths": "Unable to evaluate",
            "feedback": "Unable to evaluate"
        }
        
        try:
            # Extract scores and descriptions
            correctness_match = re.search(r'CORRECTNESS:\s*(\d+)', response)
            reasoning_match = re.search(r'REASONING:\s*(\d+)', response)
            conceptual_match = re.search(r'CONCEPTUAL:\s*(\d+)', response)
            
            if correctness_match:
                evaluation["correctness"] = int(correctness_match.group(1))
            
            if reasoning_match:
                evaluation["reasoning"] = int(reasoning_match.group(1))
            
            if conceptual_match:
                evaluation["conceptual"] = int(conceptual_match.group(1))
            
            # Extract text sections
            misconceptions_match = re.search(r'MISCONCEPTIONS:(.*?)(?:STRENGTHS:|$)', response, re.DOTALL)
            strengths_match = re.search(r'STRENGTHS:(.*?)(?:FEEDBACK:|$)', response, re.DOTALL)
            feedback_match = re.search(r'FEEDBACK:(.*?)$', response, re.DOTALL)
            
            if misconceptions_match:
                evaluation["misconceptions"] = misconceptions_match.group(1).strip()
            
            if strengths_match:
                evaluation["strengths"] = strengths_match.group(1).strip()
            
            if feedback_match:
                evaluation["feedback"] = feedback_match.group(1).strip()
            
            # Calculate derived metrics
            evaluation["overall_score"] = (
                evaluation["correctness"] * 0.4 + 
                evaluation["reasoning"] * 0.4 + 
                evaluation["conceptual"] * 0.2
            )
            
            evaluation["reasoning_quality"] = evaluation["reasoning"] / 100.0
            evaluation["is_correct"] = evaluation["correctness"] >= 70
            
        except Exception as e:
            logger.error(f"Error parsing evaluation response: {e}")
            evaluation["parsing_error"] = str(e)
        
        # Return full response if detailed evaluation requested
        if detailed:
            evaluation["full_response"] = response
        
        return evaluation

    def generate_concept_hint(self, concept_name: str, concept_description: str,
                            difficulty_level: str = "medium") -> str:
        """
        Generate a hint for understanding a concept
        
        Args:
            concept_name: Name of the concept
            concept_description: Description of the concept
            difficulty_level: Difficulty level (easy, medium, hard)
            
        Returns:
            Generated hint
        """
        system_message = self.system_messages["content_creator"]
        
        prompt = (
            f"Generate a helpful hint for understanding the concept of '{concept_name}'.\n\n"
            f"Concept description: {concept_description}\n\n"
            f"Difficulty level: {difficulty_level}\n\n"
            f"The hint should:\n"
            f"1. Provide a clue or analogy that makes the concept more accessible\n"
            f"2. Not explain the entire concept directly\n"
            f"3. Highlight a key aspect that helps unlock understanding\n"
            f"4. Be appropriate for the {difficulty_level} difficulty level\n\n"
            f"Make the hint concise (2-3 sentences) and memorable."
        )
        
        return self.generate_response(system_message, prompt, temperature=0.7)
    
    def generate_metacognitive_prompt(self, concept_name: str, learning_stage: str) -> str:
        """
        Generate a metacognitive prompt to help learners reflect on their understanding
        
        Args:
            concept_name: Name of the concept
            learning_stage: Stage of learning (beginning, middle, mastery)
            
        Returns:
            Metacognitive prompt
        """
        system_message = self.system_messages["metacognition"]
        
        prompt = (
            f"Generate a metacognitive prompt for a learner studying '{concept_name}'.\n\n"
            f"Learning stage: {learning_stage}\n\n"
            f"The prompt should help the learner reflect on:\n"
            f"1. Their current understanding of the concept\n"
            f"2. Any gaps or uncertainties in their knowledge\n"
            f"3. Connections to their prior knowledge\n"
            f"4. Effective strategies for deepening their understanding\n\n"
            f"The prompt should be appropriate for the {learning_stage} stage of learning "
            f"and encourage meaningful self-reflection."
        )
        
        return self.generate_response(system_message, prompt, temperature=0.7)
    
    def generate_learning_strategy(self, concept_name: str, learning_style: str,
                                 mastery_level: float) -> str:
        """
        Generate a personalized learning strategy for a concept
        
        Args:
            concept_name: Name of the concept
            learning_style: Learner's preferred learning style
            mastery_level: Current mastery level (0.0 to 1.0)
            
        Returns:
            Personalized learning strategy
        """
        system_message = self.system_messages["content_creator"]
        
        # Map mastery level to text description
        if mastery_level < 0.3:
            mastery_text = "beginner (just starting to learn)"
        elif mastery_level < 0.6:
            mastery_text = "intermediate (familiar but not confident)"
        else:
            mastery_text = "advanced (solid understanding but not mastered)"
        
        prompt = (
            f"Generate a personalized learning strategy for mastering the concept of '{concept_name}'.\n\n"
            f"Learning style: {learning_style}\n"
            f"Current mastery level: {mastery_text}\n\n"
            f"The strategy should:\n"
            f"1. Include 3-4 specific activities tailored to their learning style\n"
            f"2. Provide concrete next steps appropriate for their current mastery level\n"
            f"3. Suggest ways to overcome common obstacles in learning this concept\n"
            f"4. Incorporate principles of effective learning (spaced repetition, active recall, etc.)\n\n"
            f"Make the strategy practical and actionable."
        )
        
        return self.generate_response(system_message, prompt, temperature=0.7)
    
    def generate_encouragement(self, learner_name: str, recent_achievements: List[str] = None,
                             learning_challenges: List[str] = None) -> str:
        """
        Generate a personalized encouragement message
        
        Args:
            learner_name: Name of the learner
            recent_achievements: List of recent achievements (optional)
            learning_challenges: List of learning challenges (optional)
            
        Returns:
            Personalized encouragement message
        """
        system_message = self.system_messages["stoic"]
        
        achievements_text = ""
        if recent_achievements:
            achievements_text = "Recent achievements:\n- " + "\n- ".join(recent_achievements)
        
        challenges_text = ""
        if learning_challenges:
            challenges_text = "Current challenges:\n- " + "\n- ".join(learning_challenges)
        
        prompt = (
            f"Craft a brief, personalized message of encouragement for {learner_name}.\n\n"
            f"{achievements_text}\n\n"
            f"{challenges_text}\n\n"
            f"The message should:\n"
            f"1. Acknowledge their progress and/or challenges with specific references\n"
            f"2. Incorporate Stoic wisdom about learning, growth, and resilience\n"
            f"3. Provide perspective that helps them maintain motivation\n"
            f"4. Be warm and supportive while encouraging self-discipline\n\n"
            f"Keep the message concise (3-4 sentences) and genuinely encouraging."
        )
        
        return self.generate_response(system_message, prompt, temperature=0.7)

# -----------------------------
# CONTENT GENERATION
# -----------------------------

class ContentGenerator:
    """
    Enhanced educational content generation with improved error handling and validation
    """
    
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
        logger.info("Content generator initialized")
    
    def generate_question(self, concept_id: str, question_type: str = "application", 
                        difficulty: float = 0.5) -> Dict[str, Any]:
        """
        Generate a question for a concept
        
        Args:
            concept_id: ID of the concept
            question_type: Type of question to generate
            difficulty: Difficulty level (0.0 to 1.0)
            
        Returns:
            Dictionary with generated question data
        """
        # Convert question type to text
        question_type_text = question_type.capitalize()
        
        # Convert difficulty to text
        if difficulty < 0.3:
            difficulty_text = "easy"
        elif difficulty < 0.7:
            difficulty_text = "moderate"
        else:
            difficulty_text = "challenging"
        
        # Get concept information from knowledge graph (assuming kg is an attribute)
        concept = self.kg.get_concept_by_id(concept_id)
        if not concept:
            logger.warning(f"Concept with ID {concept_id} not found")
            return None
        
        concept_name = concept.get("name", "")
        concept_description = concept.get("description", "")
        
        # Generate the question
        system_message = self.llm_service.system_messages["content_creator"]
        
        prompt = (
            f"Create a {difficulty_text} {question_type_text} question about {concept_name}.\n\n"
            f"Concept description: {concept_description}\n\n"
            f"For a {question_type_text} question, the learner should need to:\n"
            )
        
        if question_type == "recall":
            prompt += "- Recall and state factual information about the concept."
        elif question_type == "application":
            prompt += "- Apply the concept to a specific situation or problem."
        elif question_type == "analysis":
            prompt += "- Analyze relationships between concepts and evaluate implications."
        
        prompt += (
            f"\n\nDifficulty should be {difficulty_text}.\n\n"
            f"Format your response as JSON with these fields:\n"
            f"1. question: The question text\n"
            f"2. answer: The comprehensive answer\n"
            f"3. hint: A helpful hint for someone struggling with the question\n"
            f"4. misconceptions: Common misconceptions to watch for\n"
        )
        
        # Get response from LLM
        response_text = self.llm_service.generate_response(
            system_message, prompt, temperature=0.7, max_tokens=500
        )
        
        # Parse JSON from the response
        try:
            # Find JSON in the response
            json_match = re.search(r'({[\s\S]*})', response_text)
            if json_match:
                json_str = json_match.group(1)
                data = json.loads(json_str)
            else:
                # Try to create structured data from unstructured response
                lines = response_text.split('\n')
                data = {}
                
                for line in lines:
                    if line.startswith("question:") or line.startswith("Question:"):
                        data["question"] = line.split(":", 1)[1].strip()
                    elif line.startswith("answer:") or line.startswith("Answer:"):
                        data["answer"] = line.split(":", 1)[1].strip()
                    elif line.startswith("hint:") or line.startswith("Hint:"):
                        data["hint"] = line.split(":", 1)[1].strip()
                    elif line.startswith("misconceptions:") or line.startswith("Misconceptions:"):
                        data["misconceptions"] = line.split(":", 1)[1].strip()
            
            # Create question data
            question_data = {
                "id": str(uuid.uuid4())[:8],
                "text": data.get("question", ""),
                "answer": data.get("answer", ""),
                "hint": data.get("hint", ""),
                "misconceptions": data.get("misconceptions", ""),
                "difficulty": difficulty,
                "type": question_type,
                "type_value": QUESTION_TYPES.get(question_type, QUESTION_TYPES["application"]),
                "discriminator_power": 0.5,  # Default value
                "related_concepts": [concept_id]
            }
            
            return question_data
            
        except Exception as e:
            logger.error(f"Error parsing question generation response: {e}")
            logger.debug(f"Response text: {response_text}")
            
            # Create a minimal valid question as fallback
            return {
                "id": str(uuid.uuid4())[:8],
                "text": f"Explain the concept of {concept_name} in your own words.",
                "answer": f"A comprehensive explanation of {concept_name} would include key aspects from the concept description.",
                "difficulty": difficulty,
                "type": question_type,
                "type_value": QUESTION_TYPES.get(question_type, QUESTION_TYPES["application"]),
                "discriminator_power": 0.5,
                "related_concepts": [concept_id]
            }
    
    def generate_resource(self, concept_id: str, media_type: str = "text") -> Dict[str, Any]:
        """
        Generate a learning resource for a concept
        
        Args:
            concept_id: ID of the concept
            media_type: Type of media to suggest
            
        Returns:
            Dictionary with generated resource data
        """
        # Get concept information from knowledge graph
        concept = self.kg.get_concept_by_id(concept_id)
        if not concept:
            logger.warning(f"Concept with ID {concept_id} not found")
            return None
        
        concept_name = concept.get("name", "")
        concept_description = concept.get("description", "")
        
        # Generate resource suggestion
        system_message = self.llm_service.system_messages["content_creator"]
        
        prompt = (
            f"Suggest a high-quality {media_type} resource for learning about {concept_name}.\n\n"
            f"Concept description: {concept_description}\n\n"
            f"Create a fictional but realistic resource that would be ideal for learning this concept.\n\n"
            f"Format your response as JSON with these fields:\n"
            f"1. title: A descriptive title for the resource\n"
            f"2. url: A realistic URL (fictional but plausible)\n"
            f"3. description: A brief description of what the resource covers\n"
            f"4. complexity: A rating from 0 to 1 of the resource's complexity\n"
            f"5. features: Key features that make this resource effective\n"
        )
        
        # Get response from LLM
        response_text = self.llm_service.generate_response(
            system_message, prompt, temperature=0.7, max_tokens=300
        )
        
        # Parse JSON from the response
        try:
            # Find JSON in the response
            json_match = re.search(r'({[\s\S]*})', response_text)
            if json_match:
                json_str = json_match.group(1)
                data = json.loads(json_str)
            else:
                # Try to create structured data from unstructured response
                lines = response_text.split('\n')
                data = {}
                
                for line in lines:
                    if line.startswith("title:") or line.startswith("Title:"):
                        data["title"] = line.split(":", 1)[1].strip()
                    elif line.startswith("url:") or line.startswith("URL:"):
                        data["url"] = line.split(":", 1)[1].strip()
                    elif line.startswith("description:") or line.startswith("Description:"):
                        data["description"] = line.split(":", 1)[1].strip()
                    elif line.startswith("complexity:") or line.startswith("Complexity:"):
                        try:
                            data["complexity"] = float(line.split(":", 1)[1].strip())
                        except:
                            data["complexity"] = 0.5
                    elif line.startswith("features:") or line.startswith("Features:"):
                        data["features"] = line.split(":", 1)[1].strip()
            
            # Ensure complexity is a valid float between 0 and 1
            complexity = 0.5
            if "complexity" in data:
                try:
                    complexity = float(data["complexity"])
                    complexity = max(0.0, min(1.0, complexity))
                except:
                    pass
            
            # Create resource data
            resource_data = {
                "id": str(uuid.uuid4())[:8],
                "title": data.get("title", f"Resource on {concept_name}"),
                "url": data.get("url", f"https://example.com/resources/{concept_name.lower().replace(' ', '-')}"),
                "description": data.get("description", ""),
                "features": data.get("features", ""),
                "quality": 0.8,  # Default high quality
                "complexity": complexity,
                "media_type": media_type,
                "media_type_value": MEDIA_TYPES.get(media_type, MEDIA_TYPES["text"]),
                "related_concepts": [concept_id]
            }
            
            return resource_data
            
        except Exception as e:
            logger.error(f"Error parsing resource generation response: {e}")
            logger.debug(f"Response text: {response_text}")
            
            # Create a minimal valid resource as fallback
            return {
                "id": str(uuid.uuid4())[:8],
                "title": f"{concept_name} - Learning Resource",
                "url": f"https://example.com/resources/{concept_name.lower().replace(' ', '-')}",
                "description": f"A comprehensive resource for learning about {concept_name}.",
                "quality": 0.8,
                "complexity": 0.5,
                "media_type": media_type,
                "media_type_value": MEDIA_TYPES.get(media_type, MEDIA_TYPES["text"]),
                "related_concepts": [concept_id]
            }
    
    def generate_learning_objectives(self, concept_id: str, num_objectives: int = 3) -> List[str]:
        """
        Generate learning objectives for a concept
        
        Args:
            concept_id: ID of the concept
            num_objectives: Number of objectives to generate
            
        Returns:
            List of learning objectives
        """
        # Get concept information
        concept = self.kg.get_concept_by_id(concept_id)
        if not concept:
            logger.warning(f"Concept with ID {concept_id} not found")
            return []
        
        concept_name = concept.get("name", "")
        concept_description = concept.get("description", "")
        
        # Generate learning objectives
        system_message = self.llm_service.system_messages["content_creator"]
        
        prompt = (
            f"Create {num_objectives} clear learning objectives for the concept of {concept_name}.\n\n"
            f"Concept description: {concept_description}\n\n"
            f"Learning objectives should:\n"
            f"1. Be specific and measurable\n"
            f"2. Use action verbs (explain, analyze, compare, etc.)\n"
            f"3. Focus on different levels of understanding (recall, application, analysis)\n"
            f"4. Be aligned with the concept description\n\n"
            f"Format your response as a numbered list, with each objective on a new line, "
            f"starting with 'By the end of this module, learners will be able to:'"
        )
        
        # Get response from LLM
        response_text = self.llm_service.generate_response(
            system_message, prompt, temperature=0.7, max_tokens=300
        )
        
        # Parse objectives from the response
        objectives = []
        
        try:
            # Look for a "learners will be able to" intro
            intro_match = re.search(r'By the end of this module,[\s\w]+:', response_text, re.IGNORECASE)
            
            if intro_match:
                # Extract everything after the intro
                content = response_text[intro_match.end():].strip()
                
                # Split by numbered points or bullet points
                lines = re.split(r'\n\s*(?:\d+[.)]|\*|-)\s*', content)
                lines = [line.strip() for line in lines if line.strip()]
                
                objectives.extend(lines)
            else:
                # Fall back to looking for numbered lines
                numbered_lines = re.findall(r'\d+[.)] (.*?)(?=\n\d+[.)]|\n\n|$)', response_text)
                if numbered_lines:
                    objectives.extend(numbered_lines)
                else:
                    # Fall back to splitting by newlines
                    lines = [line.strip() for line in response_text.split('\n') if line.strip()]
                    objectives.extend(lines)
        
        except Exception as e:
            logger.error(f"Error parsing learning objectives: {e}")
        
        # Ensure each objective starts with a verb
        verbs = ["identify", "explain", "describe", "analyze", "evaluate", "apply", "compare", 
                "contrast", "demonstrate", "calculate", "create", "design", "develop", "interpret"]
        
        for i, objective in enumerate(objectives):
            # Check if the objective starts with a verb
            if not any(objective.lower().startswith(verb) for verb in verbs):
                # Prepend a suitable verb
                objectives[i] = f"Explain {objective[0].lower()}{objective[1:]}"
        
        # Limit to requested number
        return objectives[:num_objectives]
    
    # Sets the knowledge graph reference
    def set_knowledge_graph(self, kg):
        """Set the knowledge graph reference"""
        self.kg = kg

    def generate_content_for_concept(self, concept_id: str) -> Dict[str, Any]:
        """
        Generate questions and resources for a specific concept
        
        Args:
            concept_id: ID of the concept
            
        Returns:
            Dictionary with generated content information
        """
        concept = self.kg.get_concept_by_id(concept_id)
        if not concept:
            logger.warning(f"Concept with ID {concept_id} not found")
            return None
        
        results = {
            "concept_id": concept_id,
            "name": concept.get("name", ""),
            "questions": [],
            "resources": []
        }
        
        # Generate questions (one of each type)
        question_types = ["recall", "application", "analysis"]
        for q_type in question_types:
            # Adjust difficulty based on question type
            q_difficulty = float(concept.get("difficulty", 0.5))
            if q_type == "recall":
                q_difficulty = max(0.1, q_difficulty - 0.2)
            elif q_type == "analysis":
                q_difficulty = min(0.9, q_difficulty + 0.2)
            
            # Generate and add question
            question_data = self.generate_question(
                concept_id=concept_id,
                question_type=q_type,
                difficulty=q_difficulty
            )
            
            if question_data:
                question_id = self.kg.add_question(
                    text=question_data["text"],
                    answer=question_data["answer"],
                    difficulty=q_difficulty,
                    question_type=q_type,
                    discriminator_power=0.5,
                    related_concepts=[concept_id]
                )
                
                results["questions"].append({
                    "id": question_id,
                    "text": question_data["text"],
                    "type": q_type
                })
        
        # Generate resources (text and video)
        media_types = ["text", "video"]
        for media_type in media_types:
            resource_data = self.generate_resource(
                concept_id=concept_id,
                media_type=media_type
            )
            
            if resource_data:
                resource_id = self.kg.add_resource(
                    title=resource_data["title"],
                    url=resource_data["url"],
                    description=resource_data["description"],
                    quality=0.8,
                    complexity=float(concept.get("complexity", 0.5)),
                    media_type=media_type,
                    related_concepts=[concept_id]
                )
                
                results["resources"].append({
                    "id": resource_id,
                    "title": resource_data["title"],
                    "media_type": media_type
                })
        
        return results

# -----------------------------
# MAIN SYSTEM INTEGRATION
# -----------------------------

class EducationalSystem:
    """
    Main system integrating all components with improved architecture
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        # Load configuration
        self.config = load_config(config_path)
        
        # Initialize knowledge graph
        self.knowledge_graph = KnowledgeGraph(self.config)
        
        # Initialize LLM service
        self.llm_service = LLMService(self.config)
        
        # Initialize content generator
        self.content_generator = ContentGenerator(self.llm_service)
        self.content_generator.set_knowledge_graph(self.knowledge_graph)
        
        # Feature dimensions for the model
        self.feature_dims = {
            'concept': 4,  # [difficulty, complexity, importance, has_prerequisites]
            'question': 3,  # [difficulty, type_value, discriminator_power]
            'resource': 3,  # [quality, complexity, media_type_value]
            'learner': 4    # [overall_progress, learning_style_value, persistence, experience]
        }
        
        # Edge feature dimensions
        self.edge_feature_dims = {
            ('learner', 'studies', 'concept'): 2,     # [mastery_level, recency]
            ('learner', 'answered', 'question'): 3,   # [correctness, reasoning_quality, attempts]
            ('learner', 'used', 'resource'): 3,       # [engagement, usefulness, times_accessed]
            ('concept', 'requires', 'concept'): 2,    # [strength, importance]
            ('question', 'tests', 'concept'): 1,      # [relevance]
            ('resource', 'teaches', 'concept'): 2     # [relevance, comprehensiveness]
        }
        
        # Initialize GNN model
        self.model = self.initialize_model()
        
        # Initialize model trainer
        self.model_trainer = ModelTrainer(self.model, self.config)
        
        # Initialize achievement system
        self.achievement_system = AchievementSystem(self.knowledge_graph, self.config)
        
        # Initialize recommendation system
        self.recommendation_system = RecommendationSystem(
            self.knowledge_graph, self.model, self.config, device=self.model_trainer.device  # <-- PASS THE DEVICE HERE

        )
        
        # Load saved model if available
        self.load_model()
        
        logger.info("Educational system initialized successfully")
    
    def initialize_model(self) -> HeteroGNN:
        """
        Initialize the GNN model
        
        Returns:
            Initialized HeteroGNN model
        """
        # Get metadata for heterogeneous graph
        metadata = (['concept', 'question', 'resource', 'learner'], EDGE_TYPES)
        
        # Get model configuration
        gnn_config = self.config["gnn"]
        
        # Create the model
        model = HeteroGNN(
            metadata=metadata,
            feature_dims=self.feature_dims,
            edge_feature_dims=self.edge_feature_dims,
            hidden_channels=gnn_config.get("hidden_channels", 64),
            num_layers=gnn_config.get("num_layers", 2),
            num_heads=gnn_config.get("num_heads", 4),
            dropout=gnn_config.get("dropout", 0.2),
            model_type=gnn_config.get("model_type", "hetero_gat")
        )
        
        return model
    
    def diagnose_recommendations(self, learner_id: str, concept_id: str = None) -> Dict[str, Any]:
        """
        Diagnose recommendation system issues
        
        Args:
            learner_id: ID of the learner
            concept_id: ID of the concept (optional)
            
        Returns:
            Diagnostic report
        """
        # Clear embeddings cache to force refresh
        self.recommendation_system.clear_cache()
        
        # Get diagnosis
        diagnosis = self.recommendation_system.diagnose_recommendation_system(learner_id, concept_id)
        
        # Add model information
        diagnosis["model_info"] = {
            "type": self.config["gnn"].get("model_type", "unknown"),
            "hidden_channels": self.config["gnn"].get("hidden_channels", 0),
            "num_layers": self.config["gnn"].get("num_layers", 0),
            "trained": hasattr(self.model_trainer, "best_val_loss") and self.model_trainer.best_val_loss != float('inf')
        }
        
        # Add graph structure info
        if hasattr(self.knowledge_graph, "hetero_data") and self.knowledge_graph.hetero_data is not None:
            hetero_data = self.knowledge_graph.hetero_data
            diagnosis["graph_info"] = {
                "node_counts": {node_type: data.num_nodes 
                            for node_type, data in hetero_data.node_items()},
                "edge_counts": {edge_type: data.num_edges 
                            for edge_type, data in hetero_data.edge_items()}
            }
        
        # If not recommended, add quick fixes
        if not diagnosis["recommendation_possible"]:
            diagnosis["quick_fixes"] = []
            
            # Check for simple issues
            if not diagnosis.get("learner_exists", False):
                diagnosis["quick_fixes"].append(
                    "Add a learner with: system.knowledge_graph.add_learner(name='Test User')"
                )
            
            if concept_id and not diagnosis.get("concept_exists", False):
                diagnosis["quick_fixes"].append(
                    f"Add a concept with: system.knowledge_graph.add_concept(name='Test Concept')"
                )
            
            if not diagnosis.get("model_initialized", False) or not diagnosis["model_info"].get("trained", False):
                diagnosis["quick_fixes"].append(
                    "Train the model with: system.train_model(num_epochs=10)"
                )
                
            if diagnosis["content_info"].get("total_resources", 0) == 0:
                diagnosis["quick_fixes"].append(
                    "Add example data with: system.add_example_data()"
                )
        
        return diagnosis
    # Add this method to the EducationalSystem class
    def import_concepts_from_file(self, json_path: str, overwrite: bool = False) -> Tuple[int, int]:
        """
        Import concepts from a JSON file
        
        Args:
            json_path: Path to the JSON file containing concepts
            overwrite: Whether to overwrite existing concepts with the same ID
            
        Returns:
            Tuple of (concepts_added, concepts_skipped)
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Error loading JSON file: {e}")
            return 0, 0
        
        # Check if the JSON has the expected structure
        if "concepts" not in data or not isinstance(data["concepts"], list):
            logger.error("JSON file does not have a 'concepts' array")
            return 0, 0
        
        # Get existing concept IDs to avoid duplicates
        existing_ids = {c["id"] for c in self.knowledge_graph.concepts.get("concepts", [])}
        
        # If overwriting, clear existing concepts
        if overwrite:
            self.knowledge_graph.concepts["concepts"] = []
            existing_ids = set()
            logger.info("Cleared existing concepts")
        
        # Import concepts
        concepts_added = 0
        concepts_skipped = 0
        
        for concept in data["concepts"]:
            # Skip if ID already exists and not overwriting
            if concept["id"] in existing_ids and not overwrite:
                logger.info(f"Skipping concept {concept['id']}: {concept.get('name', 'Unnamed')} (ID already exists)")
                concepts_skipped += 1
                continue
                
            # Special handling for prerequisites
            prereqs = concept.get("prerequisites", [])
            if isinstance(prereqs, list):
                # Make sure prerequisites are strings
                prereqs = [str(p) for p in prereqs]
            else:
                # If prereqs is not a list, convert to an empty list
                prereqs = []
            
            # Add concept using the knowledge graph API
            concept_id = self.knowledge_graph.add_concept(
                name=concept.get("name", f"Concept {concept['id']}"),
                description=concept.get("description", ""),
                difficulty=float(concept.get("difficulty", 0.5)),
                complexity=float(concept.get("complexity", 0.5)),
                importance=float(concept.get("importance", 0.5)),
                prerequisites=prereqs
            )
            
            logger.info(f"Added concept: {concept.get('name', 'Unnamed')} (ID: {concept_id})")
            concepts_added += 1
        
        # Save changes
        self.knowledge_graph.save()
        
        return concepts_added, concepts_skipped

    def train_model(self, num_epochs: int = None) -> Dict[str, List[float]]:
        """
        Train the GNN model
        
        Args:
            num_epochs: Number of epochs to train for (default to config)
            
        Returns:
            Dictionary of training history
        """
        if num_epochs is None:
            num_epochs = self.config["gnn"].get("epochs", 100)
        
        # Get heterogeneous graph data
        hetero_data = self.knowledge_graph.get_hetero_data()
        
        # Skip training if the graph is too small
        min_nodes = 5
        if (len(hetero_data['concept'].x) < min_nodes or 
            len(hetero_data['learner'].x) < 1):
            logger.warning("Not enough data to train the model")
            return {}
        
        logger.info(f"Training model with {len(hetero_data['concept'].x)} concepts and "
                   f"{len(hetero_data['learner'].x)} learners")
        
        # Create train/validation split
        val_ratio = self.config["gnn"].get("validation_ratio", 0.2)
        
        # For each edge type, create train/val masks
        for edge_type in hetero_data.edge_types:
            edge_index = hetero_data[edge_type].edge_index
            num_edges = edge_index.size(1)
            
            if num_edges > 0:
                # Create random permutation
                perm = torch.randperm(num_edges)
                val_size = int(num_edges * val_ratio)
                
                # Create masks
                train_mask = torch.zeros(num_edges, dtype=torch.bool)
                val_mask = torch.zeros(num_edges, dtype=torch.bool)
                
                train_mask[perm[val_size:]] = True
                val_mask[perm[:val_size]] = True
                
                # Add masks to the data
                hetero_data[edge_type].train_mask = train_mask
                hetero_data[edge_type].val_mask = val_mask
        
        # Create data loaders
        batch_size = self.config["gnn"].get("batch_size", 32)
        
        # For small graphs, use the entire graph as a batch
        if len(hetero_data['concept'].x) + len(hetero_data['learner'].x) < 100:
            train_loader = [hetero_data]
            val_loader = [hetero_data]
        else:
            # Use neighbor sampling for larger graphs
            train_loader = NeighborLoader(
                hetero_data,
                num_neighbors=[10, 5],
                batch_size=batch_size,
                input_nodes=('learner', None)
            )
            
            val_loader = NeighborLoader(
                hetero_data,
                num_neighbors=[10, 5],
                batch_size=batch_size,
                input_nodes=('learner', None)
            )
        
        # Train the model
        history = self.model_trainer.train(train_loader, val_loader, num_epochs)
        
        # Save the trained model
        self.save_model()
        
        return history
    
    def save_model(self, path: str = None) -> None:
        """
        Save the trained model
        
        Args:
            path: Path to save the model (default to data/model.pt)
        """
        if path is None:
            # Create directory if it doesn't exist
            os.makedirs("data", exist_ok=True)
            path = "data/model.pt"
        
        self.model_trainer.save_model(path)
    
    def load_model(self, path: str = None) -> bool:
        """
        Load a trained model
        
        Args:
            path: Path to load the model from (default to data/model.pt)
            
        Returns:
            True if successful, False otherwise
        """
        if path is None:
            path = "data/model.pt"
        
        if not os.path.exists(path):
            logger.info(f"No saved model found at {path}")
            return False
        
        try:
            self.model_trainer.load_model(path)
            logger.info(f"Model loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def add_example_data(self) -> bool:
        """
        Add example data for testing
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Add concepts with prerequisites
            logger.info("Adding example concepts")
            math_id = self.knowledge_graph.add_concept(
                name="Basic Mathematics",
                description="Fundamental mathematical concepts including arithmetic, algebra, and basic geometry.",
                difficulty=0.3,
                complexity=0.4,
                importance=0.9
            )
            
            algebra_id = self.knowledge_graph.add_concept(
                name="Algebra",
                description="Manipulation of mathematical symbols and rules for manipulating these symbols.",
                difficulty=0.5,
                complexity=0.6,
                importance=0.8,
                prerequisites=[math_id]
            )
            
            calculus_id = self.knowledge_graph.add_concept(
                name="Calculus",
                description="Study of continuous change and its applications, including differentiation and integration.",
                difficulty=0.7,
                complexity=0.8,
                importance=0.8,
                prerequisites=[algebra_id]
            )
            
            programming_id = self.knowledge_graph.add_concept(
                name="Programming Basics",
                description="Fundamental concepts of programming including variables, conditions, loops, and functions.",
                difficulty=0.4,
                complexity=0.5,
                importance=0.9
            )
            
            python_id = self.knowledge_graph.add_concept(
                name="Python Programming",
                description="Programming in Python including syntax, data structures, and object-oriented principles.",
                difficulty=0.5,
                complexity=0.6,
                importance=0.8,
                prerequisites=[programming_id]
            )
            
            ml_id = self.knowledge_graph.add_concept(
                name="Machine Learning Basics",
                description="Fundamental concepts of machine learning including supervised learning, unsupervised learning, and evaluation metrics.",
                difficulty=0.7,
                complexity=0.8,
                importance=0.9,
                prerequisites=[python_id, algebra_id]
            )
            
            # Add questions
            logger.info("Adding example questions")
            self.knowledge_graph.add_question(
                text="What are the basic arithmetic operations?",
                answer="The basic arithmetic operations are addition, subtraction, multiplication, and division.",
                difficulty=0.2,
                question_type="recall",
                related_concepts=[math_id]
            )
            
            self.knowledge_graph.add_question(
                text="Solve for x in the equation 2x + 5 = 15.",
                answer="2x + 5 = 15\n2x = 15 - 5\n2x = 10\nx = 5",
                difficulty=0.4,
                question_type="application",
                related_concepts=[algebra_id]
            )
            
            self.knowledge_graph.add_question(
                text="Calculate the derivative of f(x) = x^2 + 3x + 2.",
                answer="f'(x) = 2x + 3",
                difficulty=0.6,
                question_type="application",
                related_concepts=[calculus_id]
            )
            
            self.knowledge_graph.add_question(
                text="Write a Python function to calculate the factorial of a number.",
                answer="def factorial(n):\n    if n == 0 or n == 1:\n        return 1\n    else:\n        return n * factorial(n-1)",
                difficulty=0.5,
                question_type="application",
                related_concepts=[python_id]
            )
            
            # Add resources
            logger.info("Adding example resources")
            self.knowledge_graph.add_resource(
                title="Introduction to Algebra",
                url="https://www.khanacademy.org/math/algebra",
                description="Comprehensive tutorials on algebra fundamentals.",
                quality=0.9,
                complexity=0.5,
                media_type="video",
                related_concepts=[algebra_id]
            )
            
            self.knowledge_graph.add_resource(
                title="Calculus Made Easy",
                url="https://www.calc101.com",
                description="Interactive calculus tutorials with step-by-step solutions.",
                quality=0.8,
                complexity=0.7,
                media_type="interactive",
                related_concepts=[calculus_id]
            )
            
            self.knowledge_graph.add_resource(
                title="Python for Beginners",
                url="https://www.python.org/about/gettingstarted/",
                description="Official Python documentation for beginners.",
                quality=0.9,
                complexity=0.4,
                media_type="text",
                related_concepts=[python_id]
            )
            
            # Add learners
            logger.info("Adding example learners")
            learner1_id = self.knowledge_graph.add_learner(
                name="Alice Johnson",
                email="alice@example.com",
                learning_style="visual",
                persistence=0.8
            )
            
            learner2_id = self.knowledge_graph.add_learner(
                name="Bob Smith",
                email="bob@example.com",
                learning_style="balanced",
                persistence=0.6
            )
            
            # Add mastery levels
            self.knowledge_graph.update_mastery(learner1_id, math_id, 0.9)
            self.knowledge_graph.update_mastery(learner1_id, algebra_id, 0.7)
            self.knowledge_graph.update_mastery(learner1_id, programming_id, 0.8)
            self.knowledge_graph.update_mastery(learner1_id, python_id, 0.6)
            
            self.knowledge_graph.update_mastery(learner2_id, math_id, 0.8)
            self.knowledge_graph.update_mastery(learner2_id, programming_id, 0.9)
            self.knowledge_graph.update_mastery(learner2_id, python_id, 0.8)
            
            # Log question answers
            logger.info("Adding example question answers")
            questions = self.knowledge_graph.questions.get("questions", [])
            
            if questions:
                self.knowledge_graph.log_question_answer(
                    learner1_id, questions[0]["id"], True, 0.8
                )
                
                self.knowledge_graph.log_question_answer(
                    learner1_id, questions[1]["id"], True, 0.7
                )
                
                self.knowledge_graph.log_question_answer(
                    learner2_id, questions[0]["id"], True, 0.9
                )
                
                self.knowledge_graph.log_question_answer(
                    learner2_id, questions[3]["id"], True, 0.8
                )
            
            # Log resource usage
            logger.info("Adding example resource usage")
            resources = self.knowledge_graph.resources.get("resources", [])
            
            if resources:
                self.knowledge_graph.log_resource_usage(
                    learner1_id, resources[0]["id"], 0.8, 0.9
                )
                
                self.knowledge_graph.log_resource_usage(
                    learner2_id, resources[2]["id"], 0.9, 0.8
                )
            
            # Check for achievements
            logger.info("Checking for achievements")
            self.achievement_system.check_for_achievements(learner1_id)
            self.achievement_system.check_for_achievements(learner2_id)
            
            # Save all changes
            self.knowledge_graph.save()
            
            logger.info("Example data added successfully")
            return True
        except Exception as e:
            logger.error(f"Error adding example data: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_learner_stats(self, learner_id: str) -> Dict[str, Any]:
        """
        Get comprehensive stats for a learner
        
        Args:
            learner_id: ID of the learner
            
        Returns:
            Dictionary of learner statistics
        """
        learner = self.knowledge_graph.get_learner_by_id(learner_id)
        if not learner:
            return {"error": "Learner not found"}
        
        # Basic stats
        stats = {
            "id": learner_id,
            "name": learner.get("name", ""),
            "email": learner.get("email", ""),
            "overall_progress": float(learner.get("overall_progress", 0.0)),
            "points": int(learner.get("points", 0)),
            "learning_style": learner.get("learning_style", "balanced"),
            "streak": int(learner.get("study_streak", 0))
        }
        
        # Mastery statistics
        mastered_concepts = []
        in_progress_concepts = []
        
        for mastery in learner.get("concept_mastery", []):
            concept_id = mastery["concept_id"]
            level = float(mastery["level"])
            timestamp = mastery.get("timestamp", "")
            
            concept = self.knowledge_graph.get_concept_by_id(concept_id)
            if not concept:
                continue
                
            concept_data = {
                "id": concept_id,
                "name": concept.get("name", ""),
                "mastery": level,
                "last_updated": timestamp
            }
            
            if level >= self.achievement_system.mastery_threshold:
                mastered_concepts.append(concept_data)
            else:
                in_progress_concepts.append(concept_data)
        
        # Sort by mastery level (descending)
        mastered_concepts.sort(key=lambda x: x["mastery"], reverse=True)
        in_progress_concepts.sort(key=lambda x: x["mastery"], reverse=True)
        
        stats["mastered_concepts"] = mastered_concepts
        stats["in_progress_concepts"] = in_progress_concepts
        stats["num_mastered"] = len(mastered_concepts)
        stats["num_in_progress"] = len(in_progress_concepts)
        
        # Question statistics
        correct_answers = 0
        total_answers = len(learner.get("answered_questions", []))
        
        for answer in learner.get("answered_questions", []):
            if answer.get("correct", False):
                correct_answers += 1
        
        stats["question_stats"] = {
            "total_answered": total_answers,
            "correct_answers": correct_answers,
            "accuracy": correct_answers / total_answers if total_answers > 0 else 0.0
        }
        
        # Achievement statistics
        achievements = learner.get("achievements", [])
        stats["achievement_stats"] = {
            "total_achievements": len(achievements),
            "recent_achievements": achievements[-3:] if achievements else []
        }
        
        # Learning path progress
        learning_path = learner.get("learning_path", [])
        if learning_path:
            completed_steps = sum(1 for step in learning_path if step.get("completed", False))
            stats["learning_path_progress"] = {
                "total_steps": len(learning_path),
                "completed_steps": completed_steps,
                "completion_percentage": completed_steps / len(learning_path) if learning_path else 0.0
            }
        
        return stats
    
    def get_concept_stats(self, concept_id: str) -> Dict[str, Any]:
        """
        Get comprehensive stats for a concept
        
        Args:
            concept_id: ID of the concept
            
        Returns:
            Dictionary of concept statistics
        """
        concept = self.knowledge_graph.get_concept_by_id(concept_id)
        if not concept:
            return {"error": "Concept not found"}
        
        # Basic stats
        stats = {
            "id": concept_id,
            "name": concept.get("name", ""),
            "description": concept.get("description", ""),
            "difficulty": float(concept.get("difficulty", 0.5)),
            "complexity": float(concept.get("complexity", 0.5)),
            "importance": float(concept.get("importance", 0.5))
        }
        
        # Prerequisite concepts
        prerequisites = []
        for prereq_id in concept.get("prerequisites", []):
            prereq = self.knowledge_graph.get_concept_by_id(prereq_id)
            if prereq:
                prerequisites.append({
                    "id": prereq_id,
                    "name": prereq.get("name", "")
                })
        
        stats["prerequisites"] = prerequisites
        
        # Dependent concepts (concepts that have this concept as a prerequisite)
        dependents = []
        for other_concept in self.knowledge_graph.concepts.get("concepts", []):
            if concept_id in other_concept.get("prerequisites", []):
                dependents.append({
                    "id": other_concept["id"],
                    "name": other_concept.get("name", "")
                })
        
        stats["dependent_concepts"] = dependents
        
        # Related questions
        related_questions = []
        for question in self.knowledge_graph.questions.get("questions", []):
            if concept_id in question.get("related_concepts", []):
                related_questions.append({
                    "id": question["id"],
                    "text": question.get("text", ""),
                    "difficulty": float(question.get("difficulty", 0.5)),
                    "type": question.get("type", "application")
                })
        
        stats["related_questions"] = related_questions
        stats["num_questions"] = len(related_questions)
        
        # Related resources
        related_resources = []
        for resource in self.knowledge_graph.resources.get("resources", []):
            if concept_id in resource.get("related_concepts", []):
                related_resources.append({
                    "id": resource["id"],
                    "title": resource.get("title", ""),
                    "media_type": resource.get("media_type", "text"),
                    "quality": float(resource.get("quality", 0.5))
                })
        
        stats["related_resources"] = related_resources
        stats["num_resources"] = len(related_resources)
        
        # Learner mastery statistics
        learner_stats = []
        mastery_distribution = {
            "0-25%": 0,
            "26-50%": 0,
            "51-75%": 0,
            "76-100%": 0
        }
        
        total_mastery = 0.0
        num_learners = 0
        
        for learner in self.knowledge_graph.learners.get("learners", []):
            for mastery in learner.get("concept_mastery", []):
                if mastery["concept_id"] == concept_id:
                    level = float(mastery["level"])
                    
                    learner_stats.append({
                        "learner_id": learner["id"],
                        "learner_name": learner.get("name", ""),
                        "mastery_level": level
                    })
                    
                    # Update distribution
                    if level <= 0.25:
                        mastery_distribution["0-25%"] += 1
                    elif level <= 0.5:
                        mastery_distribution["26-50%"] += 1
                    elif level <= 0.75:
                        mastery_distribution["51-75%"] += 1
                    else:
                        mastery_distribution["76-100%"] += 1
                    
                    total_mastery += level
                    num_learners += 1
                    break
        
        stats["learner_stats"] = learner_stats
        stats["num_learners"] = num_learners
        stats["average_mastery"] = total_mastery / num_learners if num_learners > 0 else 0.0
        stats["mastery_distribution"] = mastery_distribution
        
        return stats
    
    def interactive_session(self, learner_id: str, concept_id: str = None) -> Dict[str, Any]:
        """
        Start an interactive learning session for a learner
        
        Args:
            learner_id: ID of the learner
            concept_id: ID of the concept (optional)
            
        Returns:
            Dictionary with session information
        """
        learner = self.knowledge_graph.get_learner_by_id(learner_id)
        if not learner:
            return {"error": "Learner not found"}
        
        # Select concept if not specified
        if not concept_id:
            recommended_concepts = self.recommendation_system.recommend_next_concepts(learner_id, top_n=1)
            if not recommended_concepts:
                return {"error": "No suitable concepts found to study"}
            
            concept_id = recommended_concepts[0]["id"]
        
        concept = self.knowledge_graph.get_concept_by_id(concept_id)
        if not concept:
            return {"error": "Concept not found"}
        
        # Get current mastery level
        mastery_level = 0.0
        for mastery in learner.get("concept_mastery", []):
            if mastery["concept_id"] == concept_id:
                mastery_level = float(mastery["level"])
                break
        
        # Get recommended resources
        resources = self.recommendation_system.recommend_resources(learner_id, concept_id, top_n=2)
        
        # Get recommended questions
        questions = self.recommendation_system.recommend_questions(learner_id, concept_id, top_n=3)
        
        # Generate an initial Socratic question
        initial_question = self.llm_service.generate_socratic_question(
            concept["name"], concept.get("description", "")
        )
        
        # Generate learning objectives
        learning_objectives = self.content_generator.generate_learning_objectives(concept_id, num_objectives=3)
        
        # Create session
        session = {
            "learner_id": learner_id,
            "learner_name": learner.get("name", ""),
            "concept_id": concept_id,
            "concept_name": concept.get("name", ""),
            "concept_description": concept.get("description", ""),
            "current_mastery": mastery_level,
            "resources": resources,
            "questions": questions,
            "initial_question": initial_question,
            "learning_objectives": learning_objectives,
            "session_id": str(uuid.uuid4())[:8],
            "started_at": datetime.datetime.now().isoformat()
        }
        
        return session
    
    def process_response(self, session_id: str, learner_id: str, concept_id: str, 
                       question: str, response: str) -> Dict[str, Any]:
        """
        Process a learner's response in an interactive session
        
        Args:
            session_id: ID of the session
            learner_id: ID of the learner
            concept_id: ID of the concept
            question: Question that was asked
            response: Learner's response
            
        Returns:
            Dictionary with evaluation and feedback
        """
        # Validate inputs
        learner = self.knowledge_graph.get_learner_by_id(learner_id)
        concept = self.knowledge_graph.get_concept_by_id(concept_id)
        
        if not learner or not concept:
            return {"error": "Learner or concept not found"}
        
        # Evaluate the response
        evaluation = self.llm_service.evaluate_response(
            concept["name"], question, response, detailed=True
        )
        
        # Generate follow-up question
        followup_question = self.llm_service.generate_followup_question(
            concept["name"], question, response
        )
        
        # Update mastery if the response is good enough
        if evaluation["overall_score"] >= 70:
            current_mastery = 0.0
            for mastery in learner.get("concept_mastery", []):
                if mastery["concept_id"] == concept_id:
                    current_mastery = float(mastery["level"])
                    break
            
            # Calculate mastery gain
            reasoning_quality = evaluation["reasoning_quality"]
            mastery_gain = 0.1 * reasoning_quality
            
            # Apply diminishing returns
            mastery_gain *= (1.0 - current_mastery * 0.5)
            
            # Update mastery
            new_mastery = min(1.0, current_mastery + mastery_gain)
            if new_mastery > current_mastery:
                self.knowledge_graph.update_mastery(learner_id, concept_id, new_mastery)
                evaluation["mastery_updated"] = True
                evaluation["mastery_gain"] = mastery_gain
                evaluation["new_mastery"] = new_mastery
            
            # Check for achievements
            if reasoning_quality >= 0.8:
                # Create a temporary question ID for the achievement
                question_id = f"session_{session_id}_{int(time.time())}"
                
                # Add to learner's answered questions
                if "answered_questions" not in learner:
                    learner["answered_questions"] = []
                
                learner["answered_questions"].append({
                    "question_id": question_id,
                    "question_text": question,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "correct": True,
                    "reasoning_quality": reasoning_quality,
                    "response_text": response[:500]  # Limit length
                })
                
                # Award deep reasoning achievement
                success, points, message = self.achievement_system.award_deep_reasoning(
                    learner_id, question_id, reasoning_quality
                )
                
                if success:
                    evaluation["achievement"] = {
                        "type": "deep_reasoning",
                        "points": points,
                        "message": message
                    }
        
        # Prepare response
        result = {
            "session_id": session_id,
            "learner_id": learner_id,
            "concept_id": concept_id,
            "question": question,
            "response": response,
            "evaluation": {
                "correctness": evaluation["correctness"],
                "reasoning": evaluation["reasoning"],
                "conceptual": evaluation["conceptual"],
                "overall_score": evaluation["overall_score"],
                "is_correct": evaluation["is_correct"],
                "feedback": evaluation["feedback"],
                "strengths": evaluation["strengths"]
            },
            "followup_question": followup_question,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Add achievement if earned
        if "achievement" in evaluation:
            result["achievement"] = evaluation["achievement"]
        
        # Add mastery update if applicable
        if evaluation.get("mastery_updated", False):
            result["mastery_update"] = {
                "gain": evaluation["mastery_gain"],
                "new_level": evaluation["new_mastery"]
            }
        
        # Save changes
        self.knowledge_graph.save()
        
        return result
    
    def generate_weekly_plan(self, learner_id: str) -> Dict[str, Any]:
        """
        Generate a weekly study plan for a learner
        
        Args:
            learner_id: ID of the learner
            
        Returns:
            Weekly plan dictionary
        """
        return self.recommendation_system.generate_weekly_plan(learner_id)
    
    def generate_learning_path(self, learner_id: str, target_concept_id: str = None) -> List[Dict[str, Any]]:
        """
        Generate a learning path for a learner
        
        Args:
            learner_id: ID of the learner
            target_concept_id: ID of the target concept (optional)
            
        Returns:
            List of learning path steps
        """
        return self.recommendation_system.generate_learning_path(learner_id, target_concept_id)
    
    def add_concept_with_content(self, name: str, description: str, difficulty: float = 0.5,
                               complexity: float = 0.5, importance: float = 0.5,
                               prerequisites: List[str] = None) -> Dict[str, Any]:
        """
        Add a new concept with automatically generated questions and resources
        
        Args:
            name: Name of the concept
            description: Description of the concept
            difficulty: Difficulty level (0.0 to 1.0)
            complexity: Complexity level (0.0 to 1.0)
            importance: Importance level (0.0 to 1.0)
            prerequisites: List of prerequisite concept IDs
            
        Returns:
            Dictionary with concept ID and generated content
        """
        # Add the concept
        concept_id = self.knowledge_graph.add_concept(
            name=name,
            description=description,
            difficulty=difficulty,
            complexity=complexity,
            importance=importance,
            prerequisites=prerequisites
        )
        
        # Generate questions
        question_types = ["recall", "application", "analysis"]
        questions = []
        
        for q_type in question_types:
            # Adjust difficulty based on question type
            q_difficulty = difficulty
            if q_type == "recall":
                q_difficulty = max(0.1, difficulty - 0.2)
            elif q_type == "analysis":
                q_difficulty = min(0.9, difficulty + 0.2)
            
            # Generate and add question
            question_data = self.content_generator.generate_question(
                concept_id=concept_id,
                question_type=q_type,
                difficulty=q_difficulty
            )
            
            if question_data:
                question_id = self.knowledge_graph.add_question(
                    text=question_data["text"],
                    answer=question_data["answer"],
                    difficulty=q_difficulty,
                    question_type=q_type,
                    discriminator_power=0.5,
                    related_concepts=[concept_id]
                )
                
                questions.append({
                    "id": question_id,
                    "text": question_data["text"],
                    "type": q_type
                })
        
        # Generate resources
        media_types = ["text", "video"]
        resources = []
        
        for media_type in media_types:
            resource_data = self.content_generator.generate_resource(
                concept_id=concept_id,
                media_type=media_type
            )
            
            if resource_data:
                resource_id = self.knowledge_graph.add_resource(
                    title=resource_data["title"],
                    url=resource_data["url"],
                    description=resource_data["description"],
                    quality=0.8,
                    complexity=complexity,
                    media_type=media_type,
                    related_concepts=[concept_id]
                )
                
                resources.append({
                    "id": resource_id,
                    "title": resource_data["title"],
                    "media_type": media_type
                })
        
        # Generate learning objectives
        learning_objectives = self.content_generator.generate_learning_objectives(
            concept_id=concept_id,
            num_objectives=3
        )
        
        # Save all changes
        self.knowledge_graph.save()
        
        # Return the created content
        return {
            "concept_id": concept_id,
            "name": name,
            "description": description,
            "questions": questions,
            "resources": resources,
            "learning_objectives": learning_objectives
        }

# -----------------------------
# COMMAND LINE INTERFACE
# -----------------------------
def main():
    """Main entry point for the command line interface with enhanced user experience."""
    import json
    import os
    import argparse
    import logging
    from datetime import datetime

    parser = argparse.ArgumentParser(
        description="Enhanced GNN-Based Educational Achievement and Recommendation System"
    )
    # General configuration
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to configuration file (default: config.yaml)")
    parser.add_argument("--log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Set logging level (default is INFO)")

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # INIT command
    init_parser = subparsers.add_parser("init", help="Initialize the system")
    init_parser.add_argument("--example-data", action="store_true",
                             help="Add example data for testing and demo purposes")

    # LIST command
    list_parser = subparsers.add_parser("list", help="List entities")
    list_parser.add_argument("entity_type", choices=["learners", "concepts", "questions", "resources"],
                             help="Type of entity to list")
    list_parser.add_argument("--limit", type=int, default=10,
                             help="Maximum number of entities to show (default: 10)")

    # SEARCH command
    search_parser = subparsers.add_parser("search", help="Search for entities")
    search_parser.add_argument("entity_type", choices=["concepts", "questions", "resources"],
                             help="Type of entity to search")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--limit", type=int, default=5,
                             help="Maximum number of results to show (default: 5)")

    # ADD ENTITY commands
    add_concept_parser = subparsers.add_parser("add-concept", help="Add a new concept")
    add_concept_parser.add_argument("--name", type=str, required=True, help="Name of the concept")
    add_concept_parser.add_argument("--description", type=str, default="", help="Description of the concept")
    add_concept_parser.add_argument("--difficulty", type=float, default=0.5, help="Difficulty level (0.0 to 1.0)")
    add_concept_parser.add_argument("--complexity", type=float, default=0.5, help="Complexity level (0.0 to 1.0)")
    add_concept_parser.add_argument("--importance", type=float, default=0.5, help="Importance level (0.0 to 1.0)")
    add_concept_parser.add_argument("--prerequisites", type=str, nargs="*",
                                    help="List of prerequisite concept IDs")
    add_concept_parser.add_argument("--generate-content", action="store_true",
                                    help="Automatically generate questions and resources for the concept")

    add_learner_parser = subparsers.add_parser("add-learner", help="Add a new learner")
    add_learner_parser.add_argument("--name", type=str, required=True, help="Name of the learner")
    add_learner_parser.add_argument("--email", type=str, default="", help="Email of the learner")
    add_learner_parser.add_argument("--learning-style", type=str,
                                    choices=["visual", "balanced", "textual"],
                                    default="balanced", help="Learning style preference")
    add_learner_parser.add_argument("--persistence", type=float, default=0.5,
                                    help="Persistence level (0.0 to 1.0)")

    # LEARNER operations
    learner_parser = subparsers.add_parser("learner", help="Learner operations")
    learner_subparsers = learner_parser.add_subparsers(dest="learner_command", help="Learner subcommands")

    learner_stats_parser = learner_subparsers.add_parser("stats", help="Show learner statistics")
    learner_stats_parser.add_argument("learner_id", type=str, help="ID of the learner")

    learner_recommend_parser = learner_subparsers.add_parser("recommend", help="Get recommendations for a learner")
    learner_recommend_parser.add_argument("learner_id", type=str, help="ID of the learner")
    learner_recommend_parser.add_argument("--type", type=str,
                                          choices=["concepts", "resources", "questions"],
                                          default="concepts", help="Type of recommendations to show")
    learner_recommend_parser.add_argument("--concept-id", type=str,
                                          help="Concept ID (required for resource/question recommendations)")
    learner_recommend_parser.add_argument("--limit", type=int, default=3,
                                          help="Maximum number of recommendations (default: 3)")

    learner_path_parser = learner_subparsers.add_parser("path", help="Generate a learning path")
    learner_path_parser.add_argument("learner_id", type=str, help="ID of the learner")
    learner_path_parser.add_argument("--target", type=str, help="Target concept ID for the learning path")

    learner_plan_parser = learner_subparsers.add_parser("plan", help="Generate a weekly study plan")
    learner_plan_parser.add_argument("learner_id", type=str, help="ID of the learner")

    learner_session_parser = learner_subparsers.add_parser("session", help="Start interactive learning session")
    learner_session_parser.add_argument("learner_id", type=str, help="ID of the learner")
    learner_session_parser.add_argument("--concept-id", type=str, help="Concept ID for the session (optional)")

    learner_update_parser = learner_subparsers.add_parser("update", help="Update learner attributes")
    learner_update_parser.add_argument("learner_id", type=str, help="ID of the learner")
    learner_update_parser.add_argument("--name", type=str, help="New name")
    learner_update_parser.add_argument("--email", type=str, help="New email")
    learner_update_parser.add_argument("--learning-style", type=str,
                                       choices=["visual", "balanced", "textual"],
                                       help="New learning style")
    learner_update_parser.add_argument("--persistence", type=float, help="New persistence level")

    # CONCEPT operations
    concept_parser = subparsers.add_parser("concept", help="Concept operations")
    concept_subparsers = concept_parser.add_subparsers(dest="concept_command", help="Concept subcommands")

    concept_stats_parser = concept_subparsers.add_parser("stats", help="Show concept statistics")
    concept_stats_parser.add_argument("concept_id", type=str, help="ID of the concept")

    concept_update_parser = concept_subparsers.add_parser("update", help="Update concept attributes")
    concept_update_parser.add_argument("concept_id", type=str, help="ID of the concept")
    concept_update_parser.add_argument("--name", type=str, help="New name")
    concept_update_parser.add_argument("--description", type=str, help="New description")
    concept_update_parser.add_argument("--difficulty", type=float, help="New difficulty level")
    concept_update_parser.add_argument("--complexity", type=float, help="New complexity level")
    concept_update_parser.add_argument("--importance", type=float, help="New importance level")
    concept_update_parser.add_argument("--prerequisites", type=str, nargs="*",
                                       help="New list of prerequisite concept IDs")

    concept_generate_parser = concept_subparsers.add_parser("generate-content", help="Generate content for a concept")
    concept_generate_parser.add_argument("concept_id", type=str, help="ID of the concept")
    concept_generate_parser.add_argument("--questions", type=int, default=3,
                                         help="Number of questions to generate (default: 3)")
    concept_generate_parser.add_argument("--resources", type=int, default=2,
                                         help="Number of resources to generate (default: 2)")

    # MODEL operations
    model_parser = subparsers.add_parser("model", help="Model operations")
    model_subparsers = model_parser.add_subparsers(dest="model_command", help="Model subcommands")

    model_train_parser = model_subparsers.add_parser("train", help="Train the GNN model")
    model_train_parser.add_argument("--epochs", type=int, help="Number of training epochs (default: 100)")
    model_train_parser.add_argument("--force", action="store_true",
                                    help="Force retraining even if a saved model exists")

    model_save_parser = model_subparsers.add_parser("save", help="Save the trained model")
    model_save_parser.add_argument("--path", type=str, help="Path to save the model (default: data/model.pt)")

    model_load_parser = model_subparsers.add_parser("load", help="Load a saved model")
    model_load_parser.add_argument("--path", type=str, help="Path to load the model from (default: data/model.pt)")

    # ACHIEVEMENT operations
    achievement_parser = subparsers.add_parser("achievement", help="Achievement operations")
    achievement_subparsers = achievement_parser.add_subparsers(dest="achievement_command",
                                                               help="Achievement subcommands")
    achievement_check_parser = achievement_subparsers.add_parser("check", help="Check for new achievements")
    achievement_check_parser.add_argument("learner_id", type=str, help="ID of the learner")

    # IMPORT operations
    import_parser = subparsers.add_parser("import", help="Import concepts from a JSON file")
    import_parser.add_argument("json_file", type=str, help="Path to the JSON file containing concepts")
    import_parser.add_argument("--overwrite", action="store_true",
                               help="Overwrite existing concepts with the same ID")
    import_parser.add_argument("--generate-content", action="store_true",
                               help="Automatically generate questions and resources for imported concepts")

    # NEW: CACHE operations
    cache_parser = subparsers.add_parser("cache", help="Cache operations")
    cache_subparsers = cache_parser.add_subparsers(dest="cache_command", help="Cache subcommands")
    cache_clear_parser = cache_subparsers.add_parser("clear", help="Clear the node embedding cache")

    args = parser.parse_args()

    # Set logging level if provided
    if args.log_level:
        logging.getLogger().setLevel(args.log_level)

    # Initialize the educational system (which builds the knowledge graph, model, etc.)
    system = EducationalSystem(args.config)

    # Process commands with clearer feedback and formatting
    if args.command == "init":
        if args.example_data:
            success = system.add_example_data()
            if success:
                print("Example data added successfully.")
            else:
                print("Error adding example data.")
        else:
            print("System initialized successfully.")

    elif args.command == "import":
        if not os.path.exists(args.json_file):
            print(f"Error: JSON file '{args.json_file}' not found.")
            return 1
        print(f"Importing concepts from {args.json_file}...")
        added, skipped = system.import_concepts_from_file(args.json_file, args.overwrite)
        print(f"Import complete: {added} concepts added; {skipped} skipped.")
        if args.generate_content and added > 0:
            print("Generating content for imported concepts...")
            with open(args.json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            generated_count = 0
            for concept in data["concepts"]:
                concept_id = concept["id"]
                if concept_id in system.knowledge_graph._concept_lookup:
                    print(f"Generating content for: {concept.get('name', 'Unnamed')} (ID: {concept_id})")
                    try:
                        result = system.content_generator.generate_content_for_concept(concept_id)
                        if result:
                            generated_count += 1
                    except Exception as e:
                        print(f"Error generating content for {concept_id}: {e}")
            print(f"Content generation complete for {generated_count} concepts.")

    elif args.command == "list":
        if args.entity_type == "learners":
            learners = system.knowledge_graph.learners.get("learners", [])
            print(f"=== Learners ({len(learners)}) ===")
            for learner in learners[:args.limit]:
                print(f"ID: {learner['id']}, Name: {learner['name']}, "
                      f"Progress: {float(learner.get('overall_progress', 0.0))*100:.1f}%, "
                      f"Points: {learner.get('points', 0)}")
        elif args.entity_type == "concepts":
            concepts = system.knowledge_graph.concepts.get("concepts", [])
            print(f"=== Concepts ({len(concepts)}) ===")
            for concept in concepts[:args.limit]:
                prereqs = ", ".join(concept.get("prerequisites", []))
                print(f"ID: {concept['id']}, Name: {concept['name']}, "
                      f"Difficulty: {float(concept.get('difficulty', 0.5)):.1f}, "
                      f"Prerequisites: {prereqs or 'None'}")
        elif args.entity_type == "questions":
            questions = system.knowledge_graph.questions.get("questions", [])
            print(f"=== Questions ({len(questions)}) ===")
            for question in questions[:args.limit]:
                print(f"ID: {question['id']}, Type: {question.get('type', 'unknown')}, "
                      f"Text: {question.get('text', '')[:50]}...")
        elif args.entity_type == "resources":
            resources = system.knowledge_graph.resources.get("resources", [])
            print(f"=== Resources ({len(resources)}) ===")
            for resource in resources[:args.limit]:
                print(f"ID: {resource['id']}, Title: {resource.get('title', 'Untitled')}, "
                      f"Media: {resource.get('media_type', 'unknown')}")

    elif args.command == "search":
        if args.entity_type == "concepts":
            results = system.knowledge_graph.search_concepts(args.query, args.limit)
            print(f"Found {len(results)} concepts matching '{args.query}':")
            for concept in results:
                print(f"ID: {concept['id']}, Name: {concept['name']}")
                if concept.get("description"):
                    print(f"  Description: {concept['description'][:100]}...")
        elif args.entity_type == "resources":
            results = system.knowledge_graph.search_resources(args.query, args.limit)
            print(f"Found {len(results)} resources matching '{args.query}':")
            for resource in results:
                print(f"ID: {resource['id']}, Title: {resource.get('title', 'Untitled')}")
                print(f"  URL: {resource.get('url', 'N/A')}")

    elif args.command == "add-concept":
        if args.generate_content:
            result = system.add_concept_with_content(
                name=args.name,
                description=args.description,
                difficulty=args.difficulty,
                complexity=args.complexity,
                importance=args.importance,
                prerequisites=args.prerequisites
            )
            print(f"Concept added with ID: {result['concept_id']}")
            print(f"Generated {len(result['questions'])} questions and {len(result['resources'])} resources.")
            print("Learning objectives:")
            for i, obj in enumerate(result['learning_objectives']):
                print(f"  {i+1}. {obj}")
            # Clear the node embeddings cache after adding new content
            system.recommendation_system.clear_cache()
            print("Cache cleared to include new content.")
        else:
            concept_id = system.knowledge_graph.add_concept(
                name=args.name,
                description=args.description,
                difficulty=args.difficulty,
                complexity=args.complexity,
                importance=args.importance,
                prerequisites=args.prerequisites
            )
            print(f"Concept added with ID: {concept_id}")

    elif args.command == "add-learner":
        learner_id = system.knowledge_graph.add_learner(
            name=args.name,
            email=args.email,
            learning_style=args.learning_style,
            persistence=args.persistence
        )
        print(f"Learner added with ID: {learner_id}")

    elif args.command == "cache":
        if args.cache_command == "clear":
            system.recommendation_system.clear_cache()
            print("Node embedding cache cleared.")

    elif args.command == "learner":
        if args.learner_command == "stats":
            stats = system.get_learner_stats(args.learner_id)
            if "error" in stats:
                print(f"Error: {stats['error']}")
            else:
                print(f"=== Learner Statistics for {stats['name']} (ID: {stats['id']}) ===")
                print(f"Overall Progress: {stats['overall_progress']*100:.1f}%")
                print(f"Points: {stats['points']}")
                print(f"Learning Style: {stats['learning_style']}")
                print(f"Current Streak: {stats['streak']} days")
                print(f"Mastered Concepts: {stats['num_mastered']}")
                print(f"In-Progress Concepts: {stats['num_in_progress']}")
                if stats.get("question_stats"):
                    q = stats["question_stats"]
                    print(f"Questions: {q['total_answered']} answered, {q['correct_answers']} correct "
                          f"({q['accuracy']*100:.1f}%)")
                if stats.get("achievement_stats"):
                    a = stats["achievement_stats"]
                    print(f"Achievements: {a['total_achievements']} earned")
                    if a.get("recent_achievements"):
                        print("Recent Achievements:")
                        for ach in a["recent_achievements"]:
                            print(f"  - {ach.get('name', 'Unknown')} (+{ach.get('points', 0)} points)")
        elif args.learner_command == "recommend":
            if args.type == "concepts":
                recs = system.recommendation_system.recommend_next_concepts(args.learner_id, args.limit)
                print(f"--- Recommended Concepts for Learner {args.learner_id} ---")
                for i, c in enumerate(recs):
                    print(f"{i+1}. {c['name']} (ID: {c['id']}) | Difficulty: {c['difficulty']:.2f}, "
                          f"Importance: {c['importance']:.2f}")
            elif args.type == "resources":
                if not args.concept_id:
                    print("Error: Please provide --concept-id for resource recommendations.")
                else:
                    recs = system.recommendation_system.recommend_resources(args.learner_id, args.concept_id, args.limit)
                    print(f"--- Recommended Resources for Concept {args.concept_id} ---")
                    for i, r in enumerate(recs):
                        print(f"{i+1}. {r['title']} (ID: {r['id']})")
                        print(f"    URL: {r.get('url', 'N/A')}, Media: {r.get('media_type', 'unknown')}")
            elif args.type == "questions":
                recs = system.recommendation_system.recommend_questions(args.learner_id, args.concept_id, top_n=args.limit)
                concept_text = f"Concept {args.concept_id}" if args.concept_id else "General"
                print(f"--- Recommended Questions for {concept_text} ---")
                for i, q in enumerate(recs):
                    print(f"{i+1}. [{q['type']}] {q['text'][:60]}... (ID: {q['id']}, Difficulty: {q['difficulty']:.2f})")
        elif args.learner_command == "path":
            path = system.generate_learning_path(args.learner_id, args.target)
            if not path:
                print("No learning path generated. Either the target is already mastered or no suitable path exists.")
            else:
                print(f"--- Learning Path for Learner {args.learner_id} ---")
                for i, step in enumerate(path):
                    print(f"Step {i+1}: {step['name']} (ID: {step['concept_id']})")
                    print(f"  Current Mastery: {step['current_mastery']*100:.1f}%")
                    print(f"  Estimated Study Time: {step['estimated_study_time_minutes']} minutes")
                    if step.get('resources'):
                        print("  Recommended Resources:")
                        for j, res in enumerate(step['resources']):
                            print(f"    {j+1}. {res['title']} ({res['media_type']})")
                    if step.get('questions'):
                        print("  Practice Questions:")
                        for j, q in enumerate(step['questions']):
                            print(f"    {j+1}. {q['text'][:50]}...")
        elif args.learner_command == "plan":
            plan = system.generate_weekly_plan(args.learner_id)
            if not plan or "error" in plan:
                print("Unable to generate weekly plan.")
            else:
                print(f"--- Weekly Study Plan for {plan['learner_name']} ---")
                print(f"Total Study Time: {plan['total_study_time_minutes']} minutes "
                      f"({plan['daily_study_time_minutes']} minutes/day)")
                print(f"Period: {plan['start_date']} to {plan['end_date']}")
                for day in plan['schedule']:
                    print(f"\n{day['day']} ({day['study_minutes']} minutes):")
                    for act in day['activities']:
                        print(f"  - {act['concept_name']} ({act['minutes']} min)")
                        if act.get('resources'):
                            res = act['resources'][0]
                            print(f"    Resource: {res['title']} (URL: {res.get('url', 'N/A')})")
                        if act.get('questions'):
                            ques = act['questions'][0]
                            print(f"    Question: {ques['text'][:50]}...")
        elif args.learner_command == "session":
            session = system.interactive_session(args.learner_id, args.concept_id)
            if "error" in session:
                print(f"Error: {session['error']}")
            else:
                print(f"\n=== Interactive Session for {session['learner_name']} ===")
                print(f"Concept: {session['concept_name']}")
                print(f"Current Mastery: {session['current_mastery']*100:.1f}%")
                
                # Display learning objectives
                print("\nLearning Objectives:")
                for i, obj in enumerate(session.get('learning_objectives', [])):
                    print(f"  {i+1}. {obj}")
                
                # NEW: Retrieve and display recommended resources
                recommended_resources = system.recommendation_system.recommend_resources(
                    args.learner_id, args.concept_id, top_n=2)
                if recommended_resources:
                    print("\nRecommended Resources:")
                    for i, res in enumerate(recommended_resources):
                        print(f"  {i+1}. {res.get('title', 'Untitled')}")
                        print(f"     URL: {res.get('url', 'N/A')}")
                else:
                    print("\nNo resource recommendations available at this time.")
                
                print("\n=== Starting Interactive Q&A ===")
                current_question = session['initial_question']
                session_id = session['session_id']
                print("\n" + "="*80)
                print(f"Question: {current_question}")
                print("="*80)
                print("Type your answer below, or type 'quit' to end the session.")
                while True:
                    user_input = input("\nYour answer: ")
                    if user_input.lower() in ['quit', 'exit', 'q']:
                        print("Session ended. Your progress has been saved.")
                        break
                    if not user_input.strip():
                        print("Please enter an answer or type 'quit' to exit.")
                        continue
                    result = system.process_response(
                        session_id=session_id,
                        learner_id=args.learner_id,
                        concept_id=session['concept_id'],
                        question=current_question,
                        response=user_input
                    )
                    print("\n" + "-"*80)
                    print("Evaluation:")
                    print(f"  Correctness: {result['evaluation']['correctness']}/100")
                    print(f"  Reasoning: {result['evaluation']['reasoning']}/100")
                    print(f"  Overall Score: {result['evaluation']['overall_score']:.1f}/100")
                    print("\nStrengths:")
                    print(result['evaluation']['strengths'])
                    print("\nFeedback:")
                    print(result['evaluation']['feedback'])
                    if 'achievement' in result:
                        print("\n🏆 Achievement Unlocked! 🏆")
                        print(f"  +{result['achievement']['points']} points: {result['achievement']['message']}")
                    if 'mastery_update' in result:
                        print(f"\nMastery increased by {result['mastery_update']['gain']*100:.1f}%!")
                        print(f"New mastery level: {result['mastery_update']['new_level']*100:.1f}%")
                    current_question = result['followup_question']
                    print("\n" + "="*80)
                    print(f"Next Question: {current_question}")
                    print("="*80)
                    print("Type your answer below, or type 'quit' to exit.")

        elif args.learner_command == "update":
            updates = {}
            if args.name is not None:
                updates["name"] = args.name
            if args.email is not None:
                updates["email"] = args.email
            if args.learning_style is not None:
                updates["learning_style"] = args.learning_style
            if args.persistence is not None:
                updates["persistence"] = args.persistence
            if updates:
                if system.knowledge_graph.update_learner(args.learner_id, updates):
                    print(f"Learner {args.learner_id} updated successfully.")
                else:
                    print(f"Error updating learner {args.learner_id}.")
            else:
                print("No updates specified.")

    elif args.command == "concept":
        if args.concept_command == "stats":
            stats = system.get_concept_stats(args.concept_id)
            if "error" in stats:
                print(f"Error: {stats['error']}")
            else:
                print(f"=== Concept: {stats['name']} (ID: {stats['id']}) ===")
                print(f"Description: {stats['description']}")
                print(f"Difficulty: {stats['difficulty']:.2f}, Complexity: {stats['complexity']:.2f}")
                print(f"Importance: {stats['importance']:.2f}")
                if stats.get("prerequisites"):
                    prereqs = ", ".join([p['name'] for p in stats['prerequisites']])
                    print(f"Prerequisites: {prereqs}")
                if stats.get("dependent_concepts"):
                    dependents = ", ".join([c['name'] for c in stats['dependent_concepts']])
                    print(f"Dependent Concepts: {dependents}")
                print(f"Related Questions: {stats['num_questions']}")
                print(f"Related Resources: {stats['num_resources']}")
                if "average_mastery" in stats:
                    print(f"Average Mastery: {stats['average_mastery']*100:.1f}%")
                    print(f"Learners Studying: {stats['num_learners']}")
                    if stats.get("mastery_distribution"):
                        print("Mastery Distribution:")
                        for rng, cnt in stats["mastery_distribution"].items():
                            print(f"  {rng}: {cnt} learners")
        elif args.concept_command == "update":
            updates = {}
            if args.name is not None:
                updates["name"] = args.name
            if args.description is not None:
                updates["description"] = args.description
            if args.difficulty is not None:
                updates["difficulty"] = args.difficulty
            if args.complexity is not None:
                updates["complexity"] = args.complexity
            if args.importance is not None:
                updates["importance"] = args.importance
            if args.prerequisites is not None:
                updates["prerequisites"] = args.prerequisites
            if updates:
                if system.knowledge_graph.update_concept(args.concept_id, updates):
                    print(f"Concept {args.concept_id} updated successfully.")
                else:
                    print(f"Error updating concept {args.concept_id}.")
            else:
                print("No updates specified.")
        elif args.concept_command == "generate-content":
            concept = system.knowledge_graph.get_concept_by_id(args.concept_id)
            if not concept:
                print(f"Error: Concept with ID {args.concept_id} not found.")
                return 1
            print(f"Generating content for concept: {concept.get('name', 'Unnamed')}")
            questions_added = 0
            for i in range(args.questions):
                question_type = ["recall", "application", "analysis"][i % 3]
                question_data = system.content_generator.generate_question(
                    args.concept_id, question_type, concept.get("difficulty", 0.5)
                )
                if question_data:
                    qid = system.knowledge_graph.add_question(
                        text=question_data["text"],
                        answer=question_data["answer"],
                        difficulty=concept.get("difficulty", 0.5),
                        question_type=question_type,
                        related_concepts=[args.concept_id]
                    )
                    print(f"Added question: {question_data['text'][:50]}...")
                    questions_added += 1
            resources_added = 0
            for i in range(args.resources):
                media_type = ["text", "video"][i % 2]
                resource_data = system.content_generator.generate_resource(args.concept_id, media_type)
                if resource_data:
                    rid = system.knowledge_graph.add_resource(
                        title=resource_data["title"],
                        url=resource_data["url"],
                        description=resource_data.get("description", ""),
                        quality=0.8,
                        complexity=concept.get("complexity", 0.5),
                        media_type=media_type,
                        related_concepts=[args.concept_id]
                    )
                    print(f"Added resource: {resource_data['title']}")
                    resources_added += 1
            objectives = system.content_generator.generate_learning_objectives(args.concept_id, 3)
            if objectives:
                print("Generated Learning Objectives:")
                for i, obj in enumerate(objectives):
                    print(f"  {i+1}. {obj}")
            system.knowledge_graph.save()
            print(f"Content generation complete: {questions_added} questions and {resources_added} resources added.")

    elif args.command == "model":
        if args.model_command == "train":
            model_path = "data/model.pt"
            if os.path.exists(model_path) and not args.force:
                print("A saved model already exists. Use --force to retrain.")
                return 0
            epochs = args.epochs if args.epochs else 100
            history = system.train_model(epochs)
            if history:
                print(f"Model trained for {len(history.get('loss', []))} epochs.")
                print(f"Final training loss: {history.get('loss', [])[-1]:.4f}")
                if 'val_loss' in history:
                    print(f"Final validation loss: {history.get('val_loss', [])[-1]:.4f}")
            else:
                print("Model training skipped (insufficient data).")
        elif args.model_command == "save":
            path = args.path if args.path else "data/model.pt"
            system.save_model(path)
            print(f"Model saved to {path}.")
        elif args.model_command == "load":
            path = args.path if args.path else "data/model.pt"
            if system.load_model(path):
                print(f"Model loaded from {path}.")
            else:
                print(f"Failed to load model from {path}.")

    elif args.command == "achievement":
        if args.achievement_command == "check":
            results = system.achievement_system.check_for_achievements(args.learner_id)
            if not results:
                print("No new achievements earned.")
            else:
                print(f"Achievements earned by learner {args.learner_id}:")
                for success, points, msg in results:
                    if success:
                        print(f"  +{points} points: {msg}")
    else:
        parser.print_help()

    return 0

if __name__ == "__main__":
    exit(main())
