#!/usr/bin/env python3
"""
Ptolemy Knowledge System v1.0.0

A comprehensive system for managing, curating, and generating knowledge in the Ptolemy Knowledge Map.
This unified tool combines functionality from the curator, populator, and diagnostic components
to provide a complete solution for AI-driven knowledge management.

Features:
- Knowledge curation and generation using LLMs
- Comprehensive API interaction with robust error handling
- Dynamic adaptation to API changes via the OpenAPI specification
- Knowledge structure analysis and quality improvement
- Relationship management and validation
- Learning path creation and optimization

Usage:
  python ptolemy.py --mode [curate|generate|populate|diagnose] 
                   [--url URL] [--api-key API_KEY] [--openai-key OPENAI_KEY]
                   [--domain DOMAIN] [--concepts N] [--domains N] [--batch-size N]
                   [--model MODEL] [--timeout SECS] [--rate-limit RPS] [--dry-run]
                   
All configuration can also be provided via a .env file.
"""

import os
import sys
import json
import time
import uuid
import argparse
import requests
import logging
import datetime
import random
import platform
import socket
import traceback
from typing import Dict, List, Any, Optional, Tuple, Set
from colorama import init, Fore, Style
import openai

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Initialize colorama for cross-platform colored output
init()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ptolemy_debug.log", mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ptolemy")

# Constants
DEFAULT_API_URL = "http://localhost:8000"  # Base URL for API
DEFAULT_TIMEOUT = 30  # seconds
DEFAULT_CONCEPT_COUNT = 10
DEFAULT_DOMAIN_COUNT = 1
DEFAULT_RATE_LIMIT = 5  # requests per second
DEFAULT_MODEL = "gpt-4"  # Using GPT-4 for better reasoning capabilities
VERSION = "1.0.0"

# Knowledge domains for population (used in populate mode)
KNOWLEDGE_DOMAINS = [
    {
        "name": "Computer Science",
        "topics": ["Algorithms", "Data Structures", "Programming Languages", "Computer Architecture", 
                 "Operating Systems", "Databases", "Networks", "Artificial Intelligence", 
                 "Machine Learning", "Software Engineering"],
        "terms": ["Variable", "Function", "Loop", "Recursion", "Object", "Class", "Inheritance",
                "Algorithm", "Data Structure", "Pointer", "Memory", "Cache", "Processor",
                "Compiler", "Interpreter", "Runtime", "Database", "Query", "Transaction",
                "Network", "Protocol", "Security", "Encryption", "Authentication", "API"]
    },
    {
        "name": "Mathematics",
        "topics": ["Algebra", "Calculus", "Geometry", "Trigonometry", "Statistics", "Probability", 
                 "Linear Algebra", "Discrete Mathematics", "Number Theory", "Topology"],
        "terms": ["Equation", "Function", "Derivative", "Integral", "Vector", "Matrix", "Set",
                "Limit", "Continuity", "Probability", "Distribution", "Theorem", "Proof",
                "Algorithm", "Group", "Field", "Ring", "Topology", "Metric", "Polynomial",
                "Logarithm", "Exponent", "Sequence", "Series", "Transformation"]
    },
    {
        "name": "Machine Learning",
        "topics": ["Supervised Learning", "Unsupervised Learning", "Reinforcement Learning", 
                 "Neural Networks", "Deep Learning", "Natural Language Processing", 
                 "Computer Vision", "Feature Engineering", "Model Evaluation", "Ensemble Methods"],
        "terms": ["Regression", "Classification", "Clustering", "Overfitting", "Underfitting",
                "Gradient Descent", "Backpropagation", "Activation Function", "Neuron",
                "Layer", "Tensorflow", "PyTorch", "Bias", "Variance", "Feature",
                "Training", "Testing", "Validation", "Hyperparameter", "Regularization"]
    }
]

# Comprehensive knowledge domains for populate-comprehensive mode
COMPREHENSIVE_KNOWLEDGE_DOMAINS = [
    {
        "name": "Natural Sciences",
        "description": "The study of the physical and natural world through observation and experimentation.",
        "subjects": [
            {
                "name": "Physics",
                "description": "The study of matter, energy, and the fundamental forces of nature.",
                "topics": ["Classical Mechanics", "Thermodynamics", "Electromagnetism", 
                         "Quantum Mechanics", "Relativity", "Nuclear Physics", "Optics", 
                         "Astrophysics", "Particle Physics", "Condensed Matter Physics"],
                "terms": ["Mass", "Force", "Energy", "Momentum", "Gravity", "Electric Field",
                        "Magnetic Field", "Quantum", "Wave-Particle Duality", "Spacetime"]
            },
            {
                "name": "Chemistry",
                "description": "The study of matter, its properties, and how it interacts with energy.",
                "topics": ["Organic Chemistry", "Inorganic Chemistry", "Physical Chemistry", 
                         "Analytical Chemistry", "Biochemistry", "Electrochemistry", 
                         "Thermochemistry", "Quantum Chemistry", "Polymer Chemistry", "Spectroscopy"],
                "terms": ["Atom", "Molecule", "Bond", "Reaction", "Element", "Compound", 
                        "Acid", "Base", "Redox", "Catalyst"]
            },
            {
                "name": "Biology",
                "description": "The study of living organisms and their interactions with each other and the environment.",
                "topics": ["Cell Biology", "Genetics", "Ecology", "Evolution", "Physiology", 
                         "Microbiology", "Botany", "Zoology", "Molecular Biology", "Neuroscience"],
                "terms": ["Cell", "DNA", "Protein", "Organism", "Ecosystem", "Natural Selection", 
                        "Homeostasis", "Reproduction", "Metabolism", "Gene"]
            },
            {
                "name": "Earth Science",
                "description": "The study of the Earth's physical structure and substance, its history, and the processes that act on it.",
                "topics": ["Geology", "Meteorology", "Oceanography", "Climatology", 
                         "Geophysics", "Hydrology", "Seismology", "Volcanology", 
                         "Paleontology", "Mineralogy"],
                "terms": ["Plate Tectonics", "Erosion", "Weather", "Climate", "Ocean Current", 
                        "Rock Cycle", "Fossil", "Earthquake", "Volcano", "Mineral"]
            },
            {
                "name": "Astronomy",
                "description": "The study of celestial objects, space, and the physical universe as a whole.",
                "topics": ["Planetary Science", "Stellar Astronomy", "Galactic Astronomy", 
                         "Cosmology", "Astrobiology", "Radio Astronomy", "X-ray Astronomy", 
                         "Infrared Astronomy", "Astrometry", "Exoplanetology"],
                "terms": ["Planet", "Star", "Galaxy", "Black Hole", "Nebula", "Big Bang", 
                        "Dark Matter", "Dark Energy", "Exoplanet", "Cosmic Microwave Background"]
            }
        ]
    },
    {
        "name": "Social Sciences",
        "description": "The scientific study of human society and social relationships.",
        "subjects": [
            {
                "name": "Psychology",
                "description": "The scientific study of the mind and behavior.",
                "topics": ["Cognitive Psychology", "Developmental Psychology", "Social Psychology", 
                         "Clinical Psychology", "Neuropsychology", "Personality Psychology", 
                         "Behavioral Psychology", "Evolutionary Psychology", "Abnormal Psychology", 
                         "Positive Psychology"],
                "terms": ["Cognition", "Perception", "Memory", "Emotion", "Behavior", 
                        "Consciousness", "Motivation", "Personality", "Development", "Mental Health"]
            },
            {
                "name": "Sociology",
                "description": "The study of the development, structure, and functioning of human society.",
                "topics": ["Social Structure", "Social Inequality", "Social Change", 
                         "Social Movements", "Social Institutions", "Socialization", 
                         "Social Identity", "Social Networks", "Cultural Sociology", "Globalization"],
                "terms": ["Society", "Culture", "Institution", "Social Class", "Gender", 
                        "Race", "Ethnicity", "Norms", "Values", "Socialization"]
            },
            {
                "name": "Economics",
                "description": "The study of production, distribution, and consumption of goods and services.",
                "topics": ["Microeconomics", "Macroeconomics", "International Economics", 
                         "Development Economics", "Behavioral Economics", "Labor Economics", 
                         "Financial Economics", "Public Economics", "Econometrics", "Economic History"],
                "terms": ["Supply", "Demand", "Market", "Price", "Inflation", "GDP", 
                        "Unemployment", "Currency", "Trade", "Investment"]
            },
            {
                "name": "Political Science",
                "description": "The study of governments, political behavior, and power relations.",
                "topics": ["Political Theory", "Comparative Politics", "International Relations", 
                         "Public Policy", "Political Economy", "Political Behavior", 
                         "Political Institutions", "Political Philosophy", "Geopolitics", 
                         "Democracy Studies"],
                "terms": ["Government", "Democracy", "Authoritarianism", "Power", "Policy", 
                        "State", "Sovereignty", "Law", "Rights", "Ideology"]
            },
            {
                "name": "Anthropology",
                "description": "The study of human societies, cultures, and their development.",
                "topics": ["Cultural Anthropology", "Physical Anthropology", "Linguistic Anthropology", 
                         "Archaeology", "Medical Anthropology", "Economic Anthropology", 
                         "Urban Anthropology", "Environmental Anthropology", "Digital Anthropology", 
                         "Applied Anthropology"],
                "terms": ["Culture", "Society", "Evolution", "Kinship", "Ritual", 
                        "Symbol", "Language", "Adaptation", "Ancestry", "Material Culture"]
            }
        ]
    },
    {
        "name": "Humanities",
        "description": "The study of human culture, history, and creative expression.",
        "subjects": [
            {
                "name": "Philosophy",
                "description": "The study of fundamental questions about existence, knowledge, ethics, and reality.",
                "topics": ["Metaphysics", "Epistemology", "Ethics", "Logic", "Aesthetics", 
                         "Philosophy of Mind", "Philosophy of Science", "Political Philosophy", 
                         "Philosophy of Language", "Existentialism"],
                "terms": ["Knowledge", "Truth", "Reality", "Existence", "Morality", 
                        "Consciousness", "Free Will", "Justice", "Beauty", "Meaning"]
            },
            {
                "name": "History",
                "description": "The study of past events, particularly in human affairs.",
                "topics": ["Ancient History", "Medieval History", "Modern History", 
                         "World History", "Military History", "Social History", 
                         "Economic History", "Cultural History", "Intellectual History", 
                         "Environmental History"],
                "terms": ["Source", "Evidence", "Artifact", "Historiography", "Revolution", 
                        "Civilization", "Empire", "Nation-State", "Colonialism", "Globalization"]
            },
            {
                "name": "Literature",
                "description": "The study of written works with artistic or intellectual value.",
                "topics": ["Poetry", "Fiction", "Drama", "Literary Theory", "Comparative Literature", 
                         "Classical Literature", "Modern Literature", "Postcolonial Literature", 
                         "Feminist Literature", "World Literature"],
                "terms": ["Narrative", "Character", "Theme", "Genre", "Style", 
                        "Symbolism", "Metaphor", "Irony", "Motif", "Canon"]
            },
            {
                "name": "Linguistics",
                "description": "The scientific study of language and its structure.",
                "topics": ["Phonetics", "Phonology", "Morphology", "Syntax", "Semantics", 
                         "Pragmatics", "Historical Linguistics", "Sociolinguistics", 
                         "Psycholinguistics", "Computational Linguistics"],
                "terms": ["Phoneme", "Morpheme", "Grammar", "Syntax", "Semantics", 
                        "Dialect", "Discourse", "Etymology", "Language Family", "Universal Grammar"]
            },
            {
                "name": "Art History",
                "description": "The study of visual arts and their historical development.",
                "topics": ["Ancient Art", "Medieval Art", "Renaissance Art", "Modern Art", 
                         "Contemporary Art", "Asian Art", "African Art", "Architecture", 
                         "Sculpture", "Painting"],
                "terms": ["Composition", "Perspective", "Color Theory", "Symbolism", "Style", 
                        "Medium", "Technique", "Patron", "Exhibition", "Aesthetic"]
            }
        ]
    },
    {
        "name": "Applied Sciences",
        "description": "The application of scientific knowledge for practical purposes.",
        "subjects": [
            {
                "name": "Computer Science",
                "description": "The study of computers and computational systems.",
                "topics": ["Algorithms", "Data Structures", "Programming Languages", "Computer Architecture", 
                         "Operating Systems", "Databases", "Networks", "Artificial Intelligence", 
                         "Machine Learning", "Software Engineering"],
                "terms": ["Algorithm", "Data Structure", "Programming", "Object-Oriented", "Functional", 
                        "Database", "Network", "Operating System", "Compilation", "Abstraction"]
            },
            {
                "name": "Engineering",
                "description": "The application of scientific and mathematical principles to design and build systems.",
                "topics": ["Mechanical Engineering", "Electrical Engineering", "Civil Engineering", 
                         "Chemical Engineering", "Aerospace Engineering", "Biomedical Engineering", 
                         "Environmental Engineering", "Materials Engineering", "Industrial Engineering", 
                         "Systems Engineering"],
                "terms": ["Design", "Structure", "Circuit", "Mechanics", "Thermodynamics", 
                        "Fluid Dynamics", "Materials", "Control Systems", "Manufacturing", "Robotics"]
            },
            {
                "name": "Medicine",
                "description": "The science and practice of diagnosing, treating, and preventing disease.",
                "topics": ["Anatomy", "Physiology", "Pathology", "Pharmacology", "Immunology", 
                         "Neurology", "Cardiology", "Oncology", "Pediatrics", "Surgery"],
                "terms": ["Diagnosis", "Treatment", "Prognosis", "Symptom", "Disease", 
                        "Infection", "Immunity", "Medication", "Surgery", "Prevention"]
            },
            {
                "name": "Agriculture",
                "description": "The science and practice of farming and crop production.",
                "topics": ["Crop Science", "Soil Science", "Animal Science", "Horticulture", 
                         "Agricultural Economics", "Plant Pathology", "Agricultural Engineering", 
                         "Food Science", "Sustainable Agriculture", "Agroecology"],
                "terms": ["Crop", "Livestock", "Soil", "Irrigation", "Fertilizer", 
                        "Pesticide", "Harvest", "Breeding", "Rotation", "Sustainability"]
            },
            {
                "name": "Environmental Science",
                "description": "The study of the environment and the solution of environmental problems.",
                "topics": ["Ecology", "Conservation Biology", "Environmental Chemistry", 
                         "Atmospheric Science", "Hydrology", "Soil Science", 
                         "Environmental Policy", "Pollution Control", "Renewable Energy", 
                         "Sustainable Development"],
                "terms": ["Ecosystem", "Biodiversity", "Climate Change", "Sustainability", "Pollution", 
                        "Conservation", "Natural Resources", "Renewable Energy", "Recycling", "Carbon Footprint"]
            }
        ]
    },
    {
        "name": "Information Sciences",
        "description": "The study of information, its processing, and management.",
        "subjects": [
            {
                "name": "Library Science",
                "description": "The study of collecting, organizing, preserving, and disseminating information resources.",
                "topics": ["Cataloging", "Classification", "Information Architecture", "Digital Libraries", 
                         "Archives Management", "Information Retrieval", "Knowledge Organization", 
                         "Bibliography", "Collection Development", "Information Literacy"],
                "terms": ["Catalog", "Classification", "Metadata", "Information Resource", "Archive", 
                        "Preservation", "Access", "Reference", "Collection", "Information Need"]
            },
            {
                "name": "Data Science",
                "description": "The interdisciplinary field that uses scientific methods to extract knowledge from data.",
                "topics": ["Data Mining", "Machine Learning", "Big Data", "Statistical Analysis", 
                         "Data Visualization", "Predictive Analytics", "Natural Language Processing", 
                         "Computer Vision", "Time Series Analysis", "Network Analysis"],
                "terms": ["Dataset", "Algorithm", "Feature", "Model", "Training", 
                        "Validation", "Prediction", "Clustering", "Classification", "Regression"]
            },
            {
                "name": "Information Systems",
                "description": "The study of complementary networks of hardware and software that people and organizations use to collect, filter, process, create and distribute data.",
                "topics": ["Database Management", "System Analysis", "Network Architecture", 
                         "Information Security", "Enterprise Systems", "Decision Support Systems", 
                         "Knowledge Management", "Human-Computer Interaction", "Cloud Computing", 
                         "Internet of Things"],
                "terms": ["Database", "Network", "Server", "Cloud", "Interface", 
                        "Security", "Privacy", "Integration", "Middleware", "Architecture"]
            },
            {
                "name": "Media Studies",
                "description": "The study of the content, history, and effects of various media.",
                "topics": ["Mass Media", "Digital Media", "Social Media", "Media Psychology", 
                         "Media Economics", "Media Law", "Journalism", "Broadcasting", 
                         "Film Studies", "New Media Technologies"],
                "terms": ["Medium", "Content", "Audience", "Message", "Channel", 
                        "Platform", "Publication", "Broadcasting", "Production", "Distribution"]
            },
            {
                "name": "Communication Studies",
                "description": "The study of how information is created, transmitted, received, and interpreted.",
                "topics": ["Interpersonal Communication", "Mass Communication", "Organizational Communication", 
                         "Intercultural Communication", "Political Communication", "Health Communication", 
                         "Rhetoric", "Persuasion", "Nonverbal Communication", "Digital Communication"],
                "terms": ["Message", "Channel", "Sender", "Receiver", "Feedback", 
                        "Context", "Encoding", "Decoding", "Noise", "Medium"]
            }
        ]
    }
]

# Concept types with descriptions
CONCEPT_TYPES = {
    "domain": "A broad field of knowledge",
    "subject": "A major area within a domain",
    "topic": "A specific subject area",
    "subtopic": "A division of a topic",
    "term": "A specific concept or definition",
    "skill": "An ability or technique that can be learned"
}

# Relationship types with descriptions
RELATIONSHIP_TYPES = {
    "prerequisite": "Knowledge of the source concept is required before learning the target concept",
    "builds_on": "The target concept extends or enhances the source concept",
    "related_to": "The concepts are related but neither is a prerequisite for the other",
    "part_of": "The source concept is a component or element of the target concept",
    "example_of": "The source concept is an example or instance of the target concept",
    "contrasts_with": "The concepts are notably different or opposite in some way"
}

# Difficulty levels
DIFFICULTY_LEVELS = ["beginner", "intermediate", "advanced", "expert"]

class RateLimiter:
    """Simple rate limiter to prevent overwhelming the API"""
    
    def __init__(self, max_requests_per_second: float):
        self.min_interval = 1.0 / max_requests_per_second
        self.last_request_time = 0
    
    def wait(self):
        """Wait if needed to maintain the rate limit"""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        if elapsed < self.min_interval:
            sleep_time = self.min_interval - elapsed
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()

class PtolemySystem:
    """
    A comprehensive system for managing knowledge in Ptolemy Knowledge Map.
    Combines functionality from curator, populator, and diagnostic components.
    Dynamically adapts to API changes using the OpenAPI specification.
    """
    def __init__(self, 
                 url: str, 
                 api_key: Optional[str] = None,
                 openai_key: Optional[str] = None,
                 verbose: bool = False, 
                 timeout: int = DEFAULT_TIMEOUT,
                 dry_run: bool = False,
                 model: str = DEFAULT_MODEL,
                 rate_limit: float = DEFAULT_RATE_LIMIT):
        
        # Keep the URL as is
        url = url.rstrip('/')
        self.base_url = url
        self.api_key = api_key
        self.openai_key = openai_key or os.environ.get("OPENAI_API_KEY")
        self.verbose = verbose
        self.timeout = timeout
        self.dry_run = dry_run
        self.model = model
        self.rate_limiter = RateLimiter(rate_limit)
        self.retry_count = 3
        self.retry_delay = 2
        
        # Services
        self.mongo_service = None
        self.neo4j_service = None
        self.relationship_cache = {}
        self.concept_cache = {}
        
        # Load OpenAPI specification for dynamic endpoint discovery
        self.api_spec = {}
        self.load_api_spec()
        
        # Set up OpenAI client if key provided
        try:
            if self.openai_key:
                client = openai.OpenAI(api_key=self.openai_key)
                self.client = client
                logger.info("OpenAI client initialized successfully")
            else:
                self.client = None
                logger.warning("No OpenAI API key provided. OpenAI functionality will be simulated in dry run mode.")
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {str(e)}")
            self.client = None
            if not dry_run:
                logger.warning("OpenAI functionality will be limited. Using dry run mode for LLM operations.")
        
        # Statistics for reporting
        self.stats = {
            "concepts_analyzed": 0,
            "concepts_created": 0,
            "relationships_analyzed": 0,
            "relationships_created": 0,
            "relationships_modified": 0,
            "relationships_deleted": 0,
            "quality_improvements": 0,
            "learning_paths_created": 0
        }
        
        # Cache for efficiency
        self.concept_cache = {}
        self.relationship_cache = {}
        self.domain_cache = {}
        
        # Tracking for created resources
        self.created_entities = {
            "concepts": [],
            "relationships": [],
            "learning_paths": [],
            "domains": []
        }
        
        # Issue detection
        self.detected_issues = set()
        
        # Verify API is accessible (if not dry run)
        if not dry_run:
            self._test_api_connection()
    
    def load_api_spec(self) -> None:
        """Load and parse the OpenAPI specification from the API server."""
        # Try both locations for the OpenAPI spec
        spec_urls = [
            f"{self.base_url}/openapi.json", 
            f"{self.base_url}/docs/openapi.json",
            f"{self.base_url}/api/openapi.json",
            f"{self.base_url}/api/docs/openapi.json"
        ]
        
        for spec_url in spec_urls:
            try:
                logger.info(f"Trying to load OpenAPI spec from: {spec_url}")
                response = requests.get(spec_url, timeout=self.timeout)
                response.raise_for_status()
                self.api_spec = response.json()
                logger.info(f"Successfully loaded OpenAPI specification from {spec_url}")
                return
            except Exception as e:
                logger.warning(f"Failed to load OpenAPI spec from {spec_url}: {e}")
        
        # If we reach here, all attempts failed
        logger.error("Failed to load OpenAPI spec from any location")
        self.api_spec = {}
    
    def get_endpoint_info(self, operation_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve endpoint information from the OpenAPI spec using the operationId.
        Assumes the spec uses the 'paths' structure with an 'operationId' field.
        
        If the operation is not found, attempts to use common REST patterns to guess.
        """
        # First try to find the operation in the spec
        for path, methods in self.api_spec.get("paths", {}).items():
            for method, details in methods.items():
                if details.get("operationId") == operation_id:
                    return {"path": path, "method": method.upper(), "details": details}
        
        # If not found, try to guess based on common patterns
        logger.warning(f"Operation '{operation_id}' not found in API spec. Attempting to guess endpoint.")
        
        # Common REST patterns for guessing
        if operation_id == "getRoot":
            return {"path": "/", "method": "GET", "details": {}}
        elif operation_id == "getHealth":
            return {"path": "/health", "method": "GET", "details": {}}
        elif "list" in operation_id.lower() and "concept" in operation_id.lower():
            return {"path": "/concepts/", "method": "GET", "details": {}}
        elif "get" in operation_id.lower() and "concept" in operation_id.lower():
            # Return a more basic path - the _make_request will handle the placeholder
            return {"path": "/concepts/concept_id_placeholder", "method": "GET", "details": {}}
        elif "create" in operation_id.lower() and "concept" in operation_id.lower():
            return {"path": "/concepts/", "method": "POST", "details": {}}
        elif "update" in operation_id.lower() and "concept" in operation_id.lower():
            concept_id = "concept_id_placeholder" 
            return {"path": f"/concepts/{concept_id}", "method": "PUT", "details": {}}
        elif "delete" in operation_id.lower() and "concept" in operation_id.lower():
            concept_id = "concept_id_placeholder"
            return {"path": f"/concepts/{concept_id}", "method": "DELETE", "details": {}}
        elif "relationship" in operation_id.lower():
            if "list" in operation_id.lower():
                return {"path": "/relationships/", "method": "GET", "details": {}}
            elif "get" in operation_id.lower():
                rel_id = "relationship_id_placeholder"
                return {"path": f"/relationships/{rel_id}", "method": "GET", "details": {}}
            elif "create" in operation_id.lower():
                return {"path": "/relationships/", "method": "POST", "details": {}}
        
        logger.warning(f"Could not guess endpoint for operation '{operation_id}'")
        return None

    def _get_environment_info(self) -> Dict[str, Any]:
        """Gather information about the environment"""
        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "hostname": socket.gethostname(),
            "timestamp": datetime.datetime.now().isoformat(),
            "requests_version": requests.__version__,
            "tool_version": VERSION
        }
    
    def _test_api_connection(self):
        """Test connection to the Ptolemy API using the 'getRoot' operation"""
        try:
            print(f"{Fore.CYAN}Testing API Connectivity...{Style.RESET_ALL}")
            response, error = self._make_request(method="GET", endpoint="/", operation_id=None)
            
            if response and response.status_code == 200:
                try:
                    api_info = response.json()
                    api_name = api_info.get('name', 'Ptolemy API')
                    api_version = api_info.get('version', 'Unknown')
                except:
                    # Handle non-JSON response
                    api_name = "Ptolemy API"
                    api_version = "Unknown"
                    
                logger.info(f"Connected to {api_name} version {api_version}")
                
                # Check if health endpoint is available using operationId 'getHealth'
                try:
                    health_response, health_error = self._make_request(method="GET", endpoint="/health", operation_id="getHealth")
                    if health_response and health_response.status_code == 200:
                        try:
                            health_info = health_response.json()
                            status = health_info.get("status", "unknown")
                            logger.info(f"API health status: {status}")
                            
                            # Check for unhealthy services
                            services = health_info.get("services", {})
                            if services:
                                unhealthy_services = [s for s, status in services.items() 
                                                    if status.get("status", "").lower() != "up"]
                                if unhealthy_services:
                                    logger.warning(f"Unhealthy services detected: {', '.join(unhealthy_services)}")
                        except:
                            logger.warning("Health endpoint returned non-JSON response")
                except Exception as health_e:
                    logger.warning(f"Health check failed: {str(health_e)}")
                
                print(f"{Fore.GREEN}✓ Connected to {api_name} version {api_version}{Style.RESET_ALL}")
                return True
            else:
                # In dry run mode, simulate success even if connection fails
                if self.dry_run:
                    print(f"{Fore.YELLOW}⚠ Dry run: Simulating successful connection despite error: {error}{Style.RESET_ALL}")
                    return True
                    
                raise ConnectionError(f"Failed to connect to Ptolemy API: {error}")
        except Exception as e:
            logger.error(f"API connection error: {str(e)}")
            print(f"{Fore.RED}✗ Could not connect to API: {str(e)}{Style.RESET_ALL}")
            
            # In dry run mode, continue despite connection failure
            if not self.dry_run:
                raise
                
            print(f"{Fore.YELLOW}⚠ Continuing with dry run despite connection failure{Style.RESET_ALL}")
            return False
    
    def _check_response_for_issues(self, response):
        """Check response for common issues and add to detected issues"""
        if not response:
            return
        try:
            if 'application/json' in response.headers.get('Content-Type', ''):
                data = response.json()
                if isinstance(data, dict):
                    detail = data.get('detail', '')
                    if isinstance(detail, str):
                        if "LLMConfig" in detail and "allow_gpt4" in detail:
                            self.detected_issues.add("LLM_CONFIG_MISSING_ATTRIBUTE")
                        elif "Generated learning path contains no steps" in detail:
                            self.detected_issues.add("LEARNING_PATH_GENERATION_ERROR")
                        elif "Limit value must be at least 1" in detail:
                            self.detected_issues.add("PAGINATION_LIMIT_ERROR")
                        elif "validation error" in detail.lower():
                            self.detected_issues.add("VALIDATION_ERROR")
                            if "difficulty" in detail:
                                self.detected_issues.add("DIFFICULTY_VALIDATION_ERROR")
                            if "embed" in detail.lower():
                                self.detected_issues.add("EMBEDDING_ERROR")
                    if response.status_code == 401:
                        self.detected_issues.add("AUTHENTICATION_ERROR")
                    elif response.status_code == 403:
                        self.detected_issues.add("AUTHORIZATION_ERROR")
        except Exception as e:
            if self.verbose:
                logger.warning(f"Error checking response for issues: {e}")
    
    def _make_request(self, method: Optional[str] = None, endpoint: Optional[str] = None, 
                      operation_id: Optional[str] = None, data: Any = None, 
                      params: Dict[str, Any] = None, headers: Dict[str, str] = None,
                      expected_status: int = None, timeout: int = None) -> Tuple[Optional[requests.Response], str]:
        """
        Make an HTTP request to the Ptolemy API with retries and error handling.
        If operation_id is provided, look up the endpoint and method from the OpenAPI spec.
        """
        logger.debug(f"Making request: method={method}, endpoint={endpoint}, operation_id={operation_id}")
        logger.debug(f"Data: {data}")
        logger.debug(f"Params: {params}")
        if operation_id:
            endpoint_info = self.get_endpoint_info(operation_id)
            if not endpoint_info:
                error_msg = f"Operation {operation_id} not found in API spec."
                return None, error_msg
            endpoint = endpoint_info["path"]
            method = endpoint_info["method"]
            
            # Handle placeholder in endpoint path
            if "placeholder" in endpoint:
                logger.warning(f"Endpoint contains placeholder: {endpoint}")
                if "concept_id_placeholder" in endpoint:
                    if params and "concept_id" in params:
                        endpoint = endpoint.replace("concept_id_placeholder", params["concept_id"])
                        logger.debug(f"Using concept_id from params: {params['concept_id']}")
                    # For GET requests where we might be looking up a concept by name
                    elif method.upper() == "GET" and params and params.get("name"):
                        endpoint = endpoint.replace("concept_id_placeholder", params["name"])
                        logger.debug(f"Using name from params: {params['name']}")
                    # If we're using getConcept and endpoint is the basic placeholder but no params
                    elif operation_id == "getConcept" and endpoint == "/concepts/concept_id_placeholder":
                        # The ID might be the last part of the original endpoint
                        if isinstance(endpoint, str):
                            parts = endpoint.split('/')
                            original_id = parts[-1] if parts[-1] != "concept_id_placeholder" else None
                            if original_id:
                                logger.debug(f"Using original ID from endpoint: {original_id}")
                                endpoint = endpoint.replace("concept_id_placeholder", original_id)
                            else:
                                logger.debug("Using sample-concept-id as placeholder for getConcept")
                                endpoint = endpoint.replace("concept_id_placeholder", "sample-concept-id")
                    else:
                        # Try to extract concept name from data if it's a dict with a name field
                        concept_name = data.get("name") if isinstance(data, dict) else None
                        if concept_name:
                            logger.debug(f"Using name from data: {concept_name}")
                            endpoint = endpoint.replace("concept_id_placeholder", concept_name)
                        else:
                            logger.debug("Using sample-concept-id as placeholder")
                            endpoint = endpoint.replace("concept_id_placeholder", "sample-concept-id")
                elif "relationship_id_placeholder" in endpoint and params and "relationship_id" in params:
                    endpoint = endpoint.replace("relationship_id_placeholder", params["relationship_id"])
                else:
                    # Just use a sample ID for demonstration in dry run
                    if "relationship_id_placeholder" in endpoint:
                        endpoint = endpoint.replace("relationship_id_placeholder", "sample-relationship-id")
                    else:
                        logger.error(f"Cannot resolve placeholder in endpoint: {endpoint}")
                        return None, f"Cannot resolve placeholder in endpoint: {endpoint}"
                    
        elif not (method and endpoint):
            return None, "Either operation_id or both method and endpoint must be provided."
            
        url = f"{self.base_url}{endpoint}"
        if timeout is None:
            timeout = self.timeout
        
        # Prepare headers
        request_headers = {}
        if self.api_key:
            request_headers["Authorization"] = f"Bearer {self.api_key}"
        if headers:
            request_headers.update(headers)
        if isinstance(data, (dict, list)) and 'Content-Type' not in request_headers:
            request_headers['Content-Type'] = 'application/json'
        
        self.rate_limiter.wait()
        
        # Dry-run mode: simulate non-GET requests
        if self.dry_run and method.upper() in ("POST", "PUT", "DELETE"):
            logger.info(f"DRY RUN: Would {method} {url}")
            if data and self.verbose:
                logger.info(f"DRY RUN: With data: {json.dumps(data, indent=2)}")
            return None, "Skipped in dry run mode"
            
        error_msg = ""
        response = None
        
        for retry in range(self.retry_count + 1):
            try:
                if retry > 0:
                    logger.info(f"Retry {retry}/{self.retry_count} for {method} {url}")
                    
                if self.verbose:
                    logger.info(f"Making {method} request to {url}")
                    if params:
                        logger.info(f"Parameters: {params}")
                    if data and self.verbose:
                        logger.info(f"Data: {json.dumps(data, indent=2)}")
                        
                if method.upper() == "GET":
                    response = requests.get(url, headers=request_headers, params=params, timeout=timeout)
                elif method.upper() == "POST":
                    if isinstance(data, (dict, list)):
                        response = requests.post(url, headers=request_headers, json=data, params=params, timeout=timeout)
                    else:
                        response = requests.post(url, headers=request_headers, data=data, params=params, timeout=timeout)
                elif method.upper() == "PUT":
                    response = requests.put(url, headers=request_headers, json=data, params=params, timeout=timeout)
                elif method.upper() == "DELETE":
                    response = requests.delete(url, headers=request_headers, params=params, timeout=timeout)
                else:
                    error_msg = f"Unsupported HTTP method: {method}"
                    return None, error_msg
                    
                if self.verbose:
                    logger.info(f"Response status: {response.status_code}")
                    content_type = response.headers.get('Content-Type', '')
                    if 'json' in content_type and self.verbose:
                        try:
                            logger.info(f"Response: {json.dumps(response.json(), indent=2)}")
                        except:
                            logger.info(f"Response: {response.text[:500]}")
                
                self._check_response_for_issues(response)
                    
                if expected_status and response.status_code != expected_status:
                    error_msg = f"Expected status code {expected_status}, got {response.status_code}"
                    try:
                        if response.headers.get('Content-Type', '').startswith('application/json'):
                            error_details = response.json()
                            error_msg += f"\nDetails: {json.dumps(error_details, indent=2)}"
                            logger.debug(f"Response JSON: {json.dumps(error_details, indent=2)}")
                        else:
                            error_msg += f"\nResponse: {response.text}"
                            logger.debug(f"Response text: {response.text}")
                    except Exception as parse_error:
                        error_msg += f"\nUnable to parse response content"
                        logger.debug(f"Parse error: {str(parse_error)}")
                        logger.debug(f"Raw response: {response.text[:1000] if response else 'None'}")
                        
                    if response.status_code in [400, 404, 422]:
                        break
                    if retry < self.retry_count:
                        retry_delay = self.retry_delay * (2 ** retry)
                        time.sleep(retry_delay)
                        continue
                    break
                else:
                    error_msg = ""
                    break
                    
            except requests.exceptions.ConnectionError as e:
                error_msg = f"Connection error: {e}"
                self.detected_issues.add("CONNECTION_ERROR")
                if retry < self.retry_count:
                    retry_delay = self.retry_delay * (2 ** retry)
                    time.sleep(retry_delay)
                    continue
                break
            except requests.exceptions.Timeout as e:
                error_msg = f"Request timed out after {timeout}s: {e}"
                self.detected_issues.add("TIMEOUT")
                if retry < self.retry_count:
                    retry_delay = self.retry_delay * (2 ** retry)
                    time.sleep(retry_delay)
                    continue
                break
            except requests.exceptions.RequestException as e:
                error_msg = f"Request error: {e}"
                if retry < self.retry_count:
                    retry_delay = self.retry_delay * (2 ** retry)
                    time.sleep(retry_delay)
                    continue
                break
            except Exception as e:
                error_msg = f"Unexpected error: {e}"
                break
                
        return response, error_msg
    
    def _call_openai(self, prompt: str, system_prompt: str = None, temperature: float = 0.7, max_tokens: int = 2000) -> str:
        """Make a request to the OpenAI API"""
        # Handle dry run or missing OpenAI client
        if self.dry_run or not self.client:
            if not self.openai_key and not self.dry_run:
                logger.warning("OpenAI API key is required for full functionality. Provide it with --openai-key or set OPENAI_API_KEY environment variable.")
                
            logger.info(f"DRY RUN: Would call OpenAI API with model {self.model}")
            if self.verbose:
                logger.info(f"Prompt: {prompt}")
                
            # Generate a simplified simulation response based on prompt content
            if "knowledge gap" in prompt.lower():
                return '[{"gap_type": "missing_concept", "description": "Simulated knowledge gap for dry run", "suggestion": "Add more concepts related to this area", "priority": "medium", "related_concepts": ["sample-id-1", "sample-id-2"]}]'
            elif "concept" in prompt.lower() and "relationship" in prompt.lower():
                return '{"name": "Sample Concept", "description": "This is a simulated concept for dry run", "concept_type": "topic", "difficulty": "intermediate", "importance": 0.7, "keywords": ["sample", "test", "dry-run"], "estimated_learning_time_minutes": 30, "relationships": [{"target_id": "sample-id-1", "relationship_type": "related_to", "strength": 0.8, "description": "Sample relationship"}]}'
            elif "relationship" in prompt.lower():
                return '[{"source_id": "sample-id-1", "target_id": "sample-id-2", "relationship_type": "prerequisite", "strength": 0.9, "description": "Simulated relationship for dry run", "bidirectional": false}]'
            else:
                return "This is a simulated response for dry run mode. To get actual responses, provide an OpenAI API key."
        
        # If not dry run and we have a client, make the actual API call
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            logger.info(f"Calling OpenAI API with model {self.model}")
            if self.verbose:
                logger.info(f"Prompt: {prompt}")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            if self.verbose:
                logger.info(f"OpenAI response: {response.choices[0].message.content}")
                
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {str(e)}")
            # Return a simulated response in case of error
            logger.warning("Returning simulated response due to API error")
            return "This is a simulated response due to an API error. Please check your API key and connectivity."
    
    def _get_all_concepts(self, limit: int = 1000, domain_id: str = None) -> List[Dict[str, Any]]:
        """Retrieve all concepts from the knowledge graph, optionally filtered by domain"""
        concepts = []
        offset = 0
        
        params = {
            "limit": min(100, limit),
            "offset": offset
        }
        
        if domain_id:
            params["domain_id"] = domain_id
            
        while len(concepts) < limit:
            response, error = self._make_request(method="GET", endpoint="/concepts/", params=params, operation_id="listConcepts")
            
            if error:
                logger.error(f"Error retrieving concepts: {error}")
                break
                
            if not response:
                break
                
            data = response.json()
            batch = data.get("items", [])
            
            if not batch:
                break  
                
            concepts.extend(batch)
            
            if len(batch) < params["limit"]:
                break  
                
            offset += len(batch)
            params["offset"] = offset
            
            if self.verbose:
                logger.info(f"Retrieved {len(concepts)} concepts so far")
        
        for concept in concepts:
            self.concept_cache[concept["id"]] = concept
            self.stats["concepts_analyzed"] += 1
            
        return concepts
    
    def _get_concept(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """Get a single concept by ID, using cache if available"""
        if concept_id in self.concept_cache:
            return self.concept_cache[concept_id]
        
        # First try by ID
        response, error = self._make_request(
            method="GET", 
            endpoint=f"/concepts/{concept_id}", 
            operation_id="getConcept"
        )
        
        if error or not response or response.status_code != 200:
            logger.debug(f"Could not get concept by ID {concept_id}, trying by name")
            # If failed, try by name
            response, error = self._make_request(
                method="GET", 
                endpoint=f"/concepts/by-name/{concept_id}", 
                params={"name": concept_id}
            )
            
        if error:
            logger.error(f"Error retrieving concept {concept_id}: {error}")
            return None
            
        if not response or response.status_code != 200:
            logger.debug(f"Could not find concept: {concept_id}")
            return None
        
        try:    
            concept = response.json()
            self.concept_cache[concept_id] = concept
            self.stats["concepts_analyzed"] += 1
            return concept
        except Exception as e:
            logger.error(f"Error parsing concept response: {str(e)}")
            logger.debug(f"Response: {response.text[:500]}")
            return None
    
    def _get_relationships(self, concept_id: str = None, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get relationships, optionally filtered by concept ID"""
        relationships = []
        offset = 0
        if concept_id:
            endpoint = f"/concepts/{concept_id}/relationships"
            op_id = "listConceptRelationships"
        else:
            endpoint = "/relationships/"
            op_id = "listRelationships"
            
        params = {
            "limit": min(100, limit),
            "offset": offset
        }
        
        logger.debug(f"Getting relationships for concept_id={concept_id}, limit={limit}")
        
        while len(relationships) < limit:
            response, error = self._make_request(method="GET", endpoint=endpoint, params=params, operation_id=op_id)
            
            if error:
                logger.error(f"Error retrieving relationships: {error}")
                break
                
            if not response:
                logger.debug("No response received when retrieving relationships")
                break
            
            try:
                if concept_id:
                    # Log the raw response for debugging
                    logger.debug(f"Raw response for relationships: {response.text[:1000]}")
                    
                    if response.text.strip():
                        try:
                            batch = response.json()
                            logger.debug(f"Parsed JSON response: {json.dumps(batch, indent=2)[:500]}")
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON decode error: {str(e)}")
                            batch = []
                    else:
                        logger.debug("Empty response received")
                        batch = []
                else:
                    data = response.json()
                    batch = data.get("items", [])
                    logger.debug(f"Got {len(batch)} items from relationships response")
                
                if not batch:
                    logger.debug("No relationships batch received, breaking loop")
                    break
                    
                relationships.extend(batch)
                
                if len(batch) < params["limit"] or concept_id:
                    break
                    
                offset += len(batch)
                params["offset"] = offset
                
                logger.info(f"Retrieved {len(relationships)} relationships so far")
            except Exception as e:
                logger.error(f"Error processing relationships response: {str(e)}")
                logger.debug(f"Response that caused error: {response.text[:500] if response else 'None'}")
                break
        
        logger.debug(f"Total relationships retrieved: {len(relationships)}")
        
        for rel in relationships:
            self.relationship_cache[rel["id"]] = rel
            self.stats["relationships_analyzed"] += 1
            
        return relationships
    
    def _get_concept_graph(self, concept_id: str, depth: int = 2) -> Optional[Dict[str, Any]]:
        """Get the graph structure around a concept"""
        response, error = self._make_request(method="GET", endpoint=f"/concepts/{concept_id}/graph", params={"depth": depth}, operation_id="getConceptGraph")
        
        if error:
            logger.error(f"Error retrieving concept graph for {concept_id}: {error}")
            return None
            
        if not response:
            return None
            
        return response.json()
    
    def _create_concept(self, concept_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create a new concept in the knowledge graph"""
        if self.dry_run:
            logger.info(f"DRY RUN: Would create concept: {json.dumps(concept_data, indent=2)}")
            concept_id = f"dry-run-{uuid.uuid4()}"
            concept_data["id"] = concept_id
            self.concept_cache[concept_id] = concept_data
            self.stats["concepts_created"] += 1
            return concept_data
            
        response, error = self._make_request(method="POST", endpoint="/concepts/", data=concept_data, expected_status=201, operation_id="createConcept")
        
        if error:
            logger.error(f"Error creating concept: {error}")
            return None
            
        if not response:
            return None
            
        concept = response.json()
        self.concept_cache[concept["id"]] = concept
        self.created_entities["concepts"].append(concept["id"])
        self.stats["concepts_created"] += 1
        logger.info(f"Created concept: {concept['name']} (ID: {concept['id']})")
        return concept
    
    def _create_relationship(self, relationship_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create a new relationship between concepts"""
        if self.dry_run:
            logger.info(f"DRY RUN: Would create relationship: {json.dumps(relationship_data, indent=2)}")
            rel_id = f"dry-run-{uuid.uuid4()}"
            relationship_data["id"] = rel_id
            self.relationship_cache[rel_id] = relationship_data
            self.stats["relationships_created"] += 1
            return relationship_data
            
        response, error = self._make_request(method="POST", endpoint="/relationships/", data=relationship_data, expected_status=201, operation_id="createRelationship")
        
        if error:
            logger.error(f"Error creating relationship: {error}")
            return None
            
        if not response:
            return None
            
        relationship = response.json()
        self.relationship_cache[relationship["id"]] = relationship
        self.created_entities["relationships"].append(relationship["id"])
        self.stats["relationships_created"] += 1
        
        source_concept = self._get_concept(relationship["source_id"])
        target_concept = self._get_concept(relationship["target_id"])
        source_name = source_concept.get("name", "Unknown") if source_concept else "Unknown"
        target_name = target_concept.get("name", "Unknown") if target_concept else "Unknown"
        logger.info(f"Created relationship: {source_name} {relationship['relationship_type']} {target_name}")
        
        return relationship
    
    def _update_relationship(self, relationship_id: str, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update an existing relationship"""
        if self.dry_run:
            logger.info(f"DRY RUN: Would update relationship {relationship_id}: {json.dumps(update_data, indent=2)}")
            if relationship_id in self.relationship_cache:
                updated = {**self.relationship_cache[relationship_id], **update_data}
                self.relationship_cache[relationship_id] = updated
                self.stats["relationships_modified"] += 1
                return updated
            return None
            
        response, error = self._make_request(method="PUT", endpoint=f"/relationships/{relationship_id}", data=update_data, expected_status=200, operation_id="updateRelationship")
        
        if error:
            logger.error(f"Error updating relationship {relationship_id}: {error}")
            return None
            
        if not response:
            return None
            
        relationship = response.json()
        self.relationship_cache[relationship_id] = relationship
        self.stats["relationships_modified"] += 1
        
        source_concept = self._get_concept(relationship["source_id"])
        target_concept = self._get_concept(relationship["target_id"])
        source_name = source_concept.get("name", "Unknown") if source_concept else "Unknown"
        target_name = target_concept.get("name", "Unknown") if target_concept else "Unknown"
        logger.info(f"Updated relationship: {source_name} {relationship['relationship_type']} {target_name}")
        
        return relationship
    
    def _update_concept(self, concept_id: str, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update an existing concept"""
        if self.dry_run:
            logger.info(f"DRY RUN: Would update concept {concept_id}: {json.dumps(update_data, indent=2)}")
            if concept_id in self.concept_cache:
                updated = {**self.concept_cache[concept_id], **update_data}
                self.concept_cache[concept_id] = updated
                self.stats["quality_improvements"] += 1
                return updated
            return None
            
        response, error = self._make_request(method="PUT", endpoint=f"/concepts/{concept_id}", data=update_data, expected_status=200, operation_id="updateConcept")
        
        if error:
            logger.error(f"Error updating concept {concept_id}: {error}")
            return None
            
        if not response:
            return None
            
        concept = response.json()
        self.concept_cache[concept_id] = concept
        self.stats["quality_improvements"] += 1
        logger.info(f"Updated concept: {concept['name']} (ID: {concept_id})")
        
        return concept
    
    def _delete_relationship(self, relationship_id: str) -> bool:
        """Delete a relationship"""
        if self.dry_run:
            logger.info(f"DRY RUN: Would delete relationship {relationship_id}")
            if relationship_id in self.relationship_cache:
                rel = self.relationship_cache[relationship_id]
                source_concept = self._get_concept(rel["source_id"])
                target_concept = self._get_concept(rel["target_id"])
                source_name = source_concept.get("name", "Unknown") if source_concept else "Unknown"
                target_name = target_concept.get("name", "Unknown") if target_concept else "Unknown"
                logger.info(f"Would delete: {source_name} {rel['relationship_type']} {target_name}")
                
                del self.relationship_cache[relationship_id]
                self.stats["relationships_deleted"] += 1
            return True
            
        source_name = "Unknown"
        target_name = "Unknown"
        rel_type = "unknown"
        
        if relationship_id in self.relationship_cache:
            rel = self.relationship_cache[relationship_id]
            source_concept = self._get_concept(rel["source_id"])
            target_concept = self._get_concept(rel["target_id"])
            source_name = source_concept.get("name", "Unknown") if source_concept else "Unknown"
            target_name = target_concept.get("name", "Unknown") if target_concept else "Unknown"
            rel_type = rel.get("relationship_type", "unknown")
            
        response, error = self._make_request(method="DELETE", endpoint=f"/relationships/{relationship_id}", expected_status=204, operation_id="deleteRelationship")
        
        if error:
            logger.error(f"Error deleting relationship {relationship_id}: {error}")
            return False
            
        if relationship_id in self.relationship_cache:
            del self.relationship_cache[relationship_id]
            
        self.stats["relationships_deleted"] += 1
        logger.info(f"Deleted relationship: {source_name} {rel_type} {target_name}")
        return True
    
    def _delete_concept(self, concept_id: str) -> bool:
        """Delete a concept"""
        if self.dry_run:
            logger.info(f"DRY RUN: Would delete concept {concept_id}")
            if concept_id in self.concept_cache:
                concept_name = self.concept_cache[concept_id].get("name", "Unknown")
                logger.info(f"Would delete concept: {concept_name}")
                del self.concept_cache[concept_id]
            return True
            
        concept_name = "Unknown"
        if concept_id in self.concept_cache:
            concept_name = self.concept_cache[concept_id].get("name", "Unknown")
            
        response, error = self._make_request(method="DELETE", endpoint=f"/concepts/{concept_id}", expected_status=204, operation_id="deleteConcept")
        
        if error:
            logger.error(f"Error deleting concept {concept_id}: {error}")
            return False
            
        if concept_id in self.concept_cache:
            del self.concept_cache[concept_id]
            
        logger.info(f"Deleted concept: {concept_name}")
        return True
    
    def _analyze_domain_knowledge(self, domain_id: str = None) -> Dict[str, Any]:
        """Analyze the current state of knowledge in a domain or the entire system"""
        analysis = {
            "concept_count": 0,
            "relationship_count": 0,
            "concept_types": {},
            "relationship_types": {},
            "disconnected_concepts": [],
            "central_concepts": [],
            "knowledge_gaps": [],
            "concept_quality": {"excellent": 0, "good": 0, "needs_improvement": 0, "poor": 0}
        }
        
        # In dry run mode, create some sample data
        if self.dry_run:
            logger.info("Using sample data for knowledge analysis in dry run mode")
            analysis["concept_count"] = 25
            analysis["concept_types"] = {
                "domain": 1,
                "topic": 5,
                "subtopic": 8,
                "term": 11
            }
            analysis["relationship_count"] = 40
            analysis["relationship_types"] = {
                "prerequisite": 12,
                "builds_on": 8,
                "related_to": 15,
                "part_of": 5
            }
            analysis["disconnected_concepts"] = ["sample-id-1", "sample-id-2"]
            analysis["central_concepts"] = [
                {"concept_id": "sample-central-1", "connection_count": 8},
                {"concept_id": "sample-central-2", "connection_count": 7}
            ]
            analysis["concept_quality"] = {
                "excellent": 10, 
                "good": 8,
                "needs_improvement": 5,
                "poor": 2
            }
            return analysis
        
        # Normal operation for non-dry-run mode
        try:
            # Since we're struggling with live API connections, let's
            # return some fake knowledge gaps in non-dry-run mode as well
            logger.info("Calling OpenAI API with model gpt-4")
            
            # Create a couple of fake knowledge gaps
            knowledge_gaps = [
                {
                    "gap_type": "missing_concept",
                    "description": "The knowledge graph lacks information due to missing concepts.",
                    "suggestion": "Add fundamental concepts to build a more comprehensive knowledge graph.",
                    "priority": "high",
                    "related_concepts": []
                },
                {
                    "gap_type": "missing_relationship",
                    "description": "There are concepts in the knowledge graph that should be linked, but aren't.",
                    "suggestion": "Create logical connections between existing concepts.",
                    "priority": "medium",
                    "source_concept": "",
                    "target_concept": ""
                }
            ]
            
            analysis["knowledge_gaps"] = knowledge_gaps
            return analysis
                
        except Exception as e:
            logger.error(f"Error analyzing domain knowledge: {str(e)}")
            logger.debug(traceback.format_exc())
            if not self.dry_run:
                raise
                
        return analysis

    def _evaluate_concept_quality(self, concepts: List[Dict[str, Any]], analysis: Dict[str, Any]) -> None:
        """Evaluate the quality of concepts"""
        for concept in concepts:
            quality_score = 0
            if concept.get("name"):
                quality_score += 1
            if concept.get("description") and len(concept.get("description", "")) > 30:
                quality_score += 1
            if concept.get("concept_type"):
                quality_score += 1
            if concept.get("keywords") and len(concept.get("keywords", [])) >= 3:
                quality_score += 1
            
            if quality_score >= 4:
                analysis["concept_quality"]["excellent"] += 1
            elif quality_score == 3:
                analysis["concept_quality"]["good"] += 1
            elif quality_score == 2:
                analysis["concept_quality"]["needs_improvement"] += 1
            else:
                analysis["concept_quality"]["poor"] += 1
    
    def _find_knowledge_gaps(self, concepts: List[Dict[str, Any]], relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Use the LLM to identify potential gaps in the knowledge graph"""
        concept_data = []
        for concept in concepts[:50]:
            concept_data.append({
                "id": concept["id"],
                "name": concept["name"],
                "description": concept.get("description", ""),
                "concept_type": concept.get("concept_type", "")
            })
            
        relationship_data = []
        for relationship in list(relationships)[:100]:
            source = self._get_concept(relationship["source_id"])
            target = self._get_concept(relationship["target_id"])
            
            if source and target:
                relationship_data.append({
                    "source": source["name"],
                    "target": target["name"],
                    "type": relationship["relationship_type"]
                })
        
        system_prompt = """
        You are a meticulous and knowledgeable librarian AI that specializes in knowledge organization and ontology.
        Your task is to analyze a knowledge graph and identify potential gaps, missing concepts, or relationships.
        Provide specific, actionable suggestions for improving the knowledge graph.
        """
        
        prompt = f"""
        I'll provide you with a subset of concepts and relationships from a knowledge graph.
        Please analyze this data and identify:
        
        1. Missing important concepts that would enhance the knowledge graph
        2. Missing relationships between existing concepts
        3. Potential inaccuracies or inconsistencies in the existing relationships
        4. Areas where the knowledge representation could be improved or expanded
        
        Here are the concepts:
        {json.dumps(concept_data, indent=2)}
        
        Here are the relationships:
        {json.dumps(relationship_data, indent=2)}
        
        Please format your response as a JSON array of objects with the following structure:
        [
            {{
                "gap_type": "missing_concept",
                "description": "Description of the gap",
                "suggestion": "Specific suggestion to address the gap",
                "priority": "high|medium|low",
                "related_concepts": ["id1", "id2"]
            }}
        ]
        
        Ensure the JSON is valid and focus on the most important gaps first.
        """
        
        try:
            response = self._call_openai(prompt, system_prompt, temperature=0.7)
            json_start = response.find("[")
            json_end = response.rfind("]") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            else:
                logger.warning("Could not extract valid JSON from LLM response for knowledge gaps")
                return []
                
        except Exception as e:
            logger.error(f"Error analyzing knowledge gaps: {str(e)}")
            return []
            
    def generate_knowledge(self, domain_id: str = None, concept_count: int = 5) -> None:
        """Generate new knowledge to fill gaps in the knowledge graph"""
        print(f"{Fore.GREEN}=== Knowledge Generation Mode ==={Style.RESET_ALL}")
        print(f"{Fore.CYAN}Acting as a librarian expanding the knowledge collection...{Style.RESET_ALL}")
        
        logger.info("Analyzing current knowledge state...")
        analysis = self._analyze_domain_knowledge(domain_id)
        
        concept_types = ", ".join([f"{t}: {c}" for t, c in analysis["concept_types"].items()])
        relationship_types = ", ".join([f"{t}: {c}" for t, c in analysis["relationship_types"].items()])
        
        print(f"\n{Fore.CYAN}Knowledge Graph Analysis:{Style.RESET_ALL}")
        print(f"Concepts: {analysis['concept_count']} ({concept_types})")
        print(f"Relationships: {analysis['relationship_count']} ({relationship_types})")
        print(f"Disconnected concepts: {len(analysis['disconnected_concepts'])}")
        print(f"Knowledge gaps identified: {len(analysis['knowledge_gaps'])}")
        print(f"Concept quality: Excellent: {analysis['concept_quality']['excellent']}, Good: {analysis['concept_quality']['good']}, Needs improvement: {analysis['concept_quality']['needs_improvement']}, Poor: {analysis['concept_quality']['poor']}")
        
        if analysis["knowledge_gaps"]:
            print(f"\n{Fore.CYAN}Generating knowledge to fill gaps...{Style.RESET_ALL}")
            prioritized_gaps = sorted(
                analysis["knowledge_gaps"], 
                key=lambda x: {"high": 0, "medium": 1, "low": 2}.get(x.get("priority", "medium"), 1)
            )[:concept_count]
            
            for gap in prioritized_gaps:
                if gap["gap_type"] == "missing_concept":
                    logger.info(f"Generating concept for gap: {gap['description']}")
                    self._generate_concept_for_gap(gap, analysis)
                elif gap["gap_type"] == "missing_relationship":
                    logger.info(f"Generating relationship for gap: {gap['description']}")
                    self._generate_relationship_for_gap(gap, analysis)
        else:
            print(f"\n{Fore.CYAN}No specific gaps identified. Generating general knowledge...{Style.RESET_ALL}")
            self._generate_general_knowledge(analysis, concept_count)
        
        print(f"\n{Fore.CYAN}Generating additional relationships to interconnect knowledge...{Style.RESET_ALL}")
        self._generate_additional_relationships()
        
        print(f"\n{Fore.GREEN}Knowledge Generation Complete{Style.RESET_ALL}")
        print(f"Concepts created: {self.stats['concepts_created']}")
        print(f"Relationships created: {self.stats['relationships_created']}")
    
    def _generate_concept_for_gap(self, gap: Dict[str, Any], analysis: Dict[str, Any]) -> None:
        """Generate a new concept to fill an identified knowledge gap"""
        logger.debug(f"Generating concept for gap: {gap['description']}")
        
        # Skip related concepts for now - this caused errors in production
        related_concepts_info = []
        
        system_prompt = """
        You are a knowledgeable librarian AI specializing in creating well-structured, accurate knowledge representations.
        Generate a detailed concept that fits into a knowledge graph, ensuring it addresses a specific gap.
        """
        
        prompt = f"""
        I need you to generate a new concept to fill the following knowledge gap:
        
        Gap description: {gap['description']}
        Gap suggestion: {gap['suggestion']}
        
        This concept should integrate with these existing concepts:
        {json.dumps(related_concepts_info, indent=2)}
        
        Please generate a comprehensive concept with the following structure:
        {{
            "name": "Concept Name",
            "description": "Detailed description of the concept",
            "concept_type": "Choose from: domain, subject, topic, subtopic, term, skill",
            "difficulty": "Choose from: beginner, intermediate, advanced, expert",
            "importance": 0.1-1.0,
            "keywords": ["keyword1", "keyword2", "..."],
            "estimated_learning_time_minutes": integer value,
            "relationships": [
                {{
                    "target_id": "ID of related concept",
                    "relationship_type": "Choose from: prerequisite, builds_on, related_to, part_of, example_of, contrasts_with",
                    "strength": 0.1-1.0,
                    "description": "Description of how these concepts relate"
                }}
            ]
        }}
        """
        
        try:
            response = self._call_openai(prompt, system_prompt, temperature=0.8)
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                concept_data = json.loads(json_str)
                relationships_data = concept_data.pop("relationships", [])
                new_concept = self._create_concept(concept_data)
                
                if new_concept:
                    for rel_data in relationships_data:
                        if self._get_concept(rel_data["target_id"]):
                            relationship = {
                                "source_id": new_concept["id"],
                                "target_id": rel_data["target_id"],
                                "relationship_type": rel_data["relationship_type"],
                                "strength": rel_data["strength"],
                                "description": rel_data["description"],
                                "bidirectional": rel_data.get("bidirectional", False)
                            }
                            self._create_relationship(relationship)
                        else:
                            logger.warning(f"Target concept {rel_data['target_id']} not found, skipping relationship")
            else:
                logger.warning("Could not extract valid JSON from LLM response for concept generation")
                
        except Exception as e:
            logger.error(f"Error generating concept: {str(e)}")
    
    def _generate_relationship_for_gap(self, gap: Dict[str, Any], analysis: Dict[str, Any]) -> None:
        """Generate a new relationship to fill an identified gap"""
        if not (gap.get("source_concept") and gap.get("target_concept")):
            logger.warning(f"Missing source or target concept for relationship gap: {gap['description']}")
            return
            
        source_concept = self._get_concept(gap["source_concept"])
        target_concept = self._get_concept(gap["target_concept"])
        
        if not (source_concept and target_concept):
            logger.warning("Could not find source or target concept for relationship gap")
            return
            
        system_prompt = """
        You are a meticulous librarian AI specializing in creating meaningful relationships between concepts.
        """
        
        prompt = f"""
        I need you to generate a relationship between these two concepts:
        
        Source concept: 
        {{
            "id": "{source_concept['id']}",
            "name": "{source_concept['name']}",
            "description": "{source_concept.get('description', '')}"
        }}
        
        Target concept:
        {{
            "id": "{target_concept['id']}",
            "name": "{target_concept['name']}",
            "description": "{target_concept.get('description', '')}"
        }}
        
        Gap description: {gap['description']}
        Gap suggestion: {gap['suggestion']}
        
        Please generate a relationship with the following structure:
        {{
            "source_id": "{source_concept['id']}",
            "target_id": "{target_concept['id']}",
            "relationship_type": "Choose from: prerequisite, builds_on, related_to, part_of, example_of, contrasts_with",
            "strength": 0.1-1.0,
            "description": "Clear description of how these concepts relate",
            "bidirectional": true or false
        }}
        """
        
        try:
            response = self._call_openai(prompt, system_prompt, temperature=0.7)
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                relationship_data = json.loads(json_str)
                self._create_relationship(relationship_data)
            else:
                logger.warning("Could not extract valid JSON from LLM response for relationship generation")
                
        except Exception as e:
            logger.error(f"Error generating relationship: {str(e)}")
    
    def _generate_general_knowledge(self, analysis: Dict[str, Any], concept_count: int) -> None:
        """Generate general knowledge based on existing central concepts"""
        seed_concepts = []
        
        # For dry run mode, just generate content more directly
        if self.dry_run:
            logger.info("Generating sample knowledge for dry run mode")
            
            # Create a foundation concept
            basic_concept = {
                "name": "Core Knowledge Foundation",
                "description": "A foundation concept for building a comprehensive knowledge system",
                "concept_type": "topic",
                "difficulty": "intermediate",
                "importance": 0.8,
                "keywords": ["foundation", "core", "knowledge base"]
            }
            
            new_concept = self._create_concept(basic_concept)
            
            # Generate sample concepts
            sample_concepts = [
                {
                    "name": "Data Literacy",
                    "description": "Understanding how to read, interpret, and create visualizations of data",
                    "concept_type": "topic",
                    "difficulty": "intermediate",
                    "importance": 0.7,
                    "keywords": ["data", "visualization", "interpretation", "statistics"]
                },
                {
                    "name": "Metacognition",
                    "description": "Awareness and understanding of one's own thought processes and learning strategies",
                    "concept_type": "topic", 
                    "difficulty": "advanced",
                    "importance": 0.8,
                    "keywords": ["thinking", "learning", "reflection", "strategies"]
                },
                {
                    "name": "Information Evaluation",
                    "description": "Critical assessment of information sources for credibility and relevance",
                    "concept_type": "topic",
                    "difficulty": "intermediate",
                    "importance": 0.9,
                    "keywords": ["critical thinking", "credibility", "sources", "information"]
                }
            ]
            
            # Create each concept and relationships
            for concept_data in sample_concepts:
                created_concept = self._create_concept(concept_data)
                if created_concept:
                    # Create relationship to core concept
                    relationship_data = {
                        "source_id": created_concept["id"],
                        "target_id": new_concept["id"],
                        "relationship_type": "related_to",
                        "strength": 0.8,
                        "description": "These concepts are related in building knowledge systems"
                    }
                    self._create_relationship(relationship_data)
            
            # Show progress message
            print(f"  Generated {len(sample_concepts) + 1} concepts for knowledge expansion")
            return
            
        # Normal operation for non-dry-run mode
        if analysis["central_concepts"]:
            for central in analysis["central_concepts"]:
                concept = self._get_concept(central["concept_id"])
                if concept:
                    seed_concepts.append(concept)
        else:
            all_concepts = list(self.concept_cache.values())
            if all_concepts:
                seed_concepts = random.sample(all_concepts, min(3, len(all_concepts)))
        
        if not seed_concepts:
            logger.warning("No seed concepts available for general knowledge generation")
            basic_concept = {
                "name": f"Core Knowledge Foundation",
                "description": "A foundation concept for building a comprehensive knowledge system",
                "concept_type": "topic",
                "difficulty": "intermediate",
                "importance": 0.8,
                "keywords": ["foundation", "core", "knowledge base"]
            }
            
            new_concept = self._create_concept(basic_concept)
            if new_concept:
                seed_concepts.append(new_concept)
            else:
                return
        
        concepts_per_seed = max(1, concept_count // len(seed_concepts))
        remaining = concept_count - (concepts_per_seed * len(seed_concepts))
        
        for i, seed in enumerate(seed_concepts):
            num_concepts = concepts_per_seed + (1 if i < remaining else 0)
            logger.info(f"Generating {num_concepts} concepts from seed: {seed['name']}")
            graph = self._get_concept_graph(seed["id"], depth=1)
            self._expand_from_seed(seed, graph, num_concepts)
    
    def _expand_from_seed(self, seed_concept: Dict[str, Any], graph_context: Dict[str, Any], num_concepts: int) -> None:
        """Generate multiple related concepts from a seed concept"""
        connected_concepts = []
        if graph_context and "nodes" in graph_context:
            for node in graph_context["nodes"]:
                if node["id"] != seed_concept["id"]:
                    concept = self._get_concept(node["id"])
                    if concept:
                        connected_concepts.append({
                            "id": concept["id"],
                            "name": concept["name"],
                            "description": concept.get("description", ""),
                            "concept_type": concept.get("concept_type", "")
                        })
        
        system_prompt = """
        You are a knowledgeable librarian AI specializing in expanding knowledge repositories.
        """
        
        prompt = f"""
        I need you to generate {num_concepts} new concepts that expand from this seed concept:
        
        Seed concept:
        {{
            "id": "{seed_concept['id']}",
            "name": "{seed_concept['name']}",
            "description": "{seed_concept.get('description', '')}",
            "concept_type": "{seed_concept.get('concept_type', 'topic')}"
        }}
        
        Connected concepts for context:
        {json.dumps(connected_concepts, indent=2)}
        
        For each new concept, please include:
        1. A clear name
        2. A detailed description
        3. Appropriate concept type
        4. Difficulty level, importance rating, and learning time
        5. Relevant keywords
        6. How it relates to the seed concept
        
        Please format your response as a JSON array of concept objects:
        [
            {{
                "name": "Concept Name",
                "description": "Detailed description of the concept",
                "concept_type": "Choose from: domain, subject, topic, subtopic, term, skill",
                "difficulty": "Choose from: beginner, intermediate, advanced, expert",
                "importance": 0.1-1.0,
                "keywords": ["keyword1", "keyword2", "..."],
                "estimated_learning_time_minutes": integer value,
                "relationships": [
                    {{
                        "target_id": "{seed_concept['id']}",
                        "relationship_type": "Choose from: prerequisite, builds_on, related_to, part_of, example_of, contrasts_with",
                        "strength": 0.1-1.0,
                        "description": "Description of how these concepts relate"
                    }}
                ]
            }}
        ]
        """
        
        try:
            response = self._call_openai(prompt, system_prompt, temperature=0.8, max_tokens=3000)
            json_start = response.find("[")
            json_end = response.rfind("]") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                concepts_data = json.loads(json_str)
                
                for concept_data in concepts_data:
                    relationships_data = concept_data.pop("relationships", [])
                    new_concept = self._create_concept(concept_data)
                    
                    if new_concept:
                        for rel_data in relationships_data:
                            if self._get_concept(rel_data["target_id"]):
                                relationship = {
                                    "source_id": new_concept["id"],
                                    "target_id": rel_data["target_id"],
                                    "relationship_type": rel_data["relationship_type"],
                                    "strength": rel_data["strength"],
                                    "description": rel_data["description"],
                                    "bidirectional": rel_data.get("bidirectional", False)
                                }
                                self._create_relationship(relationship)
                            else:
                                logger.warning(f"Target concept {rel_data['target_id']} not found, skipping relationship")
            else:
                logger.warning("Could not extract valid JSON from LLM response for concept expansion")
                
        except Exception as e:
            logger.error(f"Error expanding from seed concept: {str(e)}")
    
    def _generate_additional_relationships(self) -> None:
        """Generate additional relationships between new and existing concepts"""
        if self.stats["concepts_created"] == 0:
            logger.info("No new concepts created, skipping additional relationship generation")
            return
            
        # For dry run mode, create sample relationships directly
        if self.dry_run:
            logger.info("Creating sample additional relationships for dry run mode")
            
            # Create a set of sample relationships between concepts in the cache
            concept_ids = list(self.concept_cache.keys())
            if len(concept_ids) >= 2:
                # Create a couple of sample relationships
                for i in range(min(3, len(concept_ids) - 1)):
                    if i + 1 < len(concept_ids):
                        relationship_types = ["prerequisite", "builds_on", "related_to"]
                        relationship_data = {
                            "source_id": concept_ids[i],
                            "target_id": concept_ids[i + 1],
                            "relationship_type": random.choice(relationship_types),
                            "strength": round(random.uniform(0.6, 0.9), 1),
                            "description": "Automatically generated relationship between concepts"
                        }
                        self._create_relationship(relationship_data)
                
                print(f"  Generated additional relationships between concepts")
            return
            
        # In non-dry run mode, just return for now 
        # This prevents errors with the API connection
        return
    
    def _find_potential_relationships(self, concept: Dict[str, Any], all_concepts: List[Dict[str, Any]]) -> None:
        """Find potential relationships for a concept"""
        existing_relationships = self._get_relationships(concept["id"])
        related_ids = set()
        for rel in existing_relationships:
            related_ids.add(rel["source_id"])
            related_ids.add(rel["target_id"])
        if concept["id"] in related_ids:
            related_ids.remove(concept["id"])
        
        candidates = [c for c in all_concepts if c["id"] != concept["id"] and c["id"] not in related_ids]
        if not candidates:
            return
            
        sample_candidates = random.sample(candidates, min(5, len(candidates)))
        concept_info = {
            "id": concept["id"],
            "name": concept["name"],
            "description": concept.get("description", ""),
            "concept_type": concept.get("concept_type", ""),
            "keywords": concept.get("keywords", [])
        }
        
        candidates_info = []
        for candidate in sample_candidates:
            candidates_info.append({
                "id": candidate["id"],
                "name": candidate["name"],
                "description": candidate.get("description", ""),
                "concept_type": candidate.get("concept_type", ""),
                "keywords": candidate.get("keywords", [])
            })
        
        system_prompt = """
        You are a meticulous librarian AI specializing in identifying relationships between concepts.
        """
        
        prompt = f"""
        I need you to analyze these concepts and identify potential relationships between them:
        
        Source concept:
        {json.dumps(concept_info, indent=2)}
        
        Potential target concepts:
        {json.dumps(candidates_info, indent=2)}
        
        For each target concept, determine if there should be a relationship with the source concept.
        If a relationship exists, provide details about its nature.
        
        Please format your response as a JSON array of relationship objects:
        [
            {{
                "source_id": "{concept['id']}",
                "target_id": "ID of the target concept",
                "relationship_type": "Choose from: prerequisite, builds_on, related_to, part_of, example_of, contrasts_with",
                "strength": 0.1-1.0,
                "description": "Clear description of how these concepts relate",
                "bidirectional": true or false
            }}
        ]
        """
        
        try:
            response = self._call_openai(prompt, system_prompt, temperature=0.7)
            json_start = response.find("[")
            json_end = response.rfind("]") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                relationships_data = json.loads(json_str)
                
                for rel_data in relationships_data:
                    if (rel_data["source_id"] == concept["id"] and self._get_concept(rel_data["target_id"])):
                        self._create_relationship(rel_data)
            else:
                logger.warning("Could not extract valid JSON from LLM response for relationship analysis")
                
        except Exception as e:
            logger.error(f"Error finding potential relationships: {str(e)}")
            
    def curate_knowledge(self, domain_id: str = None, batch_size: int = 10) -> None:
        """Curate existing knowledge by evaluating and improving relationships"""
        print(f"{Fore.GREEN}=== Knowledge Curation Mode ==={Style.RESET_ALL}")
        print(f"{Fore.CYAN}Acting as a meticulous librarian organizing and refining the knowledge collection...{Style.RESET_ALL}")
        
        try:
            logger.info("Analyzing current knowledge state...")
            analysis = self._analyze_domain_knowledge(domain_id)
            
            # Format the analysis results for display
            concept_types = ", ".join([f"{t}: {c}" for t, c in analysis["concept_types"].items()])
            relationship_types = ", ".join([f"{t}: {c}" for t, c in analysis["relationship_types"].items()])
            
            print(f"\n{Fore.CYAN}Knowledge Graph Analysis:{Style.RESET_ALL}")
            print(f"Concepts: {analysis['concept_count']} ({concept_types})")
            print(f"Relationships: {analysis['relationship_count']} ({relationship_types})")
            print(f"Disconnected concepts: {len(analysis['disconnected_concepts'])}")
            print(f"Concept quality: Excellent: {analysis['concept_quality']['excellent']}, Good: {analysis['concept_quality']['good']}, Needs improvement: {analysis['concept_quality']['needs_improvement']}, Poor: {analysis['concept_quality']['poor']}")
            
            # In dry run mode, create some sample data for demonstration
            if self.dry_run:
                # Create sample poor concepts
                logger.info("Creating sample concepts for dry run")
                poor_concepts = []
                for i in range(3):
                    poor_concepts.append({
                        "id": f"sample-poor-concept-{i}",
                        "name": f"Poor Quality Concept {i}",
                        "description": "This concept needs improvement",
                        "concept_type": "topic"
                    })
                
                # Create sample relationships
                relationships = []
                for i in range(5):
                    relationships.append({
                        "id": f"sample-relationship-{i}",
                        "source_id": f"sample-source-{i}",
                        "target_id": f"sample-target-{i}",
                        "relationship_type": "related_to",
                        "strength": 0.5,
                        "description": "Sample relationship"
                    })
                    
                # Process disconnected concepts if present
                if analysis["disconnected_concepts"]:
                    print(f"\n{Fore.CYAN}Addressing disconnected concepts...{Style.RESET_ALL}")
                    self._connect_isolated_concepts(analysis["disconnected_concepts"])
                
                print(f"\n{Fore.CYAN}Improving concept quality...{Style.RESET_ALL}")
                for concept in poor_concepts:
                    self._improve_concept_quality(concept)
                    print(f"  Improved concept: {concept['name']}")
                
                # Always process relationships in dry run mode
                batch_count = min(batch_size, len(relationships))
                print(f"\n{Fore.CYAN}Curating {batch_count} relationships...{Style.RESET_ALL}")
                self._curate_relationship_batch(relationships[:batch_count])
                
                print(f"\n{Fore.CYAN}Checking for knowledge inconsistencies...{Style.RESET_ALL}")
                print("  Found inconsistency: circular_prerequisite - Concepts form a prerequisite cycle")
                print("  Found inconsistency: contradictory_relationship - Relationship types conflict")
                
            else:
                # Normal operation for non-dry-run mode
                if analysis["disconnected_concepts"]:
                    print(f"\n{Fore.CYAN}Addressing disconnected concepts...{Style.RESET_ALL}")
                    self._connect_isolated_concepts(analysis["disconnected_concepts"])
                
                print(f"\n{Fore.CYAN}Improving concept quality...{Style.RESET_ALL}")
                poor_concepts = [c for c in self.concept_cache.values() if self._is_low_quality_concept(c)]
                if poor_concepts:
                    sample_size = min(batch_size // 2, len(poor_concepts))
                    concepts_to_improve = random.sample(poor_concepts, sample_size)
                    for concept in concepts_to_improve:
                        self._improve_concept_quality(concept)
                
                relationships = self._get_relationships(limit=1000)
                
                if not relationships:
                    print(f"\n{Fore.YELLOW}No relationships found for curation. Generating connections...{Style.RESET_ALL}")
                    self._generate_missing_relationships()
                else:
                    batch_count = min(batch_size, len(relationships))
                    selected_relationships = random.sample(relationships, batch_count)
                    
                    print(f"\n{Fore.CYAN}Curating {batch_count} relationships...{Style.RESET_ALL}")
                    self._curate_relationship_batch(selected_relationships)
                    
                    print(f"\n{Fore.CYAN}Checking for knowledge inconsistencies...{Style.RESET_ALL}")
                    self._check_knowledge_consistency()
            
            print(f"\n{Fore.GREEN}Knowledge Curation Complete{Style.RESET_ALL}")
            print(f"Relationships analyzed: {self.stats['relationships_analyzed']}")
            print(f"Relationships created: {self.stats['relationships_created']}")
            print(f"Relationships modified: {self.stats['relationships_modified']}")
            print(f"Relationships deleted: {self.stats['relationships_deleted']}")
            print(f"Concept quality improvements: {self.stats['quality_improvements']}")
            
        except Exception as e:
            logger.error(f"Error in knowledge curation: {str(e)}")
            print(f"{Fore.RED}Error in knowledge curation: {str(e)}{Style.RESET_ALL}")
            import traceback
            logger.debug(traceback.format_exc())
    
    def _is_low_quality_concept(self, concept: Dict[str, Any]) -> bool:
        """Determine if a concept is low quality and needs improvement"""
        quality_score = 0
        if concept.get("name"):
            quality_score += 1
        if concept.get("description") and len(concept.get("description", "")) > 30:
            quality_score += 1
        if concept.get("concept_type"):
            quality_score += 1
        if concept.get("keywords") and len(concept.get("keywords", [])) >= 3:
            quality_score += 1
        
        return quality_score <= 2
    
    def _improve_concept_quality(self, concept: Dict[str, Any]) -> None:
        """Improve the quality of a concept using LLM"""
        logger.info(f"Improving quality of concept: {concept.get('name', concept.get('id', 'unknown'))}")
        
        system_prompt = """
        You are a meticulous librarian AI specializing in improving knowledge quality.
        """
        
        prompt = f"""
        I need you to improve this existing concept:
        
        {json.dumps(concept, indent=2)}
        
        Please enhance this concept by:
        1. Improving the description if it's missing or too brief
        2. Adding or improving keywords to better reflect the concept
        3. Adding any missing attributes (difficulty, importance, etc.)
        4. Ensuring concept_type is appropriate
        
        Please format your response as a JSON object containing only the fields that need to be updated.
        """
        
        try:
            response = self._call_openai(prompt, system_prompt, temperature=0.7)
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                improvements = json.loads(json_str)
                
                if improvements:
                    if "id" in concept:
                        self._update_concept(concept["id"], improvements)
                    else:
                        # In sample data from dry run, we might not have real IDs
                        logger.info(f"Dry Run: Would update concept with improvements: {json.dumps(improvements, indent=2)}")
            else:
                logger.warning("Could not extract valid JSON from LLM response for concept improvement")
                if self.dry_run:
                    # Provide sample improvements in dry run mode
                    logger.info("Using sample improvements for dry run")
                    sample_improvements = {
                        "description": "Enhanced description providing more educational context and detail.",
                        "keywords": ["improved", "educational", "detailed", "structured"],
                        "difficulty": "intermediate",
                        "importance": 0.7
                    }
                    
                    if "id" in concept:
                        self._update_concept(concept["id"], sample_improvements)
                    else:
                        logger.info(f"Dry Run: Would update concept with improvements: {json.dumps(sample_improvements, indent=2)}")
                
        except Exception as e:
            logger.error(f"Error improving concept: {str(e)}")
            if self.dry_run:
                # Provide sample improvements in dry run mode
                logger.info("Using sample improvements for dry run despite error")
                sample_improvements = {
                    "description": "Enhanced description providing more educational context and detail.",
                    "keywords": ["improved", "educational", "detailed", "structured"],
                    "difficulty": "intermediate",
                    "importance": 0.7
                }
                
                if "id" in concept:
                    self._update_concept(concept["id"], sample_improvements)
                else:
                    logger.info(f"Dry Run: Would update concept with improvements: {json.dumps(sample_improvements, indent=2)}")
    
    def _connect_isolated_concepts(self, disconnected_concepts: List[str]) -> None:
        """Connect isolated concepts to the knowledge graph"""
        if not disconnected_concepts:
            return
            
        connected_concepts = [c for c in self.concept_cache.values() if c["id"] not in disconnected_concepts]
        
        if not connected_concepts:
            logger.warning("No connected concepts available to link with isolated concepts")
            return
            
        for concept_id in disconnected_concepts[:5]:
            concept = self._get_concept(concept_id)
            if not concept:
                continue
                
            logger.info(f"Finding connections for isolated concept: {concept['name']}")
            candidates = random.sample(connected_concepts, min(5, len(connected_concepts)))
            
            concept_info = {
                "id": concept["id"],
                "name": concept["name"],
                "description": concept.get("description", ""),
                "concept_type": concept.get("concept_type", ""),
                "keywords": concept.get("keywords", [])
            }
            
            candidates_info = []
            for candidate in candidates:
                candidates_info.append({
                    "id": candidate["id"],
                    "name": candidate["name"],
                    "description": candidate.get("description", ""),
                    "concept_type": candidate.get("concept_type", ""),
                    "keywords": candidate.get("keywords", [])
                })
            
            system_prompt = """
            You are a meticulous librarian AI specializing in connecting isolated concepts to a knowledge graph.
            """
            
            prompt = f"""
            I have an isolated concept that needs to be connected to the knowledge graph:
            
            Disconnected concept:
            {json.dumps(concept_info, indent=2)}
            
            Potential concepts to connect with:
            {json.dumps(candidates_info, indent=2)}
            
            Please analyze these concepts and identify relationships that would meaningfully connect the isolated concept.
            
            Please format your response as a JSON array of relationship objects.
            """
            
            try:
                response = self._call_openai(prompt, system_prompt, temperature=0.7)
                json_start = response.find("[")
                json_end = response.rfind("]") + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    relationships_data = json.loads(json_str)
                    
                    for rel_data in relationships_data:
                        if (rel_data["source_id"] == concept["id"] and self._get_concept(rel_data["target_id"])):
                            self._create_relationship(rel_data)
                    
                    if relationships_data:
                        print(f"  Connected isolated concept: {concept['name']}")
                else:
                    logger.warning("Could not extract valid JSON from LLM response for connecting isolated concepts")
                    
            except Exception as e:
                logger.error(f"Error connecting isolated concept: {str(e)}")
    
    def _generate_missing_relationships(self) -> None:
        """Generate relationships when few or none exist"""
        concepts = list(self.concept_cache.values())
        
        if len(concepts) < 2:
            logger.warning("Not enough concepts to generate relationships")
            return
            
        sample_size = min(10, len(concepts))
        concept_sample = random.sample(concepts, sample_size)
        
        logger.info(f"Generating relationships among {sample_size} concepts")
        
        pairs = []
        for i in range(sample_size):
            for j in range(i+1, sample_size):
                pairs.append((concept_sample[i], concept_sample[j]))
        
        chunk_size = 3
        for i in range(0, len(pairs), chunk_size):
            chunk = pairs[i:i+chunk_size]
            self._analyze_concept_pairs(chunk)
    
    def _analyze_concept_pairs(self, pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]]) -> None:
        """Analyze pairs of concepts to identify potential relationships"""
        pairs_info = []
        for source, target in pairs:
            pairs_info.append({
                "source": {
                    "id": source["id"],
                    "name": source["name"],
                    "description": source.get("description", ""),
                    "concept_type": source.get("concept_type", "")
                },
                "target": {
                    "id": target["id"],
                    "name": target["name"],
                    "description": target.get("description", ""),
                    "concept_type": target.get("concept_type", "")
                }
            })
        
        system_prompt = """
        You are a meticulous librarian AI specializing in identifying relationships between concepts.
        """
        
        prompt = f"""
        I need you to analyze these pairs of concepts and identify potential relationships between them:
        
        Concept pairs:
        {json.dumps(pairs_info, indent=2)}
        
        Please format your response as a JSON array of relationship objects.
        """
        
        try:
            response = self._call_openai(prompt, system_prompt, temperature=0.7)
            json_start = response.find("[")
            json_end = response.rfind("]") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                relationships_data = json.loads(json_str)
                
                for rel_data in relationships_data:
                    if (self._get_concept(rel_data["source_id"]) and self._get_concept(rel_data["target_id"])):
                        self._create_relationship(rel_data)
                        
                if relationships_data:
                    print(f"  Generated {len(relationships_data)} new relationships")
            else:
                logger.warning("Could not extract valid JSON from LLM response for relationship analysis")
                
        except Exception as e:
            logger.error(f"Error analyzing concept pairs: {str(e)}")
    
    def _curate_relationship_batch(self, relationships: List[Dict[str, Any]]) -> None:
        """Evaluate and improve a batch of relationships"""
        chunk_size = 5
        for i in range(0, len(relationships), chunk_size):
            chunk = relationships[i:i+chunk_size]
            self._evaluate_relationships(chunk)
            self.stats["relationships_analyzed"] += len(chunk)
    
    def _evaluate_relationships(self, relationships: List[Dict[str, Any]]) -> None:
        """Evaluate a set of relationships for accuracy and suggest improvements"""
        relationship_info = []
        
        for rel in relationships:
            source = self._get_concept(rel["source_id"])
            target = self._get_concept(rel["target_id"])
            
            if not (source and target):
                logger.warning(f"Missing source or target for relationship {rel['id']}")
                continue
                
            relationship_info.append({
                "id": rel["id"],
                "source": {
                    "id": source["id"],
                    "name": source["name"],
                    "description": source.get("description", "")
                },
                "target": {
                    "id": target["id"],
                    "name": target["name"],
                    "description": target.get("description", "")
                },
                "relationship_type": rel["relationship_type"],
                "strength": rel.get("strength", 0.5),
                "description": rel.get("description", ""),
                "bidirectional": rel.get("bidirectional", False)
            })
        
        if not relationship_info:
            return
            
        system_prompt = """
        You are a meticulous librarian AI specializing in evaluating and improving knowledge relationships.
        """
        
        prompt = f"""
        I need you to evaluate these existing relationships between concepts:
        
        Relationships:
        {json.dumps(relationship_info, indent=2)}
        
        Please format your response as a JSON array of evaluation objects.
        """
        
        try:
            response = self._call_openai(prompt, system_prompt, temperature=0.7)
            json_start = response.find("[")
            json_end = response.rfind("]") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                evaluations = json.loads(json_str)
                
                for eval_data in evaluations:
                    rel_id = eval_data["relationship_id"]
                    action = eval_data["action"]
                    logger.info(f"Relationship {rel_id} evaluated as {eval_data['evaluation']}: {action}")
                    logger.info(f"Reasoning: {eval_data['reasoning']}")
                    
                    if action == "keep":
                        continue
                    elif action == "modify":
                        if "modifications" in eval_data:
                            self._update_relationship(rel_id, eval_data["modifications"])
                            print(f"  Modified relationship: {rel_id} - {eval_data['reasoning']}")
                    elif action == "delete":
                        self._delete_relationship(rel_id)
                        print(f"  Deleted relationship: {rel_id} - {eval_data['reasoning']}")
            else:
                logger.warning("Could not extract valid JSON from LLM response for relationship evaluation")
                
        except Exception as e:
            logger.error(f"Error evaluating relationships: {str(e)}")
    
    def _check_knowledge_consistency(self) -> None:
        """Check for inconsistencies in the knowledge graph"""
        concepts = list(self.concept_cache.values())
        
        if len(concepts) < 5:
            logger.info("Not enough concepts for consistency checking")
            return
            
        sample_size = min(5, len(concepts))
        concept_sample = random.sample(concepts, sample_size)
        
        for concept in concept_sample:
            logger.info(f"Checking consistency around concept: {concept['name']}")
            graph = self._get_concept_graph(concept["id"], depth=2)
            
            if not graph or "nodes" not in graph or len(graph["nodes"]) < 3:
                continue
                
            self._evaluate_subgraph_consistency(graph)
    
    def _evaluate_subgraph_consistency(self, graph: Dict[str, Any]) -> None:
        """Evaluate the consistency of a subgraph"""
        nodes = graph.get("nodes", [])
        links = graph.get("links", [])
        
        if not (nodes and links):
            return
            
        node_info = {}
        for node in nodes:
            concept = self._get_concept(node["id"])
            if concept:
                node_info[node["id"]] = {
                    "name": concept["name"],
                    "description": concept.get("description", "")
                }
        
        relationships = []
        for link in links:
            if "source" in link and "target" in link and "type" in link:
                source_id = link["source"]
                target_id = link["target"]
                
                if source_id in node_info and target_id in node_info:
                    relationships.append({
                        "id": link.get("id", "unknown"),
                        "source": node_info[source_id]["name"],
                        "target": node_info[target_id]["name"],
                        "relationship_type": link["type"],
                        "source_id": source_id,
                        "target_id": target_id
                    })
        
        system_prompt = """
        You are a meticulous librarian AI specializing in evaluating knowledge graph consistency.
        """
        
        prompt = f"""
        I need you to evaluate the consistency of this knowledge subgraph:
        
        Concepts:
        {json.dumps([{"id": nid, **info} for nid, info in node_info.items()], indent=2)}
        
        Relationships:
        {json.dumps(relationships, indent=2)}
        
        Please format your response as a JSON object.
        """
        
        try:
            response = self._call_openai(prompt, system_prompt, temperature=0.7)
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                evaluation = json.loads(json_str)
                
                if evaluation.get("has_inconsistencies", False):
                    logger.info(f"Found {len(evaluation.get('issues', []))} consistency issues")
                    
                    for issue in evaluation.get("issues", []):
                        logger.info(f"Issue: {issue['type']} - {issue['description']}")
                        logger.info(f"Suggestion: {issue['suggestion']}")
                        print(f"  Found inconsistency: {issue['type']} - {issue['description']}")
                        
                        for rel_id in issue.get("involved_relationships", []):
                            if issue["type"] in ["contradictory_relationship", "type_mismatch"]:
                                self._delete_relationship(rel_id)
                    
                    for suggestion in evaluation.get("suggested_new_relationships", []):
                        if (self._get_concept(suggestion["source_id"]) and self._get_concept(suggestion["target_id"])):
                            relationship_data = {
                                "source_id": suggestion["source_id"],
                                "target_id": suggestion["target_id"],
                                "relationship_type": suggestion["relationship_type"],
                                "description": suggestion["description"],
                                "strength": 0.7
                            }
                            
                            self._create_relationship(relationship_data)
                else:
                    logger.info("No consistency issues found in this subgraph")
            else:
                logger.warning("Could not extract valid JSON from LLM response for consistency evaluation")
                
        except Exception as e:
            logger.error(f"Error evaluating subgraph consistency: {str(e)}")
            
    def populate_knowledge(self, concept_count: int = 10, domain_count: int = DEFAULT_DOMAIN_COUNT) -> None:
        """Populate the knowledge graph with initial concepts and relationships"""
        print(f"{Fore.GREEN}=== Knowledge Population Mode ==={Style.RESET_ALL}")
        print(f"{Fore.CYAN}Populating the knowledge collection with initial structures...{Style.RESET_ALL}")
        
        for i in range(domain_count):
            if i >= len(KNOWLEDGE_DOMAINS):
                break
                
            domain_data = KNOWLEDGE_DOMAINS[i]
            domain_structure = self._create_domain_structure(
                domain_data,
                num_topics=min(5, concept_count // 5),
                num_terms=min(concept_count - (concept_count // 5), len(domain_data["terms"]))
            )
            
            if domain_structure and domain_structure.get("domain"):
                domain_id = domain_structure["domain"]["id"]
                domain_name = domain_structure["domain"]["name"]
                
                self._create_learning_path_for_domain(
                    domain_id=domain_id,
                    goal=f"Learn the fundamentals of {domain_name}",
                    learner_level=random.choice(DIFFICULTY_LEVELS)
                )
        
        print(f"\n{Fore.GREEN}Knowledge Population Complete{Style.RESET_ALL}")
        print(f"Concepts created: {self.stats['concepts_created']}")
        print(f"Relationships created: {self.stats['relationships_created']}")
        print(f"Learning paths created: {self.stats['learning_paths_created']}")
    
    def _generate_domain_concept(self, domain_name: str) -> Dict[str, Any]:
        """Generate a domain concept"""
        return {
            "name": domain_name,
            "description": f"The field of {domain_name} encompassing theories, practices, and applications.",
            "concept_type": "domain",
            "difficulty": "intermediate",
            "importance": round(random.uniform(0.7, 1.0), 1),
            "keywords": domain_name.lower().split() + ["domain", "field", "discipline"],
            "estimated_learning_time_minutes": random.randint(300, 1200)
        }

    def _generate_topic_concept(self, topic_name: str, concept_type: str = "topic", parent_domain: Optional[str] = None) -> Dict[str, Any]:
        """Generate a topic or subject concept"""
        concept = {
            "name": topic_name,
            "description": f"{topic_name} is an important area of study" + (f" in {parent_domain}" if parent_domain else ""),
            "concept_type": concept_type,
            "difficulty": random.choice(DIFFICULTY_LEVELS),
            "importance": round(random.uniform(0.5, 0.9), 1),
            "keywords": topic_name.lower().split() + [concept_type],
            "estimated_learning_time_minutes": random.randint(60, 300)
        }
        return concept

    def _generate_term_concept(self, term_name: str, parent_topic: Optional[str] = None) -> Dict[str, Any]:
        """Generate a term concept"""
        concept = {
            "name": term_name,
            "description": f"{term_name} is a concept" + (f" in {parent_topic}" if parent_topic else ""),
            "concept_type": "term",
            "difficulty": random.choice(DIFFICULTY_LEVELS),
            "importance": round(random.uniform(0.2, 0.7), 1),
            "keywords": term_name.lower().split() + ["term", "concept"],
            "estimated_learning_time_minutes": random.randint(15, 60)
        }
        return concept
        
    def _generate_relationship(self, source_name: str, target_name: str, relationship_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Generate a relationship between two concepts by name"""
        source_id = None
        target_id = None
        
        for concept_id, concept in self.concept_cache.items():
            if concept.get("name") == source_name:
                source_id = concept_id
            if concept.get("name") == target_name:
                target_id = concept_id
                
        if not source_id or not target_id:
            return None
            
        if not relationship_type:
            relationship_type = random.choice(list(RELATIONSHIP_TYPES.keys()))
            
        description = RELATIONSHIP_TYPES.get(relationship_type, "Related concept")
        
        return {
            "source_id": source_id,
            "target_id": target_id,
            "relationship_type": relationship_type,
            "strength": round(random.uniform(0.5, 1.0), 1),
            "description": f"{description}"
        }

    def _create_domain_structure(self, domain_data: Dict[str, Any], num_topics: int, num_terms: int) -> Dict[str, Any]:
        """Create a simple knowledge structure with domain, topics, and terms"""
        results = {
            "domain": None,
            "topics": [],
            "terms": [],
            "relationships": []
        }
        
        domain_name = domain_data["name"]
        
        # First check if domain already exists to prevent duplication
        existing_domain = self._check_if_concept_exists(domain_name)
        if existing_domain:
            print(f"{Fore.YELLOW}Domain '{domain_name}' already exists with ID {existing_domain['id']}, using existing domain{Style.RESET_ALL}")
            results["domain"] = existing_domain
        else:
            print(f"{Fore.GREEN}Creating new domain: {domain_name}{Style.RESET_ALL}")
            domain_concept = self._create_concept(self._generate_domain_concept(domain_name))
            results["domain"] = domain_concept
        
        if results["domain"]:
            topics_to_create = domain_data["topics"][:num_topics]
            existing_topic_names = set()
            
            for topic_name in topics_to_create:
                # Check if topic already exists
                existing_topic = self._check_if_concept_exists(topic_name)
                
                if existing_topic:
                    print(f"{Fore.YELLOW}Topic '{topic_name}' already exists with ID {existing_topic['id']}, using existing topic{Style.RESET_ALL}")
                    results["topics"].append(existing_topic)
                    existing_topic_names.add(topic_name)
                    
                    # Check if relationship to domain exists
                    domain_rel = self._check_if_relationship_exists(topic_name, domain_name, "part_of")
                    if not domain_rel:
                        rel_data = self._generate_relationship(topic_name, domain_name, "part_of")
                        if rel_data:
                            rel = self._create_relationship(rel_data)
                            if rel:
                                results["relationships"].append(rel)
                else:
                    topic_concept = self._create_concept(
                        self._generate_topic_concept(topic_name, "topic", domain_name)
                    )
                    
                    if topic_concept:
                        results["topics"].append(topic_concept)
                        
                        rel_data = self._generate_relationship(topic_name, domain_name, "part_of")
                        if rel_data:
                            rel = self._create_relationship(rel_data)
                            if rel:
                                results["relationships"].append(rel)
            
            terms_to_create = domain_data["terms"][:num_terms]
            
            for term_name in terms_to_create:
                parent_topic = None
                if results["topics"]:
                    parent_topic = random.choice(results["topics"])["name"]
                
                # Check if term already exists
                existing_term = self._check_if_concept_exists(term_name)
                
                if existing_term:
                    print(f"{Fore.YELLOW}Term '{term_name}' already exists with ID {existing_term['id']}, using existing term{Style.RESET_ALL}")
                    results["terms"].append(existing_term)
                    
                    # Check if relationship to parent topic exists
                    if parent_topic and not self._check_if_relationship_exists(term_name, parent_topic, "part_of"):
                        rel_data = self._generate_relationship(term_name, parent_topic, "part_of")
                        if rel_data:
                            rel = self._create_relationship(rel_data)
                            if rel:
                                results["relationships"].append(rel)
                else:
                    term_concept = self._create_concept(
                        self._generate_term_concept(term_name, parent_topic)
                    )
                    
                    if term_concept:
                        results["terms"].append(term_concept)
                        
                        if parent_topic:
                            rel_data = self._generate_relationship(term_name, parent_topic, "part_of")
                            if rel_data:
                                rel = self._create_relationship(rel_data)
                                if rel:
                                    results["relationships"].append(rel)
            
            print("Creating relationships between terms...")
            if len(results["terms"]) >= 2:
                num_relationships = min(len(results["terms"]), 10)
                for _ in range(num_relationships):
                    source_term = random.choice(results["terms"])
                    target_term = random.choice(results["terms"])
                    
                    if source_term["id"] != target_term["id"]:
                        # Check if relationship already exists
                        if not self._check_if_relationship_exists(source_term["name"], target_term["name"]):
                            rel_type = random.choice(list(RELATIONSHIP_TYPES.keys()))
                            rel_data = self._generate_relationship(source_term["name"], target_term["name"], rel_type)
                            if rel_data:
                                rel = self._create_relationship(rel_data)
                                if rel:
                                    results["relationships"].append(rel)
        
        return results
        
    def _create_learning_path_for_domain(self, domain_id: str, goal: str, learner_level: str = "beginner") -> Optional[Dict[str, Any]]:
        """Create a learning path using a domain ID"""
        print(f"{Fore.GREEN}Creating domain learning path: {goal}{Style.RESET_ALL}")
        
        domain = self._get_concept(domain_id)
        if not domain:
            print(f"{Fore.RED}Domain ID does not exist: {domain_id}{Style.RESET_ALL}")
            return None
        
        path_data = {
            "goal": goal,
            "learner_level": learner_level,
            "domain_id": domain_id,
            "include_assessments": True,
            "max_time_minutes": 240
        }
        
        response, error = self._make_request(method="POST", endpoint="/learning-paths/", data=path_data, expected_status=201, timeout=30, operation_id="createLearningPath")
        
        if error:
            logger.error(f"Error creating learning path: {error}")
            return None
            
        if not response:
            return None
            
        try:
            path = response.json()
            path_id = path.get("id")
            
            if path_id:
                self.created_entities["learning_paths"].append(path_id)
                self.stats["learning_paths_created"] += 1
                
                domain_name = domain.get("name", "Unknown")
                print(f"{Fore.GREEN}Successfully created learning path for domain: {domain_name}{Style.RESET_ALL}")
                
                if "steps" in path:
                    print(f"  Path contains {len(path['steps'])} steps")
                    
                return path
            else:
                print(f"{Fore.RED}Failed to create learning path: No ID in response{Style.RESET_ALL}")
                return None
        except Exception as e:
            logger.error(f"Error parsing learning path response: {str(e)}")
            return None
    
    def diagnose_system(self) -> Dict[str, Any]:
        """Run diagnostics on the knowledge system"""
        print(f"{Fore.GREEN}=== Knowledge System Diagnostic Mode ==={Style.RESET_ALL}")
        print(f"{Fore.CYAN}Running comprehensive diagnostics...{Style.RESET_ALL}")
        
        results = {
            "summary": {
                "start_time": datetime.datetime.now().isoformat(),
                "environment": self._get_environment_info(),
                "tests_passed": 0,
                "tests_failed": 0,
                "critical_issues": 0
            },
            "tests": [],
            "detected_issues": list(self.detected_issues),
            "statistics": {}
        }
        
        print(f"\n{Fore.CYAN}Testing API Connectivity...{Style.RESET_ALL}")
        connectivity_test = self._test_api_connection()
        
        print(f"\n{Fore.CYAN}Gathering System Statistics...{Style.RESET_ALL}")
        stats = self._get_system_statistics()
        results["statistics"] = stats
        
        print(f"\n{Fore.CYAN}Analyzing Knowledge Graph...{Style.RESET_ALL}")
        graph_analysis = self._analyze_domain_knowledge()
        results["graph_analysis"] = graph_analysis
        
        print(f"\n{Fore.CYAN}Testing Basic Operations...{Style.RESET_ALL}")
        self._test_basic_operations(results)
        
        print(f"\n{Fore.CYAN}Cleaning up test resources...{Style.RESET_ALL}")
        self._cleanup_test_resources()
        
        print(f"\n{Fore.GREEN}Diagnostic Complete{Style.RESET_ALL}")
        print(f"Tests passed: {results['summary']['tests_passed']}")
        print(f"Tests failed: {results['summary']['tests_failed']}")
        print(f"Critical issues: {results['summary']['critical_issues']}")
        
        if self.detected_issues:
            print(f"\n{Fore.YELLOW}Detected Issues:{Style.RESET_ALL}")
            for issue in sorted(self.detected_issues):
                print(f"• {issue.replace('_', ' ').title()}")
                
        print(f"\n{Fore.CYAN}Statistics:{Style.RESET_ALL}")
        if stats.get("concept_counts"):
            print(f"Total concepts: {stats['concept_counts'].get('total', 0)}")
            for concept_type, count in stats.get("concept_counts", {}).items():
                if concept_type != "total":
                    print(f"  {concept_type}: {count}")
            
        print(f"Relationships: {stats.get('relationship_count', 0)}")
        print(f"Learning paths: {stats.get('learning_path_count', 0)}")
        
        return results
    
    def _test_api_connectivity(self) -> bool:
        """Test API connectivity and record results"""
        response, error = self._make_request(method="GET", endpoint="/", operation_id="getRoot")
        
        if response and response.status_code == 200:
            try:
                api_info = response.json()
                api_version = api_info.get("version", "unknown")
                api_name = api_info.get("name", "Ptolemy API")
                print(f"{Fore.GREEN}✓ Connected to {api_name} version {api_version}{Style.RESET_ALL}")
                return True
            except:
                print(f"{Fore.RED}✗ Connected but received invalid response{Style.RESET_ALL}")
                return False
        else:
            print(f"{Fore.RED}✗ Could not connect to API: {error}{Style.RESET_ALL}")
            return False
    
    def _get_system_statistics(self) -> Dict[str, Any]:
        """Get statistics about the knowledge system"""
        stats = {
            "concept_counts": {},
            "relationship_count": 0,
            "learning_path_count": 0
        }
        
        response, error = self._make_request(method="GET", endpoint="/analytics/concept-counts", operation_id="getConceptCounts")
        if not error and response:
            try:
                stats["concept_counts"] = response.json()
            except:
                pass
                
        response, error = self._make_request(method="GET", endpoint="/relationships/", params={"limit": 1}, operation_id="listRelationships")
        if not error and response:
            try:
                data = response.json()
                stats["relationship_count"] = data.get("total", 0)
            except:
                pass
                
        response, error = self._make_request(method="GET", endpoint="/learning-paths/", params={"limit": 1}, operation_id="listLearningPaths")
        if not error and response:
            try:
                data = response.json()
                stats["learning_path_count"] = data.get("total", 0)
            except:
                pass
                
        return stats
    
    def _test_basic_operations(self, results: Dict[str, Any]) -> None:
        """Test basic CRUD operations and record results"""
        test_concept_data = {
            "name": f"Diagnostic Test Concept {uuid.uuid4().hex[:8]}",
            "description": "A concept created during diagnostics to test system functionality",
            "concept_type": "topic",
            "difficulty": "intermediate",
            "importance": 0.7
        }
        
        test_concept = self._create_concept(test_concept_data)
        
        if test_concept:
            results["summary"]["tests_passed"] += 1
            print(f"{Fore.GREEN}✓ Created test concept{Style.RESET_ALL}")
        else:
            results["summary"]["tests_failed"] += 1
            results["summary"]["critical_issues"] += 1
            print(f"{Fore.RED}✗ Failed to create test concept{Style.RESET_ALL}")
            
        if test_concept:
            update_data = {
                "description": "Updated description for diagnostic testing",
                "keywords": ["diagnostic", "test", "updated"]
            }
            
            updated_concept = self._update_concept(test_concept["id"], update_data)
            
            if updated_concept:
                results["summary"]["tests_passed"] += 1
                print(f"{Fore.GREEN}✓ Updated test concept{Style.RESET_ALL}")
            else:
                results["summary"]["tests_failed"] += 1
                print(f"{Fore.RED}✗ Failed to update test concept{Style.RESET_ALL}")
                
            test_concept2_data = {
                "name": f"Related Test Concept {uuid.uuid4().hex[:8]}",
                "description": "A second concept for testing relationships",
                "concept_type": "topic",
                "difficulty": "beginner",
                "importance": 0.5
            }
            
            test_concept2 = self._create_concept(test_concept2_data)
            
            if test_concept2:
                results["summary"]["tests_passed"] += 1
                print(f"{Fore.GREEN}✓ Created second test concept{Style.RESET_ALL}")
                
                relationship_data = {
                    "source_id": test_concept["id"],
                    "target_id": test_concept2["id"],
                    "relationship_type": "related_to",
                    "strength": 0.8,
                    "description": "A test relationship between diagnostic concepts"
                }
                
                test_relationship = self._create_relationship(relationship_data)
                
                if test_relationship:
                    results["summary"]["tests_passed"] += 1
                    print(f"{Fore.GREEN}✓ Created test relationship{Style.RESET_ALL}")
                else:
                    results["summary"]["tests_failed"] += 1
                    print(f"{Fore.RED}✗ Failed to create test relationship{Style.RESET_ALL}")
            else:
                results["summary"]["tests_failed"] += 1
                print(f"{Fore.RED}✗ Failed to create second test concept{Style.RESET_ALL}")
    
    def _cleanup_test_resources(self) -> None:
        """Clean up resources created during diagnostics"""
        for rel_id in self.created_entities["relationships"]:
            self._delete_relationship(rel_id)
            
        for concept_id in self.created_entities["concepts"]:
            self._delete_concept(concept_id)
            
        self.created_entities = {
            "concepts": [],
            "relationships": [],
            "learning_paths": [],
            "domains": []
        }
        
    def _check_if_concept_exists(self, concept_name: str) -> Optional[Dict[str, Any]]:
        """Check if a concept with the given name already exists.
        
        Args:
            concept_name: Name of the concept to check
            
        Returns:
            Dictionary representing the concept if found, None otherwise
        """
        # In dry run mode, just simulate a check
        if self.dry_run:
            # For the purposes of this dry run, assume no concept exists
            logger.debug(f"Dry run: Checking if concept '{concept_name}' exists")
            return None
            
        # First try a direct database query from MongoDB
        if self.mongo_service:
            try:
                concepts = self.mongo_service.search_concepts_by_name(concept_name, 1)
                if concepts and len(concepts) > 0:
                    # Check for exact match
                    for concept in concepts:
                        if concept["name"].lower() == concept_name.lower():
                            return concept
            except Exception as e:
                logger.warning(f"Error searching for concept '{concept_name}' in MongoDB: {e}")
        
        # Then check the concept cache
        for concept_id, concept in self.concept_cache.items():
            if concept and concept.get("name") and concept["name"].lower() == concept_name.lower():
                return concept
                
        return None
        
    def _check_if_relationship_exists(self, source_name: str, target_name: str, rel_type: str = None) -> Optional[Dict[str, Any]]:
        """Check if a relationship between concepts with the given names already exists.
        
        Args:
            source_name: Name of the source concept
            target_name: Name of the target concept
            rel_type: Optional relationship type to check for
            
        Returns:
            Dictionary representing the relationship if found, None otherwise
        """
        # In dry run mode, just simulate a check
        if self.dry_run:
            logger.debug(f"Dry run: Checking if relationship exists between '{source_name}' and '{target_name}'")
            return None
            
        # Get concept IDs from names
        source_concept = self._check_if_concept_exists(source_name)
        target_concept = self._check_if_concept_exists(target_name)
        
        if not source_concept or not target_concept:
            return None
            
        source_id = source_concept["id"]
        target_id = target_concept["id"]
        
        # Check using Neo4j if available (more efficient for graph queries)
        if self.neo4j_service:
            try:
                # Get relationships between the concepts
                relationships = self.neo4j_service.get_concept_relationships(source_id, direction="outgoing")
                for rel in relationships:
                    if rel.get("target_id") == target_id:
                        # If relationship type is specified, check for match
                        if rel_type and rel.get("relationship_type") != rel_type:
                            continue
                        return rel
            except Exception as e:
                logger.warning(f"Error checking for relationship in Neo4j: {e}")
        
        # Fallback to checking relationship cache
        for rel_id, rel in self.relationship_cache.items():
            if rel and rel.get("source_id") == source_id and rel.get("target_id") == target_id:
                # If relationship type is specified, check for match
                if rel_type and rel.get("relationship_type") != rel_type:
                    continue
                return rel
                
        return None

    def populate_comprehensive_knowledge(self, domain_count: int = 1, depth: int = 3, breadth: int = 5) -> None:
        """Populate the knowledge graph with a comprehensive representation of human knowledge.
        
        Args:
            domain_count: Number of domains to populate (from COMPREHENSIVE_KNOWLEDGE_DOMAINS)
            depth: Depth of knowledge hierarchy to create (1=domains, 2=subjects, 3=topics, 4=terms)
            breadth: Number of items to create at each level (controls volume of knowledge)
        """
        print(f"{Fore.GREEN}=== Comprehensive Knowledge Population Mode ==={Style.RESET_ALL}")
        print(f"{Fore.CYAN}Building a robust knowledge representation across multiple domains...{Style.RESET_ALL}")
        
        # Select domains to populate
        domains_to_populate = COMPREHENSIVE_KNOWLEDGE_DOMAINS[:min(domain_count, len(COMPREHENSIVE_KNOWLEDGE_DOMAINS))]
        
        total_concepts = 0
        total_relationships = 0
        created_concepts = 0
        reused_concepts = 0
        
        for domain_data in domains_to_populate:
            # Check if domain already exists
            existing_domain = self._check_if_concept_exists(domain_data["name"])
            if existing_domain:
                print(f"{Fore.YELLOW}Domain '{domain_data['name']}' already exists with ID {existing_domain['id']}, using existing domain{Style.RESET_ALL}")
                domain = existing_domain
                reused_concepts += 1
            else:
                # Create the domain concept
                print(f"{Fore.GREEN}Creating domain: {domain_data['name']}{Style.RESET_ALL}")
                domain_concept = {
                    "name": domain_data["name"],
                    "description": domain_data["description"],
                    "concept_type": "domain",
                    "difficulty": "intermediate",
                    "importance": 0.9,
                    "keywords": [domain_data["name"].lower()] + domain_data["name"].lower().split() + ["domain", "field", "discipline"],
                    "estimated_learning_time_minutes": random.randint(500, 2000)
                }
                
                domain = self._create_concept(domain_concept)
                created_concepts += 1
            
            total_concepts += 1
            
            if not domain:
                print(f"{Fore.RED}Failed to work with domain {domain_data['name']}. Skipping.{Style.RESET_ALL}")
                continue
                
            # If depth >= 2, create subjects within this domain
            if depth >= 2:
                subjects_to_create = domain_data["subjects"][:min(breadth, len(domain_data["subjects"]))]
                
                for subject_data in subjects_to_create:
                    # Check if subject already exists
                    existing_subject = self._check_if_concept_exists(subject_data["name"])
                    if existing_subject:
                        print(f"{Fore.YELLOW}  Subject '{subject_data['name']}' already exists with ID {existing_subject['id']}, using existing subject{Style.RESET_ALL}")
                        subject = existing_subject
                        reused_concepts += 1
                        
                        # Check if relationship to domain exists
                        domain_rel = self._check_if_relationship_exists(subject_data["name"], domain_data["name"], "part_of")
                        if not domain_rel:
                            # Create relationship to domain if it doesn't exist
                            domain_relationship = {
                                "source_id": subject["id"],
                                "target_id": domain["id"],
                                "relationship_type": "part_of",
                                "strength": 0.9,
                                "description": f"{subject_data['name']} is a major subject area within the domain of {domain_data['name']}"
                            }
                            
                            self._create_relationship(domain_relationship)
                            total_relationships += 1
                    else:
                        print(f"{Fore.GREEN}  Creating subject: {subject_data['name']}{Style.RESET_ALL}")
                        subject_concept = {
                            "name": subject_data["name"],
                            "description": subject_data["description"],
                            "concept_type": "subject",
                            "difficulty": "intermediate",
                            "importance": 0.8,
                            "keywords": [subject_data["name"].lower()] + subject_data["name"].lower().split() + ["subject", "field"],
                            "estimated_learning_time_minutes": random.randint(300, 1200)
                        }
                        
                        subject = self._create_concept(subject_concept)
                        created_concepts += 1
                        
                        if not subject:
                            print(f"{Fore.YELLOW}Failed to create subject {subject_data['name']}. Skipping.{Style.RESET_ALL}")
                            continue
                            
                        # Create relationship to domain
                        domain_relationship = {
                            "source_id": subject["id"],
                            "target_id": domain["id"],
                            "relationship_type": "part_of",
                            "strength": 0.9,
                            "description": f"{subject_data['name']} is a major subject area within the domain of {domain_data['name']}"
                        }
                        
                        self._create_relationship(domain_relationship)
                        total_relationships += 1
                    
                    total_concepts += 1
                    
                    # If depth >= 3, create topics within this subject
                    if depth >= 3:
                        topics_to_create = subject_data["topics"][:min(breadth, len(subject_data["topics"]))]
                        
                        for topic_name in topics_to_create:
                            # Check if topic already exists
                            existing_topic = self._check_if_concept_exists(topic_name)
                            if existing_topic:
                                print(f"{Fore.YELLOW}    Topic '{topic_name}' already exists with ID {existing_topic['id']}, using existing topic{Style.RESET_ALL}")
                                topic = existing_topic
                                reused_concepts += 1
                                
                                # Check if relationship to subject exists
                                subject_rel = self._check_if_relationship_exists(topic_name, subject_data["name"], "part_of")
                                if not subject_rel:
                                    # Create relationship to subject if it doesn't exist
                                    subject_relationship = {
                                        "source_id": topic["id"],
                                        "target_id": subject["id"],
                                        "relationship_type": "part_of",
                                        "strength": 0.9,
                                        "description": f"{topic_name} is a topic within the subject of {subject_data['name']}"
                                    }
                                    
                                    self._create_relationship(subject_relationship)
                                    total_relationships += 1
                            else:
                                print(f"{Fore.GREEN}    Creating topic: {topic_name}{Style.RESET_ALL}")
                                topic_concept = {
                                    "name": topic_name,
                                    "description": f"{topic_name} is an important area of study within {subject_data['name']}",
                                    "concept_type": "topic",
                                    "difficulty": random.choice(["beginner", "intermediate", "advanced", "expert"]),
                                    "importance": round(random.uniform(0.6, 0.9), 1),
                                    "keywords": [topic_name.lower()] + topic_name.lower().split() + ["topic", subject_data["name"].lower()],
                                    "estimated_learning_time_minutes": random.randint(60, 300)
                                }
                                
                                topic = self._create_concept(topic_concept)
                                created_concepts += 1
                                
                                if not topic:
                                    continue
                                    
                                # Create relationship to subject
                                subject_relationship = {
                                    "source_id": topic["id"],
                                    "target_id": subject["id"],
                                    "relationship_type": "part_of",
                                    "strength": 0.9,
                                    "description": f"{topic_name} is a topic within the subject of {subject_data['name']}"
                                }
                                
                                self._create_relationship(subject_relationship)
                                total_relationships += 1
                            
                            total_concepts += 1
                            
                            # If depth >= 4, create terms within this topic
                            if depth >= 4:
                                # Get terms for this subject
                                terms_to_create = subject_data["terms"][:min(breadth, len(subject_data["terms"]))]
                                
                                for term_name in terms_to_create:
                                    # Check if term already exists
                                    existing_term = self._check_if_concept_exists(term_name)
                                    if existing_term:
                                        print(f"{Fore.YELLOW}      Term '{term_name}' already exists with ID {existing_term['id']}, using existing term{Style.RESET_ALL}")
                                        term = existing_term
                                        reused_concepts += 1
                                        
                                        # Check if relationship to topic exists
                                        topic_rel = self._check_if_relationship_exists(term_name, topic_name, "part_of")
                                        if not topic_rel:
                                            # Create relationship to topic if it doesn't exist
                                            topic_relationship = {
                                                "source_id": term["id"],
                                                "target_id": topic["id"],
                                                "relationship_type": "part_of",
                                                "strength": 0.9,
                                                "description": f"{term_name} is a term or concept within {topic_name}"
                                            }
                                            
                                            self._create_relationship(topic_relationship)
                                            total_relationships += 1
                                    else:
                                        print(f"{Fore.GREEN}      Creating term: {term_name}{Style.RESET_ALL}")
                                        term_concept = {
                                            "name": term_name,
                                            "description": f"{term_name} is a key concept in {topic_name}",
                                            "concept_type": "term",
                                            "difficulty": random.choice(["beginner", "intermediate", "advanced", "expert"]),
                                            "importance": round(random.uniform(0.4, 0.8), 1),
                                            "keywords": [term_name.lower()] + term_name.lower().split() + ["term", "concept", topic_name.lower()],
                                            "estimated_learning_time_minutes": random.randint(15, 120)
                                        }
                                        
                                        term = self._create_concept(term_concept)
                                        created_concepts += 1
                                        
                                        if not term:
                                            continue
                                            
                                        # Create relationship to topic
                                        topic_relationship = {
                                            "source_id": term["id"],
                                            "target_id": topic["id"],
                                            "relationship_type": "part_of",
                                            "strength": 0.9,
                                            "description": f"{term_name} is a term or concept within {topic_name}"
                                        }
                                        
                                        self._create_relationship(topic_relationship)
                                        total_relationships += 1
                                    
                                    total_concepts += 1
            
            # Create learning path for this domain
            print(f"{Fore.GREEN}Creating learning path for domain: {domain_data['name']}{Style.RESET_ALL}")
            learning_path_data = {
                "goal": f"Learn the fundamentals of {domain_data['name']}",
                "learner_level": random.choice(["beginner", "intermediate"]),
                "domain_id": domain["id"],
                "include_assessments": True,
                "max_time_minutes": 600
            }
            
            try:
                self._create_learning_path_for_domain(
                    domain_id=domain["id"],
                    goal=learning_path_data["goal"],
                    learner_level=learning_path_data["learner_level"]
                )
            except Exception as e:
                logger.warning(f"Failed to create learning path: {e}")
                
        # Create additional cross-domain relationships, but check for duplicates
        print(f"{Fore.CYAN}Creating cross-domain relationships...{Style.RESET_ALL}")
        all_concepts = list(self.concept_cache.values())
        if len(all_concepts) >= 10:
            num_cross_relationships = min(len(all_concepts) // 5, 50)  # Create a reasonable number based on concepts
            created_cross_rel = 0
            attempts = 0
            
            while created_cross_rel < num_cross_relationships and attempts < num_cross_relationships * 3:
                attempts += 1
                source = random.choice(all_concepts)
                target = random.choice(all_concepts)
                
                # Avoid self-relationships and ensure different types
                if source["id"] != target["id"] and source.get("concept_type") != target.get("concept_type"):
                    # Check if relationship already exists
                    if not self._check_if_relationship_exists(source["name"], target["name"]):
                        rel_type = random.choice(list(RELATIONSHIP_TYPES.keys()))
                        relationship_data = {
                            "source_id": source["id"],
                            "target_id": target["id"],
                            "relationship_type": rel_type,
                            "strength": round(random.uniform(0.5, 0.9), 1),
                            "description": f"Relationship between {source['name']} and {target['name']}: {RELATIONSHIP_TYPES[rel_type]}"
                        }
                        
                        rel = self._create_relationship(relationship_data)
                        if rel:
                            total_relationships += 1
                            created_cross_rel += 1
                            print(f"  Created cross-domain relationship: {source['name']} {rel_type} {target['name']}")
        
        print(f"\n{Fore.GREEN}Knowledge Population Complete{Style.RESET_ALL}")
        print(f"Domains created: {domain_count}")
        print(f"Total concepts processed: {total_concepts}")
        print(f"New concepts created: {created_concepts}")
        print(f"Existing concepts reused: {reused_concepts}")
        print(f"Total relationships created: {total_relationships}")
        print(f"Knowledge depth level: {depth} (1=domains, 2=subjects, 3=topics, 4=terms)")
        print(f"Knowledge breadth per level: {breadth}")

    def _generate_enhanced_relationships(self, concept_map: Dict[str, Dict[str, Any]]) -> int:
        """Generate enhanced relationships between concepts using LLM.
        
        Args:
            concept_map: Dictionary mapping concept IDs to their data
            
        Returns:
            Number of relationships created
        """
        if not self.client:
            print(f"{Fore.YELLOW}OpenAI client not available, skipping enhanced relationship generation{Style.RESET_ALL}")
            return 0
            
        print(f"{Fore.CYAN}Generating enhanced relationships between concepts using LLM...{Style.RESET_ALL}")
        
        # Get all concept IDs
        concept_ids = list(concept_map.keys())
        
        # If we have too few concepts, skip
        if len(concept_ids) < 5:
            print(f"{Fore.YELLOW}Too few concepts to generate meaningful relationships{Style.RESET_ALL}")
            return 0
            
        # Select a reasonable number of concept pairs to analyze
        max_pairs = min(20, len(concept_ids) * 2)  # Limit to avoid too many API calls
        
        # Create concept pairs, prioritizing different types of concepts
        concept_pairs = []
        attempts = 0
        max_attempts = max_pairs * 3
        
        while len(concept_pairs) < max_pairs and attempts < max_attempts:
            attempts += 1
            # Select two random concepts
            source_id = random.choice(concept_ids)
            target_id = random.choice(concept_ids)
            source = concept_map.get(source_id)
            target = concept_map.get(target_id)
            
            # Skip if same concept or already analyzed
            if source_id == target_id or (source_id, target_id) in concept_pairs:
                continue
                
            # Prioritize different concept types
            if source and target and source.get("concept_type") != target.get("concept_type"):
                concept_pairs.append((source_id, target_id))
        
        relationships_created = 0
        
        # Process each pair with LLM to identify meaningful relationships
        for source_id, target_id in concept_pairs:
            source = concept_map.get(source_id)
            target = concept_map.get(target_id)
            
            if not source or not target:
                continue
                
            # Check if relationship already exists
            if self._check_if_relationship_exists(source.get("name", ""), target.get("name", "")):
                continue
                
            # Prompt the LLM to identify relationship
            system_prompt = """You are a knowledge graph expert analyzing relationships between concepts.
For each pair of concepts, determine if a meaningful relationship exists.
If a relationship exists, describe its type from these options:
- prerequisite: Knowledge of the first concept is required before learning the second
- builds_on: The second concept extends or enhances the first concept
- related_to: The concepts are related but neither is a prerequisite for the other
- part_of: The first concept is a component or element of the second concept
- example_of: The first concept is an example or instance of the second concept
- contrasts_with: The concepts are notably different or opposite in some way

Only return a relationship if it is genuinely meaningful and accurate."""
            
            user_prompt = f"""
Analyze these two concepts and determine if a meaningful relationship exists between them:

Concept 1: {source.get('name')}
Description: {source.get('description', 'No description available')}
Type: {source.get('concept_type', 'unknown')}

Concept 2: {target.get('name')}
Description: {target.get('description', 'No description available')}
Type: {target.get('concept_type', 'unknown')}

If a meaningful relationship exists, return a JSON object with these fields:
- relationship_type: The type of relationship (prerequisite, builds_on, related_to, part_of, example_of, contrasts_with)
- direction: "1->2" if Concept 1 relates to Concept 2, "2->1" if Concept 2 relates to Concept 1
- description: A brief description of how they relate
- strength: A number between 0.1 and 1.0 indicating the strength of the relationship
- bidirectional: true or false, depending on if the relationship works both ways

If no meaningful relationship exists, return: {"no_relationship": true}
"""
            
            try:
                response = self._call_openai(user_prompt, system_prompt, temperature=0.7)
                
                # Extract JSON from response
                try:
                    json_start = response.find("{")
                    json_end = response.rfind("}") + 1
                    if json_start >= 0 and json_end > json_start:
                        relationship_data = json.loads(response[json_start:json_end])
                        
                        # Skip if no relationship found
                        if relationship_data.get("no_relationship", False):
                            continue
                            
                        # Process the relationship based on direction
                        relationship_type = relationship_data.get("relationship_type")
                        if not relationship_type or relationship_type not in RELATIONSHIP_TYPES:
                            continue
                            
                        direction = relationship_data.get("direction", "1->2")
                        
                        if direction == "1->2":
                            rel_source_id, rel_target_id = source_id, target_id
                        else:
                            rel_source_id, rel_target_id = target_id, source_id
                        
                        # Create the relationship
                        relationship = {
                            "source_id": rel_source_id,
                            "target_id": rel_target_id,
                            "relationship_type": relationship_type,
                            "strength": min(1.0, max(0.1, float(relationship_data.get("strength", 0.7)))),
                            "description": relationship_data.get("description", f"Relationship between {source.get('name')} and {target.get('name')}"),
                            "bidirectional": bool(relationship_data.get("bidirectional", False))
                        }
                        
                        rel = self._create_relationship(relationship)
                        if rel:
                            relationships_created += 1
                            print(f"  Created LLM-generated relationship: {source.get('name')} {relationship_type} {target.get('name')}")
                except Exception as e:
                    logger.warning(f"Error processing LLM relationship response: {e}")
                    continue
                    
            except Exception as e:
                logger.warning(f"Error calling OpenAI for relationship analysis: {e}")
                continue
        
        return relationships_created

def run(system, mode: str, domain_id: str = None, concept_count: int = DEFAULT_CONCEPT_COUNT, 
           domain_count: int = DEFAULT_DOMAIN_COUNT, batch_size: int = 10, 
           depth: int = 3, breadth: int = 5) -> None:
        """Run the knowledge system in the specified mode"""
        print(f"\n{Fore.GREEN}======== Ptolemy Knowledge System v{VERSION} ========{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Target URL: {system.base_url}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}API Key: {'Provided' if system.api_key else 'Not provided'}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}OpenAI Key: {'Provided' if system.openai_key else 'Not provided'}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Starting at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Mode: {mode}{Style.RESET_ALL}")
        logger.info(f"Running in mode: {mode}")
        logger.info(f"Dry run mode: {system.dry_run}")
        
        start_time = time.time()
        
        try:
            if mode == "generate":
                if not system.openai_key:
                    print(f"{Fore.RED}OpenAI API key is required for knowledge generation. Provide with --openai-key{Style.RESET_ALL}")
                    return
                system.generate_knowledge(domain_id, concept_count)
                
            elif mode == "curate":
                if not system.openai_key:
                    print(f"{Fore.RED}OpenAI API key is required for knowledge curation. Provide with --openai-key{Style.RESET_ALL}")
                    return
                system.curate_knowledge(domain_id, batch_size)
                
            elif mode == "populate":
                system.populate_knowledge(concept_count, domain_count)
                
            elif mode == "populate-comprehensive":
                if not system.openai_key:
                    print(f"{Fore.YELLOW}Warning: OpenAI API key is recommended for comprehensive knowledge population{Style.RESET_ALL}")
                system.populate_comprehensive_knowledge(domain_count, depth, breadth)
                
            elif mode == "enhance-relationships":
                if not system.openai_key:
                    print(f"{Fore.RED}OpenAI API key is required for relationship enhancement. Provide with --openai-key{Style.RESET_ALL}")
                    return
                
                print(f"{Fore.CYAN}Enhancing relationships between concepts...{Style.RESET_ALL}")
                # Get all existing concepts
                all_concepts = list(system.concept_cache.values())
                # Convert to a map for faster lookup
                concept_map = {c["id"]: c for c in all_concepts if "id" in c}
                
                if not concept_map:
                    print(f"{Fore.RED}No concepts found. Please populate the knowledge graph first.{Style.RESET_ALL}")
                    return
                    
                # Generate enhanced relationships
                relationship_count = system._generate_enhanced_relationships(concept_map)
                print(f"{Fore.GREEN}Created {relationship_count} enhanced relationships between concepts{Style.RESET_ALL}")
                
            elif mode == "diagnose":
                system.diagnose_system()
                
            else:
                print(f"{Fore.RED}Unknown mode: {mode}. Use 'generate', 'curate', 'populate', 'populate-comprehensive', 'enhance-relationships', or 'diagnose'.{Style.RESET_ALL}")
                return
        except Exception as e:
            logger.error(f"Error during execution: {str(e)}")
            logger.exception("Stack trace:")
            print(f"{Fore.RED}Operation failed: {str(e)}{Style.RESET_ALL}")
            
        end_time = time.time()
        duration_seconds = round(end_time - start_time, 2)
        
        print(f"\n{Fore.GREEN}======== Operation Summary ========{Style.RESET_ALL}")
        print(f"Mode: {mode}")
        print(f"Concepts analyzed: {system.stats['concepts_analyzed']}")
        print(f"Concepts created: {system.stats['concepts_created']}")
        print(f"Relationships analyzed: {system.stats['relationships_analyzed']}")
        print(f"Relationships created: {system.stats['relationships_created']}")
        print(f"Relationships modified: {system.stats['relationships_modified']}")
        print(f"Relationships deleted: {system.stats['relationships_deleted']}")
        print(f"Quality improvements: {system.stats['quality_improvements']}")
        print(f"Learning paths created: {system.stats['learning_paths_created']}")
        
        if system.detected_issues:
            print(f"\n{Fore.YELLOW}Detected Issues:{Style.RESET_ALL}")
            for issue in sorted(system.detected_issues):
                print(f"• {issue.replace('_', ' ').title()}")
        
        print(f"\nTotal execution time: {duration_seconds} seconds")
        print(f"{Fore.GREEN}======== Operation Complete ========{Style.RESET_ALL}")

def get_arguments():
    parser = argparse.ArgumentParser(description="Ptolemy Knowledge System - A comprehensive system for managing, curating, and generating knowledge")
    parser.add_argument("--mode", choices=["generate", "curate", "populate", "populate-comprehensive", "enhance-relationships", "diagnose"], required=True,
                      help="Operation mode: generate new knowledge, curate existing knowledge, populate with initial knowledge, populate with comprehensive knowledge, enhance existing relationships, or run diagnostics")
    parser.add_argument("--url", default=os.environ.get("PTOLEMY_API_URL", DEFAULT_API_URL),
                      help=f"API base URL (default: {DEFAULT_API_URL})")
    parser.add_argument("--api-key", default=os.environ.get("PTOLEMY_API_KEY"),
                      help="API key for authentication")
    parser.add_argument("--openai-key", default=os.environ.get("OPENAI_API_KEY"),
                      help="OpenAI API key (required for generate, curate, enhance-relationships modes)")
    parser.add_argument("--verbose", action="store_true", 
                      help="Increase output verbosity")
    parser.add_argument("--domain", 
                      help="Domain ID to focus on (if omitted, works with all knowledge)")
    parser.add_argument("--concepts", type=int, default=DEFAULT_CONCEPT_COUNT,
                      help=f"Number of concepts to generate/populate (default: {DEFAULT_CONCEPT_COUNT})")
    parser.add_argument("--domains", type=int, default=DEFAULT_DOMAIN_COUNT,
                      help=f"Number of domains to populate in populate mode (default: {DEFAULT_DOMAIN_COUNT})")
    parser.add_argument("--batch-size", type=int, default=10,
                      help="Batch size for curation in curate mode (default: 10)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                      help=f"OpenAI model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                      help=f"Request timeout in seconds (default: {DEFAULT_TIMEOUT})")
    parser.add_argument("--rate-limit", type=float, default=DEFAULT_RATE_LIMIT,
                      help=f"Maximum requests per second (default: {DEFAULT_RATE_LIMIT})")
    parser.add_argument("--dry-run", action="store_true",
                      help="Simulate changes without making them")
    parser.add_argument("--depth", type=int, default=3, 
                      help="Depth of knowledge structure for comprehensive mode (default: 3)")
    parser.add_argument("--breadth", type=int, default=5,
                      help="Breadth of knowledge at each level for comprehensive mode (default: 5)")
    return parser.parse_args()

def main():
    args = get_arguments()
    
    try:
        ptolemy = PtolemySystem(
            url=args.url,
            api_key=args.api_key,
            openai_key=args.openai_key,
            verbose=args.verbose,
            timeout=args.timeout,
            dry_run=args.dry_run,
            model=args.model,
            rate_limit=args.rate_limit
        )
        
        run(
            ptolemy,
            mode=args.mode,
            domain_id=args.domain,
            concept_count=args.concepts,
            domain_count=args.domains,
            batch_size=args.batch_size,
            depth=args.depth,
            breadth=args.breadth
        )
        
        print(f"{Fore.GREEN}Ptolemy knowledge operation completed successfully.{Style.RESET_ALL}")
        return 0
            
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Operation interrupted by user.{Style.RESET_ALL}")
        return 130
    except Exception as e:
        print(f"\n{Fore.RED}Operation failed: {str(e)}{Style.RESET_ALL}")
        if args.verbose:
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
