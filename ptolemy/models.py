"""
Ptolemy Knowledge Map System - Data Models
=========================================
Defines the core data models and schemas for the knowledge graph system.
"""

import uuid
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Set
from datetime import datetime
from pydantic import BaseModel, Field, validator, root_validator, model_validator

class ConceptType(str, Enum):
    DOMAIN = "domain"          # Top-level knowledge area
    SUBJECT = "subject"        # Major subdivision of a domain
    TOPIC = "topic"            # Specific area within a subject
    SUBTOPIC = "subtopic"      # Component of a topic
    TERM = "term"              # Specific terminology or definition
    SKILL = "skill"            # Practical ability or technique

class RelationshipType(str, Enum):
    PREREQUISITE = "prerequisite"        # Must be understood before
    BUILDS_ON = "builds_on"              # Extends or enhances
    RELATED_TO = "related_to"            # Connected but not prerequisite
    PART_OF = "part_of"                  # Component of a larger concept
    EXAMPLE_OF = "example_of"            # Illustrates or instantiates
    CONTRASTS_WITH = "contrasts_with"    # Highlights differences

class DifficultyLevel(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

class LearnerLevel(str, Enum):
    NOVICE = "novice"
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

class ValidationStatus(str, Enum):
    PENDING = "pending"
    VALID = "valid"
    INVALID = "invalid"
    WARNING = "warning"

class ExportFormat(str, Enum):
    JSON = "json"
    CSV = "csv"
    XML = "xml"
    MARKDOWN = "markdown"
    GRAPHML = "graphml"
    CYPHER = "cypher"

class ActivityType(str, Enum):
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    QUERY = "query"
    EXPORT = "export"
    IMPORT = "import"
    VALIDATE = "validate"
    GENERATE = "generate"

# Input models for API requests

class ConceptCreate(BaseModel):
    """Model for creating a new concept."""
    name: str
    description: str
    concept_type: ConceptType
    difficulty: DifficultyLevel = DifficultyLevel.INTERMEDIATE
    parent_id: Optional[str] = None
    importance: float = Field(0.5, ge=0.0, le=1.0)
    complexity: float = Field(0.5, ge=0.0, le=1.0)
    keywords: List[str] = []
    estimated_learning_time_minutes: Optional[int] = None
    taxonomies: Optional[Dict[str, str]] = None
    external_references: Optional[Dict[str, str]] = None
    metadata: Optional[Dict[str, Any]] = None

class ConceptUpdate(BaseModel):
    """Model for updating an existing concept."""
    name: Optional[str] = None
    description: Optional[str] = None
    concept_type: Optional[ConceptType] = None
    difficulty: Optional[DifficultyLevel] = None
    parent_id: Optional[str] = None
    importance: Optional[float] = Field(None, ge=0.0, le=1.0)
    complexity: Optional[float] = Field(None, ge=0.0, le=1.0)
    keywords: Optional[List[str]] = None
    estimated_learning_time_minutes: Optional[int] = None
    taxonomies: Optional[Dict[str, str]] = None
    external_references: Optional[Dict[str, str]] = None
    metadata: Optional[Dict[str, Any]] = None
    validation_status: Optional[ValidationStatus] = None

class RelationshipCreate(BaseModel):
    """Model for creating a new relationship between concepts."""
    source_id: str
    target_id: str
    relationship_type: RelationshipType
    strength: float = Field(0.5, ge=0.0, le=1.0)
    description: Optional[str] = None
    bidirectional: bool = False
    metadata: Optional[Dict[str, Any]] = None

class RelationshipUpdate(BaseModel):
    """Model for updating an existing relationship."""
    relationship_type: Optional[RelationshipType] = None
    strength: Optional[float] = Field(None, ge=0.0, le=1.0)
    description: Optional[str] = None
    bidirectional: Optional[bool] = None
    metadata: Optional[Dict[str, Any]] = None

# Database models

class Concept(BaseModel):
    """Full concept model with all properties."""
    id: str
    name: str
    description: str
    concept_type: ConceptType
    difficulty: DifficultyLevel
    parent_id: Optional[str] = None
    importance: float
    complexity: float = 0.5
    keywords: List[str] = []
    estimated_learning_time_minutes: Optional[int] = None
    taxonomies: Optional[Dict[str, str]] = None
    external_references: Optional[Dict[str, str]] = None
    metadata: Optional[Dict[str, Any]] = None
    validation_status: ValidationStatus = ValidationStatus.PENDING
    created_at: datetime
    updated_at: datetime
    embedding_id: Optional[str] = None
    version: int = 1
    
    @classmethod
    def create_from_input(cls, input_data: ConceptCreate) -> 'Concept':
        """Create a new Concept from a ConceptCreate input."""
        now = datetime.now()
        return cls(
            id=str(uuid.uuid4()),
            name=input_data.name,
            description=input_data.description,
            concept_type=input_data.concept_type,
            difficulty=input_data.difficulty,
            parent_id=input_data.parent_id,
            importance=input_data.importance,
            complexity=input_data.complexity,
            keywords=input_data.keywords,
            estimated_learning_time_minutes=input_data.estimated_learning_time_minutes,
            taxonomies=input_data.taxonomies,
            external_references=input_data.external_references,
            metadata=input_data.metadata,
            created_at=now,
            updated_at=now
        )
    
    def apply_update(self, updates: ConceptUpdate) -> 'Concept':
        """Apply updates from a ConceptUpdate input."""
        update_data = updates.dict(exclude_unset=True)
        # Use pydantic's copy/update method to generate an updated instance.
        updated_concept = self.copy(update=update_data)
        updated_concept.updated_at = datetime.now()
        updated_concept.version = self.version + 1
        return updated_concept
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "name": "Object-Oriented Programming",
                "description": "A paradigm based on objects with properties and methods...",
                "concept_type": "topic",
                "difficulty": "intermediate",
                "parent_id": "123e4567-e89b-12d3-a456-426614174001",
                "importance": 0.8,
                "complexity": 0.6,
                "keywords": ["OOP", "classes", "objects", "inheritance"],
                "estimated_learning_time_minutes": 120,
                "validation_status": "valid",
                "created_at": "2023-01-01T00:00:00",
                "updated_at": "2023-01-02T00:00:00",
                "version": 1
            }
        }

# Composite models

class ConceptWithRelationships(BaseModel):
    """Model for a concept with its relationships."""
    concept: Concept
    incoming_relationships: List[Dict[str, Any]] = []
    outgoing_relationships: List[Dict[str, Any]] = []

class LearningPathStep(BaseModel):
    """Model for a step in a learning path."""
    concept_id: str
    order: int
    estimated_time_minutes: int
    reason: str
    learning_activities: List[str] = []
    assessment_suggestions: Optional[List[str]] = None
    resources: Optional[List[Dict[str, str]]] = None
    optional: bool = False
    
class LearningPath(BaseModel):
    """Model for a complete learning path."""
    id: str
    name: str
    description: str
    goal: str
    target_learner_level: LearnerLevel
    concepts: List[str]
    steps: List[LearningPathStep]
    total_time_minutes: int
    created_at: datetime
    updated_at: datetime
    metadata: Optional[Dict[str, Any]] = None

class LearningPathRequest(BaseModel):
    """Model for requesting a learning path generation."""
    goal: str = Field(..., description="Learning goal or objective")
    learner_level: LearnerLevel = Field(LearnerLevel.BEGINNER, description="Learner's experience level")
    domain_id: Optional[str] = Field(None, description="Domain ID to generate path from")
    concept_ids: Optional[List[str]] = Field(None, description="List of concept IDs to include in path")
    max_time_minutes: Optional[int] = Field(None, description="Maximum total learning time in minutes")
    include_assessments: bool = Field(False, description="Whether to include assessment items")
    include_resources: bool = Field(False, description="Whether to include additional learning resources")
    prior_knowledge: Optional[List[str]] = Field(None, description="List of concepts the learner already knows")

    @model_validator(mode='after')
    def validate_required_fields(self):
        """Validate that either domain_id or concept_ids is provided."""
        if self.domain_id is None and (not self.concept_ids or len(self.concept_ids) == 0):
            raise ValueError("Either domain_id or concept_ids must be provided")
        return self

    class Config:
        json_schema_extra = {
            "example": {
                "goal": "Learn the fundamentals of calculus",
                "learner_level": "intermediate",
                "concept_ids": ["3fa85f64-5717-4562-b3fc-2c963f66afa6", "3fa85f64-5717-4562-b3fc-2c963f66afa7"],
                "max_time_minutes": 120,
                "include_assessments": True,
                "include_resources": True,
                "prior_knowledge": ["algebra", "trigonometry"]
            }
        }

class ValidationIssue(BaseModel):
    """Model for a validation issue in the knowledge graph."""
    issue_type: str
    severity: str
    concepts_involved: List[str]
    description: str
    recommendation: str
    
class ValidationResult(BaseModel):
    """Model for the result of a validation check."""
    valid: bool
    issues: List[ValidationIssue] = []
    warnings: List[ValidationIssue] = []
    timestamp: datetime
    
class KnowledgeGap(BaseModel):
    """Model for an identified gap in the knowledge structure."""
    id: str
    domain_id: str
    title: str
    description: str
    importance: float
    related_concepts: List[str] = []
    suggested_content: Optional[str] = None
    status: str = "identified"  # identified, planned, addressed
    created_at: datetime
    updated_at: Optional[datetime] = None

class Activity(BaseModel):
    """Model for tracking user activities in the system."""
    id: str
    activity_type: ActivityType
    user_id: Optional[str] = None
    entity_type: str  # concept, relationship, domain, learning_path
    entity_id: str
    details: Dict[str, Any]
    timestamp: datetime
    
class ConceptSimilarityResult(BaseModel):
    """Model for concept similarity search results."""
    concept_id: str
    concept_name: str
    similarity: float
    concept_type: ConceptType

class DomainStructureRequest(BaseModel):
    """Model for requesting domain structure generation."""
    domain_name: str
    domain_description: str
    depth: int = Field(2, ge=1, le=3)
    generate_relationships: bool = True
    generate_learning_paths: bool = False
    concept_count: Optional[int] = None
    key_topics: Optional[List[str]] = None
    model: Optional[str] = None
    include_content_suggestions: bool = False
    difficulty_level: Optional[DifficultyLevel] = None
    target_audience: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class GraphExportRequest(BaseModel):
    """Model for requesting knowledge graph export."""
    root_concept_id: Optional[str] = None
    depth: int = Field(2, ge=1, le=5)
    include_metadata: bool = True
    format: ExportFormat = ExportFormat.JSON
    filter_types: Optional[List[ConceptType]] = None

class Relationship(BaseModel):
    """Model for relationships between concepts."""
    id: str
    source_id: str
    target_id: str
    relationship_type: RelationshipType
    strength: float
    description: Optional[str] = None
    bidirectional: bool = False
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    @classmethod
    def create_from_input(cls, input_data: RelationshipCreate) -> 'Relationship':
        """Create a new Relationship from a RelationshipCreate input."""
        now = datetime.now()
        return cls(
            id=str(uuid.uuid4()),
            source_id=input_data.source_id,
            target_id=input_data.target_id,
            relationship_type=input_data.relationship_type,
            strength=input_data.strength,
            description=input_data.description,
            bidirectional=input_data.bidirectional,
            metadata=input_data.metadata,
            created_at=now
        )
    
    def apply_update(self, updates: RelationshipUpdate) -> 'Relationship':
        """Apply updates from a RelationshipUpdate input."""
        update_data = updates.dict(exclude_unset=True)
        updated_relationship = self.copy(update=update_data)
        updated_relationship.updated_at = datetime.now()
        return updated_relationship
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174002",
                "source_id": "123e4567-e89b-12d3-a456-426614174000",
                "target_id": "123e4567-e89b-12d3-a456-426614174001",
                "relationship_type": "prerequisite",
                "strength": 0.9,
                "description": "Understanding OOP is required before learning design patterns",
                "bidirectional": False,
                "created_at": "2023-01-01T00:00:00",
                "updated_at": "2023-01-02T00:00:00"
            }
        }