"""
Template Data Models
===================
Models for content templates and template processing.
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field


class TemplateType(str, Enum):
    """Types of templates."""
    CONCEPT = "concept"
    LESSON = "lesson"
    ASSESSMENT = "assessment"
    EXERCISE = "exercise"
    SUMMARY = "summary"
    PATHWAY = "pathway"
    CUSTOM = "custom"


class SectionType(str, Enum):
    """Types of sections within templates."""
    INTRODUCTION = "introduction"
    EXPLANATION = "explanation"
    EXAMPLE = "example"
    PRACTICE = "practice"
    ASSESSMENT = "assessment"
    SUMMARY = "summary"
    MEDIA = "media"
    INTERACTIVE = "interactive"


class TemplateTarget(str, Enum):
    """Target audience or purpose for templates."""
    STUDENT = "student"
    TEACHER = "teacher"
    PARENT = "parent"
    GENERAL = "general"


class StoicElement(str, Enum):
    """Stoic philosophical elements for template content."""
    VIRTUE = "virtue"
    WISDOM = "wisdom"
    COURAGE = "courage"
    JUSTICE = "justice"
    TEMPERANCE = "temperance"
    DICHOTOMY_OF_CONTROL = "dichotomy_of_control"
    JOURNALING = "journaling"
    MEDITATION = "meditation"
    REFLECTION = "reflection"


class TemplatePlaceholder(BaseModel):
    """Model for placeholders in templates."""
    name: str = Field(..., description="Placeholder name")
    description: str = Field(..., description="Description of what should replace this placeholder")
    required: bool = Field(True, description="Whether this placeholder is required")
    default_value: Optional[str] = Field(None, description="Default value if not provided")
    example: Optional[str] = Field(None, description="Example value for documentation")


class TemplatePrompt(BaseModel):
    """Model for LLM prompts in templates."""
    id: str = Field(..., description="Unique ID for the prompt")
    prompt_text: str = Field(..., description="The prompt text with placeholders")
    system_message: Optional[str] = Field(None, description="Optional system message for LLM context")
    temperature: float = Field(0.7, description="Temperature for generation")
    max_tokens: int = Field(1000, description="Maximum tokens for response")
    placeholders: List[TemplatePlaceholder] = Field(default_factory=list, description="Placeholders in the prompt")
    output_format: Optional[Dict[str, Any]] = Field(None, description="Expected output format as a schema")


class TemplateMediaSpec(BaseModel):
    """Model for media specifications in templates."""
    id: str = Field(..., description="Unique ID for the media specification")
    media_type: str = Field(..., description="Type of media to generate")
    prompt: str = Field(..., description="Prompt for generating the media")
    style: Optional[str] = Field(None, description="Style guidelines")
    aspect_ratio: Optional[str] = Field(None, description="Preferred aspect ratio")
    resolution: Optional[str] = Field(None, description="Preferred resolution")
    placeholders: List[TemplatePlaceholder] = Field(default_factory=list, description="Placeholders in the prompt")


class TemplateInteractiveSpec(BaseModel):
    """Model for interactive element specifications in templates."""
    id: str = Field(..., description="Unique ID for the interactive specification")
    element_type: str = Field(..., description="Type of interactive element")
    prompt: str = Field(..., description="Prompt for generating the interactive element")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters for the interactive element")
    placeholders: List[TemplatePlaceholder] = Field(default_factory=list, description="Placeholders in the prompt")


class TemplateSection(BaseModel):
    """Model for a section within a template."""
    id: str = Field(..., description="Unique ID for the section")
    title: str = Field(..., description="Section title")
    section_type: SectionType = Field(..., description="Type of section")
    order: int = Field(..., description="Order in the template")
    prompts: List[TemplatePrompt] = Field(..., description="Prompts for generating section content")
    media_specs: List[TemplateMediaSpec] = Field(default_factory=list, description="Media specifications")
    interactive_specs: List[TemplateInteractiveSpec] = Field(default_factory=list, description="Interactive specifications")
    subsections: List["TemplateSection"] = Field(default_factory=list, description="Subsections")
    stoic_elements: List[StoicElement] = Field(default_factory=list, description="Stoic elements in this section")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ContentTemplate(BaseModel):
    """Model for a complete content template."""
    template_id: str = Field(..., description="Unique ID for the template")
    name: str = Field(..., description="Template name")
    description: str = Field(..., description="Template description")
    template_type: TemplateType = Field(..., description="Type of template")
    version: str = Field("1.0", description="Template version")
    target: TemplateTarget = Field(TemplateTarget.STUDENT, description="Target audience")
    sections: List[TemplateSection] = Field(..., description="Template sections")
    global_placeholders: List[TemplatePlaceholder] = Field(default_factory=list, description="Global placeholders")
    stoic_elements: List[StoicElement] = Field(default_factory=list, description="Stoic elements in this template")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_by: Optional[str] = Field(None, description="Creator of the template")
    last_modified: Optional[str] = Field(None, description="Last modification timestamp")