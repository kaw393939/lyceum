"""
Template Engine
==============
Engine for processing templates with variable substitution and content generation.
"""

import json
import logging
import re
import uuid
from typing import Dict, List, Optional, Any, Union, Tuple
import asyncio
from datetime import datetime

from models.template import ContentTemplate, TemplateSection, TemplatePrompt, SectionType
from models.content import ContentType, ContentDifficulty, ContentSection, MediaItem, InteractiveElement, AssessmentItem
from integrations.llm_service import LLMService
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class ProcessedContent:
    """Container for processed template content."""
    
    def __init__(self):
        """Initialize empty processed content."""
        self.content = ""
        self.sections = []
        self.media = []
        self.interactive_elements = []
        self.assessment_items = []
        

class TemplateEngine:
    """Engine for processing content templates."""
    
    def __init__(self):
        """Initialize the template engine."""
        self.llm_service = LLMService()
        logger.info("Template engine initialized")
    
    async def process_template(self, 
                             template: ContentTemplate,
                             context: Dict[str, Any],
                             content_type: ContentType,
                             difficulty: ContentDifficulty,
                             age_range: str,
                             rag_processor=None) -> ProcessedContent:
        """
        Process a template with context to generate content.
        
        Args:
            template: Content template to process
            context: Context data for variable substitution
            content_type: Type of content to generate
            difficulty: Difficulty level
            age_range: Target age range
            rag_processor: Optional RAG processor for enhanced content
            
        Returns:
            Processed content with sections and media
        """
        logger.info(f"Processing template: {template.name}")
        result = ProcessedContent()
        
        # Add core context variables
        full_context = self._prepare_context(context, content_type, difficulty, age_range)
        
        # Process sections
        all_sections = []
        
        # Process sections in parallel
        tasks = []
        for section in template.sections:
            task = self._process_section(section, full_context, rag_processor)
            tasks.append(task)
        
        section_results = await asyncio.gather(*tasks)
        
        # Collect results
        for section_content, media_items, interactive_elems, assessment_items in section_results:
            all_sections.append(section_content)
            result.media.extend(media_items)
            result.interactive_elements.extend(interactive_elems)
            result.assessment_items.extend(assessment_items)
        
        # Store structured sections
        result.sections = all_sections
        
        # Combine sections into full content
        result.content = self._combine_sections(all_sections)
        
        logger.info(f"Template processing complete: {len(all_sections)} sections, {len(result.media)} media items")
        return result
    
    def _prepare_context(self, context: Dict[str, Any], 
                       content_type: ContentType,
                       difficulty: ContentDifficulty, 
                       age_range: str) -> Dict[str, Any]:
        """
        Prepare the full context for template processing.
        
        Args:
            context: Raw context data
            content_type: Content type
            difficulty: Difficulty level
            age_range: Target age range
            
        Returns:
            Enriched context
        """
        full_context = dict(context)
        
        # Add standard context variables
        full_context.update({
            "content_type": content_type.value,
            "difficulty": difficulty.value,
            "age_range": age_range,
            "current_year": str(datetime.now().year)
        })
        
        # Extract concept name and description if available
        if "concept" in context:
            concept = context["concept"]
            full_context["concept_name"] = concept.get("name", "")
            full_context["concept_description"] = concept.get("description", "")
        
        return full_context
    
    async def _process_section(self, 
                             section: TemplateSection, 
                             context: Dict[str, Any],
                             rag_processor=None) -> Tuple[ContentSection, List[MediaItem], List[InteractiveElement], List[AssessmentItem]]:
        """
        Process a template section.
        
        Args:
            section: Template section to process
            context: Context data
            rag_processor: Optional RAG processor
            
        Returns:
            Tuple of (ContentSection, media_items, interactive_elements, assessment_items)
        """
        logger.debug(f"Processing section: {section.title}")
        
        # Process prompts to generate section content
        content_parts = []
        for prompt in section.prompts:
            part = await self._process_prompt(prompt, context, rag_processor)
            content_parts.append(part)
        
        # Join parts to create section content
        content = "\n\n".join(content_parts)
        
        # Generate section ID
        section_id = f"{section.section_type.value}-{uuid.uuid4()}"
        
        # Process media specifications
        media_items = []
        for media_spec in section.media_specs:
            media_item = await self._generate_media(media_spec, context)
            if media_item:
                media_items.append(media_item)
        
        # Process interactive elements
        interactive_elements = []
        for interactive_spec in section.interactive_specs:
            element = await self._generate_interactive_element(interactive_spec, context)
            if element:
                interactive_elements.append(element)
        
        # Create subsections recursively
        subsections = []
        subsection_media = []
        subsection_interactive = []
        subsection_assessment = []
        
        for subsection_template in section.subsections:
            sub_content, sub_media, sub_interactive, sub_assessment = await self._process_section(
                subsection_template, context, rag_processor
            )
            subsections.append(sub_content)
            subsection_media.extend(sub_media)
            subsection_interactive.extend(sub_interactive)
            subsection_assessment.extend(sub_assessment)
        
        # Create section object
        content_section = ContentSection(
            section_id=section_id,
            title=section.title,
            content=content,
            order=section.order,
            media=media_items + subsection_media,
            interactive_elements=interactive_elements + subsection_interactive,
            subsections=subsections,
            metadata={
                "section_type": section.section_type.value,
                "stoic_elements": [elem.value for elem in section.stoic_elements]
            }
        )
        
        # Generate assessment items if this is an assessment section
        assessment_items = []
        if section.section_type == SectionType.ASSESSMENT:
            assessment_items = await self._generate_assessment_items(section, context)
        
        return content_section, media_items + subsection_media, interactive_elements + subsection_interactive, assessment_items + subsection_assessment
    
    async def _process_prompt(self, 
                            prompt: TemplatePrompt, 
                            context: Dict[str, Any],
                            rag_processor=None) -> str:
        """
        Process a template prompt to generate content.
        
        Args:
            prompt: Template prompt
            context: Context data
            rag_processor: Optional RAG processor
            
        Returns:
            Generated content
        """
        # Replace placeholders in prompt
        processed_prompt = self._replace_placeholders(prompt.prompt_text, context)
        processed_system = prompt.system_message
        if processed_system:
            processed_system = self._replace_placeholders(processed_system, context)
        
        # Use RAG processor if available
        if rag_processor and prompt.id.startswith("rag_"):
            result = await rag_processor.process_query(
                query=processed_prompt,
                context=context,
                temperature=prompt.temperature
            )
            return result.content
        
        # Otherwise, use LLM directly
        response = await self.llm_service.generate_content(
            prompt=processed_prompt,
            system_message=processed_system,
            temperature=prompt.temperature,
            max_tokens=prompt.max_tokens,
            output_format=prompt.output_format
        )
        
        return response.content
    
    def _replace_placeholders(self, text: str, context: Dict[str, Any]) -> str:
        """
        Replace placeholders in text with values from context.
        
        Args:
            text: Text with placeholders
            context: Context data
            
        Returns:
            Text with placeholders replaced
        """
        for key, value in context.items():
            # Handle different placeholder formats
            placeholder_patterns = [
                f"{{{{ {key} }}}}", # Jinja-style: {{ key }}
                f"[{key}]",         # Custom style: [key]
                f"${key}"           # Shell-style: $key
            ]
            
            for pattern in placeholder_patterns:
                # Convert complex values to strings
                if isinstance(value, dict) or isinstance(value, list):
                    value_str = json.dumps(value)
                else:
                    value_str = str(value)
                
                text = text.replace(pattern, value_str)
        
        return text
    
    async def _generate_media(self, media_spec, context: Dict[str, Any]) -> Optional[MediaItem]:
        """
        Generate a media item based on a specification.
        
        Args:
            media_spec: Media specification
            context: Context data
            
        Returns:
            Generated media item or None if generation failed
        """
        try:
            # Replace placeholders in prompt
            processed_prompt = self._replace_placeholders(media_spec.prompt, context)
            
            # Generate media based on type
            media_type = media_spec.media_type
            media_id = f"{media_type}-{uuid.uuid4()}"
            
            # Call appropriate service based on media type
            # This is a simplified implementation
            media_data = None
            url = None
            alt_text = processed_prompt
            
            # Create media item
            return MediaItem(
                media_id=media_id,
                media_type=media_type,
                title=f"Media for {context.get('concept_name', 'concept')}",
                description=processed_prompt,
                url=url,
                data=media_data,
                alt_text=alt_text,
                metadata={
                    "style": media_spec.style,
                    "aspect_ratio": media_spec.aspect_ratio,
                    "resolution": media_spec.resolution
                }
            )
        except Exception as e:
            logger.error(f"Error generating media: {e}")
            return None
    
    async def _generate_interactive_element(self, interactive_spec, context: Dict[str, Any]) -> Optional[InteractiveElement]:
        """
        Generate an interactive element based on a specification.
        
        Args:
            interactive_spec: Interactive element specification
            context: Context data
            
        Returns:
            Generated interactive element or None if generation failed
        """
        try:
            # Replace placeholders in prompt
            processed_prompt = self._replace_placeholders(interactive_spec.prompt, context)
            
            # Generate interactive element
            element_id = f"{interactive_spec.element_type}-{uuid.uuid4()}"
            
            # Generate element data
            response = await self.llm_service.generate_content(
                prompt=f"Create an interactive {interactive_spec.element_type} element about {processed_prompt}",
                temperature=0.7,
                max_tokens=1000
            )
            
            # Create interactive element
            return InteractiveElement(
                element_id=element_id,
                element_type=interactive_spec.element_type,
                title=f"Interactive {interactive_spec.element_type.capitalize()}",
                description=processed_prompt,
                instructions=f"Interact with this {interactive_spec.element_type}",
                data={
                    "content": response.content,
                    "parameters": interactive_spec.parameters
                }
            )
        except Exception as e:
            logger.error(f"Error generating interactive element: {e}")
            return None
    
    async def _generate_assessment_items(self, section: TemplateSection, context: Dict[str, Any]) -> List[AssessmentItem]:
        """
        Generate assessment items for a section.
        
        Args:
            section: Template section
            context: Context data
            
        Returns:
            List of assessment items
        """
        assessment_items = []
        
        try:
            # Check if there are prompts specifically for assessment items
            assessment_prompts = [p for p in section.prompts if "assessment" in p.id.lower()]
            
            for prompt in assessment_prompts:
                # Replace placeholders
                processed_prompt = self._replace_placeholders(prompt.prompt_text, context)
                
                # Generate assessment items
                response = await self.llm_service.generate_content(
                    prompt=processed_prompt,
                    system_message="Create assessment items that test understanding of the concept.",
                    temperature=0.7,
                    max_tokens=1500,
                    output_format={"items": [{"question": "string", "options": ["string"], "answer": "string or int"}]}
                )
                
                # Parse response and create assessment items
                if hasattr(response, "structured_output") and "items" in response.structured_output:
                    for idx, item_data in enumerate(response.structured_output["items"]):
                        item_id = f"assessment-{uuid.uuid4()}"
                        question = item_data.get("question", f"Question {idx+1}")
                        options = item_data.get("options", [])
                        
                        # Handle different answer formats
                        correct_answer = item_data.get("answer")
                        if isinstance(correct_answer, str) and correct_answer.isdigit():
                            correct_answer = int(correct_answer)
                        
                        item = AssessmentItem(
                            item_id=item_id,
                            item_type="multiple_choice" if options else "open_ended",
                            question=question,
                            options=options if options else None,
                            correct_answer=correct_answer,
                            explanation=item_data.get("explanation", ""),
                            difficulty=context.get("difficulty", ContentDifficulty.INTERMEDIATE),
                            tags=[context.get("concept_name", "")]
                        )
                        assessment_items.append(item)
        
        except Exception as e:
            logger.error(f"Error generating assessment items: {e}")
        
        return assessment_items
    
    def _combine_sections(self, sections: List[ContentSection]) -> str:
        """
        Combine sections into a complete content document.
        
        Args:
            sections: List of content sections
            
        Returns:
            Combined content
        """
        # Sort sections by order
        sorted_sections = sorted(sections, key=lambda s: s.order)
        
        # Combine section content
        parts = []
        for section in sorted_sections:
            parts.append(f"# {section.title}\n\n{section.content}")
            
            # Add subsections if any
            if section.subsections:
                for subsection in section.subsections:
                    parts.append(f"## {subsection.title}\n\n{subsection.content}")
        
        return "\n\n".join(parts)