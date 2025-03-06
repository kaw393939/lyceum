#!/usr/bin/env python3
"""
Regenerate assets for Lyceum website.
"""

import os
import argparse
import logging
import shutil
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables 
load_dotenv()

def generate_audio():
    """Generate audio assets."""
    from utils.audio_generator import select_script, generate_audio
    
    # Get OpenAI API key
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        logger.warning("OPENAI_API_KEY not set. Cannot generate audio.")
        return None
    
    # Select script (you can change the index for different scripts)
    script = select_script(index=1)  # Using script index 1
    logger.info(f"Selected script: {script[:50]}...")
    
    # Generate audio with fable voice (high quality)
    logger.info("Generating audio with fable voice (high-definition)...")
    audio_path = generate_audio(
        script=script,
        openai_api_key=openai_api_key,
        voice="fable",  # Use the fable voice as requested
        model="tts-1-hd"  # Using the high-definition model
    )
    
    if audio_path:
        logger.info(f"Audio generated successfully: {audio_path}")
    else:
        logger.error("Failed to generate audio")
    
    return audio_path

def generate_images():
    """Generate image assets."""
    from utils.image_generator import ImageGenerator
    
    # Get OpenAI API key
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        logger.warning("OPENAI_API_KEY not set. Cannot generate images.")
        return []
    
    # Initialize image generator
    generator = ImageGenerator(api_key=openai_api_key)
    
    # Generate logo
    logger.info("Generating Lyceum logo...")
    logo_prompt = """
    A minimalist logo design for an AI educational system named 'Lyceum'. 
    The logo should incorporate elements of both ancient wisdom (like an owl or Greek column) 
    and modern AI (like neural networks or circuit patterns). 
    The design should be elegant, balanced, and sophisticated. 
    Use a color palette of deep blue, gold accents, and white.
    Make it clean and professional, suitable for a modern educational technology brand.
    """
    
    logo_paths = generator.generate_image(
        prompt=logo_prompt,
        filename="lyceum_logo",
        size="1024x1024",
        quality="standard",
        style="vivid",
        n=1
    )
    
    # Generate dialogue visualization
    logger.info("Generating dialogue visualization...")
    dialogue_prompt = """
    A visualization of a Socratic dialogue in the Lyceum educational platform. 
    Show a learner engaged in conversation with an AI mentor, with visual elements 
    representing the flow of ideas and questions. Include subtle Greek philosophical 
    imagery combined with modern digital elements. 
    Use a palette of deep blues, purples, and gold accents.
    Make it visually appealing and sophisticated, suitable for an educational technology website.
    """
    
    dialogue_paths = generator.generate_image(
        prompt=dialogue_prompt,
        filename="lyceum_dialogue",
        size="1024x1024",
        quality="standard",
        style="vivid",
        n=1
    )
    
    # Generate knowledge graph visualization
    logger.info("Generating knowledge graph visualization...")
    knowledge_prompt = """
    A visualization of an AI-powered knowledge graph for the Lyceum educational platform. 
    Show an elegant, interconnected network of concepts with nodes and edges, 
    representing how knowledge is structured and connected. 
    Use a palette of deep blues, teals, and gold highlights against a dark background.
    Make it clean, modern, and sophisticated, clearly showing the structure of knowledge.
    """
    
    knowledge_paths = generator.generate_image(
        prompt=knowledge_prompt,
        filename="lyceum_knowledge",
        size="1024x1024",
        quality="standard",
        style="vivid",
        n=1
    )
    
    all_paths = logo_paths + dialogue_paths + knowledge_paths
    logger.info(f"Generated {len(all_paths)} images")
    
    return all_paths

def generate_content():
    """Generate website content."""
    # This function would generate additional content if needed
    logger.info("Content generation not implemented yet")
    return None

def generate_page_content():
    """Generate rich content for all website pages."""
    from utils.content_generator import ContentGenerator
    
    # Get Anthropic API key
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        logger.warning("ANTHROPIC_API_KEY not set. Cannot generate page content.")
        logger.warning("Using placeholder content only.")
        return None
    
    try:
        # Initialize content generator
        generator = ContentGenerator(api_key=anthropic_api_key)
        
        # Generate content for all pages
        logger.info("Generating content for all website pages...")
        content = generator.generate_all_page_content()
        
        # Log results
        for page, page_content in content.items():
            content_length = len(page_content) if page_content else 0
            logger.info(f"Generated {content_length} characters of content for {page} page")
        
        return content
    except Exception as e:
        logger.error(f"Error generating page content: {str(e)}")
        return None

def generate_page_images():
    """Generate high-quality images for all website pages."""
    from utils.image_generator import ImageGenerator
    
    # Get OpenAI API key
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        logger.warning("OPENAI_API_KEY not set. Cannot generate page images.")
        return None
    
    try:
        # Initialize image generator
        generator = ImageGenerator(api_key=openai_api_key)
        
        # Generate images for all pages
        logger.info("Generating images for all website pages...")
        page_images = generator.generate_page_images()
        
        # Log results
        for page, paths in page_images.items():
            logger.info(f"Generated {len(paths)} images for {page} page")
        
        return page_images
    except Exception as e:
        logger.error(f"Error generating page images: {str(e)}")
        return None

def generate_fallback_assets():
    """Generate fallback assets when API keys are not available."""
    import shutil
    from pathlib import Path
    
    logger.info("Generating fallback assets...")
    
    # Paths
    project_root = Path(__file__).parent
    static_dir = project_root / "static"
    images_dir = static_dir / "images"
    content_dir = project_root / "content"
    
    # Ensure directories exist
    images_dir.mkdir(parents=True, exist_ok=True)
    content_dir.mkdir(parents=True, exist_ok=True)
    
    # Create fallback images if needed
    if not any(images_dir.glob("lyceum_vision_*.png")):
        logger.info("Creating fallback vision image")
        from PIL import Image, ImageDraw, ImageFont
        
        # Create a basic image with text
        img = Image.new('RGB', (1024, 768), color=(26, 35, 126))
        draw = ImageDraw.Draw(img)
        
        # Add text
        try:
            font = ImageFont.truetype("arial.ttf", 60)
        except:
            font = ImageFont.load_default()
            
        draw.text((512, 384), "Lyceum Vision", fill=(255, 215, 0), anchor="mm", font=font)
        
        # Save the image
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        img_path = images_dir / f"lyceum_vision_{timestamp}_0.png"
        img.save(img_path)
        logger.info(f"Created fallback image: {img_path}")
    
    # Create fallback content for each page
    pages = ["vision", "technical", "business", "agile", "contact"]
    for page in pages:
        if not any(content_dir.glob(f"lyceum_{page}_*.md")):
            logger.info(f"Creating fallback content for {page} page")
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            content_path = content_dir / f"lyceum_{page}_{timestamp}.md"
            
            # Basic content for each page
            content = f"""
# Lyceum {page.capitalize()} 

This is placeholder content for the {page} page of the Lyceum educational platform.

## Overview

Lyceum is an innovative educational platform that combines the philosophical depth of ancient learning centers with cutting-edge AI technology.

## Key Points

- Knowledge Graph: Interconnected concepts form a web of understanding
- Socratic Dialogue: Learn through conversation and questioning
- Personalized Mentorship: Guidance tailored to individual needs
- Adaptive Content: Materials that evolve with your progress
"""
            
            with open(content_path, "w") as f:
                f.write(content)
            logger.info(f"Created fallback content: {content_path}")

def update_template_styles():
    """Update templates with additional CSS for multimedia content."""
    from pathlib import Path
    
    logger.info("Updating template styles for multimedia content...")
    
    # Path to the base template
    base_template_path = Path(__file__).parent / "templates" / "base.html"
    
    if not base_template_path.exists():
        logger.warning(f"Base template not found at {base_template_path}")
        return
    
    # Read the current template
    with open(base_template_path, "r") as f:
        content = f.read()
    
    # Check if we already added the multimedia styles
    if "multimedia-content" in content and "hot-reload" in content:
        logger.info("Multimedia styles and hot-reload already present in template")
        return
    
    # Add multimedia CSS and hot-reload to the head section
    multimedia_css = """
    <!-- Multimedia content styles -->
    <style>
        .multimedia-content {
            margin: 40px 0;
        }
        
        .page-image {
            width: 100%;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            margin: 20px 0;
            transition: transform 0.3s ease;
        }
        
        .page-image:hover {
            transform: scale(1.02);
        }
        
        .image-caption {
            font-size: 0.9rem;
            color: #666;
            text-align: center;
            margin-top: 8px;
        }
        
        .content-section {
            margin: 30px 0;
        }
        
        .content-section h2 {
            color: #1a237e;
            border-bottom: 1px solid #e0e0e0;
            padding-bottom: 8px;
            margin-bottom: 20px;
        }
        
        .highlight-box {
            background-color: rgba(26, 35, 126, 0.05);
            border-left: 4px solid #1a237e;
            padding: 20px;
            margin: 20px 0;
            border-radius: 4px;
        }
        
        blockquote {
            background-color: rgba(218, 165, 32, 0.1);
            border-left: 4px solid #daa520;
            padding: 15px;
            margin: 20px 0;
            font-style: italic;
        }
        
        .two-column {
            display: flex;
            flex-wrap: wrap;
            gap: 30px;
            margin: 30px 0;
        }
        
        .two-column > div {
            flex: 1;
            min-width: 300px;
        }
        
        /* Notification for hot reload */
        .hot-reload-indicator {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background-color: rgba(26, 35, 126, 0.8);
            color: white;
            padding: 10px 15px;
            border-radius: 4px;
            font-size: 0.8rem;
            z-index: 1000;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
            transition: opacity 0.5s ease;
            opacity: 0;
            pointer-events: none;
        }
        
        .hot-reload-indicator.visible {
            opacity: 1;
        }
        
        @media (max-width: 768px) {
            .two-column > div {
                flex: 100%;
            }
        }
    </style>
    
    <!-- Hot reload script -->
    <script>
        // Check for changes every 2 seconds
        const AUTO_REFRESH_INTERVAL = 2000;
        let lastModified = new Date().getTime();
        
        // Create indicator element
        document.addEventListener('DOMContentLoaded', function() {
            const indicator = document.createElement('div');
            indicator.className = 'hot-reload-indicator';
            indicator.textContent = 'Page Updated';
            document.body.appendChild(indicator);
            
            // Store the original favicon
            const originalFavicon = document.querySelector('link[rel="icon"]')?.href || '';
            
            // Check for server changes
            function checkForChanges() {
                fetch(window.location.href, { method: 'HEAD' })
                    .then(response => {
                        const serverLastModified = response.headers.get('Last-Modified');
                        if (serverLastModified) {
                            const serverTime = new Date(serverLastModified).getTime();
                            
                            // If the server has a newer version, reload
                            if (serverTime > lastModified) {
                                console.log('Changes detected, reloading page...');
                                lastModified = serverTime;
                                
                                // Show the indicator
                                indicator.classList.add('visible');
                                
                                // Hide indicator after 3 seconds
                                setTimeout(() => {
                                    indicator.classList.remove('visible');
                                }, 3000);
                                
                                // Reload page
                                window.location.reload();
                            }
                        }
                    })
                    .catch(error => console.error('Error checking for changes:', error));
            }
            
            // Start checking for changes
            setInterval(checkForChanges, AUTO_REFRESH_INTERVAL);
        });
    </script>
    """
    
    # Insert before the closing head tag
    if "</head>" in content:
        new_content = content.replace("</head>", f"{multimedia_css}\n</head>")
        
        # Write back the updated template
        with open(base_template_path, "w") as f:
            f.write(new_content)
        
        logger.info("Successfully updated base template with multimedia styles")
    else:
        logger.warning("Could not find </head> tag in base template")

def main():
    """Main function to regenerate assets."""
    parser = argparse.ArgumentParser(description="Regenerate assets for Lyceum website")
    parser.add_argument("--audio", action="store_true", help="Generate audio assets")
    parser.add_argument("--images", action="store_true", help="Generate image assets")
    parser.add_argument("--content", action="store_true", help="Generate content assets")
    parser.add_argument("--page-images", action="store_true", help="Generate page-specific images")
    parser.add_argument("--page-content", action="store_true", help="Generate page-specific content")
    parser.add_argument("--fallback", action="store_true", help="Generate fallback assets when APIs unavailable")
    parser.add_argument("--update-templates", action="store_true", help="Update templates with multimedia styles")
    parser.add_argument("--all", action="store_true", help="Generate all assets")
    
    args = parser.parse_args()
    
    # If no specific arguments are provided, show help
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    # Generate requested assets
    if args.audio or args.all:
        generate_audio()
        
    if args.images or args.all:
        generate_images()
        
    if args.content or args.all:
        generate_content()
    
    if args.page_images or args.all:
        generate_page_images()
        
    if args.page_content or args.all:
        generate_page_content()
        
    if args.fallback or args.all:
        generate_fallback_assets()
        
    if args.update_templates or args.all:
        update_template_styles()
    
if __name__ == "__main__":
    main()