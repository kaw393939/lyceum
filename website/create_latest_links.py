#!/usr/bin/env python3
"""
Helper script to create 'latest' symlinks for assets.
This makes it easier to reference the most recent version of each asset.
"""

import os
import shutil
from pathlib import Path
import glob
from datetime import datetime

def main():
    """Create latest links for all asset types."""
    print("Creating latest links for multimedia assets...")
    
    # Project root directory
    project_root = Path(__file__).parent
    
    # Create directories if they don't exist
    images_dir = project_root / "static" / "images"
    audio_dir = project_root / "static" / "audio"
    content_dir = project_root / "content"
    
    # Ensure all directories exist
    for directory in [images_dir, audio_dir, content_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Create latest image links
    print("\nCreating image links...")
    
    # First, find the basic vision image and create a link
    vision_images = list(images_dir.glob("lyceum_vision_*_0.png"))
    if vision_images:
        # Sort by timestamp (newer first)
        vision_images.sort(reverse=True)
        latest_vision = vision_images[0]
        vision_link = images_dir / "lyceum_vision_latest.png"
        
        # Create link (using copy for better compatibility)
        if vision_link.exists():
            vision_link.unlink()
        shutil.copy2(latest_vision, vision_link)
        print(f"Created link: {vision_link.name} -> {latest_vision.name}")
    
    # Define page types and extend with numbered variants
    page_types = ["vision", "technical", "business", "agile"]
    all_types = []
    for page in page_types:
        all_types.append(page)
        for i in range(1, 4):
            all_types.append(f"{page}_{i}")
    
    # Create links for all page types
    for page_type in all_types:
        matching_images = list(images_dir.glob(f"lyceum_{page_type}_*.png"))
        
        # Skip if no matching images or if we're trying to create a link that already exists as a source
        if matching_images and not any("latest" in img.name for img in matching_images):
            # Filter out any "latest" files to prevent circular references
            valid_images = [img for img in matching_images if "latest" not in img.name]
            
            if valid_images:
                valid_images.sort(reverse=True)
                latest_image = valid_images[0]
                link_name = images_dir / f"lyceum_{page_type}_latest.png"
                
                # Check if the link already exists and points to the right file
                if link_name.exists():
                    link_name.unlink()
                
                try:
                    shutil.copy2(latest_image, link_name)
                    print(f"Created link: {link_name.name} -> {latest_image.name}")
                except FileNotFoundError:
                    print(f"Warning: Could not find source file {latest_image}")
                except Exception as e:
                    print(f"Error creating link for {page_type}: {e}")
    
    # Create latest audio links
    print("\nCreating audio links...")
    audio_files = list(audio_dir.glob("lyceum_introduction_*.mp3"))
    if audio_files:
        # Exclude any files named "latest"
        audio_files = [f for f in audio_files if "latest" not in f.name]
        if audio_files:
            audio_files.sort(reverse=True)
            latest_audio = audio_files[0]
            audio_link = audio_dir / "lyceum_introduction_latest.mp3"
            
            if audio_link.exists():
                audio_link.unlink()
                
            try:
                shutil.copy2(latest_audio, audio_link)
                print(f"Created link: {audio_link.name} -> {latest_audio.name}")
            except Exception as e:
                print(f"Error creating audio link: {e}")
    
    # Create latest content links
    print("\nCreating content links...")
    content_types = ["introduction", "vision", "technical", "business", "agile", "contact"]
    
    for content_type in content_types:
        # Try markdown files first, then text files
        matching_files = list(content_dir.glob(f"lyceum_{content_type}_*.md"))
        if not matching_files:
            matching_files = list(content_dir.glob(f"lyceum_{content_type}_*.txt"))
        
        if matching_files:
            # Exclude any files named "latest"
            matching_files = [f for f in matching_files if "latest" not in f.name]
            if matching_files:
                matching_files.sort(reverse=True)
                latest_file = matching_files[0]
                extension = latest_file.suffix
                content_link = content_dir / f"lyceum_{content_type}_latest{extension}"
                
                if content_link.exists():
                    content_link.unlink()
                    
                try:
                    shutil.copy2(latest_file, content_link)
                    print(f"Created link: {content_link.name} -> {latest_file.name}")
                except Exception as e:
                    print(f"Error creating content link for {content_type}: {e}")
            else:
                print(f"No valid source files found for {content_type}")
        else:
            print(f"No matching files found for {content_type}")
    
    print("\nLatest links created successfully!")

if __name__ == "__main__":
    main()