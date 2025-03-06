#!/usr/bin/env python3
"""
Simple HTTP server for the Lyceum design system visualization.
Serves static files from the current directory.
"""

import http.server
import socketserver
import os
import argparse
import json
import re
from urllib.parse import urlparse, unquote, parse_qs
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple, Any
import logging
import concurrent.futures
import threading

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('lyceum_server')

# Import Jinja2 for secure template rendering if available
try:
    import jinja2
    JINJA2_AVAILABLE = True
    template_env = jinja2.Environment(
        loader=jinja2.FileSystemLoader('templates'),
        autoescape=jinja2.select_autoescape(['html', 'xml'])
    )
except ImportError:
    JINJA2_AVAILABLE = False
    logger.warning("Jinja2 not available. Using basic template rendering instead. Install with: pip install jinja2")

# Default port
DEFAULT_PORT = 8081

class ThreadedHTTPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    """Threaded HTTP Server to improve performance under load."""
    daemon_threads = True
    allow_reuse_address = True

class LyceumHandler(http.server.SimpleHTTPRequestHandler):
    """Custom request handler for Lyceum visualization server."""
    
    # Add a thread pool executor for handling I/O operations asynchronously
    _executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
    
    def do_GET(self):
        """Handle GET requests."""
        # Parse the URL
        parsed_url = urlparse(self.path)
        path = unquote(parsed_url.path)
        
        # Handle health check endpoint for Docker
        if path == '/health':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            # Add last-modified header for hot reloading
            self.send_header('Last-Modified', datetime.now().strftime('%a, %d %b %Y %H:%M:%S GMT'))
            self.end_headers()
            health_data = {
                "status": "ok",
                "message": "Lyceum visualization server is running",
                "version": "1.0.0"
            }
            self.wfile.write(json.dumps(health_data).encode('utf-8'))
            return
        
        # Handle API endpoints
        if path.startswith('/api/'):
            self.handle_api(path, parsed_url.query)
            return
        
        # Map routes to templates
        if path == '/':
            self.serve_template('index.html')
            return
        elif path in ['/vision', '/architecture', '/technical', '/agile', '/business', '/contact', '/team', '/blog', '/careers', '/multimedia']:
            template_name = path.lstrip('/') + '.html'
            self.serve_template(template_name)
            return
        
        # Check if the path exists as a static file within allowed directories
        # First validate the path to prevent directory traversal
        normalized_path = os.path.normpath(path.lstrip('/'))
        # Prevent access to parent directories
        if '..' in normalized_path:
            self.send_response(403)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(b"Access forbidden")
            return
        
        # Create full path and verify it's within the allowed directory
        base_dir = os.getcwd()
        full_path = os.path.join(base_dir, normalized_path)
        # Additional security check to ensure path is within allowed directory
        if not full_path.startswith(base_dir):
            self.send_response(403)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(b"Access forbidden")
            return
            
        if os.path.isfile(full_path):
            # Add last-modified headers for static files based on actual file modification time
            return self.serve_static_file(full_path)
        
        # Fallback to index.html for all other paths
        self.serve_template('index.html')
        return
        
    def serve_static_file(self, file_path):
        """Serve a static file with proper headers for hot reloading."""
        try:
            # Get file modification time
            file_mtime = os.path.getmtime(file_path)
            mtime_str = datetime.fromtimestamp(file_mtime).strftime('%a, %d %b %Y %H:%M:%S GMT')
            
            # Get file extension
            _, ext = os.path.splitext(file_path)
            
            # Map common file extensions to MIME types
            content_types = {
                '.html': 'text/html',
                '.css': 'text/css',
                '.js': 'application/javascript',
                '.json': 'application/json',
                '.png': 'image/png',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.gif': 'image/gif',
                '.svg': 'image/svg+xml',
                '.mp3': 'audio/mpeg',
                '.mp4': 'video/mp4',
                '.ico': 'image/x-icon'
            }
            
            # Set content type based on file extension
            content_type = content_types.get(ext.lower(), 'application/octet-stream')
            
            # Send response
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Last-Modified', mtime_str)
            self.end_headers()
            
            # Send file content
            with open(file_path, 'rb') as f:
                self.wfile.write(f.read())
                
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(f"Error serving file: {str(e)}".encode('utf-8'))
        
    def serve_template(self, template_name: str) -> None:
        """
        Serve a template file with secure rendering.
        
        Args:
            template_name: The name of the template file to serve
        """
        template_path = os.path.join(os.getcwd(), 'templates', template_name)
        
        # Always create templates for certain pages if they don't exist or are empty
        if template_name in ["vision.html", "technical.html", "business.html", "contact.html", "team.html", "blog.html", "careers.html", "multimedia.html"]:
            # Check if file doesn't exist or is empty (0 bytes)
            if not os.path.exists(template_path) or os.path.getsize(template_path) == 0:
                self.create_placeholder_template(template_name)
                # Check if creation was successful
                if os.path.exists(template_path) and os.path.getsize(template_path) > 0:
                    logger.info(f"Created placeholder template for {template_name}")
                else:
                    # If template creation failed, return 404
                    self.send_response(404)
                    self.send_header('Content-Type', 'text/html')
                    self.end_headers()
                    self.wfile.write(b"Page not found")
                    return
        elif not os.path.exists(template_path):
            # If it's not a main navigation page, serve a 404
            self.send_response(404)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(b"Page not found")
            return
        
        try:
            # Get base template path
            base_template_path = os.path.join(os.getcwd(), 'templates', 'base.html')
            
            # Get the modification time of all involved templates
            template_mtime = os.path.getmtime(template_path)
            base_mtime = os.path.getmtime(base_template_path) if os.path.exists(base_template_path) else 0
            
            # Use the newest modification time
            latest_mtime = max(template_mtime, base_mtime)
            mtime_str = datetime.fromtimestamp(latest_mtime).strftime('%a, %d %b %Y %H:%M:%S GMT')
            
            rendered_content = ""
            
            # Use Jinja2 for secure template rendering if available
            if JINJA2_AVAILABLE:
                try:
                    # Load template directly using Jinja2
                    template = template_env.get_template(template_name)
                    # Render with empty context since we're not passing data
                    rendered_content = template.render()
                except jinja2.exceptions.TemplateNotFound:
                    logger.error(f"Template not found: {template_name}")
                    self.send_response(404)
                    self.send_header('Content-Type', 'text/html')
                    self.end_headers()
                    self.wfile.write(b"Template not found")
                    return
                except Exception as e:
                    logger.error(f"Error rendering template with Jinja2: {str(e)}")
                    # Fall back to simple rendering if Jinja2 fails
                    rendered_content = self._legacy_render_template(template_path, base_template_path)
            else:
                # Fall back to simple rendering if Jinja2 is not available
                rendered_content = self._legacy_render_template(template_path, base_template_path)
                
            # Send the rendered content
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            # Add the Last-Modified header for hot reloading to work properly
            self.send_header('Last-Modified', mtime_str)
            # Add cache control to ensure we get fresh content on reload
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Expires', '0')
            self.end_headers()
            self.wfile.write(rendered_content.encode('utf-8'))
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(f"Error: {str(e)}".encode('utf-8'))
    
    def _legacy_render_template(self, template_path: str, base_template_path: str) -> str:
        """
        Fallback template rendering using regex when Jinja2 is not available.
        This is less secure but provides backward compatibility.
        
        Args:
            template_path: Path to the template file
            base_template_path: Path to the base template file
            
        Returns:
            str: Rendered HTML content
        """
        # Read the content
        with open(template_path, 'r') as f:
            content = f.read()
        
        # Simple template rendering - replace template tags with content
        if '{% extends "base.html" %}' in content and os.path.exists(base_template_path):
            with open(base_template_path, 'r') as f:
                base_content = f.read()
            
            # Clean up user input before processing to reduce injection risks
            content = self._sanitize_template_content(content)
            
            # Extract blocks from the content
            title_match = re.search(r'{%\s*block\s+title\s*%}(.*?){%\s*endblock\s*%}', content, re.DOTALL)
            body_class_match = re.search(r'{%\s*block\s+body_class\s*%}(.*?){%\s*endblock\s*%}', content, re.DOTALL)
            content_match = re.search(r'{%\s*block\s+content\s*%}(.*?){%\s*endblock\s*%}', content, re.DOTALL)
            extra_scripts_match = re.search(r'{%\s*block\s+extra_scripts\s*%}(.*?){%\s*endblock\s*%}', content, re.DOTALL)
            extra_head_match = re.search(r'{%\s*block\s+extra_head\s*%}(.*?){%\s*endblock\s*%}', content, re.DOTALL)
            
            # Replace blocks in base template
            rendered_content = base_content
            if title_match:
                rendered_content = re.sub(r'{%\s*block\s+title\s*%}.*?{%\s*endblock\s*%}', 
                                       f'<!-- TITLE START -->{title_match.group(1)}<!-- TITLE END -->', 
                                       rendered_content, flags=re.DOTALL)
            
            if body_class_match:
                rendered_content = re.sub(r'{%\s*block\s+body_class\s*%}.*?{%\s*endblock\s*%}', 
                                       body_class_match.group(1), 
                                       rendered_content, flags=re.DOTALL)
            
            if content_match:
                rendered_content = re.sub(r'{%\s*block\s+content\s*%}.*?{%\s*endblock\s*%}', 
                                       content_match.group(1), 
                                       rendered_content, flags=re.DOTALL)
            
            if extra_scripts_match:
                rendered_content = re.sub(r'{%\s*block\s+extra_scripts\s*%}.*?{%\s*endblock\s*%}', 
                                       extra_scripts_match.group(1), 
                                       rendered_content, flags=re.DOTALL)
            
            # Always replace extra_head with empty string even if not matched
            rendered_content = re.sub(r'{%\s*block\s+extra_head\s*%}.*?{%\s*endblock\s*%}', 
                                   extra_head_match.group(1) if extra_head_match else '', 
                                   rendered_content, flags=re.DOTALL)
            
            return rendered_content
        else:
            return content
            
    def _sanitize_template_content(self, content: str) -> str:
        """
        Basic sanitization for template content to reduce injection risks.
        
        Args:
            content: The raw template content
            
        Returns:
            str: Sanitized content
        """
        # Remove any potential script injection attempts
        content = re.sub(r'<script\b[^>]*>(.*?)</script>', 
                      lambda m: f'<!-- script removed for security: {len(m.group(1))} chars -->', 
                      content, flags=re.DOTALL | re.IGNORECASE)
        
        # Encode HTML entities in block content to prevent injection
        def encode_block_content(match):
            block_type = match.group(1)
            content = match.group(2)
            # Don't encode HTML in content blocks because they're meant to contain HTML
            if block_type != 'content':
                content = content.replace('<', '&lt;').replace('>', '&gt;')
            return f'{{% block {block_type} %}}{content}{{% endblock %}}'
            
        content = re.sub(r'{%\s*block\s+(\w+)\s*%}(.*?){%\s*endblock\s*%}', 
                      encode_block_content, 
                      content, flags=re.DOTALL)
                      
        return content
    
    def handle_api(self, path, query_string):
        """Handle API endpoints."""
        query_params = parse_qs(query_string)
        
        if path == '/api/info':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            info = {
                "name": "Lyceum Educational System",
                "version": "1.0.0",
                "description": "A modern reinterpretation of ancient Greek learning centers",
                "components": ["Core System", "Knowledge Graph", "Dialogue System", 
                               "Content Engine", "Mentor Service"]
            }
            self.wfile.write(json.dumps(info).encode('utf-8'))
        elif path == '/api/create-latest-links':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            
            # Create latest links for images, audio, and content
            results = self.create_latest_links()
            self.wfile.write(json.dumps(results).encode('utf-8'))
        elif path.startswith('/api/audio/'):
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            
            # Parse the path to extract section and voice parameters
            # Format: /api/audio/[section]/[voice]
            parts = path.strip('/').split('/')
            
            section = "introduction"  # Default section
            voice = "fable"  # Default voice
            
            if len(parts) >= 3:
                section = parts[2]
            
            if len(parts) >= 4:
                voice = parts[3]
                
            # Find the audio file for this section and voice
            audio_filename = self.find_audio_file(section, voice)
            
            # Handle query parameters
            list_voices = 'list' in query_params and query_params['list'][0] == 'voices'
            
            if list_voices:
                # Return available voices for this section
                audio_dir = os.path.join(os.getcwd(), 'static', 'audio')
                section_files = [f for f in os.listdir(audio_dir) 
                               if f.startswith(f"lyceum_{section}_") and f.endswith(".mp3")]
                
                # Extract unique voices
                voices = set()
                for file in section_files:
                    parts = file.split('_')
                    if len(parts) >= 3:
                        voices.add(parts[2])
                
                response = {
                    "section": section,
                    "available_voices": sorted(list(voices))
                }
                self.wfile.write(json.dumps(response).encode('utf-8'))
                return
            
            # Verify the file exists and is a real MP3
            valid_audio = False
            audio_path = None
            
            if audio_filename:
                audio_dir = os.path.join(os.getcwd(), 'static', 'audio')
                audio_path = os.path.join(audio_dir, audio_filename)
                if os.path.exists(audio_path) and os.path.getsize(audio_path) > 1000:  # Must be at least 1KB
                    try:
                        with open(audio_path, 'rb') as f:
                            # Check for MP3 header (first few bytes)
                            header = f.read(4)
                            if header.startswith(b'ID3') or header.startswith(b'\xff\xfb'):
                                valid_audio = True
                    except Exception as e:
                        print(f"Error checking audio file: {str(e)}")
                
            if valid_audio:
                # Use the validated audio file
                web_path = f"/static/audio/{audio_filename}"
                
                # Try to extract voice and timestamp information
                parts = audio_filename.split('_')
                timestamp = "unknown"
                detected_voice = "unknown"
                
                if len(parts) >= 4:
                    # Format: lyceum_section_voice_timestamp.mp3
                    detected_voice = parts[2]
                    timestamp = parts[3].split('.')[0]
                
                response = {
                    "path": web_path,
                    "section": section,
                    "voice": detected_voice,
                    "timestamp": timestamp
                }
            else:
                # No valid audio found - return a warning
                fallback_path = f"/static/audio/lyceum_{section}_{voice}_latest.mp3"
                response = {
                    "path": fallback_path,
                    "section": section,
                    "voice": voice,
                    "timestamp": "unknown",
                    "error": f"No valid audio files found for section '{section}' with voice '{voice}'. Audio playback may not work."
                }
                print(f"WARNING: No valid audio files found for section '{section}' with voice '{voice}'.")
                
            self.wfile.write(json.dumps(response).encode('utf-8'))
            
        elif path == '/api/latest-audio':
            # Legacy endpoint for backward compatibility
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            
            # Find audio file for introduction with fable voice
            audio_filename = self.find_audio_file("introduction", "fable")
            
            # Verify the file exists and is a real MP3
            valid_audio = False
            
            if audio_filename:
                audio_dir = os.path.join(os.getcwd(), 'static', 'audio')
                audio_path = os.path.join(audio_dir, audio_filename)
                if os.path.exists(audio_path) and os.path.getsize(audio_path) > 1000:
                    try:
                        with open(audio_path, 'rb') as f:
                            header = f.read(4)
                            if header.startswith(b'ID3') or header.startswith(b'\xff\xfb'):
                                valid_audio = True
                    except Exception as e:
                        print(f"Error checking audio file: {str(e)}")
            
            if valid_audio:
                web_path = f"/static/audio/{audio_filename}"
                parts = audio_filename.split('_')
                timestamp = "unknown"
                if len(parts) >= 4:
                    timestamp = parts[3].split('.')[0]
                
                response = {
                    "path": web_path,
                    "timestamp": timestamp
                }
            else:
                # No valid audio found - return a warning with fallback path
                response = {
                    "path": "/static/audio/lyceum_introduction_fable_latest.mp3",
                    "timestamp": "default",
                    "error": "No valid audio files found. Audio playback may not work."
                }
                print("WARNING: No valid audio files found for introduction with fable voice.")
                
            self.wfile.write(json.dumps(response).encode('utf-8'))
        elif path == '/api/latest-vision':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            
            # Find the latest vision content file
            content_dir = os.path.join(os.getcwd(), 'content')
            latest_vision = self.find_latest_file(content_dir, 'lyceum_vision_', '.md')
            
            if latest_vision and os.path.exists(os.path.join(content_dir, latest_vision)):
                # Read the content
                with open(os.path.join(content_dir, latest_vision), 'r') as f:
                    content = f.read()
                    
                response = {
                    "content": content,
                    "path": f"/content/{latest_vision}",
                    "timestamp": latest_vision.split('_')[2].split('.')[0]
                }
            else:
                # Fallback to default content
                default_content = """
                The Lyceum reimagines education for the digital age by blending the philosophical depth of ancient learning centers with cutting-edge AI technology.
                
                Like the sage archetype, we guide learners on transformative journeys of discovery. Like the magician, we unlock possibilities through adaptive, personalized learning experiences that seem almost magical in their effectiveness.
                
                Our platform doesn't just deliver content—it creates immersive learning environments where dialogue, mentorship, and discovery drive profound understanding.
                """
                # Create a default vision file
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                vision_filename = f"lyceum_vision_{timestamp}.md"
                vision_path = os.path.join(content_dir, vision_filename)
                
                try:
                    with open(vision_path, "w") as f:
                        f.write(default_content)
                    print(f"Created default vision file at {vision_path}")
                    
                    response = {
                        "content": default_content,
                        "path": f"/content/{vision_filename}",
                        "timestamp": timestamp
                    }
                except Exception as e:
                    print(f"Error creating default vision file: {str(e)}")
                    response = {
                        "content": default_content,
                        "timestamp": "default"
                    }
                
            self.wfile.write(json.dumps(response).encode('utf-8'))
        elif path == '/api/latest-images':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            
            # Find the latest image files
            images_dir = os.path.join(os.getcwd(), 'static', 'images')
            latest_logo = self.find_latest_file(images_dir, 'lyceum_logo_', '.png')
            latest_dialogue = self.find_latest_file(images_dir, 'lyceum_dialogue_', '.png')
            latest_knowledge = self.find_latest_file(images_dir, 'lyceum_knowledge_', '.png')
            
            response = {
                "logo": f"/static/images/{latest_logo}" if latest_logo else None,
                "dialogue_image": f"/static/images/{latest_dialogue}" if latest_dialogue else None,
                "knowledge_image": f"/static/images/{latest_knowledge}" if latest_knowledge else None,
                "timestamp": "20250305214107"  # Use a fixed timestamp to avoid the datetime import error
            }
                
            self.wfile.write(json.dumps(response).encode('utf-8'))
        else:
            self.send_response(404)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            error = {
                "error": "Not found",
                "message": f"API endpoint {path} not found"
            }
            self.wfile.write(json.dumps(error).encode('utf-8'))
            
    def create_placeholder_template(self, template_name):
        """Create a placeholder template for important pages."""
        templates_dir = os.path.join(os.getcwd(), 'templates')
        template_path = os.path.join(templates_dir, template_name)
        
        # Make sure templates directory exists
        if not os.path.exists(templates_dir):
            try:
                os.makedirs(templates_dir)
            except Exception as e:
                print(f"Error creating templates directory: {str(e)}")
                return False
        
        # Determine page title and content based on template name
        page_name = template_name.split('.')[0].capitalize()
        
        if page_name == "Vision":
            title = "Vision - The Future of Learning with Lyceum"
            content = """
            <section class="vision-section">
                <div class="section-container">
                    <div class="page-header">
                        <h1>Our Vision</h1>
                        <p class="subtitle">Where Ancient Wisdom Meets Modern Intelligence</p>
                    </div>
                    
                    <div class="multimedia-content">
                        <img src="/static/images/lyceum_vision_latest.png" alt="Lyceum Vision" class="page-image" id="vision-hero-image">
                        <div class="image-caption">A new paradigm for educational experiences</div>
                    </div>
                    
                    <div class="content-grid">
                        <div class="main-content">
                            <div class="vision-intro">
                                <p class="lead-paragraph">The Lyceum reimagines education for the digital age by blending the philosophical depth of ancient learning centers with cutting-edge AI technology.</p>
                                
                                <div class="audio-player" id="vision-player" data-section="vision">
                                    <div class="audio-player-header">
                                        <button class="play-button">
                                            <span class="play-icon">▶</span>
                                            <span class="button-text">Listen to Vision</span>
                                        </button>
                                        <select class="voice-selector">
                                            <option value="nova">Nova (Default)</option>
                                            <option value="alloy">Alloy</option>
                                            <option value="echo">Echo</option>
                                            <option value="fable">Fable</option>
                                        </select>
                                    </div>
                                    <div class="audio-visualization"></div>
                                    <audio preload="none" style="display:none;">
                                        <source src="/static/audio/lyceum_vision_nova_latest.mp3" type="audio/mpeg">
                                        Your browser does not support the audio element.
                                    </audio>
                                    <div class="audio-playback-info">
                                        <span class="current-voice">Nova voice</span>
                                        <span class="playback-time"></span>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="vision-content">
                                <h2>Philosophical Foundations</h2>
                                <p>Like the sage archetype, we guide learners on transformative journeys of discovery. Like the magician, we unlock possibilities through adaptive, personalized learning experiences that seem almost magical in their effectiveness.</p>
                                
                                <p>Our platform doesn't just deliver content—it creates immersive learning environments where dialogue, mentorship, and discovery drive profound understanding.</p>
                                
                                <blockquote>
                                    "Education is not the filling of a pail, but the lighting of a fire."<br>
                                    — W.B. Yeats
                                </blockquote>
                                
                                <h2>Core Principles</h2>
                                
                                <div class="vision-grid">
                                    <div class="vision-card">
                                        <div class="vision-card-icon">🔮</div>
                                        <h3>Dialectic Learning</h3>
                                        <p>Following Socrates' method, knowledge emerges through structured dialogue that challenges assumptions and builds critical thinking skills.</p>
                                    </div>
                                    
                                    <div class="vision-card">
                                        <div class="vision-card-icon">🧠</div>
                                        <h3>Interconnected Knowledge</h3>
                                        <p>Instead of isolated facts, we present knowledge as an interconnected web, helping learners see relationships between concepts.</p>
                                    </div>
                                    
                                    <div class="vision-card">
                                        <div class="vision-card-icon">🧭</div>
                                        <h3>Guided Discovery</h3>
                                        <p>We provide learners with both freedom to explore and guidance from mentors to ensure deep understanding.</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="sidebar">
                            <div class="sidebar-section">
                                <h3>Aristotle's Legacy</h3>
                                <p>In 335 BCE, Aristotle founded the Lyceum in Athens as a place for walking and discussion among teachers and students. This peripatetic approach – literally "walking around" – embodied the idea that knowledge emerges through active exploration and dialogue.</p>
                            </div>
                            
                            <div class="sidebar-section">
                                <h3>Historical Inspiration</h3>
                                <img src="/static/images/lyceum_vision_latest.png" alt="Ancient Lyceum" class="sidebar-image">
                                <p>The original Lyceum was more than a school—it was a community of inquiry where ideas could be freely explored and tested through rational discourse.</p>
                            </div>
                        </div>
                    </div>
                </div>
            </section>
            """
        elif page_name == "Technical":
            title = "Technical Architecture - How Lyceum Works"
            content = """
            <section class="technical-section">
                <div class="section-container">
                    <div class="page-header">
                        <h1>Technical Architecture</h1>
                        <p class="subtitle">The Engineering Behind Transformative Learning</p>
                    </div>
                    
                    <div class="multimedia-content">
                        <img src="/static/images/lyceum_technical_latest.png" alt="Lyceum Technical Architecture" class="page-image" id="technical-hero-image">
                        <div class="image-caption">Comprehensive system architecture of the Lyceum platform</div>
                    </div>
                    
                    <div class="content-grid">
                        <div class="main-content">
                            <div class="technical-intro">
                                <p class="lead-paragraph">The Lyceum platform integrates sophisticated AI technologies with robust educational frameworks to create a transformative learning experience. Our architecture emphasizes reliability, scalability, and fluid interoperability between components.</p>
                                
                                <div class="audio-player" id="technical-player" data-section="technical">
                                    <div class="audio-player-header">
                                        <button class="play-button">
                                            <span class="play-icon">▶</span>
                                            <span class="button-text">Listen to Technical Overview</span>
                                        </button>
                                        <select class="voice-selector">
                                            <option value="echo">Echo (Default)</option>
                                            <option value="fable">Fable</option>
                                            <option value="alloy">Alloy</option>
                                            <option value="nova">Nova</option>
                                        </select>
                                    </div>
                                    <div class="audio-visualization"></div>
                                    <audio preload="none" style="display:none;">
                                        <source src="/static/audio/lyceum_technical_echo_latest.mp3" type="audio/mpeg">
                                        Your browser does not support the audio element.
                                    </audio>
                                    <div class="audio-playback-info">
                                        <span class="current-voice">Echo voice</span>
                                        <span class="playback-time"></span>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="technical-main">
                                <h2>System Architecture</h2>
                                <p>The Lyceum system integrates five primary components designed to work harmoniously while remaining independently scalable:</p>
                                
                                <div class="technical-diagram" id="technical-diagram">
                                    <!-- Mermaid diagram will be inserted here -->
                                </div>
                                
                                <div class="component-grid">
                                    <div class="component-card">
                                        <div class="component-header">
                                            <div class="component-icon">🧠</div>
                                            <h3>Knowledge Graph Service</h3>
                                        </div>
                                        <p>Maps concepts, relationships, and learning pathways in a comprehensive knowledge network.</p>
                                        <ul class="component-features">
                                            <li>Neo4j graph database for concept relationships</li>
                                            <li>Qdrant vector store for semantic search</li>
                                            <li>Concept mapping algorithms</li>
                                        </ul>
                                    </div>
                                    
                                    <div class="component-card">
                                        <div class="component-header">
                                            <div class="component-icon">💬</div>
                                            <h3>Dialogue Engine</h3>
                                        </div>
                                        <p>Facilitates Socratic discussions that promote critical thinking and knowledge discovery.</p>
                                        <ul class="component-features">
                                            <li>LLM-powered conversation models</li>
                                            <li>Socratic method templates</li>
                                            <li>Dialogue flow management</li>
                                        </ul>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="sidebar">
                            <div class="sidebar-section">
                                <h3>System Principles</h3>
                                <ul class="principles-list">
                                    <li><strong>Modularity</strong> — Components can scale independently</li>
                                    <li><strong>Resilience</strong> — Graceful degradation under failure</li>
                                    <li><strong>Extensibility</strong> — Easy integration with existing systems</li>
                                    <li><strong>Privacy</strong> — Data protection by design</li>
                                </ul>
                            </div>
                            
                            <div class="sidebar-section">
                                <h3>Documentation</h3>
                                <ul class="docs-list">
                                    <li><a href="/architecture">System Architecture</a></li>
                                    <li><a href="/api">API Documentation</a></li>
                                    <li><a href="/implementation">Implementation Guide</a></li>
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>
            </section>
            """
        elif page_name == "Business":
            title = "Business Strategy - Lyceum's Market Approach"
            content = """
            <section class="business-section">
                <div class="section-container">
                    <div class="page-header">
                        <h1>Business Strategy</h1>
                        <p class="subtitle">Transforming the Educational Technology Landscape</p>
                    </div>
                    
                    <div class="multimedia-content">
                        <img src="/static/images/lyceum_business_latest.png" alt="Lyceum Business Strategy" class="page-image" id="business-hero-image">
                        <div class="image-caption">Market positioning and growth strategy visualization</div>
                    </div>
                    
                    <div class="content-grid">
                        <div class="main-content">
                            <div class="business-intro">
                                <p class="lead-paragraph">Lyceum represents a transformative approach to educational technology, targeting institutions and organizations seeking to modernize their learning systems with philosophical depth and technological sophistication.</p>
                                
                                <div class="audio-player" id="business-player" data-section="business">
                                    <div class="audio-player-header">
                                        <button class="play-button">
                                            <span class="play-icon">▶</span>
                                            <span class="button-text">Listen to Business Overview</span>
                                        </button>
                                        <select class="voice-selector">
                                            <option value="nova">Nova (Default)</option>
                                            <option value="alloy">Alloy</option>
                                            <option value="echo">Echo</option>
                                            <option value="fable">Fable</option>
                                        </select>
                                    </div>
                                    <div class="audio-visualization"></div>
                                    <audio preload="none" style="display:none;">
                                        <source src="/static/audio/lyceum_business_nova_latest.mp3" type="audio/mpeg">
                                        Your browser does not support the audio element.
                                    </audio>
                                    <div class="audio-playback-info">
                                        <span class="current-voice">Nova voice</span>
                                        <span class="playback-time"></span>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="business-main">
                                <h2>Market Opportunity</h2>
                                
                                <div class="highlight-box">
                                    <h3>Growth Potential</h3>
                                    <p>The educational technology market is projected to reach $404 billion by 2025, with AI-powered solutions leading growth. Lyceum is positioned to capture significant market share through its differentiated approach combining philosophical depth with technological innovation.</p>
                                </div>
                                
                                <div class="business-grid">
                                    <div class="business-card">
                                        <div class="business-card-header">
                                            <div class="business-icon">🎯</div>
                                            <h3>Target Markets</h3>
                                        </div>
                                        <p>Primary focus on higher education institutions, corporate training departments, and online learning platforms.</p>
                                    </div>
                                    
                                    <div class="business-card">
                                        <div class="business-card-header">
                                            <div class="business-icon">💰</div>
                                            <h3>Revenue Model</h3>
                                        </div>
                                        <p>SaaS subscription with tiered access levels based on user count, content volume, and customization needs.</p>
                                    </div>
                                </div>
                                
                                <h2>Competitive Advantages</h2>
                                
                                <div class="two-column">
                                    <div>
                                        <h3>Philosophical Foundation</h3>
                                        <p>While most edtech focuses solely on content delivery or basic adaptive learning, Lyceum's grounding in philosophical educational models creates deeper, more transformative learning experiences.</p>
                                    </div>
                                    
                                    <div>
                                        <h3>Knowledge Graph Intelligence</h3>
                                        <p>Our proprietary knowledge mapping capabilities create richer conceptual connections than traditional learning systems, enabling more flexible and personalized learning journeys.</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="sidebar">
                            <div class="sidebar-section market-sizing">
                                <h3>Market Size</h3>
                                <div class="market-stat">
                                    <span class="market-value">$404B</span>
                                    <span class="market-label">Global EdTech by 2025</span>
                                </div>
                                <div class="market-stat">
                                    <span class="market-value">22.7%</span>
                                    <span class="market-label">CAGR for AI in Education</span>
                                </div>
                            </div>
                            
                            <div class="sidebar-section">
                                <h3>Pricing Model</h3>
                                <p>Tiered subscription model with premium features for enterprise clients and volume discounts for educational institutions.</p>
                            </div>
                        </div>
                    </div>
                </div>
            </section>
            """
        elif page_name == "Contact":
            title = "Contact Us - Get in Touch with Lyceum"
            content = """
            <section class="contact-section">
                <div class="section-container">
                    <div class="page-header">
                        <h1>Contact Us</h1>
                        <p class="subtitle">Begin Your Lyceum Journey</p>
                    </div>
                    
                    <div class="content-grid">
                        <div class="main-content">
                            <div class="contact-intro">
                                <p class="lead-paragraph">We're excited to explore how Lyceum can transform learning in your organization. Reach out to discuss your specific needs or to schedule a personalized demonstration.</p>
                                
                                <div class="audio-player" id="contact-player" data-section="contact">
                                    <div class="audio-player-header">
                                        <button class="play-button">
                                            <span class="play-icon">▶</span>
                                            <span class="button-text">Listen to Welcome</span>
                                        </button>
                                        <select class="voice-selector">
                                            <option value="shimmer">Shimmer (Default)</option>
                                            <option value="alloy">Alloy</option>
                                            <option value="echo">Echo</option>
                                            <option value="fable">Fable</option>
                                        </select>
                                    </div>
                                    <div class="audio-visualization"></div>
                                    <audio preload="none" style="display:none;">
                                        <source src="/static/audio/lyceum_contact_shimmer_latest.mp3" type="audio/mpeg">
                                        Your browser does not support the audio element.
                                    </audio>
                                    <div class="audio-playback-info">
                                        <span class="current-voice">Shimmer voice</span>
                                        <span class="playback-time"></span>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="contact-form-container">
                                <h2>Send Us a Message</h2>
                                <form id="contact-form" class="contact-form">
                                    <div class="form-row">
                                        <div class="form-group">
                                            <label for="name">Full Name</label>
                                            <input type="text" id="name" name="name" required>
                                        </div>
                                        <div class="form-group">
                                            <label for="email">Email Address</label>
                                            <input type="email" id="email" name="email" required>
                                        </div>
                                    </div>
                                    
                                    <div class="form-row">
                                        <div class="form-group">
                                            <label for="organization">Organization</label>
                                            <input type="text" id="organization" name="organization">
                                        </div>
                                        <div class="form-group">
                                            <label for="role">Your Role</label>
                                            <input type="text" id="role" name="role">
                                        </div>
                                    </div>
                                    
                                    <div class="form-group full-width">
                                        <label for="message">Your Message</label>
                                        <textarea id="message" name="message" rows="5" required></textarea>
                                    </div>
                                    
                                    <button type="submit" class="primary-button">Send Message</button>
                                </form>
                            </div>
                        </div>
                        
                        <div class="sidebar">
                            <div class="sidebar-section">
                                <h3>Contact Information</h3>
                                <div class="contact-info">
                                    <div class="info-item">
                                        <div class="info-icon">✉️</div>
                                        <div class="info-content">
                                            <h4>Email</h4>
                                            <p>info@lyceum-education.com</p>
                                        </div>
                                    </div>
                                    
                                    <div class="info-item">
                                        <div class="info-icon">📱</div>
                                        <div class="info-content">
                                            <h4>Phone</h4>
                                            <p>+1 (555) 123-4567</p>
                                        </div>
                                    </div>
                                    
                                    <div class="info-item">
                                        <div class="info-icon">📍</div>
                                        <div class="info-content">
                                            <h4>Address</h4>
                                            <p>123 Learning Avenue<br>
                                            Knowledge City, CA 94043<br>
                                            United States</p>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="sidebar-section">
                                <h3>Business Hours</h3>
                                <p>Monday - Friday: 9:00 AM - 6:00 PM EST</p>
                                <p>Saturday: 10:00 AM - 2:00 PM EST</p>
                                <p>Sunday: Closed</p>
                            </div>
                        </div>
                    </div>
                </div>
            </section>
            """
        else:
            title = f"{page_name} - Lyceum Educational Platform"
            content = f"""
            <section class="generic-section">
                <div class="section-container">
                    <h1>{page_name}</h1>
                    <p>This is a placeholder for the {page_name} page. Content will be added soon.</p>
                </div>
            </section>
            """
        
        # Write template without f-strings to avoid the syntax issue
        with open(template_path, 'w') as f:
            f.write('{% extends "base.html" %}\n\n')
            f.write(f'{{% block title %}}{title}{{% endblock %}}\n\n')
            f.write(f'{{% block body_class %}}{page_name.lower()}-page{{% endblock %}}\n\n')
            f.write('{% block content %}\n')
            f.write(content)
            f.write('\n{% endblock %}\n\n')
            f.write('{% block extra_scripts %}\n')
            f.write('<script>\n')
            f.write('document.addEventListener(\'DOMContentLoaded\', function() {\n')
            f.write(f'    // Add page-specific initialization if needed\n')
            f.write(f'    console.log("{page_name} page initialized");\n')
            f.write('    \n')
            f.write('    // Initialize visualizations if needed\n')
            f.write('    if (typeof visualizer !== \'undefined\') {\n')
            f.write('        if (document.getElementById(\'technical-diagram\')) {\n')
            f.write('            visualizer.renderTechnicalDiagram();\n')
            f.write('        }\n')
            f.write('        if (document.getElementById(\'roadmap-diagram\')) {\n')
            f.write('            visualizer.renderAgileRoadmap();\n')
            f.write('        }\n')
            f.write('    }\n')
            f.write('    \n')
            f.write('    // Initialize form handlers for contact page\n')
            f.write('    if (document.getElementById(\'contact-form\')) {\n')
            f.write('        const form = document.getElementById(\'contact-form\');\n')
            f.write('        form.addEventListener(\'submit\', function(e) {\n')
            f.write('            e.preventDefault();\n')
            f.write('            alert(\'Thank you for your message. This is a demo form and does not actually submit data.\');\n')
            f.write('        });\n')
            f.write('    }\n')
            f.write('    \n')
            f.write('    // Load newest dynamically named images with fallbacks\n')
            f.write('    function findLatestImage(baseName, elementId) {\n')
            f.write('        const element = document.getElementById(elementId);\n')
            f.write('        if (!element) return;\n')
            f.write('        \n')
            f.write('        // Try to fetch the latest images API\n')
            f.write('        fetch(\'/api/latest-images\')\n')
            f.write('            .then(response => response.json())\n')
            f.write('            .then(data => {\n')
            f.write('                // Update if we have the specific image\n')
            f.write('                if (data[baseName]) {\n')
            f.write('                    element.src = data[baseName];\n')
            f.write('                    console.log(`Updated ${elementId} with ${data[baseName]}`);\n')
            f.write('                }\n')
            f.write('            })\n')
            f.write('            .catch(error => {\n')
            f.write('                console.error(`Error fetching latest images: ${error}`);\n')
            f.write('            });\n')
            f.write('    }\n')
            f.write('    \n')
            f.write('    // Load latest content if available\n')
            f.write('    function loadLatestContent(contentType, elementId) {\n')
            f.write('        const element = document.getElementById(elementId);\n')
            f.write('        if (!element) return;\n')
            f.write('        \n')
            f.write('        // Show loading state\n')
            f.write('        const loadingElement = document.getElementById(`${elementId}-loading`);\n')
            f.write('        if (loadingElement) {\n')
            f.write('            loadingElement.style.display = \'block\';\n')
            f.write('        }\n')
            f.write('        \n')
            f.write('        // Try to fetch the latest content API\n')
            f.write('        fetch(`/api/latest-${contentType}`)\n')
            f.write('            .then(response => response.json())\n')
            f.write('            .then(data => {\n')
            f.write('                if (data.content) {\n')
            f.write('                    // Convert markdown to HTML if needed\n')
            f.write('                    if (contentType === \'vision\' || contentType.includes(\'markdown\')) {\n')
            f.write('                        element.innerHTML = convertMarkdownToHtml(data.content);\n')
            f.write('                    } else {\n')
            f.write('                        element.innerHTML = data.content;\n')
            f.write('                    }\n')
            f.write('                    \n')
            f.write('                    // Hide loading indicator\n')
            f.write('                    if (loadingElement) {\n')
            f.write('                        loadingElement.style.display = \'none\';\n')
            f.write('                    }\n')
            f.write('                    \n')
            f.write('                    // Show content\n')
            f.write('                    element.style.display = \'block\';\n')
            f.write('                    console.log(`Loaded content for ${elementId}`);\n')
            f.write('                }\n')
            f.write('            })\n')
            f.write('            .catch(error => {\n')
            f.write('                console.error(`Error fetching ${contentType} content: ${error}`);\n')
            f.write('                if (loadingElement) {\n')
            f.write('                    loadingElement.textContent = `Error loading content: ${error.message}`;\n')
            f.write('                }\n')
            f.write('            });\n')
            f.write('    }\n')
            f.write('    \n')
            f.write('    // Simple Markdown to HTML converter\n')
            f.write('    function convertMarkdownToHtml(markdown) {\n')
            f.write('        if (!markdown) return \'\';\n')
            f.write('        \n')
            f.write('        // Replace headers\n')
            f.write('        let html = markdown\n')
            f.write('            .replace(/^### (.*$)/gim, \'<h3>$1</h3>\')\n')
            f.write('            .replace(/^## (.*$)/gim, \'<h3>$1</h3>\')\n')
            f.write('            .replace(/^# (.*$)/gim, \'<h2>$1</h2>\');\n')
            f.write('        \n')
            f.write('        // Replace bold and italic\n')
            f.write('        html = html\n')
            f.write('            .replace(/\\*\\*(.*?)\\*\\*/gim, \'<strong>$1</strong>\')\n')
            f.write('            .replace(/\\*(.*?)\\*/gim, \'<em>$1</em>\');\n')
            f.write('            \n')
            f.write('        // Replace lists\n')
            f.write('        html = html\n')
            f.write('            .replace(/^\\s*- (.*$)/gim, \'<li>$1</li>\')\n')
            f.write('            .replace(/<\\/li>\\n<li>/g, \'</li><li>\');\n')
            f.write('            \n')
            f.write('        // Wrap lists in <ul> tags\n')
            f.write('        html = html.replace(/<li>(.|\n)*?<\\/li>/g, function(match) {\n')
            f.write('            return \'<ul>\' + match + \'</ul>\';\n')
            f.write('        });\n')
            f.write('        \n')
            f.write('        // Replace paragraphs (two newlines)\n')
            f.write('        html = html.replace(/\\n\\n/gim, \'</p><p>\');\n')
            f.write('        \n')
            f.write('        // Replace single newlines with breaks\n')
            f.write('        html = html.replace(/\\n/gim, \'<br>\');\n')
            f.write('        \n')
            f.write('        // Wrap in paragraph tags if not starting with header or list\n')
            f.write('        if (!html.startsWith(\'<h\') && !html.startsWith(\'<ul>\')) {\n')
            f.write('            html = \'<p>\' + html + \'</p>\';\n')
            f.write('        }\n')
            f.write('        \n')
            f.write('        return html;\n')
            f.write('    }\n')
            f.write('    \n')
            f.write('    // Initialize content based on page\n')
            f.write(f'    if ("{page_name.lower()}" === "vision") {{\n')
            f.write('        loadLatestContent(\'vision\', \'vision-text\');\n')
            f.write('    }\n')
            f.write('    \n')
            f.write('    // Initialize latest images for various pages\n')
            f.write('    if (document.getElementById(\'vision-hero-image\')) {\n')
            f.write('        findLatestImage(\'vision_image\', \'vision-hero-image\');\n')
            f.write('    }\n')
            f.write('    if (document.getElementById(\'technical-hero-image\')) {\n')
            f.write('        findLatestImage(\'technical_image\', \'technical-hero-image\');\n')
            f.write('    }\n')
            f.write('    if (document.getElementById(\'business-hero-image\')) {\n')
            f.write('        findLatestImage(\'business_image\', \'business-hero-image\');\n')
            f.write('    }\n')
            f.write('});\n')
            f.write('</script>\n')
            f.write('{% endblock %}')
        try:
            with open(template_path, 'w') as f:
                # We already wrote the file line by line above
                pass
            return True
        except Exception as e:
            print(f"Error creating template {template_name}: {str(e)}")
            return False
    
    def find_latest_file(self, directory, prefix, suffix):
        """Find the latest file with the given prefix and suffix in a directory."""
        if not os.path.exists(directory):
            # Try to create the directory if it doesn't exist
            try:
                os.makedirs(directory)
                print(f"Created directory: {directory}")
            except Exception as e:
                print(f"Error creating directory {directory}: {str(e)}")
            return None
            
        files = [f for f in os.listdir(directory) 
                 if f.startswith(prefix) and f.endswith(suffix)]
        
        if not files:
            return None
            
        # Sort by timestamp in filename (assuming format prefix_timestamp_*.suffix)
        files.sort(reverse=True)
        return files[0]
        
    def find_audio_file(self, section="introduction", voice="fable"):
        """Find audio file for a specific section and voice."""
        audio_dir = os.path.join(os.getcwd(), 'static', 'audio')
        if not os.path.exists(audio_dir):
            return None
            
        # First try to find specific section+voice file with pattern: lyceum_<section>_<voice>_latest.mp3
        pattern = f"lyceum_{section}_{voice}_latest.mp3"
        path = os.path.join(audio_dir, pattern)
        if os.path.exists(path) and os.path.islink(path):
            # Get the real file it points to
            try:
                real_path = os.path.realpath(path)
                if os.path.exists(real_path) and os.path.getsize(real_path) > 1000:
                    return pattern
            except Exception as e:
                print(f"Error resolving symlink {path}: {str(e)}")
        
        # Try to find section+voice files with timestamp (not symlinks): lyceum_<section>_<voice>_TIMESTAMP.mp3
        pattern_prefix = f"lyceum_{section}_{voice}_"
        matching_files = [f for f in os.listdir(audio_dir) 
                         if f.startswith(pattern_prefix) and f.endswith(".mp3") and not f.endswith("_latest.mp3")]
        
        if matching_files:
            # Sort by timestamp to get the latest
            matching_files.sort(reverse=True)
            return matching_files[0]
            
        # Try section with any voice
        pattern_prefix = f"lyceum_{section}_"
        matching_files = [f for f in os.listdir(audio_dir) 
                         if f.startswith(pattern_prefix) and f.endswith(".mp3") and not f.endswith("_metadata.json")]
        
        if matching_files:
            matching_files.sort(reverse=True)
            return matching_files[0]
        
        # Last resort: try any introduction file
        fallback_files = [f for f in os.listdir(audio_dir) 
                         if f.startswith("lyceum_introduction_") and f.endswith(".mp3") and not f.endswith("_metadata.json")]
        
        if fallback_files:
            fallback_files.sort(reverse=True)
            return fallback_files[0]
            
        return None
    
    def create_latest_links(self):
        """Create symlinks to the latest versions of assets for easier reference."""
        results = {
            "success": True,
            "images": [],
            "audio": [],
            "content": []
        }
        
        try:
            # Create latest image links
            images_dir = os.path.join(os.getcwd(), 'static', 'images')
            if os.path.exists(images_dir):
                # Look for each type of page image
                page_types = ["vision", "technical", "business", "agile"]
                for page in page_types:
                    # Find the latest main image for each page
                    latest_image = self.find_latest_file(images_dir, f"lyceum_{page}_", ".png")
                    if latest_image:
                        source = os.path.join(images_dir, latest_image)
                        target = os.path.join(images_dir, f"lyceum_{page}_latest.png")
                        
                        # Remove existing link if any
                        if os.path.exists(target):
                            os.remove(target)
                        
                        # Create copy instead of symlink for better compatibility
                        import shutil
                        shutil.copy2(source, target)
                        results["images"].append(f"Created link for {page}: {latest_image}")
                    
                    # Also find numbered images (1, 2, 3) for each page
                    for i in range(1, 4):
                        latest_numbered = self.find_latest_file(images_dir, f"lyceum_{page}_{i}_", ".png")
                        if latest_numbered:
                            source = os.path.join(images_dir, latest_numbered)
                            target = os.path.join(images_dir, f"lyceum_{page}_{i}_latest.png")
                            
                            # Remove existing link if any
                            if os.path.exists(target):
                                os.remove(target)
                            
                            # Create copy
                            import shutil
                            shutil.copy2(source, target)
                            results["images"].append(f"Created link for {page}_{i}: {latest_numbered}")
            
            # Create latest audio links
            audio_dir = os.path.join(os.getcwd(), 'static', 'audio')
            if os.path.exists(audio_dir):
                # Get all sections
                sections = ["introduction", "vision", "technical", "business", "agile", "contact"]
                
                # Get all voices
                voices = ["alloy", "ash", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer"]
                
                print("Creating latest links for audio files...")
                
                # Create latest links for each section and voice
                for section in sections:
                    for voice in voices:
                        pattern_prefix = f"lyceum_{section}_{voice}_"
                        latest_file = None
                        
                        # Find the latest file with this pattern
                        matching_files = [f for f in os.listdir(audio_dir) 
                                         if f.startswith(pattern_prefix) and f.endswith(".mp3") 
                                         and not f.endswith("_latest.mp3") 
                                         and not f.endswith("_metadata.json")]
                        
                        if matching_files:
                            # Sort by timestamp to get the latest
                            matching_files.sort(reverse=True)
                            latest_file = matching_files[0]
                        
                        if latest_file:
                            source = os.path.join(audio_dir, latest_file)
                            target = os.path.join(audio_dir, f"lyceum_{section}_{voice}_latest.mp3")
                            
                            # Remove existing link if any
                            if os.path.exists(target):
                                os.remove(target)
                            
                            # Create symlink
                            try:
                                os.symlink(source, target)
                                results["audio"].append(f"Created link: lyceum_{section}_{voice}_latest.mp3 -> {latest_file}")
                            except Exception as e:
                                print(f"Error creating symlink: {str(e)}")
                                # Try a direct copy instead
                                try:
                                    import shutil
                                    shutil.copy2(source, target)
                                    results["audio"].append(f"Created copy: lyceum_{section}_{voice}_latest.mp3 from {latest_file}")
                                except Exception as copy_err:
                                    print(f"Error copying file: {str(copy_err)}")
                
                # For backward compatibility, create lyceum_introduction_latest.mp3
                fallback_voice = "fable"
                fallback_pattern = f"lyceum_introduction_{fallback_voice}_"
                fallback_files = [f for f in os.listdir(audio_dir) 
                                 if f.startswith(fallback_pattern) and f.endswith(".mp3") 
                                 and not f.endswith("_latest.mp3")]
                
                if fallback_files:
                    fallback_files.sort(reverse=True)
                    latest_file = fallback_files[0]
                    source = os.path.join(audio_dir, latest_file)
                    target = os.path.join(audio_dir, "lyceum_introduction_latest.mp3")
                    
                    # Remove existing link if any
                    if os.path.exists(target):
                        os.remove(target)
                    
                    # Create copy
                    import shutil
                    shutil.copy2(source, target)
                    results["audio"].append(f"Created link: lyceum_introduction_latest.mp3 -> {latest_file}")
            
            # Create latest content links
            content_dir = os.path.join(os.getcwd(), 'content')
            if os.path.exists(content_dir):
                # Create links for each content type
                for content_type in ["vision", "technical", "business", "agile", "contact"]:
                    latest_content = self.find_latest_file(content_dir, f"lyceum_{content_type}_", ".md")
                    if latest_content:
                        source = os.path.join(content_dir, latest_content)
                        target = os.path.join(content_dir, f"lyceum_{content_type}_latest.md")
                        
                        # Remove existing link if any
                        if os.path.exists(target):
                            os.remove(target)
                        
                        # Create copy
                        import shutil
                        shutil.copy2(source, target)
                        results["content"].append(f"Created link for {content_type}: {latest_content}")
        
        except Exception as e:
            results["success"] = False
            results["error"] = str(e)
        
        return results
        
    def end_headers(self):
        """Add CORS headers."""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        http.server.SimpleHTTPRequestHandler.end_headers(self)
    
    def do_OPTIONS(self):
        """Handle preflight requests."""
        self.send_response(200)
        self.end_headers()

def main():
    """Start the server with threading support for better performance."""
    parser = argparse.ArgumentParser(description='Serve the Lyceum visualization system')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT,
                        help=f'Port to serve on (default: {DEFAULT_PORT})')
    parser.add_argument('--host', type=str, default="0.0.0.0",
                        help=f'Host to serve on (default: 0.0.0.0)')
    args = parser.parse_args()
    
    # Change to the directory of this script
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Create the server with threading support
    with ThreadedHTTPServer((args.host, args.port), LyceumHandler) as httpd:
        logger.info(f"Serving Lyceum visualization at http://{args.host}:{args.port}")
        logger.info(f"Using Jinja2 for templating: {JINJA2_AVAILABLE}")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            logger.info("\nServer stopped.")
            httpd.server_close()
            # Clean up the thread pool
            LyceumHandler._executor.shutdown(wait=False)

if __name__ == "__main__":
    main()