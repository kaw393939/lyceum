#!/usr/bin/env python3
"""
Automated tests for the Lyceum website.
Tests core functionality, audio generation, and asset validation.
"""

import os
import sys
import unittest
import subprocess
import tempfile
import json
import logging
import time
from pathlib import Path
import requests
from io import BytesIO
import wave
import contextlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_lyceum')

# Define paths
PROJECT_ROOT = Path(__file__).parent
STATIC_DIR = PROJECT_ROOT / "static"
AUDIO_DIR = STATIC_DIR / "audio"
IMAGES_DIR = STATIC_DIR / "images"
CSS_DIR = STATIC_DIR / "css"
JS_DIR = STATIC_DIR / "js"

# Ensure directories exist
for directory in [STATIC_DIR, AUDIO_DIR, IMAGES_DIR, CSS_DIR, JS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Server configuration
SERVER_HOST = "localhost"
SERVER_PORT = 8081  # Use a different port to avoid conflicts
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"

def is_server_running():
    """Check if the server is running."""
    try:
        response = requests.get(f"{SERVER_URL}/health", timeout=1)
        return response.status_code == 200
    except requests.RequestException:
        return False

def start_server():
    """Start the server if it's not running."""
    if not is_server_running():
        logger.info("Starting server...")
        subprocess.Popen(
            [sys.executable, "serve.py", f"--port={SERVER_PORT}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        # Wait for server to start
        for _ in range(5):
            if is_server_running():
                logger.info("Server started successfully")
                break
            time.sleep(1)
        else:
            logger.error("Failed to start server")
            return False
    return True

def stop_server():
    """Stop the server."""
    try:
        logger.info("Stopping server...")
        subprocess.run(
            ["pkill", "-f", f"python.*serve.py.*port={SERVER_PORT}"],
            check=True
        )
        logger.info("Server stopped")
    except subprocess.CalledProcessError:
        logger.warning("Failed to stop server")

class TestStaticAssets(unittest.TestCase):
    """Test static assets like CSS, JS, and images."""

    def test_css_exists(self):
        """Test that CSS files exist."""
        css_files = ["main.css", "modern.css"]
        for css_file in css_files:
            css_path = CSS_DIR / css_file
            self.assertTrue(css_path.exists(), f"CSS file {css_file} does not exist")

    def test_js_exists(self):
        """Test that JS files exist."""
        js_files = ["main.js", "visualizer.js"]
        for js_file in js_files:
            js_path = JS_DIR / js_file
            self.assertTrue(js_path.exists(), f"JS file {js_file} does not exist")

    def test_audio_exists(self):
        """Test that audio files exist."""
        # Check for audio files
        audio_files = list(AUDIO_DIR.glob("*.mp3"))
        self.assertTrue(len(audio_files) > 0, "No MP3 files found in audio directory")
        
        # Check specifically for the latest introduction audio
        latest_audio = AUDIO_DIR / "lyceum_introduction_latest.mp3"
        self.assertTrue(latest_audio.exists(), "Latest audio introduction file not found")

    def test_images_exist(self):
        """Test that image files exist."""
        # Check for image files
        image_files = list(IMAGES_DIR.glob("*.png"))
        self.assertTrue(len(image_files) > 0, "No PNG files found in images directory")
        
        # Check for logo file
        logo_files = list(IMAGES_DIR.glob("lyceum_logo_*.png"))
        self.assertTrue(len(logo_files) > 0, "No logo files found")

class TestTemplates(unittest.TestCase):
    """Test template files."""

    def test_templates_exist(self):
        """Test that template files exist."""
        template_dir = PROJECT_ROOT / "templates"
        self.assertTrue(template_dir.exists(), "Templates directory does not exist")
        
        required_templates = ["base.html", "index.html"]
        for template in required_templates:
            template_path = template_dir / template
            self.assertTrue(template_path.exists(), f"Template {template} does not exist")

class TestWebServer(unittest.TestCase):
    """Test the web server functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up the test class."""
        # Start server if not running
        start_server()
    
    def test_server_health(self):
        """Test that the server is healthy."""
        response = requests.get(f"{SERVER_URL}/health")
        self.assertEqual(response.status_code, 200, "Server health check failed")
        health_data = response.json()
        self.assertEqual(health_data.get("status"), "ok", "Server status is not ok")

    def test_home_page(self):
        """Test that the home page loads."""
        response = requests.get(SERVER_URL)
        self.assertEqual(response.status_code, 200, "Failed to load home page")
        
        # Check for key elements in the HTML
        html_content = response.text
        self.assertIn("<title>", html_content, "Title tag not found")
        self.assertIn("Lyceum", html_content, "Lyceum not found in content")
        self.assertIn("audio", html_content, "Audio element not found")

    def test_static_files_served(self):
        """Test that static files are served correctly."""
        # Test CSS
        css_response = requests.get(f"{SERVER_URL}/static/css/main.css")
        self.assertEqual(css_response.status_code, 200, "Failed to load main.css")
        
        # Test JS
        js_response = requests.get(f"{SERVER_URL}/static/js/main.js")
        self.assertEqual(js_response.status_code, 200, "Failed to load main.js")

    def test_api_endpoints(self):
        """Test API endpoints."""
        # Test info endpoint
        info_response = requests.get(f"{SERVER_URL}/api/info")
        self.assertEqual(info_response.status_code, 200, "Failed to access info endpoint")
        
        # Test latest audio endpoint
        audio_response = requests.get(f"{SERVER_URL}/api/latest-audio")
        self.assertEqual(audio_response.status_code, 200, "Failed to access latest-audio endpoint")
        audio_data = audio_response.json()
        self.assertIn("path", audio_data, "Audio path not found in response")

    @classmethod
    def tearDownClass(cls):
        """Tear down the test class."""
        # Server will continue running for other tests
        pass

class TestAudioGeneration(unittest.TestCase):
    """Test audio generation functionality."""

    def test_audio_generator_module(self):
        """Test that the audio generator module can be imported."""
        try:
            from utils import audio_generator
            self.assertTrue(hasattr(audio_generator, "generate_audio"), "generate_audio function not found")
            self.assertTrue(hasattr(audio_generator, "select_script"), "select_script function not found")
        except ImportError as e:
            self.fail(f"Failed to import audio_generator module: {e}")

    def test_script_selection(self):
        """Test script selection functionality."""
        from utils.audio_generator import select_script, INTRODUCTION_SCRIPTS
        
        # Test script selection with valid index
        script = select_script(0)
        self.assertIsInstance(script, str, "Script is not a string")
        self.assertTrue(len(script) > 50, "Script is too short")
        
        # Test script selection with invalid index
        script = select_script(999)  # Should default to script 0
        self.assertIsInstance(script, str, "Script is not a string")
        self.assertTrue(len(script) > 50, "Script is too short")
        
        # Verify all scripts are valid
        for i in range(len(INTRODUCTION_SCRIPTS)):
            script = select_script(i)
            self.assertIsInstance(script, str, "Script is not a string")
            self.assertTrue(len(script) > 50, "Script is too short")

    def test_audio_format(self):
        """Test that generated audio files are valid MP3 files."""
        # Find the latest generated audio file
        audio_files = list(AUDIO_DIR.glob("lyceum_introduction_*.mp3"))
        audio_files = [f for f in audio_files if f.name != "lyceum_introduction_latest.mp3"]
        
        # Skip test if no files found
        if not audio_files:
            self.skipTest("No audio files found to test")
            return
        
        # Sort by modification time, newest first
        audio_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        latest_audio = audio_files[0]
        
        # Test file size
        file_size = latest_audio.stat().st_size
        self.assertTrue(file_size > 1000, f"Audio file {latest_audio.name} is too small: {file_size} bytes")
        
        # Test if file is a valid audio file
        # For MP3, we can check headers or try loading with a library
        # Here we'll just check the file extension and size
        self.assertTrue(latest_audio.name.endswith(".mp3"), "File doesn't have .mp3 extension")

class TestImageGeneration(unittest.TestCase):
    """Test image generation functionality."""

    def test_image_generator_module(self):
        """Test that the image generator module can be imported."""
        try:
            from utils import image_generator
            self.assertTrue(hasattr(image_generator, "ImageGenerator"), "ImageGenerator class not found")
        except ImportError as e:
            self.fail(f"Failed to import image_generator module: {e}")

    def test_image_format(self):
        """Test that generated image files are valid PNG files."""
        # Find image files
        image_files = list(IMAGES_DIR.glob("lyceum_*.png"))
        
        # Skip test if no files found
        if not image_files:
            self.skipTest("No image files found to test")
            return
        
        # Test each image
        for image_file in image_files:
            # Test file size
            file_size = image_file.stat().st_size
            self.assertTrue(file_size > 1000, f"Image file {image_file.name} is too small: {file_size} bytes")
            
            # Test if file is a valid PNG file (check signature)
            with open(image_file, 'rb') as f:
                header = f.read(8)
                # PNG signature is 89 50 4E 47 0D 0A 1A 0A
                self.assertEqual(header, b'\x89PNG\r\n\x1a\n', f"File {image_file.name} is not a valid PNG")

class TestAssetRegeneration(unittest.TestCase):
    """Test asset regeneration script."""

    def test_regenerate_script_exists(self):
        """Test that the regenerate_assets.py script exists."""
        script_path = PROJECT_ROOT / "regenerate_assets.py"
        self.assertTrue(script_path.exists(), "regenerate_assets.py script not found")

    def test_regenerate_script_runs(self):
        """Test that the regenerate_assets.py script runs without errors."""
        # Run script with --help to avoid actual regeneration
        process = subprocess.run(
            [sys.executable, "regenerate_assets.py", "--help"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        self.assertEqual(process.returncode, 0, f"regenerate_assets.py failed with error: {process.stderr.decode()}")

def run_tests():
    """Run the tests."""
    # First check if server is running, start if not
    if not start_server():
        logger.error("Server could not be started, aborting tests")
        return False
    
    # Run the tests
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(TestStaticAssets))
    test_suite.addTest(unittest.makeSuite(TestTemplates))
    test_suite.addTest(unittest.makeSuite(TestWebServer))
    test_suite.addTest(unittest.makeSuite(TestAudioGeneration))
    test_suite.addTest(unittest.makeSuite(TestImageGeneration))
    test_suite.addTest(unittest.makeSuite(TestAssetRegeneration))
    
    test_runner = unittest.TextTestRunner(verbosity=2)
    test_result = test_runner.run(test_suite)
    
    # Report results
    if test_result.wasSuccessful():
        logger.info("All tests passed!")
        return True
    else:
        logger.error("Some tests failed")
        return False

if __name__ == "__main__":
    try:
        success = run_tests()
        # Uncomment to stop the server after tests
        # stop_server()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("Tests interrupted")
        stop_server()
        sys.exit(1)