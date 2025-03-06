#!/usr/bin/env python3
"""
Test script for the Lyceum website functionality.
Checks for page navigation, content loading, and audio playback.
"""

import unittest
import requests
import time
import subprocess
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import signal
import sys

PORT = 8081
BASE_URL = f"http://localhost:{PORT}"
SERVER_PROCESS = None

class LyceumWebsiteTests(unittest.TestCase):
    """Test cases for the Lyceum website."""
    
    @classmethod
    def setUpClass(cls):
        """Start the server and initialize the browser."""
        # Start the server
        cls.start_server()
        
        # Set up the browser
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run headless Chrome
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--mute-audio")  # Mute audio for tests
        
        cls.browser = webdriver.Chrome(options=chrome_options)
        
        # Wait for server to be ready
        max_retries = 5
        for _ in range(max_retries):
            try:
                response = requests.get(f"{BASE_URL}/health")
                if response.status_code == 200:
                    break
            except requests.exceptions.ConnectionError:
                time.sleep(1)
        else:
            raise Exception("Server failed to start")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up resources."""
        # Close the browser
        if hasattr(cls, 'browser'):
            cls.browser.quit()
        
        # Stop the server
        cls.stop_server()
    
    @classmethod
    def start_server(cls):
        """Start the server as a subprocess."""
        global SERVER_PROCESS
        SERVER_PROCESS = subprocess.Popen(
            ["python", "serve.py", "--port", str(PORT)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid
        )
        print(f"Server started with PID {SERVER_PROCESS.pid}")
    
    @classmethod
    def stop_server(cls):
        """Stop the server subprocess."""
        global SERVER_PROCESS
        if SERVER_PROCESS:
            os.killpg(os.getpgid(SERVER_PROCESS.pid), signal.SIGTERM)
            print("Server stopped")
    
    def test_homepage_loads(self):
        """Test that the homepage loads correctly."""
        self.browser.get(BASE_URL)
        self.assertIn("Lyceum", self.browser.title)
        
        # Check for key elements on the homepage
        self.assertTrue(self.browser.find_element(By.CLASS_NAME, "hero").is_displayed())
        self.assertTrue(self.browser.find_element(By.ID, "play-introduction").is_displayed())
    
    def test_navigation_menu(self):
        """Test that navigation menu links are working."""
        self.browser.get(BASE_URL)
        
        # Get all navigation links
        nav_links = self.browser.find_elements(By.CSS_SELECTOR, "nav ul li a")
        self.assertGreaterEqual(len(nav_links), 5, "Navigation should have at least 5 links")
        
        # Check all nav links to see which ones work
        working_links = []
        broken_links = []
        
        # Store the current window handle
        original_window = self.browser.current_window_handle
        
        for link in nav_links:
            href = link.get_attribute("href")
            link_text = link.text
            
            # Skip external links
            if not href.startswith(BASE_URL):
                continue
                
            # Click the link
            link.click()
            
            # Wait for the page to load
            WebDriverWait(self.browser, 2).until(
                EC.staleness_of(link)
            )
            
            # Check if page loaded successfully
            if "Page not found" in self.browser.page_source:
                broken_links.append((link_text, href))
            else:
                working_links.append((link_text, href))
            
            # Go back to the homepage
            self.browser.get(BASE_URL)
        
        # Print results
        print("\nNavigation Link Check Results:")
        print("Working Links:", working_links)
        print("Broken Links:", broken_links)
        
        self.assertGreaterEqual(len(working_links), 2, "At least 2 navigation links should work")
    
    def test_audio_player(self):
        """Test that the audio player functionality works."""
        self.browser.get(BASE_URL)
        
        # Find the audio player
        audio_player = self.browser.find_element(By.ID, "introduction-audio")
        self.assertIsNotNone(audio_player, "Audio player should be present")
        
        # Find the play button
        play_button = self.browser.find_element(By.ID, "play-introduction")
        self.assertIsNotNone(play_button, "Play button should be present")
        
        # Click the play button
        play_button.click()
        
        # Wait for the audio to start playing
        time.sleep(1)
        
        # Check if the button text changed (indicating play state)
        button_text = play_button.find_element(By.CLASS_NAME, "button-text").text
        
        # The button should now show "Pause Introduction" if audio is playing
        self.assertIn("Pause", button_text, "Button text should change to indicate pause")
        
        # Click again to pause
        play_button.click()
        
        # Wait for UI to update
        time.sleep(1)
        
        # Check if audio is paused
        button_text = play_button.find_element(By.CLASS_NAME, "button-text").text
        self.assertIn("Listen", button_text, "Button text should change back to indicate play")
    
    def test_api_endpoints(self):
        """Test that the API endpoints return valid data."""
        # Test info endpoint
        response = requests.get(f"{BASE_URL}/api/info")
        self.assertEqual(response.status_code, 200)
        info_data = response.json()
        self.assertIn("name", info_data)
        self.assertIn("version", info_data)
        
        # Test latest audio endpoint
        response = requests.get(f"{BASE_URL}/api/latest-audio")
        self.assertEqual(response.status_code, 200)
        audio_data = response.json()
        self.assertIn("path", audio_data)
        
        # Test latest vision endpoint
        response = requests.get(f"{BASE_URL}/api/latest-vision")
        self.assertEqual(response.status_code, 200)
        vision_data = response.json()
        self.assertIn("content", vision_data)
        
        # Test latest images endpoint
        response = requests.get(f"{BASE_URL}/api/latest-images")
        self.assertEqual(response.status_code, 200)
        images_data = response.json()
        self.assertIn("logo", images_data)

if __name__ == "__main__":
    unittest.main()