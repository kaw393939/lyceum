// Main JavaScript for Lyceum
document.addEventListener('DOMContentLoaded', function() {
    console.log("DOM Content Loaded - Initializing Lyceum scripts");
    
    // Initialize audio players first
    initializeAudioPlayers();
    
    // Initialize visualizations next
    setTimeout(() => {
        if (typeof visualizer !== 'undefined') {
            console.log("Visualizer found - initializing visualizations");
            initializeVisualizations();
            
            // Handle window resize for responsive visualizations
            window.addEventListener('resize', debounce(function() {
                initializeVisualizations();
            }, 250));
        } else {
            console.error("Visualizer not found");
        }
        
        // Initialize mermaid diagrams if available
        if (typeof mermaid !== 'undefined') {
            console.log("Mermaid found - initializing diagrams");
            mermaid.initialize({
                startOnLoad: true,
                securityLevel: 'loose',
                theme: 'default',
                themeVariables: {
                    primaryColor: '#1a237e',
                    primaryTextColor: '#ffffff',
                    primaryBorderColor: '#fdd835',
                    lineColor: '#3949ab',
                    secondaryColor: '#3949ab',
                    tertiaryColor: '#f5f5f5'
                }
            });
        }
        
        // Initialize AOS (Animate on Scroll)
        if (typeof AOS !== 'undefined') {
            AOS.init({
                duration: 800,
                easing: 'ease-in-out',
                once: true,
                offset: 100
            });
        }
    }, 100);
});

/**
 * Initialize visualizations for the different pages
 */
function initializeVisualizations() {
    // Home page visualization (knowledge network)
    if (document.getElementById('hero-visualization')) {
        console.log("Found hero visualization container, rendering...");
        visualizer.renderHomeVisualization();
    }
    
    // Technical page diagrams
    if (document.getElementById('technical-diagram')) {
        console.log("Found technical diagram container, rendering...");
        visualizer.renderTechnicalDiagram();
    }
    
    // Agile roadmap
    if (document.getElementById('roadmap-diagram')) {
        console.log("Found roadmap diagram container, rendering...");
        visualizer.renderAgileRoadmap();
    }
    
    // Platform diagram
    if (document.getElementById('home-platform-diagram')) {
        console.log("Found platform diagram container, rendering...");
        visualizer.renderHomePlatformDiagram();
    }
}

/**
 * Debounce function to limit how often a function can be called
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * Initialize audio players with voice selection, playback controls and visualizations
 */
function initializeAudioPlayers() {
    // Get all audio players on the page
    // Try both class names to support both old and new HTML structures
    const newAudioPlayers = document.querySelectorAll('.audio-player');
    const oldAudioPlayers = document.querySelectorAll('.audio-introduction');
    
    // Combine both collections
    const audioPlayers = [...newAudioPlayers, ...oldAudioPlayers];
    
    if (audioPlayers.length === 0) {
        console.log("No audio players found on this page");
        return;
    }
    
    console.log(`Found ${audioPlayers.length} audio player(s)`);
    
    // Process each audio player
    audioPlayers.forEach(function(playerElement) {
        // Get the section name from data attribute or default to "introduction"
        const section = playerElement.getAttribute('data-section') || 'introduction';
        console.log(`Initializing audio player for section: ${section}`);
        
        // Get key elements
        const audioElement = playerElement.querySelector('audio');
        const playButton = playerElement.querySelector('.play-button');
        // Visualization might have different class names
        const visualization = playerElement.querySelector('.audio-visualization') || 
                              playerElement.querySelector('.visualization-container');
        const playbackInfo = playerElement.querySelector('.audio-playback-info') || 
                             playerElement.querySelector('.playback-info');
        const voiceSelector = playerElement.querySelector('.voice-selector');
        
        // Get text elements, with null checks
        const currentVoiceText = playbackInfo?.querySelector('.current-voice');
        const playbackTimeText = playbackInfo?.querySelector('.playback-time');
        
        // Default voice
        let currentVoice = voiceSelector ? voiceSelector.value : 'alloy';
        
        // Debug info about the audio element
        if (audioElement) {
            console.log(`Audio element found for ${section}`);
            console.log(`Audio source: ${audioElement.querySelector('source')?.src || 'none'}`);
            console.log(`Audio preload: ${audioElement.getAttribute('preload') || 'none set'}`);
        } else {
            console.error(`No audio element found for ${section} player`);
        }
        
        // Create playback controls if required elements exist (more permissive now)
        if (audioElement && playButton) {
            // Format time (MM:SS)
            function formatTime(seconds) {
                if (isNaN(seconds)) return "0:00";
                const minutes = Math.floor(seconds / 60);
                const remainingSeconds = Math.floor(seconds % 60);
                return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
            }
            
            // Update playback time display
            function updatePlaybackTime() {
                if (audioElement.duration && playbackTimeText) {
                    const current = formatTime(audioElement.currentTime);
                    const total = formatTime(audioElement.duration);
                    playbackTimeText.textContent = `${current} / ${total}`;
                }
            }
            
            // Create visualization bars container if visualization exists
            if (visualization) {
                // Check if wave bars container already exists
                let barsContainer = visualization.querySelector('.wave-bars-container');
                
                // Create it only if it doesn't exist
                if (!barsContainer) {
                    barsContainer = document.createElement('div');
                    barsContainer.className = 'wave-bars-container';
                    visualization.appendChild(barsContainer);
                    
                    // Create wave bars
                    const barCount = 30;
                    for (let i = 0; i < barCount; i++) {
                        const bar = document.createElement('div');
                        bar.className = 'wave-bar';
                        bar.style.setProperty('--bar-index', i); // For staggered animations
                        barsContainer.appendChild(bar);
                    }
                }
            }
            
            // Play/Pause functionality
            playButton.addEventListener('click', function() {
                console.log("Play button clicked");
                
                // Direct play without first checking if file exists
                if (audioElement.paused) {
                    console.log("Audio is paused, attempting to play");
                    
                    // Reset if ended
                    if (audioElement.currentTime >= audioElement.duration) {
                        audioElement.currentTime = 0;
                    }
                    
                    // Use catch in case play fails for any reason
                    audioElement.play()
                        .then(() => {
                            console.log("Audio playback started successfully");
                            
                            // Update UI for playing state
                            if (playButton.querySelector('.play-icon')) {
                                playButton.querySelector('.play-icon').textContent = '⏸';
                            }
                            if (playButton.querySelector('.button-text')) {
                                playButton.querySelector('.button-text').textContent = 'Pause';
                            }
                            if (visualization) {
                                visualization.classList.add('active');
                                animateWaveBars();
                            }
                        })
                        .catch(error => {
                            console.error('Audio playback failed:', error);
                            showError(playerElement, 'Audio playback failed. Please try another voice or reload the page.');
                        });
                } else {
                    // Audio is already playing, so pause it
                    console.log("Audio is playing, pausing");
                    audioElement.pause();
                    
                    // Update UI for paused state
                    if (playButton.querySelector('.play-icon')) {
                        playButton.querySelector('.play-icon').textContent = '▶';
                    }
                    if (playButton.querySelector('.button-text')) {
                        playButton.querySelector('.button-text').textContent = 'Listen to ' + 
                            (section.charAt(0).toUpperCase() + section.slice(1));
                    }
                    if (visualization) {
                        visualization.classList.remove('active');
                    }
                }
            });
            
            // Voice selector change handler
            if (voiceSelector) {
                voiceSelector.addEventListener('change', function() {
                    const voice = voiceSelector.value;
                    console.log(`Changing voice to: ${voice}`);
                    
                    // Update voice display
                    if (currentVoiceText) {
                        currentVoiceText.textContent = voice.charAt(0).toUpperCase() + voice.slice(1) + ' voice';
                    }
                    
                    // Direct source change
                    const wasPlaying = !audioElement.paused;
                    if (wasPlaying) {
                        audioElement.pause();
                    }
                    
                    // Set new source directly using timestamp format
                    const timestamp = "20250306051146"; // A fixed timestamp
                    let newSource = `/static/audio/lyceum_${section}_${voice}_${timestamp}.mp3`;
                    audioElement.querySelector('source').src = newSource;
                    audioElement.load();
                    
                    console.log(`Changed source to: ${newSource}`);
                    
                    // Update current voice
                    currentVoice = voice;
                    
                    // Resume playback if needed
                    if (wasPlaying) {
                        audioElement.play()
                            .then(() => {
                                console.log("Resumed playback with new voice");
                                animateWaveBars();
                            })
                            .catch(err => {
                                console.error("Failed to resume with new voice:", err);
                                showError(playerElement, "Failed to switch voice. Please try another voice.");
                            });
                    }
                });
            }
            
            // Audio visualization animation
            function animateWaveBars() {
                if (audioElement.paused || !visualization) return;
                
                const bars = visualization.querySelectorAll('.wave-bar');
                if (bars.length === 0) return;
                
                // Animate each bar based on position and time
                bars.forEach((bar, index) => {
                    // Use mathematical waves for more natural-looking animation
                    const phase = index / bars.length * Math.PI * 2;
                    const time = Date.now() / 500;
                    const sineVal = Math.sin(time + phase);
                    const cosVal = Math.cos(time * 0.7 + phase);
                    
                    // Combine waves with slight randomness
                    const height = 5 + (sineVal * 0.5 + cosVal * 0.5 + 1) * 15 + (Math.random() * 2);
                    bar.style.height = `${height}px`;
                });
                
                // Continue animation loop
                requestAnimationFrame(animateWaveBars);
            }
            
            // Function to load audio for a specific voice
            async function loadAudioForVoice(voice) {
                // Clear any previous errors
                clearErrors(playerElement);
                
                // Show loading state
                playerElement.classList.add('loading');
                
                // Remember if we're playing so we can resume
                const wasPlaying = !audioElement.paused;
                
                // If playing, pause first
                if (wasPlaying) {
                    audioElement.pause();
                }
                
                try {
                    // Check if file exists first
                    const audioPath = `/static/audio/lyceum_${section}_${voice}_latest.mp3`;
                    console.log(`Checking if audio exists: ${audioPath}`);
                    
                    const response = await fetch(audioPath, { method: 'HEAD' });
                    
                    if (response.ok) {
                        console.log(`Audio file found: ${audioPath}`);
                        
                        // Set audio source directly
                        audioElement.src = audioPath;
                        
                        // Update current voice
                        currentVoice = voice;
                        
                        // Wait for load
                        await new Promise((resolve) => {
                            audioElement.onloadeddata = resolve;
                            audioElement.onerror = () => {
                                console.error(`Error loading audio: ${audioPath}`);
                                resolve();
                            };
                            audioElement.load();
                        });
                        
                        console.log(`Audio loaded successfully: ${audioPath}`);
                        playerElement.classList.remove('loading');
                        
                        // Update time display
                        updatePlaybackTime();
                        
                        // Resume playback if needed
                        if (wasPlaying) {
                            try {
                                await audioElement.play();
                                
                                // Update UI
                                if (playButton.querySelector('.play-icon')) {
                                    playButton.querySelector('.play-icon').textContent = '⏸';
                                }
                                if (playButton.querySelector('.button-text')) {
                                    playButton.querySelector('.button-text').textContent = 'Pause';
                                }
                                if (visualization) {
                                    visualization.classList.add('active');
                                }
                                animateWaveBars();
                            } catch (e) {
                                console.error('Error resuming playback:', e);
                                showError(playerElement, 'Playback failed after voice change. Please try again.');
                            }
                        }
                        
                        // Success!
                        return true;
                    } else {
                        throw new Error(`Audio file not found: ${audioPath}`);
                    }
                } catch (error) {
                    console.error(`Failed to load audio for ${section} with ${voice} voice:`, error);
                    
                    // Try fallback voices
                    const fallbackVoices = ['fable', 'echo', 'alloy', 'shimmer', 'nova'];
                    let fallbackFound = false;
                    
                    // Try each fallback voice (except the current one)
                    for (const fallbackVoice of fallbackVoices) {
                        if (fallbackVoice !== voice) {
                            try {
                                const fallbackPath = `/static/audio/lyceum_${section}_${fallbackVoice}_latest.mp3`;
                                console.log(`Trying fallback voice: ${fallbackPath}`);
                                
                                const fallbackResponse = await fetch(fallbackPath, { method: 'HEAD' });
                                
                                if (fallbackResponse.ok) {
                                    console.log(`Found working fallback voice: ${fallbackVoice}`);
                                    showError(playerElement, `${voice} voice not available. Using ${fallbackVoice} voice instead.`);
                                    
                                    // Update selector
                                    if (voiceSelector) {
                                        voiceSelector.value = fallbackVoice;
                                    }
                                    
                                    // Update voice display
                                    if (currentVoiceText) {
                                        currentVoiceText.textContent = fallbackVoice.charAt(0).toUpperCase() + 
                                            fallbackVoice.slice(1) + ' voice';
                                    }
                                    
                                    // Set audio source
                                    audioElement.src = fallbackPath;
                                    currentVoice = fallbackVoice;
                                    
                                    await new Promise((resolve) => {
                                        audioElement.onloadeddata = resolve;
                                        audioElement.onerror = () => resolve();
                                        audioElement.load();
                                    });
                                    
                                    // Resume if needed
                                    if (wasPlaying) {
                                        try {
                                            await audioElement.play();
                                            if (playButton.querySelector('.play-icon')) {
                                                playButton.querySelector('.play-icon').textContent = '⏸';
                                            }
                                            if (playButton.querySelector('.button-text')) {
                                                playButton.querySelector('.button-text').textContent = 'Pause';
                                            }
                                            if (visualization) {
                                                visualization.classList.add('active');
                                                animateWaveBars();
                                            }
                                        } catch (e) {
                                            console.error('Fallback playback failed:', e);
                                        }
                                    }
                                    
                                    fallbackFound = true;
                                    break;
                                }
                            } catch (e) {
                                console.warn(`Fallback ${fallbackVoice} failed:`, e);
                            }
                        }
                    }
                    
                    if (!fallbackFound) {
                        showError(playerElement, 'No audio voices available. Please try again later.');
                    }
                    
                    playerElement.classList.remove('loading');
                    return fallbackFound;
                }
            }
            
            // Handle playback end
            audioElement.addEventListener('ended', function() {
                if (playButton.querySelector('.play-icon')) {
                    playButton.querySelector('.play-icon').textContent = '▶';
                }
                if (playButton.querySelector('.button-text')) {
                    playButton.querySelector('.button-text').textContent = 'Listen to ' + 
                        (section.charAt(0).toUpperCase() + section.slice(1));
                }
                if (visualization) {
                    visualization.classList.remove('active');
                    
                    // Reset wave bars
                    const bars = visualization.querySelectorAll('.wave-bar');
                    bars.forEach(bar => {
                        bar.style.height = '3px';
                    });
                }
            });
            
            // Update time display during playback
            audioElement.addEventListener('timeupdate', updatePlaybackTime);
            
            // Load initial audio
            loadAudioForVoice(currentVoice);
        } else {
            console.error('Audio player is missing required elements');
        }
    });
}

/**
 * Show error message in player
 */
function showError(playerElement, message) {
    // Check if error container already exists
    let errorContainer = playerElement.querySelector('.error');
    
    if (!errorContainer) {
        // Create error container if it doesn't exist
        errorContainer = document.createElement('div');
        errorContainer.className = 'error';
        playerElement.appendChild(errorContainer);
    }
    
    // Set error message
    errorContainer.textContent = message;
    
    // Remove loading indicator
    playerElement.classList.remove('loading');
}

/**
 * Clear all error messages
 */
function clearErrors(playerElement) {
    const errors = playerElement.querySelectorAll('.error');
    errors.forEach(error => error.remove());
}