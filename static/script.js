// Core variables
let socket = null;
let MAX_RECORDING_DURATION = 60;
let mediaRecorder = null;
let audioChunks = [];
let recordingTimer;
let recordingSeconds = 0;

// Initialize WebSocket with reconnection logic
function initializeWebSocket(sessionId) {
    if (!sessionId) {
        console.error("No session ID provided for WebSocket connection");
        showMessage("Error: Session ID missing. Please refresh the page.", 'system-message');
        return;
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/${sessionId}`;
    
    console.log(`Initializing WebSocket connection to ${wsUrl}`);
    
    // Close existing socket if it exists
    if (socket && socket.readyState !== WebSocket.CLOSED) {
        console.log("Closing existing WebSocket connection");
        socket.close();
    }
    
    socket = new WebSocket(wsUrl);
    
    // Only send messages after the connection is open
    socket.onopen = function() {
        console.log("WebSocket connected successfully");
        // Now it's safe to send messages
        socket.send(JSON.stringify({ type: 'get_config' }));
    };
    
    socket.onmessage = handleWebSocketMessage;
    
    socket.onclose = function(event) {
        console.log(`WebSocket disconnected with code: ${event.code}, reason: ${event.reason}`);
    };
    
    socket.onerror = function(error) {
        console.error("WebSocket error:", error);
        showMessage("Connection error. Attempting to reconnect...", 'system-message', 5000);
    };
}

// Handle incoming WebSocket messages
function handleWebSocketMessage(event) {
    try {
        const data = JSON.parse(event.data);
        console.log("Received WebSocket message:", data.type);
        
        switch (data.type) {
            case 'chat_updated':
                if (data.success) {
                    updateChatMessages();
                } else {
                    console.warn("Chat update reported failure");
                    showMessage("Failed to update chat. Please try again.", 'system-message', 5000);
                }
                break;
                
            case 'config':
                // Update configuration
                console.log("Received config:", data);
                if (data.recording?.maxDuration) {
                    MAX_RECORDING_DURATION = data.recording.maxDuration;
                    console.log(`Set max recording duration to ${MAX_RECORDING_DURATION}s`);
                }

                updateUIForAudioInput();

                break;

            default:
                console.log("Unhandled message type:", data.type);
        }
    } catch (error) {
        console.error("Error parsing WebSocket message:", error);
        console.error("Raw message:", event.data);
    }
}

// Update chat messages via HTMX
function updateChatMessages() {
    const formData = new FormData(document.getElementById('chatForm'));
    const sessionId = formData.get('session_id');
    
    if (!sessionId) {
        console.error("No session ID found for updating chat messages");
        return;
    }
    
    console.log("Updating chat messages for session:", sessionId);
    
    htmx.ajax('GET', '/_chat_messages.html', { 
        target: '#chatMessages',
        swap: 'innerHTML',
        values: { session_id: sessionId },
        indicator: '#typingIndicator',
        error: function(xhr) {
            console.error("Error loading chat messages:", xhr.status, xhr.statusText);
            showMessage("Error loading chat messages. Please refresh the page.", 'system-message');
        }
    });
}
function updateUIForAudioInput() {
    const chatInput = document.querySelector('.chat-input');
    if (chatInput) chatInput.style.display = 'none';
    
    const recordButton = document.getElementById('recordButton');
    if (recordButton) recordButton.classList.add('audio-only-button');
}
// Toggle audio recording
function toggleRecording(recordButton, recordingIndicator) {
    if (!recordButton || !recordingIndicator) {
        console.error("Missing required elements for recording");
        return;
    }
    
    const recordingTime = recordingIndicator.querySelector('.recording-time');
    if (!recordingTime) {
        console.error("Recording time element not found");
        return;
    }
    
    // Stop recording if already in progress
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        console.log("Stopping active recording");
        stopRecording(recordButton, recordingIndicator);
        return;
    }
    
    console.log("Starting audio recording");
    
    // Start recording
    navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => {
            // Determine best supported format
            let mimeType = '';
            if (MediaRecorder.isTypeSupported('audio/webm')) {
                mimeType = 'audio/webm';
            } else if (MediaRecorder.isTypeSupported('audio/mp4')) {
                mimeType = 'audio/mp4';
            }
            
            console.log(`Using audio MIME type: ${mimeType || 'browser default'}`);
            const recorderOptions = mimeType ? { mimeType } : {};
            
            // Initialize recorder
            mediaRecorder = new MediaRecorder(stream, recorderOptions);
            audioChunks = [];
            
            // Set up data collection
            mediaRecorder.addEventListener('dataavailable', event => {
                audioChunks.push(event.data);
                console.log(`Audio chunk received: ${(event.data.size / 1024).toFixed(2)} KB`);
            });
            
            // Handle recording stop
            mediaRecorder.addEventListener('stop', () => {
                console.log("Recording stopped, processing audio...");
                handleRecordingEnd(mimeType, stream);
            });
            
            // Handle errors
            mediaRecorder.addEventListener('error', error => {
                console.error("MediaRecorder error:", error);
                showMessage("Error recording audio. Please try again.", 'system-message');
                stopRecording(recordButton, recordingIndicator);
            });
            
            // Set maximum recording duration
            const maxDurationTimeout = setTimeout(() => {
                if (mediaRecorder && mediaRecorder.state === 'recording') {
                    console.log("Maximum recording duration reached");
                    stopRecording(recordButton, recordingIndicator);
                    showMessage("Maximum recording duration reached.", 'system-message', 3000);
                }
            }, MAX_RECORDING_DURATION * 1000);
            
            // Start the recorder
            mediaRecorder.start(1000); // Collect data every second
            console.log("Recording started");
            
            // Update UI
            recordButton.classList.add('recording');
            recordingIndicator.style.display = 'flex';
            recordButton.querySelector('.record-text').textContent = 'Stop Recording';
            
            // Start the timer
            startRecordingTimer(recordingTime);
        })
        .catch(error => {
            console.error('Error accessing microphone:', error);
            showMessage('Error: Could not access microphone. Please check your browser permissions.', 'system-message');
        });
}

// Start the recording timer
function startRecordingTimer(recordingTime) {
    recordingSeconds = 0;
    clearInterval(recordingTimer);
    
    // Remove warning class if it exists from a previous recording
    recordingTime.classList.remove('time-warning');
    
    recordingTimer = setInterval(() => {
        recordingSeconds++;
        const minutes = Math.floor(recordingSeconds / 60);
        const seconds = recordingSeconds % 60;
        recordingTime.textContent = `${minutes}:${seconds.toString().padStart(2, '0')}`;
        
        // Show warning when approaching the limit
        if (recordingSeconds >= MAX_RECORDING_DURATION * 0.8) {
            recordingTime.classList.add('time-warning');
        }
        
        // Log for debugging long recordings
        if (recordingSeconds % 10 === 0) {
            console.log(`Recording in progress: ${recordingSeconds}s`);
        }
    }, 1000);
}

// Stop recording
function stopRecording(recordButton, recordingIndicator) {
    if (!mediaRecorder) {
        console.warn("No active mediaRecorder found");
        return;
    }
    
    if (mediaRecorder.state === 'recording') {
        console.log("Stopping recording");
        mediaRecorder.stop();
        
        // Update UI
        recordButton.classList.remove('recording');
        recordingIndicator.style.display = 'none';
        clearInterval(recordingTimer);
        recordingSeconds = 0;
        recordButton.querySelector('.record-text').textContent = 'Record Audio';
    } else {
        console.log(`MediaRecorder in state: ${mediaRecorder.state}, not stopping`);
    }
}

// Handle the end of recording
function handleRecordingEnd(mimeType, stream) {
    // Validate we have audio data
    if (!audioChunks.length) {
        console.error("No audio chunks recorded");
        showMessage("No audio recorded. Please try again.", 'system-message', 3000);
        return;
    }
    
    // Show processing message
    showMessage("[Processing your audio...]", 'user-message', 0, 'temp-processing-message');
    
    // Create audio blob
    const audioFormat = mimeType ? mimeType.split('/')[1] : 'wav';
    const audioBlob = new Blob(audioChunks, { type: mimeType || 'audio/webm' });
    
    console.log(`Audio recording complete: ${(audioBlob.size / 1024).toFixed(2)} KB ${audioFormat}`);
    
    // Convert to base64
    const reader = new FileReader();
    reader.readAsDataURL(audioBlob);
    
    reader.onloadend = function() {
        const base64Data = reader.result.split(',')[1];
        document.getElementById('typingIndicator').style.display = 'block';
        
        // Check socket connection before sending
        if (socket && socket.readyState === WebSocket.OPEN) {
            console.log("Sending audio data to server");
            
            socket.send(JSON.stringify({
                type: 'audio_recorded',
                audio_data: base64Data,
                format: audioFormat,
                duration: recordingSeconds
            }));
            
            // Remove temporary message
            const tempMessage = document.getElementById('temp-processing-message');
            if (tempMessage) tempMessage.remove();
        } else {
            const state = socket ? ["CONNECTING", "OPEN", "CLOSING", "CLOSED"][socket.readyState] : "No socket";
            console.error(`WebSocket not ready. State: ${state}`);
            
            showMessage("Error: Connection to server lost. Please refresh the page.", 'system-message');
            
            // Try to reconnect
            const sessionIdInput = document.querySelector('input[name="session_id"]');
            if (sessionIdInput) {
                initializeWebSocket(sessionIdInput.value);
            }
        }
    };
    
    reader.onerror = function() {
        console.error("Error reading audio data");
        showMessage("Error processing audio. Please try again.", 'system-message', 3000);
    };
    
    // Stop all tracks
    stream.getTracks().forEach(track => {
        track.stop();
        console.log(`Audio track ${track.id} stopped`);
    });
}

// Helper to show messages in the chat
function showMessage(text, className, autoRemoveMs = 0, id = null) {
    const chatMessages = document.getElementById('chatMessages');
    if (!chatMessages) {
        console.error("Chat messages container not found");
        return;
    }
    
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', className);
    if (id) messageDiv.id = id;
    messageDiv.textContent = text;
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    if (autoRemoveMs > 0) {
        setTimeout(() => {
            if (messageDiv.parentNode === chatMessages) {
                messageDiv.remove();
            }
        }, autoRemoveMs);
    }
    
    return messageDiv;
}

// Main initialization
document.addEventListener('DOMContentLoaded', function() {
    console.log("Initializing audio chat application");
    
    const chatForm = document.getElementById('chatForm');
    const userInput = document.getElementById('userInput');
    const recordButton = document.getElementById('recordButton');
    const recordingIndicator = document.getElementById('recordingIndicator');
    const typingIndicator = document.getElementById('typingIndicator');
    
    // Form handling
    if (chatForm && userInput) {
        chatForm.addEventListener('htmx:configRequest', function(event) {
            // Prevent empty text submissions when not using audio
            if (!userInput.value.trim() && event.detail.parameters['use_audio'] === 'false') {
                console.log("Preventing submission of empty text message");
                event.preventDefault();
                return;
            }
            
            if (typingIndicator) typingIndicator.style.display = 'block';
        });
        
        chatForm.addEventListener('htmx:afterSwap', function() {
            // Clear input and hide typing indicator
            userInput.value = '';
            
            if (typingIndicator) typingIndicator.style.display = 'none';
            
            // Scroll to bottom
            const chatMessages = document.getElementById('chatMessages');
            if (chatMessages) chatMessages.scrollTop = chatMessages.scrollHeight;
            
            // Auto-play audio elements
            document.querySelectorAll('audio[data-autoplay="true"]:not([data-played="true"])').forEach(audio => {
                audio.play()
                    .then(() => {
                        console.log("Audio playback started");
                        audio.setAttribute('data-played', 'true');
                    })
                    .catch(error => {
                        console.error("Auto-play failed:", error);
                        showMessage("Audio playback failed. Click the audio player to listen.", 'system-message', 5000);
                    });
            });
        });
    }
    
    // Audio recording button
    if (recordButton) {
        recordButton.addEventListener('click', function() {
            toggleRecording(recordButton, recordingIndicator);
        });
    }
    
    // Keyboard shortcuts
    document.addEventListener('keydown', function(event) {
        // Submit on Enter
        if (event.key === 'Enter' && document.activeElement === userInput) {
            event.preventDefault();
            if (userInput.value.trim()) {
                chatForm.dispatchEvent(new Event('submit'));
            }
        }
        
        // Stop recording on Escape
        if (event.key === 'Escape' && mediaRecorder && mediaRecorder.state === 'recording') {
            stopRecording(recordButton, recordingIndicator);
        }
    });
    
    // Initialize WebSocket
    const sessionIdInput = document.querySelector('input[name="session_id"]');
    if (sessionIdInput) {
        initializeWebSocket(sessionIdInput.value);
    } else {
        console.error("No session ID input found, WebSocket cannot be initialized");
    }
});

// Clean up on page unload
window.addEventListener('beforeunload', function() {
    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.close(1000, "Page unloaded");
        console.log("WebSocket closed due to page unload");
    }
    
    // Stop any ongoing recording
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
        console.log("Recording stopped due to page unload");
    }
});