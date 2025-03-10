/**
 * Minimal JavaScript for ND II Chat Interface
 */

// Core variables
let socket = null;
let MAX_RECORDING_DURATION = 60;
let AUDIO_ONLY_MODE = true;
let mediaRecorder;
let audioChunks = [];
let recordingTimer;
let recordingSeconds = 0;

// Initialize WebSocket
function initializeWebSocket(sessionId) {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/${sessionId}`;
    
    socket = new WebSocket(wsUrl);
    
    // Only send messages after the connection is open
    socket.onopen = function() {
        console.log("WebSocket connected");
        // Now it's safe to send messages
        socket.send(JSON.stringify({ type: 'get_config' }));
    };
    
    socket.onmessage = handleWebSocketMessage;
    
    socket.onclose = function() {
        console.log("WebSocket disconnected");
    };
    
    socket.onerror = function(error) {
        console.error("WebSocket error:", error);
    };
}

// Handle incoming WebSocket messages
function handleWebSocketMessage(event) {
    const data = JSON.parse(event.data);
    
    if (data.type === 'chat_updated' && data.success) {
        // Small delay to ensure server processing is complete
        setTimeout(updateChatMessages, 100);
        updateChatMessages();
    } else if (data.type === 'config') {
        // Update configuration
        if (data.recording?.maxDuration) {
            MAX_RECORDING_DURATION = data.recording.maxDuration;
        }
        
        if (data.audio_only_mode !== undefined) {
            AUDIO_ONLY_MODE = data.audio_only_mode;
            updateUIForAudioOnlyMode();
        }
    } else if (data.type === 'error') {
        console.error("Server error:", data.message);
        showMessage("Error: " + data.message, 'system-message');
    }
}

// Update chat messages via HTMX
function updateChatMessages() {
    const formData = new FormData(document.getElementById('chatForm'));
    htmx.ajax('GET', '/_chat_messages.html', { 
        target: '#chatMessages',
        swap: 'innerHTML',
        values: { session_id: formData.get('session_id') },
        error: function(xhr) {
            console.error("Error loading chat messages:", xhr.status);
            showMessage("Error loading chat messages. Please refresh the page.", 'system-message');
        }
    });
}

// Update UI for audio-only mode
function updateUIForAudioOnlyMode() {
    if (AUDIO_ONLY_MODE) {
        const chatInput = document.querySelector('.chat-input');
        if (chatInput) chatInput.style.display = 'none';
        
        const recordButton = document.getElementById('recordButton');
        if (recordButton) recordButton.classList.add('audio-only-button');
    }
}

// Toggle audio recording
function toggleRecording(recordButton, recordingIndicator) {
    const recordingTime = recordingIndicator.querySelector('.recording-time');
    
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        stopRecording(recordButton, recordingIndicator);
        return;
    }
    
    // Start recording
    navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => {
            // Determine best supported format
            const mimeType = MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' : '';
            const recorderOptions = mimeType ? { mimeType } : {};
            
            // Initialize recorder
            mediaRecorder = new MediaRecorder(stream, recorderOptions);
            audioChunks = [];
            
            // Set up data collection
            mediaRecorder.addEventListener('dataavailable', event => {
                audioChunks.push(event.data);
            });
            
            // Handle recording stop
            mediaRecorder.addEventListener('stop', () => handleRecordingEnd(mimeType, stream));
            
            // Set maximum recording duration
            setTimeout(() => {
                if (mediaRecorder && mediaRecorder.state === 'recording') {
                    stopRecording(recordButton, recordingIndicator);
                    showMessage("Maximum recording duration reached.", 'system-message', 3000);
                }
            }, MAX_RECORDING_DURATION * 1000);
            
            // Start the recorder
            mediaRecorder.start();
            
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
    
    recordingTimer = setInterval(() => {
        recordingSeconds++;
        const minutes = Math.floor(recordingSeconds / 60);
        const seconds = recordingSeconds % 60;
        recordingTime.textContent = `${minutes}:${seconds.toString().padStart(2, '0')}`;
        
        if (recordingSeconds >= MAX_RECORDING_DURATION * 0.8) {
            recordingTime.classList.add('time-warning');
        }
    }, 1000);
}

// Stop recording
function stopRecording(recordButton, recordingIndicator) {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
        recordButton.classList.remove('recording');
        recordingIndicator.style.display = 'none';
        clearInterval(recordingTimer);
        recordingSeconds = 0;
        recordButton.querySelector('.record-text').textContent = 'Record Audio';
    }
}

// Handle the end of recording
function handleRecordingEnd(mimeType, stream) {
    // Show processing message in audio-only mode
    if (AUDIO_ONLY_MODE) {
        showMessage("[Processing your audio...]", 'user-message', 0, 'temp-processing-message');
    }
    
    // Create audio blob
    const audioFormat = mimeType ? mimeType.split('/')[1] : 'wav';
    const audioBlob = new Blob(audioChunks, { type: mimeType || 'audio/webm' });
    
    // Convert to base64
    const reader = new FileReader();
    reader.readAsDataURL(audioBlob);
    reader.onloadend = function() {
        document.getElementById('typingIndicator').style.display = 'block';
        
        // Send only if socket is ready
        if (socket && socket.readyState === WebSocket.OPEN) {
            socket.send(JSON.stringify({
                type: 'audio_recorded',
                audio_data: reader.result.split(',')[1],
                format: audioFormat
            }));
            
            // Remove temporary message
            const tempMessage = document.getElementById('temp-processing-message');
            if (tempMessage) tempMessage.remove();
        } else {
            console.error("WebSocket not ready. State:", socket ? socket.readyState : "No socket");
            showMessage("Error: Connection to server lost. Please refresh the page.", 'system-message');
        }
    };
    
    // Stop all tracks
    stream.getTracks().forEach(track => track.stop());
}

// Helper to show messages in the chat
function showMessage(text, className, autoRemoveMs = 0, id = null) {
    const chatMessages = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', className);
    if (id) messageDiv.id = id;
    messageDiv.textContent = text;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    if (autoRemoveMs > 0) {
        setTimeout(() => messageDiv.remove(), autoRemoveMs);
    }
}

// Main initialization
document.addEventListener('DOMContentLoaded', function() {
    const chatForm = document.getElementById('chatForm');
    const userInput = document.getElementById('userInput');
    const recordButton = document.getElementById('recordButton');
    const recordingIndicator = document.getElementById('recordingIndicator');
    
    // Form handling
    if (chatForm && userInput) {
        chatForm.addEventListener('htmx:configRequest', function(event) {
            if (!userInput.value.trim() && event.detail.parameters['use_audio'] === 'false') {
                event.preventDefault();
                return;
            }
            document.getElementById('typingIndicator').style.display = 'block';
        });
        
        chatForm.addEventListener('htmx:afterSwap', function() {
            userInput.value = '';
            document.getElementById('typingIndicator').style.display = 'none';
            
            const chatMessages = document.getElementById('chatMessages');
            chatMessages.scrollTop = chatMessages.scrollHeight;
            
            // Auto-play audio elements
            document.querySelectorAll('audio[data-autoplay="true"]:not([data-played="true"])').forEach(audio => {
                audio.play().catch(e => console.error("Auto-play failed:", e));
                audio.setAttribute('data-played', 'true');
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
    }
});

// Clean up on page unload
window.addEventListener('beforeunload', function() {
    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.close();
    }
});