document.addEventListener('DOMContentLoaded', function() {
    const chatForm = document.getElementById('chatForm');
    const userInput = document.getElementById('userInput');
    const chatMessages = document.getElementById('chatMessages');
    const typingIndicator = document.getElementById('typingIndicator');
    
    // Function to add a message to the chat
    function addMessage(message, isUser) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message');
        messageDiv.classList.add(isUser ? 'user-message' : 'bot-message');
        messageDiv.textContent = message;
        
        // Insert before typing indicator
        chatMessages.insertBefore(messageDiv, typingIndicator);
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        return messageDiv;
    }
    
    // Function to show typing indicator
    function showTypingIndicator() {
        typingIndicator.style.display = 'block';
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // Function to hide typing indicator
    function hideTypingIndicator() {
        typingIndicator.style.display = 'none';
    }
    
    // Handle form submission
    chatForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const message = userInput.value.trim();
        if (!message) return;
        
        // Add user message to chat
        addMessage(message, true);
        
        // Clear input field
        userInput.value = '';
        
        // Show typing indicator
        showTypingIndicator();
        
        try {
            // Create form data for the request
            const formData = new FormData();
            formData.append('user_input', message);
            
            // Send request to get response
            const response = await fetch('/get_response', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            // Hide typing indicator
            hideTypingIndicator();
            
            // if (data.response) {
            //         // Add bot response to chat
            //         const botMessageDiv = addMessage(data.response, false);
            // }

            const botMessageDiv = addMessage(data.response, false);
            
            // Add audio player if audio is availables
            if (data.audio_base64) {
                playBase64Audio(data.audio_base64);
            }
        } catch (error) {
            // Hide typing indicator
            hideTypingIndicator();
            
            // Add error message
            addMessage("Sorry, I encountered an error. Please try again.", false);
            console.error('Error:', error);
        }
    });
});

// Function to play Audio
function playBase64Audio(base64String) {
    // Create an AudioContext
    const context = new (window.AudioContext || window.webkitAudioContext)();

    // Convert Base64 string to binary data
    let binaryString = atob(base64String);
    let len = binaryString.length;
    let bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }

    // Decode the audio data and play
    context.decodeAudioData(bytes.buffer, (audioBuffer) => {
        let source = context.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(context.destination);
        source.start(0);
    }, (error) => {
        console.error("Error decoding audio:", error);
    });
}