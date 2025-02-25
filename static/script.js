// You can move the JavaScript from index.html to this file if you prefer
// Just add a script tag in index.html: <script src="/static/script.js"></script>

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
            
            // Add bot response to chat
            const botMessageDiv = addMessage(data.response, false);
            
            // Add audio player if audio is available
            if (data.audio_path) {
                const audioPlayer = document.createElement('audio');
                audioPlayer.controls = true;
                audioPlayer.classList.add('audio-player');
                audioPlayer.src = data.audio_path;
                botMessageDiv.appendChild(audioPlayer);
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