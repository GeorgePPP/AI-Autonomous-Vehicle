/**
 * Chat-related utility functions
 */

// Function to add a message to the chat
function addMessage(message, isUser, chatMessages, typingIndicator) {
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
function showTypingIndicator(typingIndicator, chatMessages) {
    typingIndicator.style.display = 'block';
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Function to hide typing indicator
function hideTypingIndicator(typingIndicator) {
    typingIndicator.style.display = 'none';
}

// Function to send message to the server
async function sendMessageToServer(message) {
    // Create form data for the request
    const formData = new FormData();
    formData.append('user_input', message);
    
    // Send request to get response
    const response = await fetch('/get_response', {
        method: 'POST',
        body: formData
    });
    
    return await response.json();
}