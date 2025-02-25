/**
 * Main application script
 */

document.addEventListener('DOMContentLoaded', function() {
    // Get DOM elements
    const chatForm = document.getElementById('chatForm');
    const userInput = document.getElementById('userInput');
    const chatMessages = document.getElementById('chatMessages');
    const typingIndicator = document.getElementById('typingIndicator');
    
    // Handle form submission
    chatForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const message = userInput.value.trim();
        if (!message) return;
        
        // Add user message to chat
        addMessage(message, true, chatMessages, typingIndicator);
        
        // Clear input field
        userInput.value = '';
        
        // Show typing indicator
        showTypingIndicator(typingIndicator, chatMessages);
        
        try {
            // Send request to get response
            const data = await sendMessageToServer(message);
            
            // Hide typing indicator
            hideTypingIndicator(typingIndicator);
            
            // Add bot response to chat
            const botMessageDiv = addMessage(data.response, false, chatMessages, typingIndicator);
            const visualizer = createModernVisualizer();
            botMessageDiv.appendChild(visualizer);
            
            // Add audio player if audio is available
            if (data.audio_base64) {
                
                // Play audio with visualizer
                playBase64AudioWithVisualizer(data.audio_base64, visualizer);
            }
        } catch (error) {
            // Hide typing indicator
            hideTypingIndicator(typingIndicator);
            
            // Add error message
            addMessage("Sorry, I encountered an error. Please try again.", false, chatMessages, typingIndicator);
            console.error('Error:', error);
        }
    });
});