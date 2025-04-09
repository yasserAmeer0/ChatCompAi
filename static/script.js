function sendMessage() {
    const input = document.getElementById('message-input');
    const message = input.value.trim();
    if (!message) return;

    // Display user message
    const chatHistory = document.getElementById('chat-history');
    const userDiv = document.createElement('div');
    userDiv.className = 'message user-message';
    userDiv.textContent = 'You: ' + message;
    chatHistory.appendChild(userDiv);

    // Send message to backend
    fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: message })
    })
    .then(response => response.json())
    .then(data => {
        // Display AI response
        const aiDiv = document.createElement('div');
        aiDiv.className = 'message ai-message';
        aiDiv.textContent = data.response;
        chatHistory.appendChild(aiDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight; // Auto-scroll to bottom
    })
    .catch(error => {
        console.error('Error:', error);
        const errorDiv = document.createElement('div');
        errorDiv.className = 'message ai-message';
        errorDiv.textContent = 'AI: Error occurred while processing your request.';
        chatHistory.appendChild(errorDiv);
    });

    // Clear input
    input.value = '';
}

// Allow sending message with Enter key
document.getElementById('message-input').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        sendMessage();
    }
});