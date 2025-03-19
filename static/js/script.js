const chatBox = document.getElementById('chatBox');
const userInput = document.getElementById('userInput');
const AUTH_TOKEN = 'Ab@123';

function formatResponse(responseText) {
    // Split the response into lines
    const lines = responseText.split('\n');
    const formattedLines = lines.map(line => {
        // Trim the line to handle extra spaces
        line = line.trim();
        // Check if the line is a list item (starts with '-')
        if (line.startsWith('-')) {
            // Remove the '-' and wrap the rest in <strong> for bold
            const content = line.substring(1).trim();
            return `<strong>- ${content}</strong>`;
        }
        return line;
    });
    // Join lines with <br> for newlines
    return formattedLines.join('<br>');
}

function sendMessage() {
    const question = userInput.value.trim();
    if (!question) return;

    // Add user message to chat
    const userMessage = document.createElement('div');
    userMessage.className = 'message user';
    userMessage.innerHTML = `<p>${question}</p>`;
    chatBox.appendChild(userMessage);

    // Clear input
    userInput.value = '';

    // Show loading message
    const loadingMessage = document.createElement('div');
    loadingMessage.className = 'message bot';
    loadingMessage.innerHTML = '<p>Thinking...</p>';
    chatBox.appendChild(loadingMessage);
    chatBox.scrollTop = chatBox.scrollHeight;

    // Send request to backend
    fetch('/query', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${AUTH_TOKEN}`
        },
        body: JSON.stringify({ question: question })
    })
    .then(response => response.json())
    .then(data => {
        // Remove loading message
        chatBox.removeChild(loadingMessage);

        // Add bot response with formatted text
        const botMessage = document.createElement('div');
        botMessage.className = 'message bot';
        const formattedAnswer = formatResponse(data.answer || 'Error: No response');
        botMessage.innerHTML = `<p>${formattedAnswer}</p>`;
        chatBox.appendChild(botMessage);
        chatBox.scrollTop = chatBox.scrollHeight;
    })
    .catch(error => {
        console.error('Error:', error);
        chatBox.removeChild(loadingMessage);
        const errorMessage = document.createElement('div');
        errorMessage.className = 'message bot';
        errorMessage.innerHTML = '<p>Error fetching response</p>';
        chatBox.appendChild(errorMessage);
        chatBox.scrollTop = chatBox.scrollHeight;
    });
}

userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

window.addEventListener('resize', () => {
    chatBox.style.maxHeight = `${window.innerHeight - 250}px`;
});
chatBox.style.maxHeight = `${window.innerHeight - 250}px`; // Initial setting