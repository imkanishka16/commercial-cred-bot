// const chatBox = document.getElementById('chatBox');
// const userInput = document.getElementById('userInput');
// const AUTH_TOKEN = 'Ab@123';

// function formatResponse(responseText) {
//     // Split the response into lines
//     const lines = responseText.split('\n');
//     const formattedLines = lines.map(line => {
//         // Trim the line to handle extra spaces
//         line = line.trim();
//         // Check if the line is a list item (starts with '-')
//         if (line.startsWith('-')) {
//             // Remove the '-' and wrap the rest in <strong> for bold
//             const content = line.substring(1).trim();
//             return `<strong>- ${content}</strong>`;
//         }
//         return line;
//     });
//     // Join lines with <br> for newlines
//     return formattedLines.join('<br>');
// }

// function sendMessage() {
//     const question = userInput.value.trim();
//     if (!question) return;

//     // Add user message to chat
//     const userMessage = document.createElement('div');
//     userMessage.className = 'message user';
//     userMessage.innerHTML = `<p>${question}</p>`;
//     chatBox.appendChild(userMessage);

//     // Clear input
//     userInput.value = '';

//     // Show loading message
//     const loadingMessage = document.createElement('div');
//     loadingMessage.className = 'message bot';
//     loadingMessage.innerHTML = '<p>Thinking...</p>';
//     chatBox.appendChild(loadingMessage);
//     chatBox.scrollTop = chatBox.scrollHeight;

//     // Send request to backend
//     fetch('/query', {
//         method: 'POST',
//         headers: {
//             'Content-Type': 'application/json',
//             'Authorization': `Bearer ${AUTH_TOKEN}`
//         },
//         body: JSON.stringify({ question: question })
//     })
//     .then(response => response.json())
//     .then(data => {
//         // Remove loading message
//         chatBox.removeChild(loadingMessage);

//         // Add bot response with formatted text
//         const botMessage = document.createElement('div');
//         botMessage.className = 'message bot';
//         const formattedAnswer = formatResponse(data.answer || 'Error: No response');
//         botMessage.innerHTML = `<p>${formattedAnswer}</p>`;
//         chatBox.appendChild(botMessage);
//         chatBox.scrollTop = chatBox.scrollHeight;
//     })
//     .catch(error => {
//         console.error('Error:', error);
//         chatBox.removeChild(loadingMessage);
//         const errorMessage = document.createElement('div');
//         errorMessage.className = 'message bot';
//         errorMessage.innerHTML = '<p>Error fetching response</p>';
//         chatBox.appendChild(errorMessage);
//         chatBox.scrollTop = chatBox.scrollHeight;
//     });
// }

// userInput.addEventListener('keypress', (e) => {
//     if (e.key === 'Enter') {
//         sendMessage();
//     }
// });

// window.addEventListener('resize', () => {
//     chatBox.style.maxHeight = `${window.innerHeight - 250}px`;
// });
// chatBox.style.maxHeight = `${window.innerHeight - 250}px`; // Initial setting


const chatBox = document.getElementById('chatBox');
const userInput = document.getElementById('userInput');
const AUTH_TOKEN = 'Ab@123';

function formatResponse(responseText) {
    // Split the response into lines
    const lines = responseText.split('\n');
    let formattedLines = [];
    let inList = false;
    let listLevel = 0; // Track nesting level for lists

    lines.forEach(line => {
        // Trim the line to handle extra spaces
        line = line.trim();
        // Skip empty lines
        if (!line) return;

        // Check if the line starts with a Roman numeral in parentheses (e.g., (i), (ii))
        const romanNumeralMatch = line.match(/^\((i{1,3}|iv|v|vi{0,3}|viii|ix|x)\)/i);
        // Check for uppercase Roman numerals with a dot (e.g., I., II.)
        const upperRomanNumeralMatch = line.match(/^(I{1,3}|IV|V|VI{0,3}|VIII|IX|X)\./i);
        if (romanNumeralMatch || upperRomanNumeralMatch) {
            // Determine the type of Roman numeral
            const isUpper = !!upperRomanNumeralMatch;
            const match = romanNumeralMatch || upperRomanNumeralMatch;
            const prefix = match[0];

            // Start a new list if not already in one
            if (!inList) {
                formattedLines.push('<ul class="info-list">');
                inList = true;
                listLevel++;
            }
            // Preserve the Roman numeral and wrap the rest in <li>
            const content = line.substring(prefix.length).trim();
            const listClass = isUpper ? 'upper-roman' : 'lower-roman';
            formattedLines.push(`<li class="roman-numeral ${listClass}">${prefix} ${content}</li>`);
        } else {
            // If we were in a list, close it
            if (inList) {
                formattedLines.push('</ul>');
                inList = false;
                listLevel--;
            }
            // Check if the line starts with a letter followed by a parenthesis (e.g., a), b))
            const letterMatch = line.match(/^[a-z]\)/i);
            if (letterMatch) {
                // Wrap section headers in <p> with a class for styling
                formattedLines.push(`<p class="section-header">${line}</p>`);
            } else {
                // Wrap other lines in <p> tags with a class for styling
                formattedLines.push(`<p class="section-text">${line}</p>`);
            }
        }
    });

    // Close any open lists
    while (listLevel > 0) {
        formattedLines.push('</ul>');
        listLevel--;
    }

    // Filter out empty lines and join
    return formattedLines.filter(line => line !== '').join('');
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
        botMessage.innerHTML = formattedAnswer;
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