document.addEventListener('DOMContentLoaded', function() {
    console.log("DOM fully loaded and parsed");

    const chatToggle = document.getElementById('chat-toggle');
    const chatContainer = document.getElementById('chat-container');
    const chatHeader = document.getElementById('chat-header');
    const chatInput = document.getElementById('chat-input');
    const chatContent = document.getElementById('chat-content');

    if (!chatToggle || !chatContainer || !chatHeader || !chatInput || !chatContent) {
        console.error("One or more chat elements are missing. Make sure HTML elements are correctly defined.");
        return;
    }

    chatToggle.style.background = "linear-gradient(45deg, #8a2be2, #ff69b4)";
    chatToggle.style.color = "#fff";
    chatToggle.style.border = "none";
    chatToggle.style.borderRadius = "5px";
    chatToggle.style.boxShadow = "0 4px 8px rgba(0, 0, 0, 0.3)";
    chatToggle.style.transition = "box-shadow 0.3s ease";

    chatToggle.addEventListener('mouseenter', () => {
        chatToggle.style.boxShadow = "0 6px 12px rgba(0, 0, 0, 0.5)";
    });

    chatToggle.addEventListener('mouseleave', () => {
        chatToggle.style.boxShadow = "0 4px 8px rgba(0, 0, 0, 0.3)";
    });

    chatContainer.style.display = 'none';
    chatContainer.style.transition = 'all 0.5s ease';
    chatContainer.style.zIndex = '9999';
    chatToggle.style.zIndex = '10000';

    loadChatHistory();

    chatToggle.addEventListener('click', () => {
        console.log("Chat toggle clicked");
        if (chatContainer.style.display === 'none') {
            chatContainer.style.display = 'block';
            suggestRandomQuestions();
            scrollToBottom(); // Ensure it scrolls to the bottom after opening
        } else {
            chatContainer.style.display = 'none';
        }
    });

    chatHeader.addEventListener('click', () => {
        console.log("Chat header clicked to minimize");
        chatContainer.style.display = 'none';
    });

    chatInput.addEventListener('keypress', async (e) => {
        if (e.key === 'Enter' && chatInput.value.trim() !== '') {
            console.log("User pressed enter");
            const userMessage = chatInput.value.trim();
            chatInput.value = '';

            const userMessageElement = createMessageElement(userMessage, 'right', '#ff69b4');
            userMessageElement.style.fontSize = '1rem'; // Increase font size slightly
            chatContent.appendChild(userMessageElement);
            scrollToBottom(); // Ensure it scrolls to the bottom after user message
            saveChatHistory();

            // Prepare bot message element with loading animation
            const botMessageElement = document.createElement('div');
            botMessageElement.style.textAlign = 'left';
            botMessageElement.style.color = '#fff';
            botMessageElement.style.fontSize = '1rem';
            botMessageElement.style.lineHeight = '1.6';
            botMessageElement.classList.add('bot-message');
            botMessageElement.innerHTML = 'Bot is typing<span class="dots">...</span>';
            chatContent.appendChild(botMessageElement);
            scrollToBottom(); // Scroll after adding "Bot is typing..."

            // Animation for dots
            let dotsInterval = setInterval(() => {
                const dots = botMessageElement.querySelector('.dots');
                if (dots) {
                    dots.textContent = dots.textContent.length < 3 ? dots.textContent + '.' : '.';
                }
            }, 500);

            try {
                const response = await fetch('https://mackerel-striking-unduly.ngrok-free.app', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question: userMessage })
                });

                const data = await response.json();
                let botResponse = '';

                if (response.ok && data.answer) {
                    botResponse = data.answer;
                    console.log("Received bot response:", botResponse);
                } else {
                    botResponse = 'An error occurred: ' + (data.error || 'Unknown error');
                    console.warn("Bot response error:", botResponse);
                }

                clearInterval(dotsInterval); // Stop the loading animation
                botMessageElement.innerHTML = botResponse; // Replace loading text with the actual response
                scrollToBottom(); // Ensure it scrolls to bottom after appending bot response
                saveChatHistory();
            } catch (error) {
                console.error('Error:', error);
                clearInterval(dotsInterval); // Stop the loading animation
                botMessageElement.innerHTML = 'Unable to reach the server.';
                scrollToBottom(); // Ensure it scrolls to bottom after appending error message
                saveChatHistory();
            }

            // Limit chat history to last 3 messages per type (user and bot)
            limitChatHistory();
        }
    });

    function scrollToBottom() {
        chatContent.scrollTop = chatContent.scrollHeight;
    }
    // Suggest two random questions functionality
    const questions = [
        "What is Anish's work experience as a Data Scientist?",
        "What data science and machine learning skills does Anish have?",
        "What courses did Anish complete in his AI program at Conestoga College?",
        "What did Anish study in the Predictive Analytics program at Conestoga College?",
        "What were Anish’s main subjects in the Data Science program at Great Lakes Institute?",
        "What degree did Anish earn at Sathyabama Institute, and what was his focus?",
        "What certifications does Anish hold?",
        "What awards has Anish received, and for what achievements?"
    ];

    function suggestRandomQuestions() {
        const suggestionContainer = document.createElement('div');
        suggestionContainer.style.margin = '20px 0';
        suggestionContainer.style.padding = '15px';
        suggestionContainer.style.background = "#555"; // Match color with chatbot
        suggestionContainer.style.borderRadius = '10px';
        suggestionContainer.style.border = '2px solid #555'; // Light pink border
        suggestionContainer.style.textAlign = 'left';
        suggestionContainer.style.color = '#fff';
        suggestionContainer.style.fontSize = '1rem';
        suggestionContainer.innerHTML = '<strong>You could ask a question like this:</strong>';

        const randomIndices = [];
        while (randomIndices.length < 2) {
            const randomIndex = Math.floor(Math.random() * questions.length);
            if (!randomIndices.includes(randomIndex)) {
                randomIndices.push(randomIndex);
            }
        }

        randomIndices.forEach(index => {
            const randomQuestion = questions[index];
            const suggestionElement = createClickableQuestion(randomQuestion);
            suggestionContainer.appendChild(suggestionElement);
        });

        chatContent.appendChild(suggestionContainer);
        chatContent.scrollTop = chatContent.scrollHeight;
    }

    function createClickableQuestion(question) {
        const questionElement = document.createElement('div');
        questionElement.textContent = question;
        questionElement.style.margin = '10px 0';
        questionElement.style.color = '#ff69b4'; // Pink text for questions
        questionElement.style.cursor = 'pointer';
        questionElement.style.textDecoration = 'none';

        questionElement.addEventListener('click', () => {
            chatInput.value = question;
            chatInput.focus();
            const event = new KeyboardEvent('keypress', { key: 'Enter' });
            chatInput.dispatchEvent(event);
        });

        return questionElement;
    }

    function createMessageElement(message, alignment, color) {
        const messageElement = document.createElement('div');
        messageElement.textContent = message;
        messageElement.style.margin = '10px 0';
        messageElement.style.textAlign = alignment;
        messageElement.style.color = color;
        messageElement.style.padding = '10px';
        return messageElement;
    }

    function saveChatHistory() {
        const messages = [];
        chatContent.childNodes.forEach(node => {
            if (node.nodeType === Node.ELEMENT_NODE) {
                messages.push({
                    text: node.textContent,
                    alignment: node.style.textAlign,
                    color: node.style.color
                });
            }
        });
        localStorage.setItem('chatHistory', JSON.stringify(messages));
    }

    function loadChatHistory() {
        const savedHistory = localStorage.getItem('chatHistory');
        chatContent.innerHTML = '';
        if (savedHistory) {
            try {
                const messages = JSON.parse(savedHistory);
                messages.forEach(msg => {
                    const messageElement = createMessageElement(msg.text, msg.alignment, msg.color);
                    chatContent.appendChild(messageElement);
                });
            } catch (e) {
                console.error("Failed to load chat history:", e);
            }
        }
    }

    function limitChatHistory() {
        const userMessages = Array.from(chatContent.querySelectorAll('.user-message'));
        const botMessages = Array.from(chatContent.querySelectorAll('.bot-message'));
        if (userMessages.length > 3) userMessages.slice(0, userMessages.length - 3).forEach(msg => msg.remove());
        if (botMessages.length > 3) botMessages.slice(0, botMessages.length - 3).forEach(msg => msg.remove());
    }
});