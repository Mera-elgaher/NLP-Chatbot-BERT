from flask import Flask, render_template_string, request, jsonify
from simple_chatbot import SimpleChatbot

app = Flask(__name__)
chatbot = SimpleChatbot()

@app.route('/')
def index():
    return render_template_string("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🤖 AI Chatbot Demo</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                height: 100vh; display: flex; justify-content: center; align-items: center;
            }
            .chat-container {
                width: 90%; max-width: 600px; height: 500px;
                background: white; border-radius: 20px; 
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                display: flex; flex-direction: column; overflow: hidden;
            }
            .chat-header {
                background: linear-gradient(135deg, #667eea, #764ba2);
                color: white; padding: 20px; text-align: center;
            }
            .chat-messages {
                flex: 1; padding: 20px; overflow-y: auto; background: #f8f9fa;
            }
            .message { margin-bottom: 15px; display: flex; align-items: flex-start; }
            .user-message { justify-content: flex-end; }
            .message-content {
                max-width: 70%; padding: 12px 16px; border-radius: 18px;
                word-wrap: break-word;
            }
            .user-message .message-content {
                background: #667eea; color: white; border-bottom-right-radius: 4px;
            }
            .bot-message .message-content {
                background: white; color: #333; border: 1px solid #e0e0e0;
                border-bottom-left-radius: 4px;
            }
            .chat-input {
                padding: 20px; background: white; border-top: 1px solid #e0e0e0;
                display: flex; gap: 10px;
            }
            .chat-input input {
                flex: 1; padding: 12px 16px; border: 2px solid #e0e0e0;
                border-radius: 25px; font-size: 14px; outline: none;
                transition: border-color 0.3s;
            }
            .chat-input input:focus { border-color: #667eea; }
            .chat-input button {
                padding: 12px 24px; background: #667eea; color: white;
                border: none; border-radius: 25px; font-size: 14px;
                cursor: pointer; transition: background 0.3s;
            }
            .chat-input button:hover { background: #5a6fd8; }
            .typing { display: none; padding: 10px 16px; color: #666; font-style: italic; }
        </style>
    </head>
    <body>
        <div class="chat-container">
            <div class="chat-header">
                <h1>🤖 AI Chatbot</h1>
                <p>Powered by NLP & Machine Learning</p>
            </div>
            
            <div class="chat-messages" id="chatMessages">
                <div class="message bot-message">
                    <div class="message-content">
                        Hello! I'm your AI assistant. How can I help you today? 😊
                    </div>
                </div>
            </div>
            
            <div class="typing" id="typing">Bot is typing...</div>
            
            <div class="chat-input">
                <input type="text" id="messageInput" placeholder="Type your message here..." maxlength="500">
                <button onclick="sendMessage()">Send</button>
            </div>
        </div>
        
        <script>
            const chatMessages = document.getElementById('chatMessages');
            const messageInput = document.getElementById('messageInput');
            const typing = document.getElementById('typing');
            
            messageInput.focus();
            
            messageInput.addEventListener('keypress', function(e) {
                if (e.key === 'Enter') sendMessage();
            });
            
            function addMessage(content, isUser = false) {
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
                
                const contentDiv = document.createElement('div');
                contentDiv.className = 'message-content';
                contentDiv.textContent = content;
                
                messageDiv.appendChild(contentDiv);
                chatMessages.appendChild(messageDiv);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
            
            function showTyping() {
                typing.style.display = 'block';
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
            
            function hideTyping() {
                typing.style.display = 'none';
            }
            
            async function sendMessage() {
                const message = messageInput.value.trim();
                if (!message) return;
                
                addMessage(message, true);
                messageInput.value = '';
                showTyping();
                
                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({message: message})
                    });
                    
                    const data = await response.json();
                    hideTyping();
                    addMessage(data.response, false);
                    
                } catch (error) {
                    hideTyping();
                    addMessage('Sorry, I encountered an error. Please try again.', false);
                }
                
                messageInput.focus();
            }
        </script>
    </body>
    </html>
    """)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    
    interaction = chatbot.process_message(user_message)
    return jsonify({'response': interaction['response']})

if __name__ == '__main__':
    print("🌐 Starting chatbot web interface...")
    print("Visit: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
