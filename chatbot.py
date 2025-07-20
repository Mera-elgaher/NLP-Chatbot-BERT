import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    BertTokenizer, BertForSequenceClassification,
    pipeline, Conversation
)
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import re
import random
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string
import logging

class IntentClassifier:
    def __init__(self, model_name='bert-base-uncased'):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.intent_labels = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_pretrained_model(self):
        """Load pre-trained BERT model for intent classification"""
        print("🤖 Loading pre-trained BERT model...")
        
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.intent_labels) if self.intent_labels else 10
        )
        self.model.to(self.device)
        
        print(f"✅ Model loaded on {self.device}")
    
    def predict_intent(self, text, confidence_threshold=0.7):
        """Predict intent from text"""
        if not self.model or not self.tokenizer:
            print("❌ Model not loaded!")
            return "unknown", 0.0
        
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            max_length=128,
            truncation=True,
            padding=True
        ).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
        
        if confidence >= confidence_threshold and predicted_class < len(self.intent_labels):
            intent = self.intent_labels[predicted_class]
        else:
            intent = "unknown"
        
        return intent, confidence

class EntityExtractor:
    def __init__(self):
        self.ner_pipeline = pipeline("ner", 
                                   model="dbmdz/bert-large-cased-finetuned-conll03-english",
                                   aggregation_strategy="simple")
    
    def extract_entities(self, text):
        """Extract named entities from text"""
        entities = self.ner_pipeline(text)
        
        # Process and clean entities
        processed_entities = {}
        for entity in entities:
            entity_type = entity['entity_group']
            entity_text = entity['word']
            confidence = entity['score']
            
            if entity_type not in processed_entities:
                processed_entities[entity_type] = []
            
            processed_entities[entity_type].append({
                'text': entity_text,
                'confidence': confidence
            })
        
        return processed_entities

class ResponseGenerator:
    def __init__(self):
        self.responses = {
            "greeting": [
                "Hello! How can I help you today? 😊",
                "Hi there! What can I do for you?",
                "Greetings! How may I assist you?",
                "Hello! I'm here to help. What do you need?"
            ],
            "goodbye": [
                "Goodbye! Have a great day! 👋",
                "See you later! Take care!",
                "Farewell! Feel free to come back anytime.",
                "Bye! It was nice talking with you."
            ],
            "question": [
                "That's an interesting question! Let me think about that.",
                "Good question! Here's what I know about that:",
                "I'd be happy to help answer that for you.",
                "Let me provide you with some information about that."
            ],
            "request": [
                "I'll do my best to help you with that request.",
                "Sure, I can help you with that!",
                "Let me assist you with that.",
                "I'd be happy to help you with that."
            ],
            "complaint": [
                "I understand your concern. Let me see how I can help.",
                "I'm sorry to hear about that issue. How can I assist?",
                "Thank you for bringing this to my attention.",
                "I appreciate your feedback. Let me help resolve this."
            ],
            "compliment": [
                "Thank you so much! That's very kind of you to say. 😊",
                "I appreciate your kind words!",
                "Thank you! I'm glad I could help.",
                "That means a lot to me, thank you!"
            ],
            "unknown": [
                "I'm not quite sure I understand. Could you rephrase that?",
                "I didn't catch that. Can you tell me more?",
                "I'm still learning! Could you explain that differently?",
                "Hmm, I'm not sure about that. Can you provide more details?"
            ]
        }
        
        # Load conversation context
        self.conversation_history = []
        self.user_preferences = {}
    
    def generate_response(self, intent, entities, user_input, confidence):
        """Generate appropriate response based on intent and entities"""
        
        # Add to conversation history
        self.conversation_history.append({
            'user_input': user_input,
            'intent': intent,
            'confidence': confidence,
            'entities': entities,
            'timestamp': datetime.now().isoformat()
        })
        
        # Generate base response
        base_responses = self.responses.get(intent, self.responses["unknown"])
        base_response = random.choice(base_responses)
        
        # Personalize response based on entities
        personalized_response = self.personalize_response(base_response, entities, intent)
        
        # Add context from conversation history
        contextual_response = self.add_context(personalized_response, intent)
        
        return contextual_response
    
    def personalize_response(self, response, entities, intent):
        """Personalize response based on extracted entities"""
        if 'PERSON' in entities:
            names = [entity['text'] for entity in entities['PERSON']]
            if names:
                name = names[0]
                if intent == "greeting":
                    response = f"Hello {name}! How can I help you today? 😊"
                else:
                    response = f"{response} {name}!"
        
        if 'LOCATION' in entities:
            locations = [entity['text'] for entity in entities['LOCATION']]
            if locations and 'weather' in response.lower():
                location = locations[0]
                response += f" I see you're asking about {location}."
        
        return response
    
    def add_context(self, response, intent):
        """Add conversational context"""
        if len(self.conversation_history) > 1:
            previous_intent = self.conversation_history[-2]['intent']
            
            if previous_intent == "greeting" and intent == "question":
                response = "Great! " + response
            elif previous_intent == "complaint" and intent in ["question", "request"]:
                response = "I understand. " + response
        
        return response

class ConversationalChatbot:
    def __init__(self):
        self.intent_classifier = IntentClassifier()
        self.entity_extractor = EntityExtractor()
        self.response_generator = ResponseGenerator()
        
        # Initialize with basic intents
        self.intent_classifier.intent_labels = [
            "greeting", "goodbye", "question", "request", 
            "complaint", "compliment", "unknown"
        ]
        
        # Load models
        self.load_models()
        
        # Conversation state
        self.conversation_active = False
        self.conversation_id = None
        
    def load_models(self):
        """Load all required models"""
        print("🚀 Loading chatbot models...")
        
        # Load intent classifier
        self.intent_classifier.load_pretrained_model()
        
        # Initialize conversation pipeline
        self.conversational_pipeline = pipeline(
            "conversational",
            model="microsoft/DialoGPT-medium"
        )
        
        print("✅ All models loaded successfully!")
    
    def preprocess_text(self, text):
        """Preprocess input text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Basic cleaning
        text = re.sub(r'[^\w\s\?!.,]', '', text)
        
        return text
    
    def process_message(self, user_input):
        """Process user message and generate response"""
        # Preprocess input
        cleaned_input = self.preprocess_text(user_input)
        
        # Classify intent
        intent, confidence = self.intent_classifier.predict_intent(cleaned_input)
        
        # Extract entities
        entities = self.entity_extractor.extract_entities(user_input)
        
        # Generate response
        response = self.response_generator.generate_response(
            intent, entities, user_input, confidence
        )
        
        # Log interaction
        interaction = {
            'user_input': user_input,
            'intent': intent,
            'confidence': confidence,
            'entities': entities,
            'response': response,
            'timestamp': datetime.now().isoformat()
        }
        
        return interaction
    
    def generate_fallback_response(self, user_input):
        """Generate fallback response using conversational AI"""
        try:
            conversation = Conversation(user_input)
            conversation = self.conversational_pipeline(conversation)
            return conversation.generated_responses[-1]
        except Exception as e:
            print(f"Fallback generation error: {e}")
            return "I'm sorry, I couldn't understand that. Could you please rephrase?"
    
    def start_conversation(self):
        """Start interactive conversation"""
        print("🤖 Chatbot is ready! (Type 'quit' to exit)")
        print("=" * 50)
        
        self.conversation_active = True
        
        while self.conversation_active:
            user_input = input("\nYou: ")
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                final_response = self.process_message(user_input)
                print(f"Bot: {final_response['response']}")
                self.conversation_active = False
                break
            
            # Process message
            interaction = self.process_message(user_input)
            
            print(f"Bot: {interaction['response']}")
            
            # Show debug info (optional)
            if '--debug' in user_input:
                print(f"Debug - Intent: {interaction['intent']} "
                      f"(Confidence: {interaction['confidence']:.2f})")
                print(f"Debug - Entities: {interaction['entities']}")
    
    def evaluate_responses(self, test_data):
        """Evaluate chatbot performance on test data"""
        print("📊 Evaluating chatbot performance...")
        
        results = []
        
        for item in test_data:
            user_input = item['input']
            expected_intent = item['expected_intent']
            
            interaction = self.process_message(user_input)
            predicted_intent = interaction['intent']
            
            results.append({
                'input': user_input,
                'expected': expected_intent,
                'predicted': predicted_intent,
                'confidence': interaction['confidence'],
                'correct': expected_intent == predicted_intent
            })
        
        # Calculate metrics
        accuracy = sum(1 for r in results if r['correct']) / len(results)
        
        print(f"Intent Classification Accuracy: {accuracy:.2%}")
        
        return results
    
    def export_conversation_history(self, filename='conversation_history.json'):
        """Export conversation history"""
        history = self.response_generator.conversation_history
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Conversation history exported to {filename}")

# Flask Web Interface
class ChatbotWebApp:
    def __init__(self, chatbot):
        self.app = Flask(__name__)
        self.chatbot = chatbot
        self.setup_routes()
    
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            return render_template_string(self.get_html_template())
        
        @self.app.route('/chat', methods=['POST'])
        def chat():
            data = request.json
            user_message = data.get('message', '')
            
            if not user_message:
                return jsonify({'error': 'No message provided'}), 400
            
            # Process message
            interaction = self.chatbot.process_message(user_message)
            
            return jsonify({
                'response': interaction['response'],
                'intent': interaction['intent'],
                'confidence': interaction['confidence'],
                'entities': interaction['entities']
            })
        
        @self.app.route('/history', methods=['GET'])
        def get_history():
            history = self.chatbot.response_generator.conversation_history
            return jsonify(history)
    
    def get_html_template(self):
        """Return HTML template for chatbot interface"""
        return '''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>AI Chatbot</title>
            <style>
                * {
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }
                
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    height: 100vh;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                }
                
                .chat-container {
                    width: 90%;
                    max-width: 800px;
                    height: 600px;
                    background: white;
                    border-radius: 20px;
                    box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                    display: flex;
                    flex-direction: column;
                    overflow: hidden;
                }
                
                .chat-header {
                    background: linear-gradient(135deg, #667eea, #764ba2);
                    color: white;
                    padding: 20px;
                    text-align: center;
                }
                
                .chat-header h1 {
                    font-size: 1.8rem;
                    margin-bottom: 5px;
                }
                
                .chat-header p {
                    opacity: 0.9;
                }
                
                .chat-messages {
                    flex: 1;
                    padding: 20px;
                    overflow-y: auto;
                    background: #f8f9fa;
                }
                
                .message {
                    margin-bottom: 15px;
                    display: flex;
                    align-items: flex-start;
                }
                
                .user-message {
                    justify-content: flex-end;
                }
                
                .message-content {
                    max-width: 70%;
                    padding: 12px 16px;
                    border-radius: 18px;
                    word-wrap: break-word;
                }
                
                .user-message .message-content {
                    background: #667eea;
                    color: white;
                    border-bottom-right-radius: 4px;
                }
                
                .bot-message .message-content {
                    background: white;
                    color: #333;
                    border: 1px solid #e0e0e0;
                    border-bottom-left-radius: 4px;
                }
                
                .chat-input {
                    padding: 20px;
                    background: white;
                    border-top: 1px solid #e0e0e0;
                    display: flex;
                    gap: 10px;
                }
                
                .chat-input input {
                    flex: 1;
                    padding: 12px 16px;
                    border: 2px solid #e0e0e0;
                    border-radius: 25px;
                    font-size: 14px;
                    outline: none;
                    transition: border-color 0.3s;
                }
                
                .chat-input input:focus {
                    border-color: #667eea;
                }
                
                .chat-input button {
                    padding: 12px 24px;
                    background: #667eea;
                    color: white;
                    border: none;
                    border-radius: 25px;
                    font-size: 14px;
                    cursor: pointer;
                    transition: background 0.3s;
                }
                
                .chat-input button:hover {
                    background: #5a6fd8;
                }
                
                .typing-indicator {
                    display: none;
                    padding: 10px 16px;
                    color: #666;
                    font-style: italic;
                }
                
                .intent-info {
                    font-size: 12px;
                    color: #888;
                    margin-top: 5px;
                }
            </style>
        </head>
        <body>
            <div class="chat-container">
                <div class="chat-header">
                    <h1>🤖 AI Chatbot</h1>
                    <p>Powered by BERT & Transformers</p>
                </div>
                
                <div class="chat-messages" id="chatMessages">
                    <div class="message bot-message">
                        <div class="message-content">
                            Hello! I'm your AI assistant. How can I help you today? 😊
                        </div>
                    </div>
                </div>
                
                <div class="typing-indicator" id="typingIndicator">
                    Bot is typing...
                </div>
                
                <div class="chat-input">
                    <input type="text" id="messageInput" placeholder="Type your message here..." maxlength="500">
                    <button onclick="sendMessage()">Send</button>
                </div>
            </div>
            
            <script>
                const chatMessages = document.getElementById('chatMessages');
                const messageInput = document.getElementById('messageInput');
                const typingIndicator = document.getElementById('typingIndicator');
                
                // Focus on input
                messageInput.focus();
                
                // Handle Enter key
                messageInput.addEventListener('keypress', function(e) {
                    if (e.key === 'Enter') {
                        sendMessage();
                    }
                });
                
                function addMessage(content, isUser = false, intent = null, confidence = null) {
                    const messageDiv = document.createElement('div');
                    messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
                    
                    const contentDiv = document.createElement('div');
                    contentDiv.className = 'message-content';
                    contentDiv.textContent = content;
                    
                    if (!isUser && intent) {
                        const intentInfo = document.createElement('div');
                        intentInfo.className = 'intent-info';
                        intentInfo.textContent = `Intent: ${intent} (${(confidence * 100).toFixed(1)}%)`;
                        contentDiv.appendChild(intentInfo);
                    }
                    
                    messageDiv.appendChild(contentDiv);
                    chatMessages.appendChild(messageDiv);
                    
                    // Scroll to bottom
                    chatMessages.scrollTop = chatMessages.scrollHeight;
                }
                
                function showTyping() {
                    typingIndicator.style.display = 'block';
                    chatMessages.scrollTop = chatMessages.scrollHeight;
                }
                
                function hideTyping() {
                    typingIndicator.style.display = 'none';
                }
                
                async function sendMessage() {
                    const message = messageInput.value.trim();
                    if (!message) return;
                    
                    // Add user message
                    addMessage(message, true);
                    messageInput.value = '';
                    
                    // Show typing indicator
                    showTyping();
                    
                    try {
                        const response = await fetch('/chat', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({ message: message })
                        });
                        
                        const data = await response.json();
                        
                        // Hide typing and add bot response
                        hideTyping();
                        addMessage(data.response, false, data.intent, data.confidence);
                        
                    } catch (error) {
                        hideTyping();
                        addMessage('Sorry, I encountered an error. Please try again.', false);
                        console.error('Error:', error);
                    }
                    
                    // Focus back on input
                    messageInput.focus();
                }
            </script>
        </body>
        </html>
        '''
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the Flask app"""
        print(f"🌐 Starting web interface on http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug)

# Advanced features
class AdvancedChatbotFeatures:
    def __init__(self, chatbot):
        self.chatbot = chatbot
        
    def add_sentiment_analysis(self):
        """Add sentiment analysis capability"""
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest"
        )
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of text"""
        if hasattr(self, 'sentiment_pipeline'):
            result = self.sentiment_pipeline(text)[0]
            return {
                'label': result['label'],
                'score': result['score']
            }
        return None
    
    def add_emotion_detection(self):
        """Add emotion detection"""
        self.emotion_pipeline = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base"
        )
    
    def detect_emotion(self, text):
        """Detect emotion in text"""
        if hasattr(self, 'emotion_pipeline'):
            result = self.emotion_pipeline(text)[0]
            return {
                'emotion': result['label'],
                'confidence': result['score']
            }
        return None
    
    def add_language_detection(self):
        """Add language detection"""
        from langdetect import detect, detect_langs
        self.language_detector = detect
    
    def detect_language(self, text):
        """Detect language of text"""
        try:
            if hasattr(self, 'language_detector'):
                return self.language_detector(text)
        except:
            pass
        return 'en'  # default to English

# Example usage and testing
if __name__ == "__main__":
    print("🤖 Intelligent NLP Chatbot System")
    print("=" * 50)
    
    # Initialize chatbot
    chatbot = ConversationalChatbot()
    
    # Add advanced features
    advanced_features = AdvancedChatbotFeatures(chatbot)
    advanced_features.add_sentiment_analysis()
    
    # Test the chatbot
    test_messages = [
        "Hello there!",
        "How are you today?",
        "I'm having trouble with my order",
        "Thank you for your help",
        "Goodbye!"
    ]
    
    print("\n🧪 Testing chatbot with sample messages:")
    for message in test_messages:
        print(f"\nUser: {message}")
        interaction = chatbot.process_message(message)
        print(f"Bot: {interaction['response']}")
        print(f"Intent: {interaction['intent']} (Confidence: {interaction['confidence']:.2f})")
        
        # Test sentiment analysis
        sentiment = advanced_features.analyze_sentiment(message)
        if sentiment:
            print(f"Sentiment: {sentiment['label']} ({sentiment['score']:.2f})")
    
    # Option 1: Start command-line conversation
    print("\n" + "="*50)
    print("Choose an option:")
    print("1. Start command-line chat")
    print("2. Start web interface")
    print("3. Export conversation history")
    
    choice = input("\nEnter your choice (1-3): ")
    
    if choice == "1":
        chatbot.start_conversation()
    elif choice == "2":
        web_app = ChatbotWebApp(chatbot)
        web_app.run(debug=True)
    elif choice == "3":
        chatbot.export_conversation_history()
    else:
        print("Invalid choice. Starting command-line chat...")
        chatbot.start_conversation()
    
    print("\n✅ Chatbot session ended. Thank you!")
