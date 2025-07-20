import random
from datetime import datetime

class SimpleChatbot:
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
                "I'd be happy to help answer that for you."
            ],
            "compliment": [
                "Thank you so much! That's very kind of you to say. 😊",
                "I appreciate your kind words!",
                "Thank you! I'm glad I could help."
            ],
            "unknown": [
                "I'm not quite sure I understand. Could you rephrase that?",
                "I didn't catch that. Can you tell me more?",
                "I'm still learning! Could you explain that differently?"
            ]
        }
        
        self.conversation_history = []
        print("🤖 Simple Chatbot initialized successfully!")
    
    def classify_intent(self, text):
        """Simple rule-based intent classification"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['hello', 'hi', 'hey', 'good morning', 'good afternoon']):
            return 'greeting'
        elif any(word in text_lower for word in ['bye', 'goodbye', 'see you', 'farewell']):
            return 'goodbye'
        elif any(word in text_lower for word in ['what', 'how', 'why', 'when', 'where', '?']):
            return 'question'
        elif any(word in text_lower for word in ['thank', 'thanks', 'great', 'excellent', 'amazing']):
            return 'compliment'
        else:
            return 'unknown'
    
    def generate_response(self, intent, user_input):
        """Generate appropriate response"""
        responses = self.responses.get(intent, self.responses["unknown"])
        return random.choice(responses)
    
    def process_message(self, user_input):
        """Process user message and generate response"""
        intent = self.classify_intent(user_input)
        response = self.generate_response(intent, user_input)
        
        # Log interaction
        interaction = {
            'user_input': user_input,
            'intent': intent,
            'response': response,
            'timestamp': datetime.now().isoformat()
        }
        
        self.conversation_history.append(interaction)
        return interaction
    
    def start_conversation(self):
        """Start interactive conversation"""
        print("🤖 Chatbot is ready! (Type 'quit' to exit)")
        print("=" * 50)
        
        while True:
            user_input = input("\nYou: ")
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                interaction = self.process_message(user_input)
                print(f"Bot: {interaction['response']}")
                break
            
            interaction = self.process_message(user_input)
            print(f"Bot: {interaction['response']}")
            
            # Show debug info occasionally
            if len(self.conversation_history) % 5 == 0:
                print(f"💡 Intent detected: {interaction['intent']}")

if __name__ == "__main__":
    print("🤖 Simple NLP Chatbot Demo")
    print("=" * 50)
    
    # Initialize chatbot
    chatbot = SimpleChatbot()
    
    # Start conversation
    chatbot.start_conversation()
    
    print(f"\n📊 Conversation ended. Total interactions: {len(chatbot.conversation_history)}")
