import os
import re
import numpy as np
import random
import json
import pickle
from flask import Flask, render_template_string, request, jsonify
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

# Initialize Flask app
app = Flask(__name__)

# =============================================
# CHATBOT CORE COMPONENTS
# =============================================

class HealthChatbot:
    def __init__(self):
        self.model = None
        self.words = []
        self.classes = []
        self.intents = {"intents": []}
        
    def initialize(self):
        """Load or create all required components"""
        try:
            # Check if model files exist
            model_exists = os.path.exists('health_chatbot_model.h5')
            data_exists = all(os.path.exists(f) for f in ['intents.json', 'words.pkl', 'classes.pkl'])
            
            if not (model_exists and data_exists):
                self.train_model()
            
            self.model = load_model('health_chatbot_model.h5')
            with open('intents.json') as f:
                self.intents = json.load(f)
            with open('words.pkl', 'rb') as f:
                self.words = pickle.load(f)
            with open('classes.pkl', 'rb') as f:
                self.classes = pickle.load(f)
            return True
        except Exception as e:
            print(f"Initialization error: {e}")
            return False
    
    def tokenize(self, text):
        """Simple tokenizer that always returns a list"""
        try:
            return re.findall(r"\w+(?:'\w+)?|\?|\.|!|,", text)
        except:
            return []
    
    def lemmatize(self, word):
        """Basic lemmatizer that always returns a string"""
        try:
            word = str(word).lower()
            endings = {
                'ies': 'y', 'es': 'e', 's': '', 
                'ing': '', 'ed': '', 'er': '', 'est': '',
                'ly': '', 'ment': '', 'ness': '', 'ful': '', 'less': ''
            }
            for end, repl in endings.items():
                if word.endswith(end):
                    return word[:-len(end)] + repl
            return word
        except:
            return word.lower() if word else ""
    
    def train_model(self):
        """Train and save a new model"""
        try:
            # Sample training data
            intents = {
                "intents": [
                    {
                        "tag": "greeting",
                        "patterns": ["Hi", "Hello", "Hey", "How are you"],
                        "responses": ["Hello! How can I help with your health?", "Hi there!"]
                    },

                    {
                        "tag": "nutrition",
                        "patterns": ["What should I eat?", "Healthy diet"],
                        "responses": ["Eat fruits, vegetables, and whole grains."]
                    }
                ]
            }
            
            # Process data
            words = []
            classes = []
            documents = []
            
            for intent in intents['intents']:
                for pattern in intent['patterns']:
                    word_list = self.tokenize(pattern)
                    words.extend(word_list)
                    documents.append((word_list, intent['tag']))
                    if intent['tag'] not in classes:
                        classes.append(intent['tag'])

            words = sorted(set(self.lemmatize(w) for w in words if w not in ['?', '!', '.', ',']))
            classes = sorted(set(classes))

            # Save data
            with open('intents.json', 'w') as f:
                json.dump(intents, f)
            with open('words.pkl', 'wb') as f:
                pickle.dump(words, f)
            with open('classes.pkl', 'wb') as f:
                pickle.dump(classes, f)

            # Create training data
            training = []
            output_empty = [0] * len(classes)

            for doc in documents:
                bag = []
                word_patterns = [self.lemmatize(word) for word in doc[0]]
                for word in words:
                    bag.append(1) if word in word_patterns else bag.append(0)

                output_row = list(output_empty)
                output_row[classes.index(doc[1])] = 1
                training.append([bag, output_row])

            # Convert to numpy arrays
            train_x = np.array([i[0] for i in training])
            train_y = np.array([i[1] for i in training])

            # Build and train model
            model = Sequential()
            model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(64, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(len(train_y[0]), activation='softmax'))

            sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
            model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
            model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=0)
            model.save('health_chatbot_model.h5')
            return True
        except Exception as e:
            print(f"Training error: {e}")
            return False
    
    def get_response(self, message):
        """Get chatbot response with full error handling"""
        try:
            if not message or not self.model:
                return "I'm still learning. Please try again later."
            
            # Predict intent
            bow = self._create_bag_of_words(message)
            res = self.model.predict(np.array([bow]))[0]
            results = [[i, r] for i, r in enumerate(res) if r > 0.25]
            results.sort(key=lambda x: x[1], reverse=True)
            
            if not results:
                return "I'm not sure how to respond to that."
            
            # Get matching intent
            intent_tag = self.classes[results[0][0]]
            for intent in self.intents['intents']:
                if intent['tag'] == intent_tag:
                    return random.choice(intent.get('responses', ["Interesting question!"]))
            
            return "Could you rephrase that?"
        except:
            return "I'm having some trouble understanding right now."
    
    def _create_bag_of_words(self, sentence):
        """Create bag of words array with error handling"""
        try:
            sentence_words = [self.lemmatize(w) for w in self.tokenize(sentence)]
            bag = [0] * len(self.words)
            for w in sentence_words:
                for i, word in enumerate(self.words):
                    if word == w:
                        bag[i] = 1
            return np.array(bag)
        except:
            return np.zeros(len(self.words))

# Initialize chatbot instance
chatbot = HealthChatbot()
chatbot.initialize()

# =============================================
# FLASK ROUTES WITH GUARANTEED RESPONSES
# =============================================

@app.route('/')
def home():
    """Main page that always returns valid HTML"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Health Chatbot</title>
        <style>
            body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f9fc;
        }
        .chat-container {
            border: 1px solid #ddd;
            border-radius: 10px;
            height: 500px;
            overflow-y: auto;
            padding: 15px;
            background-color: white;
            margin-bottom: 15px;
        }
        .user-message {
            background-color: #e3f2fd;
            padding: 8px 12px;
            border-radius: 10px;
            margin: 5px 0;
            max-width: 70%;
            float: right;
            clear: both;
        }
        .bot-message {
            background-color: #f1f1f1;
            padding: 8px 12px;
            border-radius: 10px;
            margin: 5px 0;
            max-width: 70%;
            float: left;
            clear: both;
        }
        #userInput {
            width: 75%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        #sendButton {
            width: 20%;
            padding: 10px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        h1 {
            color: #2c7873;
            text-align: center;
        }
        </style>
    </head>
    <body>
        <h1>Health & Wellness Assistant</h1>
        <div id="chatbox">
            <div class="bot">Hello! I'm your health assistant. How can I help?</div>
        </div>
        <input type="text" id="userInput" placeholder="Ask me about health...">
        <button id="sendBtn">Send</button>
        
        <script>
            function sendMessage() {
                const input = document.getElementById('userInput');
                const message = input.value.trim();
                
                if (message) {
                    const chatbox = document.getElementById('chatbox');
                    chatbox.innerHTML += `<div class="user">You: ${message}</div>`;
                    input.value = '';
                    
                    fetch('/get_response?message=' + encodeURIComponent(message))
                        .then(response => response.text())
                        .then(response => {
                            chatbox.innerHTML += `<div class="bot">Bot: ${response}</div>`;
                            chatbox.scrollTop = chatbox.scrollHeight;
                        })
                        .catch(() => {
                            chatbox.innerHTML += `<div class="bot">Bot: Sorry, I'm having trouble responding.</div>`;
                        });
                }
            }
            
            document.getElementById('sendBtn').addEventListener('click', sendMessage);
            document.getElementById('userInput').addEventListener('keypress', (e) => {
                if (e.key === 'Enter') sendMessage();
            });
        </script>
    </body>
    </html>
    """
    return render_template_string(html)

@app.route('/get_response')
def get_bot_response():
    """API endpoint that always returns a valid response"""
    message = request.args.get('message', '')
    response = chatbot.get_response(message)
    return jsonify({"response": response})

# =============================================
# APPLICATION ENTRY POINT
# =============================================

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)