from flask import Flask, render_template, request, jsonify
from agents.general_agent import GeneralAgent
from agents.admission_cs import AdmissionAgent
from agents.ai_agent import AIAgent

# Router class
class ChatbotSystem:
    def __init__(self):
        self.general_agent = GeneralAgent()
        self.admission_agent = AdmissionAgent()
        self.ai_agent = AIAgent()

    def route_query(self, user_input):
        user_input_lower = user_input.lower()
        if "admission" in user_input_lower or "concordia" in user_input_lower or "computer science" in user_input_lower:
            return self.admission_agent.respond(user_input)
        elif "ai" in user_input_lower or "artificial intelligence" in user_input_lower or "machine learning" in user_input_lower:
            return self.ai_agent.respond(user_input)
        else:
            return self.general_agent.respond(user_input)

# Flask app
app = Flask(__name__)
chatbot = ChatbotSystem()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    message = data.get('message', '')
    if not message:
        return jsonify({'error': 'No message provided'}), 400
    response = chatbot.route_query(message)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)