import ollama
from langchain.memory import ConversationBufferMemory

class GeneralAgent:
    def __init__(self):
        self.model = "llma3"  # Change this to another model if needed
        self.memory = ConversationBufferMemory()

    def respond(self, user_input):
        # Retrieve conversation history
        history = self.memory.load_memory_variables({}).get("history", "")

        # Format prompt with history and user input
        prompt = f"Conversation History:\n{history}\nUser: {user_input}\nAI:"

        # Generate a response using Ollama
        response = ollama.chat(model=self.model, messages=[{"role": "user", "content": prompt}])

        # Store conversation in memory
        ai_response = response['message']['content']
        self.memory.save_context({"input": user_input}, {"output": ai_response})

        return ai_response
