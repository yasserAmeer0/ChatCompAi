import ollama
from langchain.memory import ConversationBufferMemory

class AdmissionAgent:
    def __init__(self):
        self.model = "tinyllama"
        self.memory = ConversationBufferMemory()
        self.context = "You are a Concordia CS admission expert. Respond as AI only, 2-3 sentences max."

    def respond(self, user_input):
        history = self.memory.load_memory_variables({}).get("history", "")
        prompt = f"{self.context}\nHistory:\n{history}\nUser: {user_input}\nAI:"
        response = ollama.chat(model=self.model, messages=[{"role": "user", "content": prompt}])
        ai_response = response['message']['content']
        self.memory.save_context({"input": user_input}, {"output": ai_response})
        return ai_response