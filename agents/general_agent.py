import ollama
from langchain.memory import ConversationBufferMemory

class GeneralAgent:
    def __init__(self):
        self.model = "tinyllama"
        self.memory = ConversationBufferMemory()

    def respond(self, user_input):
        history = self.memory.load_memory_variables({}).get("history", "")
        prompt = f"History:\n{history}\nUser: {user_input}\nAI: Respond as AI only, 2-3 sentences max."
        response = ollama.chat(model=self.model, messages=[{"role": "user", "content": prompt}])
        ai_response = response['message']['content']
        self.memory.save_context({"input": user_input}, {"output": ai_response})
        return ai_response