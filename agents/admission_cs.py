import ollama
from langchain.memory import ConversationBufferMemory
from langchain_community.retrievers import WikipediaRetriever

class AdmissionAgent:
    def __init__(self):
        self.model = "gemma3:4B"
        self.memory = ConversationBufferMemory()
        self.wiki_retriever = WikipediaRetriever(top_k_results=1)
        self.context = "You are a Concordia University Computer Science admission expert. Respond as AI only, 2-3 sentences max."

    def respond(self, user_input):
        history = self.memory.load_memory_variables({}).get("history", "")
        try:
            wiki_docs = self.wiki_retriever.invoke(user_input)
            wiki_content = wiki_docs[0].page_content[:300] if wiki_docs else "No Wikipedia info found."
        except Exception:
            wiki_content = "Couldn’t fetch Wikipedia data."
        
        prompt = (
            f"{self.context}\n"
            f"History:\n{history}\n"
            f"Wikipedia Info: {wiki_content}\n"
            f"User: {user_input}\n"
            f"AI:"
        )
        response = ollama.chat(model=self.model, messages=[{"role": "user", "content": prompt}])
        ai_response = response['message']['content']
        self.memory.save_context({"input": user_input}, {"output": ai_response})
        return ai_response