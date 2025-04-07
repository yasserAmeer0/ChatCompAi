import ollama
from langchain.memory import ConversationBufferMemory
from langchain_community.retrievers import WikipediaRetriever

class GeneralAgent:
    def __init__(self):
        self.model ="gemma3:4B"
        self.memory = ConversationBufferMemory()
        self.wiki_retriever = WikipediaRetriever(top_k_results=1)  # Fetch 1 result for speed

    def respond(self, user_input):
        history = self.memory.load_memory_variables({}).get("history", "")
        try:
            wiki_docs = self.wiki_retriever.invoke(user_input)
            wiki_content = wiki_docs[0].page_content[:300] if wiki_docs else "No Wikipedia info found."
        except Exception:
            wiki_content = "Couldn’t fetch Wikipedia data."
        
        prompt = (
            f"History:\n{history}\n"
            f"Wikipedia Info: {wiki_content}\n"
            f"User: {user_input}\n"
            f"AI: Respond as General AI agent, 2-3 sentences max."
        )
        response = ollama.chat(model=self.model, messages=[{"role": "user", "content": prompt}])
        ai_response = response['message']['content']
        self.memory.save_context({"input": user_input}, {"output": ai_response})
        return ai_response