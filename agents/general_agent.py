import ollama
from langchain.memory import ConversationBufferMemory
from langchain_community.retrievers import WikipediaRetriever
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain.prompts import PromptTemplate  

class GeneralAgent:
    def __init__(self):
        self.model = "gemma3:4B"
        self.memory = ConversationBufferMemory()
        self.wiki_retriever = WikipediaRetriever(top_k_results=1)
        # Initialize sentence transformer for embeddings
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight and effective
        # Initialize FAISS index (dimension 384 matches the embedding size of the model)
        self.dimension = 384
        self.index = faiss.IndexFlatL2(self.dimension)  # L2 distance for similarity search
        self.history_texts = []  # Store text for retrieval
        self.context = "You are a general AI agent that answers normal questions in 2-3 sentences max."
    def respond(self, user_input):
        # Load conversation history from buffer memory
        history = self.memory.load_memory_variables({}).get("history", "")

        # Fetch Wikipedia content
        try:
            wiki_docs = self.wiki_retriever.invoke(user_input)
            wiki_content = wiki_docs[0].page_content[:350] if wiki_docs else "No Wikipedia info found."
        except Exception:
            wiki_content = "Fetching Wikipedia data failed."

        # Combine history, wiki content, and user input for context
        combined_context = f"History:\n{history}\nWikipedia Info: {wiki_content}\nUser: {user_input}"

        # Generate embedding for the combined context
        context_embedding = self.embedder.encode(combined_context, convert_to_numpy=True)

        # Add the embedding to FAISS index
        self.index.add(np.array([context_embedding]))
        self.history_texts.append(combined_context)  # Store the text for later retrieval

        # Search FAISS index for the most relevant past context (if any)
        if self.index.ntotal > 1:  
            indices = self.index.search(np.array([context_embedding]), k=1)
            relevant_context = self.history_texts[indices[0][0]]
        else:
            relevant_context = "No prior context available."

        # Construct the prompt with retrieved context using PromptTemplate
        prompt_template = PromptTemplate(
            input_variables=["relevant_context", "history", "wiki_content", "user_input"],
            template=(
                "{context}\n"
                "Retrieved Context:\n{relevant_context}\n"
                "History:\n{history}\n"
                "Wikipedia Info: {wiki_content}\n"
                "User: {user_input}\n"
            )
        )

        prompt = prompt_template.format(
            context=self.context,
            relevant_context=relevant_context,
            history=history,
            wiki_content=wiki_content,
            user_input=user_input
        )

        # Generate response using Ollama last resort if none of the two above work
        # Fallback to Ollama for response generation
        try:
            response = ollama.chat(model=self.model, messages=[{"role": "user", "content": prompt}])
            ai_response = response['message']['content']
        except Exception:
             ai_response = "Sorry, I Can't generate an answer."

        # Save the conversation to buffer memory
        self.memory.save_context({"input": user_input}, {"output": ai_response})
        return ai_response