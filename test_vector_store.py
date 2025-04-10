import numpy as np
from agents.general_agent import GeneralAgent
from agents.admission_cs import AdmissionAgent
from agents.ai_agent import AIAgent



def test_agent(agent, name, queries):
    print(f"\n=== Testing {name} ===")
    for i, query in enumerate(queries, 1):
        print(f"Turn {i}: '{query}'")
        response = agent.respond(query)
        print(f"Response: {response}\n")
    
    # Check FAISS index and contexts
    print(f"Number of vectors in FAISS index: {agent.index.ntotal}")
    print(f"Stored contexts: {agent.history_texts}\n")
    
    # Test similarity search with a related query
    related_query = queries[1]  # Use second query as a test
    query_embedding = agent.embedder.encode(related_query)
    distances, indices = agent.index.search(np.array([query_embedding]), k=1)
    print(f"Similarity search for '{related_query}':")
    print(f"Closest context: {agent.history_texts[indices[0][0]]}")
    print(f"Distance: {distances[0][0]}\n")

# Initialize agents
general_agent = GeneralAgent()
admission_agent = AdmissionAgent()
ai_agent = AIAgent()

# Define test queries for each agent
general_queries = [
    "What is Python?",
    "Who created it?",
    "What is its latest version?"
]

admission_queries = [
    "What are the admission requirements for Concordia CS?",
    "What about deadlines?",
    "Are there any scholarships?"
]

ai_queries = [
    "What is machine learning?",
    "How does it work?",
    "What are some real-world examples?"
]

# Run tests
test_agent(general_agent, "GeneralAgent", general_queries)
test_agent(admission_agent, "AdmissionAgent", admission_queries)
test_agent(ai_agent, "AIAgent", ai_queries)