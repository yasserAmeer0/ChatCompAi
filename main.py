from fastapi import FastAPI
from pydantic import BaseModel
from general_agent import GeneralAgent

# Initialize FastAPI and agent
app = FastAPI()
general_agent = GeneralAgent()

class UserQuery(BaseModel):
    query: str

@app.post("/chat")
async def chat(query: UserQuery):
    response = general_agent.respond(query.query)
    return {"response": response}

