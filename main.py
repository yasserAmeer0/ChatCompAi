from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from agents.general_agent import GeneralAgent
from agents.admission_cs import AdmissionAgent
from agents.ai_agent import AIAgent

class ChatbotSystem:
    def __init__(self):
        self.general_agent = GeneralAgent()
        self.admission_agent = AdmissionAgent()
        self.ai_agent = AIAgent()

    def route_query(self, user_input):
        user_input_lower = user_input.lower()
        if "admission" in user_input_lower or "concordia" in user_input_lower or "computer science" in user_input_lower or "electives" in user_input_lower or "concordia university" in user_input_lower:
            return self.admission_agent.respond(user_input)
        elif "ai" in user_input_lower or "artificial intelligence" in user_input_lower or "machine learning" in user_input_lower or "ml" in user_input_lower or "deeplearning" in user_input_lower
        or "machinelearning" in user_input_lower or "deep learning" in user_input_lower:
            return self.ai_agent.respond(user_input)
        else:
            return self.general_agent.respond(user_input)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
chatbot = ChatbotSystem()

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    response = chatbot.route_query(request.message)
    return {"response": response}

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})