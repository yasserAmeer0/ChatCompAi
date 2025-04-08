from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from agents.general_agent import GeneralAgent
from agents.admission_cs import AdmissionAgent
from agents.ai_agent import AIAgent
import spacy

class ChatbotSystem:
    def __init__(self):
        self.general_agent = GeneralAgent()
        self.admission_agent = AdmissionAgent()
        self.ai_agent = AIAgent()
        self.last_agent = None

        # Load spaCy model for topic extraction
        self.nlp = spacy.load("en_core_web_sm")

    def detect_topic(self, text):
        doc = self.nlp(text.lower())

        # Heuristic: use entity labels or keywords
        keywords = set(token.lemma_ for token in doc)

        admission_keywords = {"concordia", "university", "admission", "program", "elective", "course", "application"}
        ai_keywords = {"ai", "artificial", "intelligence", "machine", "learning", "deep", "model"}

        if keywords & admission_keywords:
            return "admission"
        elif keywords & ai_keywords:
            return "ai"
        else:
            return "general"

    def route_query(self, user_input):
        topic = self.detect_topic(user_input)

        if topic == "admission":
            self.last_agent = self.admission_agent
        elif topic == "ai":
            self.last_agent = self.ai_agent
        elif self.last_agent:
            # Use previous agent to maintain context
            return self.last_agent.respond(user_input)
        else:
            self.last_agent = self.general_agent

        return self.last_agent.respond(user_input)

# FastAPI Setup
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
