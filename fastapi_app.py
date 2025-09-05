from fastapi import FastAPI
from pydantic import BaseModel
from main import app

fastapi_app = FastAPI()

class ResearchRequest(BaseModel):
    topic: str

@fastapi_app.post("/research")
def run_research(request: ResearchRequest):
    inputs = {"research_topic": request.topic}
    final_state = app.invoke(inputs)
    return {"report": final_state["report"]}
