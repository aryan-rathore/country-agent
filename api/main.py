import os
import time
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from agent.graph import country_agent
from agent.state import AgentState

app = FastAPI(
    title="Country Information AI Agent",
    description="Ask natural language questions about any country",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

@app.get("/")
async def serve_home():
    return FileResponse("frontend/index.html")

class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)

class AgentResponse(BaseModel):
    answer: str
    country_detected: str | None
    fields_requested: list[str]
    processing_time_ms: int

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/ask", response_model=AgentResponse)
async def ask_question(request: QuestionRequest):
    start = time.time()
    logger.info(f"Question received: {request.question}")
    try:
        initial_state: AgentState = {
            "user_question": request.question,
            "country_name": None,
            "requested_fields": None,
            "intent_error": None,
            "raw_country_data": None,
            "api_error": None,
            "final_answer": None,
        }
        result = await country_agent.ainvoke(initial_state)
        elapsed = int((time.time() - start) * 1000)
        return AgentResponse(
            answer=result["final_answer"],
            country_detected=result.get("country_name"),
            fields_requested=result.get("requested_fields") or [],
            processing_time_ms=elapsed,
        )
    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal agent error")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)