"""
FastAPI wrapper for the RAG pipeline.
Run locally:
    uvicorn api:app --reload --port 8000
Then test:
    http://127.0.0.1:8000/docs   (Swagger UI, auto-generated)
"""
import os
import tempfile
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
# Import your existing RAG pipeline
from pipeline import run_pipeline
# Import your new STT function
from stt import transcribe 
app = FastAPI(title="RAG Voice QA API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class QueryRequest(BaseModel):
    query: str
    k: int = 6
class QueryResponse(BaseModel):
    query: str
    answer: str
    blocked_reason: str | None = None
    timings: dict
@app.get("/")
def health_check():
    """Simple endpoint to confirm the server is up."""
    return {"status": "ok", "message": "RAG API is running"}
@app.post("/ask", response_model=QueryResponse)
def ask(request: QueryRequest):
    """
    Standard text endpoint.
    """
    result = run_pipeline(request.query, k=request.k)
    return QueryResponse(
        query=request.query,
        answer=result.get("answer", ""),
        blocked_reason=result.get("blocked_reason"),
        timings=result.get("timings", {}),
    )
# --- NEW ENDPOINT FOR VOICE ---
@app.post("/ask-voice", response_model=QueryResponse)
async def ask_voice(file: UploadFile = File(...)):
    """
    Accepts an audio file, transcribes it via Sarvam AI, 
    and passes the text to the RAG pipeline.
    """
    # 1. Save the uploaded audio to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(await file.read())
        temp_audio_path = temp_audio.name
    try:
        # 2. Transcribe the audio using your stt.py script
        transcribed_text = transcribe(temp_audio_path)
        
        # Fallback if the audio is completely silent or fails
        if not transcribed_text:
            return QueryResponse(
                query="",
                answer="I couldn't hear or understand the audio. Please try again.",
                blocked_reason="STT_FAILED",
                timings={}
            )
        # 3. Pass the transcribed text into your RAG pipeline
        result = run_pipeline(transcribed_text, k=6)
        # 4. Return the standard RAG response format
        return QueryResponse(
            query=transcribed_text,
            answer=result.get("answer", ""),
            blocked_reason=result.get("blocked_reason"),
            timings=result.get("timings", {}),
        )
    finally:
        # 5. Clean up the temp file so your drive doesn't fill up!
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)