"""HTTP API and a small local demo interface."""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from openai import APIError

from .service import ConfigurationError, DocumentAssistant


app = FastAPI(title="Document RAG Assistant", version="1.0.0")
assistant = DocumentAssistant()


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


@app.get("/")
def home() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask")
def ask(payload: Question) -> dict:
    try:
        answer = assistant.ask(payload.question)
    except ConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except APIError as exc:
        raise HTTPException(status_code=502, detail="The document service is temporarily unavailable. Please try again.") from exc
    return {
        "answer": answer.answer,
        "grounded": answer.grounded,
        "citations": [{"file_id": item.file_id, "filename": item.filename} for item in answer.citations],
    }

