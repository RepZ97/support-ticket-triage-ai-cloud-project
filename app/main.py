"""Support ticket triage API."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import classifier, llm

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    classifier.load()
    yield


app = FastAPI(
    title="Support Ticket Triage",
    description="Routes consumer finance complaints and drafts an agent reply.",
    version="1.0.0",
    lifespan=lifespan,
)


class TriageRequest(BaseModel):
    text: str = Field(min_length=20, max_length=20000)
    include_assessment: bool = True


@app.get("/health")
def health():
    meta = classifier.metadata()
    return {
        "status": "ok",
        "model_trained_at": meta["trained_at"],
        "assessment": llm.status(),
    }


@app.get("/model-info")
def model_info():
    meta = classifier.metadata()
    return {
        "served_model": meta["served_model"],
        "candidates": meta["candidates"],
        "classes": meta["classes"],
        "train_rows": meta["train_rows"],
        "test_rows": meta["test_rows"],
        "vectorizer": meta["vectorizer"],
        "metrics": meta["metrics"],
        "confusion_matrix": meta["confusion_matrix"],
    }


@app.post("/triage")
def triage(request: TriageRequest):
    routing = classifier.predict(request.text)
    result = {"routing": routing, "assessment": None, "assessment_error": None}

    if request.include_assessment:
        assessment, error = llm.assess(request.text, routing["category"])
        result["assessment"] = assessment
        result["assessment_error"] = error

    return result


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
