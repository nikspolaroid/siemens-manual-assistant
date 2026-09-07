"""HTTP API and static file server for the manual assistant."""
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from config import ROOT, TOP_K
from db import get_collection, get_model
from search import search
from llm import ask as call_llm
from ask import SYSTEM, build_context

STATIC = ROOT / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the embedding model at boot, not on the first question,
    # so the demo does not stall for 10 seconds on the opening query.
    print("Loading embedding model...")
    get_model()
    print(f"Ready. {get_collection().count()} chunks indexed.")
    yield


app = FastAPI(title="Siemens Manual Assistant", lifespan=lifespan)


class Query(BaseModel):
    question: str
    k: int = TOP_K


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/status")
def status():
    return {"chunks": get_collection().count()}


@app.post("/api/ask")
def api_ask(q: Query):
    question = q.question.strip()
    if not question:
        raise HTTPException(400, "Question is empty")

    hits = search(question, k=max(1, min(q.k, 12)))
    prompt = (
        f"Manual excerpts:\n\n{build_context(hits)}\n\n"
        f"Technician's question: {question}"
    )
    try:
        answer = call_llm(SYSTEM, prompt)
    except Exception as e:
        raise HTTPException(502, f"LLM call failed: {e}")

    return {
        "answer": answer,
        "sources": [
            {"source": h["source"], "page": h["page"],
             "distance": round(h["distance"], 3), "text": h["text"]}
            for h in hits
        ],
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
