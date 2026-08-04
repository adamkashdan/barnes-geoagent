"""
FastAPI service exposing the Barnes Ice Cap GeoAgent.
This enables a client or frontend to ask questions and receive maps.
"""
from __future__ import annotations
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from agent import run_agent
from tools import list_datasets

app = FastAPI(
    title="Barnes Ice Cap GeoAgent",
    description="An agentic tool for querying MCoRDS L2 ice thickness radar data over Baffin Island, Canada (Barnes Ice Cap 2015)",
    version="0.1.0"
)


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    image_base64: Optional[str] = None


@app.get("/datasets")
def get_datasets():
    """Lists the available variables and dataset layers that can be queried."""
    return list_datasets()


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """Answers a natural-language question about the Barnes Ice Cap by calling relevant GIS tools."""
    result = run_agent(req.question, verbose=False)
    return AskResponse(**result)


@app.get("/health")
def health():
    """Simple health check endpoint."""
    return {"status": "ok", "dataset_loaded": True}
