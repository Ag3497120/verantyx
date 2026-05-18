import gc
import shutil
import requests
import torch
import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
import uvicorn

# Setup same path as app.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "talkie", "src"))
from talkie import Talkie
from app import memory_engine, HistoricalReporterAgent, load_historical, unload_historical

app = FastAPI(title="Verantyx 1930s Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class NewsRequest(BaseModel):
    news_text: str
    use_hybrid: bool = True

@app.post("/api/generate")
def generate_news(req: NewsRequest):
    news_text = req.news_text
    
    try:
        # Generate using Talkie 13B MLX (Bypass mode)
        load_historical()
        html_chunks = list(HistoricalReporterAgent.generate_article(news_text, max_tokens=150))
        html_article = html_chunks[-1] if html_chunks else ""
        unload_historical()
        
        # Memory
        memory_engine.migrate_memory()
        
        return {"headline": "Special Report", "subtitle": "Authorities Investigating Strange Occurrences", "html": html_article}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Only mount public directory if it exists
if os.path.exists(os.path.join(os.path.dirname(__file__), "public")):
    app.mount("/", StaticFiles(directory="public", html=True), name="public")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
