import io
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend import pipeline as P

app = FastAPI(title="ESG / Safety Incident Intelligence Dashboard", version="1.0")
FRONTEND = Path(__file__).resolve().parents[1] / "frontend" / "index.html"


class Query(BaseModel):
    text: str = Field("", max_length=2000)


class Batch(BaseModel):
    texts: list[str] = Field(..., max_length=200)


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(FRONTEND)


@app.get("/api/health")
def health():
    return {"status": "ok", "incidents_loaded": len(P.get_model().df)}


@app.get("/api/demo")
def demo():
    return {"case_study": app.title, "status": "running", "endpoints": ["/api/dashboard", "/api/analyze", "/api/run", "/api/batch", "/api/upload", "/api/evaluation", "/docs"]}


@app.post("/api/analyze")
def analyze(q: Query):
    if not q.text.strip():
        raise HTTPException(422, "text must not be empty")
    return P.get_model().analyze([q.text])[0]


@app.post("/api/run")  # kept for compatibility with the starter contract
def run_case(q: Query):
    return analyze(q)


@app.post("/api/batch")
def batch(b: Batch):
    rows = P.get_model().analyze([t for t in b.texts if t.strip()])
    return {"results": rows, "dashboard": P.summarize(rows)}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(io.BytesIO(await file.read()))
    except Exception:
        raise HTTPException(400, "could not parse CSV")
    if "narrative" not in df.columns:
        raise HTTPException(422, "CSV needs a 'narrative' column")
    df = df.dropna(subset=["narrative"]).head(500)
    rows = P.get_model().analyze(df.narrative.astype(str).tolist())
    if "incident_id" in df.columns:
        for r, i in zip(rows, df.incident_id):
            r["incident_id"] = i
    return {"results": rows, "dashboard": P.summarize(rows)}


@app.get("/api/dashboard")
def dashboard():
    return P.corpus_dashboard()[0]


@app.get("/api/incidents")
def incidents():
    return P.corpus_dashboard()[1]


@app.get("/api/evaluation")
def evaluation():
    return P.evaluate()
