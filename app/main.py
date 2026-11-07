"""FastAPI entrypoint.  Run with:  uvicorn app.main:app --reload
Interactive docs: http://127.0.0.1:8000/docs
"""
from contextlib import asynccontextmanager
from datetime import date

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.schemas import (
    AnomalyRequest, AnomalyResponse, ChatRequest, ChatResponse, ForecastResponse, PredictRequest,
    PredictResponse, RecommendRequest, RecommendResponse,
)
from app.services.chatbot import TourismChatbot
from app.services.llm_chatbot import LLMChatbot, llm_enabled
from app.services.predictor import FootfallPredictor, ModelNotReady
from ml.config import BASE_DIR, DESTINATIONS


def _load_env_file():
    """Tiny .env loader (no extra dependency): reads KEY=VALUE lines from <project>/.env."""
    f = BASE_DIR / ".env"
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env_file()

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        state["predictor"] = FootfallPredictor()
    except ModelNotReady as e:
        print(f"[startup] {e}")
        state["predictor"] = None
    yield
    state.clear()


app = FastAPI(
    title="Jharkhand Tourism Intelligence API",
    description="Footfall prediction, crowd forecasting, anomaly detection and crowd-aware recommendations.",
    version="0.1.0",
    lifespan=lifespan,
)

# Allow the partner's frontend (any origin during development) to call this API.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def get_predictor() -> FootfallPredictor:
    p = state.get("predictor")
    if p is None:
        raise HTTPException(503, "Model not trained yet. Run: python -m ml.generate_data && python -m ml.train")
    return p


def _check_dest(name: str):
    if name not in DESTINATIONS:
        raise HTTPException(404, f"Unknown destination '{name}'. Valid: {list(DESTINATIONS)}")


@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok", "model_loaded": state.get("predictor") is not None,
            "chat_mode": "ai" if llm_enabled() else "rule_based"}


@app.get("/destinations", tags=["meta"])
def destinations():
    return [{"name": n, "category": m["category"], "capacity": m["capacity"]} for n, m in DESTINATIONS.items()]


@app.get("/model/info", tags=["meta"])
def model_info():
    p = get_predictor()
    return {k: p.meta[k] for k in ("best_model", "metrics", "feature_importances", "train_end", "n_rows", "data_note")}


@app.post("/predict", response_model=PredictResponse, tags=["prediction"])
def predict(req: PredictRequest):
    _check_dest(req.destination)
    return get_predictor().predict(req.destination, req.date, req.rainfall_mm, req.temperature_c)


@app.get("/forecast/{destination}", response_model=ForecastResponse, tags=["prediction"])
def forecast(destination: str, start: date = Query(default_factory=date.today), days: int = Query(7, ge=1, le=60)):
    _check_dest(destination)
    return get_predictor().forecast(destination, start, days)


@app.post("/recommend", response_model=RecommendResponse, tags=["recommendation"])
def recommend(req: RecommendRequest):
    return get_predictor().recommend(req.date, req.preferences, req.avoid_crowds, req.top_n)


@app.post("/chat", response_model=ChatResponse, tags=["chatbot"])
def chat(req: ChatRequest):
    predictor = get_predictor()
    if llm_enabled():
        try:
            out = LLMChatbot(predictor).respond(req.message, req.history, req.context)
            return {**out, "mode": "ai"}
        except Exception as e:  # bad key, no internet, rate limit... -> fall back so the chat never dies
            print(f"[chat] AI mode failed ({type(e).__name__}: {e}); using rule-based fallback")
    out = TourismChatbot(predictor).respond(req.message, req.context)
    return {**out, "mode": "rule_based"}


@app.get("/chat-ui", include_in_schema=False)
def chat_ui():
    """Standalone chat window. Open in a popup / new tab / iframe."""
    return FileResponse(Path(__file__).parent / "static" / "chat.html")


@app.post("/anomaly", response_model=AnomalyResponse, tags=["analytics"])
def anomaly(req: AnomalyRequest):
    _check_dest(req.destination)
    return get_predictor().check_anomaly(req.destination, req.date, req.actual_visitors, req.rainfall_mm, req.temperature_c)
