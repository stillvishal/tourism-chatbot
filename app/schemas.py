"""Pydantic request/response models (these are the API contract for the web-dev partner)."""
from datetime import date
from typing import Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    destination: str = Field(..., examples=["Dassam Falls"])
    date: date
    rainfall_mm: Optional[float] = Field(None, ge=0, description="Forecast rainfall; typical value used if omitted")
    temperature_c: Optional[float] = Field(None, ge=-5, le=55, description="Forecast temperature; typical value used if omitted")


class PredictResponse(BaseModel):
    destination: str
    date: date
    predicted_visitors: int
    interval_low: int
    interval_high: int
    capacity: int
    occupancy_pct: float
    crowd_level: str
    crowd_emoji: str
    confidence_pct: float
    drivers: list[str]


class ForecastDay(BaseModel):
    date: date
    predicted_visitors: int
    crowd_level: str


class ForecastResponse(BaseModel):
    destination: str
    days: list[ForecastDay]
    change_vs_previous_period_pct: Optional[float] = None


class RecommendRequest(BaseModel):
    date: date
    preferences: list[str] = Field(default_factory=list, description="e.g. ['waterfall','hill_station']")
    avoid_crowds: bool = True
    top_n: int = Field(3, ge=1, le=10)


class Recommendation(BaseModel):
    destination: str
    category: str
    predicted_visitors: int
    crowd_level: str
    crowd_emoji: str
    score: float
    reason: str


class RecommendResponse(BaseModel):
    date: date
    recommendations: list[Recommendation]


class ChatRequest(BaseModel):
    message: str = Field(..., max_length=500, examples=["How crowded is Dassam Falls this weekend?"])
    context: dict = Field(default_factory=dict, description="Echo back the `context` from the previous reply")
    history: list[dict] = Field(default_factory=list,
                                description="Recent turns [{role:'user'|'assistant', content:'...'}] for the AI mode")


class ChatResponse(BaseModel):
    reply: str
    cards: list[dict] = []
    suggestions: list[str] = []
    context: dict = {}
    mode: str = "rule_based"   # "ai" when Claude answered, "rule_based" for the offline fallback


class AnomalyRequest(BaseModel):
    destination: str
    date: date
    actual_visitors: int = Field(..., ge=0)
    rainfall_mm: Optional[float] = None
    temperature_c: Optional[float] = None


class AnomalyResponse(BaseModel):
    destination: str
    date: date
    expected_visitors: int
    actual_visitors: int
    z_score: float
    is_anomaly: bool
    severity: str
    message: str
