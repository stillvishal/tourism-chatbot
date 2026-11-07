"""Shared configuration for the Jharkhand tourism footfall prototype."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "tourism_data.csv"
MODEL_PATH = BASE_DIR / "models" / "footfall_model.joblib"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"

# Destination metadata used both for synthetic data generation and the API.
# base = typical daily visitors, rain_sens = how strongly rain hurts visits (negative)
# season_peak = month of peak demand, category = used by the recommender.
DESTINATIONS = {
    "Dassam Falls":   {"base": 1500, "capacity": 5000, "rain_sens": -0.35, "peak_month": 10, "category": "waterfall"},
    "Hundru Falls":   {"base": 1800, "capacity": 6000, "rain_sens": -0.30, "peak_month": 9,  "category": "waterfall"},
    "Netarhat":       {"base": 1000, "capacity": 3500, "rain_sens": -0.15, "peak_month": 12, "category": "hill_station"},
    "Deoghar":        {"base": 4500, "capacity": 20000, "rain_sens": -0.05, "peak_month": 8,  "category": "religious"},
    "Betla National Park": {"base": 900, "capacity": 3000, "rain_sens": -0.40, "peak_month": 1, "category": "wildlife"},
    "Patratu Valley": {"base": 1300, "capacity": 4500, "rain_sens": -0.20, "peak_month": 11, "category": "scenic"},
}

# Indian public holidays / festivals (fixed-date approximations for prototype purposes).
HOLIDAYS = {
    (1, 1): "New Year", (1, 26): "Republic Day", (8, 15): "Independence Day",
    (10, 2): "Gandhi Jayanti", (12, 25): "Christmas",
}
# Approximate festival windows (month, day_start, day_end).
FESTIVALS = [
    (3, 6, 8),     # Holi
    (8, 10, 20),   # Shravani Mela (Deoghar)
    (10, 15, 25),  # Durga Puja / Dussehra
    (11, 1, 5),    # Diwali / Chhath window
]

FEATURES = [
    "destination", "month", "day_of_week", "day_of_year", "is_weekend",
    "is_holiday", "is_festival", "rainfall_mm", "temperature_c",
    "is_long_weekend", "month_sin", "month_cos",
]
CATEGORICAL = ["destination"]
NUMERIC = [f for f in FEATURES if f not in CATEGORICAL]

# Crowd level thresholds as a fraction of destination capacity.
CROWD_BANDS = [
    (0.40, "LOW"),
    (0.65, "MODERATE"),
    (0.90, "HIGH"),
    (float("inf"), "VERY_HIGH"),
]
