"""Loads the trained model once and exposes prediction / crowd / anomaly logic."""
import json
from datetime import date as date_type, timedelta

import joblib
import numpy as np
import pandas as pd

from ml.config import CROWD_BANDS, DESTINATIONS, FEATURES, METADATA_PATH, MODEL_PATH
from ml.features import build_features, climatology

CROWD_EMOJI = {"LOW": "🟢", "MODERATE": "🟡", "HIGH": "🟠", "VERY_HIGH": "🔴"}


class ModelNotReady(RuntimeError):
    pass


class FootfallPredictor:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise ModelNotReady("Model not found. Run `python -m ml.generate_data && python -m ml.train` first.")
        self.model = joblib.load(MODEL_PATH)
        self.meta = json.loads(METADATA_PATH.read_text())
        self.resid_std = self.meta["residual_std_by_destination"]

    # ---- helpers ---------------------------------------------------------
    def _frame(self, destination: str, d: date_type, rain, temp) -> pd.DataFrame:
        ts = pd.Timestamp(d)
        c_rain, c_temp = climatology(ts)
        raw = pd.DataFrame([{
            "date": ts, "destination": destination,
            "rainfall_mm": c_rain if rain is None else rain,
            "temperature_c": c_temp if temp is None else temp,
        }])
        return build_features(raw)

    def _predict_count(self, destination, d, rain=None, temp=None) -> tuple[float, pd.DataFrame]:
        X = self._frame(destination, d, rain, temp)
        pred = float(self.model.predict(X[FEATURES])[0])
        return max(pred, 0.0), X

    @staticmethod
    def crowd_level(visitors: float, destination: str) -> str:
        occ = visitors / DESTINATIONS[destination]["capacity"]
        for limit, label in CROWD_BANDS:
            if occ < limit:
                return label
        return "VERY_HIGH"

    @staticmethod
    def _validate(destination: str):
        if destination not in DESTINATIONS:
            raise KeyError(destination)

    def _drivers(self, X: pd.DataFrame) -> list[str]:
        r = X.iloc[0]
        out = []
        if r["is_festival"]:
            out.append("Festival period")
        if r["is_holiday"]:
            out.append("Public holiday")
        if r["is_long_weekend"]:
            out.append("Long weekend")
        if r["is_weekend"]:
            out.append("Weekend")
        if r["rainfall_mm"] >= 20:
            out.append("Heavy rain expected (reduces visits)")
        if r["temperature_c"] > 34:
            out.append("Extreme heat (reduces visits)")
        if not out:
            out.append("Regular weekday / seasonal baseline")
        return out

    # ---- public API ------------------------------------------------------
    def predict(self, destination, d, rain=None, temp=None) -> dict:
        self._validate(destination)
        pred, X = self._predict_count(destination, d, rain, temp)
        sd = float(self.resid_std.get(destination, pred * 0.1))
        low, high = max(pred - 1.645 * sd, 0), pred + 1.645 * sd  # ~90% interval
        cap = DESTINATIONS[destination]["capacity"]
        level = self.crowd_level(pred, destination)
        # Confidence = how tight the interval is relative to the prediction.
        confidence = float(np.clip(100 * (1 - sd / max(pred, 1)), 0, 99))
        return {
            "destination": destination, "date": d,
            "predicted_visitors": int(round(pred)),
            "interval_low": int(round(low)), "interval_high": int(round(high)),
            "capacity": cap, "occupancy_pct": round(100 * pred / cap, 1),
            "crowd_level": level, "crowd_emoji": CROWD_EMOJI[level],
            "confidence_pct": round(confidence, 1),
            "drivers": self._drivers(X),
        }

    def forecast(self, destination, start: date_type, days: int) -> dict:
        self._validate(destination)
        rows = []
        for i in range(days):
            d = start + timedelta(days=i)
            pred, _ = self._predict_count(destination, d)
            rows.append({"date": d, "predicted_visitors": int(round(pred)),
                         "crowd_level": self.crowd_level(pred, destination)})
        # Compare with the immediately preceding period of equal length.
        prev = [self._predict_count(destination, start - timedelta(days=i + 1))[0] for i in range(days)]
        cur_total = sum(r["predicted_visitors"] for r in rows)
        change = None
        if sum(prev) > 0:
            change = round(100 * (cur_total - sum(prev)) / sum(prev), 1)
        return {"destination": destination, "days": rows, "change_vs_previous_period_pct": change}

    def monthly_profile(self, destination: str, year: int) -> dict:
        """Average predicted daily visitors per month (samples 6 days/month) -> seasonal demand chart."""
        self._validate(destination)
        names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        months = []
        for m in range(1, 13):
            vals = [self._predict_count(destination, date_type(year, m, d))[0] for d in (3, 8, 13, 18, 23, 28)]
            avg = sum(vals) / len(vals)
            months.append({"month": names[m - 1], "avg_visitors": int(round(avg)),
                           "crowd_level": self.crowd_level(avg, destination)})
        peak = max(months, key=lambda r: r["avg_visitors"])["month"]
        quiet = min(months, key=lambda r: r["avg_visitors"])["month"]
        return {"destination": destination, "year": year, "months": months, "peak_month": peak, "quietest_month": quiet}

    def recommend(self, d: date_type, preferences: list[str], avoid_crowds: bool, top_n: int) -> dict:
        """Score = preference match + (optionally) low-crowd bonus. ML predictions drive the ranking."""
        prefs = {p.lower() for p in preferences}
        scored = []
        for name, meta in DESTINATIONS.items():
            pred, _ = self._predict_count(name, d)
            occ = pred / meta["capacity"]
            level = self.crowd_level(pred, name)
            pref_match = 1.0 if (not prefs or meta["category"] in prefs) else 0.0
            crowd_score = max(0.0, 1.0 - occ) if avoid_crowds else 0.5
            score = 0.55 * pref_match + 0.45 * crowd_score
            if pref_match and avoid_crowds and level in ("LOW", "MODERATE"):
                reason = f"Matches your interests with {level.lower()} predicted crowds (~{int(pred):,} visitors)."
            elif pref_match:
                reason = f"Matches your interests, but expect {level.replace('_', ' ').lower()} crowds (~{int(pred):,} visitors)."
            else:
                reason = f"Outside your stated interests; predicted {level.replace('_', ' ').lower()} crowds."
            scored.append({
                "destination": name, "category": meta["category"],
                "predicted_visitors": int(round(pred)), "crowd_level": level,
                "crowd_emoji": CROWD_EMOJI[level], "score": round(score, 3), "reason": reason,
            })
        scored.sort(key=lambda r: -r["score"])
        return {"date": d, "recommendations": scored[:top_n]}

    def check_anomaly(self, destination, d, actual, rain=None, temp=None) -> dict:
        self._validate(destination)
        expected, _ = self._predict_count(destination, d, rain, temp)
        sd = float(self.resid_std.get(destination, expected * 0.1)) or 1.0
        z = (actual - expected) / sd
        az = abs(z)
        severity = "NORMAL" if az < 2 else "MODERATE" if az < 3 else "SEVERE"
        is_anom = az >= 2
        if is_anom:
            direction = "surge" if z > 0 else "drop"
            msg = (f"Unusual visitor {direction}: {actual:,} actual vs ~{int(expected):,} expected. "
                   "Check for events, viral trends, weather or road closures.")
        else:
            msg = "Visitor count is within the expected range."
        return {"destination": destination, "date": d, "expected_visitors": int(round(expected)),
                "actual_visitors": actual, "z_score": round(float(z), 2),
                "is_anomaly": is_anom, "severity": severity, "message": msg}
