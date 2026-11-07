"""Feature engineering shared by training and inference (single source of truth)."""
import numpy as np
import pandas as pd

from ml.config import FESTIVALS, HOLIDAYS, FEATURES


def is_holiday(d: pd.Timestamp) -> int:
    return int((d.month, d.day) in HOLIDAYS)


def is_festival(d: pd.Timestamp) -> int:
    for m, start, end in FESTIVALS:
        if d.month == m and start <= d.day <= end:
            return 1
    return 0


def is_long_weekend(d: pd.Timestamp) -> int:
    """Holiday adjacent to a weekend (Fri/Mon holiday, or holiday on weekend edge)."""
    if is_holiday(d) and d.dayofweek in (0, 4):
        return 1
    for delta in (-1, 1):
        n = d + pd.Timedelta(days=delta)
        if is_holiday(n) and d.dayofweek in (5, 6):
            return 1
    return 0


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Expects columns: date, destination, rainfall_mm, temperature_c."""
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["month"] = out["date"].dt.month
    out["day_of_week"] = out["date"].dt.dayofweek
    out["day_of_year"] = out["date"].dt.dayofyear
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["is_holiday"] = out["date"].apply(is_holiday)
    out["is_festival"] = out["date"].apply(is_festival)
    out["is_long_weekend"] = out["date"].apply(is_long_weekend)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    return out[["date"] + FEATURES + [c for c in out.columns if c not in FEATURES + ["date"]]]


def climatology(date: pd.Timestamp) -> tuple[float, float]:
    """Typical (rainfall_mm, temperature_c) for Jharkhand by month.

    Used as a fallback when the caller doesn't provide a weather forecast.
    """
    rain = {1: 1, 2: 2, 3: 3, 4: 5, 5: 15, 6: 180, 7: 300, 8: 290,
            9: 220, 10: 60, 11: 8, 12: 2}
    temp = {1: 16, 2: 20, 3: 26, 4: 32, 5: 35, 6: 31, 7: 28, 8: 27,
            9: 27, 10: 25, 11: 20, 12: 16}
    days = pd.Period(date, freq="M").days_in_month
    return rain[date.month] / days, float(temp[date.month])
