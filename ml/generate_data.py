"""Generate a SYNTHETIC historical dataset for prototyping.

IMPORTANT: this is simulated data with hand-written relationships (weekend lift,
rain penalty, festival spikes, seasonality). It exists only so the pipeline and API
can be demonstrated. Replace data/tourism_data.csv with real visitor records
(same columns) before making any claims about accuracy.

Columns: date, destination, visitors, rainfall_mm, temperature_c
"""
import numpy as np
import pandas as pd

from ml.config import DATA_PATH, DESTINATIONS
from ml.features import climatology, is_festival, is_holiday


def generate(start="2022-01-01", end="2025-12-31", seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    rows = []
    for name, meta in DESTINATIONS.items():
        for d in dates:
            clim_rain, clim_temp = climatology(d)
            # Rain is bursty: most days dry, some days heavy, scaled by monsoon intensity.
            rain = max(0.0, rng.gamma(shape=0.6, scale=max(clim_rain, 0.3) / 0.6) * rng.binomial(1, min(0.9, 0.15 + clim_rain / 15)))
            temp = clim_temp + rng.normal(0, 2.0)

            weekend = d.dayofweek >= 5
            holiday = is_holiday(d)
            festival = is_festival(d)

            # Seasonality: cosine peaking at the destination's peak month.
            season = 1 + 0.35 * np.cos(2 * np.pi * (d.month - meta["peak_month"]) / 12)

            mult = season
            mult *= 1.30 if weekend else 1.0
            mult *= 1.45 if holiday else 1.0
            mult *= 1.60 if festival else 1.0
            mult *= 1 + meta["rain_sens"] * min(rain / 40.0, 1.5)
            # Extreme heat discourages outdoor trips.
            if temp > 34:
                mult *= 0.85

            # Special case: Deoghar is driven by Shravani Mela (Jul-Aug).
            if name == "Deoghar" and d.month in (7, 8):
                mult *= 1.5

            mean = meta["base"] * mult
            visitors = max(0, int(rng.normal(mean, mean * 0.10)))
            rows.append((d.date(), name, visitors, round(rain, 1), round(temp, 1)))

    df = pd.DataFrame(rows, columns=["date", "destination", "visitors", "rainfall_mm", "temperature_c"])
    return df


if __name__ == "__main__":
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = generate()
    df.to_csv(DATA_PATH, index=False)
    print(f"Wrote {len(df):,} rows -> {DATA_PATH}")
    print(df.describe(include="all").T[["count", "mean", "min", "max"]])
