"""Train the footfall model, evaluate on a time-based holdout, and save artifacts.

Run from the project root:  python -m ml.train
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ml.config import CATEGORICAL, DATA_PATH, FEATURES, METADATA_PATH, MODEL_PATH, NUMERIC
from ml.features import build_features


def make_pipeline(model) -> Pipeline:
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)],
        remainder="passthrough",
    )
    return Pipeline([("pre", pre), ("model", model)])


def evaluate(pipe, X, y, label):
    pred = pipe.predict(X)
    return {
        "model": label,
        "MAE": round(float(mean_absolute_error(y, pred)), 1),
        "MAPE_%": round(float(mean_absolute_percentage_error(y, pred)) * 100, 2),
        "R2": round(float(r2_score(y, pred)), 4),
    }


def main():
    if not DATA_PATH.exists():
        raise SystemExit(f"{DATA_PATH} not found. Run: python -m ml.generate_data")

    raw = pd.read_csv(DATA_PATH)
    df = build_features(raw).sort_values("date").reset_index(drop=True)

    # Time-based split (no shuffling) -> honest estimate of forecasting ability.
    cutoff = df["date"].max() - pd.Timedelta(days=365)
    train, test = df[df["date"] <= cutoff], df[df["date"] > cutoff]
    X_tr, y_tr = train[FEATURES], train["visitors"]
    X_te, y_te = test[FEATURES], test["visitors"]
    print(f"Train: {len(train):,} rows (to {cutoff.date()})  |  Test: {len(test):,} rows")

    candidates = {
        "RandomForest": RandomForestRegressor(n_estimators=250, min_samples_leaf=2, n_jobs=-1, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=400, learning_rate=0.05, max_depth=4, random_state=42),
    }

    results, fitted = [], {}
    for name, model in candidates.items():
        pipe = make_pipeline(model).fit(X_tr, y_tr)
        fitted[name] = pipe
        res = evaluate(pipe, X_te, y_te, name)
        results.append(res)
        print(res)

    best = min(results, key=lambda r: r["MAE"])["model"]
    print(f"Best model: {best}")

    # Residual spread per destination -> used for prediction intervals and anomaly detection.
    best_pipe = fitted[best]
    test = test.assign(pred=best_pipe.predict(X_te))
    test["resid"] = test["visitors"] - test["pred"]
    resid_std = test.groupby("destination")["resid"].std().round(1).to_dict()

    # Refit the winner on ALL data for deployment.
    final = make_pipeline(candidates[best].__class__(**candidates[best].get_params()))
    final.fit(df[FEATURES], df["visitors"])

    # Feature importances (aggregate one-hot columns back to the original feature).
    names = final.named_steps["pre"].get_feature_names_out()
    imps = final.named_steps["model"].feature_importances_
    agg = {}
    for n, v in zip(names, imps):
        key = "destination" if n.startswith("cat__destination") else n.replace("remainder__", "")
        agg[key] = agg.get(key, 0.0) + float(v)
    importances = dict(sorted(((k, round(v, 4)) for k, v in agg.items()), key=lambda kv: -kv[1]))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, MODEL_PATH)
    meta = {
        "best_model": best,
        "metrics": results,
        "residual_std_by_destination": resid_std,
        "feature_importances": importances,
        "train_end": str(cutoff.date()),
        "n_rows": int(len(df)),
        "data_note": "Trained on synthetic data unless data/tourism_data.csv was replaced with real records.",
    }
    METADATA_PATH.write_text(json.dumps(meta, indent=2))
    print(f"Saved model -> {MODEL_PATH}")
    print("Top features:", list(importances.items())[:5])


if __name__ == "__main__":
    main()
