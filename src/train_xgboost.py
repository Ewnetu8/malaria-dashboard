"""
Train the XGBoost malaria-case forecasting model.

Usage:
    python src/train_xgboost.py

Reads:  data/model_frame.csv
Writes: models/xgb_malaria_model.json
        outputs/xgb_feature_importance.html
        outputs/xgb_metrics.json
"""
import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

FEATURES = [
    "temp_mean", "temp_max", "temp_min", "temp_range", "dewpoint", "rh_pct", "wind_speed",
    "precip_mm", "precip_mm_l1", "precip_mm_l2", "precip_mm_l3",
    "temp_mean_l1", "temp_mean_l2", "temp_mean_l3",
    "rh_pct_l1", "rh_pct_l2", "rh_pct_l3",
    "ndvi", "ndvi_l1", "ndvi_l2", "ndvi_l3", "evi", "lai",
    "population", "facility_count", "month",
]
CAT_FEATURES = ["burden_tier"]
TARGET = "cases"


def main():
    model_frame = pd.read_csv(os.path.join(DATA_DIR, "model_frame.csv"), parse_dates=["date"])
    ml_df = model_frame.dropna(subset=[TARGET]).copy()

    cutoff_date = ml_df["date"].quantile(0.85)
    train_mask = ml_df["date"] <= cutoff_date
    test_mask = ~train_mask

    X = ml_df[FEATURES + CAT_FEATURES].copy()
    for c in CAT_FEATURES:
        X[c] = X[c].astype("category")
    y = ml_df[TARGET].astype(float)

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    model = xgb.XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="reg:tweedie", tweedie_variance_power=1.3,
        enable_categorical=True, n_jobs=-1, random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    pred = np.clip(model.predict(X_test), 0, None)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    mae = float(mean_absolute_error(y_test, pred))
    baseline_rmse = float(np.sqrt(mean_squared_error(y_test, np.full_like(y_test, y_train.mean()))))

    print(f"XGBoost  RMSE: {rmse:,.1f}   MAE: {mae:,.1f}")
    print(f"Baseline RMSE: {baseline_rmse:,.1f}")

    model.save_model(os.path.join(MODEL_DIR, "xgb_malaria_model.json"))

    importance = pd.Series(model.feature_importances_, index=X.columns).sort_values()
    fig = px.bar(importance, orientation="h", title="XGBoost feature importance")
    fig.update_layout(showlegend=False, height=600)
    fig.write_html(os.path.join(OUT_DIR, "xgb_feature_importance.html"), include_plotlyjs='cdn')

    with open(os.path.join(OUT_DIR, "xgb_metrics.json"), "w") as f:
        json.dump({"rmse": rmse, "mae": mae, "baseline_rmse": baseline_rmse,
                    "cutoff_date": str(cutoff_date.date())}, f, indent=2)

    print("Saved model + plots + metrics.")


if __name__ == "__main__":
    main()
