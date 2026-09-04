"""
Generate a forward N-month forecast per woreda using the trained XGBoost model.

Usage:
    python src/generate_forecast.py

Reads:  data/model_frame.csv
        models/xgb_malaria_model.json
Writes: outputs/forecast_df.csv
"""
import os

import numpy as np
import pandas as pd
import xgboost as xgb

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
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
N_FORECAST = 6


def main():
    model_frame = pd.read_csv(os.path.join(DATA_DIR, "model_frame.csv"), parse_dates=["date"])
    ml_df = model_frame.dropna(subset=["cases"]).copy()

    model = xgb.XGBRegressor()
    model.load_model(os.path.join(MODEL_DIR, "xgb_malaria_model.json"))

    last_rows = ml_df.sort_values("date").groupby("ADM3_PCODE").tail(1).copy()
    state = last_rows.set_index("ADM3_PCODE").copy()

    forecast_records = []
    for step in range(1, N_FORECAST + 1):
        Xf = state[FEATURES + CAT_FEATURES].copy()
        for c in CAT_FEATURES:
            Xf[c] = Xf[c].astype("category")
        preds = np.clip(model.predict(Xf), 0, None)

        next_month = (state["month"] % 12) + 1
        next_date = state["date"] + pd.DateOffset(months=1)

        out = pd.DataFrame({
            "ADM3_PCODE": state.index, "ADM3_EN": state["ADM3_EN"].values,
            "date": next_date.values, "forecast_cases": preds, "step_ahead": step,
        })
        forecast_records.append(out)

        state["precip_mm_l3"] = state["precip_mm_l2"]
        state["precip_mm_l2"] = state["precip_mm_l1"]
        state["precip_mm_l1"] = state["precip_mm"]
        state["temp_mean_l3"] = state["temp_mean_l2"]
        state["temp_mean_l2"] = state["temp_mean_l1"]
        state["temp_mean_l1"] = state["temp_mean"]
        state["rh_pct_l3"] = state["rh_pct_l2"]
        state["rh_pct_l2"] = state["rh_pct_l1"]
        state["rh_pct_l1"] = state["rh_pct"]
        state["ndvi_l3"] = state["ndvi_l2"]
        state["ndvi_l2"] = state["ndvi_l1"]
        state["ndvi_l1"] = state["ndvi"]
        state["month"] = next_month
        state["date"] = next_date

    forecast_df = pd.concat(forecast_records, ignore_index=True)
    forecast_df.to_csv(os.path.join(OUT_DIR, "forecast_df.csv"), index=False)
    print("Saved forecast_df.csv with shape", forecast_df.shape)


if __name__ == "__main__":
    main()
