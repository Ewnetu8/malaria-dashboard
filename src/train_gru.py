"""
Train the GRU malaria-case forecasting model.

Usage:
    python src/train_gru.py

Reads:  data/model_frame.csv
Writes: models/gru_malaria_model.keras
        models/gru_scaler.json          (feature mean/std, needed to reuse the model)
        outputs/gru_training_curve.html
        outputs/gru_metrics.json
"""
import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

LOOKBACK = 6
SEQ_FEATURES = ["cases", "precip_mm", "temp_mean", "rh_pct", "ndvi", "evi"]


def make_sequences(df, group_col, feats, target, lookback):
    X_list, y_list, meta = [], [], []
    for pcode, g in df.groupby(group_col):
        g = g.sort_values("date").reset_index(drop=True)
        vals = g[feats].values.astype(float)
        tgt = g[target].values.astype(float)
        for i in range(lookback, len(g)):
            window = vals[i - lookback:i]
            t = tgt[i]
            if np.isnan(window).any() or np.isnan(t):
                continue
            X_list.append(window)
            y_list.append(t)
            meta.append((pcode, g["date"].iloc[i]))
    return np.array(X_list), np.array(y_list), meta


def main():
    model_frame = pd.read_csv(os.path.join(DATA_DIR, "model_frame.csv"), parse_dates=["date"])
    X_seq, y_seq, seq_meta = make_sequences(model_frame, "ADM3_PCODE", SEQ_FEATURES, "cases", LOOKBACK)
    seq_dates = pd.to_datetime([m[1] for m in seq_meta])
    print("Sequence tensor:", X_seq.shape)

    seq_cutoff = pd.Series(seq_dates).quantile(0.85)
    train_mask = seq_dates <= seq_cutoff
    test_mask = ~train_mask

    n_feat = X_seq.shape[-1]
    flat_train = X_seq[train_mask].reshape(-1, n_feat)
    feat_mean, feat_std = flat_train.mean(axis=0), flat_train.std(axis=0) + 1e-6

    X_scaled = (X_seq - feat_mean) / feat_std
    y_log = np.log1p(y_seq)

    Xs_train, Xs_test = X_scaled[train_mask], X_scaled[test_mask]
    ys_train, ys_test = y_log[train_mask], y_log[test_mask]
    y_test_raw = y_seq[test_mask]

    tf.random.set_seed(42)
    model = tf.keras.Sequential([
        layers.Input(shape=(LOOKBACK, n_feat)),
        layers.GRU(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.GRU(32),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")

    early_stop = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    history = model.fit(
        Xs_train, ys_train, validation_split=0.1, epochs=60, batch_size=256,
        callbacks=[early_stop], verbose=2,
    )

    pred_log = model.predict(Xs_test).flatten()
    pred = np.clip(np.expm1(pred_log), 0, None)
    rmse = float(np.sqrt(mean_squared_error(y_test_raw, pred)))
    mae = float(mean_absolute_error(y_test_raw, pred))
    print(f"GRU  RMSE: {rmse:,.1f}   MAE: {mae:,.1f}")

    model.save(os.path.join(MODEL_DIR, "gru_malaria_model.keras"))
    with open(os.path.join(MODEL_DIR, "gru_scaler.json"), "w") as f:
        json.dump({"features": SEQ_FEATURES, "lookback": LOOKBACK,
                    "mean": feat_mean.tolist(), "std": feat_std.tolist()}, f, indent=2)

    fig = px.line(pd.DataFrame(history.history), title="GRU training curves (log1p-cases MSE)")
    fig.write_html(os.path.join(OUT_DIR, "gru_training_curve.html"), include_plotlyjs='cdn')

    with open(os.path.join(OUT_DIR, "gru_metrics.json"), "w") as f:
        json.dump({"rmse": rmse, "mae": mae}, f, indent=2)

    print("Saved model + scaler + plots + metrics.")


if __name__ == "__main__":
    main()
