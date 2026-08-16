"""
src/inference.py
-----------------
Real-Time Inference Engine

Given a sequence of sensor readings for an engine,
predicts its Remaining Useful Life (RUL).

Also computes a Health Index (0-100 scale):
  HI = 100  → fully healthy
  HI = 0    → about to fail

  Formula: HI = (predicted_RUL / RUL_CAP) * 100, clipped to [0, 100]

Usage (standalone):
    python src/inference.py --dataset FD001 --engine_id 1
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import LSTMModel
from src.gold_layer import WINDOW_SIZE

RUL_CAP = 125


def load_inference_artifacts(dataset_id: str = "FD001",
                              gold_dir:  str = "data/gold",
                              model_dir: str = "models"):
    """
    Loads model + scaler + feature metadata needed for inference.
    Call this ONCE at startup, then reuse the returned objects.
    """
    meta = joblib.load(os.path.join(gold_dir, f"feature_meta_{dataset_id}.pkl"))
    feature_cols = meta["feature_cols"]
    sensor_cols  = meta["sensor_cols"]

    scaler = joblib.load(os.path.join("models", f"scaler_{dataset_id}.pkl"))

    model = LSTMModel(input_size=len(feature_cols), hidden_size=64,
                      num_layers=2, dropout=0.2)
    model.load_state_dict(
        torch.load(os.path.join(model_dir, f"lstm_{dataset_id}_best.pth"),
                   map_location="cpu")
    )
    model.eval()

    return model, scaler, feature_cols, sensor_cols


def predict_rul(sensor_df: pd.DataFrame,
                model, scaler, feature_cols: list, sensor_cols: list) -> dict:
    """
    Predict RUL from a raw sensor DataFrame for one engine.

    Args:
        sensor_df : DataFrame with at least WINDOW_SIZE rows of raw sensor readings
                    Columns must include: s1..s21 (or the subset used in training)

    Returns:
        dict with:
          - predicted_rul  : float
          - health_index   : float (0-100)
          - alert_level    : str ("CRITICAL", "WARNING", "HEALTHY")
    """
    # 1. Normalize using the training scaler
    raw_sensors = [c for c in sensor_cols if c in sensor_df.columns]
    df = sensor_df.copy()
    df[raw_sensors] = scaler.transform(df[raw_sensors])

    # 2. Add lag + rolling features
    from src.gold_layer import add_lag_features, add_rolling_features
    df = add_lag_features(df, raw_sensors)
    df = add_rolling_features(df, raw_sensors)

    # 3. Take last WINDOW_SIZE rows
    available = [c for c in feature_cols if c in df.columns]
    window = df[available].values[-WINDOW_SIZE:]

    # Pad columns if needed
    if window.shape[1] < len(feature_cols):
        pad = np.zeros((window.shape[0], len(feature_cols) - window.shape[1]))
        window = np.hstack([window, pad])

    # Pad rows if engine has fewer than WINDOW_SIZE cycles
    if len(window) < WINDOW_SIZE:
        pad_rows = WINDOW_SIZE - len(window)
        window = np.vstack([np.zeros((pad_rows, window.shape[1])), window])

    # 4. Forward pass
    X = torch.tensor(window, dtype=torch.float32).unsqueeze(0)  # (1, 30, F)
    with torch.no_grad():
        pred_rul = model(X).item()

    pred_rul = max(0.0, pred_rul)   # clip to non-negative

    # 5. Health Index
    health_index = min(100.0, (pred_rul / RUL_CAP) * 100.0)

    # 6. Alert level
    if pred_rul <= 20:
        alert = "CRITICAL"
    elif pred_rul <= 50:
        alert = "WARNING"
    else:
        alert = "HEALTHY"

    return {
        "predicted_rul": round(pred_rul, 1),
        "health_index":  round(health_index, 1),
        "alert_level":   alert
    }


def predict_fleet(dataset_id: str = "FD001",
                  silver_dir: str = "data/silver",
                  gold_dir: str   = "data/gold",
                  output_dir: str = "outputs") -> pd.DataFrame:
    """
    Runs inference on all engines in the test set.
    Returns a fleet health summary DataFrame.
    """
    os.makedirs(output_dir, exist_ok=True)
    model, scaler, feature_cols, sensor_cols = load_inference_artifacts(dataset_id, gold_dir)

    # Load test file (original with all cycles, not just last)
    from src.bronze_layer import load_raw_file, COLS
    test_raw = load_raw_file(f"data/raw/test_{dataset_id}.txt")
    rul_true = pd.read_csv(f"data/raw/RUL_{dataset_id}.txt",
                           header=None, names=["rul_true"])

    results = []
    for engine_id, group in test_raw.groupby("engine_id"):
        group = group.sort_values("cycle").reset_index(drop=True)
        result = predict_rul(group, model, scaler, feature_cols, sensor_cols)
        result["engine_id"] = engine_id

        true_rul = rul_true.iloc[engine_id - 1]["rul_true"]
        result["true_rul"] = min(true_rul, RUL_CAP)
        result["error"] = result["predicted_rul"] - result["true_rul"]
        results.append(result)

    fleet_df = pd.DataFrame(results)[
        ["engine_id", "health_index", "predicted_rul", "true_rul", "error", "alert_level"]
    ].sort_values("health_index")

    fleet_df.to_csv(os.path.join(output_dir, f"fleet_health_{dataset_id}.csv"), index=False)
    print(f"[INFERENCE] Fleet health saved to outputs/fleet_health_{dataset_id}.csv")

    return fleet_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",   default="FD001")
    args = parser.parse_args()

    fleet = predict_fleet(args.dataset)
    print("\n=== Fleet Health Summary ===")
    print(fleet.to_string(index=False))

    crit = (fleet["alert_level"] == "CRITICAL").sum()
    warn = (fleet["alert_level"] == "WARNING").sum()
    ok   = (fleet["alert_level"] == "HEALTHY").sum()
    print(f"\nCRITICAL: {crit}  |  WARNING: {warn}  |  HEALTHY: {ok}")
