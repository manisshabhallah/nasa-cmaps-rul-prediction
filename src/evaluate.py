"""
src/evaluate.py
----------------
Model Evaluation on Test Set

Metrics:
  - RMSE   : Root Mean Square Error (lower = better)
  - MAE    : Mean Absolute Error
  - MAPE   : Mean Absolute Percentage Error
  - NASA Score : Asymmetric score from C-MAPSS benchmark paper

Outputs:
  - outputs/eval_scatter_{dataset}.png   : Predicted vs Actual RUL scatter
  - outputs/eval_per_engine_{dataset}.png: Per-engine RUL comparison
  - outputs/eval_results_{dataset}.csv   : Numeric results table

NOTE: This file depends on the FIXED silver_layer.py, which now preserves
every cycle of each test engine (not just the last one) so gold_layer can
build a real 30-cycle window instead of one padded from a single repeated
row. Because test_df now has multiple rows per engine, we take the LAST
(most recent) row's ground-truth RUL for scoring, not the first.

Usage:
    python src/evaluate.py --dataset FD001
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import LSTMModel, cmapss_score
from src.gold_layer import add_lag_features, add_rolling_features, create_sequences, WINDOW_SIZE


def load_model(dataset_id: str, gold_dir: str = "data/gold",
               model_dir: str = "models") -> tuple:
    """Load trained model + feature metadata."""
    meta = joblib.load(os.path.join(gold_dir, f"feature_meta_{dataset_id}.pkl"))
    feature_cols = meta["feature_cols"]
    input_size   = len(feature_cols)

    model = LSTMModel(input_size=input_size, hidden_size=64,
                      num_layers=2, dropout=0.2)
    model.load_state_dict(
        torch.load(os.path.join(model_dir, f"lstm_{dataset_id}_best.pth"),
                   map_location="cpu")
    )
    model.eval()
    return model, meta


def predict_test_set(dataset_id: str, gold_dir: str = "data/gold",
                     silver_dir: str = "data/silver"):
    """
    Runs inference on the test set.
    Test evaluation uses only the LAST window from each engine.
    """
    model, meta = load_model(dataset_id, gold_dir)
    feature_cols = meta["feature_cols"]
    sensor_cols  = meta["sensor_cols"]

    # Load test data (silver — normalized, with ground truth RUL)
    test_df = pd.read_parquet(os.path.join(silver_dir, f"test_{dataset_id}_silver.parquet"))

    # We need to go back to the full test file to create windows
    # Use the gold test file which has engineered features
    test_gold = pd.read_parquet(os.path.join(gold_dir, f"test_{dataset_id}_gold.parquet"))

    # For engines with < WINDOW_SIZE cycles, pad with first row
    predictions, actuals, engine_ids_out = [], [], []

    for engine_id, group in test_gold.groupby("engine_id"):
        group = group.sort_values("cycle").reset_index(drop=True)

        # Check we have required features
        available_features = [c for c in feature_cols if c in group.columns]

        if len(group) < WINDOW_SIZE:
            # Pad by repeating the first row
            pad_rows = WINDOW_SIZE - len(group)
            pad_df = pd.concat([group.iloc[[0]] * 1] * pad_rows + [group], ignore_index=True)
            group = pad_df

        # Take the LAST window
        window = group[available_features].values[-WINDOW_SIZE:]

        # If we have fewer features than expected, pad with zeros
        if window.shape[1] < len(feature_cols):
            pad_cols = len(feature_cols) - window.shape[1]
            window = np.hstack([window, np.zeros((WINDOW_SIZE, pad_cols))])

        X_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0)  # (1, 30, F)

        with torch.no_grad():
            pred = model(X_tensor).item()

        # Ground truth RUL from test_df (silver test_df now keeps ALL cycles
        # per engine - FIXED silver_layer.py - so take the LAST recorded
        # cycle's rul, which equals the provided ground-truth rul_at_end)
        eng_rows = test_df[test_df["engine_id"] == engine_id].sort_values("cycle")
        true_rul = eng_rows["rul"].values
        if len(true_rul) > 0:
            predictions.append(max(0, pred))   # RUL can't be negative
            actuals.append(float(true_rul[-1]))
            engine_ids_out.append(engine_id)

    predictions = np.array(predictions)
    actuals     = np.array(actuals)

    return predictions, actuals, np.array(engine_ids_out)


def compute_metrics(y_true, y_pred) -> dict:
    """Computes all evaluation metrics."""
    rmse  = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae   = float(np.mean(np.abs(y_true - y_pred)))

    # MAPE — avoid division by zero for engines near EOL
    mask  = y_true > 0
    mape  = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    nasa  = cmapss_score(y_true, y_pred)

    return {"RMSE": rmse, "MAE": mae, "MAPE (%)": mape, "NASA Score": nasa}


def plot_scatter(y_true, y_pred, dataset_id, output_dir):
    """Predicted vs Actual RUL scatter plot."""
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(y_true, y_pred, alpha=0.6, s=30, color="steelblue", label="Predictions")
    lims = [0, max(y_true.max(), y_pred.max()) + 10]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")

    ax.set_xlabel("Actual RUL (cycles)", fontsize=12)
    ax.set_ylabel("Predicted RUL (cycles)", fontsize=12)
    ax.set_title(f"Predicted vs Actual RUL — {dataset_id}", fontsize=13)
    ax.legend()
    ax.set_xlim(lims); ax.set_ylim(lims)

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ax.text(0.05, 0.92, f"RMSE = {rmse:.2f}", transform=ax.transAxes,
            fontsize=11, color="darkred")

    plt.tight_layout()
    path = os.path.join(output_dir, f"eval_scatter_{dataset_id}.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  [OK] Scatter plot saved: {path}")


def plot_per_engine(y_true, y_pred, engine_ids, dataset_id, output_dir, top_n=30):
    """Bar chart comparing actual vs predicted RUL for first N engines."""
    n = min(top_n, len(y_true))
    idx = np.argsort(engine_ids)[:n]
    eids = engine_ids[idx]
    yp   = y_pred[idx]
    yt   = y_true[idx]

    x = np.arange(n)
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - width/2, yt, width, label="Actual RUL",    color="steelblue", alpha=0.8)
    ax.bar(x + width/2, yp, width, label="Predicted RUL", color="coral",     alpha=0.8)

    ax.set_xlabel("Engine ID")
    ax.set_ylabel("RUL (cycles)")
    ax.set_title(f"Per-Engine RUL — {dataset_id} (first {n} engines)")
    ax.set_xticks(x)
    ax.set_xticklabels(eids, rotation=45, fontsize=7)
    ax.legend()
    plt.tight_layout()

    path = os.path.join(output_dir, f"eval_per_engine_{dataset_id}.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  [OK] Per-engine chart saved: {path}")


def evaluate(dataset_id: str = "FD001", output_dir: str = "outputs"):
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[EVALUATE] Dataset: {dataset_id}")

    preds, actuals, engine_ids = predict_test_set(dataset_id)

    metrics = compute_metrics(actuals, preds)
    print("\n  === Evaluation Metrics ===")
    for k, v in metrics.items():
        print(f"  {k:15s} : {v:.4f}")

    # Save metrics CSV
    results_df = pd.DataFrame({
        "engine_id":    engine_ids,
        "actual_rul":   actuals,
        "predicted_rul": preds,
        "error":        preds - actuals,
        "abs_error":    np.abs(preds - actuals)
    })
    csv_path = os.path.join(output_dir, f"eval_results_{dataset_id}.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\n  [OK] Results saved: {csv_path}")

    # Save metrics summary
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(output_dir, f"metrics_{dataset_id}.csv"), index=False)

    # Plots
    plot_scatter(actuals, preds, dataset_id, output_dir)
    plot_per_engine(actuals, preds, engine_ids, dataset_id, output_dir)

    return metrics, results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="FD001")
    args = parser.parse_args()
    evaluate(args.dataset)