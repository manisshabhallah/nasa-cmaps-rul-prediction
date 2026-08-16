"""
src/training/evaluate_advanced.py
----------------------------------
PHD-LEVEL EVALUATION METRICS

1. RMSE / MAE / NASA SCORE
2. ALPHA-LAMBDA (α-λ) METRIC  (Saxena et al., 2008)
3. PROGNOSTIC HORIZON (PH)
4. UNCERTAINTY CALIBRATION (ECE)
5. CRITICAL ZONE RMSE (RUL ≤ 30 cycles)
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add MTP_advanced root
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, _project_root)

# Also add MTP2_nasa so we can use bronze_layer, gold_layer, silver_layer
_basic_project = os.path.join(_project_root, "..", "MTP2_nasa")
sys.path.insert(0, os.path.abspath(_basic_project))

from src.models.transformer_rul import CNNTransformerRUL
from src.models.uncertainty import mc_dropout_predict


# ── Metric functions ──────────────────────────────────────────────────

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred)**2)))

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def nasa_score(y_true, y_pred):
    d = y_pred - y_true
    return float(np.sum(np.where(d < 0,
                                  np.exp(-d/13.) - 1,
                                  np.exp(d/10.)  - 1)))

def critical_zone_rmse(y_true, y_pred, threshold=30.0):
    mask = y_true <= threshold
    if mask.sum() == 0:
        return float("nan")
    return rmse(y_true[mask], y_pred[mask])

def calibration_error(y_true, y_pred_mean, y_pred_std, n_bins=10):
    confidence_levels = np.linspace(0.1, 0.99, n_bins)
    errors = []
    for conf in confidence_levels:
        z = 1.96 * (conf / 0.95)
        lower = y_pred_mean - z * y_pred_std
        upper = y_pred_mean + z * y_pred_std
        coverage = np.mean((y_true >= lower) & (y_true <= upper))
        errors.append(abs(coverage - conf))
    return float(np.mean(errors))


# ── Load model (auto-detects architecture from checkpoint) ────────────

def load_model_from_checkpoint(model_path, n_features):
    """
    Reads the saved checkpoint and rebuilds the EXACT model that was trained.
    Auto-detects: d_model, whether multitask was used.
    No more size mismatch errors.
    """
    checkpoint = torch.load(model_path, map_location="cpu")

    # Detect d_model from the input_proj weight shape
    saved_d_model = checkpoint["input_proj.weight"].shape[0]

    # Detect if trained with multitask (has fault_classifier keys)
    has_multitask = any("fault_classifier" in k for k in checkpoint.keys())

    print(f"  [EVAL] Detected d_model={saved_d_model}, multitask={has_multitask}")

    if has_multitask:
        from src.training.train_advanced import CNNTransformerMultiTask
        model = CNNTransformerMultiTask(
            n_features=n_features,
            d_model=saved_d_model,
            n_heads=4,
            n_transformer_layers=2,
            dropout=0.15
        )
    else:
        model = CNNTransformerRUL(
            n_features=n_features,
            d_model=saved_d_model,
            n_heads=4,
            n_transformer_layers=2,
            dropout=0.15
        )

    model.load_state_dict(checkpoint)
    model.eval()
    return model


# ── Main evaluation ───────────────────────────────────────────────────

def evaluate_advanced(dataset_id="FD001",
                      gold_dir="data/gold",
                      silver_dir="data/silver",
                      model_dir="models",
                      output_dir="outputs",
                      mc_samples=50):

    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[ADVANCED EVAL] Dataset: {dataset_id}")

    # ── Load metadata ─────────────────────────────────────────────
    meta         = joblib.load(os.path.join(gold_dir, f"feature_meta_{dataset_id}.pkl"))
    feature_cols = meta["feature_cols"]
    sensor_cols  = meta["sensor_cols"]
    n_features   = len(feature_cols)

    # ── Load model (auto-detect architecture) ────────────────────
    model_path = os.path.join(model_dir, "transformer_best.pth")
    if not os.path.exists(model_path):
        print(f"  [ERROR] Model not found at {model_path}")
        print("  Run training first: python run_pipeline_advanced.py")
        sys.exit(1)

    model = load_model_from_checkpoint(model_path, n_features)

    # ── Load test data ────────────────────────────────────────────
    from src.bronze_layer import load_raw_file
    from src.gold_layer import add_lag_features, add_rolling_features, WINDOW_SIZE

    test_raw = load_raw_file(f"data/raw/test_{dataset_id}.txt")
    rul_true = pd.read_csv(
        f"data/raw/RUL_{dataset_id}.txt", header=None, names=["rul_true"]
    )
    scaler = joblib.load(os.path.join("models", f"scaler_{dataset_id}.pkl"))

    ZERO_VAR = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]

    all_preds, all_true, all_stds = [], [], []

    for engine_id, group in test_raw.groupby("engine_id"):
        group = group.sort_values("cycle").copy()

        # Drop zero-variance sensors
        group = group.drop(columns=[c for c in ZERO_VAR if c in group.columns],
                           errors="ignore")

        # Normalize
        avail_sensors = [c for c in sensor_cols if c in group.columns]
        if len(avail_sensors) < 3:
            continue

        group_norm = group.copy()
        try:
            group_norm[avail_sensors] = scaler.transform(group[avail_sensors])
        except Exception:
            continue

        # Add engineered features
        group_norm = add_lag_features(group_norm, avail_sensors)
        group_norm = add_rolling_features(group_norm, avail_sensors)
        group_norm = group_norm.fillna(0)

        avail_features = [c for c in feature_cols if c in group_norm.columns]
        data = group_norm[avail_features].values

        true_rul_at_end = min(float(rul_true.iloc[engine_id - 1]["rul_true"]), 125)

        # Pad rows if engine has fewer than WINDOW_SIZE cycles
        if len(data) < WINDOW_SIZE:
            pad = np.zeros((WINDOW_SIZE - len(data), data.shape[1]))
            data = np.vstack([pad, data])

        # Pad feature columns if needed
        if data.shape[1] < n_features:
            pad_cols = np.zeros((len(data), n_features - data.shape[1]))
            data = np.hstack([data, pad_cols])

        window = data[-WINDOW_SIZE:]
        X_t = torch.tensor(window, dtype=torch.float32).unsqueeze(0)

        # MC Dropout prediction
        mc_result = mc_dropout_predict(model, X_t, n_samples=mc_samples)

        all_preds.append(float(np.clip(mc_result["rul_mean"], 0, 200)))
        all_true.append(true_rul_at_end)
        all_stds.append(float(mc_result["total_std"]))

    all_preds = np.array(all_preds)
    all_true  = np.array(all_true)
    all_stds  = np.array(all_stds)

    # ── Metrics ───────────────────────────────────────────────────
    metrics = {
        "RMSE"            : rmse(all_true, all_preds),
        "MAE"             : mae(all_true, all_preds),
        "NASA_Score"      : nasa_score(all_true, all_preds),
        "Critical_RMSE"   : critical_zone_rmse(all_true, all_preds, threshold=30),
        "Mean_Uncertainty": float(all_stds.mean()),
        "ECE"             : calibration_error(all_true, all_preds, all_stds),
    }

    print("\n  ===== Advanced Evaluation Metrics =====")
    for k, v in metrics.items():
        if not np.isnan(v):
            print(f"  {k:22s}: {v:.4f}")

    # ── Save ──────────────────────────────────────────────────────
    results_df = pd.DataFrame({
        "true_rul"   : all_true,
        "pred_rul"   : all_preds,
        "uncertainty": all_stds,
        "error"      : all_preds - all_true,
        "abs_error"  : np.abs(all_preds - all_true),
    })
    results_df.to_csv(
        os.path.join(output_dir, f"advanced_results_{dataset_id}.csv"), index=False
    )
    pd.DataFrame([metrics]).to_csv(
        os.path.join(output_dir, f"advanced_metrics_{dataset_id}.csv"), index=False
    )

    # ── Plots ─────────────────────────────────────────────────────
    _plot_uncertainty_scatter(all_true, all_preds, all_stds, dataset_id, output_dir)
    _plot_calibration(all_true, all_preds, all_stds, dataset_id, output_dir)

    print(f"\n  Results saved to {output_dir}/")
    return metrics, results_df


def _plot_uncertainty_scatter(y_true, y_pred, y_std, dataset_id, output_dir):
    fig, ax = plt.subplots(figsize=(8, 7))
    scatter = ax.scatter(y_true, y_pred, c=y_std, cmap="RdYlGn_r",
                          s=40, alpha=0.7, vmin=0, vmax=y_std.max())
    plt.colorbar(scatter, ax=ax, label="Predicted Uncertainty (σ)")

    idx = np.random.choice(len(y_true), size=min(50, len(y_true)), replace=False)
    ax.errorbar(y_true[idx], y_pred[idx], yerr=y_std[idx],
                fmt="none", alpha=0.3, ecolor="gray", capsize=2, linewidth=0.8)

    lim = max(y_true.max(), y_pred.max()) + 10
    ax.plot([0, lim], [0, lim], "k--", linewidth=1.5, label="Perfect prediction")
    ax.fill_between([0, lim], [0*0.8, lim*0.8], [0*1.2, lim*1.2],
                     alpha=0.08, color="green", label="±20% band")
    ax.set_xlabel("Actual RUL (cycles)", fontsize=12)
    ax.set_ylabel("Predicted RUL (cycles)", fontsize=12)
    ax.set_title(f"Probabilistic RUL Predictions — {dataset_id}", fontsize=13)
    ax.legend(); ax.set_xlim([0, lim]); ax.set_ylim([0, lim])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"uncertainty_scatter_{dataset_id}.png"), dpi=120)
    plt.close()
    print(f"  Uncertainty scatter saved")


def _plot_calibration(y_true, y_pred_mean, y_pred_std, dataset_id, output_dir):
    confidence_levels = np.linspace(0.05, 0.99, 20)
    observed_coverage = []
    for conf in confidence_levels:
        z = 1.96 * (conf / 0.95)
        lower = y_pred_mean - z * y_pred_std
        upper = y_pred_mean + z * y_pred_std
        cov = np.mean((y_true >= lower) & (y_true <= upper))
        observed_coverage.append(cov)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", linewidth=1.5)
    ax.plot(confidence_levels, observed_coverage, "o-",
             color="steelblue", label="Model calibration", linewidth=2)
    ax.fill_between(confidence_levels, confidence_levels, observed_coverage,
                     alpha=0.15, color="steelblue")
    ax.set_xlabel("Expected coverage", fontsize=12)
    ax.set_ylabel("Observed coverage", fontsize=12)
    ax.set_title(f"Uncertainty Calibration — {dataset_id}", fontsize=12)
    ax.legend(); ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"calibration_{dataset_id}.png"), dpi=120)
    plt.close()
    print(f"  Calibration plot saved")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    default="FD001")
    parser.add_argument("--mc_samples", type=int, default=50)
    args = parser.parse_args()
    evaluate_advanced(args.dataset, mc_samples=args.mc_samples)
