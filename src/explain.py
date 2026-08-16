"""
src/explain.py
--------------
SHAP-Based Model Interpretability

WHY SHAP?
   Your LSTM is a black box — it outputs a number but doesn't tell you WHY.
   SHAP (SHapley Additive exPlanations) uses game theory to figure out:
   "How much did each sensor contribute to this prediction?"

   Based on the Shapley value from cooperative game theory:
   Each feature gets a "fair share" of the prediction contribution.

   SHAP value for feature i = how much adding sensor i changed the prediction
   from the baseline (average prediction)

For LSTMs:
   We use DeepLIFT-based SHAP (DeepExplainer) which works through backprop.
   We average SHAP values across the time dimension to get per-sensor importance.

Output:
   - outputs/shap_summary_{dataset}.png  : bar chart of mean |SHAP| per sensor
   - outputs/shap_values_{dataset}.npy   : raw SHAP values for further analysis

Usage:
    python src/explain.py --dataset FD001
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import LSTMModel

# Sensor descriptions from NASA C-MAPSS documentation
SENSOR_NAMES = {
    "s2":  "Total temperature fan inlet",
    "s3":  "Total temperature LPC outlet",
    "s4":  "Total temperature HPC outlet",
    "s7":  "Total pressure HPC outlet",
    "s8":  "Physical fan speed",
    "s9":  "Physical core speed",
    "s11": "Static pressure HPC outlet",
    "s12": "Ratio of fuel flow to Ps30",
    "s13": "Corrected fan speed",
    "s14": "Corrected core speed",
    "s15": "Bypass ratio",
    "s17": "Bleed enthalpy",
    "s20": "HPT coolant bleed",
    "s21": "LPT coolant bleed",
}


def compute_shap(dataset_id: str = "FD001",
                 gold_dir: str   = "data/gold",
                 model_dir: str  = "models",
                 output_dir: str = "outputs",
                 n_background: int = 100,
                 n_explain: int    = 200):
    """
    Computes SHAP values using DeepExplainer.

    n_background : number of background samples for SHAP baseline
    n_explain    : number of test samples to explain
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[SHAP] Computing explanations for {dataset_id}")

    # Load model + metadata
    meta = joblib.load(os.path.join(gold_dir, f"feature_meta_{dataset_id}.pkl"))
    feature_cols = meta["feature_cols"]
    sensor_cols  = meta["sensor_cols"]

    model = LSTMModel(input_size=len(feature_cols), hidden_size=64,
                      num_layers=2, dropout=0.2)
    model.load_state_dict(
        torch.load(os.path.join(model_dir, f"lstm_{dataset_id}_best.pth"),
                   map_location="cpu")
    )
    model.eval()

    # Load training data for SHAP background
    X_train = np.load(os.path.join(gold_dir, f"X_train_{dataset_id}.npy"))

    # Random subset for efficiency
    rng = np.random.default_rng(42)
    bg_idx  = rng.choice(len(X_train), size=min(n_background, len(X_train)), replace=False)
    exp_idx = rng.choice(len(X_train), size=min(n_explain,    len(X_train)), replace=False)

    X_bg  = torch.tensor(X_train[bg_idx],  dtype=torch.float32)
    X_exp = torch.tensor(X_train[exp_idx], dtype=torch.float32)

    # Try to use SHAP DeepExplainer
    try:
        import shap
        explainer = shap.DeepExplainer(model, X_bg)
        shap_values = explainer.shap_values(X_exp)   # shape: (n_exp, window, n_features)
        print(f"  [OK] SHAP values computed: {np.array(shap_values).shape}")
    except Exception as e:
        print(f"  [WARN] DeepExplainer failed ({e}), using GradientExplainer fallback")
        # Gradient-based fallback
        shap_values = _gradient_feature_importance(model, X_exp)

    shap_arr = np.array(shap_values)

    # Average across time dimension → per-feature importance
    # shap_arr shape: (n_samples, window_size, n_features)
    if shap_arr.ndim == 3:
        mean_abs_shap = np.abs(shap_arr).mean(axis=(0, 1))  # (n_features,)
    else:
        mean_abs_shap = np.abs(shap_arr).mean(axis=0)

    # Map feature names to readable sensor names
    readable_names = []
    for f in feature_cols:
        base = f.split("_")[0]   # strip _delta or _roll5
        suffix = "_delta" if "_delta" in f else ("_roll5" if "_roll5" in f else "")
        nice = SENSOR_NAMES.get(base, base) + suffix.replace("_", " ")
        readable_names.append(nice)

    # Top 15 features
    top_n = min(15, len(feature_cols))
    top_idx = np.argsort(mean_abs_shap)[-top_n:][::-1]
    top_names  = [readable_names[i] for i in top_idx]
    top_values = mean_abs_shap[top_idx]

    # Save raw SHAP values
    np.save(os.path.join(output_dir, f"shap_values_{dataset_id}.npy"), shap_arr)

    # Plot
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(range(top_n), top_values[::-1], color="steelblue", alpha=0.85)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Mean |SHAP Value| (impact on RUL prediction)", fontsize=10)
    ax.set_title(f"Feature Importance (SHAP) — {dataset_id}", fontsize=12)
    ax.axvline(0, color="black", linewidth=0.5)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, f"shap_summary_{dataset_id}.png")
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f"  [OK] SHAP summary plot saved: {plot_path}")

    # Save feature importance table
    importance_df = pd.DataFrame({
        "feature":   [feature_cols[i] for i in top_idx],
        "readable":  top_names,
        "mean_abs_shap": top_values
    })
    importance_df.to_csv(
        os.path.join(output_dir, f"feature_importance_{dataset_id}.csv"), index=False
    )

    return importance_df


def _gradient_feature_importance(model, X_tensor: torch.Tensor) -> np.ndarray:
    """
    Gradient-based feature importance as fallback when SHAP DeepExplainer fails.
    Computes d(output)/d(input) via backpropagation.
    """
    X_tensor = X_tensor.requires_grad_(True)
    output = model(X_tensor)
    output.sum().backward()
    gradients = X_tensor.grad.detach().numpy()   # (n, window, features)
    return gradients


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="FD001")
    args = parser.parse_args()
    df = compute_shap(args.dataset)
    print("\nTop 10 most important features:")
    print(df.head(10).to_string(index=False))
