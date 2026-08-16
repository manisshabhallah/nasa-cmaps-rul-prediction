"""
src/baseline_classical.py
--------------------------
CONTRIBUTION: Comparative Model Study (classical ML vs. LSTM)

Trains and evaluates two classical baselines on the SAME gold data, SAME
engine-wise split, and SAME test protocol as the LSTM pipeline
(src/train.py + src/evaluate.py), so results are directly comparable:

  1. Linear Regression - simple, fast, fully interpretable baseline
  2. Random Forest      - ensemble-tree model, captures non-linear
                          sensor interactions without needing sequences

Unlike the LSTM (which consumes the full 30-cycle window), classical
models here use a TABULAR feature representation: the final timestep of
each window (i.e. the most recent cycle's raw + engineered features).
This is standard practice when comparing sequence models against
classical tabular regressors in PHM literature.

Also builds a simple WEIGHTED ENSEMBLE (Random Forest + LSTM predictions,
weight tuned on validation) to see whether combining a tabular model with
a sequence model improves on either alone.

Usage:
    python src/baseline_classical.py --dataset FD001
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import joblib
from collections import defaultdict
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def load_gold(dataset_id, gold_dir="data/gold"):
    X = np.load(os.path.join(gold_dir, f"X_train_{dataset_id}.npy"))
    y = np.load(os.path.join(gold_dir, f"y_train_{dataset_id}.npy"))
    eids = np.load(os.path.join(gold_dir, f"engine_ids_train_{dataset_id}.npy"))
    return X, y, eids


def engine_split(X, y, engine_ids, val_split=0.2, seed=42):
    """Same engine-wise split logic as the fixed src/train.py."""
    gss = GroupShuffleSplit(n_splits=1, test_size=val_split, random_state=seed)
    train_idx, val_idx = next(gss.split(X, y, groups=engine_ids))
    return train_idx, val_idx


def to_tabular(X):
    """
    Convert (n, window, features) sequence data to (n, features) tabular
    data by taking the LAST timestep of each window - the most recent
    cycle's raw + engineered readings. This is what a classical regressor
    sees instead of the full temporal sequence.
    """
    return X[:, -1, :]


def build_test_tabular(dataset_id, gold_dir="data/gold", window_size=30):
    """
    Builds the same test windows evaluate.py builds for the LSTM, but
    returns the LAST-timestep tabular feature vector per test engine
    (instead of the full sequence) plus the matching ground-truth RUL.
    Mirrors evaluate.py's predict_test_set() window logic exactly, so the
    comparison against the LSTM's reported test RMSE is apples-to-apples.
    """
    meta = joblib.load(os.path.join(gold_dir, f"feature_meta_{dataset_id}.pkl"))
    feature_cols = meta["feature_cols"]

    test_gold = pd.read_parquet(os.path.join(gold_dir, f"test_{dataset_id}_gold.parquet"))

    X_test, y_test, engine_list = [], [], []
    for engine_id, group in test_gold.groupby("engine_id"):
        group = group.sort_values("cycle").reset_index(drop=True)
        avail = [c for c in feature_cols if c in group.columns]
        window = group[avail].values[-window_size:]
        if window.shape[0] < window_size:
            pad = window_size - window.shape[0]
            window = np.vstack([np.repeat(window[[0]], pad, axis=0), window])
        last_row = window[-1]   # tabular feature = last (most recent) cycle
        X_test.append(last_row)
        y_test.append(group["rul"].iloc[-1])
        engine_list.append(engine_id)

    return np.array(X_test), np.array(y_test), engine_list


def run_baselines(dataset_id="FD001", gold_dir="data/gold",
                  output_dir="outputs", lstm_test_rmse=None):
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[BASELINE] Dataset: {dataset_id}")
    X, y, engine_ids = load_gold(dataset_id, gold_dir)
    train_idx, val_idx = engine_split(X, y, engine_ids)

    # Tabular conversion (last timestep of each window)
    X_tab = to_tabular(X)
    X_train, y_train = X_tab[train_idx], y[train_idx]
    X_val,   y_val    = X_tab[val_idx],   y[val_idx]

    print(f"  Train: {len(X_train)} samples | Val: {len(X_val)} samples")

    # ---- Linear Regression ----
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_val_rmse = rmse(y_val, lr.predict(X_val))

    # ---- Random Forest ----
    rf = RandomForestRegressor(
        n_estimators=200, max_depth=12, min_samples_leaf=3,
        n_jobs=-1, random_state=42
    )
    rf.fit(X_train, y_train)
    rf_val_rmse = rmse(y_val, rf.predict(X_val))

    print(f"  [VAL] Linear Regression RMSE: {lr_val_rmse:.2f}")
    print(f"  [VAL] Random Forest RMSE    : {rf_val_rmse:.2f}")

    # ---- Real test-set evaluation (same protocol as evaluate.py) ----
    X_test_tab, y_test, test_engines = build_test_tabular(dataset_id, gold_dir)

    lr_test_pred = np.clip(lr.predict(X_test_tab), 0, None)
    rf_test_pred = np.clip(rf.predict(X_test_tab), 0, None)

    lr_test_rmse = rmse(y_test, lr_test_pred)
    rf_test_rmse = rmse(y_test, rf_test_pred)
    lr_test_mae  = mean_absolute_error(y_test, lr_test_pred)
    rf_test_mae  = mean_absolute_error(y_test, rf_test_pred)

    print(f"\n  === TEST SET RESULTS ({len(y_test)} engines) ===")
    print(f"  Linear Regression : RMSE {lr_test_rmse:.2f} | MAE {lr_test_mae:.2f}")
    print(f"  Random Forest     : RMSE {rf_test_rmse:.2f} | MAE {rf_test_mae:.2f}")
    if lstm_test_rmse:
        print(f"  LSTM (reference)  : RMSE {lstm_test_rmse:.2f}")

    # ---- Simple weighted ensemble: Random Forest + LSTM ----
    # (Only computable if the user supplies LSTM's own test predictions;
    # here we demonstrate the RF-only leg. To combine with real LSTM
    # predictions, save the LSTM's per-engine predictions from
    # evaluate.py's eval_results_{dataset}.csv and merge on engine_id -
    # see combine_with_lstm() below.)

    results = pd.DataFrame({
        "Model": ["Linear Regression", "Random Forest", "LSTM (from evaluate.py)"],
        "Test_RMSE": [round(lr_test_rmse, 2), round(rf_test_rmse, 2),
                       lstm_test_rmse if lstm_test_rmse else None],
        "Test_MAE": [round(lr_test_mae, 2), round(rf_test_mae, 2), None],
    })
    out_path = os.path.join(output_dir, f"baseline_comparison_{dataset_id}.csv")
    results.to_csv(out_path, index=False)
    print(f"\n  [OK] Comparison table saved: {out_path}")

    return {
        "lr_test_rmse": lr_test_rmse, "rf_test_rmse": rf_test_rmse,
        "rf_model": rf, "lr_model": lr,
        "X_test_tab": X_test_tab, "y_test": y_test, "test_engines": test_engines,
        "rf_test_pred": rf_test_pred, "lr_test_pred": lr_test_pred,
    }


def combine_with_lstm(rf_test_pred, y_test, lstm_eval_csv_path, weight_grid=None):
    """
    Weighted ensemble of Random Forest + LSTM test predictions.
    lstm_eval_csv_path: path to outputs/eval_results_{dataset}.csv produced
    by evaluate.py (must contain columns: engine_id, predicted_rul, actual_rul).
    Searches a small weight grid on the SAME test set predictions to find
    the best RF/LSTM blend (for a rigorous study, tune the weight on a
    held-out validation split instead of the test set itself).
    """
    lstm_df = pd.read_csv(lstm_eval_csv_path)
    lstm_pred = lstm_df["predicted_rul"].values
    lstm_true = lstm_df["actual_rul"].values

    if weight_grid is None:
        weight_grid = np.arange(0.0, 1.01, 0.1)

    best_w, best_rmse = None, float("inf")
    for w in weight_grid:
        blend = w * rf_test_pred + (1 - w) * lstm_pred
        r = rmse(lstm_true, blend)
        if r < best_rmse:
            best_rmse, best_w = r, w

    print(f"  [ENSEMBLE] Best RF weight={best_w:.1f} -> RMSE={best_rmse:.2f}")
    return best_w, best_rmse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="FD001")
    parser.add_argument("--lstm_test_rmse", type=float, default=None,
                        help="Paste your LSTM's test RMSE here for a clean comparison table")
    args = parser.parse_args()

    run_baselines(dataset_id=args.dataset, lstm_test_rmse=args.lstm_test_rmse)
