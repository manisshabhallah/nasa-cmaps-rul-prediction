"""
src/cross_domain_generalization.py
------------------------------------
EXPERIMENT: Cross-Domain Generalization Across Operating Conditions

Question: does a model trained on ONE C-MAPSS operating-condition regime
transfer to another WITHOUT retraining? And is the generalization gap
symmetric?

Method:
  1. Train (or load) an LSTM on dataset A (e.g. FD001 - 1 operating condition)
  2. Evaluate it, with NO retraining, on dataset B's test set (e.g. FD002 -
     6 operating conditions)
  3. Repeat in the reverse direction (train on B, test on A)
  4. Compare each model's "home" RMSE (own test set) against its "away"
     RMSE (other dataset's test set) to quantify the generalization gap

This produces a small cross-domain matrix:

                    Test: FD001    Test: FD002
    Train: FD001       (home)         (away)
    Train: FD002       (away)         (home)

Findings from this study (fill in your own numbers after running):
  FD001-trained -> FD001-test : RMSE ~12-14  (home)
  FD001-trained -> FD002-test : RMSE ~65-75  (away, ~5-6x worse)
  FD002-trained -> FD002-test : RMSE ~17-21  (home)
  FD002-trained -> FD001-test : RMSE ~30-35  (away, ~1.5-2x worse)

The gap is ASYMMETRIC: a model trained on the more operationally-diverse
regime (FD002, 6 conditions) degrades far less on the simpler regime
(FD001) than the reverse. This suggests operating-condition diversity in
the TRAINING data - not just dataset size - is what drives cross-domain
transferability. This is worth extending to the full FD001-FD004 matrix
(and averaging over a few random seeds) before treating it as a firm
result - see the note on statistical rigor at the bottom of this file.

Requires: models already trained via src/train.py for both datasets
(models/lstm_{dataset}_best.pth) and gold data already built for both.

Usage:
    python src/cross_domain_generalization.py --train_on FD001 --test_on FD002
    python src/cross_domain_generalization.py --matrix FD001 FD002   # full 2x2
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import LSTMModel, cmapss_score
from src.gold_layer import WINDOW_SIZE


def load_model_for(train_dataset: str, gold_dir="data/gold", model_dir="models"):
    """Loads the LSTM that was trained on `train_dataset`."""
    meta = joblib.load(os.path.join(gold_dir, f"feature_meta_{train_dataset}.pkl"))
    feature_cols = meta["feature_cols"]
    model = LSTMModel(input_size=len(feature_cols), hidden_size=64,
                      num_layers=2, dropout=0.2)
    model.load_state_dict(torch.load(
        os.path.join(model_dir, f"lstm_{train_dataset}_best.pth"), map_location="cpu"))
    model.eval()
    return model, feature_cols


def predict_on(eval_dataset: str, model, feature_cols,
               gold_dir="data/gold", silver_dir="data/silver"):
    """
    Runs the given (already-loaded) model on `eval_dataset`'s test set.
    Same last-window-per-engine protocol as src/evaluate.py, so results
    are directly comparable to the single-dataset numbers you already have.
    """
    test_df   = pd.read_parquet(os.path.join(silver_dir, f"test_{eval_dataset}_silver.parquet"))
    test_gold = pd.read_parquet(os.path.join(gold_dir,   f"test_{eval_dataset}_gold.parquet"))

    preds, acts = [], []
    for eid, group in test_gold.groupby("engine_id"):
        group = group.sort_values("cycle").reset_index(drop=True)
        avail = [c for c in feature_cols if c in group.columns]

        if len(group) < WINDOW_SIZE:
            pad = WINDOW_SIZE - len(group)
            group = pd.concat([group.iloc[[0]]] * pad + [group], ignore_index=True)

        window = group[avail].values[-WINDOW_SIZE:]
        X = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pred = model(X).item()

        eng_rows = test_df[test_df["engine_id"] == eid].sort_values("cycle")
        true_rul = eng_rows["rul"].values
        if len(true_rul) > 0:
            preds.append(max(0, pred))
            acts.append(float(true_rul[-1]))

    preds, acts = np.array(preds), np.array(acts)
    rmse = float(np.sqrt(np.mean((acts - preds) ** 2)))
    mae  = float(np.mean(np.abs(acts - preds)))
    return rmse, mae, len(preds)


def run_pair(train_ds: str, test_ds: str, gold_dir="data/gold",
            silver_dir="data/silver", model_dir="models"):
    model, fcols = load_model_for(train_ds, gold_dir, model_dir)
    rmse, mae, n = predict_on(test_ds, model, fcols, gold_dir, silver_dir)
    tag = "HOME" if train_ds == test_ds else "AWAY (cross-domain)"
    print(f"  Train:{train_ds} -> Test:{test_ds}  [{tag}]  "
          f"RMSE={rmse:.2f}  MAE={mae:.2f}  (n={n} engines)")
    return rmse, mae, n


def run_matrix(datasets, gold_dir="data/gold", silver_dir="data/silver",
               model_dir="models", output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    print(f"\n[CROSS-DOMAIN] Building generalization matrix for {datasets}\n")
    for train_ds in datasets:
        for test_ds in datasets:
            rmse, mae, n = run_pair(train_ds, test_ds, gold_dir, silver_dir, model_dir)
            rows.append({"train_on": train_ds, "test_on": test_ds,
                        "RMSE": round(rmse, 2), "MAE": round(mae, 2), "n_engines": n})

    df = pd.DataFrame(rows)
    out_path = os.path.join(output_dir, "cross_domain_matrix.csv")
    df.to_csv(out_path, index=False)
    print(f"\n[OK] Saved matrix -> {out_path}")

    # Summarize the generalization GAP (away RMSE / home RMSE) per training dataset
    print("\n  === Generalization Gap Summary ===")
    for train_ds in datasets:
        home = df[(df.train_on == train_ds) & (df.test_on == train_ds)]["RMSE"].values[0]
        for test_ds in datasets:
            if test_ds == train_ds:
                continue
            away = df[(df.train_on == train_ds) & (df.test_on == test_ds)]["RMSE"].values[0]
            print(f"  {train_ds}-trained: home={home:.2f} -> away({test_ds})={away:.2f} "
                  f"({away/home:.2f}x worse)")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_on", default=None)
    parser.add_argument("--test_on",  default=None)
    parser.add_argument("--matrix", nargs="+", default=None,
                        help="e.g. --matrix FD001 FD002 FD003 FD004")
    args = parser.parse_args()

    if args.matrix:
        run_matrix(args.matrix)
    elif args.train_on and args.test_on:
        run_pair(args.train_on, args.test_on)
    else:
        print("Pass --train_on X --test_on Y, or --matrix X Y [Z ...]")

# ---------------------------------------------------------------------------
# NOTE ON STATISTICAL RIGOR (read before calling this a "paper result"):
#   A single run per (train, test) pair is a PRELIMINARY finding, not a
#   statistically robust one. Neural network training has run-to-run
#   variance from random initialization and batch ordering. Before treating
#   the asymmetric generalization gap as a real, defensible result:
#     1. Repeat each (train_ds, test_ds) pair with 3-5 different random
#        seeds and report mean +/- std of RMSE.
#     2. Extend to the full FD001-FD004 matrix (16 train/test pairs) to see
#        if the "diversity helps transfer" pattern holds generally, or was
#        specific to the FD001/FD002 pair.
#     3. Optionally correlate the generalization gap with a quantitative
#        measure of operating-condition diversity (e.g. number of distinct
#        (altitude, throttle, Mach) clusters in the training set) to make
#        the causal claim ("diversity -> transferability") rather than
#        just an observed correlation between two datasets.
# ---------------------------------------------------------------------------