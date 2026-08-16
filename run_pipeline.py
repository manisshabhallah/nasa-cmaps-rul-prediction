"""
run_pipeline.py
----------------
Master Pipeline Runner — Run everything in one command.

This is your ONE script to rule them all.
Run this after downloading the data.

Usage:
    python run_pipeline.py --dataset FD001
    python run_pipeline.py --dataset FD001 --epochs 50 --skip_shap

Steps:
    1. Bronze  → Ingest raw data to Parquet
    2. Silver  → Clean + normalize + label
    3. Gold    → Feature engineering + sliding windows
    4. Train   → Train LSTM model
    5. Evaluate→ Compute metrics + generate plots
    6. Infer   → Predict fleet health
    7. SHAP    → Compute feature importance (optional, takes ~2 min)
"""

import argparse
import sys
import os
import time

def banner(text: str):
    """Print a section banner."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def check_data(dataset_id: str):
    """Check if raw data files exist."""
    required = [
        f"data/raw/train_{dataset_id}.txt",
        f"data/raw/test_{dataset_id}.txt",
        f"data/raw/RUL_{dataset_id}.txt",
    ]
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        print("\n❌ Missing raw data files:")
        for f in missing: print(f"   {f}")
        print("\nPlease run:  python download_data.py")
        print("Or manually download from: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps")
        sys.exit(1)
    print(f"✅ Raw data files found for {dataset_id}")


def main():
    parser = argparse.ArgumentParser(description="PHM Pipeline Runner")
    parser.add_argument("--dataset",    default="FD001", choices=["FD001", "FD002", "FD003", "FD004"])
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch_size", type=int,   default=256)
    parser.add_argument("--lr",         type=float, default=0.001)
    parser.add_argument("--patience",   type=int,   default=10)
    parser.add_argument("--skip_shap",  action="store_true",
                        help="Skip SHAP computation (saves ~2 minutes)")
    parser.add_argument("--skip_train", action="store_true",
                        help="Skip training (use existing model)")
    args = parser.parse_args()

    print(f"\n🚀 PHM Pipeline | Dataset: {args.dataset}")
    start_total = time.time()

    # ── 0. Check data ──────────────────────────────────────────────
    check_data(args.dataset)

    # ── 1. Bronze Layer ────────────────────────────────────────────
    banner("STEP 1: Bronze Layer — Raw Ingestion")
    from src.bronze_layer import ingest_to_bronze
    ingest_to_bronze(args.dataset)

    # ── 2. Silver Layer ────────────────────────────────────────────
    banner("STEP 2: Silver Layer — Cleaning & Labeling")
    from src.silver_layer import process_silver
    process_silver(args.dataset)

    # ── 3. Gold Layer ──────────────────────────────────────────────
    banner("STEP 3: Gold Layer — Feature Engineering")
    from src.gold_layer import process_gold
    process_gold(args.dataset)

    # ── 4. Train ───────────────────────────────────────────────────
    if not args.skip_train:
        banner("STEP 4: Training LSTM Model")
        from src.train import train_model
        train_model(
            dataset_id  = args.dataset,
            epochs      = args.epochs,
            batch_size  = args.batch_size,
            lr          = args.lr,
            patience    = args.patience,
        )
    else:
        print("\n[SKIP] Training skipped (--skip_train flag)")

    # ── 5. Evaluate ────────────────────────────────────────────────
    banner("STEP 5: Model Evaluation")
    from src.evaluate import evaluate
    metrics, _ = evaluate(args.dataset)

    # ── 6. Fleet Inference ─────────────────────────────────────────
    banner("STEP 6: Fleet Health Inference")
    from src.inference import predict_fleet
    fleet = predict_fleet(args.dataset)
    crit = (fleet["alert_level"] == "CRITICAL").sum()
    warn = (fleet["alert_level"] == "WARNING").sum()
    ok   = (fleet["alert_level"] == "HEALTHY").sum()
    print(f"\n  Fleet: 🔴 {crit} CRITICAL | 🟡 {warn} WARNING | 🟢 {ok} HEALTHY")

    # ── 7. SHAP ────────────────────────────────────────────────────
    if not args.skip_shap:
        banner("STEP 7: SHAP Explainability")
        from src.explain import compute_shap
        compute_shap(args.dataset)
    else:
        print("\n[SKIP] SHAP skipped (--skip_shap flag)")

    # ── Summary ────────────────────────────────────────────────────
    elapsed = time.time() - start_total
    banner("✅ PIPELINE COMPLETE")
    print(f"\n  Dataset  : {args.dataset}")
    print(f"  RMSE     : {metrics.get('RMSE', '?'):.4f}")
    print(f"  MAE      : {metrics.get('MAE', '?'):.4f}")
    print(f"  Duration : {elapsed:.1f} seconds")
    print(f"\n  Outputs saved in: outputs/")
    print(f"\n  👉 Launch dashboard:\n     streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
