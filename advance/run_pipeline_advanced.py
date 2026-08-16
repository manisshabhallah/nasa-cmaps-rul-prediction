"""
run_pipeline_advanced.py
-------------------------
ADVANCED PHD-LEVEL PIPELINE RUNNER

Additions over basic pipeline:
  - Trains on ALL FOUR datasets (FD001-FD004) jointly
  - Data augmentation (6× more training data)
  - CNN-Transformer with dual-axis attention
  - Curriculum learning + multi-task learning
  - Uncertainty quantification (MC Dropout)
  - Advanced PHM metrics (α-λ, Prognostic Horizon, ECE)

IMPORTANT:
  This uses models/ from the BASIC pipeline for the bronze/silver/gold layers.
  You must run the basic pipeline first for each dataset:
    python run_pipeline.py --dataset FD001
    python run_pipeline.py --dataset FD002
    python run_pipeline.py --dataset FD003
    python run_pipeline.py --dataset FD004

Then run this:
    python run_pipeline_advanced.py

Or quick version (FD001 only, no augmentation):
    python run_pipeline_advanced.py --datasets FD001 --no_augmentation
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def banner(text):
    print("\n" + "=" * 65)
    print(f"  {text}")
    print("=" * 65)


def check_gold_data(datasets):
    missing = []
    for d in datasets:
        if not os.path.exists(f"data/gold/X_train_{d}.npy"):
            missing.append(d)
    if missing:
        print(f"\n❌ Gold data not found for: {missing}")
        print("Run the basic pipeline first:")
        for d in missing:
            print(f"  python run_pipeline.py --dataset {d}")
        sys.exit(1)
    print(f"✅ Gold data found for: {datasets}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        default=["FD001", "FD002", "FD003", "FD004"])
    parser.add_argument("--epochs",         type=int,   default=80)
    parser.add_argument("--batch_size",     type=int,   default=512)
    parser.add_argument("--d_model",        type=int,   default=64)
    parser.add_argument("--n_heads",        type=int,   default=4)
    parser.add_argument("--n_layers",       type=int,   default=2)
    parser.add_argument("--eval_dataset",   default="FD001",
                        help="Which dataset to evaluate on (default: FD001)")
    parser.add_argument("--no_augmentation", action="store_true")
    parser.add_argument("--no_curriculum",   action="store_true")
    parser.add_argument("--no_multitask",    action="store_true")
    parser.add_argument("--mc_samples",     type=int,   default=50)
    parser.add_argument("--skip_train",     action="store_true")
    args = parser.parse_args()

    print("\n🚀 Advanced PHM Pipeline")
    print(f"   Datasets     : {args.datasets}")
    print(f"   Augmentation : {not args.no_augmentation}")
    print(f"   Curriculum   : {not args.no_curriculum}")
    print(f"   Multi-task   : {not args.no_multitask}")
    t0 = time.time()

    # ── Check inputs ──────────────────────────────────────────────
    check_gold_data(args.datasets)

    # ── Augmentation summary ──────────────────────────────────────
    if not args.no_augmentation and len(args.datasets) > 1:
        banner("STEP 1: Data Augmentation Preview")
        from src.data.augmentation import full_augmentation
        X, y = full_augmentation(args.datasets, use_noise=True, use_jitter=True)
        print(f"  Final augmented training set: {len(X):,} sequences")
        print(f"  Input shape: {X.shape}")

    # ── Train ─────────────────────────────────────────────────────
    if not args.skip_train:
        banner("STEP 2: Train CNN-Transformer with Advanced Techniques")
        from src.training.train_advanced import train_advanced
        model, best_rmse = train_advanced(
            datasets        = args.datasets,
            epochs          = args.epochs,
            batch_size      = args.batch_size,
            d_model         = args.d_model,
            n_heads         = args.n_heads,
            n_layers        = args.n_layers,
            use_augmentation= not args.no_augmentation,
            use_curriculum  = not args.no_curriculum,
            use_multitask   = not args.no_multitask,
        )
        print(f"\n  ✅ Best validation RMSE: {best_rmse:.4f}")
    else:
        print("\n[SKIP] Training skipped")

    # ── Advanced Evaluation ───────────────────────────────────────
    banner("STEP 3: Advanced Evaluation (α-λ, PH, ECE, Critical Zone RMSE)")
    from src.training.evaluate_advanced import evaluate_advanced
    metrics, _ = evaluate_advanced(
        dataset_id  = args.eval_dataset,
        mc_samples  = args.mc_samples,
    )

    # ── Summary ───────────────────────────────────────────────────
    elapsed = time.time() - t0
    banner("✅ ADVANCED PIPELINE COMPLETE")
    print(f"\n  Training on  : {args.datasets}")
    print(f"  Evaluated on : {args.eval_dataset}")
    print(f"\n  === RESULTS ===")
    for k, v in metrics.items():
        print(f"  {k:22s}: {v:.4f}")
    print(f"\n  Duration: {elapsed:.1f}s")
    print(f"\n  👉 Launch dashboard:")
    print(f"     streamlit run dashboard/app_advanced.py")


if __name__ == "__main__":
    main()
