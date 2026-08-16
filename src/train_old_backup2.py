"""
src/train.py
------------
Training Pipeline for LSTM RUL Predictor

Steps:
1. Load gold data (X_train, y_train, engine_ids_train)
2. Split into train / validation by ENGINE (not by window!) - 80/20
3. Build LSTM model
4. Train with early stopping
5. Save best model checkpoint
6. Plot loss curves

Early Stopping:
  We stop training if validation loss doesn't improve for `patience` epochs.
  This prevents overfitting on the training engines.

IMPORTANT FIX (vs. original version):
  The original code used torch's `random_split`, which shuffles individual
  sliding-window SAMPLES into train/val at random. Because windows from the
  SAME engine overlap heavily (window i and window i+1 share 29/30 cycles),
  this leaks near-identical data between train and val. That made val RMSE
  look great (~12) while real test RMSE (on fully unseen engines, last
  window only) was much worse (~28).

  Fix: split by ENGINE ID using GroupShuffleSplit. All windows belonging to
  one engine go entirely into train OR entirely into val - never split
  across both. This makes validation behave like the real test set (unseen
  engines), so val RMSE during training now reflects true generalization.

Usage:
    python src/train.py --dataset FD001 --epochs 100 --patience 15
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import GroupShuffleSplit
import matplotlib
matplotlib.use("Agg")   # headless backend for saving plots
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import LSTMModel, cmapss_score


def load_gold_data(dataset_id: str, gold_dir: str = "data/gold"):
    X = np.load(os.path.join(gold_dir, f"X_train_{dataset_id}.npy"))
    y = np.load(os.path.join(gold_dir, f"y_train_{dataset_id}.npy"))
    eids_path = os.path.join(gold_dir, f"engine_ids_train_{dataset_id}.npy")
    engine_ids = np.load(eids_path)
    print(f"[TRAIN] Loaded X:{X.shape}  y:{y.shape}  engine_ids:{engine_ids.shape}")
    return X, y, engine_ids


def train_model(dataset_id: str = "FD001",
                epochs: int     = 50,
                batch_size: int = 256,
                lr: float       = 0.001,
                val_split: float = 0.2,
                patience: int   = 10,
                gold_dir: str   = "data/gold",
                model_dir: str  = "models",
                output_dir: str = "outputs",
                seed: int       = 42):

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # -------- Data --------
    X, y, engine_ids = load_gold_data(dataset_id, gold_dir)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)

    # -------- ENGINE-WISE 80/20 split (FIXED) --------
    # GroupShuffleSplit ensures every window from a given engine_id lands
    # entirely in either train or val, never both. This mirrors how the
    # real test set works (fully unseen engines), so val RMSE becomes a
    # trustworthy estimate of test-time performance.
    gss = GroupShuffleSplit(n_splits=1, test_size=val_split, random_state=seed)
    train_idx, val_idx = next(gss.split(X, y, groups=engine_ids))

    train_ds = Subset(dataset, train_idx)
    val_ds   = Subset(dataset, val_idx)

    train_size = len(train_idx)
    val_size   = len(val_idx)

    n_train_engines = len(set(engine_ids[train_idx].tolist()))
    n_val_engines   = len(set(engine_ids[val_idx].tolist()))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    print(f"  Train samples: {train_size}  |  Val samples: {val_size}")
    print(f"  Train engines: {n_train_engines}  |  Val engines: {n_val_engines}  "
          f"(no engine appears in both)")

    # -------- Model --------
    input_size = X.shape[2]   # number of features
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")

    model = LSTMModel(input_size=input_size, hidden_size=64,
                      num_layers=2, dropout=0.2).to(device)

    # Adam optimizer — standard choice for LSTMs
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Learning rate scheduler — reduce LR when val loss stagnates
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    # MSE loss for regression
    criterion = nn.MSELoss()

    # -------- Training Loop --------
    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0

    print(f"\n[TRAIN] Starting training for {epochs} epochs...")
    print(f"  Early stopping patience: {patience}")

    for epoch in range(1, epochs + 1):

        # --- Train ---
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()

            # Gradient clipping — prevents exploding gradients in LSTM
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            running_loss += loss.item() * len(X_batch)

        train_loss = running_loss / train_size

        # --- Validate ---
        model.eval()
        val_preds_all, val_true_all = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                preds = model(X_batch).cpu().numpy()
                val_preds_all.extend(preds)
                val_true_all.extend(y_batch.numpy())

        val_preds_all = np.array(val_preds_all)
        val_true_all  = np.array(val_true_all)
        val_loss = np.mean((val_preds_all - val_true_all) ** 2)
        val_rmse = np.sqrt(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"Train MSE: {train_loss:.2f} | "
                  f"Val RMSE: {val_rmse:.2f}")

        # --- Early Stopping ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(),
                       os.path.join(model_dir, f"lstm_{dataset_id}_best.pth"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  [EARLY STOP] No improvement for {patience} epochs. Stopping at epoch {epoch}.")
                break

    print(f"\n  Best model from epoch {best_epoch} | Val RMSE: {np.sqrt(best_val_loss):.4f}")
    print(f"  (This Val RMSE is now a realistic estimate of test RMSE, "
          f"since val engines were never seen in training.)")

    # -------- Plot Loss Curves --------
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label="Train MSE", color="steelblue")
    plt.plot(val_losses,   label="Val MSE",   color="coral")
    plt.axvline(x=best_epoch - 1, color="green", linestyle="--",
                label=f"Best (epoch {best_epoch})")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(f"Training Loss Curve — {dataset_id}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"loss_curve_{dataset_id}.png"), dpi=120)
    plt.close()
    print(f"  [OK] Loss curve saved to outputs/loss_curve_{dataset_id}.png")

    return model, train_losses, val_losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM for RUL prediction")
    parser.add_argument("--dataset",    default="FD001", help="Dataset ID (FD001, FD002, ...)")
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch_size", type=int,   default=256)
    parser.add_argument("--lr",         type=float, default=0.001)
    parser.add_argument("--patience",   type=int,   default=10)
    args = parser.parse_args()

    train_model(
        dataset_id  = args.dataset,
        epochs      = args.epochs,
        batch_size  = args.batch_size,
        lr          = args.lr,
        patience    = args.patience,
    )