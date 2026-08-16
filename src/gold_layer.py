"""
src/gold_layer.py
------------------
GOLD LAYER — Feature-Engineered Data Ready for Deep Learning

Takes silver (clean, normalized) data and creates:
1. Sliding window sequences of length WINDOW_SIZE
2. Lag delta features (rate of change between cycles)
3. Rolling mean features (smoothed trends)

WHY sliding windows for LSTM?
   LSTM needs to "see" a sequence to learn temporal patterns.
   A single cycle row has no context.
   Window of 30 cycles = model sees last 30 snapshots before predicting.

   Input shape per sample: (30 cycles, 14 sensors)
   Output: single RUL value for that window

Example:
   Engine 1, cycles 1-30  → RUL label at cycle 30
   Engine 1, cycles 2-31  → RUL label at cycle 31
   Engine 1, cycles 3-32  → RUL label at cycle 32
   ...
"""

import os
import numpy as np
import pandas as pd
import joblib

WINDOW_SIZE = 30    # Number of past cycles to look at


def add_lag_features(df: pd.DataFrame, sensor_cols: list) -> pd.DataFrame:
    """
    Adds delta (difference from previous cycle) for each sensor.
    This captures RATE OF CHANGE — important for detecting sudden degradation.

    e.g., s3_delta = s3(t) - s3(t-1)
    """
    df = df.copy().sort_values(["engine_id", "cycle"])
    for col in sensor_cols:
        df[f"{col}_delta"] = df.groupby("engine_id")[col].diff().fillna(0)
    print(f"  [GOLD] Added {len(sensor_cols)} lag-delta features")
    return df


def add_rolling_features(df: pd.DataFrame, sensor_cols: list,
                          window: int = 5) -> pd.DataFrame:
    """
    Rolling mean over last `window` cycles per sensor per engine.
    Smooths out noise to reveal underlying degradation trend.
    """
    df = df.copy().sort_values(["engine_id", "cycle"])
    for col in sensor_cols:
        df[f"{col}_roll{window}"] = (
            df.groupby("engine_id")[col]
              .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )
    print(f"  [GOLD] Added {len(sensor_cols)} rolling-mean features (window={window})")
    return df


def create_sequences(df: pd.DataFrame, feature_cols: list,
                     window_size: int = WINDOW_SIZE):
    """
    Creates overlapping sliding window sequences per engine.

    Returns:
        X: np.array shape (N_samples, window_size, n_features)
        y: np.array shape (N_samples,)   — RUL at last step of window
        engine_ids: for tracking which engine each sample came from
    """
    X_list, y_list, eid_list = [], [], []

    for engine_id, group in df.groupby("engine_id"):
        group = group.sort_values("cycle")
        data = group[feature_cols].values
        labels = group["rul"].values
        n_cycles = len(group)

        # Slide window across cycles
        for i in range(n_cycles - window_size + 1):
            window_data = data[i : i + window_size]         # shape: (30, n_features)
            rul_label   = labels[i + window_size - 1]       # RUL at the LAST cycle of window
            X_list.append(window_data)
            y_list.append(rul_label)
            eid_list.append(engine_id)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    eids = np.array(eid_list)

    print(f"  [GOLD] Created {len(X)} sequences | X shape: {X.shape} | y shape: {y.shape}")
    return X, y, eids


def process_gold(dataset_id: str = "FD001",
                 silver_dir: str = "data/silver",
                 gold_dir: str = "data/gold") -> dict:
    """
    Full gold pipeline:
    1. Load from silver
    2. Add lag + rolling features
    3. Create sliding window sequences
    4. Save X, y as .npy arrays
    """
    os.makedirs(gold_dir, exist_ok=True)

    print(f"\n[GOLD] Processing dataset: {dataset_id}")

    train_df = pd.read_parquet(os.path.join(silver_dir, f"train_{dataset_id}_silver.parquet"))
    test_df  = pd.read_parquet(os.path.join(silver_dir, f"test_{dataset_id}_silver.parquet"))

    # Identify sensor columns (normalized ones, no metadata)
    sensor_cols = [c for c in train_df.columns
                   if c.startswith("s") and c[1:].isdigit()]

    # Add engineered features
    train_df = add_lag_features(train_df, sensor_cols)
    train_df = add_rolling_features(train_df, sensor_cols)

    # All feature columns for model input
    feature_cols = (
        sensor_cols
        + [f"{c}_delta" for c in sensor_cols]
        + [f"{c}_roll5"  for c in sensor_cols]
    )

    # Remove NaN rows (rolling creates some at the start)
    train_df = train_df.dropna(subset=feature_cols)

    # Create sequences for train
    X_train, y_train, eids_train = create_sequences(train_df, feature_cols, WINDOW_SIZE)

    # For test: only last window per engine (we predict at the last known cycle)
    # We need at least WINDOW_SIZE cycles per engine
    test_df = add_lag_features(test_df, [c for c in sensor_cols if c in test_df.columns])
    test_df = add_rolling_features(test_df, [c for c in sensor_cols if c in test_df.columns])

    # Save feature column names for later use in inference
    feature_meta = {
        "feature_cols": feature_cols,
        "sensor_cols": sensor_cols,
        "window_size": WINDOW_SIZE,
        "dataset_id": dataset_id
    }
    joblib.dump(feature_meta, os.path.join(gold_dir, f"feature_meta_{dataset_id}.pkl"))

    # Save arrays
    np.save(os.path.join(gold_dir, f"X_train_{dataset_id}.npy"), X_train)
    np.save(os.path.join(gold_dir, f"y_train_{dataset_id}.npy"), y_train)
    np.save(os.path.join(gold_dir, f"engine_ids_train_{dataset_id}.npy"), eids_train)

    # Save test silver for inference use
    test_df.to_parquet(os.path.join(gold_dir, f"test_{dataset_id}_gold.parquet"), index=False)

    print(f"  [OK] Gold data saved to {gold_dir}/")
    print(f"  Feature count: {len(feature_cols)}")

    return {
        "X_train": X_train,
        "y_train": y_train,
        "feature_cols": feature_cols,
        "test_df": test_df,
        "sensor_cols": sensor_cols
    }


if __name__ == "__main__":
    result = process_gold("FD001")
    print(f"\nX_train shape : {result['X_train'].shape}")
    print(f"y_train range : {result['y_train'].min():.1f} – {result['y_train'].max():.1f}")
    print(f"Features used : {len(result['feature_cols'])}")
