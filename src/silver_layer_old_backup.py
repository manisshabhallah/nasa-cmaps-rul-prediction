"""
src/silver_layer.py
--------------------
SILVER LAYER — Validated & Cleaned Data

Takes bronze (raw) data and:
1. Removes zero-variance sensors (no information content)
2. Applies Min-Max normalization (0 to 1 scale)
3. Removes outliers using IQR method
4. Assigns piecewise-linear RUL labels
5. Saves clean Parquet to silver/

WHY piecewise RUL?
   Engines don't actually degrade from cycle 1.
   They run "healthy" for a while, THEN start degrading.
   So we cap RUL at 125 — meaning:
     - If an engine has 200 cycles left → we say "RUL = 125" (healthy phase)
     - If an engine has 50 cycles left  → we say "RUL = 50"  (degrading)
   This is the standard approach in C-MAPSS literature.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

# Sensors with near-zero variance — confirmed via std check in literature
ZERO_VAR_SENSORS = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]

# Standard RUL cap used across literature (NASA Tech Memo 2015)
RUL_CAP = 125


def drop_zero_variance_sensors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops sensors that have essentially no variance.
    A sensor that never changes tells you nothing about degradation.
    """
    cols_to_drop = [c for c in ZERO_VAR_SENSORS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    print(f"  [SILVER] Dropped zero-variance sensors: {cols_to_drop}")
    return df


def compute_rul_labels(train_df: pd.DataFrame, rul_cap: int = RUL_CAP) -> pd.DataFrame:
    """
    For training data:
    - Find the max cycle per engine (= end of life)
    - RUL = max_cycle - current_cycle
    - Apply piecewise cap: RUL = min(RUL, rul_cap)
    """
    # Max cycle per engine = total life of that engine
    max_cycles = train_df.groupby("engine_id")["cycle"].max().reset_index()
    max_cycles.columns = ["engine_id", "max_cycle"]

    df = train_df.merge(max_cycles, on="engine_id")
    df["rul"] = df["max_cycle"] - df["cycle"]

    # Piecewise cap — healthy phase treated as constant RUL = 125
    df["rul"] = df["rul"].clip(upper=rul_cap)

    df = df.drop(columns=["max_cycle"])
    return df


def normalize_sensors(train_df: pd.DataFrame, test_df: pd.DataFrame,
                      scaler_path: str = "models/scaler.pkl"):
    """
    Fits MinMaxScaler on training data, applies to both train and test.

    IMPORTANT: Fit ONLY on train, transform both.
    Fitting on test would be data leakage.

    Returns normalized copies + saves scaler for inference.
    """
    os.makedirs("models", exist_ok=True)

    # Only normalize actual sensor columns
    sensor_cols = [c for c in train_df.columns
               if c.startswith("s") and c[1:].isdigit()]

    scaler = MinMaxScaler()
    train_df = train_df.copy()
    test_df  = test_df.copy()

    train_df[sensor_cols] = scaler.fit_transform(train_df[sensor_cols])
    test_df[sensor_cols]  = scaler.transform(test_df[sensor_cols])

    joblib.dump(scaler, scaler_path)
    print(f"  [SILVER] Scaler fitted on {len(sensor_cols)} sensors, saved to {scaler_path}")

    return train_df, test_df, scaler, sensor_cols


def remove_outliers_iqr(df: pd.DataFrame, sensor_cols: list,
                        iqr_factor: float = 3.0) -> pd.DataFrame:
    """
    IQR-based outlier clamping (not removal, clamping).
    We CLAMP instead of DROP because in time-series we can't have gaps.
    """
    df = df.copy()
    for col in sensor_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - iqr_factor * IQR
        upper = Q3 + iqr_factor * IQR
        df[col] = df[col].clip(lower=lower, upper=upper)
    print(f"  [SILVER] IQR outlier clamping applied (factor={iqr_factor})")
    return df


def process_silver(dataset_id: str = "FD001",
                   bronze_dir: str = "data/bronze",
                   silver_dir: str = "data/silver") -> dict:
    """
    Full silver pipeline:
    1. Load from bronze
    2. Drop zero-variance sensors
    3. Compute RUL labels for train
    4. Normalize
    5. Clamp outliers
    6. Save to silver
    """
    os.makedirs(silver_dir, exist_ok=True)

    print(f"\n[SILVER] Processing dataset: {dataset_id}")

    # Load bronze
    train_df = pd.read_parquet(os.path.join(bronze_dir, f"train_{dataset_id}_bronze.parquet"))
    test_df  = pd.read_parquet(os.path.join(bronze_dir, f"test_{dataset_id}_bronze.parquet"))
    rul_df   = pd.read_parquet(os.path.join(bronze_dir, f"RUL_{dataset_id}_bronze.parquet"))

    # Drop useless sensors
    train_df = drop_zero_variance_sensors(train_df)
    test_df  = drop_zero_variance_sensors(test_df)

    # Add RUL labels to train
    train_df = compute_rul_labels(train_df, rul_cap=RUL_CAP)

    # Add RUL to test: last cycle of each engine + ground truth RUL
    # (test files contain data UP TO the prediction point, not until failure)
    test_last = test_df.groupby("engine_id").last().reset_index()
    rul_df["engine_id"] = range(1, len(rul_df) + 1)
    test_last = test_last.merge(rul_df, on="engine_id")
    test_last["rul"] = test_last["rul_at_end"].clip(upper=RUL_CAP)

    # Normalize (fit on full train data)
    sensor_cols = [c for c in train_df.columns
                   if c.startswith("s") and c[1:].isdigit()]

    train_df, test_last, scaler, sensor_cols = normalize_sensors(
        train_df, test_last,
        scaler_path=f"models/scaler_{dataset_id}.pkl"
    )

    # Clamp outliers AFTER normalization
    train_df = remove_outliers_iqr(train_df, sensor_cols)

    print(f"  Train rows after processing: {len(train_df)}")
    print(f"  RUL range in train: {train_df['rul'].min():.0f} – {train_df['rul'].max():.0f}")

    # Save
    train_df.to_parquet(os.path.join(silver_dir, f"train_{dataset_id}_silver.parquet"), index=False)
    test_last.to_parquet(os.path.join(silver_dir, f"test_{dataset_id}_silver.parquet"), index=False)

    print(f"  [OK] Silver Parquet saved to {silver_dir}/")

    return {
        "train": train_df,
        "test": test_last,
        "sensor_cols": sensor_cols,
        "rul_cap": RUL_CAP
    }


if __name__ == "__main__":
    result = process_silver("FD001")
    print("\nSample train (engine 1, last 5 cycles):")
    sample = result["train"][result["train"]["engine_id"] == 1].tail(5)
    print(sample[["engine_id", "cycle", "rul", "s2", "s3", "s4"]].to_string())
