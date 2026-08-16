"""
src/bronze_layer.py
--------------------
BRONZE LAYER — Raw Data Ingestion

In a real production setup, this would be Apache Kafka consuming
live sensor streams from turbofan gateways.

For this M.Tech project, we simulate the streaming ingestion by:
1. Reading the raw .txt files row-by-row (simulating event streaming)
2. Adding ingestion timestamps + metadata
3. Saving as Parquet (simulating Delta Lake Bronze tables)

Think of this as: "data landed exactly as received, nothing changed"

Column names for C-MAPSS:
  - engine_id   : which engine unit
  - cycle        : current operational cycle
  - setting_1/2/3 : operational settings (altitude, mach, TRA)
  - s1..s21      : 21 sensor readings
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

# ---- Column names as defined in NASA C-MAPSS documentation ----
COLS = (
    ["engine_id", "cycle", "setting_1", "setting_2", "setting_3"]
    + [f"s{i}" for i in range(1, 22)]          # s1 to s21
)

# These 2 sensors have ZERO variance across all datasets (confirmed in literature)
# They carry no useful information, so we drop them.
SENSORS_TO_DROP = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]


def load_raw_file(filepath: str) -> pd.DataFrame:
    """
    Reads a raw C-MAPSS .txt file.
    Data is space-separated, no header row.
    """
    df = pd.read_csv(
        filepath,
        sep=r"\s+",       # one or more spaces
        header=None,
        names=COLS
    )
    return df


def ingest_to_bronze(dataset_id: str = "FD001", raw_dir: str = "data/raw",
                     bronze_dir: str = "data/bronze") -> dict:
    """
    Simulates Bronze Layer ingestion:
    - Reads raw file
    - Tags each row with ingestion_timestamp + source metadata
    - Saves as Parquet (immutable, append-only in real Delta Lake)

    Returns: dict with train/test DataFrames
    """
    os.makedirs(bronze_dir, exist_ok=True)

    train_path = os.path.join(raw_dir, f"train_{dataset_id}.txt")
    test_path  = os.path.join(raw_dir, f"test_{dataset_id}.txt")
    rul_path   = os.path.join(raw_dir, f"RUL_{dataset_id}.txt")

    print(f"\n[BRONZE] Ingesting dataset: {dataset_id}")

    # --- Load train ---
    train_df = load_raw_file(train_path)
    train_df["source_file"]          = f"train_{dataset_id}.txt"
    train_df["ingestion_timestamp"]  = datetime.utcnow().isoformat()
    train_df["dataset_id"]           = dataset_id
    train_df["split"]                = "train"

    # --- Load test ---
    test_df = load_raw_file(test_path)
    test_df["source_file"]           = f"test_{dataset_id}.txt"
    test_df["ingestion_timestamp"]   = datetime.utcnow().isoformat()
    test_df["dataset_id"]            = dataset_id
    test_df["split"]                 = "test"

    # --- Load RUL ground truth for test set ---
    rul_df = pd.read_csv(rul_path, header=None, names=["rul_at_end"])

    print(f"  Train rows : {len(train_df)}")
    print(f"  Test rows  : {len(test_df)}")
    print(f"  Engines (train): {train_df['engine_id'].nunique()}")
    print(f"  Engines (test) : {test_df['engine_id'].nunique()}")

    # Save to bronze Parquet (immutable, timestamped)
    train_df.to_parquet(os.path.join(bronze_dir, f"train_{dataset_id}_bronze.parquet"), index=False)
    test_df.to_parquet(os.path.join(bronze_dir, f"test_{dataset_id}_bronze.parquet"), index=False)
    rul_df.to_parquet(os.path.join(bronze_dir, f"RUL_{dataset_id}_bronze.parquet"), index=False)

    print(f"  [OK] Saved bronze Parquet files to {bronze_dir}/")

    return {"train": train_df, "test": test_df, "rul": rul_df}


if __name__ == "__main__":
    result = ingest_to_bronze("FD001")
    print("\nSample train data (first 3 rows):")
    print(result["train"][["engine_id", "cycle", "s2", "s3", "s4"]].head(3))
