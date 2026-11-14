"""Adapter for the public Google Cluster five-minute aggregate workload CSV.

The source file has no timestamps or request counts. We create a relative
five-minute index (not an observed wall-clock timestamp) and do not invent
request counts. CPU/memory fractions are converted to percentages.
"""
from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "avg_cpu", "avg_mem", "avg_assigned_mem",
    "avg_cycles_per_instruction",
]

def load_google_cluster_dataset(file_path="data/google_cluster_usage.csv"):
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Google Cluster CSV not found: {path}")

    raw = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in raw.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if raw.empty:
        raise ValueError("Dataset is empty")

    raw = raw[REQUIRED_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if raw.isna().any().any():
        raise ValueError("Dataset contains missing or non-numeric values")
    if not np.isfinite(raw.to_numpy()).all():
        raise ValueError("Dataset contains infinite values")

    # The supplied trace reports CPU and memory as ratios in [0, 1].
    for column in ["avg_cpu", "avg_mem", "avg_assigned_mem"]:
        if raw[column].min() < 0 or raw[column].max() > 1:
            raise ValueError(
                f"Expected {column} to be a fraction between 0 and 1; "
                "inspect source units before converting."
            )

    # Keep canonical names expected by the simulator. The third ML feature
    # is assigned-memory percentage, not a fabricated request count.
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=len(raw), freq="5min"),
        "cpu_usage": raw["avg_cpu"].to_numpy() * 100.0,
        "ram_usage": raw["avg_mem"].to_numpy() * 100.0,
        "assigned_mem_usage": raw["avg_assigned_mem"].to_numpy() * 100.0,
        "avg_cycles_per_instruction": raw["avg_cycles_per_instruction"].to_numpy(),
    })
    print(f"Loaded {len(df)} Google Cluster observations (5-minute intervals)")
    print("Note: timestamps are relative indices; the source CSV has no timestamp column.")
    return df
