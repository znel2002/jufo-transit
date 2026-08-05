"""Hybrid head-start: build the delay-prediction pipeline NOW on the ready-made
Deutsche Bahn dataset, while the BVG logger accrues the novel Berlin dataset.

Dataset: piebro/deutsche-bahn-data on Hugging Face (CC BY 4.0), one Parquet per
month under monthly_processed_data/. ~60 GB total, so we load ONE month at a
time via the hf:// filesystem (never the whole thing).

What this does:
  1. Load one month.
  2. Print the schema (column names differ from our guesses -> adjust COLS once,
     confirmed on first run).
  3. Define the delay ground truth.
  4. Compute the baseline: mean delay per (line/train-category x hour x weekday),
     and its MAE vs. a naive "always 0" predictor on a held-out slice.

The baseline is the number every later model must beat honestly -- exactly the
"Ergebnisse klar vom bisherigen Erkenntnisstand getrennt" the jury rewards.

Run:
    pip install "pandas>=2" pyarrow "huggingface_hub>=0.24"
    python analysis/rail_baseline.py --month 2025-01
"""
from __future__ import annotations

import argparse

import pandas as pd

DATASET = "piebro/deutsche-bahn-data"

# Best-guess column names from the dataset README. CONFIRM against the printed
# schema on first run and adjust here -- everything downstream reads from COLS.
COLS = {
    "planned": "planned_departure",   # scheduled departure time
    "actual": "departure",            # actual/estimated departure time
    "delay_min": "departure_delay",   # delay in minutes (if present, preferred)
    "line": "train_type",             # ICE / IC / RE ... (our grouping key)
    "cancelled": "cancelled",
}


def load_month(month: str) -> pd.DataFrame:
    """Load one monthly Parquet directly from the HF filesystem."""
    path = f"hf://datasets/{DATASET}/monthly_processed_data/data-{month}.parquet"
    print(f"loading {path} ...")
    return pd.read_parquet(path)


def add_delay(df: pd.DataFrame) -> pd.DataFrame:
    """Ground-truth delay in seconds, from an explicit delay col or planned/actual."""
    if COLS["delay_min"] in df.columns:
        df["delay_s"] = pd.to_numeric(df[COLS["delay_min"]], errors="coerce") * 60
    else:
        planned = pd.to_datetime(df[COLS["planned"]], errors="coerce", utc=True)
        actual = pd.to_datetime(df[COLS["actual"]], errors="coerce", utc=True)
        df["delay_s"] = (actual - planned).dt.total_seconds()
        df["_planned_dt"] = planned
    return df


def baseline(df: pd.DataFrame) -> None:
    planned = df.get("_planned_dt")
    if planned is None:
        planned = pd.to_datetime(df[COLS["planned"]], errors="coerce", utc=True)
    df = df.assign(hour=planned.dt.hour, weekday=planned.dt.weekday)
    df = df.dropna(subset=["delay_s", "hour", "weekday"])
    if COLS["cancelled"] in df.columns:
        df = df[df[COLS["cancelled"]] != True]  # noqa: E712 -- keep only ran trains

    # Time-based split: earlier half train, later half test (no leakage).
    df = df.sort_values("_planned_dt") if "_planned_dt" in df else df
    cut = int(len(df) * 0.7)
    train, test = df.iloc[:cut], df.iloc[cut:]

    key = [COLS["line"], "hour", "weekday"]
    means = train.groupby(key)["delay_s"].mean()
    pred = test.set_index(key).index.map(means).astype("float")
    pred = pd.Series(pred, index=test.index).fillna(train["delay_s"].mean())

    mae_baseline = (test["delay_s"] - pred).abs().mean()
    mae_naive = test["delay_s"].abs().mean()  # predict 0 delay
    print(f"\nrows used: {len(df):,}  (train {len(train):,} / test {len(test):,})")
    print(f"MAE naive (predict 0): {mae_naive/60:.2f} min")
    print(f"MAE baseline (line x hour x weekday mean): {mae_baseline/60:.2f} min")
    print("baseline should beat naive; every later model must beat baseline.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--month", default="2025-01", help="YYYY-MM to load")
    ap.add_argument("--schema-only", action="store_true",
                    help="just print columns + a sample, then exit")
    args = ap.parse_args()

    df = load_month(args.month)
    print(f"\nloaded {len(df):,} rows")
    print("columns:", list(df.columns))
    print(df.head(3).to_string())
    if args.schema_only:
        return

    missing = [c for c in COLS.values() if c not in df.columns]
    if missing:
        print(f"\n!! these expected columns are missing: {missing}")
        print("   Adjust COLS at the top of this file to the real names above, then re-run.")
        return

    df = add_delay(df)
    baseline(df)


if __name__ == "__main__":
    main()
