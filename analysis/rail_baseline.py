"""Hybrid head-start: build the delay-prediction pipeline NOW on the ready-made
Deutsche Bahn dataset, while the BVG logger accrues the novel Berlin dataset.

Dataset: piebro/deutsche-bahn-data on Hugging Face (CC BY 4.0), one Parquet per
month under monthly_processed_data/. ~61 GB / 206M rows total, so we load ONE
month at a time via the hf:// filesystem, and only the columns we actually use
(pyarrow prunes them at the file level, so we never transfer the full month).

What this does:
  1. Load one month.
  2. Check what `delay_in_min` actually measures (the dataset does not say).
  3. Define the delay ground truth.
  4. Compute the baseline: mean delay per (train_type x hour x weekday), and its
     MAE vs. a naive "always 0" predictor on a held-out, time-ordered slice.

The baseline is the number every later model must beat honestly -- exactly the
"Ergebnisse klar vom bisherigen Erkenntnisstand getrennt" the jury rewards.

Run:
    pip install "pandas>=2" pyarrow "huggingface_hub>=0.24"
    python analysis/rail_baseline.py --schema-only --month 2025-01
    python analysis/rail_baseline.py --month 2025-01
    python analysis/rail_baseline.py --month 2025-01 --berlin-only
"""
from __future__ import annotations

import argparse

import pandas as pd

DATASET = "piebro/deutsche-bahn-data"

# Confirmed 2026-08-10 against the live schema served by
# datasets-server.huggingface.co/info (NOT the dataset README, which is stale --
# it documents a `train_name` column that the actual Parquet does not have).
COLS = {
    "planned": "departure_planned_time",
    "actual": "departure_change_time",
    "delay_min": "delay_in_min",
    "line": "train_type",
    "cancelled": "is_canceled",
    "station": "station_name",
}

# Only these are read from the Parquet. `arrival_*` exists solely to settle the
# delay-semantics question in check_delay_semantics().
NEEDED = list(COLS.values()) + ["arrival_planned_time", "arrival_change_time"]


def load_month(month: str, *, all_columns: bool = False) -> pd.DataFrame:
    """Load one monthly Parquet directly from the HF filesystem."""
    path = f"hf://datasets/{DATASET}/monthly_processed_data/data-{month}.parquet"
    cols = None if all_columns else NEEDED
    print(f"loading {path}")
    print(f"  columns: {'ALL' if all_columns else len(cols)} of 17")
    return pd.read_parquet(path, columns=cols)


def _elapsed_min(planned: pd.Series, changed: pd.Series) -> pd.Series:
    """Minutes late, treating a missing change-time as 'ran exactly to plan'.

    The dataset leaves `*_change_time` null when nothing changed, i.e. precisely
    for the punctual rows. Subtracting without filling would turn every on-time
    train into NaN and drop it -- which would bias the measured delay upwards and
    make the baseline look far better than it is.
    """
    return (changed.fillna(planned) - planned).dt.total_seconds() / 60


def check_delay_semantics(df: pd.DataFrame) -> None:
    """`delay_in_min` is undocumented: arrival delay or departure delay?

    Each row is one station stop carrying both arrival and departure times, and
    the dataset README never says which the delay refers to. Rather than assume,
    measure it: reconstruct both candidates and see which one `delay_in_min`
    matches. The answer belongs in the write-up as a data-provenance check.
    """
    dep = _elapsed_min(df["departure_planned_time"], df["departure_change_time"])
    arr = _elapsed_min(df["arrival_planned_time"], df["arrival_change_time"])
    reported = pd.to_numeric(df[COLS["delay_min"]], errors="coerce")

    print("\nwhat does `delay_in_min` measure?")
    for label, cand in (("departure", dep), ("arrival", arr)):
        mask = cand.notna() & reported.notna()
        if not mask.any():
            print(f"  {label:<10} no overlapping rows")
            continue
        exact = (cand[mask].round() == reported[mask]).mean()
        corr = cand[mask].corr(reported[mask])
        print(f"  {label:<10} n={mask.sum():>10,}  exact match {exact:6.1%}  corr {corr:5.3f}")
    print("  -> the higher pair is what the column reports; record it in docs/entscheidungen.md")


def add_delay(df: pd.DataFrame) -> pd.DataFrame:
    """Ground-truth delay in seconds, from the explicit delay column."""
    df = df.copy()
    df["delay_s"] = pd.to_numeric(df[COLS["delay_min"]], errors="coerce") * 60
    df["_planned_dt"] = pd.to_datetime(df[COLS["planned"]], errors="coerce")
    return df


def baseline(df: pd.DataFrame) -> None:
    planned = df["_planned_dt"]
    df = df.assign(hour=planned.dt.hour, weekday=planned.dt.weekday)
    df = df.dropna(subset=["delay_s", "hour", "weekday", "_planned_dt"])
    if COLS["cancelled"] in df.columns:
        df = df[~df[COLS["cancelled"]].astype(bool)]  # keep only trains that ran

    # Time-based split: earlier 70% train, later 30% test. Never random -- the same
    # ride appears at many stations, so a random split would leak it across the cut.
    df = df.sort_values("_planned_dt")
    cut = int(len(df) * 0.7)
    train, test = df.iloc[:cut], df.iloc[cut:]

    key = [COLS["line"], "hour", "weekday"]
    y = test["delay_s"]

    def group_pred(stat: str) -> pd.Series:
        """Per-(train_type, hour, weekday) statistic, fitted on train only."""
        table = train.groupby(key)["delay_s"].agg(stat)
        fallback = getattr(train["delay_s"], stat)()
        return pd.Series(
            test.set_index(key).index.map(table).astype("float"), index=test.index
        ).fillna(fallback)  # group unseen in train -> global train statistic

    candidates = {
        "predict 0 (naive)": pd.Series(0.0, index=test.index),
        "global median": pd.Series(train["delay_s"].median(), index=test.index),
        "global mean": pd.Series(train["delay_s"].mean(), index=test.index),
        "group median (type x hour x wd)": group_pred("median"),
        "group mean   (type x hour x wd)": group_pred("mean"),
    }

    print(f"\nrows used: {len(df):,}  (train {len(train):,} / test {len(test):,})")
    print(f"  test window: {test['_planned_dt'].min()} -> {test['_planned_dt'].max()}")
    print(f"  mean delay {df['delay_s'].mean()/60:5.2f} min | "
          f"median {df['delay_s'].median()/60:5.2f} min | "
          f"share > 5 min {(df['delay_s'] > 300).mean():.1%}")

    # Report MAE *and* RMSE, because they disagree about what a good predictor is:
    # MAE is minimised by the conditional median, RMSE by the conditional mean.
    # On a right-skewed delay distribution a mean-baseline can lose to "predict 0"
    # on MAE while winning on RMSE -- reporting only one number hides that.
    print(f"\n{'predictor':<34} {'MAE (min)':>10} {'RMSE (min)':>11}")
    print("-" * 57)
    for label, pred in candidates.items():
        mae = (y - pred).abs().mean() / 60
        rmse = ((y - pred) ** 2).mean() ** 0.5 / 60
        print(f"{label:<34} {mae:>10.3f} {rmse:>11.3f}")

    best_mae = min(candidates, key=lambda k: (y - candidates[k]).abs().mean())
    best_rmse = min(candidates, key=lambda k: ((y - candidates[k]) ** 2).mean())
    print(f"\nbest MAE:  {best_mae}")
    print(f"best RMSE: {best_rmse}")
    print("Every later model must beat the best baseline on the SAME metric.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--month", default="2025-01", help="YYYY-MM to load")
    ap.add_argument("--schema-only", action="store_true",
                    help="load all columns, print schema + a sample, then exit")
    ap.add_argument("--berlin-only", action="store_true",
                    help="restrict to Berlin stations (comparable to the BVG data)")
    args = ap.parse_args()

    df = load_month(args.month, all_columns=args.schema_only)
    print(f"\nloaded {len(df):,} rows")
    print("columns:", list(df.columns))
    if args.schema_only:
        print(df.dtypes.to_string())
        print(df.head(3).to_string())
        return

    if args.berlin_only:
        before = len(df)
        df = df[df[COLS["station"]].str.contains("Berlin", case=False, na=False)]
        print(f"berlin-only: {len(df):,} of {before:,} rows "
              f"({df[COLS['station']].nunique()} stations)")

    missing = [c for c in COLS.values() if c not in df.columns]
    if missing:
        print(f"\n!! these expected columns are missing: {missing}")
        print("   Adjust COLS at the top of this file to the real names above, then re-run.")
        return

    check_delay_semantics(df)
    baseline(add_delay(df))


if __name__ == "__main__":
    main()
