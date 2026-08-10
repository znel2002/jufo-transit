"""Turn the logged polls into one modelling table: transit x weather x calendar.

    python analysis/build_dataset.py
    python analysis/build_dataset.py --out data/dataset.parquet

Pipeline:
  1. Read every gzipped poll file under data/observations/.
  2. Collapse the append-only observations to ONE row per departure.
  3. Join hourly DWD weather on the UTC hour.
  4. Add Berlin calendar features.
  5. Write Parquet + print a data-quality report.

GROUND TRUTH, AND ITS KNOWN WEAKNESS
The BVG API reports the *current estimate* of a departure's delay. Once a vehicle
has departed it drops off the board, so the realised delay is never observed
directly: our label is the last estimate seen before departure. With a 30-minute
look-ahead polled every 15 minutes, that estimate is typically 0-15 minutes old.
`lead_time_s` records exactly how old it is for every row, so the error can be
quantified instead of assumed away.

There is a selection effect on top of that, and it is worth stating plainly in the
write-up: a delayed vehicle stays on the departure board *longer*, so it tends to be
observed closer to its real departure than a punctual one. Label quality is
therefore not uniform across the target range -- it is best exactly where the delays
are largest.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
from pathlib import Path

import pandas as pd

import calendar_de
import fetch_weather

ROOT = Path(__file__).resolve().parent.parent
OBS_GLOB = str(ROOT / "data" / "observations" / "*" / "*.ndjson.gz")
DEFAULT_OUT = ROOT / "data" / "dataset.parquet"

KEY = ["trip_id", "stop_id", "planned_when"]
# Carried through from the last observation of each departure.
CARRY = ["line_name", "product", "direction", "platform", "planned_platform"]


def load_observations(pattern: str = OBS_GLOB) -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no poll files matched {pattern} -- has the logger run?")
    rows = []
    for path in files:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            rows.extend(json.loads(line) for line in fh)
    df = pd.DataFrame(rows)
    print(f"{len(files):,} poll files -> {len(df):,} raw observations")
    return df


def collapse_to_departures(df: pd.DataFrame) -> pd.DataFrame:
    """One row per departure, from the append-only observation stream.

    Uses first/last *rows* rather than a groupby aggregation on purpose: pandas'
    groupby `first`/`last` skip NaN, which would silently paper over departures that
    never had a realtime estimate. Those nulls are information -- keep them.
    """
    df = df.dropna(subset=["trip_id", "planned_when"]).copy()
    df["observed_at"] = pd.to_datetime(df["observed_at"], utc=True, errors="coerce")
    df["planned_when"] = pd.to_datetime(df["planned_when"], utc=True, errors="coerce")
    df = df.dropna(subset=["observed_at", "planned_when"]).sort_values("observed_at")

    last = df.drop_duplicates(KEY, keep="last")
    first = df.drop_duplicates(KEY, keep="first")[KEY + ["delay_s", "observed_at"]]

    out = last[KEY + CARRY + ["delay_s", "observed_at", "cancelled"]].rename(
        columns={"delay_s": "final_delay_s", "observed_at": "last_observed_at"}
    ).merge(
        first.rename(columns={"delay_s": "first_delay_s",
                              "observed_at": "first_observed_at"}),
        on=KEY, how="left",
    )

    counts = df.groupby(KEY, observed=True).size().rename("n_obs").reset_index()
    out = out.merge(counts, on=KEY, how="left")

    # Any observation flagging a cancellation cancels the departure.
    cancelled_any = df.groupby(KEY, observed=True)["cancelled"].max().rename(
        "cancelled_any").reset_index()
    out = out.merge(cancelled_any, on=KEY, how="left")
    out["cancelled"] = out["cancelled_any"].fillna(0).astype(int)
    out = out.drop(columns=["cancelled_any"])

    # How stale is the label? Positive = last seen before the planned departure.
    out["lead_time_s"] = (
        out["planned_when"] - out["last_observed_at"]
    ).dt.total_seconds()
    # How much the estimate moved while we watched -- a feature for the nowcast only.
    out["delay_drift_s"] = out["final_delay_s"] - out["first_delay_s"]

    print(f"collapsed to {len(out):,} departures "
          f"({len(df)/max(len(out),1):.2f} observations each on average)")
    return out


WEATHER_COLS = ["temp_c", "humidity_pct", "precip_mm",
                "precip_form", "wind_ms", "wind_dir_deg"]


def join_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Join DWD hourly weather on the UTC hour of the planned departure.

    DWD's MESS_DATUM is a UTC hour stamp, so this is a direct merge -- no timezone
    conversion, which is also why the join is done on UTC while every *calendar*
    feature is built from local Berlin time.

    Fetches the whole `/recent/` archive rather than only the dates we need: the
    DWD endpoint serves one zip per category regardless, so a date filter would
    save no bandwidth and could only ever discard usable rows.
    """
    print("\nfetching DWD weather (full /recent/ archive)")
    rows = fetch_weather.fetch(None)
    if not rows:
        print("  ! no weather returned -- continuing without it")
        return df.assign(**{c: pd.NA for c in WEATHER_COLS})

    w = pd.DataFrame(rows)
    w["hour_utc"] = pd.to_datetime(w["timestamp_utc"], format="%Y%m%d%H", utc=True)
    for c in WEATHER_COLS:
        w[c] = pd.to_numeric(w.get(c), errors="coerce")
    w = w.drop(columns=["timestamp_utc"])

    df = df.copy()
    df["hour_utc"] = df["planned_when"].dt.floor("h")
    merged = df.merge(w, on="hour_utc", how="left")

    first, latest = w["hour_utc"].min(), w["hour_utc"].max()
    matched = merged["temp_c"].notna().mean()
    print(f"  {len(w):,} hourly rows covering {first} -> {latest}")
    print(f"  weather matched for {matched:.1%} of departures")
    if matched < 0.95:
        ahead = (df["hour_utc"] > latest).mean()
        print(f"  -> {ahead:.1%} of departures are later than the newest weather hour. "
              f"DWD publishes with a ~16 h lag, so the most recent day is never "
              f"available yet; re-run after the lag has passed to fill these in.")
    return merged


def add_derived_weather(df: pd.DataFrame) -> pd.DataFrame:
    """A few interpretable weather flags, kept explicit for the driver analysis."""
    out = df.copy()
    out["is_snow"] = out["precip_form"].isin([7, 8])       # 7 = snow, 8 = rain+snow
    out["is_rain"] = out["precip_form"].isin([6, 8])
    out["is_wet"] = out["precip_mm"].fillna(0) > 0
    out["is_freezing"] = out["temp_c"] < 0
    out["is_hot"] = out["temp_c"] > 28                      # rail/overhead-line heat
    return out


def report(df: pd.DataFrame) -> None:
    print("\n" + "=" * 62)
    print("DATA QUALITY REPORT")
    print("=" * 62)
    print(f"departures            {len(df):,}")
    print(f"planned_when span     {df['planned_when'].min()}  ->  {df['planned_when'].max()}")
    print(f"stops / lines         {df['stop_id'].nunique()} / {df['line_name'].nunique()}")
    print(f"cancelled             {int(df['cancelled'].sum()):,} "
          f"({df['cancelled'].mean():.2%})")

    print("\nobservations per departure:")
    print(df["n_obs"].value_counts().sort_index().to_string())

    print("\nlead_time_s (planned - last seen; how stale the label is):")
    lt = df["lead_time_s"].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
    print((lt / 60).round(2).to_string() + "   [minutes]")
    print(f"  negative (seen at/after planned departure): {(df['lead_time_s'] < 0).mean():.1%}")

    live = df[df["cancelled"] == 0]
    d = live["final_delay_s"].dropna()
    if len(d):
        print(f"\nfinal_delay_s (non-cancelled, n={len(d):,}, "
              f"{live['final_delay_s'].isna().mean():.1%} without realtime):")
        print(f"  mean {d.mean()/60:6.2f} min | median {d.median()/60:6.2f} min | "
              f"p95 {d.quantile(0.95)/60:6.2f} min | max {d.max()/60:6.2f} min")
        print(f"  share > 3 min {(d > 180).mean():.1%} | > 5 min {(d > 300).mean():.1%} | "
              f"early (< 0) {(d < 0).mean():.1%}")
        print("  NOTE: MAE is minimised by the median, RMSE by the mean. On a "
              "distribution this skewed they disagree -- report both.")

    print("\nby product:")
    prod = live.groupby("product")["final_delay_s"].agg(["size", "mean", "median"])
    print((prod.assign(mean=lambda x: x["mean"] / 60,
                       median=lambda x: x["median"] / 60).round(2)).to_string())

    nan_cols = df.columns[df.isna().any()]
    if len(nan_cols):
        print("\nmissing values:")
        for c in nan_cols:
            print(f"  {c:<20} {df[c].isna().mean():6.1%}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="Parquet output path")
    ap.add_argument("--no-weather", action="store_true",
                    help="skip the DWD fetch (offline / quick structural check)")
    args = ap.parse_args()

    df = collapse_to_departures(load_observations())

    if args.no_weather:
        print("\nskipping weather (--no-weather)")
    else:
        df = add_derived_weather(join_weather(df))

    df = calendar_de.add_calendar_features(df, "planned_when")
    report(df)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"\nwrote {out}  ({len(df):,} rows x {len(df.columns)} cols, "
          f"{out.stat().st_size/1024:.0f} kB)")


if __name__ == "__main__":
    main()
