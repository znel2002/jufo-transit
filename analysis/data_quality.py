"""Data-quality audit of the modelling table. Read-only; prints a report.

    python analysis/data_quality.py

Checks what a juror (or the Fehlerquellen chapter) would ask about the dataset:
volume and spread over time, per-source contribution, completeness of every
feature, how stale the label is, internal consistency, duplicates, and whether the
mid-series changes (source outage, GTFS-RT scope change on 2026-09-21) shift the
data in ways an analysis must account for.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SCOPE_CHANGE = pd.Timestamp("2026-09-21 09:40", tz="UTC")


def pct(x: float) -> str:
    return f"{x:6.1%}"


def main() -> int:
    df = pd.read_parquet(ROOT / "data" / "dataset.parquet")
    df["day"] = df.planned_when.dt.tz_convert("Europe/Berlin").dt.date
    live = df[df.cancelled == 0]
    lab = live[live.final_delay_s.notna()]

    print("=" * 70)
    print("1. VOLUME")
    print("=" * 70)
    print(f"departures            {len(df):,}")
    print(f"  cancelled           {int(df.cancelled.sum()):,}  ({df.cancelled.mean():.2%})")
    print(f"  with a delay label  {len(lab):,}  ({len(lab)/len(live):.1%} of non-cancelled)")
    print(f"span                  {df.planned_when.min():%Y-%m-%d} -> {df.planned_when.max():%Y-%m-%d}"
          f"  ({df.day.nunique()} days)")
    per_day = df.groupby("day").size()
    print(f"per day               median {per_day.median():,.0f}, min {per_day.min():,} "
          f"({per_day.idxmin()}), max {per_day.max():,}")
    thin = per_day[per_day < 0.5 * per_day.median()]
    print(f"days under half the median volume: {len(thin)} -> "
          + ", ".join(f"{d} ({n:,})" for d, n in thin.items()))

    print("\nby source:")
    print(df.groupby("source").size().rename("departures").to_string())
    print("\nby product (non-cancelled, labelled):")
    print(lab.groupby("product").size().sort_values(ascending=False).to_string())
    print("\nby stop:")
    print(df.groupby("stop_id").size().to_string())

    print("\n" + "=" * 70)
    print("2. COMPLETENESS (share missing)")
    print("=" * 70)
    cols = ["line_name", "product", "direction", "final_delay_s", "first_delay_s",
            "temp_c", "precip_mm", "wind_ms", "humidity_pct"]
    for src, g in [("all", df)] + list(df.groupby("source")):
        print(f"  {src:<16}" + "  ".join(f"{c[:10]:>10} {pct(g[c].isna().mean())}"
                                          for c in cols[:4]))
        print(f"  {'':<16}" + "  ".join(f"{c[:10]:>10} {pct(g[c].isna().mean())}"
                                          for c in cols[4:8]))

    print("\n" + "=" * 70)
    print("3. LABEL QUALITY")
    print("=" * 70)
    v = lab.final_delay_s
    print(f"quantised to whole minutes: {(v % 60 == 0).mean():.2%}")
    print(f"distribution: exactly 0 {(v == 0).mean():.1%} | late {(v > 0).mean():.1%} | "
          f"early {(v < 0).mean():.1%} | >=3 min {(v >= 180).mean():.1%} | "
          f">=10 min {(v >= 600).mean():.2%}")
    print(f"extreme values: max {v.max()/60:.0f} min, min {v.min()/60:.0f} min, "
          f">60 min: {(v > 3600).sum():,}, < -10 min: {(v < -600).sum():,}")
    lt = lab.lead_time_s / 60
    print("label age (lead_time = minutes between last sighting and planned departure):")
    print(f"  median {lt.median():.1f} | p75 {lt.quantile(.75):.1f} | p95 {lt.quantile(.95):.1f}"
          f" | label <= 10 min old: {(lt <= 10).mean():.1%} | > 30 min old: {(lt > 30).mean():.1%}")
    for src, g in lab.groupby("source"):
        l2 = g.lead_time_s / 60
        print(f"  {src:<16} median {l2.median():5.1f} min, > 30 min old {(l2 > 30).mean():.1%}")
    # Does a stale label bias the delay downward? (delays surface late)
    bins = pd.cut(lt, [-1e9, 0, 10, 20, 30, 1e9], labels=["<=0", "0-10", "10-20", "20-30", ">30"])
    t = lab.assign(b=bins).groupby("b", observed=True).final_delay_s.agg(
        n="size", late3=lambda s: (s >= 180).mean())
    print("  share >=3 min late by label age (stale labels understate delay?):")
    for b, r in t.iterrows():
        print(f"    {b:>6} min  n={int(r.n):>7,}  late>=3min {r.late3:.1%}")

    print("\n" + "=" * 70)
    print("4. CONSISTENCY")
    print("=" * 70)
    key = ["trip_id", "stop_id", "planned_when"]
    print(f"duplicate (trip, stop, planned) keys: {int(df.duplicated(key).sum()):,}")
    k = df.stop_id.astype(str) + "|" + df.planned_when.astype(str) + "|" + df.line_name.astype(str)
    cross = df.assign(k=k).groupby("k").source.nunique()
    print(f"cross-source duplicate keys:          {int((cross > 1).sum()):,}")
    if "first_delay_s" in lab and "delay_drift_s" in lab:
        ok = (lab.final_delay_s - lab.first_delay_s - lab.delay_drift_s).abs().fillna(0) < 1
        print(f"drift == final - first:               {ok.mean():.2%}")
    print(f"planned_when in the future (> build): "
          f"{int((df.planned_when > pd.Timestamp.now(tz='UTC')).sum()):,}")

    print("\n" + "=" * 70)
    print("5. STABILITY OVER TIME (what an analysis must account for)")
    print("=" * 70)
    wk = lab.assign(w=lab.planned_when.dt.tz_convert("Europe/Berlin").dt.to_period("W"))
    g = wk.groupby("w").final_delay_s.agg(n="size", late3=lambda s: (s >= 180).mean(),
                                          late10=lambda s: (s >= 600).mean())
    g["gtfsrt_share"] = wk.groupby("w").source.apply(lambda s: (s == "vbb-gtfsrt").mean())
    print(f"{'week':<23}{'n':>8}{'>=3min':>9}{'>=10min':>9}{'GTFS-RT':>9}")
    for w, r in g.iterrows():
        print(f"{str(w):<23}{int(r.n):>8,}{r.late3:>9.1%}{r.late10:>9.2%}{r.gtfsrt_share:>9.1%}")
    rt = lab[lab.source == "vbb-gtfsrt"]
    pre, post = rt[rt.planned_when < SCOPE_CHANGE], rt[rt.planned_when >= SCOPE_CHANGE]
    print(f"\nGTFS-RT before/after the 2026-09-21 scope change:")
    for name, part in (("before", pre), ("after", post)):
        if len(part):
            print(f"  {name:<7} n={len(part):>7,}  tram {(part['product']=='tram').mean():.1%}  "
                  f"bus {(part['product']=='bus').mean():.1%}  regional "
                  f"{(part['product']=='regional').mean():.1%}  late>=3min "
                  f"{(part.final_delay_s>=180).mean():.1%}")
    print("\n(Cross-source agreement on the SAME departure needs pre-dedupe data:\n"
          " see analysis/source_agreement.py.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
