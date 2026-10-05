"""Do the two sources agree about the same departure? Read-only.

    python analysis/source_agreement.py

The dataset keeps one row per physical departure (logger row preferred), so it
cannot show whether transport.rest and VBB GTFS-RT report the same delay for the
same vehicle. That matters: the merged dataset mixes rows from both, and from
2026-08-20 to 08-25 (and every transport.rest dropout since) the record is
GTFS-RT-only. If the sources disagreed systematically, the mix would shift with
the outage pattern and look like a real change in punctuality.

Both sources are collapsed separately with build_dataset's own collapse, then
matched on (stop, planned departure, line).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_dataset as bd  # noqa: E402


def main() -> int:
    a = bd.collapse_to_departures(bd.load_observations().assign(source="transport.rest"))
    b = bd.collapse_to_departures(bd.load_gtfsrt())
    k = ["stop_id", "planned_when", "line_name"]
    for d in (a, b):
        d["stop_id"] = d.stop_id.astype(str)
        d.drop_duplicates(k, keep=False, inplace=True)   # ambiguous keys out
    m = a[k + ["final_delay_s", "lead_time_s", "product"]].merge(
        b[k + ["final_delay_s", "lead_time_s"]], on=k, suffixes=("_tr", "_rt"))
    m = m.dropna(subset=["final_delay_s_tr", "final_delay_s_rt"])
    d = (m.final_delay_s_tr - m.final_delay_s_rt) / 60
    print(f"\ndepartures seen by BOTH sources with a label in both: {len(m):,}")
    print(f"  identical delay            {(d == 0).mean():.1%}")
    print(f"  within 1 minute            {(d.abs() <= 1).mean():.1%}")
    print(f"  differ by > 3 minutes      {(d.abs() > 3).mean():.1%}")
    print(f"  mean difference (tr - rt)  {d.mean():+.2f} min   median {d.median():+.1f}")
    print(f"  same >=3 min verdict       "
          f"{((m.final_delay_s_tr >= 180) == (m.final_delay_s_rt >= 180)).mean():.1%}")
    print(f"  late rate >=3 min          transport.rest {(m.final_delay_s_tr>=180).mean():.1%}"
          f" | GTFS-RT {(m.final_delay_s_rt>=180).mean():.1%}")
    # Disagreement should mostly be timing: whichever source saw the trip later
    # has the fresher estimate.
    fresher = (m.lead_time_s_tr < m.lead_time_s_rt)
    print(f"  transport.rest label is the fresher one in {fresher.mean():.1%} of pairs")
    big = d.abs() > 3
    if big.any():
        lt_gap = (m.lead_time_s_tr - m.lead_time_s_rt).abs()[big] / 60
        print(f"  among >3-min disagreements, median label-age gap {lt_gap.median():.1f} min")
    print("\nby product:")
    g = m.assign(d=d).groupby("product").agg(n=("d", "size"),
        exact=("d", lambda s: (s == 0).mean()), within1=("d", lambda s: (s.abs() <= 1).mean()),
        mean=("d", "mean"))
    print(g.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
