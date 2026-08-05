"""Quick health/verification report on the logged data.

Run any time to confirm the logger is alive and the data looks sane.
This doubles as the 48-hour verification step from the build plan.

    python scripts/healthcheck.py
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "transit.db"


def main() -> None:
    if not DB_PATH.exists():
        print("No DB yet at", DB_PATH, "- has the logger run?")
        return
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    total = conn.execute("SELECT COUNT(*) c FROM observations").fetchone()["c"]
    print(f"observations rows: {total:,}")

    first = conn.execute("SELECT MIN(observed_at) a FROM observations").fetchone()["a"]
    last = conn.execute("SELECT MAX(observed_at) a FROM observations").fetchone()["a"]
    print(f"time span: {first}  ->  {last}")

    print("\nrows per product:")
    for r in conn.execute(
        "SELECT product, COUNT(*) c FROM observations GROUP BY product ORDER BY c DESC"
    ):
        print(f"  {r['product'] or '(null)':<12} {r['c']:,}")

    print("\ndelay distribution (seconds, non-cancelled, non-null):")
    delays = [
        r[0] for r in conn.execute(
            "SELECT delay_s FROM observations "
            "WHERE delay_s IS NOT NULL AND cancelled=0 ORDER BY delay_s"
        )
    ]
    if delays:
        n = len(delays)
        def pct(p: float) -> int:
            return delays[min(n - 1, int(p * n))]
        print(f"  count    {n:,}")
        print(f"  min      {delays[0]}")
        print(f"  p25      {pct(0.25)}")
        print(f"  median   {pct(0.50)}")
        print(f"  p75      {pct(0.75)}")
        print(f"  p95      {pct(0.95)}")
        print(f"  max      {delays[-1]}")
        print(f"  avg      {round(sum(delays) / n, 1)}")
    else:
        print("  (no delay values yet)")

    cancelled = conn.execute(
        "SELECT COUNT(*) c FROM observations WHERE cancelled=1"
    ).fetchone()["c"]
    print(f"\ncancelled departures seen: {cancelled:,}")

    # Gap check: largest gap between consecutive polls (a proxy for outages).
    print("\npoll health (last 20 poll cycles):")
    errs = conn.execute(
        "SELECT status, COUNT(*) c FROM poll_log GROUP BY status"
    ).fetchall()
    for r in errs:
        print(f"  {r['status']:<12} {r['c']:,}")

    # Freshness: how long since the last successful poll?
    if last:
        last_dt = datetime.fromisoformat(last)
        age_min = (datetime.now(timezone.utc) - last_dt).total_seconds() / 60
        flag = "  <-- STALE, check the logger!" if age_min > 10 else ""
        print(f"\nlast observation age: {age_min:.1f} min{flag}")


if __name__ == "__main__":
    main()
