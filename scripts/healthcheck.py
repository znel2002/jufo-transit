"""Quick freshness check on the live collection. Run it any time.

    python scripts/healthcheck.py

Rewritten 2026-09-12. It previously read only `data/transit.db`, the SQLite
backend, which stopped being the production path the moment the logger moved to
GitHub Actions on 2026-08-10. Running it since then printed a healthy-looking
report ending on 2026-08-10 -- it would have shown the same thing whether
collection was fine or had been dead for a month.

That was the same class of defect as the coverage figure that hid a four-day
outage and the exit code that reported recovery as failure: a monitor that cannot
tell "working" from "stopped" is worse than no monitor, because it reassures.

For the full picture (coverage, gaps, per-source redundancy, delay distributions)
use `scripts/collection_dashboard.py`. This is the 2-second version.
"""
from __future__ import annotations

import glob
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCES = {
    "transport.rest": (ROOT / "data" / "observations", "n_rows"),
    "VBB GTFS-RT": (ROOT / "data" / "gtfsrt", "rows_kept"),
}
STALE_MIN = 25          # a cycle is due every 15 min; allow one late run


def latest(dir_: Path, key: str):
    files = sorted(glob.glob(str(dir_ / "*" / "*.poll.json")))
    if not files:
        return None
    for path in reversed(files[-40:]):          # newest first
        try:
            m = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        return m, len(files)
    return None


def main() -> int:
    now = datetime.now(timezone.utc)
    print(f"collection healthcheck  {now:%Y-%m-%d %H:%M} UTC\n")
    worst = 0.0
    any_data = False

    for name, (dir_, key) in SOURCES.items():
        got = latest(dir_, key)
        if not got:
            print(f"  {name:<16} no data on disk")
            continue
        meta, n_files = got
        t = datetime.fromisoformat(meta["cycle_at"]).astimezone(timezone.utc)
        age = (now - t).total_seconds() / 60
        rows = meta.get(key, 0)
        worst = max(worst, age) if rows else worst
        any_data = any_data or rows > 0
        flag = "" if age <= STALE_MIN else "   <-- STALE"
        status = f"{rows:>5,} rows" if rows else "    0 rows  (source down)"
        print(f"  {name:<16} {n_files:>5,} cycles | last {t:%m-%d %H:%M} "
              f"({age:5.1f} min ago) | {status}{flag}")

    print()
    if not any_data:
        print("  !! NEITHER source returned data in its last cycle")
        return 1
    if worst > STALE_MIN:
        print(f"  !! newest usable data is {worst:.0f} min old - check the workflow")
        return 1
    print("  OK - at least one source is delivering fresh data")
    print("  full report: python scripts/collection_dashboard.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
