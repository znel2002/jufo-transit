"""Fetch DWD hourly weather observations for Berlin and join them to one table.

WHY THIS IS A SEPARATE, ON-DEMAND SCRIPT (not a second logger):
DWD publishes station observations *retroactively* with roughly a one-day lag, so
weather for any past date can be fetched whenever we like. That is the opposite
of the BVG realtime feed, which is gone forever if not captured at the time.
So: log transit continuously, fetch weather in one pass at analysis time.

Station choice (verified 2026-08-10 against DWD's own station list):
    00433  Berlin-Tempelhof   1951-01-01 -> present   <- default: central, longest record
    00403  Berlin-Dahlem (FU) 2002-01-01 -> present
    00400  Berlin-Buch        1991-01-01 -> present
    00420  Berlin-Marzahn     2007-08-01 -> present
    00427  Berlin Brandenburg 1973-01-01 -> present
  Do NOT use Berlin-Alexanderplatz (00399, ended 2011) or Berlin-Tegel (00430,
  ended 2021) — plausible names, dead stations.

Usage:
    python analysis/fetch_weather.py --check              # verify feed is reachable
    python analysis/fetch_weather.py --from 2026-08-10 --out data/weather.csv
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone

BASE = ("https://opendata.dwd.de/climate_environment/CDC/"
        "observations_germany/climate/hourly")

STATION = "00433"          # Berlin-Tempelhof
MISSING = {"-999", "-999.0"}

# category -> (url path, zip filename prefix, {DWD column: friendly name})
# Chosen for delay relevance: rain/snow, freezing temps, wind gusts.
CATEGORIES = {
    "air_temperature": ("air_temperature", "TU", {
        "TT_TU": "temp_c",
        "RF_TU": "humidity_pct",
    }),
    "precipitation": ("precipitation", "RR", {
        "R1": "precip_mm",
        "WRTR": "precip_form",     # 6 = rain, 7 = snow, 8 = rain+snow
    }),
    "wind": ("wind", "FF", {
        "F": "wind_ms",
        "D": "wind_dir_deg",
    }),
}


def _download(category: str, prefix: str) -> dict[str, dict[str, str]]:
    """Return {MESS_DATUM: {col: value}} for one DWD category."""
    url = f"{BASE}/{category}/recent/stundenwerte_{prefix}_{STATION}_akt.zip"
    with urllib.request.urlopen(url, timeout=90) as resp:
        blob = resp.read()

    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        name = next(n for n in zf.namelist() if n.startswith("produkt_"))
        text = zf.read(name).decode("latin-1")

    rows: dict[str, dict[str, str]] = {}
    reader = csv.DictReader(io.StringIO(text), delimiter=";")
    for raw in reader:
        rec = {k.strip(): (v.strip() if v else "") for k, v in raw.items() if k}
        stamp = rec.get("MESS_DATUM", "")
        if stamp:
            rows[stamp] = rec
    return rows


def fetch(date_from: str | None = None) -> list[dict[str, str]]:
    """Fetch and merge all categories on the hourly timestamp."""
    merged: dict[str, dict[str, str]] = {}

    for label, (path, prefix, cols) in CATEGORIES.items():
        try:
            data = _download(path, prefix)
        except Exception as exc:
            print(f"  ! {label}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        print(f"  {label:16} {len(data):>6} hourly rows")
        for stamp, rec in data.items():
            out = merged.setdefault(stamp, {"timestamp_utc": stamp})
            for src, friendly in cols.items():
                val = rec.get(src, "")
                out[friendly] = "" if val in MISSING else val

    rows = []
    cutoff = date_from.replace("-", "") if date_from else None
    for stamp in sorted(merged):
        if cutoff and stamp[:8] < cutoff:
            continue
        rows.append(merged[stamp])
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="verify the feed works and show the latest observation")
    ap.add_argument("--from", dest="date_from",
                    help="earliest date to keep, YYYY-MM-DD")
    ap.add_argument("--out", help="write CSV here")
    args = ap.parse_args()

    print(f"DWD station {STATION} (Berlin-Tempelhof)")
    rows = fetch(args.date_from)
    if not rows:
        print("no rows returned", file=sys.stderr)
        return 1

    print(f"\n{len(rows)} merged hourly rows"
          f"  ({rows[0]['timestamp_utc']} -> {rows[-1]['timestamp_utc']})")

    if args.check:
        print("\nlatest 3 observations:")
        fields = ["timestamp_utc", "temp_c", "humidity_pct",
                  "precip_mm", "precip_form", "wind_ms"]
        print("  " + " | ".join(f"{f:>13}" for f in fields))
        for r in rows[-3:]:
            print("  " + " | ".join(f"{r.get(f, ''):>13}" for f in fields))
        latest = datetime.strptime(rows[-1]["timestamp_utc"], "%Y%m%d%H").replace(
            tzinfo=timezone.utc)
        lag = (datetime.now(timezone.utc) - latest).total_seconds() / 3600
        print(f"\npublication lag: {lag:.1f} hours behind now")

    if args.out:
        fields = ["timestamp_utc", "temp_c", "humidity_pct",
                  "precip_mm", "precip_form", "wind_ms", "wind_dir_deg"]
        with open(args.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
