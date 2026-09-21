"""Work out which VBB station IDs actually belong to each logged interchange.

WHY THIS EXISTS
Berlin does not file a whole interchange under one ID. The GTFS-RT feed was being
filtered on the hub ID alone, so at Alexanderplatz it captured ZERO trams and ZERO
buses, and regional/express services were missing everywhere -- they sit on sibling
station IDs such as:

    900100005  "U Alexanderplatz (Berlin) [Tram]"
    900100031  "S+U Alexanderplatz Bhf/Memhardstr. (Berlin)"
    900003200  "S+U Berlin Hauptbahnhof"              (the regional platforms)

The transport.rest API resolves a whole station server-side and was unaffected, so
this silently degraded only the BACKUP source -- the one whose entire purpose is to
cover the primary's outages.

THE RULE, deliberately conservative: a station counts as part of an interchange if
it lies within RADIUS_M of the hub AND its name contains the hub's distinguishing
word. Proximity alone would swallow genuinely separate stops that merely sit nearby
(Littenstr., Memhardstr., Helsingforser Platz); the name test excludes them.

    python analysis/derive_stop_siblings.py --gtfs <vbb_gtfs.zip>

Prints a STOP_SIBLINGS block to paste into transit_logger/stops.py, so the result
is reviewable in a diff rather than recomputed at runtime from a 77 MB download.
"""
from __future__ import annotations

import argparse
import csv
import io
import math
import re
import sys
import zipfile

RADIUS_M = 400

# Real VBB station numbers only. stops.txt also carries IFOPT sub-elements
# (entrances, pathways) under ids like "000300003020" which share the station
# name but are not stations and never appear as a GTFS-RT stop_id.
STATION_ID = re.compile(r"^900\d{6}$")

# hub id -> the word a sibling's name must contain
HUBS = {
    "900100003": "Alexanderplatz",
    "900023201": "Zoologischer Garten",
    "900120004": "Warschauer",
    "900003201": "Hauptbahnhof",
}


def _metres(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    return math.hypot((lat1 - lat2) * 111320,
                      (lon1 - lon2) * 111320 * math.cos(math.radians(lat1)))


def base_id(stop_id: str) -> str:
    parts = stop_id.split(":")
    return parts[2] if len(parts) > 2 else stop_id


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gtfs", required=True, help="path to the VBB GTFS zip")
    args = ap.parse_args()

    with zipfile.ZipFile(args.gtfs) as zf:
        rows = list(csv.DictReader(
            io.StringIO(zf.read("stops.txt").decode("utf-8-sig"))))

    coords: dict[str, tuple[float, float]] = {}
    names: dict[str, str] = {}
    for r in rows:
        b = base_id(r["stop_id"])
        if not STATION_ID.match(b):
            continue
        try:
            lat, lon = float(r["stop_lat"]), float(r["stop_lon"])
        except (TypeError, ValueError, KeyError):
            continue
        coords.setdefault(b, (lat, lon))
        names.setdefault(b, r["stop_name"])

    print(f"# Derived by analysis/derive_stop_siblings.py from the VBB static GTFS.")
    print(f"# Rule: within {RADIUS_M} m of the hub AND name contains the hub word.")
    print("STOP_SIBLINGS = {")
    for hub, word in HUBS.items():
        if hub not in coords:
            print(f"!! hub {hub} not in GTFS", file=sys.stderr)
            continue
        hlat, hlon = coords[hub]
        found = sorted(
            b for b, (lat, lon) in coords.items()
            if _metres(hlat, hlon, lat, lon) <= RADIUS_M
            and word.lower() in names[b].lower()
        )
        print(f'    "{hub}": [')
        for b in found:
            tag = "   <- hub" if b == hub else ""
            print(f'        "{b}",'.ljust(24) + f"# {names[b]}{tag}")
        print("    ],")
    print("}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
