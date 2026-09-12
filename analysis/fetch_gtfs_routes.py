"""Build the route_id -> (line name, product) lookup the GTFS-RT feed needs.

WHY THIS EXISTS
The VBB GTFS-Realtime feed identifies a trip only by `route_id` (e.g.
"10148_109"). It carries no line name and no vehicle type, and those two are
exactly the features the delay model leans on hardest -- stripping identity drops
ROC-AUC from 0.801 to 0.609. Without this mapping the second data source cannot be
used for modelling at all, which is why ~1 million captured rows sat unused.

The static GTFS feed supplies both. It is a 74 MB download, so it is fetched ONCE
and reduced to a small committed CSV (~1,250 routes). Analysis then needs no
network access and stays reproducible for a jury.

    python analysis/fetch_gtfs_routes.py            # refresh analysis/vbb_routes.csv

Source: https://www.vbb.de/vbbgtfs (VBB open data). Verified 2026-09-12: every one
of the route_ids in a live GTFS-RT cycle resolved against it (37/37).
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

GTFS_URL = "https://www.vbb.de/vbbgtfs"
# vbb.de answers urllib's default User-Agent with 403, so send a real one that
# also says who is asking -- the same courtesy the GTFS-RT feed asks for.
USER_AGENT = ("jufo-transit-delay-study/0.1 (Jugend forscht student research "
              "project; +https://github.com/znel2002/jufo-transit)")
OUT = Path(__file__).resolve().parent / "vbb_routes.csv"

# Extended GTFS route types -> the product vocabulary the self-logged source uses,
# so both sources describe a vehicle the same way.
ROUTE_TYPE_TO_PRODUCT = {
    "0": "tram", "900": "tram", "901": "tram", "902": "tram",
    "1": "subway", "400": "subway", "401": "subway", "402": "subway",
    "109": "suburban", "110": "suburban",
    "2": "regional", "100": "regional", "101": "express", "102": "express",
    "103": "regional", "105": "express", "106": "regional", "107": "regional",
    "3": "bus", "700": "bus", "701": "bus", "702": "bus", "704": "bus",
    "715": "bus", "1500": "bus",
    "4": "ferry", "1000": "ferry", "1200": "ferry",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=GTFS_URL)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    print(f"downloading {args.url} (~74 MB, once) ...")
    req = urllib.request.Request(args.url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=600) as resp:
        blob = resp.read()
    print(f"  {len(blob)/1e6:.1f} MB")

    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        text = zf.read("routes.txt").decode("utf-8-sig")

    rows, unknown = [], set()
    for r in csv.DictReader(io.StringIO(text)):
        rtype = (r.get("route_type") or "").strip()
        product = ROUTE_TYPE_TO_PRODUCT.get(rtype)
        if product is None:
            unknown.add(rtype)
            product = ""
        rows.append({
            "route_id": r["route_id"],
            "line_name": (r.get("route_short_name") or "").strip(),
            "route_type": rtype,
            "product": product,
        })

    out = Path(args.out)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["route_id", "line_name", "route_type", "product"])
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {out}  ({len(rows):,} routes, {out.stat().st_size/1024:.0f} kB)")
    if unknown:
        print(f"  ! unmapped route_type values (product left blank): {sorted(unknown)}")
        print("    add them to ROUTE_TYPE_TO_PRODUCT if they appear in the logged stops")
    return 0


if __name__ == "__main__":
    sys.exit(main())
