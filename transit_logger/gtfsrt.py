"""Fallback/parallel logger: VBB's official GTFS-Realtime feed.

WHY THIS EXISTS
`v6.bvg.transport.rest` -- the project's original source -- went down on
2026-08-20 and stayed down. Every transport.rest instance (bvg/vbb/db, v5 and v6)
resolves to a single machine (thuya.jannisr.de / 162.55.47.160), so one
community-run server was a single point of failure for the whole dataset. Four
days and ~348 polls were lost before it was noticed.

VBB publishes an official GTFS-Realtime feed that needs no authentication:
    https://production.gtfsrt.vbb.de/data     (CC-BY 4.0, 60 requests/minute)

This module captures it in parallel with the original logger, deliberately
WITHOUT committing to a full migration. Realtime data not captured now is gone
forever, whereas parsing decisions can be revisited at any time -- the same
asymmetry that made the project log transit live but backfill weather.

WHAT IS STORED
One gzipped NDJSON file per cycle under ``data/gtfsrt/<UTC day>/``, holding the
stop-time updates for the logged stops within HORIZON_MIN. Fields are kept close
to the raw feed rather than force-fitted to the old schema, so nothing is thrown
away before the migration question is settled.

Measured 2026-08-23: the full feed is ~6.6 MB (6,560 trips, 131,689 stop-time
updates). Restricted to our four stops and a 65-minute horizon that becomes 409
records, of which 70.7% carry a departure delay. The unfiltered figure looks far
worse (8.4%) only because most of the feed is trips hours in the future that have
no realtime yet.

Usage:
    python -m transit_logger.gtfsrt --once
    python -m transit_logger.gtfsrt --once --stats     # print, do not write
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

from .stops import STOP_IDS

FEED_URL = "https://production.gtfsrt.vbb.de/data"
HORIZON_MIN = 65           # matches the original logger's look-ahead window
HTTP_TIMEOUT_S = 60        # the feed is ~6.6 MB
HTTP_RETRIES = 3
RETRY_BACKOFF_S = 4

# VBB asks for an informative User-Agent and says generic ones may be blocked.
USER_AGENT = ("jufo-transit-delay-study/0.1 (Jugend forscht student research "
              "project; +https://github.com/znel2002/jufo-transit)")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = DATA_DIR / "gtfsrt"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iso(epoch: int | None) -> str | None:
    if not epoch:
        return None
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def _base_stop_id(stop_id: str) -> str | None:
    """VBB DHIDs look like ``de:11000:900100003::3`` or ``de:11000:900120004:2:53``.

    The third colon-separated field is the plain VBB stop number the rest of the
    project already uses, so the two sources stay joinable.
    """
    parts = stop_id.split(":")
    return parts[2] if len(parts) > 2 else None


def fetch(client: httpx.Client) -> tuple[bytes | None, str]:
    """Download the feed, retrying transient failures. Returns (body, detail)."""
    detail = ""
    for attempt in range(1, HTTP_RETRIES + 1):
        try:
            resp = client.get(FEED_URL)
            if resp.status_code == 200:
                return resp.content, ""
            detail = f"status={resp.status_code}"
            if resp.status_code < 500:
                return None, detail
        except Exception as exc:
            detail = repr(exc)[:300]
        if attempt < HTTP_RETRIES:
            import time
            time.sleep(RETRY_BACKOFF_S * attempt)
    return None, f"{detail} after {HTTP_RETRIES} attempts"


def parse(body: bytes, observed_at: str, stop_ids: set[str],
          horizon_min: int = HORIZON_MIN) -> tuple[list[dict], dict]:
    """Extract stop-time updates for the wanted stops inside the horizon."""
    from google.transit import gtfs_realtime_pb2 as rt

    feed = rt.FeedMessage()
    feed.ParseFromString(body)
    now = datetime.now(timezone.utc)
    rows: list[dict] = []
    seen_trips = 0

    for entity in feed.entity:
        if not entity.HasField("trip_update"):
            continue
        seen_trips += 1
        tu = entity.trip_update
        trip = tu.trip
        for stu in tu.stop_time_update:
            base = _base_stop_id(stu.stop_id)
            if base not in stop_ids:
                continue
            dep = stu.departure if stu.HasField("departure") else None
            arr = stu.arrival if stu.HasField("arrival") else None
            ev = dep or arr
            if ev is None or not ev.HasField("time"):
                continue
            ahead = (datetime.fromtimestamp(ev.time, timezone.utc) - now).total_seconds() / 60
            if ahead > horizon_min:
                continue

            dep_delay = dep.delay if dep is not None and dep.HasField("delay") else None
            dep_time = dep.time if dep is not None and dep.HasField("time") else None
            # GTFS-RT reports the predicted time plus its deviation, so the
            # scheduled time is recovered by subtracting the delay.
            planned = dep_time - dep_delay if (dep_time and dep_delay is not None) else None

            rows.append({
                "observed_at": observed_at,
                "feed_timestamp": _iso(feed.header.timestamp or None),
                "source": "vbb-gtfsrt",
                "stop_id": base,
                "stop_id_full": stu.stop_id,
                "trip_id": trip.trip_id or None,
                "route_id": trip.route_id or None,
                "direction_id": trip.direction_id if trip.HasField("direction_id") else None,
                "start_date": trip.start_date or None,
                "start_time": trip.start_time or None,
                "stop_sequence": stu.stop_sequence if stu.HasField("stop_sequence") else None,
                "planned_when": _iso(planned),
                "when_est": _iso(dep_time),
                "delay_s": dep_delay,
                "arrival_when_est": _iso(arr.time if arr is not None and arr.HasField("time") else None),
                "arrival_delay_s": arr.delay if arr is not None and arr.HasField("delay") else None,
                "schedule_relationship": int(stu.schedule_relationship),
                "trip_schedule_relationship": int(trip.schedule_relationship),
            })

    stats = {
        "feed_bytes": len(body),
        "feed_timestamp": _iso(feed.header.timestamp or None),
        "trips_in_feed": seen_trips,
        "rows_kept": len(rows),
        "rows_with_delay": sum(1 for r in rows if r["delay_s"] is not None),
        "horizon_min": horizon_min,
    }
    return rows, stats


def _write(observed_at: str, rows: list[dict], meta: dict) -> Path:
    """One write-once gzipped file per cycle, plus its status sidecar."""
    out_dir = OUT_DIR / observed_at[:10]
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = observed_at[:19].replace(":", "").replace("-", "") + "Z"
    path = out_dir / f"{stamp}.ndjson.gz"
    n = 1
    while path.exists():
        path = out_dir / f"{stamp}_{n}.ndjson.gz"
        n += 1

    tmp = path.with_suffix(".gz.tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)

    meta_path = path.with_name(path.name.replace(".ndjson.gz", ".poll.json"))
    tmp_meta = meta_path.with_suffix(".json.tmp")
    tmp_meta.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp_meta, meta_path)
    return path


def poll_once(client: httpx.Client, *, write: bool = True) -> int:
    observed_at = _utc_now_iso()
    body, detail = fetch(client)
    if body is None:
        print(f"  ! gtfs-rt feed unavailable: {detail}", flush=True)
        if write:
            _write(observed_at, [], {"cycle_at": observed_at, "source": "vbb-gtfsrt",
                                     "status": "error", "detail": detail, "rows_kept": 0})
        return 0

    rows, stats = parse(body, observed_at, set(STOP_IDS))
    stats.update({"cycle_at": observed_at, "source": "vbb-gtfsrt", "status": "ok"})
    if write:
        _write(observed_at, rows, stats)
    pct = stats["rows_with_delay"] / stats["rows_kept"] if stats["rows_kept"] else 0
    print(f"[{observed_at}] gtfs-rt: {stats['feed_bytes']/1e6:.1f} MB, "
          f"{stats['trips_in_feed']:,} trips -> {len(rows)} rows at {len(STOP_IDS)} stops "
          f"({pct:.0%} with delay)", flush=True)
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--once", action="store_true", help="single fetch then exit")
    ap.add_argument("--stats", action="store_true", help="print only, write nothing")
    args = ap.parse_args()

    with httpx.Client(timeout=HTTP_TIMEOUT_S, headers={"User-Agent": USER_AGENT},
                      follow_redirects=True) as client:
        poll_once(client, write=not args.stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
