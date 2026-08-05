"""BVG/VBB real-time departure logger for the Jugend forscht transit-delay project.

Polls the public v6.bvg.transport.rest departures endpoint for a small set of
high-traffic stops, once per cycle, and records every departure's planned vs.
estimated time and delay. Also archives the raw JSON response so nothing about
the original signal is ever lost.

Run it on a loop via cron/systemd (recommended, see scripts/) OR standalone with
--loop for quick local testing. Rate limit is 100 req/min; we poll a few stops
once a minute, far under the limit, with polite spacing between stops.

Usage:
    python -m transit_logger.logger --once        # single poll, then exit
    python -m transit_logger.logger --loop 60     # poll every 60s forever
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from .db import connect, insert_observations, log_poll
from .stops import STOPS, STOP_IDS

API_BASE = "https://v6.bvg.transport.rest"
# How far ahead to look and how many departures to pull per stop.
# 30 min / 60 results captures a stop's full board without paging.
DURATION_MIN = 30
RESULTS = 60
# Politeness: small gap between per-stop requests so we never burst the limit.
STOP_SPACING_S = 0.7
HTTP_TIMEOUT_S = 20

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DATA_DIR / "transit.db"
RAW_DIR = DATA_DIR / "raw"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _archive_raw(observed_at: str, stop_id: str, payload: object) -> None:
    """Write the raw response gzipped, partitioned by UTC day, for provenance."""
    day = observed_at[:10]
    out_dir = RAW_DIR / day
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = observed_at.replace(":", "").replace("-", "")
    with gzip.open(out_dir / f"{stop_id}_{stamp}.json.gz", "wt", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)


def _parse_departures(observed_at: str, stop_id: str, departures: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for d in departures:
        line = d.get("line") or {}
        rows.append(
            {
                "observed_at": observed_at,
                "stop_id": stop_id,
                "trip_id": d.get("tripId"),
                "line_name": line.get("name"),
                "product": line.get("product"),
                "direction": d.get("direction"),
                "planned_when": d.get("plannedWhen"),
                "when_est": d.get("when"),
                "delay_s": d.get("delay"),
                "cancelled": 1 if d.get("cancelled") else 0,
                "platform": d.get("platform"),
                "planned_platform": d.get("plannedPlatform"),
            }
        )
    return rows


def poll_once(conn, client: httpx.Client) -> int:
    """Poll every stop once. Returns total departures recorded this cycle."""
    total = 0
    for stop_id in STOP_IDS:
        observed_at = _utc_now_iso()
        try:
            resp = client.get(
                f"{API_BASE}/stops/{stop_id}/departures",
                params={"duration": DURATION_MIN, "results": RESULTS},
            )
            if resp.status_code != 200:
                log_poll(conn, observed_at, stop_id, "http_error", None,
                         f"status={resp.status_code}")
                continue
            payload = resp.json()
            departures = payload.get("departures", payload) if isinstance(payload, dict) else payload
            _archive_raw(observed_at, stop_id, payload)
            rows = _parse_departures(observed_at, stop_id, departures)
            insert_observations(conn, rows)
            log_poll(conn, observed_at, stop_id, "ok", len(rows))
            total += len(rows)
        except Exception as exc:  # never let one stop kill the cycle
            log_poll(conn, observed_at, stop_id, "exception", None, repr(exc)[:500])
        time.sleep(STOP_SPACING_S)
    return total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--once", action="store_true", help="single poll then exit (for cron)")
    g.add_argument("--loop", type=int, metavar="SECONDS",
                   help="poll every N seconds forever (for a supervised service)")
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = connect(DB_PATH)
    headers = {"User-Agent": "jufo-transit-delay-study/0.1 (student research project)"}

    with httpx.Client(timeout=HTTP_TIMEOUT_S, headers=headers) as client:
        if args.once:
            n = poll_once(conn, client)
            print(f"[{_utc_now_iso()}] logged {n} departures across {len(STOPS)} stops")
            return 0
        # --loop
        interval = args.loop
        print(f"[{_utc_now_iso()}] starting loop, every {interval}s, {len(STOPS)} stops")
        while True:
            start = time.monotonic()
            n = poll_once(conn, client)
            print(f"[{_utc_now_iso()}] logged {n} departures", flush=True)
            elapsed = time.monotonic() - start
            time.sleep(max(0.0, interval - elapsed))


if __name__ == "__main__":
    sys.exit(main())
