"""BVG/VBB real-time departure logger for the Jugend forscht transit-delay project.

Polls the public v6.bvg.transport.rest departures endpoint for a small set of
high-traffic stops and records every departure's planned vs. estimated time and
delay. Two storage backends, chosen by where it runs:

* ``--out sqlite`` (default) — append to a local SQLite DB, and archive the raw
  JSON of each poll. Best for an always-on host (own machine / Pi / VPS) via
  ``--loop``.
* ``--out ndjson`` — append parsed rows to a per-day newline-delimited JSON file
  under ``data/observations/``. This is the **GitHub Actions** backend: a runner
  polls once, appends the file, and the workflow commits it back to the repo, so
  no persistent disk is needed and no credit card / paid host is involved.

Rate limit is 100 req/min; we poll a few stops per run, far under the limit.

Usage:
    python -m transit_logger.logger --once                     # one poll -> sqlite
    python -m transit_logger.logger --once --out ndjson        # one poll -> ndjson (CI)
    python -m transit_logger.logger --loop 60                  # every 60s -> sqlite
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
DURATION_MIN = 30          # look-ahead window per stop
RESULTS = 60               # max departures per stop
STOP_SPACING_S = 0.7       # polite gap between per-stop requests
HTTP_TIMEOUT_S = 20

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DATA_DIR / "transit.db"
RAW_DIR = DATA_DIR / "raw"
NDJSON_DIR = DATA_DIR / "observations"


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


def _append_ndjson(observed_at: str, rows: list[dict]) -> None:
    """Append parsed rows to a per-UTC-day NDJSON file (the CI backend)."""
    NDJSON_DIR.mkdir(parents=True, exist_ok=True)
    path = NDJSON_DIR / f"{observed_at[:10]}.ndjson"
    with path.open("a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def poll_once(client: httpx.Client, *, out: str, conn=None, archive_raw: bool = True) -> int:
    """Poll every stop once. Returns total departures recorded this cycle.

    ``out`` is "sqlite" or "ndjson". Failures on one stop never kill the cycle.
    """
    total = 0
    for stop_id in STOP_IDS:
        observed_at = _utc_now_iso()
        try:
            resp = client.get(
                f"{API_BASE}/stops/{stop_id}/departures",
                params={"duration": DURATION_MIN, "results": RESULTS},
            )
            if resp.status_code != 200:
                if conn is not None:
                    log_poll(conn, observed_at, stop_id, "http_error", None,
                             f"status={resp.status_code}")
                continue
            payload = resp.json()
            departures = payload.get("departures", payload) if isinstance(payload, dict) else payload
            rows = _parse_departures(observed_at, stop_id, departures)

            if out == "sqlite":
                if archive_raw:
                    _archive_raw(observed_at, stop_id, payload)
                insert_observations(conn, rows)
                log_poll(conn, observed_at, stop_id, "ok", len(rows))
            else:  # ndjson
                _append_ndjson(observed_at, rows)
            total += len(rows)
        except Exception as exc:  # never let one stop kill the cycle
            if conn is not None:
                log_poll(conn, observed_at, stop_id, "exception", None, repr(exc)[:500])
        time.sleep(STOP_SPACING_S)
    return total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--once", action="store_true", help="single poll then exit (cron / CI)")
    g.add_argument("--loop", type=int, metavar="SECONDS",
                   help="poll every N seconds forever (supervised service)")
    ap.add_argument("--out", choices=["sqlite", "ndjson"], default="sqlite",
                    help="storage backend (default: sqlite; use ndjson on GitHub Actions)")
    ap.add_argument("--no-raw", action="store_true",
                    help="skip archiving raw responses (sqlite backend only)")
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = connect(DB_PATH) if args.out == "sqlite" else None
    archive_raw = not args.no_raw
    headers = {"User-Agent": "jufo-transit-delay-study/0.1 (student research project)"}

    with httpx.Client(timeout=HTTP_TIMEOUT_S, headers=headers) as client:
        if args.once:
            n = poll_once(client, out=args.out, conn=conn, archive_raw=archive_raw)
            print(f"[{_utc_now_iso()}] logged {n} departures ({args.out}) across {len(STOPS)} stops")
            return 0
        interval = args.loop
        print(f"[{_utc_now_iso()}] starting loop, every {interval}s, {len(STOPS)} stops -> {args.out}")
        while True:
            start = time.monotonic()
            n = poll_once(client, out=args.out, conn=conn, archive_raw=archive_raw)
            print(f"[{_utc_now_iso()}] logged {n} departures", flush=True)
            time.sleep(max(0.0, interval - (time.monotonic() - start)))


if __name__ == "__main__":
    sys.exit(main())
