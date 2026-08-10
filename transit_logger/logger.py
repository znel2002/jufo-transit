"""BVG/VBB real-time departure logger for the Jugend forscht transit-delay project.

Polls the public v6.bvg.transport.rest departures endpoint for a small set of
high-traffic stops and records every departure's planned vs. estimated time and
delay. Two storage backends, chosen by where it runs:

* ``--out sqlite`` (default) — append to a local SQLite DB, and archive the raw
  JSON of each poll. Best for an always-on host (own machine / Pi / VPS) via
  ``--loop``.
* ``--out ndjson`` — write each poll cycle to its own gzipped newline-delimited JSON
  file under ``data/observations/<UTC day>/``. This is the **GitHub Actions**
  backend: a runner polls once, writes one new file, and the workflow commits it
  back to the repo, so no persistent disk is needed and no credit card / paid host
  is involved. One file per cycle (rather than appending to a per-day file) keeps
  the git history small and makes concurrent runs conflict-free.

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
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from .db import connect, insert_observations, log_poll
from .stops import STOPS, STOP_IDS

API_BASE = "https://v6.bvg.transport.rest"
# Look-ahead window per stop. Raised from 30 to 65 min on 2026-08-10: GitHub was
# dropping most */15 slots and effectively polling hourly, so a 30-minute window
# left departures that fell between two polls completely unobserved. A window
# longer than the worst-case poll interval guarantees every departure is seen at
# least once, even when a scheduled run is skipped.
DURATION_MIN = 65
# Must comfortably exceed the departures a single stop returns in DURATION_MIN --
# roughly 58 per stop per 30 min at these interchanges, so ~125 for 65 min. If this
# cap is ever hit the data is silently truncated, which is why poll_once warns.
RESULTS = 250
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


def _write_ndjson(observed_at: str, rows: list[dict]) -> Path:
    """Write one poll cycle to its own gzipped NDJSON file (the CI backend).

    One write-once file per cycle, partitioned by UTC day:
        data/observations/2026-08-10/2026-08-10T151719Z.ndjson.gz

    Deliberately *not* appended to a per-day file. Appending would make git store a
    full new blob of the whole day on every one of the ~96 daily commits (~1 GB
    working tree by February), and would make concurrent runs touch the same path,
    which is the only reason the workflow ever needed a rebase before pushing.
    Unique filenames make write conflicts structurally impossible.
    """
    out_dir = NDJSON_DIR / observed_at[:10]
    out_dir.mkdir(parents=True, exist_ok=True)
    # Second precision is unambiguous at any sane poll interval; the suffix guard
    # only ever matters if two cycles somehow start within the same second.
    stamp = observed_at[:19].replace(":", "").replace("-", "") + "Z"
    path = out_dir / f"{stamp}.ndjson.gz"
    n = 1
    while path.exists():
        path = out_dir / f"{stamp}_{n}.ndjson.gz"
        n += 1
    # Write-then-rename: the CI job runs the logger under `timeout`, so the process
    # can be killed mid-write. A half-written .gz would be an unreadable hole in the
    # record; an atomic rename means a cycle is either complete or absent.
    tmp = path.with_suffix(".gz.tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    return path


def poll_once(client: httpx.Client, *, out: str, conn=None, archive_raw: bool = True) -> int:
    """Poll every stop once. Returns total departures recorded this cycle.

    ``out`` is "sqlite" or "ndjson". Failures on one stop never kill the cycle.
    """
    total = 0
    cycle_at = _utc_now_iso()      # names the cycle's ndjson file
    cycle_rows: list[dict] = []    # all stops of this cycle, written in one go
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
            if len(rows) >= RESULTS:
                # Hitting the cap means the API truncated the window and we are
                # silently losing departures -- raise RESULTS if this ever appears.
                print(f"  ! stop {stop_id} returned {len(rows)} rows, at the "
                      f"RESULTS={RESULTS} cap: data may be truncated", flush=True)

            if out == "sqlite":
                if archive_raw:
                    _archive_raw(observed_at, stop_id, payload)
                insert_observations(conn, rows)
                log_poll(conn, observed_at, stop_id, "ok", len(rows))
            else:  # ndjson
                cycle_rows.extend(rows)
            total += len(rows)
        except Exception as exc:  # never let one stop kill the cycle
            if conn is not None:
                log_poll(conn, observed_at, stop_id, "exception", None, repr(exc)[:500])
        time.sleep(STOP_SPACING_S)

    if out == "ndjson" and cycle_rows:
        _write_ndjson(cycle_at, cycle_rows)
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
