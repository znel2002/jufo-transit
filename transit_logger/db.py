"""SQLite storage for the transit-delay logger.

Design notes (the reasons matter for the Jugend forscht write-up):

* Append-only *observations*: every poll writes one row per departure seen, so we
  keep the full history of how a delay estimate evolves as the vehicle approaches.
  This is deliberately NOT deduplicated at write time -- the raw signal is data.
* A separate `latest` view/dedup happens at analysis time on
  (trip_id, stop_id, planned_when), keeping the observation closest to departure.
  Doing it in analysis rather than at write time means we can also *study* how the
  prognosis firms up, which is itself an interesting sub-question.
* Raw JSON of each poll is archived to disk (see logger.py), not the DB, to keep
  the DB queryable and small while never losing the original response.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS observations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    observed_at   TEXT NOT NULL,   -- ISO8601 UTC, when we polled
    stop_id       TEXT NOT NULL,
    trip_id       TEXT,
    line_name     TEXT,
    product       TEXT,            -- subway / suburban / tram / bus / regional / express
    direction     TEXT,
    planned_when  TEXT,            -- ISO8601, scheduled departure
    when_est      TEXT,            -- ISO8601, current best estimate (null if cancelled)
    delay_s       INTEGER,         -- seconds; positive = late, can be null
    cancelled     INTEGER NOT NULL DEFAULT 0,
    platform      TEXT,
    planned_platform TEXT
);
-- Fast lookups for the analysis-time dedup and for gap monitoring.
CREATE INDEX IF NOT EXISTS idx_obs_trip ON observations(trip_id, stop_id, planned_when);
CREATE INDEX IF NOT EXISTS idx_obs_observed ON observations(observed_at);

CREATE TABLE IF NOT EXISTS poll_log (
    observed_at   TEXT NOT NULL,
    stop_id       TEXT NOT NULL,
    status        TEXT NOT NULL,   -- ok / http_error / exception
    n_departures  INTEGER,
    detail        TEXT
);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.execute("PRAGMA journal_mode=WAL;")      # concurrent read while writing
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.executescript(SCHEMA)
    return conn


def insert_observations(conn: sqlite3.Connection, rows: list[dict]) -> None:
    conn.executemany(
        """INSERT INTO observations
           (observed_at, stop_id, trip_id, line_name, product, direction,
            planned_when, when_est, delay_s, cancelled, platform, planned_platform)
           VALUES
           (:observed_at, :stop_id, :trip_id, :line_name, :product, :direction,
            :planned_when, :when_est, :delay_s, :cancelled, :platform, :planned_platform)""",
        rows,
    )
    conn.commit()


def log_poll(conn: sqlite3.Connection, observed_at: str, stop_id: str,
             status: str, n_departures: int | None, detail: str = "") -> None:
    conn.execute(
        "INSERT INTO poll_log (observed_at, stop_id, status, n_departures, detail) "
        "VALUES (?,?,?,?,?)",
        (observed_at, stop_id, status, n_departures, detail),
    )
    conn.commit()
