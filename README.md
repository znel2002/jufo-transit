# jufo-transit — Berlin transit-delay prediction

**Jugend forscht 2027 project (Q), Fachgebiet Mathematik/Informatik.**

## Research question
Wie gut lassen sich Verspätungen im Berliner Nahverkehr aus offenen Echtzeit-
und Wetterdaten vorhersagen — und **welche Faktoren treiben sie tatsächlich?**

The prediction is the headline; the driver analysis (which factors actually
cause delay) is the research payload. Precedent: T. Döllmann won a Bavarian
Mathematik/Informatik Landessieg with the analogous rail project "Bahn-Vorhersage".

## Data strategy (hybrid)
- **Berlin U-Bahn / tram / bus, self-logged** — the *novel* dataset. These are
  BVG-operated and appear in no public historical delay dataset (only GTFS
  schedules), so the logger's months-long record is a genuine contribution. They
  are ~2/3 of every poll (bus 55, subway 46, tram 46 of 231 departures). Every
  week the logger runs adds data that can never be backfilled — hence it starts
  early.
- **Berlin S-Bahn — *not* novel, and used deliberately.** Corrected 2026-08-10:
  S-Bahn Berlin is a DB subsidiary, so it *is* already covered historically by
  `piebro/deutsche-bahn-data` (296,163 rows of `train_type = S` at 11 Berlin
  stations in January 2025 alone, back to 2024-07). Our logger records S-Bahn at
  Zoologischer Garten and Hauptbahnhof, both of which that dataset also covers —
  so the overlap becomes an **external validation** of the self-collected data
  against an independent source. That is worth more than the original claim was.
- **Deutsche Bahn rail, ready-made** — `piebro/deutsche-bahn-data` (Hugging Face,
  CC BY 4.0). Used to build/validate the modelling pipeline *now* and as a
  full-scale fallback. See `analysis/`.

## Hosting — free, no credit card
The logger runs on **GitHub Actions** (`.github/workflows/logger.yml`): a runner
polls every 15 min in `--out ndjson` mode and commits the day's file to
`data/observations/`. No paid host, no card. Those commits also keep the
scheduled workflow from being auto-disabled after 60 days idle. Trade-off: 5-min
minimum interval and best-effort timing, so occasional gaps appear in
`observed_at` — analysed, not hidden. **Keep the repo public** for unlimited
Actions minutes.

(An always-on machine / Pi or an Oracle always-free VM — the latter needs a card
for ID only — remain more reliable alternatives if ever wanted; the `--out sqlite`
backend + `scripts/transit-logger.service` cover that path.)

## Eigenanteil / disclosure (for the jury)
- **Mine:** the logging design, schema, feature engineering, models, and analysis.
- **Libraries:** `httpx` (HTTP), and at analysis time pandas/scikit-learn/LightGBM/SHAP.
- **Data source:** the public `v6.bvg.transport.rest` API (community-run mirror of
  VBB/BVG real-time data) and VBB open GTFS. No private or paid access.

## Layout
```
transit_logger/
  logger.py     # polls departures, records observations + archives raw JSON
  db.py         # SQLite schema & inserts (append-only observations + poll_log)
  stops.py      # the 4 logged interchange stops (IDs resolved 2026-08-05)
analysis/
  rail_baseline.py  # DB rail dataset: ground truth + the baseline every model must beat
  fetch_weather.py  # DWD hourly weather for Berlin (backfillable, fetched on demand)
  build_dataset.py  # transit x weather x calendar -> data/dataset.parquet
  calendar_de.py    # Berlin school holidays + German public holidays
scripts/
  healthcheck.py          # data sanity + freshness report (also the 48h verification)
  backup.sh               # daily gzipped DB snapshot (cron)
  transit-logger.service  # systemd unit for the VPS
docs/
  entscheidungen.md       # decision log -> becomes Methodik + Fehlerquellen in January
```

## Run locally
```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
./.venv/bin/python -m transit_logger.logger --once              # single poll -> sqlite
./.venv/bin/python -m transit_logger.logger --once --out ndjson # single poll -> ndjson
./.venv/bin/python scripts/healthcheck.py                       # verify (sqlite)
```

## Deploy (GitHub Actions — chosen, no card)
1. Push this repo to GitHub as a **public** repo (unlimited Actions minutes).
2. That's it — `.github/workflows/logger.yml` starts polling every 15 min and
   commits data to `data/observations/`. Trigger a first run manually from the
   Actions tab ("Run workflow") to confirm it commits.

## Deploy (always-on host — optional alternative)
```bash
# on a Pi / always-on machine / VM, in /opt/jufo-transit:
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
sudo cp scripts/transit-logger.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now transit-logger
journalctl -u transit-logger -f
crontab -e   # 5 3 * * * /opt/jufo-transit/scripts/backup.sh >> /var/log/jufo-backup.log 2>&1
```

## Data model
`observations` is **append-only** — one row per departure per poll, so we keep
the full evolution of each delay estimate. Analysis-time dedup on
`(trip_id, stop_id, planned_when)` keeps the observation closest to departure as
the "final" delay. `delay_s` is seconds (positive = late, negative = early, null
= no realtime yet). `poll_log` records every poll's outcome for gap/outage
monitoring.

## Status
- Logger verified end-to-end on both backends (one poll = ~231–242 departures across
  bus/S-Bahn/U-Bahn/tram/express/regional; delays and cancellations captured).
- **Deployed 2026-08-10**, collecting continuously via GitHub Actions.
- Storage migrated to one gzipped file per poll (2026-08-10) — see
  `docs/entscheidungen.md`.
- Rail baseline run on 2025-01: **MAE 2.845 min / RMSE 7.376 min** to beat. The
  delay column's meaning was confirmed empirically, and the original mean-baseline
  turned out to lose to "predict 0" on MAE (see `analysis/README.md`).
- Next: transit+weather+calendar join, then gradient-boosted trees and the driver
  analysis.
