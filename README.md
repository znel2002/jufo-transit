# jufo-transit — Berlin transit-delay prediction

**Jugend forscht 2027 project (Q), Fachgebiet Mathematik/Informatik.**

## Research question
Wie gut lassen sich Verspätungen im Berliner Nahverkehr aus offenen Echtzeit-
und Wetterdaten vorhersagen — und **welche Faktoren treiben sie tatsächlich?**

The prediction is the headline; the driver analysis (which factors actually
cause delay) is the research payload. Precedent: T. Döllmann won a Bavarian
Mathematik/Informatik Landessieg with the analogous rail project "Bahn-Vorhersage".

## Why a self-collected dataset
No published Berlin delay dataset exists at this granularity. The core asset of
this project is a **months-long, self-logged** time series — which is why the
logger runs from early September, long before any modelling. Every week it runs
adds data that can never be backfilled.

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
scripts/
  healthcheck.py          # data sanity + freshness report (also the 48h verification)
  backup.sh               # daily gzipped DB snapshot (cron)
  transit-logger.service  # systemd unit for the VPS
```

## Run locally
```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
./.venv/bin/python -m transit_logger.logger --once     # single poll
./.venv/bin/python scripts/healthcheck.py              # verify
```

## Deploy on the VPS (recommended)
```bash
# on the box, in /opt/jufo-transit:
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
sudo cp scripts/transit-logger.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now transit-logger
journalctl -u transit-logger -f        # watch it
# nightly backup:
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
Logger written and **verified end-to-end 2026-08-05** (one poll = 225 departures
across bus/S-Bahn/U-Bahn/tram/express/regional; delays and cancellations captured
correctly). Next: deploy to the VPS so continuous collection starts.
