# jufo-transit — Berlin transit-delay prediction

**Jugend forscht 2027 project (Q), Fachgebiet Mathematik/Informatik.**

## Research question
Wie gut lassen sich Verspätungen im Berliner Nahverkehr aus offenen Echtzeit-
und Wetterdaten vorhersagen — und **welche Faktoren treiben sie tatsächlich?**

The prediction is the headline; the driver analysis (which factors actually
cause delay) is the research payload. Precedent: T. Döllmann won a Bavarian
Mathematik/Informatik Landessieg with the analogous rail project "Bahn-Vorhersage".

## Data strategy (hybrid)
- **Berlin local (BVG), self-logged** — the *novel* dataset. Confirmed 2026-08-05:
  no public historical BVG delay dataset exists (only GTFS schedules), so the
  logger's months-long record is a genuine contribution. Every week it runs adds
  data that can never be backfilled — hence it starts early.
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
scripts/
  healthcheck.py          # data sanity + freshness report (also the 48h verification)
  backup.sh               # daily gzipped DB snapshot (cron)
  transit-logger.service  # systemd unit for the VPS
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
Logger written and **verified end-to-end 2026-08-05**, both backends (one poll =
~225–232 departures across bus/S-Bahn/U-Bahn/tram/express/regional; delays and
cancellations captured correctly). GitHub Actions workflow ready. Next: push to a
public GitHub repo and trigger the first Actions run so continuous collection
starts.
