# Analysis — the hybrid strategy

Two data sources feed one project:

1. **Berlin local (BVG), self-logged** — the *novel* dataset. No public historical
   BVG delay dataset exists (only GTFS schedules), so the logger's months-long
   record is a genuine contribution. This is the headline result.
2. **Deutsche Bahn national rail, ready-made** — `piebro/deutsche-bahn-data` on
   Hugging Face (CC BY 4.0, 100M–1B rows since 2024-07). Used to **build and
   validate the full modelling pipeline immediately**, with no waiting for data,
   and as a **full-scale fallback** if the self-logged set is thin by January.

`rail_baseline.py` is the head start: it loads one month of the rail dataset,
defines the delay ground truth, and computes the `line × hour × weekday` baseline
MAE against a naive "predict 0" model. The same pipeline then transfers to the
BVG NDJSON data once enough has accrued.

Framing for the write-up: the rail dataset is prior work (Döllmann's territory);
the **Berlin local angle is what's new**, plus whatever driver analysis / modelling
finding comes out of it. Keep the two clearly separated in the report.

```
pip install "pandas>=2" pyarrow "huggingface_hub>=0.24"
python analysis/rail_baseline.py --schema-only --month 2025-01   # confirm columns first
python analysis/rail_baseline.py --month 2025-01                 # then the baseline
```

## Weather data — backfillable, so fetched on demand

The research question asks about delays "aus offenen Echtzeit- **und Wetterdaten**", so weather
is a required input. But it does **not** need a second continuous logger:

**DWD publishes station observations retroactively** (verified 2026-08-10: ~16-hour lag, and the
`/recent/` archive holds ~13,200 hourly rows ≈ 550 days). So weather for any past date can be
fetched at analysis time. That is the exact opposite of the BVG realtime feed, which is
unrecoverable if not captured live — hence: **log transit continuously, fetch weather in one pass.**

```bash
python analysis/fetch_weather.py --check                       # verify the feed
python analysis/fetch_weather.py --from 2026-08-10 --out data/weather.csv
```

**Station: 00433 Berlin-Tempelhof** — central and the longest record (since 1951). Alternatives:
00403 Dahlem, 00400 Buch, 00420 Marzahn, 00427 BER.
⚠️ **Berlin-Alexanderplatz (00399) and Berlin-Tegel (00430) are dead stations** (ended 2011 and
2021) — plausible names, no current data.

Variables pulled, chosen for delay relevance: temperature, humidity, precipitation amount,
**precipitation form** (6 = rain, 7 = snow, 8 = both — snow is the interesting one for delays),
wind speed and direction. Join to the transit observations on the hourly timestamp.
