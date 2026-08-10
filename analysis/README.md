# Analysis — the hybrid strategy

Two data sources feed one project:

1. **Berlin U-Bahn / tram / bus, self-logged** — the *novel* dataset. BVG-operated,
   absent from every public historical delay dataset, ~2/3 of each poll. This is
   the headline result.
2. **Deutsche Bahn national rail, ready-made** — `piebro/deutsche-bahn-data` on
   Hugging Face (CC BY 4.0, 206M rows since 2024-07). Used to **build and
   validate the full modelling pipeline immediately**, with no waiting for data,
   and as a **full-scale fallback** if the self-logged set is thin by January.
   It also covers **Berlin S-Bahn** (see below), which the self-logged set
   overlaps — so it doubles as an independent check on our own measurements.

`rail_baseline.py` is the head start: it loads one month of the rail dataset,
settles what the delay column actually measures, defines the ground truth, and
computes the `train_type × hour × weekday` baseline against a naive "predict 0"
model. The same pipeline then transfers to the BVG data once enough has accrued.

Framing for the write-up: the rail dataset is prior work (Döllmann's territory);
the **Berlin U-Bahn/tram/bus angle is what's new**, plus the driver analysis.
Keep the two clearly separated in the report.

```
pip install "pandas>=2" pyarrow "huggingface_hub>=0.24"
python analysis/rail_baseline.py --schema-only --month 2025-01   # confirm columns first
python analysis/rail_baseline.py --month 2025-01                 # national baseline
python analysis/rail_baseline.py --month 2025-01 --berlin-only   # Berlin subset
```

### Baseline results (2025-01, run 2026-08-10)

`delay_in_min` was confirmed empirically to be the **departure** delay (100.0% exact
match, corr 1.000 against `departure_change_time − departure_planned_time`; arrival
only matches 61.6%). `*_change_time` is null for punctual trains, so it must be
filled with the planned time before subtracting — otherwise every on-time train
drops out and the measured delay is biased upward.

| Predictor | MAE (min) | RMSE (min) |
|---|---|---|
| predict 0 (naive) | 3.126 | 8.201 |
| global median | 2.897 | 7.879 |
| global mean | 3.720 | 7.597 |
| **group median** (type × hour × weekday) | **2.845** | 7.627 |
| **group mean** | 3.483 | **7.376** |

**The mean-baseline loses to predicting zero on MAE.** That is not a bug: MAE is
minimised by the median, RMSE by the mean, and the delay distribution is strongly
right-skewed (mean 3.20 min, median 1.00 min, 14.0% over 5 min). Hence both metrics
are reported everywhere, and each model is compared against the baseline that wins
on the *same* metric. Berlin-only subset: MAE 1.540 / RMSE 4.610 — Berlin trains are
markedly more punctual (mean 1.68 min, 5.6% over 5 min).

### Berlin S-Bahn is already in the rail dataset

296,163 rows of `train_type = S` at 11 Berlin stations in January 2025 alone
(Ostkreuz, Friedrichstraße, Hauptbahnhof, Südkreuz, Gesundbrunnen, Ostbahnhof,
Lichtenberg, Zoologischer Garten, Potsdamer Platz, Wannsee, Spandau), back to
2024-07. S-Bahn Berlin is a DB subsidiary, so it appears in DB's timetable API.
Zoologischer Garten and Hauptbahnhof are logged by us too — use the overlap to
validate the self-collected data against an independent source.

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
