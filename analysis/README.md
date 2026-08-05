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
