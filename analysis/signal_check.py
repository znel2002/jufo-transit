"""Is the delay tail predictable from open data at all? Re-runnable checkpoint.

    python analysis/signal_check.py
    python analysis/signal_check.py --threshold 300     # "late" = 5 min instead of 3

WHY THIS EXISTS SEPARATELY FROM THE FULL MODEL
The target turned out to be extremely zero-inflated (74% of departures are exactly
0, and delay is quantised to whole minutes), so an MAE regression is close to
meaningless: predicting a constant zero already scores ~0.50 min. The honest task
is rare-event classification -- "will this departure be at least N minutes late?"
-- and the first question is simply whether ANY signal exists in openly available
features. That question deserves a cheap, repeatable answer rather than a guess.

Run it every few weeks. Two numbers decide the project's direction:
  * ROC-AUC / PR-AUC -- is the tail predictable, and does gradient boosting beat
    the trivial "this line at this hour is historically late" baseline?
  * the weather block in the driver table -- the research question promises weather
    as an input, and as of the summer data it contributes nothing. Autumn storms
    and winter ice are the real test, and they arrive before the January deadline.

Deliberately excludes every realtime field about the departure being predicted
(first_delay_s, delay_drift_s, n_obs, lead_time_s). Those belong to the separate
nowcast experiment; including them here would let the model echo the answer and
would make the driver analysis meaningless.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "dataset.parquet"

CAT = ["product", "line_name", "stop_id"]
NUM = ["hour", "weekday", "month", "day_of_year", "is_weekend", "is_school_holiday",
       "is_public_holiday", "is_morning_peak", "is_evening_peak",
       "is_full_traffic_day", "temp_c", "humidity_pct", "precip_mm", "wind_ms",
       "is_rain", "is_wet", "is_snow", "is_freezing"]
WEATHER = ["temp_c", "humidity_pct", "precip_mm", "wind_ms",
           "is_rain", "is_wet", "is_snow", "is_freezing"]


def load(threshold_s: int):
    df = pd.read_parquet(DATASET)
    df = df[(df.cancelled == 0) & df.final_delay_s.notna()].sort_values("planned_when")
    df["y"] = (df.final_delay_s >= threshold_s).astype(int)
    num = [c for c in NUM if c in df.columns]
    X = df[CAT + num].copy()
    for c in CAT:
        X[c] = X[c].astype("category")
    for c in num:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
    return df, X, df.y.values


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--threshold", type=int, default=180,
                    help="seconds of delay counted as 'late' (default 180)")
    ap.add_argument("--repeats", type=int, default=5,
                    help="permutation-importance repeats")
    ap.add_argument("--sweep", action="store_true",
                    help="compare thresholds first: severity vs predictability")
    args = ap.parse_args()

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import average_precision_score, roc_auc_score

    if args.sweep:
        # The central empirical result: severity and predictability rise together.
        # Sub-minute jitter is noise; serious disruption is systematic. Re-run as
        # data accumulates -- the >=10 min row rests on the fewest events and is
        # therefore the least settled.
        print(f"{'target':<18}{'pos rate':>9}{'events':>8}"
              f"{'GBM AUC':>9}{'lookup AUC':>11}{'gain':>7}")
        print("-" * 62)
        for th in (60, 180, 300, 600):
            d, X, y = load(th)
            if y.sum() < 150:
                print(f"delay >= {th//60:>2} min      too few events yet")
                continue
            cut = int(len(d) * 0.7)
            Xtr, Xte, ytr, yte = X[:cut], X[cut:], y[:cut], y[cut:]
            for c in CAT:
                Xte[c] = Xte[c].cat.set_categories(Xtr[c].cat.categories)
            mm = HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.06, random_state=0,
                categorical_features=[X.columns.get_loc(c) for c in CAT]).fit(Xtr, ytr)
            auc = roc_auc_score(yte, mm.predict_proba(Xte)[:, 1])
            tr, te = d.iloc[:cut], d.iloc[cut:]
            rate = tr.groupby(["line_name", "hour"], observed=True).y.mean()
            base = pd.Series(te.set_index(["line_name", "hour"]).index.map(rate).astype(float),
                             index=te.index).fillna(tr.y.mean()).values
            bauc = roc_auc_score(yte, base)
            print(f"delay >= {th//60:>2} min {y.mean():>13.2%}{y.sum():>8,}"
                  f"{auc:>9.3f}{bauc:>11.3f}{auc-bauc:>+7.3f}")
        print()

    df, X, y = load(args.threshold)
    print(f"target: delay >= {args.threshold//60} min")
    print(f"rows {len(df):,} | positives {y.mean():.2%} ({y.sum():,} events)")
    if y.sum() < 200:
        print("!! fewer than 200 positive events -- treat everything below as noise")

    # Time-ordered split. Never random: the same line recurs constantly, and a
    # random split would let the model see the future of its own test rows.
    cut = int(len(df) * 0.7)
    Xtr, Xte, ytr, yte = X[:cut], X[cut:], y[:cut], y[cut:]
    for c in CAT:
        Xte[c] = Xte[c].cat.set_categories(Xtr[c].cat.categories)
    print(f"train {len(Xtr):,} / test {len(Xte):,} "
          f"| test window {df.planned_when.iloc[cut]:%Y-%m-%d} -> "
          f"{df.planned_when.iloc[-1]:%Y-%m-%d} | positive rate {yte.mean():.2%}")

    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, random_state=0,
        categorical_features=[X.columns.get_loc(c) for c in CAT]).fit(Xtr, ytr)
    p = model.predict_proba(Xte)[:, 1]

    # Baseline with real explanatory power: how often is THIS line late at THIS
    # hour, historically? Beating chance is easy; beating this is the honest bar.
    tr, te = df.iloc[:cut], df.iloc[cut:]
    rate = tr.groupby(["line_name", "hour"], observed=True).y.mean()
    base = pd.Series(te.set_index(["line_name", "hour"]).index.map(rate).astype(float),
                     index=te.index).fillna(tr.y.mean()).values

    print(f"\n{'model':<34}{'ROC-AUC':>9}{'PR-AUC':>9}")
    print("-" * 52)
    print(f"{'always predict base rate':<34}{0.5:>9.3f}{yte.mean():>9.3f}")
    print(f"{'line x hour historical rate':<34}"
          f"{roc_auc_score(yte, base):>9.3f}{average_precision_score(yte, base):>9.3f}")
    print(f"{'gradient boosting (+weather/cal)':<34}"
          f"{roc_auc_score(yte, p):>9.3f}{average_precision_score(yte, p):>9.3f}")
    print(f"\nPR-AUC lift over chance: {average_precision_score(yte, p)/yte.mean():.1f}x")

    imp = pd.Series(
        permutation_importance(model, Xte, yte, scoring="roc_auc",
                               n_repeats=args.repeats, random_state=0,
                               n_jobs=2).importances_mean,
        index=X.columns).sort_values(ascending=False)
    print("\nDRIVER ANALYSIS - permutation importance (AUC drop when shuffled)")
    for k, v in imp.head(10).items():
        print(f"  {k:<22}{v:+7.4f}  {'#' * max(0, int(v * 400))}")

    w = imp[[c for c in WEATHER if c in imp.index]].sum()
    print(f"\nweather block combined: {w:+.4f}")
    print("  -> " + ("weather is contributing" if w > 0.005 else
                     "weather contributes nothing measurable yet. Expected while the "
                     "record is summer-only (no snow, no ice). This is THE number to "
                     "watch before the 2026-11-30 registration, because the research "
                     "question promises weather as an input."))


if __name__ == "__main__":
    main()
