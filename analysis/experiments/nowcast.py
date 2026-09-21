"""Experiment (B): how much is realtime data actually worth?

RESEARCH QUESTION
-----------------
Task (A) -- SCHEDULE-TIME prediction -- has already been run to death: predict
"will this departure be >= T seconds late?" from line/stop/calendar/weather
only, with no realtime information about the trip.  Result: a gradient-boosting
model is level with a trivial (line, hour) historical lookup at T = 180 s
(ROC-AUC 0.731 vs 0.732) and modestly ahead at T = 600 s (0.818 vs 0.788).

Task (B) -- NOWCAST -- was defined at the same time and never run.  It is the
identical prediction, but the model is additionally allowed to see the delay
that was reported for that trip the *first* time it was observed, up to ~65
minutes before the planned departure.

The gap between (A) and (B) is the actual result: it is a measurement of how
much the realtime feed is worth, in AUC, over knowing nothing but the timetable.
Nobody has measured that on this data.

WHAT COUNTS AS A LEGITIMATE NOWCAST FEATURE  (read this before trusting numbers)
--------------------------------------------------------------------------------
The dataset ships four "nowcast" columns.  They are NOT equally honest:

  first_delay_s   LEGITIMATE.  The delay reported at the first observation of
                  the trip.  Known well before departure (median ~52 min).
  first_lead_s    LEGITIMATE (derived here, = planned_when - first_observed_at).
                  How far ahead that first estimate was taken.  Known at the
                  moment you take the estimate.
  n_obs           HINDSIGHT.  How many times the trip was seen *in total*, i.e.
                  counted up to departure.  Not knowable 65 min ahead.
  lead_time_s     HINDSIGHT.  Measured from the LAST observation, which is by
                  definition the observation that produced the target.
  delay_drift_s   PURE LEAK.  It is exactly final_delay_s - first_delay_s.
                  first_delay_s + delay_drift_s reconstructs the target with
                  zero error.  Verified numerically in this script.

So the headline nowcast model uses schedule features + first_delay_s only.  The
hindsight and leaking variants are still run, explicitly labelled, because
showing *why* they are disqualified is part of the write-up.

THE MISSINGNESS TRAP
--------------------
first_delay_s is missing for ~37 % of rows: a trip first seen 65 minutes ahead
often has no realtime estimate attached yet.  That missingness is NOT random --
rows with a first estimate are >=180 s late 9.3 % of the time, rows without one
only 4.3 % of the time.  Silently dropping the missing rows would therefore
compare the nowcast against a *harder-to-predict* population than the
schedule-time model saw, and would flatter it badly.

This script therefore:
  * keeps every row, imputes first_delay_s and adds an explicit
    first_delay_missing indicator (the absence of an estimate is itself
    information a real system would have), and
  * ALSO reports every model on the sub-population where a first estimate
    exists, so the two are compared on identical rows.
Both tables are printed and labelled.

SPLIT
-----
Strictly time-ordered, first 70 % train / last 30 % test, sorted by
planned_when.  A random split would be catastrophically optimistic: the same
line at the same stop at the same minute recurs daily, so near-duplicate rows
would straddle the split.  Identical split for (A), (B) and the cancellation
task, so all numbers are comparable row for row.

Usage
-----
    ./.venv/bin/python analysis/experiments/nowcast.py
    ./.venv/bin/python analysis/experiments/nowcast.py --thresholds 180
    ./.venv/bin/python analysis/experiments/nowcast.py --skip-cancel
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import OrdinalEncoder

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

REPO = Path(__file__).resolve().parents[2]
DEFAULT_DATA = REPO / "data" / "dataset.parquet"
OUTDIR = Path(__file__).resolve().parent

# --------------------------------------------------------------------------
# Feature whitelists.  Hard allow-lists, not deny-lists, so a column added to
# the dataset later can never silently leak into a model.
# --------------------------------------------------------------------------
CAT_FEATURES = ["product", "line_name", "stop_id", "direction"]

CALENDAR_FEATURES = [
    "hour", "minute", "weekday", "month", "day_of_year",
    "is_weekend", "is_school_holiday", "is_public_holiday",
    "is_morning_peak", "is_evening_peak", "is_full_traffic_day",
]

WEATHER_FEATURES = [
    "temp_c", "humidity_pct", "precip_mm", "precip_form", "wind_ms",
    "wind_dir_deg", "is_rain", "is_wet", "is_snow", "is_freezing", "is_hot",
]

# (A) schedule time: nothing about this particular trip's realtime state.
SCHEDULE_NUM = CALENDAR_FEATURES + WEATHER_FEATURES

# (A-) same, minus day_of_year and month.  These two are *calendar indices*,
# not cyclical features: the test period lies entirely after the training
# period, so every test row falls off the right-hand end of every split the
# tree learned on them.  The tree then dumps the whole test set into whatever
# leaf the last training days produced.  On the cancellation target that
# single design choice moves the Brier score from 0.1114 (worse than a
# constant) to 0.0360 (better than a constant), so both variants are reported.
NOINDEX_NUM = [c for c in SCHEDULE_NUM if c not in ("day_of_year", "month")]

# (B) nowcast, honest version: + the first realtime estimate and how old it is.
NOWCAST_NUM = SCHEDULE_NUM + ["first_delay_s", "first_delay_missing", "first_lead_s"]

# (B') nowcast + observation-process features that are only knowable in
# hindsight.  Reported as a ceiling, not as the headline.
HINDSIGHT_NUM = NOWCAST_NUM + ["n_obs", "lead_time_s"]

# (B'') includes the algebraic leak.  Run only to show it is a leak.
LEAK_NUM = HINDSIGHT_NUM + ["delay_drift_s"]

# HGB config held fixed across every model so that (A) vs (B) differs only in
# the feature set, never in tuning effort.
HGB_KW = dict(
    max_iter=300,
    learning_rate=0.06,
    max_leaf_nodes=31,
    min_samples_leaf=40,
    l2_regularization=1.0,
    early_stopping=False,
    random_state=0,
)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    """ROC-AUC (ranking), PR-AUC (ranking under class imbalance), Brier
    (calibration).  PR-AUC is the honest headline: the positive rate is 7.4 %
    at 180 s and 1.6 % at 600 s, where ROC-AUC flatters everything."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    if y.sum() == 0 or y.sum() == len(y):
        return dict(n=len(y), pos=int(y.sum()), base=float(y.mean()),
                    roc=np.nan, pr=np.nan, brier=np.nan)
    return dict(
        n=len(y), pos=int(y.sum()), base=float(y.mean()),
        roc=float(roc_auc_score(y, p)),
        pr=float(average_precision_score(y, p)),
        brier=float(brier_score_loss(y, np.clip(p, 0, 1))),
    )


def load(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # first_lead_s: how far ahead of the planned departure the FIRST estimate
    # was taken.  The shipped lead_time_s is measured from the LAST observation
    # (the one that produced the target), so it is useless as a forecast
    # horizon; this is the column task 2 actually needs.
    df["first_lead_s"] = (df["planned_when"] - df["first_observed_at"]).dt.total_seconds()
    df["first_delay_missing"] = df["first_delay_s"].isna().astype(int)
    df = df.sort_values("planned_when", kind="mergesort").reset_index(drop=True)
    return df


def time_split(df: pd.DataFrame, frac: float = 0.70):
    cut = int(len(df) * frac)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


MAX_CARD = 254  # HGB refuses native categoricals with cardinality > 255


def _cap(tr: pd.Series, te: pd.Series):
    """HGB caps native categoricals at 255 levels; line_name has 275 and
    direction 274 once cancelled rows are included.  Keep the MAX_CARD most
    frequent levels *as seen in TRAIN* and fold the rest into one bucket -- the
    tail is made of lines with a handful of departures each, which a tree
    cannot learn a rate for anyway.  Train-only frequencies, so this is not a
    route for test information to leak back."""
    tr, te = tr.astype(str), te.astype(str)
    keep = set(tr.value_counts().index[:MAX_CARD])
    return (tr.where(tr.isin(keep), "__RARE__"),
            te.where(te.isin(keep), "__RARE__"))


def encode(train: pd.DataFrame, test: pd.DataFrame, num_cols: list[str]):
    """Ordinal-encode the categoricals (HGB treats them natively) and hand the
    numerics through untouched; HGB handles NaN internally, so the imputation
    of first_delay_s is 'learned split direction', paired with the explicit
    missing indicator."""
    tr_c, te_c = {}, {}
    for c in CAT_FEATURES:
        tr_c[c], te_c[c] = _cap(train[c], test[c])
    tr_c, te_c = pd.DataFrame(tr_c), pd.DataFrame(te_c)
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                         encoded_missing_value=-1)
    tr_cat = enc.fit_transform(tr_c)
    te_cat = enc.transform(te_c)
    tr = np.hstack([tr_cat, train[num_cols].astype(float).to_numpy()])
    te = np.hstack([te_cat, test[num_cols].astype(float).to_numpy()])
    cat_mask = np.array([True] * len(CAT_FEATURES) + [False] * len(num_cols))
    return tr, te, cat_mask


def fit_hgb(train, test, y_tr, num_cols):
    Xtr, Xte, cat_mask = encode(train, test, num_cols)
    clf = HistGradientBoostingClassifier(categorical_features=cat_mask, **HGB_KW)
    clf.fit(Xtr, y_tr)
    return clf.predict_proba(Xte)[:, 1]


def lookup_line_hour(train, test, y_tr, min_n: int = 1):
    """The trivial reference model: historical positive rate per (line, hour),
    computed on TRAIN only, global train rate for unseen combinations.  This is
    the thing model (A) failed to beat, so it stays in every table."""
    t = train[["line_name", "hour"]].copy()
    t["y"] = y_tr
    g = t.groupby(["line_name", "hour"], observed=True)["y"].agg(["mean", "size"])
    g = g[g["size"] >= min_n]["mean"]
    key = pd.MultiIndex.from_arrays([test["line_name"], test["hour"]])
    return g.reindex(key).to_numpy(dtype=float), float(np.mean(y_tr))


def lookup_first_delay(train, test, y_tr):
    """Task 3's control: the single feature first_delay_s turned into a
    probability by a train-only lookup on whole minutes (the target is
    quantised to minutes anyway).  If the nowcast model is merely echoing
    first_delay_s, this row will match it."""
    b_tr = np.where(train["first_delay_s"].isna(), np.nan,
                    np.clip(train["first_delay_s"] / 60.0, -5, 30).round())
    b_te = np.where(test["first_delay_s"].isna(), np.nan,
                    np.clip(test["first_delay_s"] / 60.0, -5, 30).round())
    t = pd.DataFrame({"b": b_tr, "y": y_tr})
    rate = t.groupby("b", dropna=False)["y"].agg(["mean", "size"])
    # 'missing' is its own bucket: no estimate existing is informative in itself.
    miss_rate = float(t.loc[t["b"].isna(), "y"].mean()) if t["b"].isna().any() else float(np.mean(y_tr))
    m = rate["mean"].to_dict()
    out = np.array([m.get(b, np.nan) if not np.isnan(b) else miss_rate for b in b_te])
    return out


def fill(p: np.ndarray, default: float) -> np.ndarray:
    p = np.asarray(p, dtype=float).copy()
    p[~np.isfinite(p)] = default
    return p


def row(name, y, p, mask=None):
    if mask is not None:
        y, p = np.asarray(y)[mask], np.asarray(p)[mask]
    r = metrics(y, p)
    r["model"] = name
    return r


def show(rows, title):
    df = pd.DataFrame(rows)[["model", "n", "pos", "base", "roc", "pr", "brier"]]
    print(f"\n{title}")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return df


# --------------------------------------------------------------------------
# main experiment
# --------------------------------------------------------------------------
def run_threshold(train, test, thr, results):
    y_tr = (train["final_delay_s"] >= thr).astype(int).to_numpy()
    y_te = (test["final_delay_s"] >= thr).astype(int).to_numpy()
    print(f"\n{'='*78}\nTHRESHOLD >= {thr} s   train n={len(train)} pos={y_tr.sum()} "
          f"({y_tr.mean():.4f})   test n={len(test)} pos={y_te.sum()} ({y_te.mean():.4f})\n{'='*78}")

    preds = {}
    lk, glob = lookup_line_hour(train, test, y_tr)
    preds["baseline (line,hour) lookup"] = fill(lk, glob)
    preds["first_delay_s lookup (1 feature)"] = fill(lookup_first_delay(train, test, y_tr), glob)

    preds["constant train base rate"] = np.full(len(test), float(np.mean(y_tr)))

    for label, cols in [
        ("(A) schedule-time HGB", SCHEDULE_NUM),
        ("(A-) schedule, no day_of_year/month", NOINDEX_NUM),
        ("(B) nowcast HGB [first_delay]", NOWCAST_NUM),
        ("(B') + hindsight n_obs/lead_time", HINDSIGHT_NUM),
        ("(B'') + delay_drift_s [LEAK]", LEAK_NUM),
    ]:
        t0 = time.time()
        preds[label] = fit_hgb(train, test, y_tr, cols)
        print(f"  fitted {label:34s} in {time.time()-t0:5.1f}s", flush=True)

    have_first = test["first_delay_s"].notna().to_numpy()

    full = show([row(k, y_te, v) for k, v in preds.items()],
                f"[{thr}s] FULL TEST SET (missing first_delay kept, imputed + flagged)")
    sub = show([row(k, y_te, v, have_first) for k, v in preds.items()],
               f"[{thr}s] SUBSET WITH A FIRST ESTIMATE (n={have_first.sum()}) "
               f"- same rows for every model")
    full["population"], sub["population"] = "full_test", "has_first_delay"
    full["threshold"], sub["threshold"] = thr, thr
    results.append(pd.concat([full, sub]))

    # ---- task 2: does the nowcast's edge decay with forecast horizon? -------
    # Bucket by first_lead_s = how long BEFORE departure the first estimate was
    # taken.  A bucket at <=0 s means the "first" estimate was only recorded at
    # or after the planned departure -- that is not a forecast at all, and it
    # is where any spuriously good nowcast number will hide.
    edges = [-np.inf, 0, 600, 1200, 1800, 2400, 3000, 3600, np.inf]
    labels = ["<=0 (not a forecast)", "0-10m", "10-20m", "20-30m",
              "30-40m", "40-50m", "50-60m", ">60m"]
    lead_bucket = pd.cut(test["first_lead_s"], bins=edges, labels=labels)
    dec = []
    for lab in labels:
        m = (lead_bucket == lab).to_numpy() & have_first
        if m.sum() < 300:
            continue
        r = {"threshold": thr, "lead_bucket": lab, "n": int(m.sum()),
             "pos": int(y_te[m].sum()), "base": float(y_te[m].mean())}
        for k, short in [("baseline (line,hour) lookup", "baseline"),
                         ("(A) schedule-time HGB", "A_sched"),
                         ("(B) nowcast HGB [first_delay]", "B_nowcast"),
                         ("first_delay_s lookup (1 feature)", "first_only")]:
            mm = metrics(y_te[m], preds[k][m])
            r[f"roc_{short}"] = mm["roc"]
            r[f"pr_{short}"] = mm["pr"]
        r["roc_gain_B_minus_A"] = r["roc_B_nowcast"] - r["roc_A_sched"]
        dec.append(r)
    ddf = pd.DataFrame(dec)
    print(f"\n[{thr}s] LEAD-TIME DECAY (rows with a first estimate, bucketed by "
          f"how far ahead it was taken)")
    print(ddf.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    ddf.to_csv(OUTDIR / f"nowcast_leaddecay_{thr}s.csv", index=False)
    return ddf


def run_cancellation(df, results):
    """Separate target: is a departure CANCELLED?  Cancellations may be far
    more systematic than delays (planned diversions, line closures, a whole
    day of works), and unlike delay this is schedule-time predictable or it is
    nothing -- no realtime column applies."""
    d = df.copy()  # every row, including cancelled ones (obviously)
    y = d["cancelled"].astype(int)
    train, test = time_split(d)
    y_tr = y.iloc[:len(train)].to_numpy()
    y_te = y.iloc[len(train):].to_numpy()
    print(f"\n{'='*78}\nCANCELLATION  total n={len(d)} pos={int(y.sum())} "
          f"({y.mean():.4f})\n  train pos={y_tr.sum()} ({y_tr.mean():.4f})   "
          f"test pos={y_te.sum()} ({y_te.mean():.4f})\n{'='*78}")
    lk, glob = lookup_line_hour(train, test, y_tr)
    preds = {"baseline (line,hour) lookup": fill(lk, glob)}
    # a second trivial lookup: cancellations are plausibly a per-line-per-DAY
    # phenomenon, but day is unknowable ahead; (line, stop) is not.
    t = train[["line_name", "stop_id"]].copy(); t["y"] = y_tr
    g = t.groupby(["line_name", "stop_id"], observed=True)["y"].mean()
    key = pd.MultiIndex.from_arrays([test["line_name"], test["stop_id"]])
    preds["baseline (line,stop) lookup"] = fill(g.reindex(key).to_numpy(dtype=float), glob)
    preds["constant train base rate"] = np.full(len(test), float(np.mean(y_tr)))
    for label, cols in [("schedule-time HGB", SCHEDULE_NUM),
                        ("schedule-time HGB, no day_of_year/month", NOINDEX_NUM)]:
        t0 = time.time()
        preds[label] = fit_hgb(train, test, y_tr, cols)
        print(f"  fitted {label:40s} in {time.time()-t0:.1f}s  "
              f"mean_pred={preds[label].mean():.4f} (actual {y_te.mean():.4f})", flush=True)
    out = show([row(k, y_te, v) for k, v in preds.items()],
               "CANCELLATION, schedule-time features only, same 70/30 time split")
    out["population"] = "all_rows"; out["threshold"] = "cancelled"
    results.append(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--thresholds", type=int, nargs="*", default=[180, 600])
    ap.add_argument("--skip-cancel", action="store_true")
    args = ap.parse_args()

    df = load(args.data)
    print(f"loaded {len(df)} rows  {df.planned_when.min()} .. {df.planned_when.max()}")

    # Verify the leak claim numerically rather than asserting it in prose.
    chk = df.dropna(subset=["delay_drift_s", "first_delay_s", "final_delay_s"])
    err = (chk["delay_drift_s"] - (chk["final_delay_s"] - chk["first_delay_s"])).abs().max()
    print(f"LEAK CHECK: max |delay_drift_s - (final - first)| = {err:.6f} over "
          f"{len(chk)} rows  -> delay_drift_s reconstructs the target exactly.")

    model = df.query("cancelled == 0 and final_delay_s.notna()").reset_index(drop=True)
    print(f"delay-modelling rows after filter: {len(model)}  "
          f"first_delay_s missing: {model.first_delay_s.isna().mean():.4f}")
    print("MISSINGNESS IS NOT RANDOM: P(>=180s) = "
          f"{(model.loc[model.first_delay_s.notna(),'final_delay_s']>=180).mean():.4f} "
          "with a first estimate vs "
          f"{(model.loc[model.first_delay_s.isna(),'final_delay_s']>=180).mean():.4f} without.")
    print("first_lead_s (planned - first_observed) quantiles [s]: "
          + ", ".join(f"{q}%={model.first_lead_s.quantile(q/100):.0f}"
                      for q in (5, 25, 50, 75, 95)))

    train, test = time_split(model)
    print(f"split: train {len(train)} (<= {train.planned_when.max()}), "
          f"test {len(test)} (>= {test.planned_when.min()})")

    results = []
    for thr in args.thresholds:
        run_threshold(train, test, thr, results)
    if not args.skip_cancel:
        run_cancellation(df, results)

    allres = pd.concat(results, ignore_index=True)
    allres.to_csv(OUTDIR / "nowcast_results.csv", index=False)
    print(f"\nwrote {OUTDIR/'nowcast_results.csv'}")


if __name__ == "__main__":
    main()
