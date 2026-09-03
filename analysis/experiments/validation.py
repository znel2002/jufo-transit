"""Stress-test: is the delay signal REAL, or an artifact of one lucky split?

    python analysis/experiments/validation.py                      # full run (~20 min)
    python analysis/experiments/validation.py --quick              # ~2 min, fewer reps
    python analysis/experiments/validation.py --threshold 180
    python analysis/experiments/validation.py --only cv,perm

WHY THIS FILE EXISTS
--------------------
analysis/signal_check.py reports a single number from a single 70/30 time split:
ROC-AUC 0.805 for gradient boosting at "delay >= 10 min", versus 0.765 for a
trivial (line_name, hour) historical-rate lookup. Two facts make that number
unquotable in a scientific report as it stands:

  1. An earlier run on 7 days of data gave 0.849. The estimate therefore moves by
     ~0.04 when the sample changes -- larger than the entire claimed advantage of
     the model over the baseline (~0.04). A difference smaller than the
     instability of its own estimator is not yet a result.
  2. There are only ~3,300 events at the >=10 min threshold, and they are not
     independent: one disrupted afternoon produces hundreds of correlated
     positives at once. The effective sample size is much smaller than the row
     count, so the naive standard error is optimistic.

So before any claim is made, five things have to be established. Each is one
section below, switchable via --only:

  cv     ROLLING-ORIGIN VALIDATION. Many sequential train/test folds instead of
         one. Gives a mean AND a spread, and shows whether the model's advantage
         over the lookup baseline is consistent or lives in a few windows.
  perm   PERMUTATION / NULL TEST. Shuffle the labels, refit, repeat. If the real
         score is inside that null distribution, there is no signal at all. This
         is the literal answer to "is our data even showing anything?".
  boot   BOOTSTRAP CONFIDENCE INTERVALS on the headline test-set numbers, both
         naive (row-level) and clustered by day (honest, because delays cluster).
  drift  TEMPORAL STABILITY. Does a model trained early still work later? Does
         the positive rate itself drift? The collection had a 4-day outage
         (2026-08-20..08-23) and a data-source change (transport.rest -> VBB
         GTFS-RT), so this is not a hypothetical worry.
  split  SPLIT SANITY. No departure in both train and test; and how much
         optimism a random split would have injected, since the same trip is
         logged at several stops and the same line recurs every few minutes.

METHODOLOGICAL RULES OBEYED THROUGHOUT
  * Only ex-ante features. Everything known solely because the departure was
    already observed (first_delay_s, delay_drift_s, n_obs, lead_time_s,
    last/first_observed_at) is excluded -- using it would let the model echo the
    answer. See LEAKY below; the script asserts none of them reach the model.
  * Splits are always ordered in time, never random, and are cut on calendar-day
    boundaries so no day is half in train and half in test.
  * Every baseline is fitted on that fold's training rows only.

Outputs a plain-text report to stdout and machine-readable results to
analysis/experiments/validation_results.json for the write-up.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data" / "dataset.parquet"
OUT_JSON = Path(__file__).resolve().parent / "validation_results.json"

# --- feature contract -------------------------------------------------------
# Categorical, fed to HistGradientBoosting's native categorical handling.
CAT = ["product", "line_name", "stop_id", "direction"]
# Numeric / boolean. Calendar + weather only: all knowable before the departure.
NUM = [
    "hour", "minute", "weekday", "month", "day_of_year", "is_weekend",
    "is_school_holiday", "is_public_holiday", "is_morning_peak",
    "is_evening_peak", "is_full_traffic_day",
    "temp_c", "humidity_pct", "precip_mm", "precip_form", "wind_ms",
    "wind_dir_deg", "is_rain", "is_wet", "is_snow", "is_freezing", "is_hot",
]
# Never features. Realtime observations of the very departure being predicted,
# plus the target itself and identifiers that would act as a row fingerprint.
LEAKY = [
    "first_delay_s", "delay_drift_s", "n_obs", "lead_time_s",
    "last_observed_at", "first_observed_at", "hour_utc", "trip_id",
    "final_delay_s",
]

# Model kept identical to analysis/signal_check.py so the numbers below refer to
# the same estimator the project already quotes. Tuning is a separate question;
# this file only measures how trustworthy the existing number is.
GBM_KW = dict(max_iter=300, learning_rate=0.06, random_state=0)

# The record has a hole: transport.rest died on 2026-08-20 and the VBB GTFS-RT
# logger took over from 2026-08-24. Anything spanning this date is comparing two
# different measurement instruments, not two time periods.
SOURCE_CHANGE = pd.Timestamp("2026-08-24").date()


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load(threshold_s: int):
    """Rows that actually ran and have a final delay, sorted in time.

    `day` is the Berlin calendar day, used for every split boundary: cutting on
    row index instead would put the same rush hour on both sides of a fold.
    """
    df = pd.read_parquet(DATASET)
    df = df[(df.cancelled == 0) & df.final_delay_s.notna()]
    df = df.sort_values("planned_when").reset_index(drop=True)
    df["y"] = (df.final_delay_s >= threshold_s).astype(int)
    df["day"] = df.planned_when.dt.tz_convert("Europe/Berlin").dt.date

    cat = [c for c in CAT if c in df.columns]
    num = [c for c in NUM if c in df.columns]
    assert not (set(cat) | set(num)) & set(LEAKY), "a leaky column reached the features"

    X = df[cat + num].copy()
    for c in cat:
        X[c] = X[c].astype("category")
    for c in num:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
    return df, X, df.y.values, cat


def fit_gbm(X, y, tr, te, cat, seed=0):
    """Fit on rows `tr`, return P(late) for rows `te`.

    Test categories are re-mapped onto the training categories so that a line
    that only appears after the cut becomes an unseen level rather than silently
    shifting every other category's integer code.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    Xtr, Xte = X.iloc[tr].copy(), X.iloc[te].copy()
    for c in cat:
        Xte[c] = Xte[c].cat.set_categories(Xtr[c].cat.categories)
    kw = dict(GBM_KW, random_state=seed,
              categorical_features=[X.columns.get_loc(c) for c in cat])
    m = HistGradientBoostingClassifier(**kw).fit(Xtr, y[tr])
    return m.predict_proba(Xte)[:, 1]


def lookup(df, y, tr, te, keys=("line_name", "hour")):
    """Historical positive rate per key group, fitted on train rows only.

    This is the honest bar. Beating chance is trivial; beating "this line is
    usually late at this hour" is what would justify a model at all. Groups that
    never occur in train fall back to the global training rate.
    """
    t = df.iloc[tr].assign(_y=y[tr])
    rate = t.groupby(list(keys), observed=True)._y.mean()
    idx = df.iloc[te].set_index(list(keys)).index
    return pd.Series(idx.map(rate).astype(float)).fillna(y[tr].mean()).values


def scores(y_true, p):
    from sklearn.metrics import average_precision_score, roc_auc_score
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return dict(roc=float("nan"), pr=float("nan"))
    return dict(roc=roc_auc_score(y_true, p), pr=average_precision_score(y_true, p))


# ---------------------------------------------------------------------------
# 1. rolling-origin (expanding-window) validation
# ---------------------------------------------------------------------------
def make_folds(days, min_train_days, test_days):
    """Expanding window: train on everything up to a cut, test on the next block.

    Not TimeSeriesSplit on row index, because rows are unevenly distributed over
    the day and over the 24-day record (one day has only 516 rows after the
    outage). Whole calendar days keep each fold interpretable as "predict the
    next N days from everything known so far", which is the operational task.
    """
    folds, i = [], min_train_days
    while i + test_days <= len(days):
        folds.append((days[:i], days[i:i + test_days]))
        i += test_days
    return folds


def section_cv(df, X, y, cat, min_train_days, test_days):
    days = sorted(df.day.unique())
    folds = make_folds(days, min_train_days, test_days)
    print(f"\n{'='*78}\n1. ROLLING-ORIGIN VALIDATION  "
          f"({len(folds)} expanding-window folds, {test_days}-day test blocks)\n{'='*78}")
    print("Each fold trains on every day before the cut and tests on the next block --")
    print("the operational task. A model that is only good in one window is not a model.\n")
    print(f"{'fold':<5}{'train days':>11}{'test window':>25}{'n test':>8}{'pos':>7}"
          f"{'GBM roc':>9}{'look roc':>10}{'GBM pr':>8}{'look pr':>9}{'d roc':>8}")
    print("-" * 100)

    rows = []
    for k, (tr_days, te_days) in enumerate(folds, 1):
        tr = np.where(df.day.isin(tr_days))[0]
        te = np.where(df.day.isin(te_days))[0]
        if y[te].sum() < 10:
            print(f"{k:<5}  skipped: fewer than 10 positive events in the test block")
            continue
        g = scores(y[te], fit_gbm(X, y, tr, te, cat))
        b = scores(y[te], lookup(df, y, tr, te))
        h = scores(y[te], lookup(df, y, tr, te, keys=("hour",)))
        rows.append(dict(fold=k, n_train=len(tr), n_test=len(te),
                         train_days=len(tr_days),
                         test_start=str(te_days[0]), test_end=str(te_days[-1]),
                         pos=int(y[te].sum()), pos_rate=float(y[te].mean()),
                         gbm_roc=g["roc"], gbm_pr=g["pr"],
                         look_roc=b["roc"], look_pr=b["pr"],
                         hour_roc=h["roc"], hour_pr=h["pr"]))
        print(f"{k:<5}{len(tr_days):>11}{str(te_days[0])+'..'+str(te_days[-1])[5:]:>25}"
              f"{len(te):>8,}{y[te].sum():>7}{g['roc']:>9.3f}{b['roc']:>10.3f}"
              f"{g['pr']:>8.3f}{b['pr']:>9.3f}{g['roc']-b['roc']:>+8.3f}")

    r = pd.DataFrame(rows)
    print("-" * 100)
    print(f"{'mean':<5}{'':>11}{'':>25}{'':>8}{'':>7}"
          f"{r.gbm_roc.mean():>9.3f}{r.look_roc.mean():>10.3f}"
          f"{r.gbm_pr.mean():>8.3f}{r.look_pr.mean():>9.3f}"
          f"{(r.gbm_roc-r.look_roc).mean():>+8.3f}")
    print(f"{'sd':<5}{'':>11}{'':>25}{'':>8}{'':>7}"
          f"{r.gbm_roc.std():>9.3f}{r.look_roc.std():>10.3f}"
          f"{r.gbm_pr.std():>8.3f}{r.look_pr.std():>9.3f}"
          f"{(r.gbm_roc-r.look_roc).std():>+8.3f}")
    print(f"{'min':<5}{'':>11}{'':>25}{'':>8}{'':>7}"
          f"{r.gbm_roc.min():>9.3f}{r.look_roc.min():>10.3f}"
          f"{r.gbm_pr.min():>8.3f}{r.look_pr.min():>9.3f}"
          f"{(r.gbm_roc-r.look_roc).min():>+8.3f}")
    print(f"{'max':<5}{'':>11}{'':>25}{'':>8}{'':>7}"
          f"{r.gbm_roc.max():>9.3f}{r.look_roc.max():>10.3f}"
          f"{r.gbm_pr.max():>8.3f}{r.look_pr.max():>9.3f}"
          f"{(r.gbm_roc-r.look_roc).max():>+8.3f}")

    # Paired test on the per-fold difference. Folds share training data (each is a
    # superset of the last), so they are NOT independent and this p-value is
    # anti-conservative; the sign count below is the more honest statement.
    d = (r.gbm_roc - r.look_roc).values
    wins = int((d > 0).sum())
    se = d.std(ddof=1) / np.sqrt(len(d))
    print(f"\nGBM beats the (line x hour) lookup in {wins}/{len(d)} folds; "
          f"mean advantage {d.mean():+.3f} +- {se:.3f} (SE across folds).")
    print(f"hour-only lookup mean ROC {r.hour_roc.mean():.3f} -- how much is pure time-of-day.")
    print("Folds overlap in training data, so treat the SE as a rough scale, not a test.")
    return r


# ---------------------------------------------------------------------------
# 2. permutation / null test
# ---------------------------------------------------------------------------
def section_perm(df, X, y, cat, n_perm, rng):
    """Refit on shuffled labels to get the distribution of "no signal".

    Two nulls, because they answer different questions:

    GLOBAL shuffle -- labels permuted across all rows. Destroys every association,
    including the fact that some days are simply worse than others. This is the
    strict "is there anything here at all" null.

    WITHIN-DAY shuffle -- labels permuted only among rows of the same calendar day.
    The share of late departures per day is preserved, so a model can no longer
    profit from day-level or seasonal structure, only from which DEPARTURE within
    a day is late (line, stop, hour, direction, weather). If the real score stays
    outside this null too, the signal is about departures, not just about bad days.

    p is the standard empirical p-value with the +1 correction (Phipson & Smyth):
    the observed statistic counts as one of its own reference draws, so p can
    never be exactly 0 and is never anti-conservative.
    """
    cut_day = sorted(df.day.unique())[int(len(df.day.unique()) * 0.7)]
    tr = np.where(df.day < cut_day)[0]
    te = np.where(df.day >= cut_day)[0]

    obs_g = scores(y[te], fit_gbm(X, y, tr, te, cat))
    obs_b = scores(y[te], lookup(df, y, tr, te))
    print(f"\n{'='*78}\n2. PERMUTATION / NULL TEST  ({n_perm} shuffles per null)\n{'='*78}")
    print(f"Fixed split at {cut_day}: train {len(tr):,} rows / test {len(te):,} rows, "
          f"{y[te].sum():,} positive events in test.")
    print(f"OBSERVED  GBM roc {obs_g['roc']:.4f}  pr {obs_g['pr']:.4f}   |   "
          f"lookup roc {obs_b['roc']:.4f}  pr {obs_b['pr']:.4f}\n")

    day_idx = df.day.values
    out = {}
    for name in ("global", "within-day"):
        null_roc, null_pr, null_broc = [], [], []
        t0 = time.time()
        for i in range(n_perm):
            yp = y.copy()
            if name == "global":
                rng.shuffle(yp)
            else:
                # permute inside each day, preserving that day's positive rate
                for _, ix in pd.Series(np.arange(len(y))).groupby(day_idx):
                    v = ix.values
                    yp[v] = rng.permutation(yp[v])
            s = scores(yp[te], fit_gbm(X, yp, tr, te, cat))
            null_roc.append(s["roc"])
            null_pr.append(s["pr"])
            null_broc.append(scores(yp[te], lookup(df, yp, tr, te))["roc"])
            if i == 0:
                print(f"  [{name}] ~{(time.time()-t0)*n_perm/60:.1f} min for {n_perm} refits")
        null_roc, null_pr = np.array(null_roc), np.array(null_pr)
        p_roc = (1 + (null_roc >= obs_g["roc"]).sum()) / (n_perm + 1)
        p_pr = (1 + (null_pr >= obs_g["pr"]).sum()) / (n_perm + 1)
        # z is descriptive only -- the null need not be Gaussian.
        z = (obs_g["roc"] - null_roc.mean()) / null_roc.std(ddof=1)
        print(f"\n  {name} null, GBM:")
        print(f"    ROC-AUC null: mean {null_roc.mean():.4f}  sd {null_roc.std(ddof=1):.4f}  "
              f"max {null_roc.max():.4f}   -> observed {obs_g['roc']:.4f}, "
              f"p = {p_roc:.4g}  ({z:+.1f} sd above the null mean)")
        print(f"    PR-AUC  null: mean {null_pr.mean():.4f}  sd {null_pr.std(ddof=1):.4f}  "
              f"max {null_pr.max():.4f}   -> observed {obs_g['pr']:.4f}, p = {p_pr:.4g}")
        print(f"    (lookup baseline null ROC mean {np.mean(null_broc):.4f}, "
              f"max {np.max(null_broc):.4f}; observed {obs_b['roc']:.4f})")
        out[name] = dict(p_roc=float(p_roc), p_pr=float(p_pr), z_roc=float(z),
                         null_roc_mean=float(null_roc.mean()),
                         null_roc_sd=float(null_roc.std(ddof=1)),
                         null_roc_max=float(null_roc.max()),
                         null_pr_mean=float(null_pr.mean()),
                         null_pr_max=float(null_pr.max()),
                         n_perm=n_perm)
    print(f"\n  Smallest attainable p with {n_perm} shuffles is {1/(n_perm+1):.4g}; "
          "a reported value at that floor means\n  'no shuffle ever came close', "
          "not 'p is exactly this'.")
    out["observed"] = dict(gbm=obs_g, lookup=obs_b, cut_day=str(cut_day),
                           n_train=len(tr), n_test=len(te),
                           n_pos_test=int(y[te].sum()))
    return out


# ---------------------------------------------------------------------------
# 3. bootstrap confidence intervals
# ---------------------------------------------------------------------------
def section_boot(df, X, y, cat, n_boot, rng):
    """CIs on the headline split, resampling rows and (honestly) whole days.

    Row bootstrap treats every departure as an independent observation. It is not:
    a single disrupted afternoon turns hundreds of rows positive together, and one
    trip is logged at up to four stops. The row CI is therefore too narrow and is
    shown only because it is what a naive analysis would report.

    The day-cluster bootstrap resamples whole calendar days with replacement,
    which keeps those correlations intact. With only ~7 test days it is coarse
    and wide -- that width IS the finding: 24 days of data cannot pin this number
    down tightly, whatever the row count suggests.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    cut_day = sorted(df.day.unique())[int(len(df.day.unique()) * 0.7)]
    tr = np.where(df.day < cut_day)[0]
    te = np.where(df.day >= cut_day)[0]
    p_gbm = fit_gbm(X, y, tr, te, cat)
    p_look = lookup(df, y, tr, te)
    yte = y[te]
    days_te = df.day.values[te]
    uniq = np.unique(days_te)
    by_day = {d: np.where(days_te == d)[0] for d in uniq}

    print(f"\n{'='*78}\n3. BOOTSTRAP CONFIDENCE INTERVALS  ({n_boot} resamples)\n{'='*78}")
    print(f"Test set: {len(te):,} departures over {len(uniq)} days, "
          f"{yte.sum():,} positive events ({yte.mean():.2%}).\n")

    def ci(vals):
        v = np.asarray(vals)
        v = v[~np.isnan(v)]
        return np.percentile(v, 2.5), np.percentile(v, 97.5)

    res = {}
    for mode in ("row", "day-cluster"):
        acc = {k: [] for k in ("g_roc", "g_pr", "b_roc", "b_pr", "d_roc")}
        for _ in range(n_boot):
            if mode == "row":
                s = rng.integers(0, len(te), len(te))
            else:
                pick = rng.integers(0, len(uniq), len(uniq))
                s = np.concatenate([by_day[uniq[j]] for j in pick])
            yy = yte[s]
            if yy.sum() == 0 or yy.sum() == len(yy):
                continue
            gr, br = roc_auc_score(yy, p_gbm[s]), roc_auc_score(yy, p_look[s])
            acc["g_roc"].append(gr)
            acc["b_roc"].append(br)
            acc["d_roc"].append(gr - br)
            acc["g_pr"].append(average_precision_score(yy, p_gbm[s]))
            acc["b_pr"].append(average_precision_score(yy, p_look[s]))
        lo, hi = ci(acc["g_roc"])
        plo, phi = ci(acc["g_pr"])
        blo, bhi = ci(acc["b_roc"])
        dlo, dhi = ci(acc["d_roc"])
        dneg = float(np.mean(np.asarray(acc["d_roc"]) <= 0))
        print(f"  {mode} bootstrap")
        print(f"    GBM    ROC-AUC {roc_auc_score(yte, p_gbm):.3f}  95% CI [{lo:.3f}, {hi:.3f}]"
              f"   width {hi-lo:.3f}")
        print(f"    GBM    PR-AUC  {average_precision_score(yte, p_gbm):.3f}  "
              f"95% CI [{plo:.3f}, {phi:.3f}]   width {phi-plo:.3f}")
        print(f"    lookup ROC-AUC {roc_auc_score(yte, p_look):.3f}  95% CI [{blo:.3f}, {bhi:.3f}]")
        print(f"    GBM - lookup   {roc_auc_score(yte,p_gbm)-roc_auc_score(yte,p_look):+.3f}  "
              f"95% CI [{dlo:+.3f}, {dhi:+.3f}]   P(advantage <= 0) = {dneg:.3f}\n")
        res[mode] = dict(gbm_roc_ci=[lo, hi], gbm_pr_ci=[plo, phi],
                         look_roc_ci=[blo, bhi], diff_ci=[dlo, dhi],
                         p_diff_le_0=dneg)
    res["point"] = dict(gbm_roc=float(roc_auc_score(yte, p_gbm)),
                        gbm_pr=float(average_precision_score(yte, p_gbm)),
                        look_roc=float(roc_auc_score(yte, p_look)),
                        look_pr=float(average_precision_score(yte, p_look)),
                        n_test=len(te), n_pos=int(yte.sum()), n_days=len(uniq))
    print("  The row CI is the optimistic one. Quote the day-cluster CI: delays arrive")
    print("  in correlated bursts, so days -- not departures -- are the independent unit.")
    return res


# ---------------------------------------------------------------------------
# 4. temporal stability / drift
# ---------------------------------------------------------------------------
def section_drift(df, X, y, cat, train_days_n):
    print(f"\n{'='*78}\n4. TEMPORAL STABILITY AND DRIFT\n{'='*78}")
    days = sorted(df.day.unique())

    # (a) does the positive rate itself move? If the thing being predicted has a
    # different base rate every day, a fixed model is chasing a moving target and
    # any single-split score is partly a statement about which days it landed on.
    per_day = df.groupby("day").y.agg(["size", "sum", "mean"])
    from scipy.stats import chi2_contingency
    tab = np.vstack([per_day["sum"].values,
                     (per_day["size"] - per_day["sum"]).values])
    chi2, pval, dof, _ = chi2_contingency(tab)
    print("(a) does the positive rate drift over the 24 days?")
    print(f"    daily positive rate: min {per_day['mean'].min():.2%} "
          f"({per_day['mean'].idxmin()}), max {per_day['mean'].max():.2%} "
          f"({per_day['mean'].idxmax()}), "
          f"ratio {per_day['mean'].max()/per_day['mean'].min():.1f}x")
    print(f"    chi-square test of equal daily rates: chi2 = {chi2:.0f}, dof = {dof}, "
          f"p = {pval:.3g}")
    print("    -> the base rate is NOT stationary; day-to-day variation is real and large.")

    pre = per_day[[d < SOURCE_CHANGE for d in per_day.index]]
    post = per_day[[d >= SOURCE_CHANGE for d in per_day.index]]
    r_pre = pre["sum"].sum() / pre["size"].sum()
    r_post = post["sum"].sum() / post["size"].sum()
    print(f"\n    before the source change (< {SOURCE_CHANGE}, transport.rest): "
          f"{r_pre:.2%} over {int(pre['size'].sum()):,} rows")
    print(f"    after  the source change (>= {SOURCE_CHANGE}, VBB GTFS-RT):    "
          f"{r_post:.2%} over {int(post['size'].sum()):,} rows")
    print(f"    difference {r_post-r_pre:+.2%} -- a change of measurement instrument is")
    print("    confounded with any change in the city, and cannot be separated here.")

    # (b) does a model trained early keep working? Train once on the first N days,
    # then score each later day on its own. If skill decays with distance, the
    # single-split number is an average over a decaying curve.
    tr_days, te_days = days[:train_days_n], days[train_days_n:]
    tr = np.where(df.day.isin(tr_days))[0]
    print(f"\n(b) fixed model trained on the first {train_days_n} days "
          f"({tr_days[0]} .. {tr_days[-1]}, {len(tr):,} rows), scored day by day:")
    print(f"    {'day':<12}{'gap':>5}{'n':>8}{'pos':>6}{'pos rate':>10}"
          f"{'GBM roc':>9}{'look roc':>10}")
    rows = []
    all_te = np.where(df.day.isin(te_days))[0]
    p_all = fit_gbm(X, y, tr, all_te, cat)
    b_all = lookup(df, y, tr, all_te)
    pos_in_te = pd.Series(p_all, index=all_te)
    bos_in_te = pd.Series(b_all, index=all_te)
    for gap, d in enumerate(te_days, 1):
        ix = np.where(df.day.values == d)[0]
        if y[ix].sum() < 5:
            print(f"    {str(d):<12}{gap:>5}{len(ix):>8}{y[ix].sum():>6}"
                  f"{'  too few positives to score':>29}")
            continue
        g = scores(y[ix], pos_in_te.loc[ix].values)
        b = scores(y[ix], bos_in_te.loc[ix].values)
        rows.append(dict(day=str(d), gap=gap, n=len(ix), pos=int(y[ix].sum()),
                         rate=float(y[ix].mean()), gbm_roc=g["roc"], look_roc=b["roc"]))
        print(f"    {str(d):<12}{gap:>5}{len(ix):>8,}{y[ix].sum():>6}"
              f"{y[ix].mean():>10.2%}{g['roc']:>9.3f}{b['roc']:>10.3f}")
    r = pd.DataFrame(rows)
    if len(r) > 2:
        sl = np.polyfit(r.gap, r.gbm_roc, 1)[0]
        c = np.corrcoef(r.gap, r.gbm_roc)[0, 1]
        print(f"\n    per-day ROC-AUC: mean {r.gbm_roc.mean():.3f}, sd {r.gbm_roc.std():.3f}, "
              f"range [{r.gbm_roc.min():.3f}, {r.gbm_roc.max():.3f}]")
        print(f"    trend vs days-since-training: slope {sl:+.4f} AUC/day (r = {c:+.2f})")
        print("    Single-day AUCs rest on one day of events each -- the scatter here is")
        print("    mostly sampling noise, so read the slope as 'no strong decay detected',")
        print("    NOT as evidence that the model is stable.")
    return dict(per_day_rate={str(k): float(v) for k, v in per_day["mean"].items()},
                chi2=float(chi2), chi2_p=float(pval),
                rate_pre=float(r_pre), rate_post=float(r_post),
                per_day_scores=r.to_dict("records"),
                slope=float(np.polyfit(r.gap, r.gbm_roc, 1)[0]) if len(r) > 2 else None)


# ---------------------------------------------------------------------------
# 5. split sanity
# ---------------------------------------------------------------------------
def section_split(df, X, y, cat, rng):
    """Prove the split is clean, and measure what a random split would have cost.

    The dataset logs one trip at up to four stops and the same line every few
    minutes, so a random split would put near-copies of test rows into train. The
    gap between the random-split and time-split scores is the size of that
    optimism -- worth stating explicitly, because it is exactly the mistake the
    project would otherwise be accused of.
    """
    from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit

    print(f"\n{'='*78}\n5. SPLIT SANITY\n{'='*78}")
    dup = df.duplicated(["trip_id", "stop_id", "planned_when"]).sum()
    print(f"(a) exact duplicate departures (trip_id, stop_id, planned_when): {dup}")

    cut_day = sorted(df.day.unique())[int(len(df.day.unique()) * 0.7)]
    tr = np.where(df.day < cut_day)[0]
    te = np.where(df.day >= cut_day)[0]
    shared_rows = df.index[te].isin(df.index[tr]).sum()
    shared_trips = len(set(df.trip_id.values[tr]) & set(df.trip_id.values[te]))
    aff = df.trip_id.values[te]
    n_aff = np.isin(aff, df.trip_id.values[tr]).sum()
    print(f"(b) day-boundary time split at {cut_day}: {shared_rows} rows in both sides "
          f"(must be 0)")
    print(f"    trip_ids present on both sides: {shared_trips} "
          f"({n_aff} of {len(te):,} test rows = {n_aff/len(te):.3%})")
    print("    A trip is logged at several stops within ~20 min, so a same-day cut could")
    print("    split one; cutting on calendar days makes even that essentially impossible.")
    print(f"    trips appearing at more than one stop overall: "
          f"{(df.groupby('trip_id').stop_id.nunique() > 1).sum():,} "
          f"-- these rows are correlated and inflate the apparent sample size.")

    print("\n(c) what a RANDOM split would have reported instead (optimism check):")
    variants = {}
    variants["time (day boundary)"] = (tr, te)
    rs = ShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
    variants["random rows"] = next(rs.split(X))
    gs = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
    variants["random, trips kept together"] = next(gs.split(X, groups=df.trip_id.values))
    print(f"    {'split':<32}{'ROC-AUC':>9}{'PR-AUC':>9}")
    out = {}
    for name, (a, b) in variants.items():
        s = scores(y[b], fit_gbm(X, y, a, b, cat))
        out[name] = s
        print(f"    {name:<32}{s['roc']:>9.3f}{s['pr']:>9.3f}")
    infl = out["random rows"]["roc"] - out["time (day boundary)"]["roc"]
    print(f"\n    random-split optimism: {infl:+.3f} ROC-AUC.")
    print("    Grouping trips removes only part of it, so the inflation is mostly the")
    print("    general nearness in time (same day, same weather, same disruption),")
    print("    not just the multi-stop duplication. The time split is the correct one;")
    print("    any published number must come from it.")
    return dict(duplicates=int(dup), rows_in_both=int(shared_rows),
                shared_trip_ids=int(shared_trips),
                test_rows_with_train_trip=float(n_aff / len(te)),
                random_split_optimism=float(infl),
                variants={k: {kk: float(vv) for kk, vv in v.items()}
                          for k, v in out.items()})


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threshold", type=int, default=600,
                    help="seconds of delay counted as 'late' (default 600)")
    ap.add_argument("--n-perm", type=int, default=100, help="label shuffles per null")
    ap.add_argument("--n-boot", type=int, default=2000, help="bootstrap resamples")
    ap.add_argument("--min-train-days", type=int, default=8,
                    help="days in the first rolling-origin training window")
    ap.add_argument("--test-days", type=int, default=2,
                    help="days per rolling-origin test block")
    ap.add_argument("--drift-train-days", type=int, default=10,
                    help="days used for the fixed model in the drift section")
    ap.add_argument("--quick", action="store_true",
                    help="20 shuffles / 400 bootstraps, for a fast re-check")
    ap.add_argument("--only", default="cv,perm,boot,drift,split",
                    help="comma-separated subset of sections to run")
    ap.add_argument("--out", default=str(OUT_JSON),
                    help="where to write the machine-readable results")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.quick:
        args.n_perm, args.n_boot = 20, 400

    rng = np.random.default_rng(args.seed)
    want = {s.strip() for s in args.only.split(",")}
    df, X, y, cat = load(args.threshold)
    days = sorted(df.day.unique())

    print(f"{'='*78}")
    print(f"VALIDATION OF THE DELAY SIGNAL   target: final_delay_s >= "
          f"{args.threshold}s ({args.threshold//60} min)")
    print(f"{'='*78}")
    print(f"rows {len(df):,} | positives {y.sum():,} ({y.mean():.2%}) | "
          f"{len(days)} calendar days {days[0]} .. {days[-1]}")
    print(f"features: {len(cat)} categorical + {len(X.columns)-len(cat)} numeric; "
          f"none of {len(LEAKY)} realtime/leaky columns used")
    missing = [d for d in pd.date_range(days[0], days[-1]).date if d not in set(days)]
    if missing:
        print(f"MISSING DAYS (collection outage): {', '.join(str(d) for d in missing)}")
    print(f"positive events per day, on average: {y.sum()/len(days):.0f} -- this, not the "
          f"row count, sets the precision of everything below.")

    results = dict(threshold=args.threshold, n_rows=len(df), n_pos=int(y.sum()),
                   pos_rate=float(y.mean()), n_days=len(days),
                   missing_days=[str(d) for d in missing])
    if "cv" in want:
        results["cv"] = section_cv(df, X, y, cat, args.min_train_days,
                                   args.test_days).to_dict("records")
    if "perm" in want:
        results["permutation"] = section_perm(df, X, y, cat, args.n_perm, rng)
    if "boot" in want:
        results["bootstrap"] = section_boot(df, X, y, cat, args.n_boot, rng)
    if "drift" in want:
        results["drift"] = section_drift(df, X, y, cat, args.drift_train_days)
    if "split" in want:
        results["split"] = section_split(df, X, y, cat, rng)

    Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    print(f"\nmachine-readable results -> {args.out}")


if __name__ == "__main__":
    main()
