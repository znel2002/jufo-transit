"""Fair, leakage-free comparison of model families for "will this departure be late?".

RESEARCH QUESTION
-----------------
Predict, *at schedule time* (i.e. before any realtime observation of the trip
exists), whether a departure will end up at least T seconds late.  T = 180 s and
T = 600 s are the two thresholds of interest: 180 s is "noticeably late",
600 s is "your connection is gone".

WHY THIS SCRIPT EXISTS
----------------------
A previous pass found that an *untuned* HistGradientBoostingClassifier beats a
simple (line_name, hour) historical-rate lookup table on ROC-AUC but *loses* to
it on PR-AUC at the >=600 s threshold.  With a positive rate of ~1.6 % that is
the more meaningful metric, so the untuned model was effectively worse than a
five-line group-by.  This script tests whether that is
  (a) a real ceiling of the allowed feature set, or
  (b) an artefact of a badly configured model (no imbalance handling, no tuning).
Both answers are publishable; (a) would be the more interesting one.

DESIGN DECISIONS (all deliberate, all defensible to a jury)
-----------------------------------------------------------
1. LEAKAGE.  Everything that is only knowable *after* the trip started running
   is excluded by construction: first_delay_s, delay_drift_s, n_obs,
   lead_time_s, first/last_observed_at, hour_utc (a collection artefact, not a
   calendar feature) and of course final_delay_s itself.  The allowed feature
   list below is a hard whitelist, not a blacklist, so a newly added column can
   never silently leak in.
2. SPLIT.  Strictly time-ordered 70/30 by planned_when.  A random split would be
   catastrophically optimistic here: the same line, at the same stop, at the same
   minute, recurs every single day, so a random split leaks almost-duplicate rows
   into the test set.
3. MODEL SELECTION.  All tuning happens on an inner, also time-ordered,
   train/validation split carved out of TRAIN.  The test set is touched exactly
   once per model, for the final numbers.  No metric on the test set ever feeds
   back into a choice.
4. METRICS.  PR-AUC (average_precision) is the headline.  With a 1.6 % positive
   rate ROC-AUC is inflated and easy to feel good about; PR-AUC is not.  Brier
   score + a reliability table are reported because for rare-event warning
   systems a *calibrated* probability ("this one is 30 % likely to be 10 min
   late") is more useful than a good ranking.
5. The lookup table is reproduced exactly as specified so it is a like-for-like
   reference row, not a strawman.

Usage
-----
    ./.venv/bin/python analysis/experiments/model_comparison.py
    ./.venv/bin/python analysis/experiments/model_comparison.py --thresholds 180
    ./.venv/bin/python analysis/experiments/model_comparison.py --quick
    ./.venv/bin/python analysis/experiments/model_comparison.py --n-search 60
"""

from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

# sklearn >= 1.6 replaces CalibratedClassifierCV(cv="prefit") with FrozenEstimator.
try:  # pragma: no cover - depends on installed sklearn
    from sklearn.frozen import FrozenEstimator

    _HAS_FROZEN = True
except ImportError:  # pragma: no cover
    _HAS_FROZEN = False

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

REPO = Path(__file__).resolve().parents[2]
DEFAULT_DATA = REPO / "data" / "dataset.parquet"

# --------------------------------------------------------------------------
# Feature whitelist.  Anything not listed here is never shown to any model.
# --------------------------------------------------------------------------
CAT_FEATURES = ["product", "line_name", "stop_id", "direction"]

CALENDAR_FEATURES = [
    "hour",
    "minute",
    "weekday",
    "month",
    "day_of_year",
    "is_weekend",
    "is_school_holiday",
    "is_public_holiday",
    "is_morning_peak",
    "is_evening_peak",
    "is_full_traffic_day",
]

WEATHER_FEATURES = [
    "temp_c",
    "humidity_pct",
    "precip_mm",
    "precip_form",
    "wind_ms",
    "wind_dir_deg",
    "is_rain",
    "is_wet",
    "is_snow",
    "is_freezing",
    "is_hot",
]

NUM_FEATURES = CALENDAR_FEATURES + WEATHER_FEATURES
ALL_FEATURES = CAT_FEATURES + NUM_FEATURES

# Columns that must NEVER be used.  Kept explicit so the script self-documents
# the leakage argument, and so the assertion below can actually check it.
FORBIDDEN = [
    "first_delay_s",
    "delay_drift_s",
    "n_obs",
    "lead_time_s",
    "last_observed_at",
    "first_observed_at",
    "hour_utc",
    "trip_id",
    "final_delay_s",
]


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_data(path: Path) -> pd.DataFrame:
    """Load, apply the analysis filter, and sort by scheduled departure time.

    The filter (cancelled == 0, final_delay_s present) is the project-wide
    convention: a cancellation is a different event from a delay and modelling
    them jointly would conflate two questions.
    """
    df = pd.read_parquet(path)
    n_raw = len(df)
    df = df[(df["cancelled"] == 0) & df["final_delay_s"].notna()].copy()
    df = df.sort_values("planned_when", kind="mergesort").reset_index(drop=True)
    print(f"rows: {n_raw:,} raw -> {len(df):,} usable "
          f"({df['planned_when'].min()} .. {df['planned_when'].max()})")

    assert not set(ALL_FEATURES) & set(FORBIDDEN), "leaky column in feature list"
    missing = [c for c in ALL_FEATURES if c not in df.columns]
    assert not missing, f"missing feature columns: {missing}"
    return df


def make_X(df: pd.DataFrame) -> pd.DataFrame:
    """Whitelisted feature frame; booleans cast to int so every estimator copes."""
    X = df[ALL_FEATURES].copy()
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(np.int8)
    for c in CAT_FEATURES:
        X[c] = X[c].astype("string").fillna("__NA__")
    return X


def time_split(n: int, frac: float) -> tuple[np.ndarray, np.ndarray]:
    """Index arrays for a time-ordered prefix/suffix split (data must be sorted)."""
    k = int(n * frac)
    return np.arange(k), np.arange(k, n)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
@dataclass
class Result:
    name: str
    roc_auc: float
    pr_auc: float
    brier: float
    pos_rate: float
    lift_vs_base: float          # PR-AUC divided by the no-skill PR-AUC (= base rate)
    fit_seconds: float
    notes: str = ""
    proba: np.ndarray | None = field(default=None, repr=False)


def evaluate(name: str, y_true: np.ndarray, p: np.ndarray, secs: float,
             notes: str = "") -> Result:
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    base = float(np.mean(y_true))
    ap = average_precision_score(y_true, p)
    return Result(
        name=name,
        roc_auc=roc_auc_score(y_true, p),
        pr_auc=ap,
        brier=brier_score_loss(y_true, p),
        pos_rate=base,
        lift_vs_base=ap / base if base > 0 else float("nan"),
        fit_seconds=secs,
        notes=notes,
        proba=p,
    )


def bootstrap_vs_bar(y_true: np.ndarray, results: list[Result], bar: Result,
                     n_boot: int, seed: int) -> pd.DataFrame:
    """Paired bootstrap of (PR-AUC of model) - (PR-AUC of the lookup table).

    Why this is not optional: at >=600 s the test set holds ~1,100 positives, so
    a PR-AUC gap of 0.005 is well inside sampling noise.  Declaring a winner on
    the point estimate alone would be exactly the kind of overclaiming this
    project is trying to avoid.  Test rows are resampled *jointly* for all models
    (paired), which is far more sensitive than comparing two independent CIs.
    """
    n = len(y_true)
    P = np.column_stack([r.proba for r in results])
    bar_p = bar.proba

    def chunk(sub_seed: int, reps: int) -> np.ndarray:
        rng = np.random.default_rng(sub_seed)
        out = np.full((reps, P.shape[1]), np.nan)
        for b in range(reps):
            idx = rng.integers(0, n, n)
            yb = y_true[idx]
            if yb.sum() == 0:      # degenerate resample; PR-AUC undefined
                continue
            base = average_precision_score(yb, bar_p[idx])
            for j in range(P.shape[1]):
                out[b, j] = average_precision_score(yb, P[idx, j]) - base
        return out

    # Bootstrap replicates are embarrassingly parallel; serially this dominates
    # the whole script's runtime.
    n_workers = 10
    per = int(np.ceil(n_boot / n_workers))
    with threadpool_limits(limits=1):
        parts = Parallel(n_jobs=n_workers)(
            delayed(chunk)(seed + 1000 * w, per) for w in range(n_workers))
    D = np.vstack(parts)[:n_boot]

    rows = []
    for j, r in enumerate(results):
        d = D[:, j][~np.isnan(D[:, j])]
        lo, hi = np.percentile(d, [2.5, 97.5])
        rows.append({
            "model": r.name,
            "pr_auc": r.pr_auc,
            "delta_vs_bar": r.pr_auc - bar.pr_auc,
            "ci95_lo": lo, "ci95_hi": hi,
            # Fraction of resamples where the model loses: a one-sided bootstrap
            # p-value for "this model is no better than the lookup table".
            "p_not_better": float((d <= 0).mean()),
            "significant": bool(lo > 0),
        })
    return pd.DataFrame(rows).sort_values("delta_vs_bar", ascending=False)


def reliability_table(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Decile reliability check.

    Equal-count (quantile) bins rather than equal-width: with a 1.6 % positive
    rate almost every prediction sits in the lowest equal-width bin, which makes
    an equal-width diagram uninformative.
    """
    q = pd.qcut(pd.Series(p).rank(method="first"), n_bins, labels=False)
    out = pd.DataFrame({"bin": q, "p": p, "y": y_true})
    g = out.groupby("bin").agg(n=("y", "size"), mean_pred=("p", "mean"),
                               obs_rate=("y", "mean"))
    g["gap"] = g["mean_pred"] - g["obs_rate"]
    return g


# --------------------------------------------------------------------------
# Model 2: the bar to beat -- historical positive rate per (line_name, hour)
# --------------------------------------------------------------------------
def lookup_table_predict(train: pd.DataFrame, apply_to: pd.DataFrame, y_train: np.ndarray,
                         *, smoothing: float = 0.0) -> np.ndarray:
    """Historical positive rate per (line_name, hour), fitted on TRAIN only.

    Unseen (line, hour) groups fall back to the global train rate.  With
    smoothing m > 0 the group rate is shrunk toward that global rate as
    (k + m*prior) / (n + m), which is the standard empirical-Bayes correction for
    groups with very few observations -- the untuned version happily reports 100 %
    for a group that was late once out of one observation.
    """
    prior = float(y_train.mean())
    tmp = pd.DataFrame({
        "line_name": train["line_name"].to_numpy(),
        "hour": train["hour"].to_numpy(),
        "y": y_train,
    })
    g = tmp.groupby(["line_name", "hour"])["y"].agg(["sum", "size"])
    rate = (g["sum"] + smoothing * prior) / (g["size"] + smoothing)

    key = pd.MultiIndex.from_arrays(
        [apply_to["line_name"].to_numpy(), apply_to["hour"].to_numpy()]
    )
    out = rate.reindex(key)
    # Guard against a silent dtype mismatch turning the whole lookup into the
    # constant prior (which would look like a plausible-but-wrong result).
    miss = float(out.isna().mean())
    assert miss < 0.25, f"lookup fell back to the prior for {miss:.1%} of rows"
    return out.fillna(prior).to_numpy()


def oof_group_rate(train: pd.DataFrame, y_train: np.ndarray, test: pd.DataFrame,
                   keys: list[str], *, n_folds: int = 5,
                   smoothing: float = 20.0) -> tuple[np.ndarray, np.ndarray]:
    """Target encoding of a group key, safe to hand to a learner as a feature.

    For TRAIN rows the encoding is computed out-of-fold over *time-ordered*
    folds, so a row never contributes to its own encoding; for TEST rows it is
    computed on all of TRAIN.  Without the out-of-fold step the tree would just
    memorise the target through this column and validation would look great for
    the wrong reason.
    """
    prior = float(y_train.mean())

    def fit_rate(idx: np.ndarray) -> pd.Series:
        t = train.iloc[idx][keys].copy()
        t["y"] = y_train[idx]
        g = t.groupby(keys)["y"].agg(["sum", "size"])
        return (g["sum"] + smoothing * prior) / (g["size"] + smoothing)

    def apply_rate(rate: pd.Series, frame: pd.DataFrame) -> np.ndarray:
        key = pd.MultiIndex.from_frame(frame[keys]) if len(keys) > 1 \
            else pd.Index(frame[keys[0]])
        return rate.reindex(key).fillna(prior).to_numpy()

    tr_enc = np.full(len(train), prior)
    bounds = np.linspace(0, len(train), n_folds + 1).astype(int)
    for i in range(n_folds):
        lo, hi = bounds[i], bounds[i + 1]
        hold = np.arange(lo, hi)
        rest = np.concatenate([np.arange(0, lo), np.arange(hi, len(train))])
        tr_enc[hold] = apply_rate(fit_rate(rest), train.iloc[hold])

    te_enc = apply_rate(fit_rate(np.arange(len(train))), test)
    return tr_enc, te_enc


# --------------------------------------------------------------------------
# Pipelines
# --------------------------------------------------------------------------
def linear_pipeline(class_weight=None) -> Pipeline:
    """One-hot categoricals + median-imputed, standardised numerics.

    min_frequency=20 collapses the long tail of rare lines/directions into a
    single "infrequent" level: with 245 lines and 235 directions the raw one-hot
    would otherwise hand the model hundreds of columns seen a handful of times.
    handle_unknown='infrequent_if_exist' makes the 26 lines that only appear in
    the test period land in that same bucket instead of crashing.
    """
    cat = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20,
                        sparse_output=True)
    num = Pipeline([("imp", SimpleImputer(strategy="median")),
                    ("sc", StandardScaler())])
    pre = ColumnTransformer([("cat", cat, CAT_FEATURES), ("num", num, NUM_FEATURES)])
    return Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=2000, class_weight=class_weight,
                                   solver="lbfgs", C=1.0)),
    ])


def forest_pipeline(estimator) -> Pipeline:
    """Ordinal-encoded categoricals + median-imputed numerics for the forests.

    Forests cannot take NaN, and one-hot over 245 lines would make every split
    a near-useless binary question, so ordinal codes are the pragmatic choice.
    They impose a meaningless order, which is a real (and reportable) handicap
    for the forests relative to HistGB's native categorical splits.
    """
    cat = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                         encoded_missing_value=-1)
    num = SimpleImputer(strategy="median")
    pre = ColumnTransformer([("cat", cat, CAT_FEATURES), ("num", num, NUM_FEATURES)])
    return Pipeline([("pre", pre), ("clf", estimator)])


def hgb_encode(train_X: pd.DataFrame, other_X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ordinal-encode the categoricals for HistGB's *native* categorical support.

    Unknown/missing categories become NaN, which HistGB routes to its own
    missing-value branch -- exactly the behaviour we want for the lines that only
    exist in the test period.  Returns (train, other, categorical mask).
    """
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                         encoded_missing_value=np.nan)
    tr_cat = enc.fit_transform(train_X[CAT_FEATURES])
    ot_cat = enc.transform(other_X[CAT_FEATURES])
    tr = np.hstack([tr_cat, train_X[NUM_FEATURES].to_numpy(dtype=float)])
    ot = np.hstack([ot_cat, other_X[NUM_FEATURES].to_numpy(dtype=float)])
    mask = np.array([True] * len(CAT_FEATURES) + [False] * len(NUM_FEATURES))
    return tr, ot, mask


# --------------------------------------------------------------------------
# Tuning (inner, time-ordered validation split of TRAIN)
# --------------------------------------------------------------------------
HGB_GRID = {
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "max_iter": [100, 200, 400, 800],
    "max_leaf_nodes": [7, 15, 31, 63],
    "min_samples_leaf": [20, 50, 100, 200, 500],
    "l2_regularization": [0.0, 0.1, 1.0, 10.0],
    # Whether to reweight the classes is treated as just another hyperparameter,
    # because that is exactly the question: does the missing imbalance handling
    # actually buy anything on PR-AUC, or does it only wreck calibration?
    "balanced": [False, True],
}


def sample_params(rng: np.random.Generator, n: int) -> list[dict]:
    seen, out = set(), []
    while len(out) < n:
        p = {k: v[int(rng.integers(len(v)))] for k, v in HGB_GRID.items()}
        key = tuple(sorted(p.items()))
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _one_config(p: dict, Xtr, ytr, Xva, yva, mask, seed: int) -> dict:
    """Fit one hyperparameter draw and score it on the inner validation slice."""
    kw = {k: v for k, v in p.items() if k != "balanced"}
    m = HistGradientBoostingClassifier(categorical_features=mask, random_state=seed,
                                       early_stopping=False, **kw)
    sw = compute_sample_weight("balanced", ytr) if p["balanced"] else None
    t0 = time.time()
    with threadpool_limits(limits=1):  # one thread per worker; parallelism is over configs
        m.fit(Xtr, ytr, sample_weight=sw)
        pv = m.predict_proba(Xva)[:, 1]
    return {**p, "val_pr_auc": average_precision_score(yva, pv),
            "val_roc_auc": roc_auc_score(yva, pv), "secs": time.time() - t0}


def tune_hgb(Xtr, ytr, Xva, yva, mask, n_search: int, seed: int,
             n_jobs: int = -1) -> tuple[dict, pd.DataFrame]:
    """Random search scored by PR-AUC on the inner validation split.

    Random search rather than grid: with 6 dimensions a full grid is 3200 fits,
    and random search over ~40 draws is the well-established better use of the
    same budget.  Selection metric is average_precision because that is the
    metric the project cares about -- tuning on ROC-AUC and then reporting PR-AUC
    would be selecting for the wrong thing.

    The draws are run as parallel *processes* with one thread each: HistGB's own
    OpenMP parallelism barely saturates one core on this data shape, so spreading
    independent fits across cores is far more effective (~8x wall-clock here).
    """
    rng = np.random.default_rng(seed)
    draws = sample_params(rng, n_search)
    rows = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_one_config)(p, Xtr, ytr, Xva, yva, mask, seed) for p in draws
    )
    df = pd.DataFrame(rows).sort_values("val_pr_auc", ascending=False)
    best = {k: df.iloc[0][k] for k in HGB_GRID}
    # numpy scalars round-trip badly into sklearn's int/bool params
    best = {k: (bool(v) if k == "balanced" else
                int(v) if k in ("max_iter", "max_leaf_nodes", "min_samples_leaf")
                else float(v)) for k, v in best.items()}
    return best, df


# --------------------------------------------------------------------------
# One full threshold run
# --------------------------------------------------------------------------
def run_threshold(df: pd.DataFrame, thr: int, args) -> tuple[list[Result], pd.DataFrame, dict]:
    y = (df["final_delay_s"] >= thr).to_numpy().astype(np.int8)
    X = make_X(df)
    tr_idx, te_idx = time_split(len(df), args.train_frac)

    # Inner split of TRAIN for model selection.  Time-ordered again, so tuning
    # mirrors the real deployment situation (fit on the past, judge on the future).
    itr, iva = time_split(len(tr_idx), args.val_frac)

    df_tr, df_te = df.iloc[tr_idx], df.iloc[te_idx]
    X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    print(f"\n{'='*78}\nTHRESHOLD >= {thr} s\n{'='*78}")
    print(f"train {len(tr_idx):,} rows, pos rate {y_tr.mean():.4%}  "
          f"({df_tr['planned_when'].min().date()} .. {df_tr['planned_when'].max().date()})")
    print(f"test  {len(te_idx):,} rows, pos rate {y_te.mean():.4%}  "
          f"({df_te['planned_when'].min().date()} .. {df_te['planned_when'].max().date()})")
    print(f"inner: fit {len(itr):,} / validate {len(iva):,}")

    results: list[Result] = []

    def add(name, p, secs, notes=""):
        r = evaluate(name, y_te, p, secs, notes)
        results.append(r)
        print(f"  {name:<44s} ROC {r.roc_auc:.3f}  PR {r.pr_auc:.4f}  "
              f"Brier {r.brier:.5f}  ({secs:.1f}s)")
        return r

    # -- 1. Dummy -----------------------------------------------------------
    t0 = time.time()
    dm = DummyClassifier(strategy="prior").fit(X_tr, y_tr)
    add("1. Dummy (train base rate)", dm.predict_proba(X_te)[:, 1], time.time() - t0,
        "no-skill floor; PR-AUC == test base rate by construction")

    # -- 2. Lookup tables ---------------------------------------------------
    t0 = time.time()
    p_lut = lookup_table_predict(df_tr, df_te, y_tr, smoothing=0.0)
    add("2. Lookup (line_name, hour)  [THE BAR]", p_lut, time.time() - t0,
        "historical train rate per group, global rate for unseen groups")

    t0 = time.time()
    p_lut_s = lookup_table_predict(df_tr, df_te, y_tr, smoothing=20.0)
    add("2b. Lookup (line, hour) + smoothing m=20", p_lut_s, time.time() - t0,
        "empirical-Bayes shrink of small groups toward the global rate")

    t0 = time.time()
    p_lut3 = lookup_table_predict(
        df_tr.assign(line_name=df_tr["line_name"] + "|" + df_tr["stop_id"]),
        df_te.assign(line_name=df_te["line_name"] + "|" + df_te["stop_id"]),
        y_tr, smoothing=20.0)
    add("2c. Lookup (line, stop, hour) + smoothing", p_lut3, time.time() - t0,
        "finer group; tests whether the bar itself can simply be raised")

    # -- 3. Logistic regression --------------------------------------------
    for cw, tag in [(None, ""), ("balanced", " [class_weight=balanced]")]:
        t0 = time.time()
        lr = linear_pipeline(class_weight=cw).fit(X_tr, y_tr)
        add(f"3. LogisticRegression{tag}", lr.predict_proba(X_te)[:, 1],
            time.time() - t0, "one-hot cats (min_freq 20), standardised numerics")

    # -- 4. Random forest ---------------------------------------------------
    # Small selection pass on the inner split: min_samples_leaf and class_weight
    # are the two knobs that actually matter for a rare-event forest.
    rf_best, rf_best_ap = None, -np.inf
    rf_rows = []
    for msl in ([50] if args.quick else [20, 50, 200]):
        for cw in [None, "balanced_subsample"]:
            est = RandomForestClassifier(
                n_estimators=args.rf_trees, min_samples_leaf=msl, class_weight=cw,
                n_jobs=-1, random_state=args.seed)
            pipe = forest_pipeline(est).fit(X_tr.iloc[itr], y_tr[itr])
            ap = average_precision_score(y_tr[iva], pipe.predict_proba(X_tr.iloc[iva])[:, 1])
            rf_rows.append({"min_samples_leaf": msl, "class_weight": str(cw),
                            "val_pr_auc": ap})
            if ap > rf_best_ap:
                rf_best, rf_best_ap = (msl, cw), ap
    print(f"  [RF inner search] best {rf_best} val PR-AUC {rf_best_ap:.4f}")

    # Both weighting variants are always reported at the selected leaf size, since
    # "does imbalance handling help?" is one of the questions under test.
    for cw in [None, "balanced_subsample"]:
        msl = rf_best[0]
        t0 = time.time()
        est = RandomForestClassifier(n_estimators=args.rf_trees, min_samples_leaf=msl,
                                     class_weight=cw, n_jobs=-1, random_state=args.seed)
        pipe = forest_pipeline(est).fit(X_tr, y_tr)
        tag = f"class_weight={cw}, min_samples_leaf={msl}"
        add(f"4. RandomForest ({tag})", pipe.predict_proba(X_te)[:, 1], time.time() - t0,
            "ordinal-encoded cats" + (" [inner-search pick]" if cw == rf_best[1] else ""))

    # -- 5. HistGradientBoosting -------------------------------------------
    # ONE ordinal encoder, fit on TRAIN, then sliced.  Fitting a second encoder on
    # the inner-train subset would assign different integer codes to the same
    # category, so a model fitted there could not be applied to A_te at all -- a
    # silent, catastrophic bug rather than an error.  (Fitting the encoder on all
    # of TRAIN only shares the category *vocabulary* with the inner validation
    # slice, never the target, so it is not target leakage.)
    A_tr, A_te, mask = hgb_encode(X_tr, X_te)
    A_itr, A_iva = A_tr[itr], A_tr[iva]

    t0 = time.time()
    hgb0 = HistGradientBoostingClassifier(categorical_features=mask,
                                          random_state=args.seed).fit(A_tr, y_tr)
    add("5. HistGB untuned (sklearn defaults)", hgb0.predict_proba(A_te)[:, 1],
        time.time() - t0, "reproduction of the previously reported model")

    t0 = time.time()
    sw = compute_sample_weight("balanced", y_tr)
    hgb_b = HistGradientBoostingClassifier(categorical_features=mask,
                                           random_state=args.seed).fit(A_tr, y_tr,
                                                                       sample_weight=sw)
    add("5b. HistGB untuned + balanced sample_weight",
        hgb_b.predict_proba(A_te)[:, 1], time.time() - t0,
        "isolates the effect of imbalance handling alone")

    # --reuse-search reloads a previous search instead of repeating it.  The search
    # only ever touched TRAIN, so reusing it changes nothing methodologically; it
    # just makes re-running the downstream comparison cheap.
    cache = args.out_dir / f"hgb_search_{thr}s.csv"
    if args.reuse_search and cache.exists():
        search_df = pd.read_csv(cache).sort_values("val_pr_auc", ascending=False)
        best = {k: search_df.iloc[0][k] for k in HGB_GRID}
        best = {k: (bool(v) if k == "balanced" else
                    int(v) if k in ("max_iter", "max_leaf_nodes", "min_samples_leaf")
                    else float(v)) for k, v in best.items()}
        print(f"  [HistGB search reused from {cache.name} "
              f"({len(search_df)} draws)]")
    else:
        print(f"  [HistGB random search, {args.n_search} draws, PR-AUC on inner val]")
        best, search_df = tune_hgb(A_itr, y_tr[itr], A_iva, y_tr[iva], mask,
                                   args.n_search, args.seed, n_jobs=args.n_jobs)
    print(f"  best params: {best}  (val PR-AUC {search_df.iloc[0]['val_pr_auc']:.4f})")

    kw = {k: v for k, v in best.items() if k != "balanced"}
    t0 = time.time()
    hgb_t = HistGradientBoostingClassifier(categorical_features=mask, random_state=args.seed,
                                           early_stopping=False, **kw)
    sw_t = compute_sample_weight("balanced", y_tr) if best["balanced"] else None
    hgb_t.fit(A_tr, y_tr, sample_weight=sw_t)
    r_tuned = add("5c. HistGB TUNED (random search on inner val)",
                  hgb_t.predict_proba(A_te)[:, 1], time.time() - t0, str(best))

    # -- 6. Extra models ----------------------------------------------------
    # 6a. ExtraTrees: more randomised splits, a genuinely different bias.
    if not args.quick:
        t0 = time.time()
        et = forest_pipeline(ExtraTreesClassifier(
            n_estimators=args.rf_trees, min_samples_leaf=rf_best[0], n_jobs=-1,
            random_state=args.seed)).fit(X_tr, y_tr)
        add("6a. ExtraTrees", et.predict_proba(X_te)[:, 1], time.time() - t0)

    # 6b. HistGB *given* the lookup table as a feature.  This is the decisive
    # experiment: if the boosting model cannot beat the lookup table on its own
    # but can once the group rate is handed to it, the gap was an inability to
    # carve out 245x24 groups from raw features, not a lack of signal.
    tr_enc, te_enc = oof_group_rate(df_tr, y_tr, df_te, ["line_name", "hour"])
    B_tr = np.hstack([A_tr, tr_enc[:, None]])
    B_te = np.hstack([A_te, te_enc[:, None]])
    mask_b = np.concatenate([mask, [False]])
    t0 = time.time()
    hgb_e = HistGradientBoostingClassifier(categorical_features=mask_b,
                                           random_state=args.seed, early_stopping=False,
                                           **kw)
    sw_e = compute_sample_weight("balanced", y_tr) if best["balanced"] else None
    hgb_e.fit(B_tr, y_tr, sample_weight=sw_e)
    add("6b. HistGB tuned + OOF (line,hour) rate feature",
        hgb_e.predict_proba(B_te)[:, 1], time.time() - t0,
        "target encoding computed out-of-fold on train only")

    # 6c. Simple average of the tuned model and the lookup table.  Cheap test of
    # whether the two carry complementary information.
    p_blend = 0.5 * (r_tuned.proba + p_lut_s)
    add("6c. Blend: 0.5 * (tuned HistGB + smoothed lookup)", p_blend, 0.0,
        "rank-and-probability average of the two best single models")

    # -- 7. Calibration -----------------------------------------------------
    # Fitted on the inner validation slice, which the tuned model never saw for
    # gradient steps.  (It did inform hyperparameter choice; that is a mild,
    # disclosed dependency, and the alternative -- calibrating on the test set --
    # would be outright cheating.)
    if _HAS_FROZEN:
        base_cal = HistGradientBoostingClassifier(categorical_features=mask,
                                                  random_state=args.seed,
                                                  early_stopping=False, **kw)
        base_cal.fit(A_itr, y_tr[itr], sample_weight=(
            compute_sample_weight("balanced", y_tr[itr]) if best["balanced"] else None))
        for method in ["sigmoid", "isotonic"]:
            t0 = time.time()
            cal = CalibratedClassifierCV(FrozenEstimator(base_cal), method=method)
            cal.fit(A_iva, y_tr[iva])
            add(f"7. HistGB tuned + {method} calibration",
                cal.predict_proba(A_te)[:, 1], time.time() - t0,
                "calibrator fit on the held-out inner validation slice")
    else:
        print("  [skip] sklearn.frozen.FrozenEstimator unavailable")

    table = pd.DataFrame([{
        "model": r.name, "roc_auc": r.roc_auc, "pr_auc": r.pr_auc,
        "brier": r.brier, "pos_rate": r.pos_rate,
        "pr_auc_lift_vs_base": r.lift_vs_base, "fit_s": r.fit_seconds,
        "notes": r.notes,
    } for r in results]).sort_values("pr_auc", ascending=False)

    extras = {"hgb_search": search_df, "rf_search": pd.DataFrame(rf_rows), "y_te": y_te}
    return results, table, extras


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--thresholds", type=int, nargs="+", default=[180, 600],
                    help="delay thresholds in seconds")
    ap.add_argument("--train-frac", type=float, default=0.70,
                    help="time-ordered fraction used for training")
    ap.add_argument("--val-frac", type=float, default=0.80,
                    help="fraction OF TRAIN used to fit during tuning; the rest is "
                         "the inner validation slice")
    ap.add_argument("--n-search", type=int, default=40,
                    help="random-search draws for HistGB")
    ap.add_argument("--rf-trees", type=int, default=300)
    ap.add_argument("--n-jobs", type=int, default=-1,
                    help="parallel workers for the hyperparameter search")
    ap.add_argument("--n-boot", type=int, default=500,
                    help="paired bootstrap resamples for the PR-AUC comparison")
    ap.add_argument("--reuse-search", action="store_true",
                    help="reload hgb_search_<thr>s.csv instead of re-running the "
                         "random search (the search only ever used TRAIN)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true",
                    help="smaller searches, skip ExtraTrees (for a fast smoke run)")
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent,
                    help="where the result CSVs are written")
    args = ap.parse_args()
    if args.quick:
        args.n_search = min(args.n_search, 8)
        args.rf_trees = 100

    df = load_data(args.data)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for thr in args.thresholds:
        results, table, extras = run_threshold(df, thr, args)
        y_te_arr = extras["y_te"]

        print(f"\n--- RESULTS  (delay >= {thr} s), sorted by PR-AUC ---")
        show = table.drop(columns=["notes"]).copy()
        for c in ["roc_auc", "pr_auc", "pos_rate", "pr_auc_lift_vs_base"]:
            show[c] = show[c].map(lambda v: f"{v:.4f}")
        show["brier"] = show["brier"].map(lambda v: f"{v:.5f}")
        show["fit_s"] = show["fit_s"].map(lambda v: f"{v:.1f}")
        print(show.to_string(index=False))

        bar = next(r for r in results if r.name.startswith("2. "))
        print(f"\n  bar (line,hour lookup) PR-AUC = {bar.pr_auc:.4f}")
        # Only the contenders are bootstrapped: every extra model multiplies the
        # cost, and a model well below the bar needs no interval to be dismissed.
        shortlist = [r for r in results
                     if r.pr_auc >= bar.pr_auc * 0.95 and not r.name.startswith("1.")]
        print(f"  paired bootstrap vs the bar ({args.n_boot} resamples, "
              f"{len(shortlist)} contenders within 5% of the bar):")
        boot = bootstrap_vs_bar(y_te_arr, shortlist, bar, args.n_boot, args.seed)
        b_show = boot.copy()
        for c in ["pr_auc", "delta_vs_bar", "ci95_lo", "ci95_hi", "p_not_better"]:
            b_show[c] = b_show[c].map(lambda v: f"{v:+.4f}")
        print(b_show.to_string(index=False))
        boot.to_csv(args.out_dir / f"bootstrap_vs_bar_{thr}s.csv", index=False)

        sig = boot[(boot.significant) & (~boot.model.str.startswith("2"))]
        if len(sig):
            print("\n    SIGNIFICANTLY beats the lookup table (95% CI excludes 0):")
            for _, r in sig.iterrows():
                print(f"      {r['model']}  delta {r['delta_vs_bar']:+.4f} "
                      f"[{r['ci95_lo']:+.4f}, {r['ci95_hi']:+.4f}]")
        else:
            print("\n    NO model significantly beats the (line_name, hour) lookup "
                  "table on PR-AUC.")

        # Reliability check for the three models a report would actually discuss.
        for r in results:
            if r.name.startswith(("2. ", "5. ", "5c.", "7. HistGB tuned + isotonic")):
                print(f"\n  reliability -- {r.name}")
                print(reliability_table(y_te_arr, r.proba).to_string())

        out = args.out_dir / f"results_delay_{thr}s.csv"
        table.to_csv(out, index=False)
        extras["hgb_search"].to_csv(args.out_dir / f"hgb_search_{thr}s.csv", index=False)
        print(f"\n  written: {out}")


if __name__ == "__main__":
    main()
