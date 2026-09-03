"""Feature-block ablation and driver analysis for the Berlin delay model.

    ./.venv/bin/python analysis/experiments/ablation.py                # everything
    ./.venv/bin/python analysis/experiments/ablation.py --stage weather
    ./.venv/bin/python analysis/experiments/ablation.py --stage ablation --stage drivers

WHY THIS SCRIPT EXISTS
`analysis/signal_check.py` answers "is the delay tail predictable at all?" and
says yes (ROC-AUC ~0.80). That number on its own is close to worthless for the
research question, because it does not say WHAT the model learned. Three
specific worries motivated every experiment below, and each one is a way the
headline number could be honest-looking but scientifically empty:

  W1  The model may only have memorised which lines are chronically late.
      A lookup table of historical rates would then be the whole result, and
      "machine learning predicts delays" would be a dressed-up group mean.
      -> stage `generalise`: strip identity, and evaluate on lines the model
         has never seen. A lookup table scores chance on an unseen line; a
         model that learned transferable structure does not.

  W2  The apparent weather contribution may be a confound. Temperature is not
      an independent variable here: it correlates 0.47 with hour-of-day, and
      with only ~21 usable days the hourly weather series is very nearly a
      unique fingerprint of "which day is this". A tree can use temp_c as a
      date index and get credit for "weather".
      -> stage `weather`: day-block placebo shuffles that keep the marginal
         weather distribution and the diurnal shape but destroy the real
         day->delay pairing, plus residualisation against hour and day.

  W3  Permutation importance is unreliable when features are correlated (it
      evaluates the model off the data manifold), and hour/temperature here
      certainly are. A single importance ranking would be overconfident.
      -> stage `drivers`: permutation, SHAP and drop-column importance side by
         side, and the disagreements reported rather than hidden.

LEAKAGE POLICY
Every field describing the realtime observation of the departure being predicted
is excluded by construction: first_delay_s, delay_drift_s, n_obs, lead_time_s,
first/last_observed_at, hour_utc, trip_id and the target itself. Those belong to
a separate nowcast experiment. `assert_no_leakage()` fails loudly rather than
trusting the feature lists to stay correct as the dataset evolves.

SPLIT
Always chronological, first 70% train / last 30% test, never random. Departures
on the same day share weather and share whatever disruption happened that day,
so a random split would let the model see the answer for the day it is scoring.
"""
from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data" / "dataset.parquet"

SEED = 0
TRAIN_FRAC = 0.70

# ---------------------------------------------------------------- feature blocks
# `direction` is deliberately counted as IDENTITY, not as its own thing: the
# direction string is a terminus name, which is close to a unique key for the
# line-and-branch. Treating it as a neutral feature would smuggle line identity
# into every "no identity" model and quietly answer W1 the wrong way.
BLOCKS: dict[str, list[str]] = {
    "IDENTITY": ["line_name", "stop_id", "product", "direction"],
    "TIME": ["hour", "minute", "weekday", "month", "day_of_year",
             "is_weekend", "is_morning_peak", "is_evening_peak"],
    "CALENDAR": ["is_school_holiday", "is_public_holiday", "is_full_traffic_day"],
    "WEATHER": ["temp_c", "humidity_pct", "precip_mm", "precip_form", "wind_ms",
                "wind_dir_deg", "is_rain", "is_wet", "is_snow", "is_freezing",
                "is_hot"],
}
CATEGORICAL = {"line_name", "stop_id", "product", "direction"}

# Never features. Checked at runtime, not just documented.
FORBIDDEN = ["first_delay_s", "delay_drift_s", "n_obs", "lead_time_s",
             "last_observed_at", "first_observed_at", "hour_utc", "trip_id",
             "final_delay_s"]

THRESHOLDS = [180, 600]

# Fixed everywhere so ablation deltas reflect the features, not a retuned model.
HGB_KW = dict(random_state=SEED, max_iter=300, learning_rate=0.06,
              early_stopping=False, categorical_features="from_dtype")


# ------------------------------------------------------------------------ data
def load() -> pd.DataFrame:
    df = pd.read_parquet(DATASET)
    df = df[(df.cancelled == 0) & df.final_delay_s.notna()]
    df = df.sort_values("planned_when").reset_index(drop=True)
    df["date"] = df.planned_when.dt.date
    return df


def all_features() -> list[str]:
    return [f for b in BLOCKS.values() for f in b]


def assert_no_leakage(cols) -> None:
    bad = sorted(set(cols) & set(FORBIDDEN))
    if bad:
        raise AssertionError(f"leaking features in matrix: {bad}")


def design_matrix(df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    assert_no_leakage(feats)
    X = df[feats].copy()
    for c in feats:
        if c in CATEGORICAL:
            X[c] = X[c].astype("category")
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
    return X


def split_idx(n: int) -> tuple[np.ndarray, np.ndarray]:
    cut = int(n * TRAIN_FRAC)
    return np.arange(cut), np.arange(cut, n)


# --------------------------------------------------------------------- scoring
def fit_score(Xtr, ytr, Xte, yte, return_model=False):
    """One chronological fit. Returns (roc, pr) or (roc, pr, model, p)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score

    if Xtr.shape[1] == 0:                      # empty feature set -> constant
        p = np.full(len(yte), ytr.mean())
        return (0.5, yte.mean()) if not return_model else (0.5, yte.mean(), None, p)
    m = HistGradientBoostingClassifier(**HGB_KW).fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    roc, pr = roc_auc_score(yte, p), average_precision_score(yte, p)
    return (roc, pr, m, p) if return_model else (roc, pr)


def day_block_ci(y, p, days, n_boot=400, rng=None):
    """Day-block bootstrap CI for ROC-AUC.

    WHY blocks and not plain row resampling: departures on the same day share
    the weather and share whatever went wrong that day, so rows are nowhere near
    independent. Resampling rows would give a CI of ~+/-0.005 and badly overstate
    the precision. The test window holds only ~7 days, so even this CI is coarse
    -- it is reported to show the order of magnitude of the noise, not as a
    publication-grade interval.
    """
    from sklearn.metrics import roc_auc_score
    rng = rng or np.random.default_rng(SEED)
    uniq = np.unique(days)
    idx_by_day = {d: np.where(days == d)[0] for d in uniq}
    out = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        ii = np.concatenate([idx_by_day[d] for d in pick])
        if 0 < y[ii].sum() < len(ii):
            out.append(roc_auc_score(y[ii], p[ii]))
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))) if out else (np.nan, np.nan)


def header(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def sc(df, feats, y, tr, te):
    """Shorthand: build the matrix and score one chronological fit."""
    X = design_matrix(df, feats)
    return fit_score(X.iloc[tr], y[tr], X.iloc[te], y[te])


# --------------------------------------------------------------- stage: dataset
def stage_data(df: pd.DataFrame) -> None:
    header("STAGE 0 -- what the data can and cannot support")
    tr, te = split_idx(len(df))
    per_day = df.groupby("date").size()
    real_days = per_day[per_day > 500]
    print(f"rows {len(df):,}   span {df.planned_when.min():%Y-%m-%d} .. {df.planned_when.max():%Y-%m-%d}")
    print(f"calendar days {per_day.size}, but only {len(real_days)} with >500 departures")
    print("  thin days (logger outage):",
          ", ".join(f"{d} n={n}" for d, n in per_day[per_day <= 500].items()))
    print(f"\nsplit  train {df.planned_when.iloc[0]:%m-%d} .. {df.planned_when.iloc[tr[-1]]:%m-%d} "
          f"({len(tr):,})   test {df.planned_when.iloc[te[0]]:%m-%d} .. "
          f"{df.planned_when.iloc[-1]:%m-%d} ({len(te):,})")
    for t in THRESHOLDS:
        y = (df.final_delay_s >= t).astype(int).values
        print(f"  >={t:>3}s  overall {y.mean():.4f}   train {y[tr].mean():.4f}   "
              f"test {y[te].mean():.4f}   test events {int(y[te].sum()):,}")

    # The single most important limitation for the weather half of the question.
    print(f"\nEFFECTIVE SAMPLE SIZE FOR WEATHER: weather is constant within an hour and "
          f"nearly constant within a day.\n  n={len(df):,} departures, but only "
          f"{len(real_days)} independent weather-days ({df.groupby('date').temp_c.nunique().sum()} "
          f"distinct hourly weather rows).\n  Any weather claim rests on ~{len(real_days)} points, "
          "not 200k. Treat weather effect sizes as anecdotal.")

    # Zero-variance features are worth naming: they cannot contribute and their
    # 0.000 importance is a property of August, not evidence about winter.
    const = [f for f in all_features() if df[f].nunique(dropna=False) <= 1]
    print(f"\nZERO-VARIANCE features in this window (cannot contribute by construction): {const}")
    print("  -> is_snow / is_freezing / is_public_holiday are structurally untestable in August.")

    nat = df[df.temp_c.isna()]
    if len(nat):
        print(f"\nWEATHER MISSING for {len(nat):,} rows ({len(nat)/len(df):.1%}), dates "
              f"{sorted(set(nat.date))} -- all inside the TEST window.")
        print("  DWD publishes with a lag; the last logged day has no weather yet.")

    print(f"\ncorr(temp_c, hour)        = {df[['temp_c','hour']].corr().iloc[0,1]:+.3f}   "
          "<- temperature is largely a time-of-day variable here")
    print(f"corr(temp_c, day_of_year) = {df[['temp_c','day_of_year']].corr().iloc[0,1]:+.3f}   "
          "<- 24 days of late summer show no seasonal trend to speak of")


# -------------------------------------------------------------- stage: ablation
def stage_ablation(df: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-block-out and block-alone, for both severity thresholds.

    Both directions are needed and they answer different questions. LOO measures
    UNIQUE contribution (what is lost that nothing else can supply); alone
    measures TOTAL information content (including what is shared with other
    blocks). A block can be individually strong and still have LOO ~0 if another
    block carries the same information -- reporting only one of the two is the
    classic way to overstate a feature's role.
    """
    header("STAGE 1 -- feature-block ablation")
    tr, te = split_idx(len(df))
    rows = []
    for t in THRESHOLDS:
        y = (df.final_delay_s >= t).astype(int).values
        base = y[te].mean()
        full_roc, full_pr = sc(df, all_features(), y, tr, te)
        print(f"\n--- target: final_delay_s >= {t}s   (test positive rate {base:.4f}) ---")
        print(f"{'config':<20}{'ROC-AUC':>9}{'dROC':>9}{'PR-AUC':>9}{'dPR':>9}{'PR lift':>9}")

        configs = [("FULL", all_features()), ("NONE (base rate)", [])]
        configs += [(f"-{b}", [f for f in all_features() if f not in fs])
                    for b, fs in BLOCKS.items()]
        configs += [(f"{b} alone", fs) for b, fs in BLOCKS.items()]
        for name, feats in configs:
            roc, pr = (full_roc, full_pr) if name == "FULL" else sc(df, feats, y, tr, te)
            rows.append(dict(threshold=t, config=name, n_feat=len(feats), roc=roc, pr=pr))
            print(f"{name:<20}{roc:>9.4f}{roc-full_roc:>+9.4f}{pr:>9.4f}"
                  f"{pr-full_pr:>+9.4f}{pr/base:>9.1f}x", flush=True)

        got = {r["config"]: r for r in rows if r["threshold"] == t}
        ident = got["IDENTITY alone"]["roc"]
        print(f"\n  identity alone recovers {(ident-0.5)/(full_roc-0.5):.1%} of the full model's "
              f"ROC-AUC gain over chance ({ident:.4f} vs {full_roc:.4f}, chance 0.5)")
        print(f"  identity alone recovers {got['IDENTITY alone']['pr']/full_pr:.1%} of full PR-AUC")
    return pd.DataFrame(rows)


# ------------------------------------------------------- stage: does it generalise
def lookup_baseline(df, y, tr, te, keys, prior_w=20.0):
    """Smoothed historical-rate lookup: the 'lookup table with extra steps'.

    This is the null hypothesis for W1. If gradient boosting cannot beat a table
    of historical rates by a meaningful margin, the modelling adds nothing and
    the honest headline is "some lines are chronically late", not "delays are
    predictable". Empirical-Bayes smoothing toward the global rate keeps rare
    keys from scoring 0 or 1 on two observations; unseen keys fall back to the
    global rate, which is exactly what a lookup table must do.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score
    g = df.iloc[tr].assign(_y=y[tr]).groupby(keys, observed=True)._y.agg(["sum", "size"])
    prior = y[tr].mean()
    rate = (g["sum"] + prior_w * prior) / (g["size"] + prior_w)
    key_te = df.iloc[te].set_index(keys).index
    p = pd.Series(key_te.map(rate), dtype=float).fillna(prior).values
    return roc_auc_score(y[te], p), average_precision_score(y[te], p)


def stage_generalise(df: pd.DataFrame) -> None:
    """W1: is this a predictive model, or a lookup table with extra steps?

    Two independent tests, because either alone is arguable:
      (a) remove identity entirely -- what predictive power survives?
      (b) hold whole lines out of TRAINING and score only those lines in the
          test window. A lookup table is structurally incapable here: it has no
          row for an unseen line. A model that learned "buses at 17:00 in the
          rain run late" transfers; one that memorised "M41 is late" does not.
    """
    header("STAGE 2 -- THE KEY QUESTION: beyond memorising chronically late lines?")
    tr, te = split_idx(len(df))
    ident_strict = ["line_name", "stop_id"]          # the memorisable keys
    ident_full = BLOCKS["IDENTITY"]                  # + product + direction

    for t in THRESHOLDS:
        y = (df.final_delay_s >= t).astype(int).values
        print(f"\n--- target >= {t}s ---")

        # ---- (a) what survives without identity, and what a table achieves
        variants = {
            "FULL": all_features(),
            "no line_name/stop_id": [f for f in all_features() if f not in ident_strict],
            "no IDENTITY at all": [f for f in all_features() if f not in ident_full],
            "product+TIME+WEATHER": ["product"] + BLOCKS["TIME"] + BLOCKS["WEATHER"],
        }
        print(f"{'model':<24}{'ROC-AUC':>9}{'PR-AUC':>9}   {'day-block 95% CI (ROC)':>24}")
        days_te = df.date.values[te]
        for name, feats in variants.items():
            X = design_matrix(df, feats)
            roc, pr, _, p = fit_score(X.iloc[tr], y[tr], X.iloc[te], y[te], return_model=True)
            lo, hi = day_block_ci(y[te], p, days_te)
            print(f"{name:<24}{roc:>9.4f}{pr:>9.4f}   [{lo:.4f}, {hi:.4f}]")

        print(f"\n{'lookup table (no ML)':<24}{'ROC-AUC':>9}{'PR-AUC':>9}")
        for keys in (["line_name"], ["line_name", "hour"],
                     ["line_name", "hour", "weekday"], ["line_name", "stop_id", "hour"]):
            roc, pr = lookup_baseline(df, y, tr, te, keys)
            print(f"{'x'.join(keys):<24}{roc:>9.4f}{pr:>9.4f}")

        # ---- (b) unseen-line generalisation
        # Hold out lines by a hash of the name, not at random per run, so the
        # held-out set is reproducible. Only lines with enough test rows are
        # scored, otherwise the AUC is noise.
        rng = np.random.default_rng(SEED)
        lines = df.line_name.dropna().unique()
        held = set(rng.choice(lines, size=max(1, int(0.25 * len(lines))), replace=False))
        is_held = df.line_name.isin(held).values

        tr_seen = tr[~is_held[tr]]                     # training sees no held line
        te_held = te[is_held[te]]
        te_seen = te[~is_held[te]]
        if len(te_held) < 500 or y[te_held].sum() < 20:
            print(f"\n  unseen-line test skipped: only {len(te_held)} rows / "
                  f"{int(y[te_held].sum())} events in held-out lines")
            continue

        print(f"\nUNSEEN-LINE TEST: {len(held)} of {len(lines)} lines removed from training "
              f"entirely.\n  scored on {len(te_held):,} test rows of those lines "
              f"({int(y[te_held].sum())} events, rate {y[te_held].mean():.4f})")
        print(f"{'trained w/o held lines':<24}{'unseen lines':>14}{'seen lines':>12}")
        for name, feats in (("with identity", all_features()),
                            ("no line_name/stop_id",
                             [f for f in all_features() if f not in ident_strict]),
                            ("no IDENTITY at all",
                             [f for f in all_features() if f not in ident_full])):
            X = design_matrix(df, feats)
            # Categories must cover held-out lines so the matrix stays alignable;
            # HGB simply never saw them at fit time, which is the point.
            _, _, m, _ = fit_score(X.iloc[tr_seen], y[tr_seen],
                                   X.iloc[te_held], y[te_held], return_model=True)
            from sklearn.metrics import roc_auc_score
            ph = m.predict_proba(X.iloc[te_held])[:, 1]
            ps = m.predict_proba(X.iloc[te_seen])[:, 1]
            lo, hi = day_block_ci(y[te_held], ph, df.date.values[te_held])
            print(f"{name:<24}{roc_auc_score(y[te_held], ph):>14.4f}"
                  f"{roc_auc_score(y[te_seen], ps):>12.4f}   unseen 95% CI "
                  f"[{lo:.4f}, {hi:.4f}]")
        rocl, _ = lookup_baseline(df.iloc[np.concatenate([tr_seen, te_held])],
                                  y[np.concatenate([tr_seen, te_held])],
                                  np.arange(len(tr_seen)),
                                  np.arange(len(tr_seen), len(tr_seen) + len(te_held)),
                                  ["line_name", "hour"])
        print(f"{'lookup line x hour':<24}{rocl:>14.4f}   <- a table cannot see an unseen line; "
              "this is the memorisation-only floor")


# ------------------------------------------------------------- stage: weather
def stage_weather(df: pd.DataFrame) -> None:
    """W2: is the weather contribution real, or a proxy for time and day?

    The design of the placebo matters. A plain row-wise shuffle of the weather
    columns is too weak a null: it destroys the diurnal temperature curve as
    well, so beating it only proves temperature knows the time of day. The
    day-block shuffle reassigns each day's ENTIRE hourly weather series to a
    different day. It preserves the marginal distribution, the within-day
    diurnal shape and the autocorrelation, and destroys only the true pairing of
    weather to delays. Weather gain above THAT null is the real weather effect.
    """
    header("STAGE 3 -- is weather real, or a confound?")
    tr, te = split_idx(len(df))
    wx = BLOCKS["WEATHER"]
    non_wx = [f for f in all_features() if f not in wx]

    print("Correlation structure of the 'weather' block with time (n rows, not n days):")
    cc = df[["temp_c", "humidity_pct", "wind_ms", "precip_mm",
             "hour", "day_of_year"]].corr()
    print(cc.loc[["temp_c", "humidity_pct", "wind_ms", "precip_mm"], ["hour", "day_of_year"]]
          .round(3).to_string())
    print("\n  temp_c vs hour = strong. temp_c vs day_of_year = negligible.")
    print("  => in THIS window the confound is DIURNAL, not seasonal. The data span")
    print("     2026-08-10..09-03; there is no autumn or winter in it to create a")
    print("     seasonal trend. Any claim about seasonality is out of sample.")

    for t in THRESHOLDS:
        y = (df.final_delay_s >= t).astype(int).values
        print(f"\n--- target >= {t}s ---")

        # Context matters: a block's value depends on what it is added to.
        contexts = {
            "WEATHER alone": (wx, []),
            "TIME -> +WEATHER": (BLOCKS["TIME"] + wx, BLOCKS["TIME"]),
            "IDENTITY+TIME -> +WEATHER": (BLOCKS["IDENTITY"] + BLOCKS["TIME"] + wx,
                                          BLOCKS["IDENTITY"] + BLOCKS["TIME"]),
            "FULL -> -WEATHER": (all_features(), non_wx),
        }
        print(f"{'context':<28}{'with wx':>9}{'without':>9}{'gain':>9}")
        gains = {}
        for name, (fa, fb) in contexts.items():
            ra, _ = sc(df, fa, y, tr, te)
            rb, _ = sc(df, fb, y, tr, te) if fb else (0.5, 0.0)
            gains[name] = ra - rb
            print(f"{name:<28}{ra:>9.4f}{rb:>9.4f}{ra-rb:>+9.4f}", flush=True)

        # ---- placebo: null distribution of the weather gain
        real_gain = gains["FULL -> -WEATHER"]
        rb, _ = sc(df, non_wx, y, tr, te)
        rng = np.random.default_rng(SEED)
        days = np.array(sorted(df.date.unique()))
        null = []
        n_perm = 10          # each permutation is a full refit; 10 is enough to
                             # see whether the real gain is inside the null cloud
        # weather lookup keyed by (day, hour); the placebo swaps only the day part
        wx_by_dh = df.groupby(["date", "hour"])[wx].first()
        for k in range(n_perm):
            # derangement-ish: a random permutation of days, retried so that few
            # days map to themselves (a self-map would leak real weather back in)
            for _ in range(20):
                perm = rng.permutation(days)
                if (perm == days).mean() < 0.1:
                    break
            mapping = dict(zip(days, perm))
            key = pd.MultiIndex.from_arrays([df.date.map(mapping).values, df.hour.values])
            fake = df.copy()
            fake[wx] = wx_by_dh.reindex(key).values      # donor day, SAME hour
            rf, _ = sc(fake, all_features(), y, tr, te)
            null.append(rf - rb)
        null = np.array(null)
        p_emp = (np.sum(null >= real_gain) + 1) / (len(null) + 1)
        print(f"\n  day-block placebo (weather series swapped between days, {n_perm} draws):")
        print(f"    real weather gain      {real_gain:+.4f}")
        print(f"    placebo gain mean/sd   {null.mean():+.4f} / {null.std():.4f}   "
              f"range [{null.min():+.4f}, {null.max():+.4f}]")
        print(f"    empirical p            {p_emp:.3f}  "
              f"({'weather beats its own placebo' if p_emp <= 0.1 else 'INDISTINGUISHABLE from a fake weather series'})")

        # ---- residualised temperature: strip the diurnal and day-level means
        # If temp_c only matters as a clock or a date, the residual is noise and
        # a model given the residual instead of the raw value should lose the gain.
        d2 = df.copy()
        d2["temp_resid"] = (df.temp_c
                            - df.groupby("hour").temp_c.transform("mean")
                            - df.groupby("date").temp_c.transform("mean")
                            + df.temp_c.mean())
        r_raw, _ = sc(df, non_wx + ["temp_c"], y, tr, te)
        r_res, _ = sc(d2, non_wx + ["temp_resid"], y, tr, te)
        print(f"\n  non-weather + raw temp_c      {r_raw:.4f}  (gain {r_raw-rb:+.4f})")
        print(f"  non-weather + residual temp   {r_res:.4f}  (gain {r_res-rb:+.4f})")
        print("    residual = temp_c minus its hour-of-day mean minus its day mean:")
        print("    what is left is deviation from the normal temperature at that hour on that day.")

    # ---- day-level picture, where the real n lives
    print("\nDAY-LEVEL VIEW (this is the honest resolution of the weather question):")
    g = df.groupby("date").agg(n=("final_delay_s", "size"),
                               late180=("final_delay_s", lambda s: (s >= 180).mean()),
                               late600=("final_delay_s", lambda s: (s >= 600).mean()),
                               temp=("temp_c", "mean"), rain_h=("is_rain", "mean"),
                               precip=("precip_mm", "mean"), wind=("wind_ms", "mean"))
    g = g[g.n > 500]
    print(g.round(4).to_string())
    # THIS is the test with the correct sample size. n ~ 21 days, so only a very
    # large correlation could reach significance -- which is itself the finding.
    from scipy import stats as st
    print(f"\n  n = {len(g)} days. Pearson r across days (the honest unit of analysis):")
    for c in ["temp", "rain_h", "precip", "wind"]:
        sub = g[[c, "late180", "late600"]].dropna()
        if len(sub) > 3:
            r1, p1 = st.pearsonr(sub[c], sub.late180)
            r6, p6 = st.pearsonr(sub[c], sub.late600)
            print(f"    {c:<8} vs late180 r={r1:+.3f} (p={p1:.3f})   "
                  f"vs late600 r={r6:+.3f} (p={p6:.3f})   n={len(sub)}")
    print("    NOTE: 4 predictors x 2 targets = 8 tests, uncorrected. Treat p<0.05 here")
    print("    as a hint to re-test on autumn data, not as an established effect.")


# ------------------------------------------------------------- stage: drivers
def stage_drivers(df: pd.DataFrame) -> None:
    """W3: which factors actually drive delay, cross-checked three ways.

    Permutation importance breaks correlations and evaluates the model on
    impossible rows (temp 30 C at 04:00). Drop-column importance retrains and so
    stays on the manifold, but charges a feature nothing when a correlated
    feature can replace it. SHAP attributes the model's actual output. They
    measure different things; where they disagree, the disagreement is the
    finding.
    """
    header("STAGE 4 -- driver analysis (permutation / SHAP / drop-column)")
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import roc_auc_score

    tr, te = split_idx(len(df))
    feats = all_features()
    X = design_matrix(df, feats)

    for t in THRESHOLDS:
        y = (df.final_delay_s >= t).astype(int).values
        roc, pr, m, p = fit_score(X.iloc[tr], y[tr], X.iloc[te], y[te], return_model=True)
        print(f"\n--- target >= {t}s   full model ROC {roc:.4f} / PR {pr:.4f} ---")

        # Permutation importance, scored by ROC-AUC on the TEST set only.
        r = permutation_importance(m, X.iloc[te], y[te], scoring="roc_auc",
                                   n_repeats=5, random_state=SEED, n_jobs=1)
        perm = pd.Series(r.importances_mean, index=feats).sort_values(ascending=False)
        perm_sd = pd.Series(r.importances_std, index=feats)

        # Drop-column: refit without the feature. Expensive, so only the top ones.
        top = list(perm.head(8).index)
        drop = {}
        for f in top:
            rr, _ = sc(df, [g for g in feats if g != f], y, tr, te)
            drop[f] = roc - rr

        # SHAP if available; mean |value| over a test subsample.
        shap_imp = {}
        try:
            import shap
            sub = np.random.default_rng(SEED).choice(te, size=min(4000, len(te)), replace=False)
            ex = shap.TreeExplainer(m)
            sv = ex.shap_values(X.iloc[sub])
            sv = sv[..., 1] if getattr(sv, "ndim", 2) == 3 else sv
            shap_imp = dict(zip(feats, np.abs(sv).mean(0)))
        except Exception as e:                          # shap is optional
            print(f"  (shap unavailable: {type(e).__name__}: {e})")

        print(f"{'feature':<20}{'perm dROC':>11}{'+/-sd':>8}{'drop-col':>10}{'SHAP |v|':>10}")
        for f in perm.index[:14]:
            d = f"{drop[f]:+.4f}" if f in drop else "     -"
            s = f"{shap_imp[f]:.4f}" if f in shap_imp else "     -"
            print(f"{f:<20}{perm[f]:>+11.4f}{perm_sd[f]:>8.4f}{d:>10}{s:>10}")

        # Block-level permutation: shuffle a whole block at once, so correlated
        # members cannot cover for each other. This is the number to quote for
        # "how much does weather matter", not the sum of single-feature values.
        print(f"\n  block-level permutation (whole block shuffled together):")
        rng = np.random.default_rng(SEED)
        for b, fs in BLOCKS.items():
            if not fs:
                continue
            drops = []
            for _ in range(5):
                Xp = X.iloc[te].copy()
                order = rng.permutation(len(Xp))
                for f in fs:
                    Xp[f] = Xp[f].values[order]
                drops.append(roc - roc_auc_score(y[te], m.predict_proba(Xp)[:, 1]))
            print(f"    {b:<12}{np.mean(drops):>+9.4f}  (sd {np.std(drops):.4f})")


# ------------------------------------------------------- stage: effects for >=600
def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def stage_effects(df: pd.DataFrame) -> None:
    """Effect sizes and direction for SERIOUS delay (>=600s), the payload of the
    research question. Model-based partial dependence AND raw empirical rates are
    both shown: PDP is what the model believes, empirical rates are what happened.
    Where they diverge, believe the data.
    """
    header("STAGE 5 -- what drives SERIOUS delay (>= 600 s = 10 min)")
    t = 600
    y = (df.final_delay_s >= t).astype(int).values
    base = y.mean()
    print(f"baseline rate {base:.4f} ({int(y.sum()):,} events in {len(df):,} departures)\n")

    def table(col, label, min_n=800, top=12, order_by="rate"):
        g = df.assign(_y=y).groupby(col, observed=True)._y.agg(["sum", "size"])
        g = g[g["size"] >= min_n]
        g["rate"] = g["sum"] / g["size"]
        g["rr"] = g["rate"] / base
        g[["lo", "hi"]] = pd.DataFrame([wilson(a, b) for a, b in zip(g["sum"], g["size"])],
                                       index=g.index)
        g = g.sort_values(order_by, ascending=False)
        # show the worst `top` and, if there are more, the best few for contrast
        show = pd.concat([g.head(top), g.tail(4)])
        show = show[~show.index.duplicated()]
        print(f"-- {label} (>= {min_n} departures) --")
        print(f"{'value':<22}{'n':>8}{'rate':>8}{'x base':>8}{'95% CI':>18}")
        for k, r in show.iterrows():
            print(f"{str(k):<22}{int(r['size']):>8,}{r['rate']:>8.4f}{r['rr']:>8.2f}"
                  f"   [{r['lo']:.4f},{r['hi']:.4f}]")
        print()

    table("product", "product", min_n=1000, top=6)
    table("line_name", "line (top and bottom)", min_n=1500, top=10)
    table("hour", "hour of day (local)", min_n=2000, top=24)
    table("stop_id", "stop", min_n=1000, top=6)
    table("weekday", "weekday (0=Mon)", min_n=2000, top=7)
    table("is_rain", "raining in that hour", min_n=1000, top=2)
    table("is_wet", "wet", min_n=1000, top=2)
    table("is_hot", "hot", min_n=1000, top=2)
    table("is_school_holiday", "school holiday", min_n=1000, top=2)
    table("is_weekend", "weekend", min_n=1000, top=2)

    # temperature binned: the direction of the temp effect, unadjusted
    d = df.assign(_y=y, tbin=pd.cut(df.temp_c, [-50, 12, 16, 20, 24, 28, 50]))
    g = d.groupby("tbin", observed=True)._y.agg(["sum", "size"])
    g["rate"] = g["sum"] / g["size"]
    print("-- temperature bins (UNADJUSTED: confounded with hour of day) --")
    print(f"{'bin':<16}{'n':>9}{'rate':>8}{'x base':>8}")
    for k, r in g.iterrows():
        print(f"{str(k):<16}{int(r['size']):>9,}{r['rate']:>8.4f}{r['rate']/base:>8.2f}")

    # ...and adjusted, by holding hour fixed. If the temperature effect is a
    # clock effect, it collapses within an hour band.
    print("\n-- temperature effect WITHIN fixed hour bands (adjusted) --")
    print(f"{'hour band':<12}{'cool half':>12}{'warm half':>12}{'ratio':>8}{'n':>10}")
    for lo, hi in [(6, 10), (10, 15), (15, 19), (19, 23)]:
        s = d[(d.hour >= lo) & (d.hour < hi) & d.temp_c.notna()]
        if len(s) < 2000:
            continue
        med = s.temp_c.median()
        cool, warm = s[s.temp_c <= med]._y, s[s.temp_c > med]._y
        ratio = warm.mean() / cool.mean() if cool.mean() > 0 else np.nan
        print(f"{f'{lo:02d}-{hi:02d}':<12}{cool.mean():>12.4f}{warm.mean():>12.4f}"
              f"{ratio:>8.2f}{len(s):>10,}")

    # model-based partial dependence for the top numerics
    print("\n-- partial dependence (full model, test window) --")
    from sklearn.inspection import partial_dependence
    tr, te = split_idx(len(df))
    X = design_matrix(df, all_features())
    _, _, m, _ = fit_score(X.iloc[tr], y[tr], X.iloc[te], y[te], return_model=True)
    sub = X.iloc[np.random.default_rng(SEED).choice(te, size=min(6000, len(te)), replace=False)]
    for f in ["hour", "temp_c", "wind_ms", "humidity_pct", "precip_mm", "day_of_year"]:
        try:
            pd_res = partial_dependence(m, sub, [f], kind="average", grid_resolution=8)
            gv = pd_res["grid_values"][0]
            av = pd_res["average"][0]
            print(f"  {f:<14}" + "  ".join(f"{a:.0f}:{b:.3f}" for a, b in zip(gv, av)))
        except Exception as e:
            print(f"  {f}: {type(e).__name__}")
    print("  (values are predicted P(delay>=600s); read the SHAPE, not the level)")


# ----------------------------------------------------------------------- main
STAGES = {"data": stage_data, "ablation": stage_ablation, "generalise": stage_generalise,
          "weather": stage_weather, "drivers": stage_drivers, "effects": stage_effects}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", action="append", choices=list(STAGES),
                    help="run only these stages (default: all, in order)")
    args = ap.parse_args()
    stages = args.stage or list(STAGES)

    df = load()
    assert_no_leakage(all_features())
    t0 = time.time()
    for s in stages:
        STAGES[s](df)
    print(f"\n[done in {time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
