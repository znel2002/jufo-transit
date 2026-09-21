"""Does delay PROPAGATE through the network? A viability test for a project pivot.

    ./.venv/bin/python analysis/experiments/propagation.py            # full run
    ./.venv/bin/python analysis/experiments/propagation.py --sample 150000

WHY THIS EXISTS
---------------
The current framing ("predict lateness from calendar + weather") is a dead end for a
write-up: a gradient-boosting model beats a two-column (line, hour) lookup table by
+0.007 ROC-AUC at >=3 min, and an ablation showed line identity carries nearly all the
signal. That is a correlational result about *who* is late, not about *why* or *how*
lateness spreads.

This script tests a different research question: delay as something that PROPAGATES.
A vehicle late at one stop should still be late at the next; congestion on shared
infrastructure should spill between lines. That is a claim about mechanism, it is
falsifiable, and it uses the multi-stop structure the dataset already has (one row per
(trip_id, stop_id, planned_when), 4 Berlin interchange stops, so a Stadtbahn train is
observed up to four times on one run).

It is organised as four sections matching four questions:
  S0  Is the multi-stop structure real, or an artefact of the feed?
  S1  How much delay is carried / gained / lost between consecutive stops?
  S2  Does the upstream observation beat the (line, hour) lookup baseline?
  S3  Does recent delay on OTHER lines at the same stop predict this one?

THE CENTRAL METHODOLOGICAL DANGER
---------------------------------
The BVG/HAFAS feed reports a *current estimate* of each departure's delay. A realtime
feed can, and often does, propagate a vehicle's current delay forward onto its own
future stops. If it does that naively, then "delay at stop A predicts delay at stop B"
would measure the FEED'S internal model, not Berlin's transit network. S0 tests this
directly, and every number downstream has to be read in its light.

Second danger: leakage. The upstream observation must be genuinely earlier in WALL-CLOCK
time than the moment we claim to predict at. We enforce that on `last_observed_at`
(when the number actually became visible to us), not on `planned_when` (the timetable),
and we report a strict variant that additionally demands >=5 minutes of real lead.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "dataset.parquet"

# Lateness thresholds, in seconds. 180 s is the German "puenktlich" convention
# (a departure counts as on time below 6 min in practice, but 3 min is the stricter
# cut the rest of this project already uses); 600 s is "the trip is ruined".
THRESHOLDS = [180, 600]

# Contagion look-back window: how far into the past a "recent delay on other lines"
# aggregate may reach. 30 min is roughly one congestion episode on a Berlin corridor.
CONTAGION_WINDOW_S = 1800

# Strict-variant lead requirement (see docstring): the upstream number must have been
# visible at least this long before the target's scheduled departure.
STRICT_LEAD_S = 300

STOP_NAMES = {
    "900100003": "Alexanderplatz",
    "900023201": "Zoologischer Garten",
    "900120004": "Warschauer Str.",
    "900003201": "Hauptbahnhof",
}


def log(*a):
    print(*a, flush=True)


def rule(title: str):
    log("\n" + "=" * 78)
    log(title)
    log("=" * 78)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load(sample: int | None) -> pd.DataFrame:
    df = pd.read_parquet(DATA)
    # Cancelled departures have no meaningful delay, and rows whose last estimate was
    # never captured have no label. Both are dropped rather than imputed: imputing a
    # delay would invent exactly the quantity under study.
    df = df[(df["cancelled"] == 0) & df["final_delay_s"].notna()].copy()
    if sample:
        # Subsample by TRIP, never by row -- sampling rows would destroy the
        # multi-stop structure that is the whole point of this analysis.
        trips = df["trip_id"].drop_duplicates()
        keep = trips.sample(min(sample, len(trips)), random_state=0)
        df = df[df["trip_id"].isin(set(keep))].copy()
    df = df.sort_values(["trip_id", "planned_when"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# S0  Is the multi-stop structure real?
# ---------------------------------------------------------------------------
def section0(df: pd.DataFrame) -> dict:
    rule("S0  STRUCTURE, AND WHETHER THE FEED IS FAKING IT")
    per_trip_stops = df.groupby("trip_id")["stop_id"].nunique()
    dist = per_trip_stops.value_counts().sort_index()
    log("trips observed at N distinct stops (after dropping cancelled/unlabelled):")
    for n, c in dist.items():
        log(f"   {n} stop(s): {c:>7,} trips")
    multi = df[df["trip_id"].map(per_trip_stops) > 1]
    log(f"\nmulti-stop trips: {multi.trip_id.nunique():,}  rows: {len(multi):,}")
    log("product mix of multi-stop rows:")
    for p, c in multi["product"].value_counts().items():
        log(f"   {p:<10} {c:>7,}")

    # THE ARTEFACT TEST. If the feed simply broadcast one trip-level delay to every
    # stop, then every multi-stop trip would show an identical number everywhere.
    # A high identical-rate overall proves nothing (most departures are simply on
    # time at every stop); the test that bites is restricted to trips that were
    # actually delayed somewhere.
    g = multi.groupby("trip_id")["final_delay_s"]
    agg = pd.DataFrame({"nuniq": g.nunique(), "mx": g.max(), "mn": g.min(), "n": g.size()})
    agg = agg[agg["n"] > 1]
    out = {
        "multi_stop_trips": int(multi.trip_id.nunique()),
        "identical_all": float((agg["nuniq"] == 1).mean()),
    }
    log(f"\nidentical delay at every stop, all multi-stop trips: {out['identical_all']:.1%}")
    for cut, label in [(1, "any nonzero delay"), (180, "max delay >= 180 s"), (600, "max delay >= 600 s")]:
        sub = agg[agg["mx"] >= cut]
        frac = float((sub["nuniq"] == 1).mean()) if len(sub) else float("nan")
        spread = (sub["mx"] - sub["mn"]).median() if len(sub) else float("nan")
        out[f"identical_ge{cut}"] = frac
        log(f"   restricted to trips with {label:<20} n={len(sub):>6,}  identical: {frac:.1%}"
            f"   median (max-min) spread: {spread:.0f} s")
    log("\nREADING: if the restricted identical-rates were near 100% the feed would be")
    log("broadcasting one number per trip and everything below would be circular.")
    return out


# ---------------------------------------------------------------------------
# S1  How much delay is carried between consecutive stops?
# ---------------------------------------------------------------------------
def section1(df: pd.DataFrame) -> pd.DataFrame:
    rule("S1  DELAY AT STOP k  ->  DELAY AT STOP k+1")
    d = df.sort_values(["trip_id", "planned_when"]).copy()
    d["k"] = d.groupby("trip_id").cumcount()
    d["n_in_trip"] = d.groupby("trip_id")["trip_id"].transform("size")
    m = d[d["n_in_trip"] > 1].copy()

    # Consecutive pairs within a trip, in timetable order. shift(-1) inside the
    # groupby keeps pairs from crossing trip boundaries.
    for col, new in [("final_delay_s", "next_delay"), ("planned_when", "next_planned"),
                     ("stop_id", "next_stop"), ("last_observed_at", "next_obs")]:
        m[new] = m.groupby("trip_id")[col].shift(-1)
    pairs = m[m["next_delay"].notna()].copy()
    pairs["gap_s"] = (pairs["next_planned"] - pairs["planned_when"]).dt.total_seconds()
    # A "pair" whose two stops are scheduled at the same minute is not a run between
    # stops; and >1 h apart is almost certainly a trip_id collision, not a vehicle.
    pairs = pairs[(pairs["gap_s"] > 0) & (pairs["gap_s"] <= 3600)]
    pairs["change"] = pairs["next_delay"] - pairs["final_delay_s"]
    log(f"consecutive stop-pairs usable: {len(pairs):,}")

    overall_r = pairs["final_delay_s"].corr(pairs["next_delay"])
    overall_rho = pairs["final_delay_s"].corr(pairs["next_delay"], method="spearman")
    log(f"overall Pearson r = {overall_r:.3f}   Spearman rho = {overall_rho:.3f}")

    rows = []
    for prod, sub in pairs.groupby("product"):
        if len(sub) < 200:
            continue
        # Conditional persistence: given the vehicle was already >=180 s late at stop k,
        # how often is it still >=180 s late at k+1? This is the number a reader
        # actually understands, more than a correlation coefficient.
        late_k = sub[sub["final_delay_s"] >= 180]
        ontime_k = sub[sub["final_delay_s"] < 180]
        rows.append({
            "product": prod,
            "pairs": len(sub),
            "r": sub["final_delay_s"].corr(sub["next_delay"]),
            "rho": sub["final_delay_s"].corr(sub["next_delay"], method="spearman"),
            "mean_change_s": sub["change"].mean(),
            "median_change_s": sub["change"].median(),
            # Raw drift is not comparable across products: the median scheduled gap
            # between two of our four stops is ~7 min for S-Bahn but ~35 min for tram,
            # so a tram has five times as long to accumulate delay. Normalising by
            # scheduled runtime gives the quantity a mechanism claim actually needs.
            "drift_s_per_min": (sub["change"] / (sub["gap_s"] / 60.0)).mean(),
            "p_late_next_given_late": (late_k["next_delay"] >= 180).mean() if len(late_k) else np.nan,
            "p_late_next_given_ontime": (ontime_k["next_delay"] >= 180).mean() if len(ontime_k) else np.nan,
            "n_late_k": len(late_k),
            "median_gap_s": sub["gap_s"].median(),
        })
    tab = pd.DataFrame(rows).sort_values("pairs", ascending=False)
    log("\nby product (change = delay at k+1 minus delay at k, seconds):")
    log(tab.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))

    # Growth vs decay conditional on being late: does a late vehicle recover?
    late = pairs[pairs["final_delay_s"] >= 180]
    if len(late):
        log(f"\nconditional on >=180 s late at stop k (n={len(late):,}):")
        log(f"   mean change to next stop : {late['change'].mean():+.1f} s")
        log(f"   median change            : {late['change'].median():+.1f} s")
        log(f"   recovered to <180 s      : {(late['next_delay'] < 180).mean():.1%}")
        log(f"   got worse                : {(late['change'] > 0).mean():.1%}")
    early = pairs[pairs["final_delay_s"] < 180]
    if len(early):
        log(f"conditional on <180 s at stop k (n={len(early):,}):")
        log(f"   became >=180 s late      : {(early['next_delay'] >= 180).mean():.1%}")
    return tab


# ---------------------------------------------------------------------------
# S2 helpers: build a leak-free upstream feature
# ---------------------------------------------------------------------------
def attach_upstream(df: pd.DataFrame) -> pd.DataFrame:
    """For every row, attach the same vehicle's most recent EARLIER stop observation.

    Legitimacy rules, all enforced on wall-clock time:
      * the upstream row belongs to the same trip_id and is earlier in the timetable;
      * the upstream number became visible (`last_observed_at`) STRICTLY BEFORE the
        target's scheduled departure -- so a passenger standing at stop B really could
        have known it;
      * the two stops are at most 1 h apart in the timetable (trip_id sanity).

    `upstream_lead_s` records how much genuine warning the feature gave, so the strict
    variant can demand >= STRICT_LEAD_S of it.
    """
    d = df.sort_values(["trip_id", "planned_when"]).copy()
    for col, new in [("final_delay_s", "up_delay"), ("planned_when", "up_planned"),
                     ("last_observed_at", "up_obs"), ("stop_id", "up_stop"),
                     ("first_delay_s", "up_first_delay")]:
        d[new] = d.groupby("trip_id")[col].shift(1)
    d["has_up"] = d["up_delay"].notna()
    d["up_gap_s"] = (d["planned_when"] - d["up_planned"]).dt.total_seconds()
    d["upstream_lead_s"] = (d["planned_when"] - d["up_obs"]).dt.total_seconds()
    valid = (
        d["has_up"]
        & (d["up_gap_s"] > 0)
        & (d["up_gap_s"] <= 3600)
        & (d["upstream_lead_s"] > 0)  # the hard anti-leak condition
    )
    d["up_valid"] = valid
    d.loc[~valid, ["up_delay", "up_gap_s", "upstream_lead_s", "up_first_delay"]] = np.nan

    # The departure's own first estimate, usable only where it predates the scheduled
    # departure (a handful of rows are first seen late, after their planned time).
    d["own_lead_s"] = (d["planned_when"] - d["first_observed_at"]).dt.total_seconds()
    d["own_first_delay"] = d["first_delay_s"].where(d["own_lead_s"] > 0)
    d.loc[d["own_lead_s"] <= 0, "own_lead_s"] = np.nan
    return d


def attach_contagion(df: pd.DataFrame) -> pd.DataFrame:
    """Recent delay on OTHER lines at the same stop, as of just before this departure.

    Built with cumulative sums over observations sorted by `last_observed_at` (the
    moment the number became visible), then sliced with searchsorted at the target's
    scheduled departure. Nothing at or after that instant can enter the aggregate, so
    the feature cannot leak. The same-line contribution is subtracted rather than
    filtered, which keeps the whole thing O(n log n) instead of O(n^2).
    """
    d = df.copy()
    for c in ["xline_mean_delay", "xline_frac_late", "xline_n",
              "sameline_mean_delay", "sameline_n"]:
        d[c] = np.nan

    for stop, idx in d.groupby("stop_id").groups.items():
        sub = d.loc[idx]
        reveal = sub["last_observed_at"].values.astype("datetime64[s]").astype(np.int64)
        order = np.argsort(reveal, kind="stable")
        r = reveal[order]
        val = sub["final_delay_s"].values[order]
        late = (val >= 180).astype(float)
        cs_v = np.concatenate([[0.0], np.cumsum(val)])
        cs_l = np.concatenate([[0.0], np.cumsum(late)])
        cs_n = np.arange(len(r) + 1, dtype=float)

        target = sub["planned_when"].values.astype("datetime64[s]").astype(np.int64)
        hi = np.searchsorted(r, target, side="left")
        lo = np.searchsorted(r, target - CONTAGION_WINDOW_S, side="left")
        tot_v = cs_v[hi] - cs_v[lo]
        tot_l = cs_l[hi] - cs_l[lo]
        tot_n = cs_n[hi] - cs_n[lo]

        # Same-line slice, computed per line inside this stop and scattered back.
        lines = sub["line_name"].values[order]
        same_v = np.zeros(len(sub))
        same_l = np.zeros(len(sub))
        same_n = np.zeros(len(sub))
        for line in pd.unique(lines):
            mask = lines == line
            rr = r[mask]
            vv = val[mask]
            ll = late[mask]
            c_v = np.concatenate([[0.0], np.cumsum(vv)])
            c_l = np.concatenate([[0.0], np.cumsum(ll)])
            c_n = np.arange(len(rr) + 1, dtype=float)
            # order[mask] maps this line's rows, still in reveal-time order, back to
            # their positions inside `sub` -- so the slice results scatter correctly.
            orig = order[mask]
            tg = target[orig]
            h = np.searchsorted(rr, tg, side="left")
            l_ = np.searchsorted(rr, tg - CONTAGION_WINDOW_S, side="left")
            same_v[orig] = c_v[h] - c_v[l_]
            same_l[orig] = c_l[h] - c_l[l_]
            same_n[orig] = c_n[h] - c_n[l_]

        # SELF-EXCLUSION. A row's own `last_observed_at` is usually earlier than its own
        # `planned_when` (the feed is polled ahead of departure), so the target row
        # falls inside its own look-back window and would contribute its own label to
        # the same-line aggregate -- a direct leak, and a large one when only one
        # same-line departure is in the window. It cancels in the other-lines figure
        # (present in both `tot` and `same`), but it must be removed explicitly here.
        self_rev = sub["last_observed_at"].values.astype("datetime64[s]").astype(np.int64)
        self_in = ((self_rev < target) & (self_rev >= target - CONTAGION_WINDOW_S)).astype(float)
        self_val = sub["final_delay_s"].values
        tot_v = tot_v - self_val * self_in
        tot_l = tot_l - (self_val >= 180).astype(float) * self_in
        tot_n = tot_n - self_in
        same_v = same_v - self_val * self_in
        same_l = same_l - (self_val >= 180).astype(float) * self_in
        same_n = same_n - self_in

        oth_n = tot_n - same_n
        with np.errstate(invalid="ignore", divide="ignore"):
            d.loc[idx, "xline_mean_delay"] = np.where(oth_n > 0, (tot_v - same_v) / oth_n, np.nan)
            d.loc[idx, "xline_frac_late"] = np.where(oth_n > 0, (tot_l - same_l) / oth_n, np.nan)
            d.loc[idx, "sameline_mean_delay"] = np.where(same_n > 0, same_v / same_n, np.nan)
        d.loc[idx, "xline_n"] = oth_n
        d.loc[idx, "sameline_n"] = same_n
    return d


# ---------------------------------------------------------------------------
# Modelling
# ---------------------------------------------------------------------------
BASE_FEATURES = [
    "hour", "minute", "weekday", "is_weekend", "day_of_year",
    "is_school_holiday", "is_public_holiday", "is_morning_peak", "is_evening_peak",
    "temp_c", "precip_mm", "wind_ms", "is_rain", "is_snow",
    "line_code", "product_code", "stop_code",
]
UP_FEATURES = ["up_delay", "up_gap_s", "upstream_lead_s", "up_first_delay"]
# THE HARD BASELINE. `first_delay_s` is the feed's FIRST estimate for this very
# departure, made at `first_observed_at`. Where that instant precedes the scheduled
# departure it is a perfectly legitimate, leak-free predictor -- and a far tougher one
# than a (line, hour) lookup. Any claim that "upstream propagation is informative" is
# only worth making if the upstream feature still adds something ON TOP of the
# departure's own earlier estimate. Without this column the pivot would look far
# stronger than it is.
OWN_FEATURES = ["own_first_delay", "own_lead_s"]
X_FEATURES = ["xline_mean_delay", "xline_frac_late", "xline_n",
              "sameline_mean_delay", "sameline_n"]
CATS = ["line_code", "product_code", "stop_code"]


def encode(df: pd.DataFrame) -> pd.DataFrame:
    # Plain ordinal codes for the tree model's native categorical support. This uses
    # no target information, so fitting the encoder on all rows is not leakage.
    d = df.copy()
    d["line_code"] = d["line_name"].astype("category").cat.codes
    d["product_code"] = d["product"].astype("category").cat.codes
    d["stop_code"] = d["stop_id"].astype("category").cat.codes
    return d


def lookup_baseline(train: pd.DataFrame, test: pd.DataFrame, y_col: str) -> np.ndarray:
    """The thing to beat: historical lateness rate of this (line, hour), nothing else.

    Smoothed towards the global rate so a line/hour cell seen twice in training does
    not get a confident 0 or 1. Fitted on train only.
    """
    prior = train[y_col].mean()
    grp = train.groupby(["line_name", "hour"])[y_col].agg(["sum", "count"])
    k = 20.0  # pseudo-count; with 41 days of data most cells are far larger than this
    grp["rate"] = (grp["sum"] + k * prior) / (grp["count"] + k)
    key = pd.MultiIndex.from_arrays([test["line_name"], test["hour"]])
    return grp["rate"].reindex(key).fillna(prior).to_numpy()


def fit_eval(train, test, feats, y_col, seed=0):
    cat_mask = [f in CATS for f in feats]
    clf = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.08, max_leaf_nodes=31,
        early_stopping=True, validation_fraction=0.12, n_iter_no_change=15,
        categorical_features=cat_mask, random_state=seed,
    )
    clf.fit(train[feats], train[y_col])
    p = clf.predict_proba(test[feats])[:, 1]
    return p, clf


def scores(y, p) -> tuple[float, float]:
    if y.nunique() < 2:
        return float("nan"), float("nan")
    return roc_auc_score(y, p), average_precision_score(y, p)


def time_split(d: pd.DataFrame, frac=0.7):
    """Time-ordered split: the first 70% of scheduled departures train, the last 30%
    test. Random splitting would let the model see the same rush hour on both sides."""
    d = d.sort_values("planned_when")
    cut = int(len(d) * frac)
    return d.iloc[:cut].copy(), d.iloc[cut:].copy()


def section2(df: pd.DataFrame) -> pd.DataFrame:
    rule("S2  IS PROPAGATION PREDICTIVE?  (evaluated only on rows with a legitimate upstream)")
    d = df[df["up_valid"]].copy()
    log(f"rows with a leak-free upstream observation: {len(d):,} "
        f"({len(d)/len(df):.1%} of labelled rows)")
    log(f"upstream lead (scheduled departure minus time the number was visible), seconds:")
    log("   " + d["upstream_lead_s"].describe(percentiles=[.1, .25, .5, .75, .9]).to_string().replace("\n", "\n   "))

    results = []
    for variant, sub in [("all-upstream", d),
                         (f"strict lead>={STRICT_LEAD_S}s", d[d["upstream_lead_s"] >= STRICT_LEAD_S])]:
        if len(sub) < 5000:
            log(f"\n[{variant}] only {len(sub):,} rows -- skipped")
            continue
        tr, te = time_split(sub)
        log(f"\n[{variant}] n={len(sub):,}  train={len(tr):,} test={len(te):,}  "
            f"test period {te.planned_when.min():%Y-%m-%d} .. {te.planned_when.max():%Y-%m-%d}")
        for thr in THRESHOLDS:
            y = f"y{thr}"
            tr[y] = (tr["final_delay_s"] >= thr).astype(int)
            te[y] = (te["final_delay_s"] >= thr).astype(int)
            base_rate = te[y].mean()
            if te[y].nunique() < 2 or tr[y].sum() < 50:
                log(f"   >={thr}s: too few positives, skipped")
                continue

            p_lookup = lookup_baseline(tr, te, y)
            r_lookup = scores(te[y], p_lookup)
            # Two zero-learning sanity scores: the raw upstream number, and the
            # departure's own first estimate. If a gradient-boosting machine cannot
            # beat these it is adding nothing.
            r_raw = scores(te[y], te["up_delay"].fillna(0).to_numpy())
            r_own_raw = scores(te[y], te["own_first_delay"].fillna(0).to_numpy())
            fits = [
                ("GBM no-upstream", BASE_FEATURES),
                ("GBM +upstream", BASE_FEATURES + UP_FEATURES),
                ("GBM +own-estimate", BASE_FEATURES + OWN_FEATURES),
                ("GBM +own +upstream", BASE_FEATURES + OWN_FEATURES + UP_FEATURES),
                ("GBM +own +upstream +crossline", BASE_FEATURES + OWN_FEATURES + UP_FEATURES + X_FEATURES),
            ]
            got = {}
            for name, feats in fits:
                p, _ = fit_eval(tr, te, feats, y)
                got[name] = scores(te[y], p)

            log(f"   >={thr}s  positives in test: {base_rate:.2%}")
            table = [("(line,hour) lookup", r_lookup), ("raw upstream delay", r_raw),
                     ("raw own first estimate", r_own_raw)] + list(got.items())
            for name, (roc, pr) in table:
                results.append({"variant": variant, "thr_s": thr, "model": name,
                                "roc_auc": roc, "pr_auc": pr, "pos_rate": base_rate,
                                "n_test": len(te)})
                log(f"      {name:<32} ROC {roc:.3f}  PR {pr:.3f}")
            # The two deltas that decide the pivot: upstream over the weak lookup
            # baseline, and upstream over the hard own-estimate baseline.
            d_lookup = got["GBM +own +upstream"][0] - r_lookup[0]
            d_own = got["GBM +own +upstream"][0] - got["GBM +own-estimate"][0]
            d_noup = got["GBM +upstream"][0] - got["GBM no-upstream"][0]
            log(f"      -> dROC upstream vs (line,hour) lookup : {d_noup:+.3f}")
            log(f"      -> dROC upstream ON TOP OF own estimate: {d_own:+.3f}  "
                f"(full model vs lookup {d_lookup:+.3f})")
    return pd.DataFrame(results)


def section3(df: pd.DataFrame) -> pd.DataFrame:
    rule("S3  CROSS-LINE CONTAGION  (all labelled rows, upstream feature deliberately absent)")
    d = df.copy()
    log(f"rows with >=1 other-line observation in the preceding {CONTAGION_WINDOW_S//60} min: "
        f"{(d['xline_n'] > 0).sum():,} / {len(d):,}")

    # Marginal association first, before any model: does a busy-and-late neighbourhood
    # coincide with this departure being late?
    sub = d[d["xline_n"] >= 3]
    for thr in THRESHOLDS:
        y = (sub["final_delay_s"] >= thr).astype(int)
        r = np.corrcoef(sub["xline_frac_late"].fillna(0), y)[0, 1]
        q = pd.qcut(sub["xline_frac_late"], 5, duplicates="drop")
        rate = y.groupby(q, observed=True).mean()
        log(f"\n>={thr}s  point-biserial r with other-lines late-fraction: {r:+.3f}")
        log("   lateness rate by quintile of other-lines late-fraction:")
        for lab, v in rate.items():
            log(f"      {str(lab):<22} {v:.2%}")

    results = []
    tr, te = time_split(d)
    log(f"\ntime-split n={len(d):,} train={len(tr):,} test={len(te):,}")
    for thr in THRESHOLDS:
        y = f"y{thr}"
        tr[y] = (tr["final_delay_s"] >= thr).astype(int)
        te[y] = (te["final_delay_s"] >= thr).astype(int)
        if te[y].nunique() < 2:
            continue
        r_lookup = scores(te[y], lookup_baseline(tr, te, y))
        got = {}
        # Cross-line contagion is tested twice: on its own against a calendar/weather
        # model, and on top of the departure's own earlier estimate. Only the second
        # answers "does the neighbourhood tell us anything the feed had not already
        # said about this very departure?"
        for name, feats in [("GBM base", BASE_FEATURES),
                            ("GBM +crossline", BASE_FEATURES + X_FEATURES),
                            ("GBM +own-estimate", BASE_FEATURES + OWN_FEATURES),
                            ("GBM +own +crossline", BASE_FEATURES + OWN_FEATURES + X_FEATURES)]:
            p, _ = fit_eval(tr, te, feats, y)
            got[name] = scores(te[y], p)
        log(f"\n>={thr}s  positives {te[y].mean():.2%}")
        for name, (roc, pr) in [("(line,hour) lookup", r_lookup)] + list(got.items()):
            results.append({"thr_s": thr, "model": name, "roc_auc": roc, "pr_auc": pr,
                            "pos_rate": te[y].mean(), "n_test": len(te)})
            log(f"   {name:<22} ROC {roc:.3f} PR {pr:.3f}")
        log(f"   -> dROC crossline vs base      : {got['GBM +crossline'][0]-got['GBM base'][0]:+.3f}")
        log(f"   -> dROC crossline on top of own: {got['GBM +own +crossline'][0]-got['GBM +own-estimate'][0]:+.3f}")
    return pd.DataFrame(results)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None,
                    help="subsample this many TRIPS (not rows) for a fast re-run")
    ap.add_argument("--outdir", type=str, default=str(Path(__file__).resolve().parent))
    args = ap.parse_args()

    df = load(args.sample)
    log(f"labelled rows: {len(df):,}   trips: {df.trip_id.nunique():,}   "
        f"{df.planned_when.min():%Y-%m-%d} .. {df.planned_when.max():%Y-%m-%d}")

    s0 = section0(df)
    t1 = section1(df)

    df = attach_upstream(df)
    df = attach_contagion(df)
    df = encode(df)

    t2 = section2(df)
    t3 = section3(df)

    out = Path(args.outdir)
    t1.to_csv(out / "prop_s1_by_product.csv", index=False)
    t2.to_csv(out / "prop_s2_models.csv", index=False)
    t3.to_csv(out / "prop_s3_crossline.csv", index=False)
    (out / "prop_s0_structure.json").write_text(json.dumps(s0, indent=2))
    log(f"\nwrote prop_s0_structure.json, prop_s1_by_product.csv, "
        f"prop_s2_models.csv, prop_s3_crossline.csv to {out}")


if __name__ == "__main__":
    main()
