"""How much is live information worth, as a function of how far ahead you predict?

    python analysis/experiments/horizon_sweep.py
    python analysis/experiments/horizon_sweep.py --thresholds 180 --horizons 10 30 60

WHY THIS EXISTS
Two results from 2026-09-21 looked like the project's headline, but both came from
a single time split produced by agents that stalled:
  * the nowcast -- knowing a trip's own earlier realtime estimate -- doubled PR-AUC;
  * "cross-line" network state -- how late OTHER lines at the same stop are right
    now -- beat the (line, hour) lookup table that nothing else could beat.
The second was also mislabelled as schedule-time: it used network state from just
before departure, which only exists at a short horizon. My own 30-minute-lag
version on 2026-08-17 had found almost nothing (+0.015 ROC).

Both are really the same question -- what is live information worth, and how fast
does that value decay with forecast horizon -- so this script answers it once,
properly: every model is re-fitted at every horizon, on rolling-origin folds, with
day-clustered bootstrap confidence intervals on the gains.

THE DESIGN DECISION THAT MATTERS MOST
At horizon h the prediction is made at T = planned_when - h, and only information
observed at or before T may be used. But the LABEL (final_delay_s) is itself only
the last estimate we saw, and it can be many minutes old. If the label was observed
before T, the "latest estimate at T" simply IS the label and short horizons look
spectacular for no reason. So the evaluation population is restricted to
departures whose label was observed within MAX_LEAD_S of departure (or later), and
horizons start at 10 min: every prediction point is then strictly before the label
it is scored against, and the population is the SAME at every horizon, so the
curve is comparable across h.

MODELS (all HistGradientBoosting, class-balanced, identical settings)
  lookup      historical late rate per (line, hour) on the training days
  A           schedule-time: line / stop / product, time of day, calendar, weather
  own-only    ONLY the trip's latest realtime estimate at T and its age
  A+own       A plus the trip's own latest estimate            (the nowcast)
  A+net       A plus network state at T: late share and mean delay among OTHER
              lines at the same stop, from the most recent board snapshot <= T
  A+own+net   both
day_of_year and month are deliberately excluded: over seven weeks they act as a
drift proxy, not a cause, and removing day_of_year did not hurt (2026-09-20).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))
import build_dataset as bd  # noqa: E402  (reuse the exact loaders the dataset uses)

OUT_DIR = Path(__file__).resolve().parent
MAX_LEAD_S = 600            # label observed <= 10 min before departure (or later)
NET_MAX_AGE_MIN = 30        # a board snapshot older than this counts as "unknown"
MAX_CATS = 250              # HistGradientBoosting accepts at most 255 categories

CAT = ["product", "line_name", "stop_id"]
BASE_NUM = ["hour", "weekday", "is_weekend", "is_school_holiday", "is_public_holiday",
            "is_morning_peak", "is_evening_peak", "is_full_traffic_day",
            "temp_c", "humidity_pct", "precip_mm", "wind_ms", "is_rain", "is_wet"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def naive_utc(s: pd.Series) -> pd.Series:
    """tz-aware -> naive UTC at microsecond resolution, so merge_asof keys match."""
    s = pd.to_datetime(s, utc=True, errors="coerce")
    return s.dt.tz_localize(None).dt.as_unit("us")


def key(trip, stop, planned) -> pd.Series:
    return (trip.astype(str) + "|" + stop.astype(str) + "|"
            + planned.dt.as_unit("s").astype("int64").astype(str))


# --------------------------------------------------------------------------- data

def load(max_lead_s: int):
    dep = pd.read_parquet(ROOT / "data" / "dataset.parquet")
    dep = dep[(dep.cancelled == 0) & dep.final_delay_s.notna()].copy()
    n_all = len(dep)
    dep = dep[dep.lead_time_s < max_lead_s].copy()
    log(f"departures: {n_all:,} usable -> {len(dep):,} with a label observed "
        f"<= {max_lead_s // 60} min before departure ({len(dep) / n_all:.1%})")
    dep["planned"] = naive_utc(dep.planned_when)
    dep["k"] = key(dep.trip_id, dep.stop_id, dep.planned)
    dep["day"] = dep.planned.dt.floor("D")

    obs = bd.load_observations()
    obs["source"] = "transport.rest"
    rt = bd.load_gtfsrt()
    raw = pd.concat([obs, rt], ignore_index=True)[
        ["observed_at", "stop_id", "trip_id", "line_name", "planned_when",
         "delay_s", "source"]]
    raw["observed_at"] = naive_utc(raw.observed_at)
    raw["planned"] = naive_utc(raw.planned_when)
    raw["delay_s"] = pd.to_numeric(raw.delay_s, errors="coerce")
    raw = raw.dropna(subset=["observed_at", "planned", "trip_id"])
    raw["stop_id"] = raw.stop_id.astype(str)
    raw["k"] = key(raw.trip_id, raw.stop_id, raw.planned)
    log(f"raw observations: {len(raw):,}")
    return dep, raw


def build_snapshots(raw: pd.DataFrame):
    """Board state per (source, stop, poll): how late is everything on the board?"""
    d = raw[raw.delay_s.notna()].copy()
    d["l180"] = (d.delay_s >= 180).astype(np.int32)
    d["l600"] = (d.delay_s >= 600).astype(np.int32)
    d["line_name"] = d.line_name.astype(str)
    g = ["source", "stop_id", "observed_at"]
    snap = d.groupby(g).agg(n=("delay_s", "size"), s180=("l180", "sum"),
                            s600=("l600", "sum"), sd=("delay_s", "sum")).reset_index()
    snapl = d.groupby(g + ["line_name"]).agg(
        nl=("delay_s", "size"), s180l=("l180", "sum"),
        s600l=("l600", "sum"), sdl=("delay_s", "sum")).reset_index()
    log(f"board snapshots: {len(snap):,} (per-line rows {len(snapl):,})")
    return snap, snapl


def features_at(dep, own_src, snap, snapl, h: int) -> pd.DataFrame:
    """All live features available at T = planned - h minutes. Nothing after T."""
    probe = dep.planned - pd.Timedelta(minutes=h)
    left = pd.DataFrame({"k": dep.k.values, "stop_id": dep.stop_id.astype(str).values,
                         "line_name": dep.line_name.astype(str).values,
                         "probe": probe.values, "i": np.arange(len(dep))})
    left["probe"] = left.probe.astype("datetime64[us]")
    left = left.sort_values("probe")

    # own trip: latest AVAILABLE realtime estimate at or before T
    own = pd.merge_asof(left, own_src, left_on="probe", right_on="observed_at",
                        by="k", direction="backward")
    own_est = own.set_index("i")["delay_s"].reindex(np.arange(len(dep))).values
    own_age = ((own.probe - own.observed_at).dt.total_seconds() / 60)
    own_age = own_age.set_axis(own.i).reindex(np.arange(len(dep))).values

    # network: most recent board snapshot at this stop at or before T, other lines only
    net = pd.merge_asof(left, snap.sort_values("observed_at"), left_on="probe",
                        right_on="observed_at", by="stop_id", direction="backward")
    net = net.merge(snapl, on=["source", "stop_id", "observed_at", "line_name"],
                    how="left")
    for c in ("nl", "s180l", "s600l", "sdl"):
        net[c] = net[c].fillna(0)
    n_other = net.n - net.nl
    age = (net.probe - net.observed_at).dt.total_seconds() / 60
    ok = (n_other > 0) & (age <= NET_MAX_AGE_MIN)
    net["net_late180"] = np.where(ok, (net.s180 - net.s180l) / n_other.where(ok, 1), np.nan)
    net["net_late600"] = np.where(ok, (net.s600 - net.s600l) / n_other.where(ok, 1), np.nan)
    net["net_mean_delay"] = np.where(ok, (net.sd - net.sdl) / n_other.where(ok, 1), np.nan)
    net["net_age"] = np.where(ok, age, np.nan)
    net = net.set_index("i").reindex(np.arange(len(dep)))

    return pd.DataFrame({
        "own_est": own_est, "own_age": own_age,
        "net_late180": net.net_late180.values, "net_late600": net.net_late600.values,
        "net_mean_delay": net.net_mean_delay.values, "net_age": net.net_age.values,
    }, index=dep.index)


# ------------------------------------------------------------------------- models

def encode(tr: pd.DataFrame, te: pd.DataFrame, cols: list[str]):
    Xtr, Xte = tr[cols].copy(), te[cols].copy()
    cat_idx = []
    for j, c in enumerate(cols):
        if c in CAT:
            top = Xtr[c].astype(str).value_counts().index[:MAX_CATS]
            cats = pd.Index(list(top) + ["__other__"])
            conv = lambda s: pd.Categorical(
                s.astype(str).where(s.astype(str).isin(top), "__other__"), categories=cats)
            Xtr[c], Xte[c] = conv(Xtr[c]), conv(Xte[c])
            cat_idx.append(j)
        else:
            Xtr[c] = pd.to_numeric(Xtr[c], errors="coerce").astype(float)
            Xte[c] = pd.to_numeric(Xte[c], errors="coerce").astype(float)
    return Xtr, Xte, cat_idx


def fit_predict(tr, te, cols, y_tr) -> np.ndarray:
    Xtr, Xte, cat_idx = encode(tr, te, cols)
    m = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=100,
        class_weight="balanced", random_state=0,
        categorical_features=cat_idx or None)
    m.fit(Xtr, y_tr)
    return m.predict_proba(Xte)[:, 1]


def lookup_pred(tr, te, y_tr) -> np.ndarray:
    t = tr.assign(y=y_tr)
    rate = t.groupby(["line_name", "hour"], observed=True).y.mean()
    idx = pd.MultiIndex.from_arrays([te.line_name, te.hour])
    return pd.Series(idx.map(rate).astype(float), index=te.index).fillna(t.y.mean()).values


# ---------------------------------------------------------------------- main loop

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--thresholds", type=int, nargs="+", default=[180, 600])
    ap.add_argument("--horizons", type=int, nargs="+", default=[10, 15, 20, 30, 45, 60])
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--fold-days", type=int, default=4)
    ap.add_argument("--boot", type=int, default=400)
    args = ap.parse_args()

    dep, raw = load(MAX_LEAD_S)
    own_src = (raw[raw.delay_s.notna()][["k", "observed_at", "delay_s"]]
               .sort_values("observed_at"))
    snap, snapl = build_snapshots(raw)
    del raw

    live = {}
    for h in args.horizons:
        live[h] = features_at(dep, own_src, snap, snapl, h)
        f = live[h]
        log(f"h={h:>2} min: own estimate available {f.own_est.notna().mean():.1%}, "
            f"network state available {f.net_late180.notna().mean():.1%}")

    days = np.sort(dep.day.unique())
    blocks = [days[len(days) - (args.folds - i) * args.fold_days:
                   len(days) - (args.folds - i - 1) * args.fold_days]
              for i in range(args.folds)]
    base = CAT + BASE_NUM
    fold_rows, oof = [], []

    for th in args.thresholds:
        y_all = (dep.final_delay_s >= th).astype(int)
        for fi, block in enumerate(blocks):
            tr_m = dep.day < block[0]
            te_m = dep.day.isin(block)
            tr, te = dep[tr_m], dep[te_m]
            ytr, yte = y_all[tr_m].values, y_all[te_m].values
            if yte.sum() < 20:
                log(f"th={th} fold {fi}: only {yte.sum()} positives, skipped")
                continue
            preds = {"lookup": lookup_pred(tr, te, ytr),
                     "A": fit_predict(tr, te, base, ytr)}
            for h in args.horizons:
                L = live[h]
                trh, teh = tr.join(L), te.join(L)
                preds[f"own-only@{h}"] = fit_predict(trh, teh, ["own_est", "own_age"], ytr)
                preds[f"A+own@{h}"] = fit_predict(trh, teh, base + ["own_est", "own_age"], ytr)
                netc = ["net_late180", "net_late600", "net_mean_delay", "net_age"]
                preds[f"A+net@{h}"] = fit_predict(trh, teh, base + netc, ytr)
                preds[f"A+own+net@{h}"] = fit_predict(
                    trh, teh, base + ["own_est", "own_age"] + netc, ytr)
            for name, p in preds.items():
                fold_rows.append({"threshold": th, "fold": fi,
                                  "test_start": str(pd.Timestamp(block[0]).date()),
                                  "test_end": str(pd.Timestamp(block[-1]).date()),
                                  "model": name, "n_test": len(yte), "pos": int(yte.sum()),
                                  "roc": roc_auc_score(yte, p),
                                  "pr": average_precision_score(yte, p)})
            o = pd.DataFrame(preds, index=te.index)
            o["y"], o["day"], o["threshold"] = yte, te.day.values, th
            oof.append(o)
            log(f"th={th} fold {fi} ({pd.Timestamp(block[0]).date()}.."
                f"{pd.Timestamp(block[-1]).date()}): {len(yte):,} test rows, "
                f"{yte.sum():,} positives, {len(preds)} models")

    folds = pd.DataFrame(fold_rows)
    folds.to_csv(OUT_DIR / "horizon_sweep_folds.csv", index=False)

    # pooled out-of-fold metrics + day-clustered bootstrap CIs on the gains
    rng = np.random.default_rng(0)
    pooled = []
    for th in args.thresholds:
        O = pd.concat([o for o in oof if o.threshold.iloc[0] == th])
        y = O.y.values
        day_idx = {d: np.flatnonzero(O.day.values == d) for d in np.unique(O.day.values)}
        dkeys = list(day_idx)
        samples = [np.concatenate([day_idx[d] for d in rng.choice(dkeys, len(dkeys))])
                   for _ in range(args.boot)]
        models = [c for c in O.columns if c not in ("y", "day", "threshold")]
        for mname in models:
            pooled.append({"threshold": th, "model": mname, "n": len(y), "pos": int(y.sum()),
                           "roc": roc_auc_score(y, O[mname]),
                           "pr": average_precision_score(y, O[mname])})

        def gain(a, b):
            d_pr = [average_precision_score(y[s], O[a].values[s]) -
                    average_precision_score(y[s], O[b].values[s]) for s in samples]
            d_roc = [roc_auc_score(y[s], O[a].values[s]) -
                     roc_auc_score(y[s], O[b].values[s]) for s in samples]
            return (np.percentile(d_pr, [2.5, 50, 97.5]),
                    np.percentile(d_roc, [2.5, 50, 97.5]))

        for h in args.horizons:
            for a, b, label in ((f"A+own@{h}", "A", "own vs A"),
                                (f"A+net@{h}", "A", "net vs A"),
                                (f"A+own+net@{h}", f"A+own@{h}", "net on top of own")):
                pr_ci, roc_ci = gain(a, b)
                pooled.append({"threshold": th, "model": f"GAIN {label} @{h}",
                               "n": len(y), "pos": int(y.sum()),
                               "pr": pr_ci[1], "pr_lo": pr_ci[0], "pr_hi": pr_ci[2],
                               "roc": roc_ci[1], "roc_lo": roc_ci[0], "roc_hi": roc_ci[2]})
            log(f"th={th} h={h}: bootstrap CIs done")

    pd.DataFrame(pooled).to_csv(OUT_DIR / "horizon_sweep_pooled.csv", index=False)
    meta = {"max_lead_s": MAX_LEAD_S, "net_max_age_min": NET_MAX_AGE_MIN,
            "thresholds": args.thresholds, "horizons": args.horizons,
            "folds": args.folds, "fold_days": args.fold_days, "boot": args.boot,
            "n_population": len(dep),
            "availability": {str(h): {"own": float(live[h].own_est.notna().mean()),
                                      "net": float(live[h].net_late180.notna().mean())}
                             for h in args.horizons}}
    (OUT_DIR / "horizon_sweep_meta.json").write_text(json.dumps(meta, indent=2))
    log("done -> horizon_sweep_folds.csv, horizon_sweep_pooled.csv, horizon_sweep_meta.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
