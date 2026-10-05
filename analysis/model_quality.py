"""How good are the prediction models? Rolling-origin evaluation on the current data.

    python analysis/model_quality.py
    python analysis/model_quality.py --folds 4 --fold-days 7

Answers four questions with the same folds throughout:
  1. Schedule-time prediction: does gradient boosting beat the (line, hour) lookup
     table, at >=3 min and >=10 min? (ROC-AUC, PR-AUC, fold wins)
  2. Does the label definition change the verdict? Evaluated on ALL labelled
     departures and on FRESH labels only (last seen <= 10 min before departure, or
     after it). The audit of 2026-10-05 showed stale labels understate delay
     sharply (>=3 min late: 25.2% if seen at/after departure, 1.9% if last seen
     >30 min before), so a model scored on stale labels is partly scored on noise.
  3. Calibration: when the model says "20% chance of >=3 min delay", is that true?
     Brier score and a binned reliability table on pooled out-of-fold predictions.
  4. Cancellations: is the model's advantage over the lookup robust across folds?

Excludes departures planned after the newest observation in the data: their label
is still a forecast, not an outcome.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis" / "experiments"
FRESH_S = 600
MAX_CATS = 250
CAT = ["product", "line_name", "stop_id"]
NUM = ["hour", "weekday", "is_weekend", "is_school_holiday", "is_public_holiday",
       "is_morning_peak", "is_evening_peak", "is_full_traffic_day",
       "temp_c", "humidity_pct", "precip_mm", "wind_ms", "is_rain", "is_wet"]


def encode(tr, te, cols):
    Xtr, Xte, cat_idx = tr[cols].copy(), te[cols].copy(), []
    for j, c in enumerate(cols):
        if c in CAT:
            top = Xtr[c].astype(str).value_counts().index[:MAX_CATS]
            cats = pd.Index(list(top) + ["__other__"])
            f = lambda s: pd.Categorical(s.astype(str).where(s.astype(str).isin(top), "__other__"),
                                         categories=cats)
            Xtr[c], Xte[c] = f(Xtr[c]), f(Xte[c])
            cat_idx.append(j)
        else:
            Xtr[c] = pd.to_numeric(Xtr[c], errors="coerce").astype(float)
            Xte[c] = pd.to_numeric(Xte[c], errors="coerce").astype(float)
    return Xtr, Xte, cat_idx


def gbm(tr, te, y, balanced=True):
    Xtr, Xte, ci = encode(tr, te, CAT + NUM)
    m = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, max_leaf_nodes=31,
                                       min_samples_leaf=100, random_state=0,
                                       class_weight="balanced" if balanced else None,
                                       categorical_features=ci)
    return m.fit(Xtr, y).predict_proba(Xte)[:, 1]


def lookup(tr, te, y, keys=("line_name", "hour")):
    t = tr.assign(y=y)
    r = t.groupby(list(keys), observed=True).y.mean()
    idx = pd.MultiIndex.from_arrays([te[k] for k in keys])
    return pd.Series(idx.map(r).astype(float), index=te.index).fillna(t.y.mean()).values


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--fold-days", type=int, default=7)
    args = ap.parse_args()

    df = pd.read_parquet(ROOT / "data" / "dataset.parquet")
    df = df[df.planned_when <= df.last_observed_at.max()].copy()
    df["day"] = df.planned_when.dt.tz_convert("Europe/Berlin").dt.floor("D")
    days = np.sort(df.day.unique())
    blocks = [days[len(days) - (args.folds - i) * args.fold_days:
                   len(days) - (args.folds - i - 1) * args.fold_days] for i in range(args.folds)]
    print(f"departures {len(df):,}; folds: " +
          ", ".join(f"{pd.Timestamp(b[0]):%m-%d}..{pd.Timestamp(b[-1]):%m-%d}" for b in blocks))

    lab = df[(df.cancelled == 0) & df.final_delay_s.notna()]
    pops = {"all labels": lab, "fresh labels": lab[lab.lead_time_s < FRESH_S]}
    rows, oof = [], {}

    for pname, pop in pops.items():
        for th in (180, 600):
            y_all = (pop.final_delay_s >= th).astype(int)
            preds = []
            for fi, b in enumerate(blocks):
                trm, tem = pop.day < b[0], pop.day.isin(b)
                tr, te = pop[trm], pop[tem]
                ytr, yte = y_all[trm].values, y_all[tem].values
                pl, pg = lookup(tr, te, ytr), gbm(tr, te, ytr)
                pu = gbm(tr, te, ytr, balanced=False)          # for calibration
                for name, p in (("lookup", pl), ("GBM", pg)):
                    rows.append({"population": pname, "target": f">={th//60} min", "fold": fi,
                                 "model": name, "n": len(yte), "pos_rate": yte.mean(),
                                 "roc": roc_auc_score(yte, p), "pr": average_precision_score(yte, p)})
                preds.append(pd.DataFrame({"y": yte, "lookup": pl, "GBM": pg, "GBM_unw": pu}))
            oof[(pname, th)] = pd.concat(preds)
            print(f"  done: {pname}, >= {th//60} min")

    # cancellations (all departures, not just labelled ones)
    yc_all = df.cancelled.astype(int)
    cpreds = []
    for fi, b in enumerate(blocks):
        trm, tem = df.day < b[0], df.day.isin(b)
        tr, te = df[trm], df[tem]
        ytr, yte = yc_all[trm].values, yc_all[tem].values
        pl = lookup(tr, te, ytr, ("line_name", "stop_id"))
        pg = gbm(tr, te, ytr)
        for name, p in (("lookup", pl), ("GBM", pg)):
            rows.append({"population": "all departures", "target": "cancelled", "fold": fi,
                         "model": name, "n": len(yte), "pos_rate": yte.mean(),
                         "roc": roc_auc_score(yte, p), "pr": average_precision_score(yte, p)})
    print("  done: cancellations")

    R = pd.DataFrame(rows)
    R.to_csv(OUT / "model_quality_folds.csv", index=False)

    print("\n" + "=" * 78)
    print("ROLLING-ORIGIN RESULTS (mean over folds; wins = folds where GBM beats lookup)")
    print("=" * 78)
    for (pop, tgt), g in R.groupby(["population", "target"], sort=False):
        L, G = g[g.model == "lookup"].set_index("fold"), g[g.model == "GBM"].set_index("fold")
        print(f"\n{pop} | {tgt} | base rate {L.pos_rate.mean():.2%} | n/fold ~{int(L.n.mean()):,}")
        print(f"   lookup  ROC {L.roc.mean():.3f} ± {L.roc.std():.3f}   PR {L.pr.mean():.3f} ± {L.pr.std():.3f}")
        print(f"   GBM     ROC {G.roc.mean():.3f} ± {G.roc.std():.3f}   PR {G.pr.mean():.3f} ± {G.pr.std():.3f}")
        print(f"   GBM wins: ROC {(G.roc > L.roc).sum()}/{len(G)}, PR {(G.pr > L.pr).sum()}/{len(G)}"
              f"   | mean gain ROC {(G.roc - L.roc).mean():+.3f}, PR {(G.pr - L.pr).mean():+.3f}")

    print("\n" + "=" * 78)
    print("CALIBRATION (pooled out-of-fold, fresh labels)")
    print("=" * 78)
    for th in (180, 600):
        O = oof[("fresh labels", th)]
        base = O.y.mean()
        print(f"\n>= {th//60} min  (base rate {base:.2%})")
        for m in ("lookup", "GBM_unw", "GBM"):
            print(f"   Brier {m:<8} {brier_score_loss(O.y, O[m]):.4f}"
                  f"   (constant base rate: {brier_score_loss(O.y, np.full(len(O), base)):.4f})")
        bins = pd.qcut(O.GBM_unw, 10, duplicates="drop")
        t = O.groupby(bins, observed=True).agg(pred=("GBM_unw", "mean"), obs=("y", "mean"), n=("y", "size"))
        print("   reliability, unweighted GBM (decile: predicted -> observed):")
        for _, r in t.iterrows():
            print(f"     predicted {r.pred:6.1%}  observed {r.obs:6.1%}  n={int(r.n):,}")
        top = O.nlargest(int(0.05 * len(O)), "GBM_unw")
        print(f"   top 5% riskiest departures: {top.y.mean():.1%} actually late "
              f"({top.y.mean()/base:.1f}x base rate), capturing {top.y.sum()/O.y.sum():.1%} of all late ones")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
