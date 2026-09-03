"""The two questions the ablation run never answered.

Q1  Does the model do anything beyond memorising which lines are chronically late?
Q2  Is the apparent weather effect real, or is temperature standing in for season?

Both are run at the >=180s threshold, which the validation work identified as the
defensible target (7/7 folds beat the baseline on both metrics; fold sd 0.024
versus 0.089 at >=600s).

Deliberately cheap: the earlier agent runs stalled on long searches, and these two
questions do not need a hyperparameter sweep to be answered.
"""
from __future__ import annotations
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

THRESH = 180
IDENT = ["line_name", "stop_id", "product", "direction"]
TIME  = ["hour", "minute", "weekday", "month", "is_weekend",
         "is_morning_peak", "is_evening_peak"]
SEASON = ["day_of_year"]
CAL   = ["is_school_holiday", "is_public_holiday", "is_full_traffic_day"]
WX    = ["temp_c", "humidity_pct", "precip_mm", "precip_form", "wind_ms",
         "wind_dir_deg", "is_rain", "is_wet", "is_snow", "is_freezing", "is_hot"]

df = pd.read_parquet("data/dataset.parquet")
df = df[(df.cancelled == 0) & df.final_delay_s.notna()].sort_values("planned_when")
df["y"] = (df.final_delay_s >= THRESH).astype(int)
cut = int(len(df) * 0.7)
print(f"rows {len(df):,} | positives {df.y.mean():.2%} | train {cut:,} test {len(df)-cut:,}")

def fit_eval(cols, tr=None, te=None, label=""):
    tr = df.iloc[:cut] if tr is None else tr
    te = df.iloc[cut:] if te is None else te
    cats = [c for c in cols if c in IDENT]
    X = pd.concat([tr[cols], te[cols]])
    for c in cats:
        X[c] = X[c].astype("category")
    for c in [c for c in cols if c not in cats]:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
    Xtr, Xte = X.iloc[:len(tr)], X.iloc[len(tr):]
    m = HistGradientBoostingClassifier(
        max_iter=150, learning_rate=0.08, random_state=0,
        categorical_features=[X.columns.get_loc(c) for c in cats] or None,
        class_weight="balanced").fit(Xtr, tr.y.values)
    p = m.predict_proba(Xte)[:, 1]
    roc, pr = roc_auc_score(te.y, p), average_precision_score(te.y, p)
    if label:
        print(f"  {label:<44}{roc:>8.3f}{pr:>9.3f}")
    return roc, pr

print(f"\nQ1  BLOCK ABLATION (target: delay >= {THRESH//60} min)")
print(f"  {'feature set':<44}{'ROC-AUC':>8}{'PR-AUC':>9}")
full = fit_eval(IDENT+TIME+SEASON+CAL+WX, label="everything")
fit_eval(TIME+SEASON+CAL+WX,               label="WITHOUT identity (no line/stop/product)")
fit_eval(IDENT,                            label="identity ONLY")
fit_eval(IDENT+TIME,                       label="identity + time")
fit_eval(TIME+SEASON+CAL,                  label="no identity, no weather")

# Held-out lines: can it score a line it has never seen?
print("\nQ1b HELD-OUT LINES (generalisation to unseen lines)")
lines = df.line_name.dropna().unique()
rng = np.random.default_rng(0); rng.shuffle(lines)
held = set(lines[:max(1, len(lines)//4)])
tr = df.iloc[:cut]; tr = tr[~tr.line_name.isin(held)]
te = df.iloc[cut:]; te_h = te[te.line_name.isin(held)]
print(f"  {len(held)} of {len(lines)} lines held out | test rows on unseen lines: {len(te_h):,}"
      f" | positives {te_h.y.mean():.2%}")
if te_h.y.nunique() > 1:
    print(f"  {'feature set':<44}{'ROC-AUC':>8}{'PR-AUC':>9}")
    fit_eval(IDENT+TIME+SEASON+CAL+WX, tr, te_h, "all features, UNSEEN lines")
    fit_eval(TIME+SEASON+CAL+WX,       tr, te_h, "no identity, UNSEEN lines")

print("\nQ2  WEATHER vs SEASON")
print(f"  {'feature set':<44}{'ROC-AUC':>8}{'PR-AUC':>9}")
a = fit_eval(IDENT+TIME+CAL,          label="baseline (no weather, no day_of_year)")
b = fit_eval(IDENT+TIME+CAL+WX,       label="+ weather")
c = fit_eval(IDENT+TIME+CAL+SEASON,   label="+ day_of_year (season proxy)")
d = fit_eval(IDENT+TIME+CAL+SEASON+WX,label="+ day_of_year + weather")
print(f"\n  weather alone adds        : {b[0]-a[0]:+.4f} ROC   {b[1]-a[1]:+.4f} PR")
print(f"  season alone adds         : {c[0]-a[0]:+.4f} ROC   {c[1]-a[1]:+.4f} PR")
print(f"  weather ON TOP of season  : {d[0]-c[0]:+.4f} ROC   {d[1]-c[1]:+.4f} PR")
print("\n  If weather's gain collapses once day_of_year is present, the 'weather effect'")
print("  is seasonal confounding, not weather -- the data spans only late summer.")
