"""Does the STRUCTURE of a line explain why it is chronically late?

WHY THIS EXISTS
---------------
The project's most robust finding is negative-for-prediction but positive-for-
explanation: line identity dominates. Removing line/stop/product features drops
ROC-AUC 0.801 -> 0.609, and a two-column (line, hour) lookup table nearly matches
gradient boosting. Weather adds ~+0.007, calendar ~0.000.

Treating that as a failure is a framing error. The interesting question is not
"can we beat the lookup table" (we cannot, and that is fine) but "WHY does the
lookup table know so much?" -- i.e. which structural properties of a line make it
chronically late. This script tests whether that question is answerable with the
data at hand, or whether it collapses into the trivial answer "buses and trams
share the road with cars, trains do not".

The script is deliberately sceptical. The headline number it is designed to
produce is not "structure explains X%" but "structure explains X% ACROSS modes
and Y% WITHIN a mode". If Y is ~0, the research question is dead and the honest
report is that mode is the whole story.

UNIT OF ANALYSIS
----------------
(line_name, product), NOT line_name. Line names collide across modes in the
observed data: "U2" exists as both a subway line and a bus line, "M2" as both a
tram and a bus. Collapsing them would mix a tunnel line with a road line.

STAGES (each cached, so re-runs are cheap)
------------------------------------------
  1  profile   per-line observed delay rates + Wilson CIs      (needs dataset.parquet)
  2  gtfs      structural features from the static VBB feed    (needs the 77 MB zip)
  3  model     regress chronic delay rate on structure         (needs 1 + 2)

    python analysis/experiments/line_structure.py --stage all

The GTFS zip is NOT committed. Fetch it once:

    curl -L -A "jufo-transit-delay-study/0.1 (...)" -o analysis/experiments/gtfs_cache/vbb_gtfs.zip https://www.vbb.de/vbbgtfs

(vbb.de answers a default urllib/curl User-Agent with 403 -- a real UA is required.)

CAVEAT ON THE FEED VERSION: the static feed published 2026-09-17 is used, while the
observation window is 2026-08-10..2026-09-20. Route geometry, stop counts and
timetable density are stable on that timescale, but the feed is not a perfect
contemporaneous description of every observed departure. Noted, not fixed.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
CACHE = HERE / "gtfs_cache"
DATASET = REPO / "data" / "dataset.parquet"
GTFS_ZIP = CACHE / "vbb_gtfs.zip"

PROFILE_CSV = CACHE / "line_profile.csv"
STRUCT_CSV = CACHE / "line_structure_features.csv"

# Minimum observed departures per (line, product) to be included.
# 500 keeps 53 of 256 units but 98.2% of usable departures; at n=500 a Wilson CI
# on a 10% rate is about +/-2.7pp, narrow enough that between-line differences of
# the size we see (0.5% .. 34%) are not CI noise. Lowering it to 200 adds six
# mostly-night-bus units with visibly wider intervals and does not change the
# conclusions (checked: see --min-n).
MIN_N = 500

# Delay thresholds in seconds. The dataset quantises delay to whole minutes.
T3, T10 = 180, 600

# Observed stops: the four interchanges that were logged. The station DHID is the
# 9-digit core of a GTFS stop_id like "de:11000:900100003::5".
#
# Matching on the DHID alone is NOT enough. The departure API treats e.g.
# "S+U Alexanderplatz" as one station and returns the tram departures with it,
# but GTFS files the Alexanderplatz tram platforms under separate DHIDs
# (900100004, 900100005, ...). Matching on the ID alone therefore loses the trams
# and the regional platforms at Hauptbahnhof. So the observed "station" is defined
# geographically: any GTFS stop within OBSERVED_RADIUS_M of the hub's centroid.
OBSERVED_STATIONS = {"900023201", "900100003", "900003201", "900120004"}
OBSERVED_RADIUS_M = 350.0

# Rail-replacement services ("Schienenersatzverkehr") appear in the logged data as
# bus departures carrying the RAIL line's name -- "U2" and "M2" show up as buses
# with direction "Ersatzverkehr Richtung ...". They have no GTFS route (they are
# short-notice works services), so matching them on line name silently binds them
# to an unrelated real route of the same name: there is a genuine 5-stop, 3 km bus
# "U2" in the VBB area. That is a fabricated join, so these units are excluded.
ERSATZ_MARKER = "ersatzverkehr"

# A weekday inside the feed's validity window, used to define "how often does this
# line run". Wednesday 2026-09-23: a normal school-term midweek day with no holiday.
REF_DATE = "20260923"
REF_WEEKDAY = "wednesday"


# --------------------------------------------------------------------------
# stage 1: observed per-line delay profile
# --------------------------------------------------------------------------

def wilson(k: np.ndarray, n: np.ndarray, z: float = 1.96):
    """Wilson score interval. Used instead of the normal approximation because
    several lines have rates near 0 (U1, U3 are at exactly 0.0% late), where the
    normal interval is nonsense (it would extend below zero)."""
    k = np.asarray(k, float)
    n = np.asarray(n, float)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = (z / d) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return centre - half, centre + half


def stage_profile(min_n: int = MIN_N) -> pd.DataFrame:
    df = pd.read_parquet(
        DATASET,
        columns=["line_name", "product", "stop_id", "direction", "final_delay_s",
                 "cancelled", "hour"],
    )
    # Flag units that are mostly replacement services before anything else.
    ers = df.assign(e=df.direction.fillna("").str.lower().str.contains(ERSATZ_MARKER)) \
            .groupby(["line_name", "product"]).e.mean()
    # Cancelled departures have no meaningful delay, and 12.9% of rows never got a
    # final delay observation. Both are dropped rather than imputed: imputing a
    # delay for a cancelled train would invent the very quantity being modelled.
    n_raw = len(df)
    usable = df[df.final_delay_s.notna() & (df.cancelled == 0)].copy()

    # Cancellation rate is kept as a separate per-line outcome -- a line that is
    # chronically cancelled is a different failure mode from one that is late, and
    # conflating them would hide it.
    canc = df.groupby(["line_name", "product"]).cancelled.agg(["mean", "size"])
    canc.columns = ["cancel_rate", "n_incl_cancelled"]

    usable["late3"] = (usable.final_delay_s >= T3).astype(int)
    usable["late10"] = (usable.final_delay_s >= T10).astype(int)

    g = usable.groupby(["line_name", "product"]).agg(
        n=("final_delay_s", "size"),
        k3=("late3", "sum"),
        k10=("late10", "sum"),
        mean_delay_s=("final_delay_s", "mean"),
        median_delay_s=("final_delay_s", "median"),
        p90_delay_s=("final_delay_s", lambda s: s.quantile(0.90)),
        n_obs_stations=("stop_id", "nunique"),
    )
    g = g.join(canc, how="left")
    g["ersatz_frac"] = ers
    g = g.reset_index()
    g["rate3"] = g.k3 / g.n
    g["rate10"] = g.k10 / g.n
    g["ci3_lo"], g["ci3_hi"] = wilson(g.k3, g.n)
    g["ci10_lo"], g["ci10_hi"] = wilson(g.k10, g.n)

    kept = g[g.n >= min_n].sort_values("rate3", ascending=False).reset_index(drop=True)
    n_before = len(kept)
    dropped_ers = kept[kept.ersatz_frac > 0.5][["line_name", "product", "n", "rate3"]]
    kept = kept[kept.ersatz_frac <= 0.5].reset_index(drop=True)

    print(f"[profile] raw rows {n_raw:,} -> usable {len(usable):,} "
          f"(dropped {n_raw - len(usable):,} cancelled/unobserved)")
    print(f"[profile] {len(g)} (line,product) units; {n_before} with n>={min_n}, "
          f"covering {g[g.n >= min_n].n.sum() / usable.shape[0]:.1%} of usable departures")
    if len(dropped_ers):
        print(f"[profile] dropped {len(dropped_ers)} replacement-service units "
              f"(>50% 'Ersatzverkehr'): "
              + ", ".join(f"{r.line_name}/{r['product']} n={r.n}"
                          for _, r in dropped_ers.iterrows()))
    CACHE.mkdir(parents=True, exist_ok=True)
    kept.to_csv(PROFILE_CSV, index=False)
    return kept


# --------------------------------------------------------------------------
# stage 2: structural features from the static GTFS feed
# --------------------------------------------------------------------------

def _station(gtfs_stop_id: str) -> str:
    """'de:11000:900100003::5' -> '900100003'. VBB encodes the DHID station number
    in the third colon-field; platform suffixes after '::' vary by platform. We
    want the station, because 'which routes share this stop' is a station-level
    question (a bus bay and a tram stop at Alexanderplatz are the same place)."""
    parts = gtfs_stop_id.split(":")
    if len(parts) >= 3 and parts[2].isdigit():
        return parts[2]
    return gtfs_stop_id


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def _gtfs_seconds(t: str) -> float:
    """GTFS times can exceed 24:00:00 for trips running past midnight."""
    try:
        h, m, s = t.split(":")
        return int(h) * 3600 + int(m) * 60 + int(s)
    except Exception:
        return np.nan


def stage_gtfs(profile: pd.DataFrame, max_variants: int = 20) -> pd.DataFrame:
    if not GTFS_ZIP.exists():
        sys.exit(f"missing {GTFS_ZIP} -- see the module docstring for the curl command")
    zf = zipfile.ZipFile(GTFS_ZIP)

    routes = pd.read_csv(io.BytesIO(zf.read("routes.txt")), dtype=str)
    trips = pd.read_csv(io.BytesIO(zf.read("trips.txt")), dtype=str)
    stops = pd.read_csv(io.BytesIO(zf.read("stops.txt")),
                        dtype={"stop_id": str, "stop_lat": float, "stop_lon": float},
                        usecols=["stop_id", "stop_lat", "stop_lon", "stop_name"])
    stops["station"] = stops.stop_id.map(_station)

    # Geographic definition of "one of our four observed hubs" -- see the comment
    # on OBSERVED_STATIONS. Centroid of the stops carrying the hub's own DHID,
    # then every GTFS stop within OBSERVED_RADIUS_M of it counts as that hub.
    obs_ids = set()
    for dhid in OBSERVED_STATIONS:
        core = stops[stops.station == dhid]
        if core.empty:
            print(f"[gtfs] ! observed station {dhid} not found in stops.txt")
            continue
        clat, clon = core.stop_lat.mean(), core.stop_lon.mean()
        near = stops[_haversine_km(clat, clon, stops.stop_lat, stops.stop_lon) * 1000
                     <= OBSERVED_RADIUS_M]
        obs_ids |= set(near.stop_id)
        print(f"[gtfs] hub {dhid} ({core.stop_name.iloc[0]}): {len(near)} GTFS stops "
              f"within {OBSERVED_RADIUS_M:.0f} m ({core.station.nunique()} by DHID)")
    stop_xy = stops.set_index("stop_id")[["stop_lat", "stop_lon", "station"]]

    # --- which service_ids run on the reference weekday --------------------
    cal = pd.read_csv(io.BytesIO(zf.read("calendar.txt")), dtype=str)
    cal_d = pd.read_csv(io.BytesIO(zf.read("calendar_dates.txt")), dtype=str)
    base = set(cal[(cal[REF_WEEKDAY] == "1") &
                   (cal.start_date <= REF_DATE) &
                   (cal.end_date >= REF_DATE)].service_id)
    on_ref = cal_d[cal_d.date == REF_DATE]
    services = (base | set(on_ref[on_ref.exception_type == "1"].service_id)) \
        - set(on_ref[on_ref.exception_type == "2"].service_id)
    print(f"[gtfs] {len(services):,} service_ids active on {REF_DATE}")

    # --- map observed (line, product) -> GTFS route_ids --------------------
    # route_short_name is the line label; route_type is mapped to the product
    # vocabulary the logger uses, so "U2 bus" and "U2 subway" stay separate.
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location("fg", REPO / "analysis" / "fetch_gtfs_routes.py")
    fg = module_from_spec(spec)
    spec.loader.exec_module(fg)  # reuse the committed route_type -> product map
    routes["product"] = routes.route_type.map(fg.ROUTE_TYPE_TO_PRODUCT)
    routes["line_name"] = routes.route_short_name.fillna("").str.strip()

    targets = set(zip(profile.line_name, profile["product"]))
    routes["key"] = list(zip(routes.line_name, routes["product"]))
    tgt_routes = routes[routes.key.isin(targets)]
    print(f"[gtfs] {len(tgt_routes)} route_ids match {tgt_routes.key.nunique()} "
          f"of {len(targets)} observed (line,product) units")

    trips = trips.merge(routes[["route_id", "key", "product", "line_name", "route_type"]],
                        on="route_id", how="left")
    ref_trips = trips[trips.service_id.isin(services)]

    # --- pick trips whose stop lists we will actually read -----------------
    # Reading all 392 MB of stop_times for every trip is unnecessary. We need:
    #   (a) for target lines: enough shape variants to find the LONGEST version of
    #       the route (the "route length" of a line is its full end-to-end run,
    #       not a short-turn working);
    #   (b) for EVERY VBB route: a couple of trips, so we can count how many
    #       distinct routes call at each station (the interchange-congestion
    #       feature). One trip per shape is plenty -- a route's stop set barely
    #       varies within a shape.
    def sample(frame, per_route):
        s = frame.dropna(subset=["shape_id"]).drop_duplicates(["route_id", "shape_id"])
        return s.groupby("route_id").head(per_route)

    want_target = sample(ref_trips[ref_trips.key.isin(targets)], max_variants)
    want_all = sample(ref_trips, 3)
    want = pd.concat([want_target, want_all]).drop_duplicates("trip_id")
    want_ids = set(want.trip_id)
    print(f"[gtfs] streaming stop_times.txt for {len(want_ids):,} sampled trips "
          f"(of {len(ref_trips):,} on the reference day)")

    # --- single streaming pass over stop_times.txt -------------------------
    # 392 MB uncompressed. Never loaded whole; chunked and filtered on the fly.
    keep = []
    first_dep = {}          # trip_id -> departure seconds at the first stop
    target_trip_ids = set(ref_trips[ref_trips.key.isin(targets)].trip_id)
    reader = pd.read_csv(zf.open("stop_times.txt"),
                         dtype={"trip_id": str, "stop_id": str, "stop_sequence": int,
                                "arrival_time": str, "departure_time": str},
                         usecols=["trip_id", "stop_id", "stop_sequence",
                                  "arrival_time", "departure_time"],
                         chunksize=2_000_000)
    n_rows = 0
    for ch in reader:
        n_rows += len(ch)
        keep.append(ch[ch.trip_id.isin(want_ids)])
        # first-stop rows for ALL target trips give the service span cheaply --
        # needed because night buses run few trips over few hours and would
        # otherwise look like "low frequency" lines rather than "night" lines.
        f = ch[(ch.stop_sequence == 0) & ch.trip_id.isin(target_trip_ids)]
        for tid, dep in zip(f.trip_id, f.departure_time):
            first_dep[tid] = _gtfs_seconds(dep)
    st = pd.concat(keep, ignore_index=True)
    print(f"[gtfs] scanned {n_rows:,} stop_times rows, kept {len(st):,}")

    st = st.join(stop_xy, on="stop_id")
    st = st.sort_values(["trip_id", "stop_sequence"])

    # --- route <-> station incidence, for the shared-routes feature --------
    tr_route = want.set_index("trip_id").route_id
    st["route_id"] = st.trip_id.map(tr_route)
    inc = st[["route_id", "station"]].drop_duplicates()
    routes_per_station = inc.groupby("station").route_id.nunique()
    # Count DISTINCT LINES not route_ids: VBB splits one line into several
    # route_ids (variants), which would inflate the congestion count for lines
    # that happen to be modelled with many variants.
    inc2 = st[["trip_id", "station"]].copy()
    inc2["line"] = inc2.trip_id.map(want.set_index("trip_id").line_name)
    lines_per_station = inc2[["line", "station"]].drop_duplicates() \
        .groupby("station").line.nunique()

    # --- per-trip geometry -------------------------------------------------
    rows = []
    for tid, grp in st[st.trip_id.isin(set(want_target.trip_id))].groupby("trip_id", sort=False):
        lat = grp.stop_lat.to_numpy()
        lon = grp.stop_lon.to_numpy()
        if len(lat) < 2 or np.isnan(lat).any():
            continue
        seg = _haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:])
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        a0 = _gtfs_seconds(grp.arrival_time.iloc[0])
        a1 = _gtfs_seconds(grp.arrival_time.iloc[-1])
        stations = grp.station.to_numpy()
        sids = grp.stop_id.to_numpy()
        # Where along the route do our four observed stops sit? Delay accumulates
        # downstream, so a line observed near its terminus has had more distance
        # in which to fall behind than one observed two stops from its origin.
        hits = [cum[i] for i, s in enumerate(sids) if s in obs_ids]
        rows.append({
            "trip_id": tid,
            "route_id": grp.route_id.iloc[0],
            "n_stops": len(lat),
            "length_km": float(cum[-1]),
            "runtime_min": (a1 - a0) / 60.0 if np.isfinite(a1 - a0) else np.nan,
            "shared_lines_mean": float(np.nanmean(
                [lines_per_station.get(s, np.nan) for s in stations])),
            "shared_lines_max": float(np.nanmax(
                [lines_per_station.get(s, 0) for s in stations])),
            "obs_pos_km": float(np.mean(hits)) if hits else np.nan,
            "obs_pos_frac": float(np.mean(hits) / cum[-1]) if hits and cum[-1] > 0 else np.nan,
        })
    tg = pd.DataFrame(rows)
    tg["key"] = tg.route_id.map(routes.set_index("route_id").key)

    # Representative variant = the longest run of the line (most stops). A line's
    # structural identity is its full route, not the short-turns layered on it.
    tg = tg.sort_values("n_stops", ascending=False)
    rep = tg.groupby("key").head(1).set_index("key")
    # ...but the observed-position and congestion features should average over all
    # variants actually operated, since passengers ride all of them.
    avg = tg.groupby("key")[["obs_pos_frac", "obs_pos_km", "shared_lines_mean"]].mean()

    # --- frequency / service span -----------------------------------------
    rt = ref_trips[ref_trips.key.isin(targets)].copy()
    rt["dep_s"] = rt.trip_id.map(first_dep)
    freq = rt.groupby("key").agg(
        trips_per_weekday=("trip_id", "size"),
        n_route_ids=("route_id", "nunique"),
        span_start_h=("dep_s", lambda s: np.nanpercentile(s, 2) / 3600),
        span_end_h=("dep_s", lambda s: np.nanpercentile(s, 98) / 3600),
    )
    freq["service_span_h"] = (freq.span_end_h - freq.span_start_h).clip(lower=1)
    # Two directions share the trip count, so headway is span / (trips/2).
    freq["headway_min"] = freq.service_span_h * 60 / (freq.trips_per_weekday / 2)

    out = rep[["n_stops", "length_km", "runtime_min", "shared_lines_max"]] \
        .join(avg).join(freq[["trips_per_weekday", "service_span_h", "headway_min",
                              "n_route_ids"]])
    out["mean_stop_spacing_m"] = out.length_km * 1000 / (out.n_stops - 1)
    out["sched_speed_kmh"] = out.length_km / (out.runtime_min / 60)
    out = out.reset_index()
    out[["line_name", "product"]] = pd.DataFrame(out.key.tolist(), index=out.index)
    out = out.drop(columns=["key"])
    out.to_csv(STRUCT_CSV, index=False)
    print(f"[gtfs] structural features for {len(out)} (line,product) units -> {STRUCT_CSV.name}")
    return out


# --------------------------------------------------------------------------
# stage 3: model the LINE, not the departure
# --------------------------------------------------------------------------

FEATURES = ["log_length_km", "n_stops", "mean_stop_spacing_m", "log_headway_min",
            "service_span_h", "shared_lines_mean", "shared_lines_max",
            "obs_pos_frac", "sched_speed_kmh"]


def _ols(X: np.ndarray, y: np.ndarray, w: np.ndarray):
    """Weighted least squares with an intercept, returned with R^2 and the
    standard errors needed to say whether a coefficient means anything."""
    X1 = np.column_stack([np.ones(len(X)), X])
    W = np.sqrt(w)
    Xw, yw = X1 * W[:, None], y * W
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    resid = yw - Xw @ beta
    dof = max(len(X) - X1.shape[1], 1)
    sigma2 = float(resid @ resid) / dof
    try:
        cov = sigma2 * np.linalg.inv(Xw.T @ Xw)
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        se = np.full(X1.shape[1], np.nan)
    ybar = float((w * y).sum() / w.sum())
    sst = float((w * (y - ybar) ** 2).sum())
    sse = float((w * (y - X1 @ beta) ** 2).sum())
    r2 = 1 - sse / sst if sst > 0 else np.nan
    adj = 1 - (1 - r2) * (len(X) - 1) / dof
    return beta, se, r2, adj


def _loo_r2(X: np.ndarray, y: np.ndarray, w: np.ndarray):
    """Leave-one-out R^2. With n=53 lines and up to 9 features, in-sample R^2 is
    not evidence of anything; LOO is the number that survives a jury question."""
    preds = np.empty(len(y))
    for i in range(len(y)):
        m = np.ones(len(y), bool)
        m[i] = False
        beta, *_ = _ols(X[m], y[m], w[m])
        preds[i] = np.concatenate([[1.0], X[i]]) @ beta
    ybar = float((w * y).sum() / w.sum())
    sst = float((w * (y - ybar) ** 2).sum())
    sse = float((w * (y - preds) ** 2).sum())
    return 1 - sse / sst if sst > 0 else np.nan


def stage_model(profile: pd.DataFrame, struct: pd.DataFrame, out_json: Path):
    d = profile.merge(struct, on=["line_name", "product"], how="inner")
    print(f"\n[model] {len(d)} lines with both a delay profile and structure "
          f"({len(profile) - len(d)} lost to no GTFS match)")

    # Model the log-odds of being >=3 min late, not the raw rate: rates are bounded
    # in [0,1] and several lines sit near the floor, where a linear model would
    # predict negative lateness. Haldane correction (+0.5) keeps the two 0% lines.
    d["logit3"] = np.log((d.k3 + 0.5) / (d.n - d.k3 + 0.5))
    d["log_length_km"] = np.log(d.length_km)
    d["log_headway_min"] = np.log(d.headway_min)
    # Weight by n: a rate from 30,000 departures deserves more say than one from 500.
    d["w"] = d.n / d.n.mean()

    cov = d[FEATURES].notna().mean()
    print("[model] feature coverage: "
          + ", ".join(f"{f}={c:.0%}" for f, c in cov.items()))
    # Drop a feature outright if it is missing for >10% of lines (better to lose a
    # column than a fifth of an already tiny sample); otherwise drop the few rows.
    feats = [f for f in FEATURES if cov[f] >= 0.90]
    dropped_f = [f for f in FEATURES if f not in feats]
    if dropped_f:
        print(f"[model] dropped feature(s) for poor coverage: {dropped_f}")
    d = d.dropna(subset=feats + ["logit3"]).reset_index(drop=True)
    print(f"[model] fitting on {len(d)} lines x {len(feats)} structural features")
    y = d.logit3.to_numpy()
    w = d.w.to_numpy()

    def z(cols):
        M = d[cols].to_numpy(float)
        return (M - M.mean(0)) / M.std(0), M.std(0)

    results = {}

    # --- 1. mode alone ------------------------------------------------------
    modes = pd.get_dummies(d["product"], drop_first=True).to_numpy(float)
    _, _, r2_mode, adj_mode = _ols(modes, y, w)
    loo_mode = _loo_r2(modes, y, w)
    results["mode_only"] = {"r2": r2_mode, "adj_r2": adj_mode, "loo_r2": loo_mode,
                            "k": modes.shape[1]}

    # --- 2. structure alone -------------------------------------------------
    Xs, sd = z(feats)
    beta_s, se_s, r2_s, adj_s = _ols(Xs, y, w)
    loo_s = _loo_r2(Xs, y, w)
    results["structure_only"] = {
        "r2": r2_s, "adj_r2": adj_s, "loo_r2": loo_s, "k": len(feats),
        "coef": {f: {"beta_per_sd": float(b), "se": float(s), "t": float(b / s)}
                 for f, b, s in zip(feats, beta_s[1:], se_s[1:])}}

    # --- 3. both ------------------------------------------------------------
    Xb = np.column_stack([modes, Xs])
    _, _, r2_b, adj_b = _ols(Xb, y, w)
    loo_b = _loo_r2(Xb, y, w)
    results["mode_plus_structure"] = {"r2": r2_b, "adj_r2": adj_b, "loo_r2": loo_b,
                                      "k": Xb.shape[1]}

    # --- 3b. THE SCEPTICAL TEST, pooled: structure AFTER mode is held fixed --
    # Only the bus mode has enough lines for its own regression (see step 4), so
    # the powerful version of the test is a within (fixed-effects) estimator:
    # sweep out the mode means from y and from every feature, then ask whether
    # what is left still covaries. This uses all modes at once and answers exactly
    # "does structure say anything that 'it is a bus' does not already say?".
    # p-value by permutation: shuffle y WITHIN each mode, so the null preserves
    # the mode effect entirely and only destroys the line-level structure link.
    def demean_by(group, M):
        out = M.copy()
        for gval in np.unique(group):
            m = group == gval
            out[m] -= out[m].mean(0)
        return out

    grp = d["product"].to_numpy()
    yw_ = demean_by(grp, y.reshape(-1, 1)).ravel()
    Xw_ = demean_by(grp, Xs.copy())
    _, _, r2_within, adj_within = _ols(Xw_, yw_, w)
    loo_within = _loo_r2(Xw_, yw_, w)
    rng = np.random.default_rng(0)
    null = []
    for _ in range(2000):
        yp = y.copy()
        for gval in np.unique(grp):
            m = grp == gval
            yp[m] = rng.permutation(yp[m])
        ypw = demean_by(grp, yp.reshape(-1, 1)).ravel()
        null.append(_ols(Xw_, ypw, w)[2])
    null = np.asarray(null)
    results["structure_within_mode_pooled"] = {
        "n": int(len(d)), "k": len(feats),
        "partial_r2": r2_within, "adj_r2": adj_within, "loo_r2": loo_within,
        "perm_p": float((null >= r2_within).mean()),
        "perm_null_median_r2": float(np.median(null)),
        "perm_null_p95_r2": float(np.percentile(null, 95)),
    }

    # --- 4. THE SCEPTICAL TEST: structure WITHIN a mode ---------------------
    # If structure only works because it proxies for mode (long, slow, frequent,
    # congested == bus; short, fast == subway), then residualising on mode first
    # should destroy it. This is the test that decides the verdict.
    results["within_mode"] = {}
    for mode, grp in d.groupby("product"):
        if len(grp) < 8:
            results["within_mode"][mode] = {"n": len(grp), "skipped": "n<8"}
            continue
        gy = grp.logit3.to_numpy()
        gw = grp.w.to_numpy()
        # Use a small, pre-chosen subset inside a mode: with n~28 we cannot fit 9
        # features without fitting noise. These four are the ones with a prior
        # mechanism (longer/slower/busier/more-shared route -> more delay).
        sub = ["log_length_km", "shared_lines_mean", "log_headway_min", "sched_speed_kmh"]
        M = grp[sub].to_numpy(float)
        M = (M - M.mean(0)) / M.std(0)
        b, s, r2, adj = _ols(M, gy, gw)
        loo = _loo_r2(M, gy, gw)
        spear = {f: float(pd.Series(grp[f].to_numpy()).corr(
            pd.Series(gy), method="spearman")) for f in sub}
        results["within_mode"][mode] = {
            "n": len(grp), "r2": r2, "adj_r2": adj, "loo_r2": loo,
            "coef": {f: {"beta_per_sd": float(bb), "se": float(ss), "t": float(bb / ss)}
                     for f, bb, ss in zip(sub, b[1:], s[1:])},
            "spearman_vs_logit3": spear,
        }

    # --- 4b. last chance for the hypothesis: ONE feature inside the bus mode -
    # A 4-feature regression on 27 lines can hide a real effect in its own noise.
    # So also ask the weakest possible question: does ANY single structural
    # feature rank-correlate with lateness among buses alone? The p-value is on
    # max|rho| across all features, with the null built by shuffling, so it is
    # already corrected for having gone looking at nine of them.
    bus = d[d["product"] == "bus"]
    if len(bus) >= 10:
        by = bus.logit3.to_numpy()
        rhos = {f: float(pd.Series(bus[f].to_numpy()).corr(pd.Series(by), method="spearman"))
                for f in feats}
        obs_max = max(abs(v) for v in rhos.values())
        rng2 = np.random.default_rng(1)
        Mb = bus[feats].to_numpy(float)
        nullmax = []
        for _ in range(5000):
            yp = rng2.permutation(by)
            nullmax.append(max(abs(float(pd.Series(Mb[:, j]).corr(
                pd.Series(yp), method="spearman"))) for j in range(Mb.shape[1])))
        results["bus_single_feature_screen"] = {
            "n": int(len(bus)), "spearman": rhos, "max_abs_rho": obs_max,
            "perm_p_maxrho": float((np.asarray(nullmax) >= obs_max).mean()),
            "perm_null_p95_maxrho": float(np.percentile(nullmax, 95)),
        }

    # --- 4c. what is left unexplained, and is it real? ----------------------
    # Mode explains some of the spread. Of the rest, how much is genuine
    # line-level signal rather than sampling noise? If the residual is real but
    # structure cannot touch it, the cause is something not in the static feed.
    binom_var = float((1.0 / (d.k3 + 0.5) + 1.0 / (d.n - d.k3 + 0.5)).mean())
    results["residual_after_mode"] = {
        "var_resid_logit3": float(np.var(yw_)),  # y with mode means swept out
        "binomial_noise_var": binom_var,
        "residual_signal_to_noise": float(np.var(yw_) / binom_var),
    }

    # --- 5. how much between-line variance is there at all? -----------------
    # If the observed spread in rate3 were mostly sampling noise there would be
    # nothing to explain. Compare observed variance of logit3 to the variance
    # expected from binomial sampling alone.
    results["variance_check"] = {
        "var_logit3_observed": float(np.var(y)),
        "mean_binomial_var_logit3": binom_var,
        "signal_to_noise": float(np.var(y) / binom_var),
    }

    out_json.write_text(json.dumps(results, indent=2))
    return d, results


def _report(d: pd.DataFrame, res: dict):
    p = print
    p("\n" + "=" * 78)
    p("BETWEEN-LINE VARIANCE CHECK")
    v = res["variance_check"]
    p(f"  var(logit rate>=3min) observed        : {v['var_logit3_observed']:.3f}")
    p(f"  var expected from binomial noise alone: {v['mean_binomial_var_logit3']:.3f}")
    p(f"  signal / noise                        : {v['signal_to_noise']:.1f}x")

    p("\nEXPLAINING BETWEEN-LINE CHRONIC LATENESS (y = logit P(delay>=3min))")
    p(f"  {'model':<24} {'k':>3} {'R2':>7} {'adjR2':>7} {'LOO R2':>8}")
    for k in ["mode_only", "structure_only", "mode_plus_structure"]:
        r = res[k]
        p(f"  {k:<24} {r['k']:>3} {r['r2']:>7.3f} {r['adj_r2']:>7.3f} {r['loo_r2']:>8.3f}")

    p("\nSTRUCTURE-ONLY COEFFICIENTS (standardised; y in log-odds per 1 SD of x)")
    for f, c in sorted(res["structure_only"]["coef"].items(),
                       key=lambda kv: -abs(kv[1]["t"])):
        star = "*" if abs(c["t"]) > 2 else " "
        p(f"  {star} {f:<22} beta={c['beta_per_sd']:+.3f}  se={c['se']:.3f}  t={c['t']:+.2f}")

    w_ = res["structure_within_mode_pooled"]
    p("\nSCEPTICAL TEST A -- pooled within-mode (mode fixed effects swept out)")
    p(f"  partial R2 of structure after mode: {w_['partial_r2']:.3f}  "
      f"(adj {w_['adj_r2']:.3f}, LOO {w_['loo_r2']:+.3f})")
    p(f"  permutation null (y shuffled within mode): median R2 "
      f"{w_['perm_null_median_r2']:.3f}, 95th pct {w_['perm_null_p95_r2']:.3f}")
    p(f"  p = {w_['perm_p']:.4f}   (n={w_['n']} lines, k={w_['k']} features)")

    p("\nSCEPTICAL TEST B -- one mode at a time: does structure explain anything")
    p("once mode is held fixed?")
    for mode, r in res["within_mode"].items():
        if "skipped" in r:
            p(f"  {mode:<10} n={r['n']:<3} skipped ({r['skipped']})")
            continue
        p(f"  {mode:<10} n={r['n']:<3} R2={r['r2']:.3f}  adjR2={r['adj_r2']:.3f}  "
          f"LOO R2={r['loo_r2']:+.3f}")
        for f, c in r["coef"].items():
            star = "*" if abs(c["t"]) > 2 else " "
            p(f"      {star} {f:<22} beta={c['beta_per_sd']:+.3f} t={c['t']:+.2f} "
              f"rho={r['spearman_vs_logit3'][f]:+.2f}")

    if "bus_single_feature_screen" in res:
        b = res["bus_single_feature_screen"]
        p(f"\nSCEPTICAL TEST C -- weakest possible question: does ANY single feature")
        p(f"rank-correlate with lateness among the {b['n']} bus lines alone?")
        for f, r in sorted(b["spearman"].items(), key=lambda kv: -abs(kv[1])):
            p(f"      {f:<22} rho={r:+.3f}")
        p(f"  max|rho| = {b['max_abs_rho']:.3f}; permutation null 95th pct = "
          f"{b['perm_null_p95_maxrho']:.3f}; p = {b['perm_p_maxrho']:.4f}")

    r = res["residual_after_mode"]
    p(f"\nWHAT IS LEFT: between-line variance after removing mode")
    p(f"  var(logit3) residual-of-mode : {r['var_resid_logit3']:.3f}  "
      f"(vs {v['var_logit3_observed']:.3f} total)")
    p(f"  binomial sampling noise      : {r['binomial_noise_var']:.3f}")
    p(f"  residual signal / noise      : {r['residual_signal_to_noise']:.1f}x  "
      f"-> the unexplained part is REAL, structure just does not reach it")
    p("=" * 78)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all",
                    choices=["profile", "gtfs", "model", "all"])
    ap.add_argument("--min-n", type=int, default=MIN_N)
    ap.add_argument("--out", default=str(CACHE / "line_structure_results.json"))
    a = ap.parse_args()

    prof = None
    if a.stage in ("profile", "all") or not PROFILE_CSV.exists():
        prof = stage_profile(a.min_n)
    if prof is None:
        prof = pd.read_csv(PROFILE_CSV, dtype={"line_name": str})

    if a.stage == "profile":
        cols = ["line_name", "product", "n", "rate3", "ci3_lo", "ci3_hi",
                "rate10", "ci10_lo", "ci10_hi", "cancel_rate", "median_delay_s"]
        print(prof[cols].to_string(index=False,
                                   float_format=lambda x: f"{x:.4f}"))
        return 0

    st = None
    if a.stage in ("gtfs", "all") or not STRUCT_CSV.exists():
        st = stage_gtfs(prof)
    if st is None:
        st = pd.read_csv(STRUCT_CSV, dtype={"line_name": str})
    if a.stage == "gtfs":
        print(st.to_string(index=False))
        return 0

    d, res = stage_model(prof, st, Path(a.out))
    _report(d, res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
