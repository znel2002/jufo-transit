"""Calendar features for Berlin: school holidays, public holidays, rush hour.

WHY THE SCHOOL HOLIDAYS ARE HARDCODED:
They are fetched once from an authoritative source and committed, rather than looked
up at analysis time. Two reasons, one of them learned the hard way on 2026-08-10:

  * ferien-api.de -- the obvious choice -- returns an empty list for Berlin 2026 and
    2027. A third-party API that silently returns `[]` would not crash the pipeline;
    it would quietly mark every day as a school day and weaken a real effect.
  * The analysis has to be reproducible in January 2027 and re-runnable by a jury
    afterwards. A committed table cannot go offline.

Source: OpenHolidays API (https://openholidaysapi.org), an EU-funded open dataset,
queried 2026-08-10 for subdivision DE-BE. Refresh with:

    python analysis/calendar_de.py --refresh

which re-queries and prints an updated SCHOOL_HOLIDAYS block to paste in, so the
provenance stays visible in the diff instead of hiding inside a network call.

Public holidays come from the `holidays` package (`holidays.Germany(subdiv="BE")`) --
it is offline, versioned and knows Berlin's specifics such as Frauentag (8 March).

ALL CALENDAR FEATURES USE LOCAL BERLIN TIME, NOT UTC. Rush hour is a human
phenomenon and follows the clock on the wall; using UTC hours would shift every
feature by an hour when DST ends (2026-10-25, inside the logging period).
"""
from __future__ import annotations

import argparse
import json
import urllib.request

import pandas as pd

TZ = "Europe/Berlin"

# (start, end, name) -- both ends INCLUSIVE. Berlin (DE-BE), fetched 2026-08-10.
SCHOOL_HOLIDAYS = [
    ("2026-07-09", "2026-08-22", "Sommerferien"),
    ("2026-10-19", "2026-10-31", "Herbstferien"),
    ("2026-12-23", "2027-01-02", "Weihnachtsferien"),
    ("2027-02-01", "2027-02-06", "Winterferien"),
    ("2027-03-22", "2027-04-02", "Osterferien"),
    ("2027-05-07", "2027-05-07", "Unterrichtsfreier Tag"),
    ("2027-05-18", "2027-05-19", "Pfingstferien"),
    ("2027-07-01", "2027-08-14", "Sommerferien"),
]

# Berlin peaks, local time. Deliberately coarse: the model gets the raw `hour`
# anyway, so these only need to be honest, not finely tuned.
MORNING_PEAK = (6, 9)     # 06:00-08:59
EVENING_PEAK = (15, 19)   # 15:00-18:59


def _local(ts: pd.Series) -> pd.Series:
    """Timestamps as tz-aware Europe/Berlin, whatever they came in as."""
    out = pd.to_datetime(ts, errors="coerce", utc=True)
    return out.dt.tz_convert(TZ)


def is_school_holiday(ts: pd.Series) -> pd.Series:
    days = _local(ts).dt.normalize().dt.tz_localize(None)
    flag = pd.Series(False, index=ts.index)
    for start, end, _ in SCHOOL_HOLIDAYS:
        flag |= days.between(pd.Timestamp(start), pd.Timestamp(end))
    return flag


def is_public_holiday(ts: pd.Series) -> pd.Series:
    import holidays

    local = _local(ts)
    years = sorted(local.dt.year.dropna().unique().astype(int))
    if not years:
        return pd.Series(False, index=ts.index)
    be = holidays.Germany(subdiv="BE", years=years)
    return local.dt.date.map(lambda d: d in be).astype(bool)


def add_calendar_features(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    """Attach calendar features derived from `ts_col` (local Berlin time)."""
    local = _local(df[ts_col])
    out = df.copy()
    out["hour"] = local.dt.hour
    out["minute"] = local.dt.minute
    out["weekday"] = local.dt.weekday          # 0 = Monday
    out["is_weekend"] = out["weekday"] >= 5
    out["month"] = local.dt.month
    out["day_of_year"] = local.dt.dayofyear
    out["is_school_holiday"] = is_school_holiday(df[ts_col])
    out["is_public_holiday"] = is_public_holiday(df[ts_col])
    out["is_morning_peak"] = out["hour"].between(*MORNING_PEAK, inclusive="left")
    out["is_evening_peak"] = out["hour"].between(*EVENING_PEAK, inclusive="left")
    # A working day is the interesting contrast: not weekend, not public holiday,
    # not school holiday -- i.e. full commuter *and* school traffic.
    out["is_full_traffic_day"] = ~(
        out["is_weekend"] | out["is_public_holiday"] | out["is_school_holiday"]
    )
    return out


def _refresh(date_from: str, date_to: str) -> None:
    url = (
        "https://openholidaysapi.org/SchoolHolidays"
        f"?countryIsoCode=DE&subdivisionCode=DE-BE"
        f"&validFrom={date_from}&validTo={date_to}&languageIsoCode=DE"
    )
    with urllib.request.urlopen(url, timeout=60) as resp:
        data = json.load(resp)
    if not data:
        print("!! the API returned nothing -- do NOT paste an empty table, "
              "check the source before touching SCHOOL_HOLIDAYS")
        return
    print(f"# Berlin (DE-BE), fetched from openholidaysapi.org for {date_from}..{date_to}")
    print("SCHOOL_HOLIDAYS = [")
    for h in data:
        name = next((n["text"] for n in h["name"] if n["language"] == "DE"), "?")
        print(f'    ("{h["startDate"]}", "{h["endDate"]}", "{name}"),')
    print("]")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true",
                    help="re-query OpenHolidays and print a SCHOOL_HOLIDAYS block")
    ap.add_argument("--from", dest="date_from", default="2026-06-01")
    ap.add_argument("--to", dest="date_to", default="2027-08-31")
    args = ap.parse_args()

    if args.refresh:
        _refresh(args.date_from, args.date_to)
        return

    # Self-check: show the features for a week that spans a school-holiday boundary.
    ts = pd.Series(pd.date_range("2026-08-19 08:00", periods=8, freq="D", tz=TZ))
    demo = add_calendar_features(pd.DataFrame({"planned_when": ts}), "planned_when")
    cols = ["planned_when", "weekday", "is_weekend", "is_school_holiday",
            "is_public_holiday", "is_full_traffic_day"]
    print("Sommerferien 2026 end on 2026-08-22 -- the flag must flip after that:")
    print(demo[cols].to_string(index=False))


if __name__ == "__main__":
    main()
