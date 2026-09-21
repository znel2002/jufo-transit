"""Stops to log, chosen for a diverse product mix across high-traffic interchanges.

IDs resolved from the VBB/BVG public API (v6.bvg.transport.rest) on 2026-08-05.
Keep this list small (4-6 stops): depth of a clean time series beats breadth.
Every stop below is a multi-product interchange, so a handful of stops already
covers U-Bahn, S-Bahn, tram, bus and regional without extra requests.
"""

STOPS = [
    # id,          human name,                         products present
    ("900100003", "S+U Alexanderplatz",               "subway, tram, bus, suburban"),
    ("900023201", "S+U Zoologischer Garten",          "suburban, subway, bus, regional"),
    ("900120004", "S+U Warschauer Str.",              "suburban, subway, tram, bus"),
    ("900003201", "S+U Berlin Hauptbahnhof",          "suburban, subway, tram, express, regional"),
]

# IDs only, for iteration. The transport.rest API resolves a whole station
# server-side from these, so the hub id alone is enough for that source.
STOP_IDS = [s[0] for s in STOPS]

# ---------------------------------------------------------------------------
# Sibling station ids, for the GTFS-RT source ONLY.
#
# Berlin does not file a whole interchange under one id, and the GTFS-RT feed
# names the individual station rather than the interchange. Filtering it on the
# hub id alone (as this logger did from 2026-08-23 to 2026-09-21) silently
# captured ZERO trams and ZERO buses at Alexanderplatz and dropped regional and
# express services everywhere -- Hauptbahnhof's regional platforms are a separate
# station id, 900003200.
#
# That degraded only the BACKUP source, whose entire purpose is to cover the
# primary's outages, so a future outage would have been covered without trams,
# buses or regional trains. The transport.rest source was never affected.
#
# Derived by analysis/derive_stop_siblings.py from the VBB static GTFS feed:
# within 400 m of the hub AND the name contains the hub's distinguishing word.
# Proximity alone would swallow genuinely separate stops nearby (Littenstr.,
# Memhardstr., Helsingforser Platz); the name test keeps them out. Checked in as
# a literal rather than computed at runtime, so a change is visible in a diff and
# no 77 MB download is needed to run the logger.
STOP_SIBLINGS = {
    "900100003": [
        "900100003",    # S+U Alexanderplatz Bhf (Berlin)   <- hub
        "900100005",    # U Alexanderplatz (Berlin) [Tram]
        "900100006",    # S+U Alexanderplatz Bhf/Grunerstr. (Berlin)
        "900100024",    # S+U Alexanderplatz Bhf/Dircksenstr. (Berlin)
        "900100026",    # S+U Alexanderplatz Bhf/Gontardstr. (Berlin)
        "900100031",    # S+U Alexanderplatz Bhf/Memhardstr. (Berlin)
    ],
    "900023201": [
        "900023172",    # S+U Zoologischer Garten/Jebensstr. (Berlin)
        "900023201",    # S+U Zoologischer Garten Bhf (Berlin)   <- hub
    ],
    "900120004": [
        "900120004",    # S+U Warschauer Str. (Berlin)   <- hub
        "900120011",    # S Warschauer Str. (Berlin)
    ],
    "900003201": [
        "900003200",    # S+U Berlin Hauptbahnhof   (regional/long-distance)
        "900003201",    # S+U Berlin Hauptbahnhof   <- hub
        "900003205",    # S+U Hauptbahnhof/Washingtonplatz (Berlin)
    ],
}

# Flat set for GTFS-RT filtering, plus the reverse map so every captured row can
# still be attributed to the interchange it belongs to.
GTFSRT_STOP_IDS = {sid for ids in STOP_SIBLINGS.values() for sid in ids}
SIBLING_TO_HUB = {sid: hub for hub, ids in STOP_SIBLINGS.items() for sid in ids}
