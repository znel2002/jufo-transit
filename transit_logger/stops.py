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
    ("900003201", "S+U Berlin Hauptbahnhof",          "suburban, subway, tram, bus, express, regional"),
]

# IDs only, for iteration.
STOP_IDS = [s[0] for s in STOPS]
