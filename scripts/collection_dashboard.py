"""Collection dashboard: how much has been logged, and how complete is it?

    python scripts/collection_dashboard.py                  # terminal summary + HTML
    python scripts/collection_dashboard.py --no-gh          # skip the GitHub API call
    python scripts/collection_dashboard.py --open           # and open it in a browser

Volume alone ("we have N rows") says nothing about whether the record is usable.
GitHub Actions runs schedules on a best-effort basis, so the real question is
COVERAGE: of the 15-minute slots that should have produced a poll, how many did?
Every missing slot is a hole in the time series that can never be backfilled, so
this dashboard measures the holes as prominently as the volume, and the numbers it
produces feed the Fehlerquellen chapter directly.

Standard library only, on purpose: no pandas, no matplotlib. That keeps it runnable
anywhere -- including inside the Actions runner itself, should the dashboard ever be
published automatically.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import html
import json
import subprocess
import webbrowser
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OBS_DIR = ROOT / "data" / "observations"
DEFAULT_OUT = ROOT / "docs" / "dashboard.html"

POLL_INTERVAL_S = 15 * 60      # the workflow's */15 schedule
CYCLE_GAP_S = 5 * 60           # observations closer than this belong to one cycle
EXPECTED_PER_HOUR = 3600 // POLL_INTERVAL_S

# Jugend forscht 2027 milestones -- the dashboard projects the record forward to these.
MILESTONES = [
    ("Anmeldung (Titel + Kurzbeschreibung)", "2026-11-30"),
    ("Langfassung + Kurzfassung", "2027-01-31"),
    ("Regionalwettbewerb Berlin", "2027-02-15"),
]


# --------------------------------------------------------------------------- load

def load_rows(obs_dir: Path = OBS_DIR) -> list[dict]:
    """Read every poll file. Handles the current gzip layout and the legacy one."""
    rows: list[dict] = []
    files = sorted(glob.glob(str(obs_dir / "*" / "*.ndjson.gz")))
    for path in files:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            rows.extend(json.loads(line) for line in fh if line.strip())
    # Pre-2026-08-10 layout, still present in older checkouts.
    legacy = sorted(glob.glob(str(obs_dir / "*.ndjson")))
    for path in legacy:
        with open(path, encoding="utf-8") as fh:
            rows.extend(json.loads(line) for line in fh if line.strip())
    if not rows:
        raise SystemExit(f"no observations found under {obs_dir} -- has the logger run?")
    print(f"read {len(files)} poll files"
          + (f" + {len(legacy)} legacy files" if legacy else ""))
    return rows


def load_meta(obs_dir: Path = OBS_DIR) -> list[dict]:
    """Per-cycle poll status sidecars written by the logger.

    Without these a cycle is only visible through the rows it produced, so a poll
    that ran and failed looks exactly like a slot GitHub never fired -- and the
    coverage figure blames the scheduler for an upstream outage. Files written
    before 2026-08-10 have no sidecar, hence the row-timestamp fallback.
    """
    metas = []
    for path in sorted(glob.glob(str(obs_dir / "*" / "*.poll.json"))):
        try:
            metas.append(json.loads(Path(path).read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return metas


def _parse(ts: str) -> datetime | None:
    try:
        return datetime.fromisoformat(ts).astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None


# ------------------------------------------------------------------------ analyse

def find_cycles(rows: list[dict], metas: list[dict] | None = None) -> list[datetime]:
    """Cluster observation timestamps into poll cycles.

    One cycle writes all four stops within a few seconds. Clustering on the
    timestamps (rather than trusting filenames) means the same code works for both
    storage layouts and reflects when polling actually happened. Cycles that
    produced no rows contribute no observations, so their timestamps come from the
    status sidecars -- otherwise a failed poll would silently count as a gap.
    """
    stamped = {t for t in (_parse(r.get("observed_at", "")) for r in rows) if t}
    for m in metas or []:
        t = _parse(m.get("cycle_at", ""))
        if t:
            stamped.add(t)
    stamps = sorted(stamped)
    cycles: list[datetime] = []
    for s in stamps:
        if not cycles or (s - cycles[-1]).total_seconds() > CYCLE_GAP_S:
            cycles.append(s)
    return cycles


def coverage(cycles: list[datetime]) -> dict:
    """Actual vs. expected polls, and the gaps in between."""
    first, last = cycles[0], cycles[-1]
    span_s = (last - first).total_seconds()
    expected = int(span_s // POLL_INTERVAL_S) + 1
    gaps = []
    for a, b in zip(cycles, cycles[1:]):
        delta = (b - a).total_seconds()
        if delta > POLL_INTERVAL_S * 1.6:      # missed at least one slot
            gaps.append((a, b, delta, int(delta // POLL_INTERVAL_S) - 1))
    gaps.sort(key=lambda g: -g[2])
    return {
        "first": first, "last": last, "span_s": span_s,
        "actual": len(cycles), "expected": expected,
        "completeness": len(cycles) / expected if expected else 0.0,
        "missed": max(0, expected - len(cycles)),
        "gaps": gaps,
        "longest_gap_s": gaps[0][2] if gaps else 0.0,
    }


def departures(rows: list[dict]) -> dict:
    """Collapse observations to departures, keeping the last estimate for each."""
    latest: dict[tuple, dict] = {}
    for r in rows:
        key = (r.get("trip_id"), r.get("stop_id"), r.get("planned_when"))
        seen = latest.get(key)
        if seen is None or (r.get("observed_at") or "") > (seen.get("observed_at") or ""):
            latest[key] = r
    live = [r for r in latest.values() if not r.get("cancelled")]
    delays = sorted(r["delay_s"] for r in live if r.get("delay_s") is not None)
    return {
        "n": len(latest),
        "cancelled": sum(1 for r in latest.values() if r.get("cancelled")),
        "no_realtime": sum(1 for r in live if r.get("delay_s") is None),
        "delays": delays,
        "by_product": Counter(r.get("product") or "?" for r in latest.values()),
        "delay_by_product": _delay_by_product(live),
        "lines": len({r.get("line_name") for r in latest.values()}),
        "stops": len({r.get("stop_id") for r in latest.values()}),
    }


def _delay_by_product(live: list[dict]) -> dict[str, dict]:
    buckets: dict[str, list[int]] = defaultdict(list)
    for r in live:
        if r.get("delay_s") is not None:
            buckets[r.get("product") or "?"].append(r["delay_s"])
    out = {}
    for product, vals in buckets.items():
        vals.sort()
        out[product] = {
            "n": len(vals),
            "median": vals[len(vals) // 2] / 60,
            "mean": sum(vals) / len(vals) / 60,
            "p95": vals[min(len(vals) - 1, int(0.95 * len(vals)))] / 60,
            "share_late": sum(1 for v in vals if v > 180) / len(vals),
        }
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["n"]))


def hourly_grid(cycles: list[datetime]) -> tuple[list[str], dict]:
    """polls per (day, hour) -- the coverage heatmap's data."""
    grid: dict[tuple[str, int], int] = Counter()
    for c in cycles:
        grid[(c.strftime("%Y-%m-%d"), c.hour)] += 1
    start, end = cycles[0].date(), cycles[-1].date()
    days = [(start + timedelta(days=i)).isoformat()
            for i in range((end - start).days + 1)]
    return days, grid


def gh_runs(limit: int = 100) -> list[dict] | None:
    """Recent workflow runs, straight from the GitHub API via the gh CLI."""
    try:
        out = subprocess.run(
            ["gh", "run", "list", "--workflow=logger.yml", f"--limit={limit}",
             "--json", "event,status,conclusion,createdAt"],
            capture_output=True, text=True, timeout=60, cwd=ROOT,
        )
        if out.returncode != 0:
            return None
        return json.loads(out.stdout)
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return None


PROJECTION_MIN_SPAN_S = 86400   # refuse to extrapolate from less than a full day


def project(cov: dict, dep: dict, rows_n: int) -> tuple[list[dict], bool]:
    """Extrapolate the record forward to each competition milestone.

    Returns (rows, reliable). Extrapolating a daily total from a few polls would be
    nonsense: the logger's 30-minute look-ahead overlaps between cycles, so the share
    of *new* departures per poll only settles once a full day (with its night lull
    and rush hours) is in the data. Below that threshold the numbers are withheld
    rather than shown with a caveat nobody reads.
    """
    now = datetime.now(timezone.utc)
    reliable = cov["span_s"] >= PROJECTION_MIN_SPAN_S
    days = max(cov["span_s"] / 86400, 1e-9)
    out = []
    for label, date in MILESTONES:
        target = datetime.fromisoformat(date).replace(tzinfo=timezone.utc)
        days_left = (target - now).total_seconds() / 86400
        row = {"label": label, "date": date, "days_left": days_left,
               "departures": None, "observations": None}
        if reliable:
            extra = max(0.0, days_left)
            row["departures"] = dep["n"] + dep["n"] / days * extra
            row["observations"] = rows_n + rows_n / days * extra
        out.append(row)
    return out, reliable


# ------------------------------------------------------------------------- render

CSS = """
:root{--bg:#fbfbfa;--fg:#1a1a19;--muted:#6b6b68;--card:#fff;--line:#e5e4e1;
--ok:#2f855a;--warn:#b7791f;--bad:#c53030;--accent:#2b6cb0;}
@media(prefers-color-scheme:dark){:root{--bg:#161615;--fg:#eeeeec;--muted:#9a9a96;
--card:#1f1f1e;--line:#33322f;--ok:#68d391;--warn:#f6ad55;--bad:#fc8181;--accent:#63b3ed;}}
*{box-sizing:border-box}
body{margin:0;padding:32px 24px 64px;background:var(--bg);color:var(--fg);
font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
.wrap{max-width:1080px;margin:0 auto}
h1{font-size:26px;margin:0 0 4px} h2{font-size:17px;margin:36px 0 12px;font-weight:600}
.sub{color:var(--muted);margin:0 0 28px;font-size:14px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.kpi .v{font-size:25px;font-weight:640;letter-spacing:-.5px;font-variant-numeric:tabular-nums}
.kpi .l{color:var(--muted);font-size:12.5px;margin-top:3px}
.kpi .n{color:var(--muted);font-size:11.5px;margin-top:6px}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{text-align:right;padding:7px 10px;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}
th{color:var(--muted);font-weight:600;font-size:12.5px}
td{font-variant-numeric:tabular-nums}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
.ok{color:var(--ok)} .warn{color:var(--warn)} .bad{color:var(--bad)}
.note{color:var(--muted);font-size:13px;margin-top:10px}
.legend{display:flex;gap:14px;align-items:center;color:var(--muted);font-size:12.5px;margin-top:10px;flex-wrap:wrap}
.sw{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:5px;vertical-align:-1px}
code{background:var(--card);border:1px solid var(--line);border-radius:4px;padding:1px 5px;font-size:13px}
"""


def _fmt(n: float, dp: int = 0) -> str:
    return f"{n:,.{dp}f}"


def _dur(seconds: float) -> str:
    m = seconds / 60
    if m < 90:
        return f"{m:.0f} min"
    h = m / 60
    return f"{h:.1f} h" if h < 48 else f"{h/24:.1f} d"


def kpi(value: str, label: str, note: str = "", cls: str = "") -> str:
    note_html = f'<div class="n">{html.escape(note)}</div>' if note else ""
    return (f'<div class="kpi"><div class="v {cls}">{value}</div>'
            f'<div class="l">{html.escape(label)}</div>{note_html}</div>')


def heatmap_svg(days: list[str], grid: dict, first: datetime, last: datetime) -> str:
    """Hours as rows, days as columns. Blank = outside the logging window."""
    cw, ch, left, top = 15, 15, 34, 18
    w, h = left + len(days) * cw + 8, top + 24 * ch + 26
    parts = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
             f'xmlns="http://www.w3.org/2000/svg" font-size="9" '
             f'font-family="ui-monospace,monospace">']

    for hh in range(0, 24, 3):
        y = top + hh * ch + 11
        parts.append(f'<text x="0" y="{y}" fill="currentColor" opacity=".55">'
                     f'{hh:02d}:00</text>')

    for di, day in enumerate(days):
        x = left + di * cw
        for hh in range(24):
            cell = datetime.fromisoformat(f"{day}T{hh:02d}:00:00").replace(
                tzinfo=timezone.utc)
            y = top + hh * ch
            # Only hours inside the logging window can be "missing".
            if cell + timedelta(hours=1) <= first or cell > last:
                fill, op = "currentColor", ".06"
            else:
                n = grid.get((day, hh), 0)
                if n == 0:
                    fill, op = "var(--bad)", ".85"
                elif n >= EXPECTED_PER_HOUR:
                    fill, op = "var(--ok)", ".9"
                else:
                    fill, op = "var(--warn)", str(0.35 + 0.5 * n / EXPECTED_PER_HOUR)
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cw-2}" height="{ch-2}" rx="2" '
                f'fill="{fill}" opacity="{op}"><title>{day} {hh:02d}:00 UTC — '
                f'{grid.get((day, hh), 0)} of {EXPECTED_PER_HOUR} polls</title></rect>')
        if di == 0 or di == len(days) - 1 or (len(days) > 6 and di == len(days) // 2):
            parts.append(f'<text x="{x}" y="{top + 24*ch + 14}" fill="currentColor" '
                         f'opacity=".55">{day[5:]}</text>')

    parts.append("</svg>")
    return "".join(parts)


def bars_svg(pairs: list[tuple[str, float]], unit: str = "") -> str:
    if not pairs:
        return ""
    top = max(v for _, v in pairs) or 1
    bw, gap, h = 30, 8, 120
    w = len(pairs) * (bw + gap) + 40
    parts = [f'<svg width="{w}" height="{h+34}" viewBox="0 0 {w} {h+34}" '
             f'xmlns="http://www.w3.org/2000/svg" font-size="10" '
             f'font-family="ui-monospace,monospace">']
    for i, (label, v) in enumerate(pairs):
        bh = max(2, v / top * h)
        x, y = 20 + i * (bw + gap), h - bh + 12
        parts.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="3" '
                     f'fill="var(--accent)" opacity=".85"><title>{html.escape(label)}: '
                     f'{v:,.0f}{unit}</title></rect>')
        parts.append(f'<text x="{x+bw/2}" y="{y-4}" text-anchor="middle" '
                     f'fill="currentColor" opacity=".7">{v:,.0f}</text>')
        parts.append(f'<text x="{x+bw/2}" y="{h+26}" text-anchor="middle" '
                     f'fill="currentColor" opacity=".55">{html.escape(label)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def hist_svg(delays: list[int]) -> str:
    if not delays:
        return "<p class='note'>no realtime delays recorded yet</p>"
    edges = [-120, 0, 60, 120, 180, 300, 600, 900, 10**9]
    labels = ["früh", "0–1", "1–2", "2–3", "3–5", "5–10", "10–15", "15+"]
    counts = [0] * (len(edges) - 1)
    for d in delays:
        for i in range(len(edges) - 1):
            if edges[i] <= d < edges[i + 1]:
                counts[i] += 1
                break
    total = sum(counts) or 1
    rows = []
    for label, c in zip(labels, counts):
        pct = c / total
        colour = "var(--ok)" if label in ("früh", "0–1", "1–2") else (
            "var(--warn)" if label in ("2–3", "3–5") else "var(--bad)")
        rows.append(
            f'<tr><td>{label} min</td><td>{c:,}</td><td>{pct:.1%}</td>'
            f'<td style="width:55%"><div style="background:{colour};opacity:.8;'
            f'height:10px;border-radius:3px;width:{max(pct*100,0.6):.2f}%"></div></td></tr>')
    return ('<table><tr><th>Verspätung</th><th>n</th><th>Anteil</th><th></th></tr>'
            + "".join(rows) + "</table>")


def render(rows: list[dict], cycles: list[datetime], cov: dict, dep: dict,
           runs: list[dict] | None, obs_dir: Path = OBS_DIR,
           metas: list[dict] | None = None) -> str:
    now = datetime.now(timezone.utc)
    age_min = (now - cov["last"]).total_seconds() / 60
    fresh_cls = "ok" if age_min <= 20 else ("warn" if age_min <= 60 else "bad")
    comp = cov["completeness"]
    comp_cls = "ok" if comp >= 0.97 else ("warn" if comp >= 0.9 else "bad")

    days, grid = hourly_grid(cycles)
    per_day = Counter(c.strftime("%Y-%m-%d") for c in cycles)
    obs_day = Counter()
    for r in rows:
        t = _parse(r.get("observed_at", ""))
        if t:
            obs_day[t.strftime("%Y-%m-%d")] += 1

    delays = dep["delays"]
    median = delays[len(delays) // 2] / 60 if delays else 0.0
    gz_bytes = sum(p.stat().st_size for p in obs_dir.rglob("*.ndjson.gz"))

    k = [
        kpi(_fmt(len(rows)), "Beobachtungen", "eine Zeile pro Abfahrt pro Poll"),
        kpi(_fmt(dep["n"]), "Abfahrten", "dedupliziert je Fahrt"),
        kpi(_fmt(cov["actual"]), "Poll-Zyklen", f"erwartet {_fmt(cov['expected'])}"),
        kpi(f"{comp:.1%}", "Abdeckung", f"{cov['missed']} Slots fehlen", comp_cls),
        kpi(_dur((cov['last'] - cov['first']).total_seconds()), "Messdauer",
            f"seit {cov['first']:%Y-%m-%d %H:%M} UTC"),
        kpi(f"{age_min:.0f} min", "letzter Poll",
            "frisch" if age_min <= 20 else "ÜBERFÄLLIG — Logger prüfen", fresh_cls),
        kpi(f"{median:.1f} min", "Median-Verspätung",
            f"{dep['stops']} Halte, {dep['lines']} Linien"),
        kpi(_fmt(sum(1 for m in (metas or []) if m.get("n_failed"))),
            "Polls mit Stop-Fehlern",
            "API-Ausfall, nicht GitHub" if any(m.get("n_failed") for m in (metas or []))
            else "alle Stops erreichbar",
            "warn" if any(m.get("n_failed") for m in (metas or [])) else "ok"),
        kpi(f"{gz_bytes/1024:,.0f} kB", "Datenvolumen",
            f"{gz_bytes/max(cov['span_s'],1)*86400/1024:,.0f} kB/Tag hochgerechnet"
            if cov["span_s"] >= PROJECTION_MIN_SPAN_S
            else "Tagesrate ab einem vollen Messtag"),
    ]

    # coverage
    gap_rows = "".join(
        f"<tr><td>{a:%Y-%m-%d %H:%M}</td><td>{b:%H:%M}</td><td>{_dur(d)}</td>"
        f"<td>{missed}</td></tr>"
        for a, b, d, missed in cov["gaps"][:10]
    ) or '<tr><td colspan="4" class="ok">keine Lücken</td></tr>'

    prod_rows = "".join(
        f"<tr><td>{html.escape(p)}</td><td>{s['n']:,}</td><td>{s['median']:.1f}</td>"
        f"<td>{s['mean']:.1f}</td><td>{s['p95']:.1f}</td><td>{s['share_late']:.1%}</td></tr>"
        for p, s in dep["delay_by_product"].items()
    )

    # github
    if runs is None:
        gh_html = ('<p class="note">GitHub-Daten nicht abrufbar '
                   '(<code>gh</code> nicht installiert oder nicht angemeldet).</p>')
    elif not runs:
        gh_html = '<p class="note">noch keine Workflow-Läufe.</p>'
    else:
        ev = Counter(r["event"] for r in runs)
        con = Counter(r["conclusion"] or r["status"] for r in runs)
        sched = ev.get("schedule", 0)
        gh_html = (
            '<div class="kpis">'
            + kpi(_fmt(len(runs)), "Läufe (letzte 100)")
            + kpi(_fmt(sched), "davon per Zeitplan",
                  "Cron läuft" if sched else "noch kein Cron-Lauf",
                  "ok" if sched else "warn")
            + kpi(_fmt(con.get("success", 0)), "erfolgreich", cls="ok")
            + kpi(_fmt(con.get("failure", 0)), "fehlgeschlagen",
                  cls="bad" if con.get("failure") else "")
            + "</div>")

    bad = [m for m in (metas or []) if m.get("n_failed")]
    if not metas:
        fail_html = ('<p class="note">Keine Status-Dateien vorhanden — diese Daten '
                     'stammen aus der Zeit vor der Einführung des Poll-Protokolls '
                     '(2026-08-10). Für sie lässt sich Ausfall nicht von Lücke '
                     'unterscheiden.</p>')
    elif not bad:
        fail_html = (f'<p class="note ok">Alle {len(metas):,} protokollierten Polls '
                     f'haben jeden Stop erreicht.</p>')
    else:
        fail_rows = "".join(
            f"<tr><td>{html.escape(m['cycle_at'][:19])}</td>"
            f"<td>{m['n_failed']}</td><td>{m.get('n_rows', 0):,}</td>"
            f"<td style='text-align:left'>"
            + html.escape(", ".join(
                f"{s['stop_id']}: {s.get('detail', s['status'])}"
                for s in m["stops"] if s["status"] != "ok")) + "</td></tr>"
            for m in bad[-15:]
        )
        fail_html = (
            f'<p class="note"><strong>{len(bad):,} von {len(metas):,} Polls</strong> '
            f'hatten mindestens einen nicht erreichbaren Stop.</p>'
            '<table><tr><th>Zeitpunkt (UTC)</th><th>Stops betroffen</th>'
            '<th>Zeilen</th><th>Ursache</th></tr>' + fail_rows + "</table>")

    proj, proj_ok = project(cov, dep, len(rows))
    proj_rows = "".join(
        f"<tr><td>{html.escape(m['label'])}</td><td>{m['date']}</td>"
        f"<td>{m['days_left']:.0f}</td>"
        f"<td>{_fmt(m['departures']) if proj_ok else '—'}</td>"
        f"<td>{_fmt(m['observations']) if proj_ok else '—'}</td></tr>"
        for m in proj
    )
    proj_note = (
        f"Lineare Fortschreibung der bisherigen Rate — sie setzt voraus, dass der "
        f"Logger durchläuft. Bei {comp:.0%} Abdeckung ist das eine Obergrenze, "
        f"keine Zusage."
        if proj_ok else
        "<strong>Noch keine Hochrechnung.</strong> Dafür braucht es mindestens einen "
        "vollständigen Messtag. Da sich die 30-Minuten-Vorschau aufeinanderfolgender "
        "Polls überlappt, stabilisiert sich der Anteil <em>neuer</em> Abfahrten pro "
        "Poll erst über einen ganzen Tag mit Nachtflaute und Berufsverkehr. Eine "
        "Zahl aus wenigen Polls wäre frei erfunden."
    )

    day_bars = bars_svg([(d[5:], obs_day.get(d, 0)) for d in days[-14:]])
    poll_bars = bars_svg([(d[5:], per_day.get(d, 0)) for d in days[-14:]])

    return f"""<!doctype html><html lang="de"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>jufo-transit — Datenerhebung</title><style>{CSS}</style></head><body><div class="wrap">
<h1>Datenerhebung — jufo-transit</h1>
<p class="sub">Stand {now:%Y-%m-%d %H:%M} UTC · Quelle: <code>data/observations/</code>
im Repo <code>znel2002/jufo-transit</code></p>

<div class="kpis">{"".join(k)}</div>

<h2>Abdeckung — welche 15-Minuten-Slots wurden tatsächlich gemessen?</h2>
<div class="card"><div class="scroll">{heatmap_svg(days, grid, cov['first'], cov['last'])}</div>
<div class="legend">
<span><span class="sw" style="background:var(--ok)"></span>4/4 Polls</span>
<span><span class="sw" style="background:var(--warn)"></span>1–3 Polls</span>
<span><span class="sw" style="background:var(--bad)"></span>0 Polls — Lücke</span>
<span><span class="sw" style="background:currentColor;opacity:.15"></span>außerhalb des Messzeitraums</span>
</div>
<p class="note">Zeilen = Stunde (UTC), Spalten = Tag. GitHub Actions garantiert keine
pünktliche Ausführung; jede rote Zelle ist eine Lücke, die sich nicht nachträglich
füllen lässt. Diese Zahlen gehören in das Kapitel Fehlerquellen.</p></div>

<h2>Längste Lücken</h2>
<div class="card"><table>
<tr><th>von</th><th>bis</th><th>Dauer</th><th>verpasste Slots</th></tr>{gap_rows}</table>
<p class="note">Längste Lücke: {_dur(cov['longest_gap_s'])} ·
{cov['missed']} von {_fmt(cov['expected'])} erwarteten Polls fehlen.</p></div>

<h2>Fehlgeschlagene Polls (Stop-Ebene)</h2>
<div class="card">{fail_html}
<p class="note">Ein Poll, der lief und an dem die API scheiterte, ist etwas anderes als
ein Slot, den GitHub nie ausgelöst hat. Ohne diese Unterscheidung würde ein
API-Ausfall als Planungslücke gezählt und im Kapitel Fehlerquellen der falschen
Ursache zugeschrieben.</p></div>

<h2>Volumen der letzten 14 Tage</h2>
<div class="card"><div class="scroll">{day_bars}</div>
<p class="note">Beobachtungen pro Tag</p><div class="scroll">{poll_bars}</div>
<p class="note">Poll-Zyklen pro Tag (Soll: {24*EXPECTED_PER_HOUR})</p></div>

<h2>Verspätung nach Verkehrsmittel</h2>
<div class="card"><table>
<tr><th>Produkt</th><th>n</th><th>Median (min)</th><th>Mittel (min)</th>
<th>p95 (min)</th><th>&gt; 3 min</th></tr>{prod_rows}</table>
<p class="note">Median und Mittelwert weichen stark ab — die Verteilung ist rechtsschief.
Deshalb wird durchgehend MAE <em>und</em> RMSE berichtet.</p></div>

<h2>Verteilung der Verspätungen</h2>
<div class="card">{hist_svg(delays)}
<p class="note">{_fmt(dep['no_realtime'])} Abfahrten ohne Echtzeitwert,
{_fmt(dep['cancelled'])} ausgefallen.</p></div>

<h2>GitHub Actions</h2>
<div class="card">{gh_html}</div>

<h2>Hochrechnung auf die Wettbewerbstermine</h2>
<div class="card"><table>
<tr><th>Meilenstein</th><th>Datum</th><th>Tage</th><th>Abfahrten</th><th>Beobachtungen</th></tr>
{proj_rows}</table>
<p class="note">{proj_note}</p></div>

</div></body></html>"""


# --------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--obs-dir", default=str(OBS_DIR),
                    help="observations directory (default: data/observations)")
    ap.add_argument("--no-gh", action="store_true", help="skip the GitHub API query")
    ap.add_argument("--open", action="store_true", help="open the HTML when done")
    args = ap.parse_args()

    rows = load_rows(Path(args.obs_dir))
    metas = load_meta(Path(args.obs_dir))
    cycles = find_cycles(rows, metas)
    cov = coverage(cycles)
    dep = departures(rows)
    runs = None if args.no_gh else gh_runs()

    age = (datetime.now(timezone.utc) - cov["last"]).total_seconds() / 60
    print(f"\n{'COLLECTION SUMMARY':-^58}")
    print(f"  observations      {len(rows):,}")
    print(f"  departures        {dep['n']:,}  ({dep['cancelled']:,} cancelled)")
    print(f"  poll cycles       {cov['actual']:,} of {cov['expected']:,} expected")
    print(f"  coverage          {cov['completeness']:.1%}"
          f"   ({cov['missed']} slots missed)")
    print(f"  span              {cov['first']:%Y-%m-%d %H:%M} -> "
          f"{cov['last']:%Y-%m-%d %H:%M} UTC")
    print(f"  last poll         {age:.0f} min ago"
          + ("" if age <= 20 else "   <-- OVERDUE, check the logger"))
    if cov["gaps"]:
        a, b, d, missed = cov["gaps"][0]
        print(f"  longest gap       {_dur(d)} ({missed} slots) at {a:%Y-%m-%d %H:%M}")
    bad = [m for m in metas if m.get("n_failed")]
    if metas:
        print(f"  failed polls      {len(bad)} of {len(metas)} logged"
              + ("   <-- API outage, not a scheduling gap" if bad else ""))
    if runs is not None:
        sched = sum(1 for r in runs if r["event"] == "schedule")
        print(f"  gh runs           {len(runs)} total, {sched} scheduled")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(rows, cycles, cov, dep, runs, Path(args.obs_dir), metas),
                   encoding="utf-8")
    print(f"\nwrote {out}  ({out.stat().st_size/1024:.0f} kB)")
    if args.open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
