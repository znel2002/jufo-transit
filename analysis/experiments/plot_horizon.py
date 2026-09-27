"""Headline figure: what live information is worth, by forecast horizon.

    python analysis/experiments/plot_horizon.py

Reads horizon_sweep_pooled.csv / horizon_sweep_folds.csv and writes
horizon_sweep.png (+ .pdf for the printed Langfassung).

Design choices (checked against the data-viz method, and re-done after looking at
the first render, whose end-of-line labels collided and whose "both" line sat on
top of "own estimate"):
  * form: lines over an ordered x (minutes before departure); one y-axis per panel;
  * y starts at ZERO and the chance level (base rate) is drawn: for PR-AUC the
    honest floor is the base rate, not the bottom of the data, and a truncated axis
    would exaggerate the decline;
  * only the two quantities that vary with horizon get series colours -- the
    trip's own realtime estimate and network state (categorical slots 1 and 2,
    validated all-pairs). "Both" is omitted from the drawing on purpose: it lies
    within +0.016 PR-AUC of "own estimate" everywhere, i.e. invisible, and that
    near-zero increment is itself the finding, stated in the text instead;
  * horizon-independent references (lookup table, schedule-time model, chance) are
    neutral grey rules named in the legend, never series colours;
  * the two series are direct-labelled where they are far apart (x = 15 min), with
    distinct markers, so identity survives greyscale printing and CVD;
  * text stays in ink colours; grid is a recessive hairline;
  * whiskers span the rolling-origin folds, so fold-to-fold variation is visible.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES = [  # (model prefix, legend label, short direct label, colour, marker)
    ("A+own", "Fahrplan + eigene Echtzeitschätzung der Fahrt",
     "eigene Echtzeitschätzung", "#2a78d6", "o"),
    ("A+net", "Fahrplan + Netzzustand (andere Linien am Halt)",
     "Netzzustand", "#eb6834", "s"),
]
# Direct labels are kept SHORT on purpose: the first render used the full legend
# text, and its label box covered a stretch of the other series' line. The legend
# carries the full description; a label must never hide data.
REFS = [  # (model, label, dash)
    ("lookup", "Nachschlagetabelle (Linie × Stunde)", (0, (1.5, 1.5))),
    ("A", "nur Fahrplan (Gradient Boosting)", (0, (5, 2))),
]
TITLES = {180: "Verspätung ≥ 3 min", 600: "Verspätung ≥ 10 min"}
LABEL_AT = 15          # horizon at which series are far enough apart to label


def de(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def main() -> int:
    pooled = pd.read_csv(HERE / "horizon_sweep_pooled.csv")
    folds = pd.read_csv(HERE / "horizon_sweep_folds.csv")
    ths = sorted(pooled.threshold.unique())

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    })
    fig, axes = plt.subplots(1, len(ths), figsize=(4.4 * len(ths), 3.6),
                             facecolor=SURFACE, squeeze=False)

    for ax, th in zip(axes[0], ths):
        P = pooled[(pooled.threshold == th) & ~pooled.model.str.startswith("GAIN")]
        P = P.set_index("model")
        F = folds[folds.threshold == th]
        n, pos = int(P.loc["A", "n"]), int(P.loc["A", "pos"])
        base = pos / n

        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

        horizons = sorted(int(m.split("@")[1]) for m in P.index if m.startswith("A+net@"))

        ax.axhline(base, color=AXIS, linewidth=1.0, zorder=1)
        for key, _, dash in REFS:
            ax.axhline(P.loc[key, "pr"], color=MUTED, linewidth=1.2, linestyle=dash, zorder=1)

        ymax = 0
        for prefix, label, short, colour, marker in SERIES:
            ys, lo, hi = [], [], []
            for h in horizons:
                m = f"{prefix}@{h}"
                y = P.loc[m, "pr"]
                fv = F[F.model == m].pr
                ys.append(y)
                lo.append(max(0.0, y - fv.min()))
                hi.append(max(0.0, fv.max() - y))
                ymax = max(ymax, fv.max())
            ax.errorbar(horizons, ys, yerr=[lo, hi], color=colour, linewidth=2,
                        marker=marker, markersize=6, markeredgecolor=SURFACE,
                        markeredgewidth=1.2, elinewidth=0.8, capsize=0, zorder=3)
            i = horizons.index(LABEL_AT)
            ax.annotate(short, (horizons[i], ys[i]),
                        xytext=(6, 7), textcoords="offset points", fontsize=7.5,
                        color=INK, zorder=4,
                        bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none"))

        ax.set_ylim(0, ymax * 1.08)
        ax.set_xlim(min(horizons) - 3, max(horizons) + 3)
        ax.set_xticks(horizons)
        ax.set_xlabel("Vorhersage so viele Minuten vor der Abfahrt")
        ax.set_ylabel("PR-AUC (höher = besser)")
        ax.set_title(f"{TITLES.get(th, f'≥ {th // 60} min')}  ·  n = {de(n)}, "
                     f"davon {de(pos)} verspätet", fontsize=9, color=INK, loc="left")
        ax.text(max(horizons) + 2.5, base, f"Zufall ({base:.0%})".replace(".", ","),
                color=MUTED, fontsize=7, ha="right", va="bottom")

    handles = [mlines.Line2D([], [], color=c, marker=mk, linewidth=2, markersize=6,
                             markeredgecolor=SURFACE, label=lab)
               for _, lab, _s, c, mk in SERIES]
    handles += [mlines.Line2D([], [], color=MUTED, linewidth=1.2, linestyle=d, label=lab)
                for _, lab, d in REFS]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               fontsize=7.5, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"horizon_sweep.{ext}", dpi=200, facecolor=SURFACE,
                    bbox_inches="tight")
    print("wrote horizon_sweep.png / .pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
