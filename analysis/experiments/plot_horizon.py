"""Headline figure: what live information is worth, by forecast horizon.

    python analysis/experiments/plot_horizon.py

Reads horizon_sweep_pooled.csv / horizon_sweep_folds.csv and writes
horizon_sweep.png (+ .pdf for the printed Langfassung).

Design choices (from the data-viz method, checked rather than by taste):
  * form: a line over an ordered x (minutes before departure) -- one y-axis only;
  * the two horizon-independent references (lookup table, schedule-time model) are
    NOT given series colours: they are neutral dashed rules, so colour is spent only
    on the three models that actually vary with horizon;
  * those three take the first three categorical slots of the reference palette,
    which validate all-pairs (worst CVD dE 9.2, normal-vision 24.0);
  * aqua is below 3:1 contrast on the surface, which obligates visible labels, so
    every line is direct-labelled at its end AND has a distinct marker -- identity
    survives greyscale printing and colour-vision deficiency, never colour alone;
  * text stays in ink colours, never series colours; grid is a recessive hairline;
  * whiskers are the spread across rolling-origin folds, so the reader sees the
    uncertainty, not just a point.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES = [  # (model prefix, label, colour, marker) -- fixed order, never cycled
    ("A+net", "Fahrplan + Netzzustand", "#2a78d6", "o"),
    ("A+own", "Fahrplan + eigene Echtzeitschätzung", "#eb6834", "s"),
    ("A+own+net", "Fahrplan + beides", "#1baf7a", "^"),
]
TITLES = {180: "Verspätung ≥ 3 min", 600: "Verspätung ≥ 10 min"}


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
    fig, axes = plt.subplots(1, len(ths), figsize=(3.6 * len(ths) + 1.2, 3.4),
                             facecolor=SURFACE, squeeze=False)

    for ax, th in zip(axes[0], ths):
        P = pooled[pooled.threshold == th].set_index("model")
        F = folds[folds.threshold == th]
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

        horizons = sorted(int(m.split("@")[1]) for m in P.index if m.startswith("A+net@"))
        xmax = max(horizons)

        # neutral reference rules
        for key, label, ls in (("lookup", "Nachschlagetabelle (Linie × Stunde)", (0, (2, 2))),
                               ("A", "nur Fahrplan", (0, (5, 2)))):
            if key in P.index:
                y = P.loc[key, "pr"]
                ax.axhline(y, color=MUTED, linewidth=1.2, linestyle=ls, zorder=1)
                ax.text(xmax + 1.5, y, label, color=INK2, va="center", fontsize=7.5)

        ends = []
        for prefix, label, colour, marker in SERIES:
            ys, lo, hi = [], [], []
            for h in horizons:
                m = f"{prefix}@{h}"
                ys.append(P.loc[m, "pr"])
                fv = F[F.model == m].pr
                lo.append(P.loc[m, "pr"] - fv.min())
                hi.append(fv.max() - P.loc[m, "pr"])
            ax.errorbar(horizons, ys, yerr=[lo, hi], color=colour, linewidth=2,
                        marker=marker, markersize=6, markeredgecolor=SURFACE,
                        markeredgewidth=1.2, elinewidth=0.8, capsize=0, zorder=3,
                        label=label)
            ends.append((ys[-1], label))

        # direct labels at the right end, nudged apart so they never collide
        ends.sort()
        placed = []
        span = ax.get_ylim()[1] - ax.get_ylim()[0]
        for y, label in ends:
            y_adj = max([y] + [p + 0.06 * span for p in placed])
            placed.append(y_adj)
            ax.text(xmax + 1.5, y_adj, label, color=INK, va="center", fontsize=7.5)

        ax.set_xlim(min(horizons) - 2, xmax + 1)
        ax.set_xticks(horizons)
        ax.set_xlabel("Vorhersage so viele Minuten vor der Abfahrt")
        ax.set_ylabel("PR-AUC (höher = besser)")
        pos = int(P.loc["A", "pos"]) if "A" in P.index else 0
        n = int(P.loc["A", "n"]) if "A" in P.index else 0
        ax.set_title(f"{TITLES.get(th, f'≥ {th // 60} min')}   "
                     f"(n = {n:,}, davon {pos:,} verspätet)".replace(",", "."),
                     fontsize=9, color=INK, loc="left")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               fontsize=7.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 0.86, 1))
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"horizon_sweep.{ext}", dpi=200, facecolor=SURFACE,
                    bbox_inches="tight")
    print("wrote horizon_sweep.png / .pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
