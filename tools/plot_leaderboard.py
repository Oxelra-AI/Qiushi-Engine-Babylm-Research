"""Draw the homepage comparison from the report's fixed leaderboard table.

Run with the report's Python dependencies. This script makes no network calls
and never changes the report's numerical data or PDFs.
"""
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
FRONTIER = "Qiushi-Engine-Frontier-Advancement"
GUIDED = "Qiushi-Engine-Principle-Guided-Frontier-Advancement"
MODELS = {FRONTIER, GUIDED}


def read_scores():
    with (ROOT / "results/leaderboard_comparison.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 10 or len({row["model"] for row in rows}) != 10:
        raise ValueError("The report comparison must have ten distinct models")
    if not MODELS.issubset({row["model"] for row in rows}):
        raise ValueError("Both representative model generations are required")
    for row in rows:
        row["score"] = float(row["Overall"])
        if not math.isfinite(row["score"]):
            raise ValueError("Every plotted score must be a finite recorded value")
    return sorted(rows, key=lambda row: (-row["score"], row["model"]))


def draw(rows):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "svg.fonttype": "none", "axes.unicode_minus": False})
    colors = {GUIDED: "#285c91", FRONTIER: "#7768ad"}
    scores = {row["model"]: row["score"] for row in rows}
    best_external = max(value for model, value in scores.items() if model not in MODELS)
    fig = plt.figure(figsize=(12, 8.8), facecolor="white")
    fig.text(.055, .947, "Qiushi Engine on BabyLM 2026", fontsize=21, weight="bold", color="#223040")
    fig.text(.055, .909, "Strict-Small · Two model generations from one autonomous research program", fontsize=12, color="#526173")
    cards = [("Principle-guided model", f"{scores[GUIDED]:.2f}", colors[GUIDED]),
             ("Frontier model", f"{scores[FRONTIER]:.2f}", colors[FRONTIER]),
             ("Principle-guided vs. best external", f"{scores[GUIDED]-best_external:+.2f}", "#223040")]
    for index, (label, value, color) in enumerate(cards):
        x = .055 + index * .304
        fig.add_artist(FancyBboxPatch((x, .768), .285, .105,
                                     boxstyle="round,pad=0.006,rounding_size=0.009",
                                     transform=fig.transFigure, linewidth=.7,
                                     edgecolor="#e2e8ef", facecolor="#f7f9fc"))
        fig.text(x+.016, .840, label, fontsize=10, color="#526173")
        fig.text(x+.016, .786, value, fontsize=26, weight="bold", color=color)

    ax = fig.add_axes([.43, .155, .46, .53])
    lo, hi = min(scores.values()), max(scores.values())
    margin = max((hi-lo)*.12, .12)
    ax.set_xlim(lo-margin, hi+margin)
    ax.set_ylim(9.65, -.65)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.grid(axis="x", color="#e5eaf0", linewidth=.8)
    ax.set_axisbelow(True)
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="x", length=0, labelsize=10, colors="#526173")
    ax.set_xlabel("Overall score", fontsize=11, labelpad=13, color="#526173")
    ax.axvline(best_external, color="#b5bfcb", linestyle=(0, (3, 3)), linewidth=1)
    ax.text(-.805, 1.065, "MODEL", transform=ax.transAxes,
            fontsize=10, color="#526173", weight="bold")
    ax.text(1.12, 1.065, "OVERALL", transform=ax.transAxes, ha="right",
            fontsize=10, color="#526173", weight="bold")

    for index, row in enumerate(rows):
        model, score = row["model"], row["score"]
        own = model in MODELS
        color = colors.get(model, "#98a5b5")
        name = row["display_label"].split(" / ")[-1]
        if model == GUIDED:
            name = "Qiushi Engine · Principle-guided"
        elif model == FRONTIER:
            name = "Qiushi Engine · Frontier"
        publisher = "Qiushi Engine" if own else row["display_label"].split(" / ")[0]
        if own:
            ax.axhspan(index-.40, index+.40, color=color, alpha=.055, linewidth=0)
        ax.scatter([score], [index], s=100 if own else 62, color=color,
                   edgecolors="white", linewidth=1, zorder=4)
        ax.text(-.875, index, str(index+1), transform=ax.get_yaxis_transform(),
                va="center", ha="right", color="#8b96a4", fontsize=11)
        ax.text(-.805, index-.09, name, transform=ax.get_yaxis_transform(),
                va="center", fontsize=11.5 if own else 10.8,
                weight="bold" if own else "normal", color=color if own else "#334155")
        ax.text(-.805, index+.23, publisher, transform=ax.get_yaxis_transform(),
                va="center", fontsize=8.8, color="#778596")
        ax.text(1.12, index, f"{score:.2f}", transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=13, color=color if own else "#334155",
                weight="bold" if own else "normal")

    fig.text(.055, .065, "Report snapshot · 8 September 2026 · Two representative Qiushi generations + eight leading external entries.",
             fontsize=9, color="#64748b")
    fig.text(.055, .035, "Ordered by Overall after excluding earlier Qiushi submissions. Dotted line: highest external score.",
             fontsize=9, color="#64748b")
    output = ROOT / "assets/leaderboard"
    output.mkdir(exist_ok=True)
    fig.savefig(output / "report-snapshot.svg", metadata={"Date": "2026-09-08"})
    fig.savefig(output / "report-snapshot.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    data = read_scores()
    draw(data)
    print("Homepage figure generated from the unchanged 8 September report table.")
