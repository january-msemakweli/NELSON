"""
Boxplots of edible-part plant-uptake reduction by feedstock category.

X-axis is the standard feedstock class. Y-axis is the percent reduction
in edible-tissue PTE concentration. Values are the same extraction x PTE
estimates used in the plant-metal Cleveland plot.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "figures"
SCRIPTS = Path(__file__).resolve().parent
OUT = FIGDIR / "plant_uptake_reduction_by_feedstock"

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from plot_feedstock_composition import CATEGORY_ORDER, FEEDSTOCK_COLORS
from plot_plant_metal_reduction import load_effects

LABEL_COLOR = "#1A1A1A"


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.10,
        }
    )


def wrap_label(name: str, n: int, width: int = 18) -> str:
    wrapped = textwrap.wrap(
        name.replace("/", " / "),
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )
    wrapped.append(f"(n = {n})")
    return "\n".join(wrapped)


def category_series(effects: pd.DataFrame) -> tuple[list[str], list[np.ndarray]]:
    cats = []
    series = []
    for cat in CATEGORY_ORDER:
        vals = effects.loc[effects["feedstock_group"] == cat, "pct_reduction"].to_numpy(
            dtype=float
        )
        if len(vals) == 0:
            continue
        cats.append(cat)
        series.append(vals)
    return cats, series


def plot_boxes(effects: pd.DataFrame) -> None:
    cats, series = category_series(effects)
    n_points = int(sum(len(v) for v in series))
    n_treat = int(effects["extraction_id"].nunique())
    n_study = int(effects["study_id"].nunique())

    fig, ax = plt.subplots(figsize=(13.8, 7.2))
    positions = np.arange(1, len(cats) + 1)
    boxed = [vals if len(vals) else np.array([np.nan]) for vals in series]
    bp = ax.boxplot(
        boxed,
        positions=positions,
        tick_labels=[""] * len(cats),
        vert=True,
        patch_artist=True,
        widths=0.58,
        showfliers=False,
        medianprops={"color": "#1A1A1A", "linewidth": 1.25},
        whiskerprops={"color": "#333333", "linewidth": 0.7},
        capprops={"color": "#333333", "linewidth": 0.7},
        zorder=3,
    )
    for patch, cat in zip(bp["boxes"], cats):
        patch.set_facecolor(FEEDSTOCK_COLORS.get(cat, "#1B7F4E"))
        patch.set_edgecolor("#1A1A1A")
        patch.set_linewidth(0.7)
        patch.set_alpha(0.88)

    rng = np.random.default_rng(7)
    for i, (cat, vals) in enumerate(zip(cats, series), start=1):
        jitter = rng.normal(0, 0.07, size=len(vals))
        ax.scatter(
            np.full(len(vals), i) + jitter,
            vals,
            s=16,
            color=FEEDSTOCK_COLORS.get(cat, "#333333"),
            edgecolors="#1A1A1A",
            linewidths=0.25,
            alpha=0.72,
            zorder=4,
        )

    ax.axhline(0, color="#888888", linewidth=0.8, zorder=1)
    ax.set_ylim(-8, 105)
    ax.set_yticks(range(0, 101, 20))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(10))
    ax.set_xlim(0.4, len(cats) + 0.6)
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [wrap_label(cat, len(vals)) for cat, vals in zip(cats, series)],
        fontsize=8,
        color="#333333",
    )
    ax.tick_params(axis="x", length=0, pad=4)
    ax.tick_params(axis="y", length=3.5, width=0.6, labelsize=9, colors="#333333")
    ax.set_ylabel(
        "Reduction in edible-part PTE concentration (%)",
        fontsize=11,
    )
    ax.set_xlabel("Feedstock category", fontsize=11, labelpad=10)
    ax.set_title(
        "Plant-uptake reduction by feedstock category  "
        f"(n = {n_points} PTE estimates, {n_treat} treatments, {n_study} studies)",
        loc="left",
        fontsize=11,
        fontweight="bold",
        color=LABEL_COLOR,
        pad=10,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#E4E4E4", linewidth=0.7)
    fig.subplots_adjust(left=0.07, right=0.995, top=0.90, bottom=0.26)
    fig.savefig(OUT.with_suffix(".pdf"))
    fig.savefig(OUT.with_suffix(".png"))
    plt.close(fig)
    print(f"wrote {OUT.with_suffix('.pdf')}")
    print(f"wrote {OUT.with_suffix('.png')}")


def main() -> None:
    style()
    effects = load_effects()
    FIGDIR.mkdir(exist_ok=True)
    print(
        effects.groupby("feedstock_group")
        .agg(points=("pct_reduction", "size"), treatments=("extraction_id", "nunique"))
        .reindex(CATEGORY_ORDER)
        .to_string()
    )
    plot_boxes(effects)


if __name__ == "__main__":
    main()
