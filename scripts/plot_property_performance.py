"""
Property-performance relationship matrix.

Four scatter plots, one point per treatment:
  A  Biochar pH vs edible-part PTE uptake reduction
  B  Surface area vs uptake reduction
  C  Biochar CEC vs uptake reduction
  D  Ash content vs uptake reduction

Y is the treatment-level median percent reduction from the plant-metal
parser. Marker color and shape are the standard feedstock class.
CEC is restricted to values reported for the biochar.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "ANALYSIS SET.csv"
FIGDIR = ROOT / "figures"
SCRIPTS = Path(__file__).resolve().parent
OUT = FIGDIR / "property_performance_matrix"

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from plot_feedstock_composition import (
    CATEGORY_ORDER,
    FEEDSTOCK_COLORS,
    FEEDSTOCK_MARKERS,
)
from plot_plant_metal_reduction import load_effects as load_plant

LABEL_COLOR = "#1A1A1A"

PANELS = [
    ("ph", "A   Biochar pH", "Biochar pH", (5.0, 11.4)),
    ("ssa", "B   Surface area", "Surface area (m2/g)", (0.0, 260.0)),
    ("cec", "C   Biochar CEC", "CEC (cmolc/kg)", (0.0, 155.0)),
    ("ash", "D   Ash content", "Ash content (%)", (0.0, 72.0)),
]


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


def load_treatments() -> pd.DataFrame:
    df = pd.read_csv(DATA, dtype=str, keep_default_na=False)
    meta = pd.DataFrame(
        {
            "extraction_id": df["extraction_id"],
            "ph": pd.to_numeric(df["biochar_ph"], errors="coerce"),
            "ssa": pd.to_numeric(df["surface_area_m2_g"], errors="coerce"),
            "ash": pd.to_numeric(df["ash_content_pct"], errors="coerce"),
        }
    )
    cec = pd.to_numeric(df["cec_cmolc_kg"], errors="coerce")
    meta["cec"] = cec.where(df["cec_reported_for"] == "biochar")

    plant = load_plant().groupby("extraction_id", as_index=False).agg(
        pct_reduction=("pct_reduction", "median"),
        feedstock_group=("feedstock_group", "first"),
        study_id=("study_id", "first"),
    )
    return plant.merge(meta, on="extraction_id", how="left")


def draw_panel(ax, pts: pd.DataFrame, column: str, title: str, xlabel: str, xlim: tuple[float, float]) -> None:
    sub = pts.dropna(subset=[column, "pct_reduction"]).copy()
    ax.axhline(0, color="#888888", linewidth=0.8, zorder=1)
    for _, rec in sub.iterrows():
        grp = rec["feedstock_group"]
        ax.scatter(
            rec[column],
            rec["pct_reduction"],
            s=42,
            marker=FEEDSTOCK_MARKERS.get(grp, "o"),
            color=FEEDSTOCK_COLORS.get(grp, "#333333"),
            edgecolors="#1A1A1A",
            linewidths=0.35,
            alpha=0.85,
            zorder=3,
        )
    ax.set_xlim(*xlim)
    ax.set_ylim(-5, 108)
    ax.set_yticks(range(0, 101, 20))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(10))
    ax.tick_params(axis="both", length=3.5, width=0.6, labelsize=8.5, colors="#333333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#E4E4E4", linewidth=0.7)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Reduction in edible-part PTE concentration (%)", fontsize=10)
    ax.set_title(
        f"{title}  (n = {len(sub)} treatments)",
        loc="left",
        fontsize=11,
        fontweight="bold",
        color=LABEL_COLOR,
        pad=8,
    )


def feedstock_handles(pts: pd.DataFrame) -> list[Line2D]:
    used = set(pts["feedstock_group"])
    return [
        Line2D(
            [0],
            [0],
            marker=FEEDSTOCK_MARKERS[g],
            color="none",
            markerfacecolor=FEEDSTOCK_COLORS[g],
            markeredgecolor="#1A1A1A",
            markeredgewidth=0.35,
            markersize=8.0,
            label=g,
        )
        for g in CATEGORY_ORDER
        if g in used
    ]


def plot_matrix(pts: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(13.4, 10.6))
    gs = GridSpec(
        3,
        2,
        height_ratios=[1.0, 1.0, 0.38],
        hspace=0.36,
        wspace=0.24,
        left=0.08,
        right=0.99,
        top=0.94,
        bottom=0.05,
        figure=fig,
    )
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]
    for ax, (column, title, xlabel, xlim) in zip(axes, PANELS):
        draw_panel(ax, pts, column, title, xlabel, xlim)

    ax_leg = fig.add_subplot(gs[2, :])
    ax_leg.set_axis_off()
    present = pts.dropna(subset=["ph", "ssa", "cec", "ash"], how="all")
    ax_leg.legend(
        handles=feedstock_handles(present),
        loc="upper left",
        bbox_to_anchor=(0.00, 0.95),
        ncol=3,
        frameon=False,
        fontsize=8.5,
        handletextpad=0.40,
        columnspacing=1.15,
        labelspacing=0.55,
        title="Feedstock category",
        title_fontsize=10,
    )

    fig.savefig(OUT.with_suffix(".pdf"))
    fig.savefig(OUT.with_suffix(".png"))
    plt.close(fig)
    print(f"wrote {OUT.with_suffix('.pdf')}")
    print(f"wrote {OUT.with_suffix('.png')}")


def main() -> None:
    style()
    pts = load_treatments()
    FIGDIR.mkdir(exist_ok=True)
    for column, title, _xlabel, _xlim in PANELS:
        n = int(pts[column].notna().sum())
        print(f"{title}: n={n}")
    plot_matrix(pts)


if __name__ == "__main__":
    main()
