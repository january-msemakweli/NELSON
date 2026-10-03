"""
Pyrolysis temperature versus performance.

Two-panel bubble plot, one point per treatment:
  A  Edible-part plant-metal reduction
  B  Soil-bioavailability reduction

X is pyrolysis temperature. Y is the treatment-level median percent
reduction. Marker color and shape are the standard feedstock class.
Bubble size is application rate as % w/w.
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
OUT = FIGDIR / "temperature_vs_reduction"

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from plot_feedstock_composition import (
    CATEGORY_ORDER,
    FEEDSTOCK_COLORS,
    FEEDSTOCK_MARKERS,
)
from plot_plant_metal_reduction import load_effects as load_plant
from plot_soil_bioavailability import load_effects as load_soil

LABEL_COLOR = "#1A1A1A"
DEFAULT_RATE = 2.5
SIZE_RATES = (1.0, 2.5, 5.0, 10.0)


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


def bubble_size(rate: float) -> float:
    return 50.0 * np.sqrt(max(rate, 0.5) / DEFAULT_RATE)


def load_meta() -> pd.DataFrame:
    df = pd.read_csv(DATA, dtype=str, keep_default_na=False)
    return pd.DataFrame(
        {
            "extraction_id": df["extraction_id"],
            "temp_c": pd.to_numeric(df["pyrolysis_temp_c"], errors="coerce"),
            "rate_ww": pd.to_numeric(df["application_rate_pct_ww"], errors="coerce"),
        }
    )


def treatment_points(effects: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    mid = effects.groupby("extraction_id", as_index=False).agg(
        pct_reduction=("pct_reduction", "median"),
        feedstock_group=("feedstock_group", "first"),
        study_id=("study_id", "first"),
        n_pte=("metal", "nunique"),
    )
    out = mid.merge(meta, on="extraction_id", how="left")
    out = out.dropna(subset=["temp_c", "pct_reduction"]).copy()
    out["rate_plot"] = out["rate_ww"].fillna(DEFAULT_RATE)
    out["rate_missing"] = out["rate_ww"].isna()
    return out


def draw_panel(ax, pts: pd.DataFrame, title: str, xlabel: bool) -> None:
    rng = np.random.default_rng(7)
    drawn = pts.sort_values("rate_plot", ascending=False).reset_index(drop=True)
    x = drawn["temp_c"].to_numpy() + rng.normal(0, 6.0, size=len(drawn))
    y = drawn["pct_reduction"].to_numpy()

    ax.axhline(0, color="#888888", linewidth=0.8, zorder=1)
    for i, rec in drawn.iterrows():
        grp = rec["feedstock_group"]
        ax.scatter(
            x[i],
            y[i],
            s=bubble_size(float(rec["rate_plot"])),
            marker=FEEDSTOCK_MARKERS.get(grp, "o"),
            color=FEEDSTOCK_COLORS.get(grp, "#333333"),
            edgecolors="#1A1A1A",
            linewidths=0.35,
            alpha=0.82,
            zorder=3,
        )

    ax.set_xlim(275, 675)
    ax.set_ylim(-35, 108)
    ax.set_xticks(range(300, 651, 50))
    ax.set_yticks(range(-20, 101, 20))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(10))
    ax.tick_params(axis="both", length=3.5, width=0.6, labelsize=8.5, colors="#333333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#E4E4E4", linewidth=0.7)
    if xlabel:
        ax.set_xlabel("Pyrolysis temperature (°C)", fontsize=11)
    ax.set_ylabel("Reduction (%)", fontsize=11)
    ax.set_title(
        f"{title}  (n = {len(pts)} treatments)",
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


def size_handles() -> list[Line2D]:
    handles = []
    for rate in SIZE_RATES:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#888888",
                markeredgecolor="#1A1A1A",
                markeredgewidth=0.35,
                markersize=np.sqrt(bubble_size(rate)),
                label=f"{rate:g}% w/w",
            )
        )
    return handles


def plot_figure(plant: pd.DataFrame, soil: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(13.6, 8.0))
    gs = GridSpec(
        2,
        2,
        height_ratios=[3.8, 1.15],
        hspace=0.22,
        wspace=0.22,
        left=0.07,
        right=0.99,
        top=0.93,
        bottom=0.06,
        figure=fig,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    draw_panel(ax_a, plant, "A   Plant-uptake reduction", xlabel=True)
    draw_panel(ax_b, soil, "B   Soil-bioavailability reduction", xlabel=True)
    ax_a.xaxis.labelpad = 6
    ax_b.xaxis.labelpad = 6

    ax_leg = fig.add_subplot(gs[1, :])
    ax_leg.set_axis_off()
    feed = feedstock_handles(pd.concat([plant, soil], ignore_index=True))
    sizes = size_handles()
    leg1 = ax_leg.legend(
        handles=feed,
        loc="upper left",
        bbox_to_anchor=(0.00, 0.98),
        ncol=3,
        frameon=False,
        fontsize=8.5,
        handletextpad=0.40,
        columnspacing=1.15,
        labelspacing=0.55,
        title="Feedstock category",
        title_fontsize=10,
    )
    ax_leg.add_artist(leg1)
    ax_leg.legend(
        handles=sizes,
        loc="upper left",
        bbox_to_anchor=(0.74, 0.98),
        ncol=1,
        frameon=False,
        fontsize=8.5,
        handletextpad=0.55,
        labelspacing=0.85,
        title="Application rate",
        title_fontsize=10,
    )

    fig.savefig(OUT.with_suffix(".pdf"))
    fig.savefig(OUT.with_suffix(".png"))
    plt.close(fig)
    print(f"wrote {OUT.with_suffix('.pdf')}")
    print(f"wrote {OUT.with_suffix('.png')}")


def main() -> None:
    style()
    meta = load_meta()
    plant = treatment_points(load_plant(), meta)
    soil = treatment_points(load_soil(), meta)
    FIGDIR.mkdir(exist_ok=True)
    print(
        f"plant treatments={len(plant)} studies={plant.study_id.nunique()} "
        f"rate_missing={int(plant.rate_missing.sum())}"
    )
    print(
        f"soil treatments={len(soil)} studies={soil.study_id.nunique()} "
        f"rate_missing={int(soil.rate_missing.sum())}"
    )
    plot_figure(plant, soil)


if __name__ == "__main__":
    main()
