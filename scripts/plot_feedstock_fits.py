"""
Exploratory feedstock-restricted fits.

Only feedstock classes with more than 10 treatments in that panel are
drawn. Each class gets its own OLS line and R2. This is a descriptive
check, not a clustered meta-regression.
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
FIGDIR = ROOT / "figures"
SCRIPTS = Path(__file__).resolve().parent
OUT = FIGDIR / "feedstock_restricted_fits"

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from plot_application_conditions import load_meta as load_app_meta
from plot_application_conditions import treatment_points as app_treatments
from plot_feedstock_composition import (
    CATEGORY_ORDER,
    FEEDSTOCK_COLORS,
    FEEDSTOCK_MARKERS,
)
from plot_plant_metal_reduction import load_effects as load_plant
from plot_property_performance import load_treatments as load_properties
from plot_soil_bioavailability import load_effects as load_soil
from plot_temperature_reduction import load_meta as load_temp_meta
from plot_temperature_reduction import treatment_points as temp_treatments

LABEL_COLOR = "#1A1A1A"
MIN_N = 11

SHORT_NAME = {
    "Cereal field residue": "Cereal field",
    "Cereal processing residue": "Cereal processing",
    "Fruit and nut processing residue": "Fruit/nut",
    "Oilseed processing residue": "Oilseed",
}


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


def r_squared(x: np.ndarray, y: np.ndarray) -> float:
    pred = np.polyval(np.polyfit(x, y, 1), x)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def qualifying(df: pd.DataFrame, xcol: str) -> list[str]:
    out = []
    for cat in CATEGORY_ORDER:
        n = int(df.loc[df["feedstock_group"] == cat, xcol].notna().sum())
        if n >= MIN_N:
            out.append(cat)
    return out


def draw_panel(
    ax,
    df: pd.DataFrame,
    xcol: str,
    title: str,
    xlabel: str,
    xlim: tuple[float, float],
) -> None:
    cats = qualifying(df, xcol)
    ax.axhline(0, color="#888888", linewidth=0.8, zorder=1)
    handles = []
    for cat in cats:
        sub = df.loc[
            (df["feedstock_group"] == cat) & df[xcol].notna(),
            [xcol, "pct_reduction"],
        ].dropna()
        x = sub[xcol].to_numpy(dtype=float)
        y = sub["pct_reduction"].to_numpy(dtype=float)
        color = FEEDSTOCK_COLORS[cat]
        ax.scatter(
            x,
            y,
            s=36,
            marker=FEEDSTOCK_MARKERS[cat],
            color=color,
            edgecolors="#1A1A1A",
            linewidths=0.35,
            alpha=0.85,
            zorder=3,
        )
        slope, intercept = np.polyfit(x, y, 1)[0], np.polyfit(x, y, 1)[1]
        xs = np.linspace(float(x.min()), float(x.max()), 40)
        ax.plot(xs, slope * xs + intercept, color=color, linewidth=1.4, zorder=4)
        r2 = r_squared(x, y)
        handles.append(
            Line2D(
                [0],
                [0],
                marker=FEEDSTOCK_MARKERS[cat],
                color=color,
                markerfacecolor=color,
                markeredgecolor="#1A1A1A",
                markeredgewidth=0.35,
                markersize=7.5,
                linewidth=1.4,
                label=f"{SHORT_NAME.get(cat, cat)}  n={len(sub)}  R2={r2:.2f}",
            )
        )
    ax.set_xlim(*xlim)
    ax.set_ylim(-8, 108)
    ax.set_yticks(range(0, 101, 20))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(10))
    ax.tick_params(axis="both", length=3.5, width=0.6, labelsize=8.5, colors="#333333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#E4E4E4", linewidth=0.7)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Reduction (%)", fontsize=10)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=LABEL_COLOR, pad=8)
    if handles:
        ax.legend(
            handles=handles,
            loc="upper left",
            frameon=True,
            fancybox=False,
            edgecolor="#DDDDDD",
            facecolor="white",
            framealpha=0.92,
            fontsize=7.2,
            handlelength=1.5,
            handletextpad=0.40,
            labelspacing=0.28,
            borderaxespad=0.25,
            borderpad=0.35,
        )


def assemble() -> list[tuple[pd.DataFrame, str, str, str, tuple[float, float]]]:
    plant = load_plant()
    soil = load_soil()
    temp_meta = load_temp_meta()
    plant_temp = temp_treatments(plant, temp_meta)
    soil_temp = temp_treatments(soil, temp_meta)
    props = load_properties()
    app = app_treatments(plant, load_app_meta())
    return [
        (
            plant_temp,
            "temp_c",
            "A   Pyrolysis temperature, plant uptake",
            "Pyrolysis temperature (°C)",
            (275.0, 675.0),
        ),
        (
            soil_temp,
            "temp_c",
            "B   Pyrolysis temperature, soil bioavailability",
            "Pyrolysis temperature (°C)",
            (275.0, 675.0),
        ),
        (
            props,
            "ph",
            "C   Biochar pH",
            "Biochar pH",
            (5.0, 11.4),
        ),
        (
            props,
            "ssa",
            "D   Surface area",
            "Surface area (m2/g)",
            (0.0, 260.0),
        ),
        (
            app,
            "rate_ww",
            "E   Application rate",
            "Application rate (% w/w)",
            (-0.2, 10.8),
        ),
        (
            app,
            "soil_ph",
            "F   Initial soil pH",
            "Initial soil pH",
            (3.6, 9.2),
        ),
    ]


def plot_figure(panels: list[tuple[pd.DataFrame, str, str, str, tuple[float, float]]]) -> None:
    fig = plt.figure(figsize=(13.8, 9.6))
    gs = GridSpec(
        2,
        3,
        wspace=0.28,
        hspace=0.42,
        left=0.06,
        right=0.99,
        top=0.94,
        bottom=0.08,
        figure=fig,
    )
    for i, (df, xcol, title, xlabel, xlim) in enumerate(panels):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        draw_panel(ax, df, xcol, title, xlabel, xlim)
    fig.savefig(OUT.with_suffix(".pdf"))
    fig.savefig(OUT.with_suffix(".png"))
    plt.close(fig)
    print(f"wrote {OUT.with_suffix('.pdf')}")
    print(f"wrote {OUT.with_suffix('.png')}")


def main() -> None:
    style()
    panels = assemble()
    FIGDIR.mkdir(exist_ok=True)
    for df, xcol, title, _xlabel, _xlim in panels:
        cats = qualifying(df, xcol)
        print(title)
        for cat in cats:
            n = int(df.loc[df["feedstock_group"] == cat, xcol].notna().sum())
            print(f"  {n:3}  {cat}")
    plot_figure(panels)


if __name__ == "__main__":
    main()
