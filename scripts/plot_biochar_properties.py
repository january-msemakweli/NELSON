"""
Biochar physicochemical characteristics.

Four-panel box-and-jitter figure, extraction-level:
  A  Biochar pH by feedstock category
  B  Surface area by feedstock category
  C  Ash content by feedstock category
  D  Biochar CEC by feedstock category

Also writes standalone A-D files.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "ANALYSIS SET.csv"
FIGDIR = ROOT / "figures"
SCRIPTS = Path(__file__).resolve().parent

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from plot_feedstock_composition import CATEGORY_ORDER, SPECIFIC_TO_CATEGORY

OUT_COMBINED = FIGDIR / "biochar_physicochemical_properties"
OUT_A = FIGDIR / "biochar_physicochemical_A_ph"
OUT_B = FIGDIR / "biochar_physicochemical_B_surface_area"
OUT_C = FIGDIR / "biochar_physicochemical_C_ash"
OUT_D = FIGDIR / "biochar_physicochemical_D_cec"

LABEL_COLOR = "#1A1A1A"

GREEN = {
    "fill": "#1B7F4E",
    "edge": "#145C39",
    "median": "#0D3B24",
    "point": "#3A9A68",
}
ORANGE = {
    "fill": "#E07B39",
    "edge": "#B85F28",
    "median": "#8C4A1F",
    "point": "#F0A36A",
}

PANELS = [
    ("ph", "Biochar pH", "Biochar pH", "A", (2.5, 11.5), GREEN),
    ("ssa", "Surface area", "Surface area (m2/g)", "B", (0, 260), ORANGE),
    ("ash", "Ash content", "Ash content (%)", "C", (0, 72), GREEN),
    ("cec", "CEC", "CEC (cmolc/kg)", "D", (0, 160), ORANGE),
]


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 11,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
        }
    )


def add_panel_header(fig, ax, tag: str, title: str) -> None:
    pos = ax.get_position()
    fig.text(
        pos.x0,
        pos.y1 + 0.012,
        f"{tag}   {title}",
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=LABEL_COLOR,
    )


def hide_spines(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", length=3.5, width=0.6, labelsize=10.5, colors="#333333")
    ax.tick_params(axis="y", length=0, pad=5, labelsize=11, colors="#333333")
    ax.set_axisbelow(True)


def load_treatments() -> pd.DataFrame:
    df = pd.read_csv(DATA, dtype=str, keep_default_na=False)
    out = pd.DataFrame(
        {
            "extraction_id": df["extraction_id"],
            "feedstock_group": df["specific_feedstock"].map(
                lambda name: SPECIFIC_TO_CATEGORY.get(name.strip(), "Mixed residues")
            ),
            "ph": pd.to_numeric(df["biochar_ph"], errors="coerce"),
            "ssa": pd.to_numeric(df["surface_area_m2_g"], errors="coerce"),
            "ash": pd.to_numeric(df["ash_content_pct"], errors="coerce"),
        }
    )
    cec = pd.to_numeric(df["cec_cmolc_kg"], errors="coerce")
    out["cec"] = cec.where(df["cec_reported_for"] == "biochar")
    return out


def category_series(df: pd.DataFrame, column: str) -> tuple[list[np.ndarray], list[str], int]:
    series = []
    labels = []
    total = 0
    for cat in reversed(CATEGORY_ORDER):
        vals = df.loc[df["feedstock_group"] == cat, column].dropna().to_numpy()
        series.append(vals)
        labels.append(f"{cat}  (n = {len(vals)})")
        total += len(vals)
    return series, labels, total


def draw_property_box(
    ax,
    df: pd.DataFrame,
    column: str,
    xlabel: str,
    heading: str,
    xlim: tuple[float, float],
    colors: dict[str, str],
    title: bool = True,
    show_yticklabels: bool = True,
) -> str:
    series, labels, n_total = category_series(df, column)
    names = [cat for cat in reversed(CATEGORY_ORDER)]
    boxed = [vals if len(vals) else np.array([np.nan]) for vals in series]
    ax.boxplot(
        boxed,
        tick_labels=names if show_yticklabels else [""] * len(names),
        vert=False,
        patch_artist=True,
        widths=0.55,
        showfliers=False,
        medianprops={"color": colors["median"], "linewidth": 1.3},
        boxprops={"facecolor": colors["fill"], "edgecolor": colors["edge"], "linewidth": 0.7, "alpha": 0.85},
        whiskerprops={"color": colors["edge"], "linewidth": 0.7},
        capprops={"color": colors["edge"], "linewidth": 0.7},
        zorder=3,
    )
    rng = np.random.default_rng(7)
    x0, x1 = xlim
    for i, vals in enumerate(series, start=1):
        if len(vals):
            jitter = rng.normal(0, 0.07, size=len(vals))
            ax.scatter(
                vals,
                np.full(len(vals), i) + jitter,
                s=12,
                color=colors["point"],
                alpha=0.55,
                edgecolors="none",
                zorder=4,
            )
        ax.text(
            x1,
            i,
            f"n={len(vals)}",
            ha="left",
            va="center",
            fontsize=9,
            color="#555555",
            clip_on=False,
        )
    pad = 0.08 * (x1 - x0)
    ax.set_xlim(x0, x1 + pad)
    ax.set_xlabel(xlabel, fontsize=11)
    full_title = f"{heading}  (n = {n_total} treatments)"
    if title:
        ax.set_title(full_title, loc="left", fontsize=12, fontweight="bold", color=LABEL_COLOR, pad=8)
    hide_spines(ax)
    ax.xaxis.grid(True, color="#E4E4E4", linewidth=0.7)
    if not show_yticklabels:
        ax.tick_params(axis="y", labelleft=False)
    return full_title


def save_figure(fig, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"))
    plt.close(fig)
    print(f"wrote {stem.with_suffix('.pdf')}")
    print(f"wrote {stem.with_suffix('.png')}")


def plot_combined(df: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(12.2, 8.4))
    gs = GridSpec(
        2,
        2,
        wspace=0.24,
        hspace=0.38,
        left=0.26,
        right=0.93,
        top=0.91,
        bottom=0.09,
        figure=fig,
    )
    axes = [fig.add_subplot(gs[i, j]) for i, j in ((0, 0), (0, 1), (1, 0), (1, 1))]
    titles = []
    for idx, (ax, (column, heading, xlabel, _tag, xlim, colors)) in enumerate(zip(axes, PANELS)):
        titles.append(
            draw_property_box(
                ax,
                df,
                column,
                xlabel,
                heading,
                xlim,
                colors,
                title=False,
                show_yticklabels=idx in (0, 2),
            )
        )
    fig.canvas.draw()
    for ax, (_column, _heading, _xlabel, tag, _xlim, _colors), title in zip(axes, PANELS, titles):
        add_panel_header(fig, ax, tag, title)
    save_figure(fig, OUT_COMBINED)


def plot_standalone(
    df: pd.DataFrame,
    column: str,
    heading: str,
    xlabel: str,
    xlim: tuple[float, float],
    colors: dict[str, str],
    out: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    fig.subplots_adjust(left=0.34, right=0.90, top=0.86, bottom=0.16)
    draw_property_box(ax, df, column, xlabel, heading, xlim, colors, title=True)
    save_figure(fig, out)


def main() -> None:
    style()
    df = load_treatments()
    FIGDIR.mkdir(exist_ok=True)
    plot_combined(df)
    plot_standalone(df, "ph", "Biochar pH", "Biochar pH", (2.5, 11.5), GREEN, OUT_A)
    plot_standalone(df, "ssa", "Surface area", "Surface area (m2/g)", (0, 260), ORANGE, OUT_B)
    plot_standalone(df, "ash", "Ash content", "Ash content (%)", (0, 72), GREEN, OUT_C)
    plot_standalone(df, "cec", "CEC", "CEC (cmolc/kg)", (0, 160), ORANGE, OUT_D)
    for column, heading, _xlabel, _tag, _xlim, _colors in PANELS:
        n = int(df[column].notna().sum())
        print(f"{heading}: n={n}")


if __name__ == "__main__":
    main()
