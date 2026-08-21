"""
Biochar production conditions.

Four-panel figure, extraction-level (one row = one Extraction_ID):
  A  Histogram of pyrolysis temperature (left, top)
  B  Boxplot of pyrolysis temperature by feedstock category (left, bottom)
  C  Horizontal bar chart of biochar modification (right, top; same height as A)
  D  Radar of median biochar properties, unmodified vs modified (right, bottom)

Also writes standalone A, B, C, and D files.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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

OUT_COMBINED = FIGDIR / "biochar_production_conditions"
OUT_A = FIGDIR / "biochar_production_A_temperature"
OUT_B = FIGDIR / "biochar_production_B_temperature_by_feedstock"
OUT_C = FIGDIR / "biochar_production_C_modification"
OUT_D = FIGDIR / "biochar_production_D_properties_radar"

BAR_COLOR = "#1B7F4E"
BAR_EDGE = "#145C39"
LABEL_COLOR = "#1A1A1A"
POINT_COLOR = "#3A9A68"

FE_CLASSES = {
    "Fe modified",
    "Fe-Mn modified",
    "Fe-Mg modified",
    "FeCl3 modified",
    "nZVI",
}

MOD_DISPLAY_ORDER = [
    "Unmodified",
    "Fe modified",
    "Phosphate modified",
    "Mineral modified",
    "Microbial/co-amended",
    "Compost-associated",
    "Particle-size modified",
    "Other",
]

TEMP_BINS = np.arange(275, 701, 50)

RADAR_AXES = [
    ("Temp.", "temp_c"),
    ("Time", "residence_min"),
    ("pH", "biochar_ph"),
    ("SSA", "ssa"),
    ("Ash", "ash"),
    ("CEC", "cec"),
]

RADAR_SERIES = [
    ("Unmodified", "#1B7F4E"),
    ("Modified/co-amended", "#E07B39"),
]

SPOKE_COLORS = ["#1B7F4E", "#E07B39", "#2E86AB", "#B4539A", "#C9A227", "#4A8B8B"]


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
            "savefig.pad_inches": 0.08,
        }
    )


def add_panel_header(fig, ax, tag: str, title: str) -> None:
    """Place the panel letter and title on the left edge of that panel."""
    pos = ax.get_position()
    y_off = 0.038 if getattr(ax, "name", "") == "polar" else 0.012
    fig.text(
        pos.x0,
        pos.y1 + y_off,
        f"{tag}   {title}",
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=LABEL_COLOR,
    )


def hide_spines(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", length=3.5, width=0.6, labelsize=8.5, colors="#333333")
    ax.set_axisbelow(True)


def feedstock_category(specific: str) -> str:
    return SPECIFIC_TO_CATEGORY.get(specific.strip(), "Mixed residues")


def classify_modification(row: pd.Series) -> str:
    """One mutually exclusive class per extraction.

    Char modification is applied first. If the char is unmodified, the
    co-amendment is used so compost, inoculants, and minerals are visible.
    """
    mclass = str(row["modification_class"]).strip()
    co = str(row["co_amendment"]).strip().lower()
    coclass = str(row["co_amendment_class"]).strip()

    if mclass in FE_CLASSES:
        return "Fe modified"
    if mclass == "Phosphate-enriched":
        return "Phosphate modified"
    if mclass == "Particle-size reduced (nano)":
        return "Particle-size modified"
    if mclass == "Microbially loaded":
        return "Microbial/co-amended"
    if mclass not in {"None", "", "NR"}:
        return "Other"

    if "compost" in co:
        return "Compost-associated"
    if coclass == "Microbial":
        return "Microbial/co-amended"
    if coclass == "Phosphate mineral":
        return "Mineral modified"
    if coclass == "Mineral":
        return "Mineral modified"
    if coclass in {"Organic", "Fertilizer", "Chemical", "Metal/chemical"}:
        return "Other"
    return "Unmodified"


def load_treatments() -> pd.DataFrame:
    df = pd.read_csv(DATA, dtype=str, keep_default_na=False)
    out = df[
        [
            "extraction_id",
            "specific_feedstock",
            "modification_class",
            "co_amendment",
            "co_amendment_class",
            "pyrolysis_temp_c",
            "residence_time_min",
            "biochar_ph",
            "surface_area_m2_g",
            "ash_content_pct",
            "cec_cmolc_kg",
            "cec_reported_for",
        ]
    ].copy()
    out["temp_c"] = pd.to_numeric(out["pyrolysis_temp_c"], errors="coerce")
    out["residence_min"] = pd.to_numeric(out["residence_time_min"], errors="coerce")
    out["biochar_ph"] = pd.to_numeric(out["biochar_ph"], errors="coerce")
    out["ssa"] = pd.to_numeric(out["surface_area_m2_g"], errors="coerce")
    out["ash"] = pd.to_numeric(out["ash_content_pct"], errors="coerce")
    cec = pd.to_numeric(out["cec_cmolc_kg"], errors="coerce")
    out["cec"] = cec.where(out["cec_reported_for"] == "biochar")
    out["feedstock_group"] = out["specific_feedstock"].map(feedstock_category)
    out["mod_group"] = out.apply(classify_modification, axis=1)
    out["mod_series"] = np.where(
        out["mod_group"] == "Unmodified",
        "Unmodified",
        "Modified/co-amended",
    )
    return out


def radar_medians(df: pd.DataFrame) -> dict[str, list[float]]:
    medians: dict[str, list[float]] = {}
    for series_name, _color in RADAR_SERIES:
        sub = df.loc[df["mod_series"] == series_name]
        values = []
        for _label, column in RADAR_AXES:
            values.append(float(sub[column].median()))
        medians[series_name] = values
    return medians


def modification_counts(df: pd.DataFrame) -> pd.DataFrame:
    counts = (
        df.groupby("mod_group", as_index=False)
        .size()
        .rename(columns={"size": "n"})
    )
    counts["mod_group"] = pd.Categorical(
        counts["mod_group"], categories=MOD_DISPLAY_ORDER, ordered=True
    )
    return counts.sort_values("n", ascending=True)


def draw_histogram(ax, temps: pd.Series, title: bool = True) -> str:
    counts, edges, patches = ax.hist(
        temps,
        bins=TEMP_BINS,
        color=BAR_COLOR,
        edgecolor=BAR_EDGE,
        linewidth=0.45,
        zorder=3,
    )
    for patch in patches:
        patch.set_zorder(3)
    ax.set_xlim(275, 700)
    ax.set_xticks([300, 350, 400, 450, 500, 550, 600, 650])
    ymax = max(int(counts.max()) + 5, 10)
    ax.set_ylim(0, ymax)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_xlabel("Pyrolysis temperature (°C)", fontsize=9)
    ax.set_ylabel("Number of biochar treatments", fontsize=9)
    heading = f"Pyrolysis temperature  (n = {int(temps.notna().sum())} treatments)"
    if title:
        ax.set_title(heading, loc="left", fontsize=10, fontweight="bold", color=LABEL_COLOR, pad=8)
    hide_spines(ax)
    ax.yaxis.grid(True, color="#E4E4E4", linewidth=0.7)
    ax.yaxis.set_label_coords(-0.10, 0.5)
    for count, left, right in zip(counts, edges[:-1], edges[1:]):
        if count <= 0:
            continue
        ax.text(
            (left + right) / 2,
            count + 0.6,
            str(int(count)),
            ha="center",
            va="bottom",
            fontsize=8,
            color=LABEL_COLOR,
        )
    return heading


def draw_boxplot(ax, df: pd.DataFrame, title: bool = True) -> str:
    with_temp = df.dropna(subset=["temp_c"]).copy()
    series = []
    labels = []
    # Largest / manuscript-first category at the top
    for cat in reversed(CATEGORY_ORDER):
        vals = with_temp.loc[with_temp["feedstock_group"] == cat, "temp_c"].to_numpy()
        series.append(vals)
        labels.append(f"{cat} (n = {len(vals)})")

    ax.boxplot(
        series,
        tick_labels=labels,
        vert=False,
        patch_artist=True,
        widths=0.55,
        medianprops={"color": "#0D3B24", "linewidth": 1.3},
        boxprops={"facecolor": BAR_COLOR, "edgecolor": BAR_EDGE, "linewidth": 0.7, "alpha": 0.85},
        whiskerprops={"color": BAR_EDGE, "linewidth": 0.7},
        capprops={"color": BAR_EDGE, "linewidth": 0.7},
        flierprops={
            "marker": "o",
            "markersize": 3.5,
            "markerfacecolor": BAR_COLOR,
            "markeredgecolor": BAR_EDGE,
            "markeredgewidth": 0.4,
        },
        zorder=3,
    )
    rng = np.random.default_rng(7)
    for i, vals in enumerate(series, start=1):
        if len(vals) == 0:
            continue
        jitter = rng.normal(0, 0.07, size=len(vals))
        ax.scatter(
            vals,
            np.full(len(vals), i) + jitter,
            s=11,
            color=POINT_COLOR,
            alpha=0.45,
            edgecolors="none",
            zorder=4,
        )
    ax.set_xlim(275, 700)
    ax.set_xticks([300, 350, 400, 450, 500, 550, 600, 650])
    ax.set_xlabel("Pyrolysis temperature (°C)", fontsize=9)
    heading = f"Temperature by feedstock category  (n = {len(with_temp)} treatments)"
    if title:
        ax.set_title(heading, loc="left", fontsize=10, fontweight="bold", color=LABEL_COLOR, pad=8)
    hide_spines(ax)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.xaxis.grid(True, color="#E4E4E4", linewidth=0.7)
    return heading


def draw_modification(ax, counts: pd.DataFrame, title: bool = True) -> str:
    y = range(len(counts))
    bars = ax.barh(
        list(y),
        counts["n"],
        height=0.62,
        color=BAR_COLOR,
        edgecolor=BAR_EDGE,
        linewidth=0.4,
        zorder=3,
    )
    ax.set_yticks(list(y))
    ax.set_yticklabels(counts["mod_group"].astype(str), fontsize=8.5)
    ax.set_xlabel("Number of biochar treatments", fontsize=9)
    xmax = max(int(counts["n"].max()) + 8, 12)
    ax.set_xlim(0, xmax)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    heading = f"Biochar modification  (n = {int(counts['n'].sum())} treatments)"
    if title:
        ax.set_title(heading, loc="left", fontsize=10, fontweight="bold", color=LABEL_COLOR, pad=8)
    hide_spines(ax)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.xaxis.grid(True, color="#E4E4E4", linewidth=0.7)
    for bar, value in zip(bars, counts["n"]):
        ax.text(
            bar.get_width() + 0.6,
            bar.get_y() + bar.get_height() / 2,
            str(int(value)),
            ha="left",
            va="center",
            fontsize=8,
            color=LABEL_COLOR,
        )
    return heading


def draw_radar(ax, df: pd.DataFrame, title: bool = True) -> str:
    medians = radar_medians(df)
    labels = [label for label, _column in RADAR_AXES]
    n_axes = len(labels)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False)
    scale = np.array(
        [
            max(medians[name][i] for name, _color in RADAR_SERIES) or 1.0
            for i in range(n_axes)
        ]
    )

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles), labels)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels([])
    ax.tick_params(axis="x", pad=8, labelsize=8)
    ax.grid(color="#D0D0D0", linewidth=0.7)
    ax.spines["polar"].set_color("#A8A8A8")
    ax.spines["polar"].set_linewidth(0.8)
    for tick, color in zip(ax.get_xticklabels(), SPOKE_COLORS):
        tick.set_color(color)
        tick.set_fontweight("bold")

    for series_name, color in RADAR_SERIES:
        raw = np.array(medians[series_name], dtype=float)
        scaled = np.divide(raw, scale, out=np.zeros_like(raw), where=scale > 0)
        closed = np.concatenate([scaled, scaled[:1]])
        closed_angles = np.concatenate([angles, angles[:1]])
        ax.plot(closed_angles, closed, color=color, linewidth=2.0, zorder=4)
        ax.fill(closed_angles, closed, color=color, alpha=0.22, zorder=3)
        ax.scatter(angles, scaled, s=28, color=color, edgecolors="white", linewidths=0.5, zorder=5)

    n_unmod = int((df["mod_series"] == "Unmodified").sum())
    n_mod = int((df["mod_series"] == "Modified/co-amended").sum())
    heading = "Biochar properties"
    if title:
        ax.set_title(heading, loc="left", fontsize=10, fontweight="bold", color=LABEL_COLOR, pad=28)
    ax.legend(
        handles=[
            plt.Line2D([0], [0], color=color, lw=2.4, label=f"{name} (n = {n})")
            for (name, color), n in zip(RADAR_SERIES, (n_unmod, n_mod))
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        frameon=False,
        fontsize=7.5,
        handlelength=1.5,
        columnspacing=1.2,
    )
    return heading


def save_figure(fig, stem: Path, tight: bool = True) -> None:
    kwargs = {} if tight else {"bbox_inches": None, "pad_inches": 0.0}
    fig.savefig(stem.with_suffix(".pdf"), **kwargs)
    fig.savefig(stem.with_suffix(".png"), **kwargs)
    plt.close(fig)
    print(f"wrote {stem.with_suffix('.pdf')}")
    print(f"wrote {stem.with_suffix('.png')}")


def plot_combined(df: pd.DataFrame, mods: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(12.6, 8.0))
    gs = GridSpec(
        2,
        2,
        width_ratios=[1.12, 1.08],
        height_ratios=[0.92, 1.38],
        wspace=0.32,
        hspace=0.42,
        left=0.17,
        right=0.985,
        top=0.91,
        bottom=0.10,
        figure=fig,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[0, 1])
    ax_d = fig.add_subplot(gs[1, 1], projection="polar")

    title_a = draw_histogram(ax_a, df["temp_c"].dropna(), title=False)
    title_b = draw_boxplot(ax_b, df, title=False)
    title_c = draw_modification(ax_c, mods, title=False)
    title_d = draw_radar(ax_d, df, title=False)
    fig.canvas.draw()

    pos_b = ax_b.get_position()
    pos_a = ax_a.get_position()
    pos_c = ax_c.get_position()
    pos_d = ax_d.get_position()
    ax_a.set_position([pos_b.x0, pos_a.y0, pos_b.width, pos_a.height])
    ax_c.set_position([pos_c.x0, pos_a.y0, pos_c.width, pos_a.height])
    pos_c = ax_c.get_position()
    # Enlarge D to a square so the radar fills the bottom-right space.
    gap = 0.045
    left = pos_b.x0 + pos_b.width + gap
    right = 0.99
    bottom = 0.07
    top = pos_d.y1 - 0.012
    side = min(right - left, top - bottom)
    ax_d.set_position([left + (right - left - side) / 2, bottom, side, side])

    add_panel_header(fig, ax_a, "A", title_a)
    add_panel_header(fig, ax_b, "B", title_b)
    title_x = left
    fig.text(
        title_x,
        pos_c.y1 + 0.012,
        f"C   {title_c}",
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=LABEL_COLOR,
    )
    fig.text(
        title_x,
        top + 0.018,
        f"D   {title_d}",
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=LABEL_COLOR,
    )
    save_figure(fig, OUT_COMBINED, tight=True)


def plot_panel_a(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 4.0))
    fig.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.16)
    draw_histogram(ax, df["temp_c"].dropna(), title=True)
    save_figure(fig, OUT_A)


def plot_panel_b(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    fig.subplots_adjust(left=0.38, right=0.97, top=0.88, bottom=0.16)
    draw_boxplot(ax, df, title=True)
    save_figure(fig, OUT_B)


def plot_panel_c(mods: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    fig.subplots_adjust(left=0.30, right=0.97, top=0.88, bottom=0.16)
    draw_modification(ax, mods, title=True)
    save_figure(fig, OUT_C)


def plot_panel_d(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.6), subplot_kw={"projection": "polar"})
    fig.subplots_adjust(left=0.10, right=0.90, top=0.84, bottom=0.14)
    draw_radar(ax, df, title=True)
    save_figure(fig, OUT_D)


def main() -> None:
    style()
    df = load_treatments()
    mods = modification_counts(df)
    FIGDIR.mkdir(exist_ok=True)
    plot_combined(df, mods)
    plot_panel_a(df)
    plot_panel_b(df)
    plot_panel_c(mods)
    plot_panel_d(df)
    print(f"treatments={len(df)}")
    print(f"treatments with temperature={int(df['temp_c'].notna().sum())}")
    print(mods.sort_values("n", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
