"""
Postharvest agricultural wastes used for biochar production.

Hierarchical two-panel figure:
  A  Unique study-category combinations
  B  Unique Study_ID x specific-feedstock combinations

Also writes standalone A and B files.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "ANALYSIS SET.csv"
FIGDIR = ROOT / "figures"

OUT_COMBINED = FIGDIR / "feedstock_composition"
OUT_A = FIGDIR / "feedstock_composition_A_categories"
OUT_B = FIGDIR / "feedstock_composition_B_specific"

BAR_COLOR = "#1B7F4E"
BAR_EDGE = "#145C39"
LABEL_COLOR = "#1A1A1A"

SPECIFIC_DISPLAY = {
    "Rice husk": "Rice husk",
    "Rice straw": "Rice straw",
    "Wheat straw": "Wheat straw",
    "Maize straw": "Maize straw",
    "Maize stover": "Corn stalk/stover",
    "Maize stalk": "Corn stalk/stover",
    "Peanut shell": "Peanut shell",
    "Pistachio shell": "Pistachio shell",
    "Sugarcane bagasse": "Sugarcane bagasse",
    "Coconut husk": "Coconut husk",
    "Coconut shell": "Coconut shell",
    "Plantain peel": "Plantain peel",
    "Banana peel": "Banana peel",
    "Orange peel": "Orange peel",
    "Orange bagasse": "Orange bagasse",
    "Hazelnut husk": "Hazelnut husk",
    "Wheat husk": "Wheat husk",
    "Acai seed": "Acai seed",
    "Cereal and grass seed residues": "Cereal and grass seed residues",
    "Camellia oleifera shell": "Camellia oleifera shell",
    "Tobacco straw": "Tobacco straw",
    "Rice stem": "Rice stem",
    "Cotton stalk": "Cotton stalk",
    "Pigeon pea stalk": "Pigeon pea stalk",
    "Oil palm bunch": "Oil palm bunch",
    "Sugarcane filter cake": "Sugarcane filter cake",
    "Licorice root pulp": "Licorice root pulp",
    "Sugar beet pulp": "Sugar beet pulp",
    "Lemon waste": "Lemon waste",
    "Vegetable waste": "Vegetable waste",
    "Vegetable waste + thiourea": "Vegetable waste + thiourea",
    "Wheat straw + orange peel + rice husk": "Wheat straw + orange peel + rice husk",
    "Maize straw + cow dung": "Maize straw + cow dung",
}

SPECIFIC_TO_CATEGORY = {
    "Rice husk": "Seed/husk/shell residues",
    "Wheat husk": "Seed/husk/shell residues",
    "Hazelnut husk": "Seed/husk/shell residues",
    "Coconut husk": "Seed/husk/shell residues",
    "Peanut shell": "Seed/husk/shell residues",
    "Pistachio shell": "Seed/husk/shell residues",
    "Coconut shell": "Seed/husk/shell residues",
    "Camellia oleifera shell": "Seed/husk/shell residues",
    "Acai seed": "Seed/husk/shell residues",
    "Cereal and grass seed residues": "Seed/husk/shell residues",
    "Rice straw": "Crop residue",
    "Wheat straw": "Crop residue",
    "Maize straw": "Crop residue",
    "Tobacco straw": "Crop residue",
    "Rice stem": "Crop residue",
    "Cotton stalk": "Crop residue",
    "Maize stover": "Crop residue",
    "Maize stalk": "Crop residue",
    "Pigeon pea stalk": "Crop residue",
    "Banana peel": "Fruit waste",
    "Plantain peel": "Fruit waste",
    "Orange peel": "Fruit waste",
    "Orange bagasse": "Fruit waste",
    "Lemon waste": "Fruit waste",
    "Sugarcane bagasse": "Processing residue",
    "Sugarcane filter cake": "Processing residue",
    "Licorice root pulp": "Processing residue",
    "Sugar beet pulp": "Processing residue",
    "Oil palm bunch": "Processing residue",
    "Vegetable waste": "Vegetable residue",
    "Vegetable waste + thiourea": "Vegetable residue",
    "Wheat straw + orange peel + rice husk": "Mixed residues",
    "Maize straw + cow dung": "Mixed residues",
}

CATEGORY_ORDER = [
    "Seed/husk/shell residues",
    "Crop residue",
    "Processing residue",
    "Fruit waste",
    "Vegetable residue",
    "Mixed residues",
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
            "savefig.pad_inches": 0.08,
        }
    )


def add_panel_tag(ax, tag: str, x: float = -0.02, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        tag,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=LABEL_COLOR,
        clip_on=False,
    )


def classified_pairs(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, rec in df.iterrows():
        specific = rec["specific_feedstock"].strip()
        display = SPECIFIC_DISPLAY.get(specific, specific)
        category = SPECIFIC_TO_CATEGORY.get(specific)
        if category is None:
            category = "Mixed residues" if "+" in specific else "Crop residue"
        rows.append(
            {
                "study_id": rec["study_id"],
                "specific": display,
                "category": category,
            }
        )
    return pd.DataFrame(rows)


def category_counts(pairs: pd.DataFrame) -> pd.DataFrame:
    counts = (
        pairs[["study_id", "category"]]
        .drop_duplicates()
        .groupby("category", as_index=False)
        .size()
        .rename(columns={"size": "n"})
    )
    counts["category"] = pd.Categorical(
        counts["category"], categories=CATEGORY_ORDER, ordered=True
    )
    return counts.sort_values("n", ascending=True)


def specific_counts(pairs: pd.DataFrame) -> pd.DataFrame:
    counts = (
        pairs[["study_id", "specific"]]
        .drop_duplicates()
        .groupby("specific", as_index=False)
        .size()
        .rename(columns={"size": "n"})
        .sort_values(["n", "specific"], ascending=[True, False])
    )
    return counts


def draw_hbar(ax, labels, values, xlabel: str, title: str, tag: str | None) -> None:
    y = range(len(labels))
    bars = ax.barh(
        list(y),
        values,
        height=0.62,
        color=BAR_COLOR,
        edgecolor=BAR_EDGE,
        linewidth=0.4,
        zorder=3,
    )
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel(xlabel, fontsize=9)
    xmax = max(int(max(values)) + 3, 8)
    ax.set_xlim(0, xmax)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold", color=LABEL_COLOR, pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", length=3.5, width=0.6, labelsize=8, colors="#333333")
    ax.tick_params(axis="y", length=0)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color="#E4E4E4", linewidth=0.7)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width() + 0.25,
            bar.get_y() + bar.get_height() / 2,
            str(int(value)),
            ha="left",
            va="center",
            fontsize=8,
            color=LABEL_COLOR,
        )
    if tag:
        add_panel_tag(ax, tag, x=-0.02, y=1.08)


def save_figure(fig, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"))
    plt.close(fig)
    print(f"wrote {stem.with_suffix('.pdf')}")
    print(f"wrote {stem.with_suffix('.png')}")


def plot_combined(cat: pd.DataFrame, spec: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(8.4, 13.2))
    gs = GridSpec(
        2,
        1,
        height_ratios=[1.15, 3.55],
        hspace=0.16,
        left=0.32,
        right=0.97,
        top=0.945,
        bottom=0.05,
        figure=fig,
    )
    fig.suptitle(
        "Types of postharvest agricultural wastes used for biochar production",
        x=0.32,
        ha="left",
        fontsize=12,
        fontweight="bold",
        color=LABEL_COLOR,
    )
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    draw_hbar(
        ax_a,
        cat["category"].astype(str),
        cat["n"],
        "Number of unique study-category combinations",
        "Feedstock categories",
        "A",
    )
    draw_hbar(
        ax_b,
        spec["specific"].astype(str),
        spec["n"],
        "Number of unique study-feedstock combinations",
        "Specific feedstocks",
        "B",
    )
    save_figure(fig, OUT_COMBINED)


def plot_panel_a(cat: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    fig.subplots_adjust(left=0.32, right=0.97, top=0.86, bottom=0.16)
    draw_hbar(
        ax,
        cat["category"].astype(str),
        cat["n"],
        "Number of unique study-category combinations",
        "Feedstock categories",
        None,
    )
    save_figure(fig, OUT_A)


def plot_panel_b(spec: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 9.4))
    fig.subplots_adjust(left=0.36, right=0.97, top=0.94, bottom=0.07)
    draw_hbar(
        ax,
        spec["specific"].astype(str),
        spec["n"],
        "Number of unique study-feedstock combinations",
        "Specific feedstocks",
        None,
    )
    save_figure(fig, OUT_B)


def main() -> None:
    style()
    df = pd.read_csv(DATA, dtype=str, keep_default_na=False)
    pairs = classified_pairs(df)
    cat = category_counts(pairs)
    spec = specific_counts(pairs)
    FIGDIR.mkdir(exist_ok=True)
    plot_combined(cat, spec)
    plot_panel_a(cat)
    plot_panel_b(spec)
    print(cat.sort_values("n", ascending=False).to_string(index=False))
    print(spec.sort_values("n", ascending=False).to_string(index=False))
    print(f"study-category pairs={int(cat['n'].sum())}")
    print(f"study-feedstock pairs={int(spec['n'].sum())}")


if __name__ == "__main__":
    main()
