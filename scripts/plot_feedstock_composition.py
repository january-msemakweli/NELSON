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

# Standard classes from the corrected extraction file / ANALYSIS SET.
CATEGORY_ORDER = [
    "Cereal field residue",
    "Cereal processing residue",
    "Fruit and nut processing residue",
    "Oilseed processing residue",
    "Sugar-crop processing residue",
    "Oil-palm processing residue",
    "Agricultural/root processing residue",
    "Vegetable processing residue",
    "Industrial/fiber crop residue",
    "Legume crop residue",
    "Seed-processing residue",
    "Mixed postharvest agricultural residues",
    "Mixed eligible and non-eligible feedstocks",
]

SPECIFIC_TO_CATEGORY = {
    "Rice husk": "Cereal processing residue",
    "Wheat husk": "Cereal processing residue",
    "Rice straw": "Cereal field residue",
    "Rice stem": "Cereal field residue",
    "Wheat straw": "Cereal field residue",
    "Maize straw": "Cereal field residue",
    "Maize stover": "Cereal field residue",
    "Maize stalk": "Cereal field residue",
    "Peanut shell": "Oilseed processing residue",
    "Camellia oleifera shell": "Oilseed processing residue",
    "Pistachio shell": "Fruit and nut processing residue",
    "Coconut shell": "Fruit and nut processing residue",
    "Coconut husk": "Fruit and nut processing residue",
    "Hazelnut husk": "Fruit and nut processing residue",
    "Acai seed": "Fruit and nut processing residue",
    "Lemon waste": "Fruit and nut processing residue",
    "Orange peel": "Fruit and nut processing residue",
    "Orange bagasse": "Fruit and nut processing residue",
    "Banana peel": "Fruit and nut processing residue",
    "Plantain peel": "Fruit and nut processing residue",
    "Cotton stalk": "Industrial/fiber crop residue",
    "Tobacco straw": "Industrial/fiber crop residue",
    "Pigeon pea stalk": "Legume crop residue",
    "Oil palm bunch": "Oil-palm processing residue",
    "Sugarcane bagasse": "Sugar-crop processing residue",
    "Sugarcane filter cake": "Sugar-crop processing residue",
    "Sugar beet pulp": "Sugar-crop processing residue",
    "Licorice root pulp": "Agricultural/root processing residue",
    "Cereal and grass seed residues": "Seed-processing residue",
    "Vegetable waste": "Vegetable processing residue",
    "Vegetable waste + thiourea": "Vegetable processing residue",
    "Wheat straw + orange peel + rice husk": "Mixed postharvest agricultural residues",
    "Maize straw + cow dung": "Mixed eligible and non-eligible feedstocks",
}

FEEDSTOCK_MARKERS = {
    "Cereal field residue": "o",
    "Cereal processing residue": "s",
    "Fruit and nut processing residue": "^",
    "Oilseed processing residue": "D",
    "Sugar-crop processing residue": "P",
    "Oil-palm processing residue": "v",
    "Agricultural/root processing residue": "h",
    "Vegetable processing residue": "X",
    "Industrial/fiber crop residue": "*",
    "Legume crop residue": "p",
    "Seed-processing residue": "8",
    "Mixed postharvest agricultural residues": "<",
    "Mixed eligible and non-eligible feedstocks": "d",
}

FEEDSTOCK_COLORS = {
    "Cereal field residue": "#1B7F4E",
    "Cereal processing residue": "#55A868",
    "Fruit and nut processing residue": "#C44E52",
    "Oilseed processing residue": "#E07B39",
    "Sugar-crop processing residue": "#4C72B0",
    "Oil-palm processing residue": "#8172B3",
    "Agricultural/root processing residue": "#937860",
    "Vegetable processing residue": "#6A9A23",
    "Industrial/fiber crop residue": "#CCB974",
    "Legume crop residue": "#DA8BC3",
    "Seed-processing residue": "#64B5CD",
    "Mixed postharvest agricultural residues": "#8C8C8C",
    "Mixed eligible and non-eligible feedstocks": "#6B3FA0",
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


def standard_category(specific: str, recorded: str = "") -> str:
    recorded = str(recorded).strip()
    if recorded and recorded != "NR":
        return recorded
    specific = str(specific).strip()
    if specific in SPECIFIC_TO_CATEGORY:
        return SPECIFIC_TO_CATEGORY[specific]
    return "Mixed postharvest agricultural residues"


def classified_pairs(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, rec in df.iterrows():
        specific = rec["specific_feedstock"].strip()
        display = SPECIFIC_DISPLAY.get(specific, specific)
        category = standard_category(specific, rec.get("feedstock_category", ""))
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
    fig = plt.figure(figsize=(16.6, 10.8))
    gs = GridSpec(
        1,
        2,
        width_ratios=[1.08, 1.0],
        wspace=0.58,
        left=0.22,
        right=0.985,
        top=0.90,
        bottom=0.08,
        figure=fig,
    )
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    n_cat = int(cat["n"].sum())
    n_spec = int(spec["n"].sum())
    draw_hbar(
        ax_a,
        cat["category"].astype(str),
        cat["n"],
        "Number of unique study-category combinations",
        f"Feedstock categories  (n = {n_cat} combinations)",
        "A",
    )
    draw_hbar(
        ax_b,
        spec["specific"].astype(str),
        spec["n"],
        "Number of unique study-feedstock combinations",
        f"Specific feedstocks  (n = {n_spec} combinations)",
        "B",
    )
    save_figure(fig, OUT_COMBINED)


def plot_panel_a(cat: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    fig.subplots_adjust(left=0.44, right=0.97, top=0.88, bottom=0.12)
    n_cat = int(cat["n"].sum())
    draw_hbar(
        ax,
        cat["category"].astype(str),
        cat["n"],
        "Number of unique study-category combinations",
        f"Feedstock categories  (n = {n_cat} combinations)",
        None,
    )
    save_figure(fig, OUT_A)


def plot_panel_b(spec: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 9.4))
    fig.subplots_adjust(left=0.36, right=0.97, top=0.94, bottom=0.07)
    n_spec = int(spec["n"].sum())
    draw_hbar(
        ax,
        spec["specific"].astype(str),
        spec["n"],
        "Number of unique study-feedstock combinations",
        f"Specific feedstocks  (n = {n_spec} combinations)",
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
