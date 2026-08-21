"""
Contamination sources: unique included studies by contamination source.

Horizontal bar chart. Counts are study-level (one row per study_id).
Each study is assigned one mutually exclusive category:

  Artificially spiked
  Naturally contaminated
  Mining
  Industrial
  Wastewater irrigation
  Mixed/other

Naturally contaminated soils with a stated origin (mine, industrial,
or wastewater irrigation) are placed in that origin category.
Studies with both spiked and natural arms, or coded Both, are Mixed/other.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "ANALYSIS SET.csv"
FIGDIR = ROOT / "figures"
OUT = FIGDIR / "contamination_sources"

BAR_COLOR = "#1B7F4E"
BAR_EDGE = "#145C39"
LABEL_COLOR = "#1A1A1A"

SOURCE_ORDER = [
    "Artificially spiked",
    "Naturally contaminated",
    "Mining",
    "Industrial",
    "Wastewater irrigation",
    "Mixed/other",
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


def classify_study(group: pd.DataFrame) -> str:
    sources = set(group["contamination_source"].tolist())
    classes = set(group["soil_type_class"].tolist())
    soil = " ".join(group["soil_type"].tolist()).lower()

    mixed_sources = sources == {"Artificially spiked", "Naturally contaminated"}
    if mixed_sources or "Both" in sources:
        return "Mixed/other"

    mining = "Mine-affected" in classes or any(
        key in soil for key in ("mine", "mining", "tailing")
    )
    industrial = "Industrial" in classes or "industrial" in soil
    wastewater = "Wastewater-irrigated" in classes or any(
        key in soil for key in ("wastewater", "sewage")
    )

    if sources == {"Artificially spiked"}:
        return "Artificially spiked"
    if mining:
        return "Mining"
    if industrial:
        return "Industrial"
    if wastewater:
        return "Wastewater irrigation"
    return "Naturally contaminated"


def study_counts_by_source(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    assigned = (
        df.groupby("study_id", sort=False)
        .apply(classify_study, include_groups=False)
        .rename("source")
        .reset_index()
    )
    counts = assigned.groupby("source", as_index=False).size()
    counts = counts.rename(columns={"size": "n_studies"})
    counts["source"] = pd.Categorical(counts["source"], categories=SOURCE_ORDER, ordered=True)
    counts = counts.sort_values("source")
    return counts, int(assigned["study_id"].nunique())


def plot_sources(counts: pd.DataFrame, n_studies: int) -> None:
    shown = counts.loc[counts["n_studies"] > 0].sort_values("n_studies", ascending=True)
    fig, ax = plt.subplots(figsize=(7.6, 3.9))

    y = range(len(shown))
    bars = ax.barh(
        list(y),
        shown["n_studies"],
        height=0.62,
        color=BAR_COLOR,
        edgecolor=BAR_EDGE,
        linewidth=0.4,
        zorder=3,
    )

    ax.set_yticks(list(y))
    ax.set_yticklabels(shown["source"], fontsize=10)
    ax.set_xlabel("Number of unique studies", fontsize=10)
    xmax = max(int(shown["n_studies"].max()) + 4, 8)
    ax.set_xlim(0, xmax)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_title(
        f"Contamination sources  (n = {n_studies} studies)",
        loc="left",
        fontsize=11,
        fontweight="bold",
        color=LABEL_COLOR,
        pad=8,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", length=3.5, width=0.6, labelsize=8.5, colors="#333333")
    ax.tick_params(axis="y", length=0)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color="#E4E4E4", linewidth=0.7)

    for bar, value in zip(bars, shown["n_studies"]):
        ax.text(
            bar.get_width() + 0.35,
            bar.get_y() + bar.get_height() / 2,
            str(int(value)),
            ha="left",
            va="center",
            fontsize=8.5,
            color=LABEL_COLOR,
        )

    fig.tight_layout()
    FIGDIR.mkdir(exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"))
    fig.savefig(OUT.with_suffix(".png"))
    plt.close(fig)
    print(f"wrote {OUT.with_suffix('.pdf')}")
    print(f"wrote {OUT.with_suffix('.png')}")
    print(shown.iloc[::-1].to_string(index=False))


def main() -> None:
    style()
    df = pd.read_csv(DATA, dtype=str, keep_default_na=False)
    counts, n_studies = study_counts_by_source(df)
    plot_sources(counts, n_studies)


if __name__ == "__main__":
    main()
