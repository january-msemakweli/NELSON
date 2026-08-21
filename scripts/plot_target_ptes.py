"""
Target PTEs: unique included studies investigating each metal.

Vertical bar chart. A study that reports several metals is counted once
for each metal. Counts are study-level (union across extractions).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "ANALYSIS SET.csv"
FIGDIR = ROOT / "figures"
OUT = FIGDIR / "target_ptes"

BAR_COLOR = "#1B7F4E"
BAR_EDGE = "#145C39"
LABEL_COLOR = "#1A1A1A"

# Manuscript order first, then remaining PTEs present in the data
PTE_ORDER = [
    "Cd",
    "Pb",
    "Cr",
    "Ni",
    "As",
    "Cu",
    "Zn",
    "Fe",
    "Mn",
    "Al",
    "F",
    "Hg",
    "Cr(VI)",
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


def split_metals(value: str) -> set[str]:
    found = set()
    for part in str(value).split(";"):
        metal = part.strip()
        if metal and metal != "NR":
            found.add(metal)
    return found


def study_counts_by_pte(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    rows = []
    for _, group in df.groupby("study_id", sort=False):
        metals: set[str] = set()
        for value in group["target_metals"]:
            metals.update(split_metals(value))
        for metal in metals:
            rows.append({"study_id": group["study_id"].iloc[0], "metal": metal})
    long = pd.DataFrame(rows)
    counts = long.groupby("metal", as_index=False).size()
    counts = counts.rename(columns={"size": "n_studies"})
    counts["metal"] = pd.Categorical(counts["metal"], categories=PTE_ORDER, ordered=True)
    counts = counts.dropna(subset=["metal"]).sort_values("metal")
    return counts, int(df["study_id"].nunique())


def plot_ptes(counts: pd.DataFrame, n_studies: int) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 4.3))

    bars = ax.bar(
        counts["metal"].astype(str),
        counts["n_studies"],
        width=0.72,
        color=BAR_COLOR,
        edgecolor=BAR_EDGE,
        linewidth=0.4,
        zorder=3,
    )

    ymax = max(int(counts["n_studies"].max()) + 5, 8)
    ax.set_ylim(0, ymax)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_xlabel("Target PTE", fontsize=10)
    ax.set_ylabel("Number of unique studies", fontsize=10)
    ax.set_title(
        f"Target PTEs  (n = {n_studies} studies)",
        loc="left",
        fontsize=11,
        fontweight="bold",
        color=LABEL_COLOR,
        pad=8,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", length=3.5, width=0.6, labelsize=8.5, colors="#333333")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#E4E4E4", linewidth=0.7)

    for bar, value in zip(bars, counts["n_studies"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.4,
            str(int(value)),
            ha="center",
            va="bottom",
            fontsize=8,
            color=LABEL_COLOR,
        )

    fig.tight_layout()
    FIGDIR.mkdir(exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"))
    fig.savefig(OUT.with_suffix(".png"))
    plt.close(fig)
    print(f"wrote {OUT.with_suffix('.pdf')}")
    print(f"wrote {OUT.with_suffix('.png')}")
    print(counts.to_string(index=False))


def main() -> None:
    style()
    df = pd.read_csv(DATA, dtype=str, keep_default_na=False)
    counts, n_studies = study_counts_by_pte(df)
    plot_ptes(counts, n_studies)


if __name__ == "__main__":
    main()
