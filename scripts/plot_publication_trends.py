"""
Publication trends: unique included studies by publication year.

One vertical bar chart. Counts are study-level (one row per study_id),
not extraction-level.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "ANALYSIS SET.csv"
FIGDIR = ROOT / "figures"
OUT = FIGDIR / "publication_trends"

BAR_COLOR = "#1B7F4E"
BAR_EDGE = "#145C39"
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
            "savefig.pad_inches": 0.08,
        }
    )


def study_counts_by_year(df: pd.DataFrame) -> pd.DataFrame:
    studies = df.drop_duplicates(subset="study_id")[["study_id", "year"]].copy()
    studies["year"] = studies["year"].astype(int)
    counts = studies.groupby("year", as_index=False).size()
    counts = counts.rename(columns={"size": "n_studies"})
    years = pd.DataFrame({"year": range(int(counts["year"].min()), int(counts["year"].max()) + 1)})
    counts = years.merge(counts, on="year", how="left")
    counts["n_studies"] = counts["n_studies"].fillna(0).astype(int)
    return counts


def plot_trends(counts: pd.DataFrame) -> None:
    n = int(counts["n_studies"].sum())
    fig, ax = plt.subplots(figsize=(7.4, 4.2))

    bars = ax.bar(
        counts["year"],
        counts["n_studies"],
        width=0.72,
        color=BAR_COLOR,
        edgecolor=BAR_EDGE,
        linewidth=0.4,
        zorder=3,
    )

    ymax = max(int(counts["n_studies"].max()) + 2, 4)
    ax.set_ylim(0, ymax)
    ax.set_xlim(counts["year"].min() - 0.7, counts["year"].max() + 0.7)
    ax.set_xticks(counts["year"])
    ax.set_xticklabels(counts["year"], rotation=0)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_xlabel("Publication year", fontsize=10)
    ax.set_ylabel("Number of unique studies", fontsize=10)
    ax.set_title(
        f"Publication trends  (n = {n} studies)",
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
        if value <= 0:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.18,
            str(value),
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
    counts = study_counts_by_year(df)
    plot_trends(counts)


if __name__ == "__main__":
    main()
