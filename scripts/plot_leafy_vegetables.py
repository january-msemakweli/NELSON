"""
Leafy vegetable species: unique study-species combinations.

Horizontal bar chart. A study that grew several species is counted once
for each species. Multiple treatment rows of the same study and species
count as one combination.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "ANALYSIS SET.csv"
FIGDIR = ROOT / "figures"
OUT = FIGDIR / "leafy_vegetables"

BAR_COLOR = "#1B7F4E"
BAR_EDGE = "#145C39"
LABEL_COLOR = "#1A1A1A"

NAMED = {
    "Lettuce": "Lettuce",
    "Spinach": "Spinach",
    "Mustard": "Mustard",
    "Pak choi": "Pak choi",
    "Chinese cabbage": "Chinese cabbage",
    "Water spinach": "Water spinach",
    "Amaranth": "Amaranth",
}

SPECIES_ORDER = [
    "Lettuce",
    "Spinach",
    "Mustard",
    "Pak choi",
    "Chinese cabbage",
    "Water spinach",
    "Amaranth",
    "Others",
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


def map_species(name: str) -> str:
    return NAMED.get(name.strip(), "Others")


def study_species_counts(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    rows = []
    for _, rec in df.iterrows():
        for name in [p.strip() for p in rec["species_common"].split(";") if p.strip()]:
            rows.append(
                {
                    "study_id": rec["study_id"],
                    "species": map_species(name),
                    "raw": name,
                }
            )
    long = pd.DataFrame(rows)
    # Unique study-species combinations. Named species collapse by category.
    # Others stay as unique (study, raw species) pairs so cabbage and broccoli
    # in the same study are two combinations.
    named = long.loc[long["species"] != "Others", ["study_id", "species"]].drop_duplicates()
    other = long.loc[long["species"] == "Others", ["study_id", "raw"]].drop_duplicates()
    other = other.assign(species="Others")
    pairs = pd.concat(
        [named, other[["study_id", "species"]]],
        ignore_index=True,
    )
    counts = pairs.groupby("species", as_index=False).size()
    counts = counts.rename(columns={"size": "n_combinations"})
    counts["species"] = pd.Categorical(
        counts["species"], categories=SPECIES_ORDER, ordered=True
    )
    counts = counts.sort_values("species")
    return counts, int(pairs.shape[0])


def plot_species(counts: pd.DataFrame, n_combinations: int) -> None:
    shown = counts.iloc[::-1].copy()
    fig, ax = plt.subplots(figsize=(7.6, 4.2))

    y = range(len(shown))
    bars = ax.barh(
        list(y),
        shown["n_combinations"],
        height=0.62,
        color=BAR_COLOR,
        edgecolor=BAR_EDGE,
        linewidth=0.4,
        zorder=3,
    )

    ax.set_yticks(list(y))
    ax.set_yticklabels(shown["species"], fontsize=10)
    ax.set_xlabel("Number of unique study-species combinations", fontsize=10)
    xmax = max(int(shown["n_combinations"].max()) + 4, 8)
    ax.set_xlim(0, xmax)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_title(
        f"Leafy vegetable species investigated  (n = {n_combinations} combinations)",
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

    for bar, value in zip(bars, shown["n_combinations"]):
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
    print(counts.to_string(index=False))
    print(f"combinations={n_combinations}")


def main() -> None:
    style()
    df = pd.read_csv(DATA, dtype=str, keep_default_na=False)
    counts, n_combinations = study_species_counts(df)
    plot_species(counts, n_combinations)


if __name__ == "__main__":
    main()
