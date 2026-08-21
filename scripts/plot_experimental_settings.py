"""
Experimental settings: unique included studies by experimental setting.

Horizontal bar chart. Counts are study-level (one row per study_id).
Greenhouse and pot are kept separate using the original extraction wording.
There were no standalone incubation experiments, so that category is omitted.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "ANALYSIS SET.csv"
RAW = ROOT / "Engineer Nelsons Review  (Responses) - Form Responses.csv"
FIGDIR = ROOT / "figures"
OUT = FIGDIR / "experimental_settings"

BAR_COLOR = "#1B7F4E"
BAR_EDGE = "#145C39"
LABEL_COLOR = "#1A1A1A"

# Display order: most frequent first
SETTING_ORDER = [
    "Greenhouse",
    "Pot",
    "Field",
    "Other eligible settings",
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


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().lower()


def classify_setting(value: str) -> str:
    text = clean_text(value)
    if "lysimeter" in text or "chamber" in text:
        return "Other eligible settings"
    if "field" in text:
        return "Field"
    if "greenhouse" in text:
        return "Greenhouse"
    if "pot" in text:
        return "Pot"
    return "Other eligible settings"


def study_counts_by_setting() -> pd.DataFrame:
    analysis = pd.read_csv(ANALYSIS, dtype=str, keep_default_na=False)
    raw = pd.read_csv(RAW, dtype=str, keep_default_na=False)
    sid_col = [c for c in raw.columns if "Study_ID" in c][0]
    raw = raw.rename(columns={sid_col: "study_id", "Experiment_type": "experiment_type_raw"})
    raw["study_id"] = raw["study_id"].str.strip()
    raw["setting"] = raw["experiment_type_raw"].map(classify_setting)

    studies = raw.drop_duplicates(subset="study_id")[["study_id", "setting"]]
    # keep only IDs present in the analysis set
    keep = set(analysis["study_id"].astype(str).str.strip())
    studies = studies.loc[studies["study_id"].isin(keep)].copy()

    counts = studies.groupby("setting", as_index=False).size()
    counts = counts.rename(columns={"size": "n_studies"})
    counts["setting"] = pd.Categorical(counts["setting"], categories=SETTING_ORDER, ordered=True)
    counts = counts.sort_values("setting")
    return counts, int(studies["study_id"].nunique())


def plot_settings(counts: pd.DataFrame, n_studies: int) -> None:
    counts = counts.iloc[::-1].copy()
    fig, ax = plt.subplots(figsize=(7.4, 3.6))

    y = range(len(counts))
    bars = ax.barh(
        list(y),
        counts["n_studies"],
        height=0.62,
        color=BAR_COLOR,
        edgecolor=BAR_EDGE,
        linewidth=0.4,
        zorder=3,
    )

    ax.set_yticks(list(y))
    ax.set_yticklabels(counts["setting"], fontsize=10)
    ax.set_xlabel("Number of unique studies", fontsize=10)
    xmax = max(int(counts["n_studies"].max()) + 4, 8)
    ax.set_xlim(0, xmax)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_title(
        f"Experimental settings  (n = {n_studies} studies)",
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

    for bar, value in zip(bars, counts["n_studies"]):
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
    print(counts.iloc[::-1].to_string(index=False))


def main() -> None:
    style()
    counts, n_studies = study_counts_by_setting()
    plot_settings(counts, n_studies)


if __name__ == "__main__":
    main()
