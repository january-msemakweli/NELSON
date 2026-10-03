"""
Evidence-gap bubble matrix.

Rows are standard feedstock classes. Columns are major outcomes.
Bubble size is the number of unique studies. Empty cells are gaps.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "ANALYSIS SET.csv"
FIGDIR = ROOT / "figures"
SCRIPTS = Path(__file__).resolve().parent
OUT = FIGDIR / "evidence_gap_map"

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from plot_feedstock_composition import CATEGORY_ORDER, standard_category

LABEL_COLOR = "#1A1A1A"
BUBBLE = "#1B7F4E"
SIZE_NS = (1, 5, 10, 20)

OUTCOMES = [
    "Soil bioavailability",
    "Edible-tissue PTE",
    "BCF/BAF/TF",
    "Plant growth",
    "HQ",
    "HI",
    "CR",
    "Long-term performance",
    "Field validation",
    "Advanced mechanistic characterization",
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
            "savefig.pad_inches": 0.10,
        }
    )


def has_value(text: str) -> bool:
    s = str(text).strip()
    return s not in {"", "NR"}


def metric_reported(text: str) -> bool:
    if not has_value(text):
        return False
    return not bool(
        re.search(
            r"not reported separately|not explicitly (reported|tabulated)|"
            r"no [a-z0-9 +\-/]+-specific numerical",
            text,
            flags=re.I,
        )
    )


def reported_flag(text: str) -> bool:
    return str(text).strip() in {"Yes", "Graphical only"}


def mentions_growth(rec: pd.Series) -> bool:
    blob = f"{rec['main_findings']} {rec['immobilization_mechanism']}".lower()
    return bool(
        re.search(
            r"\bbiomas|\byield\b|plant growth|edible biomass|shoot dry|dry weight",
            blob,
        )
    )


def is_long_term(rec: pd.Series) -> bool:
    dur = pd.to_numeric(rec["experimental_duration_days"], errors="coerce")
    inc = pd.to_numeric(rec["soil_incubation_days"], errors="coerce")
    if pd.notna(dur) and dur >= 180:
        return True
    if pd.notna(inc) and inc >= 180:
        return True
    blob = f"{rec['experimental_duration']} {rec['soil_incubation']}".lower()
    return bool(re.search(r"long[- ]term|\b15 months\b|\bageing\b|\baging\b", blob))


def is_advanced(rec: pd.Series) -> bool:
    blob = " ".join(
        [
            rec["functional_groups"],
            rec["immobilization_mechanism"],
            rec["bioavailability_method_std"],
            rec["main_findings"],
        ]
    ).lower()
    return bool(
        re.search(
            r"\bxrd\b|\bxps\b|\bsem\b|\btem\b|\bxanes\b|\bexafs\b|\bxas\b|\bnmr\b|\bxrf\b|synchrotron",
            blob,
        )
    )


def outcome_flags(rec: pd.Series) -> dict[str, bool]:
    return {
        "Soil bioavailability": has_value(rec["change_in_soil_bioavailability"]),
        "Edible-tissue PTE": has_value(rec["change_in_plant_metal_uptake"]),
        "BCF/BAF/TF": any(
            reported_flag(rec[c]) for c in ("bcf_reported", "baf_reported", "tf_reported")
        ),
        "Plant growth": mentions_growth(rec),
        "HQ": metric_reported(rec["hq"]),
        "HI": metric_reported(rec["hi"]),
        "CR": metric_reported(rec["cr"]),
        "Long-term performance": is_long_term(rec),
        "Field validation": rec["experiment_type"] == "Field",
        "Advanced mechanistic characterization": is_advanced(rec),
    }


def study_matrix(df: pd.DataFrame) -> pd.DataFrame:
    pairs = set()
    for _, rec in df.iterrows():
        feed = standard_category(rec["specific_feedstock"], rec["feedstock_category"])
        flags = outcome_flags(rec)
        for outcome, hit in flags.items():
            if hit:
                pairs.add((rec["study_id"], feed, outcome))
    mat = pd.DataFrame(0, index=CATEGORY_ORDER, columns=OUTCOMES, dtype=int)
    for _study, feed, outcome in pairs:
        if feed in mat.index:
            mat.loc[feed, outcome] += 1
    return mat


def bubble_size(n: int) -> float:
    return 28.0 * n


def wrap_outcome(name: str) -> str:
    if name == "Advanced mechanistic characterization":
        return "Advanced mechanistic\ncharacterization"
    if name == "Long-term performance":
        return "Long-term\nperformance"
    if name == "Soil bioavailability":
        return "Soil\nbioavailability"
    if name == "Edible-tissue PTE":
        return "Edible-tissue\nPTE"
    if name == "Field validation":
        return "Field\nvalidation"
    if name == "Plant growth":
        return "Plant\ngrowth"
    return name


def plot_matrix(mat: pd.DataFrame, n_studies: int) -> None:
    n_r, n_c = mat.shape
    fig, ax = plt.subplots(figsize=(12.6, 8.4))
    ymax = int(mat.to_numpy().max())
    for i, feed in enumerate(mat.index):
        for j, outcome in enumerate(mat.columns):
            n = int(mat.loc[feed, outcome])
            if n == 0:
                continue
            ax.scatter(
                j,
                i,
                s=bubble_size(n),
                color=BUBBLE,
                edgecolors="#145C39",
                linewidths=0.45,
                alpha=0.88,
                zorder=3,
            )
            ax.text(
                j,
                i,
                str(n),
                ha="center",
                va="center",
                fontsize=8,
                color="white" if n >= 8 else LABEL_COLOR,
                zorder=4,
            )

    ax.set_xticks(range(n_c))
    ax.set_xticklabels([wrap_outcome(c) for c in mat.columns], fontsize=8.5)
    ax.xaxis.tick_top()
    ax.set_yticks(range(n_r))
    ax.set_yticklabels(mat.index.tolist(), fontsize=9)
    ax.set_xlim(-0.7, n_c - 0.3)
    ax.set_ylim(n_r - 0.45, -0.55)
    ax.tick_params(axis="both", length=0, pad=4)
    ax.set_axisbelow(True)
    ax.grid(True, color="#E8E8E8", linewidth=0.7, zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_title(
        f"Evidence-gap map  (n = {n_studies} studies)",
        loc="left",
        fontsize=11,
        fontweight="bold",
        color=LABEL_COLOR,
        pad=10,
    )
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=BUBBLE,
            markeredgecolor="#145C39",
            markeredgewidth=0.45,
            markersize=np.sqrt(bubble_size(n)) * 0.55,
            label=f"{n} stud{'y' if n == 1 else 'ies'}",
        )
        for n in SIZE_NS
        if n <= ymax
    ]
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.50, -0.10),
        ncol=len(handles),
        frameon=False,
        fontsize=8.5,
        handletextpad=0.45,
        columnspacing=1.4,
        title="Number of studies",
        title_fontsize=9.5,
    )
    fig.subplots_adjust(left=0.30, right=0.98, top=0.82, bottom=0.12)
    fig.savefig(OUT.with_suffix(".pdf"))
    fig.savefig(OUT.with_suffix(".png"))
    plt.close(fig)
    print(f"wrote {OUT.with_suffix('.pdf')}")
    print(f"wrote {OUT.with_suffix('.png')}")


def main() -> None:
    style()
    df = pd.read_csv(DATA, dtype=str, keep_default_na=False)
    mat = study_matrix(df)
    FIGDIR.mkdir(exist_ok=True)
    print(mat.to_string())
    print()
    print(mat.sum().to_string())
    plot_matrix(mat, int(df["study_id"].nunique()))


if __name__ == "__main__":
    main()
