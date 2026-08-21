"""
Risk-of-bias traffic-light plot with a domain-level stacked-bar summary.

Combined figure (A + B) plus standalone panels:
  A  Percent of studies in each judgement, by domain
  B  Study-level traffic-light ratings

Study-level collapse uses the more conservative rating when a study
has more than one extraction (Huang et al. 2024).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "ANALYSIS SET.csv"
FIGDIR = ROOT / "figures"

OUT_COMBINED = FIGDIR / "risk_of_bias_traffic_light"
OUT_A = FIGDIR / "risk_of_bias_A_summary"
OUT_B = FIGDIR / "risk_of_bias_B_traffic_light"

DOMAINS = [
    ("Randomization", "randomization_reported", "item"),
    ("Replication", "replication_reported", "item"),
    ("Controls", "control_treatment_present", "item"),
    ("Biochar characterization", "biochar_characterization_adequate", "item"),
    ("Statistics", "statistical_analysis_reported", "item"),
    ("Overall risk", "overall_risk_of_bias", "overall"),
]

ITEM_WORST = {"Yes": 0, "NR": 1, "No": 2}
OVERALL_WORST = {"Very low": 0, "Low": 1, "Moderate": 2, "High": 3}

ITEM_TO_JUDGEMENT = {
    "Yes": "Low risk",
    "NR": "Some concerns",
    "No": "High risk",
}
OVERALL_TO_JUDGEMENT = {
    "Very low": "Very low",
    "Low": "Low risk",
    "Moderate": "Some concerns",
    "High": "High risk",
}

# Colorblind-safer traffic-light palette
COLORS = {
    "Very low": "#1B7F4E",
    "Low risk": "#5BA86D",
    "Some concerns": "#E6B422",
    "High risk": "#C0392B",
}
JUDGEMENT_ORDER = ["Very low", "Low risk", "Some concerns", "High risk"]

def worst(series: pd.Series, ranking: dict) -> str:
    return max(series.tolist(), key=lambda x: ranking.get(x, -1))


def study_level(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for study_id, g in df.groupby("study_id", sort=False):
        rec = {
            "study_id": study_id,
            "citation": g["citation"].iloc[0],
            "year": int(g["year"].iloc[0]),
            "first_author": g["first_author"].iloc[0],
        }
        for _label, col, kind in DOMAINS:
            rec[col] = worst(g[col], OVERALL_WORST if kind == "overall" else ITEM_WORST)
        rows.append(rec)
    out = pd.DataFrame(rows)
    dup = out["citation"].duplicated(keep=False)
    out["label"] = out["citation"]
    out.loc[dup, "label"] = out.loc[dup].apply(
        lambda r: f"{r['citation']} ({r['study_id']})", axis=1
    )
    out = out.sort_values(
        ["first_author", "year", "study_id"], kind="mergesort"
    ).reset_index(drop=True)
    return out


def judgements(studies: pd.DataFrame) -> pd.DataFrame:
    mapped = studies.copy()
    for label, col, kind in DOMAINS:
        if kind == "item":
            mapped[label] = mapped[col].map(ITEM_TO_JUDGEMENT)
        else:
            mapped[label] = mapped[col].map(OVERALL_TO_JUDGEMENT)
    return mapped


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
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
        color="#1A1A1A",
        clip_on=False,
    )


def draw_summary(ax, mapped: pd.DataFrame, show_tag: bool = True) -> None:
    n = len(mapped)
    labels = [d[0] for d in DOMAINS]
    y = np.arange(len(labels))[::-1]

    left = np.zeros(len(labels))
    # Overall uses a Very low slice; items have none
    for judgement in JUDGEMENT_ORDER:
        widths = []
        for label, _col, kind in DOMAINS:
            if kind == "item" and judgement == "Very low":
                widths.append(0.0)
                continue
            widths.append(100.0 * (mapped[label] == judgement).mean())
        widths = np.array(widths)
        ax.barh(
            y,
            widths,
            left=left,
            height=0.62,
            color=COLORS[judgement],
            edgecolor="white",
            linewidth=0.4,
            label=judgement,
        )
        for yi, w, lo in zip(y, widths, left):
            if w >= 7.5:
                ax.text(
                    lo + w / 2,
                    yi,
                    f"{w:.0f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white" if judgement != "Some concerns" else "#2B2B2B",
                    fontweight="bold",
                )
        left += widths

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.7, len(labels) - 0.3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Studies (%)", fontsize=8)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.tick_params(axis="x", length=3, width=0.6, labelsize=7.5)
    ax.tick_params(axis="y", length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color="#E4E4E4", linewidth=0.6)
    ax.set_title(
        f"Summary of judgements  (n = {n} studies)",
        loc="left",
        fontsize=9,
        fontweight="bold",
        pad=8,
        color="#1A1A1A",
    )
    if show_tag:
        add_panel_tag(ax, "A", x=-0.22, y=1.06)


def draw_traffic_light(ax, mapped: pd.DataFrame, show_tag: bool = True) -> None:
    labels = [d[0] for d in DOMAINS]
    n = len(mapped)
    x = np.arange(len(labels))
    y = np.arange(n)

    for i in range(n):
        if i % 2 == 0:
            ax.add_patch(
                Rectangle(
                    (-0.5, i - 0.5),
                    len(labels),
                    1.0,
                    facecolor="#F6F7F5",
                    edgecolor="none",
                    zorder=0,
                )
            )

    # Divider before overall
    ax.axvline(4.5, color="#C8C8C8", linewidth=0.7, linestyle=(0, (2, 2)), zorder=1)

    for j, label in enumerate(labels):
        for i, judgement in enumerate(mapped[label]):
            ax.scatter(
                j,
                i,
                s=42,
                c=COLORS[judgement],
                edgecolors="white",
                linewidths=0.45,
                zorder=3,
            )

    ax.set_xlim(-0.55, len(labels) - 0.45)
    ax.set_ylim(-0.65, n - 0.35)
    ax.invert_yaxis()
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            "Randomization",
            "Replication",
            "Controls",
            "Biochar\ncharacterization",
            "Statistics",
            "Overall\nrisk",
        ],
        fontsize=7.2,
        ha="center",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(mapped["label"], fontsize=6.2)
    ax.tick_params(axis="x", length=0, pad=4)
    ax.tick_params(axis="y", length=0, pad=3)
    for spine in ax.spines.values():
        spine.set_color("#D0D0D0")
        spine.set_linewidth(0.6)
    ax.set_title(
        "Study-level traffic-light ratings",
        loc="left",
        fontsize=9,
        fontweight="bold",
        pad=8,
        color="#1A1A1A",
    )
    if show_tag:
        add_panel_tag(ax, "B", x=-0.22, y=1.012)


def legend_handles():
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=COLORS[k],
            markeredgecolor="white",
            markeredgewidth=0.6,
            markersize=8,
            label=k,
        )
        for k in JUDGEMENT_ORDER
    ]


def draw_legend(fig, bbox_to_anchor=(0.58, 0.016)) -> None:
    fig.legend(
        handles=legend_handles(),
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=8,
        handletextpad=0.4,
        columnspacing=1.4,
        bbox_to_anchor=bbox_to_anchor,
    )


def save_figure(fig, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"))
    plt.close(fig)
    print(f"wrote {stem.with_suffix('.pdf')}")
    print(f"wrote {stem.with_suffix('.png')}")


def plot_combined(mapped: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(8.8, 16.6))
    gs = GridSpec(
        2,
        1,
        height_ratios=[1.05, 6.55],
        hspace=0.11,
        left=0.30,
        right=0.975,
        top=0.965,
        bottom=0.052,
        figure=fig,
    )
    ax_bar = fig.add_subplot(gs[0])
    ax_tl = fig.add_subplot(gs[1])
    draw_summary(ax_bar, mapped, show_tag=True)
    draw_traffic_light(ax_tl, mapped, show_tag=True)
    draw_legend(fig, bbox_to_anchor=(0.58, 0.016))
    save_figure(fig, OUT_COMBINED)


def plot_panel_a(mapped: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(8.4, 4.15))
    ax = fig.add_axes([0.26, 0.22, 0.70, 0.68])
    draw_summary(ax, mapped, show_tag=False)
    fig.legend(
        handles=legend_handles(),
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=8,
        handletextpad=0.4,
        columnspacing=1.4,
        bbox_to_anchor=(0.58, 0.03),
    )
    save_figure(fig, OUT_A)


def plot_panel_b(mapped: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(8.8, 14.6))
    ax = fig.add_axes([0.30, 0.058, 0.675, 0.90])
    draw_traffic_light(ax, mapped, show_tag=False)
    fig.legend(
        handles=legend_handles(),
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=8,
        handletextpad=0.4,
        columnspacing=1.4,
        bbox_to_anchor=(0.58, 0.012),
    )
    save_figure(fig, OUT_B)


def main() -> None:
    style()
    df = pd.read_csv(DATA, dtype=str, keep_default_na=False)
    studies = study_level(df)
    mapped = judgements(studies)

    FIGDIR.mkdir(exist_ok=True)
    plot_combined(mapped)
    plot_panel_a(mapped)
    plot_panel_b(mapped)

    print(f"studies={len(mapped)}")
    for label, _col, _kind in DOMAINS:
        print(label, mapped[label].value_counts().to_dict())


if __name__ == "__main__":
    main()
