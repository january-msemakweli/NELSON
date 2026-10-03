"""
Study-level soil-bioavailability change.

Cleveland dot plot: one point per extraction x PTE with a numeric
percentage change. Positive values are reductions. Facets are
standard feedstock classes. Point shape (and color) is the PTE.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "ANALYSIS SET.csv"
FIGDIR = ROOT / "figures"
SCRIPTS = Path(__file__).resolve().parent
OUT = FIGDIR / "soil_bioavailability_change"

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from plot_feedstock_composition import CATEGORY_ORDER, standard_category
from plot_target_ptes import PTE_ORDER

LABEL_COLOR = "#1A1A1A"

METAL_ALT = r"Cr\(VI\)|Cd|Pb|Ni|As|Cu|Zn|Fe|Mn|Al|Hg|F|Cr"
NUM = r"(\d+(?:\.\d+)?)(?:\s*-\s*(\d+(?:\.\d+)?))?"

MARKERS = {
    "Cd": "o",
    "Pb": "s",
    "Cr": "D",
    "Ni": "^",
    "As": "v",
    "Cu": "P",
    "Zn": "X",
    "Fe": "*",
    "Mn": "h",
    "Al": "p",
    "F": "8",
    "Hg": "<",
    "Cr(VI)": "d",
}

PTE_COLORS = {
    "Cd": "#C44E52",
    "Pb": "#4C72B0",
    "Cr": "#8172B3",
    "Ni": "#55A868",
    "As": "#CCB974",
    "Cu": "#E07B39",
    "Zn": "#64B5CD",
    "Fe": "#8C8C8C",
    "Mn": "#937860",
    "Al": "#DA8BC3",
    "F": "#8C8C8C",
    "Hg": "#2E8B57",
    "Cr(VI)": "#6B3FA0",
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
            "savefig.pad_inches": 0.10,
        }
    )


def feedstock_group(specific: str, recorded: str = "") -> str:
    return standard_category(specific, recorded)


def midpoint(a: str, b: str | None) -> float:
    if b:
        return (float(a) + float(b)) / 2.0
    return float(a)


METAL_NAMES = {
    "CR(VI)": "Cr(VI)",
    "CD": "Cd",
    "PB": "Pb",
    "NI": "Ni",
    "AS": "As",
    "CU": "Cu",
    "ZN": "Zn",
    "FE": "Fe",
    "MN": "Mn",
    "AL": "Al",
    "HG": "Hg",
    "F": "F",
    "CR": "Cr",
}


def norm_metal(token: str) -> str:
    return METAL_NAMES.get(token.strip().upper(), token.strip())


def clause_weight(clause: str) -> int:
    low = clause.lower()
    if re.search(r"stable fraction|residual fraction|reduced from \d", low):
        return 0
    if re.search(r"bioavailable|available|dtpa|cacl2|exchangeable|nh4", low):
        return 3
    if re.search(r"pore[- ]water|water-soluble|h2o-extractable|immobilization|stabilization", low):
        return 2
    return 1


def sole_metal(targets: str) -> str | None:
    metals = [p.strip() for p in targets.split(";") if p.strip()]
    if metals == ["Cr", "Cr(VI)"]:
        return None
    if len(metals) == 1:
        return metals[0]
    return None


def parse_effects(text: str, targets: str) -> list[tuple[str, float]]:
    text = str(text).strip()
    if not text or text == "NR" or text.startswith("NR ") or text == "Reported graphically":
        return []
    if "%" not in text:
        return []

    pairs: list[tuple[str, float, int]] = []
    single = sole_metal(targets)
    metal_re = r"(Cr\(VI\)|\bCd\b|\bPb\b|\bNi\b|\bAs\b|\bCu\b|\bZn\b|\bFe\b|\bMn\b|\bAl\b|\bHg\b|\bF\b|\bCr\b)"

    for clause in re.split(r";", text):
        clause = clause.strip()
        if not clause or "%" not in clause:
            continue
        weight = clause_weight(clause)
        increased = bool(re.search(r"\bincreased\b|\bmobilization\b", clause, re.I))
        reduced = bool(
            re.search(
                r"\breduced\b|\bdecreased\b|\breduction\b|\bimmobilization\b|"
                r"\bstabilization\b|\blower\b|\bdeclined\b",
                clause,
                re.I,
            )
        )
        sign = -1.0 if increased and not reduced else 1.0

        metal_pcts = list(
            re.finditer(
                rf"{metal_re}\s*(?:[-]extractable)?"
                rf".{{0,40}}?(?:reduced|decreased|increased|lower|declined)?"
                rf"(?: by)?\s*{NUM}\s*%",
                clause,
                flags=re.I,
            )
        )
        if not metal_pcts:
            metal_pcts = list(
                re.finditer(
                    rf"{metal_re}\s*[:\-]?\s*(?:approximately\s+)?{NUM}\s*%",
                    clause,
                    flags=re.I,
                )
            )

        if metal_pcts:
            for m in metal_pcts:
                local = clause[max(0, m.start() - 28) : m.end() + 28]
                if re.search(r"\bincreased\b|\bmobilization\b", local, re.I) and not re.search(
                    r"\breduced\b|\bdecreased\b|\breduction\b", local, re.I
                ):
                    local_sign = -1.0
                elif re.search(r"\bincreased\b", local, re.I) and "increase" in local.lower():
                    local_sign = -1.0
                else:
                    local_sign = sign
                pairs.append(
                    (norm_metal(m.group(1)), local_sign * midpoint(m.group(2), m.group(3)), weight)
                )
            continue

        nums = list(re.finditer(rf"{NUM}\s*%", clause))
        metals = [norm_metal(m.group(1)) for m in re.finditer(metal_re, clause, flags=re.I)]
        if len(nums) == 1 and len(metals) == 1:
            pairs.append((metals[0], sign * midpoint(nums[0].group(1), nums[0].group(2)), weight))
            continue
        if len(nums) >= 1 and len(metals) == 0 and single:
            values = [sign * midpoint(n.group(1), n.group(2)) for n in nums]
            pairs.append((single, float(np.median(values)), weight))
            continue
        if len(nums) >= 1 and len(metals) == 1:
            values = [sign * midpoint(n.group(1), n.group(2)) for n in nums]
            pairs.append((metals[0], float(np.median(values)), weight))

    collapsed: dict[str, list[tuple[float, int]]] = {}
    for metal, value, weight in pairs:
        collapsed.setdefault(metal, []).append((value, weight))
    out = []
    for metal, items in collapsed.items():
        best_w = max(w for _v, w in items)
        vals = [v for v, w in items if w == best_w]
        if vals:
            out.append((metal, float(np.median(vals))))
    return out


def load_effects() -> pd.DataFrame:
    df = pd.read_csv(DATA, dtype=str, keep_default_na=False)
    rows = []
    for _, rec in df.iterrows():
        for metal, value in parse_effects(rec["change_in_soil_bioavailability"], rec["target_metals"]):
            rows.append(
                {
                    "extraction_id": rec["extraction_id"],
                    "study_id": rec["study_id"],
                    "citation": rec["citation"],
                    "year": rec["year"],
                    "feedstock_group": feedstock_group(
                        rec["specific_feedstock"], rec["feedstock_category"]
                    ),
                    "specific_feedstock": rec["specific_feedstock"],
                    "metal": metal,
                    "pct_reduction": value,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["y_label"] = out["citation"]
    return out


def draw_facet(ax, sub: pd.DataFrame, show_xlabel: bool) -> None:
    treatments = (
        sub.groupby(["y_label", "extraction_id", "year"], as_index=False)
        .agg(mid=("pct_reduction", "median"))
        .sort_values(["mid", "year", "extraction_id"])
    )
    order = treatments["extraction_id"].tolist()
    labels = treatments["y_label"].tolist()
    ypos = {eid: i for i, eid in enumerate(order)}

    ax.axvline(0, color="#888888", linewidth=0.8, zorder=1)
    for _, rec in sub.iterrows():
        metal = rec["metal"]
        ax.scatter(
            rec["pct_reduction"],
            ypos[rec["extraction_id"]],
            marker=MARKERS.get(metal, "o"),
            s=36,
            color=PTE_COLORS.get(metal, "#333333"),
            edgecolors="#1A1A1A",
            linewidths=0.35,
            zorder=3,
        )
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.tick_params(axis="x", length=3.5, width=0.6, labelsize=8.5, colors="#333333")
    ax.tick_params(axis="y", length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color="#E4E4E4", linewidth=0.7)
    if show_xlabel:
        ax.set_xlabel("Reduction in soil bioavailability (%)", fontsize=11)
    n_t = treatments["extraction_id"].nunique()
    n_p = len(sub)
    ax.set_title(
        f"{sub['feedstock_group'].iloc[0]}  (n = {n_t} treatments, {n_p} PTE points)",
        loc="left",
        fontsize=10,
        fontweight="bold",
        color=LABEL_COLOR,
        pad=6,
    )


def legend_handles(effects: pd.DataFrame) -> list[Line2D]:
    used = [m for m in PTE_ORDER if m in set(effects["metal"])]
    return [
        Line2D(
            [0],
            [0],
            marker=MARKERS[m],
            color="none",
            markerfacecolor=PTE_COLORS[m],
            markeredgecolor="#1A1A1A",
            markeredgewidth=0.35,
            markersize=8.5,
            label=m,
        )
        for m in used
    ]


def plot_effects(effects: pd.DataFrame) -> None:
    present = [c for c in CATEGORY_ORDER if c in set(effects["feedstock_group"])]
    n_by_cat = {
        cat: int(effects.loc[effects["feedstock_group"] == cat, "extraction_id"].nunique())
        for cat in present
    }
    left_cats = present[0::2]
    right_cats = present[1::2]

    inch_per = 0.175
    title_in = 0.32
    gap_in = 0.42
    xlab_in = 0.42
    top_m = 0.18
    bot_m = 0.16
    ylim_extra = 0.4

    def stack_in(cats: list[str], extra_in: float = 0.0) -> float:
        h = top_m + bot_m + extra_in
        for i, cat in enumerate(cats):
            h += title_in + (n_by_cat[cat] + ylim_extra) * inch_per
            if i < len(cats) - 1:
                h += gap_in
        return h

    fig_w = 16.2
    fig_h = max(stack_in(left_cats, xlab_in), stack_in(right_cats, xlab_in + 1.6))
    fig = plt.figure(figsize=(fig_w, fig_h))

    xmin = min(-10.0, float(effects["pct_reduction"].min()) - 5)
    xmax = max(100.0, float(effects["pct_reduction"].max()) + 5)
    left_x, width = 0.195, 0.295
    right_x = 0.685

    def place_column(cats: list[str], x0: float) -> float:
        y = 1.0 - (top_m / fig_h)
        for i, cat in enumerate(cats):
            data_h = (n_by_cat[cat] + ylim_extra) * inch_per / fig_h
            y -= title_in / fig_h
            ax = fig.add_axes([x0, y - data_h, width, data_h])
            sub = effects.loc[effects["feedstock_group"] == cat].copy()
            draw_facet(ax, sub, show_xlabel=(i == len(cats) - 1))
            ax.set_xlim(xmin, xmax)
            y -= data_h
            if i < len(cats) - 1:
                y -= gap_in / fig_h
        return y

    place_column(left_cats, left_x)
    y_right = place_column(right_cats, right_x)

    ax_leg = fig.add_axes(
        [right_x, 0.035, width, max(y_right - gap_in / fig_h - 0.035, 0.08)]
    )
    ax_leg.set_axis_off()
    ax_leg.set_clip_on(False)
    handles = legend_handles(effects)
    n_rows = 2
    n_cols = int(np.ceil(len(handles) / n_rows))
    leg = ax_leg.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.12, 1.0),
        ncol=n_cols,
        frameon=True,
        fancybox=False,
        edgecolor="#1A1A1A",
        facecolor="white",
        framealpha=1,
        fontsize=10,
        handletextpad=0.45,
        columnspacing=1.35,
        borderpad=0.8,
        labelspacing=0.75,
        borderaxespad=0.15,
        title="Target PTE",
        title_fontsize=11,
    )
    leg.set_clip_on(False)
    frame = leg.get_frame()
    frame.set_visible(True)
    frame.set_linewidth(1.1)
    frame.set_edgecolor("#1A1A1A")
    frame.set_facecolor("white")
    frame.set_alpha(1)

    fig.savefig(OUT.with_suffix(".pdf"))
    fig.savefig(OUT.with_suffix(".png"))
    plt.close(fig)
    print(f"wrote {OUT.with_suffix('.pdf')}")
    print(f"wrote {OUT.with_suffix('.png')}")


def main() -> None:
    style()
    effects = load_effects()
    FIGDIR.mkdir(exist_ok=True)
    print(effects.groupby("feedstock_group").size().reindex(CATEGORY_ORDER).to_string())
    print(f"points={len(effects)} treatments={effects['extraction_id'].nunique()} studies={effects['study_id'].nunique()}")
    print(effects["metal"].value_counts().to_string())
    print(effects["pct_reduction"].describe().to_string())
    plot_effects(effects)


if __name__ == "__main__":
    main()
