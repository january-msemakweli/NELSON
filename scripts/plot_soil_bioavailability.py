"""
Study-level soil-bioavailability change.

Cleveland dot plot: one point per extraction x PTE with a numeric
percentage change. Positive values are reductions. Facets are
hierarchical feedstock categories. Point shape (and color) is the PTE.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "ANALYSIS SET.csv"
FIGDIR = ROOT / "figures"
SCRIPTS = Path(__file__).resolve().parent
OUT = FIGDIR / "soil_bioavailability_change"

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from plot_feedstock_composition import CATEGORY_ORDER, SPECIFIC_TO_CATEGORY
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


def feedstock_group(specific: str) -> str:
    return SPECIFIC_TO_CATEGORY.get(specific.strip(), "Mixed residues")


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
                    "feedstock_group": feedstock_group(rec["specific_feedstock"]),
                    "specific_feedstock": rec["specific_feedstock"],
                    "metal": metal,
                    "pct_reduction": value,
                    "label": f"{rec['citation']} ({rec['extraction_id']})",
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # Prefer a short label when a study has one extraction in the plot
    counts = out.groupby("citation")["extraction_id"].nunique()
    out["y_label"] = out.apply(
        lambda r: r["citation"] if counts[r["citation"]] == 1 else r["label"],
        axis=1,
    )
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
        fontsize=11,
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
    n_by_cat = {
        cat: int(effects.loc[effects["feedstock_group"] == cat, "extraction_id"].nunique())
        for cat in CATEGORY_ORDER
        if cat in set(effects["feedstock_group"])
    }
    seed_n = n_by_cat.get("Seed/husk/shell residues", 3)
    crop_n = n_by_cat.get("Crop residue", 3)
    proc_n = n_by_cat.get("Processing residue", 3)
    veg_n = n_by_cat.get("Vegetable residue", 2)

    fig = plt.figure(figsize=(16.2, 12.6))
    gs = GridSpec(
        3,
        2,
        height_ratios=[max(seed_n, crop_n), proc_n, max(veg_n, 3)],
        hspace=0.32,
        wspace=0.42,
        left=0.20,
        right=0.99,
        top=0.96,
        bottom=0.06,
        figure=fig,
    )
    xmin = min(-10.0, float(effects["pct_reduction"].min()) - 5)
    xmax = max(100.0, float(effects["pct_reduction"].max()) + 5)

    def add_cat(ax, cat: str, xlabel: bool) -> None:
        sub = effects.loc[effects["feedstock_group"] == cat].copy()
        draw_facet(ax, sub, show_xlabel=xlabel)
        ax.set_xlim(xmin, xmax)

    ax_seed = fig.add_subplot(gs[0, 0])
    ax_crop = fig.add_subplot(gs[0, 1])
    ax_proc = fig.add_subplot(gs[1, 0])
    ax_veg = fig.add_subplot(gs[2, 0])
    add_cat(ax_seed, "Seed/husk/shell residues", False)
    add_cat(ax_crop, "Crop residue", False)
    add_cat(ax_proc, "Processing residue", False)
    add_cat(ax_veg, "Vegetable residue", True)

    pos_seed = ax_seed.get_position()
    pos_proc = ax_proc.get_position()
    pos_crop = ax_crop.get_position()
    pos_veg = ax_veg.get_position()
    gap = pos_seed.y0 - pos_proc.y1
    ylim_extra = 0.4

    def panel_height(n_treat: int) -> float:
        return pos_proc.height * (n_treat + ylim_extra) / (proc_n + ylim_extra)

    fruit_n = n_by_cat.get("Fruit waste", 2)
    mixed_n = n_by_cat.get("Mixed residues", 2)
    fruit_h = panel_height(fruit_n)
    mixed_h = panel_height(mixed_n)
    veg_h = panel_height(veg_n)
    ax_veg.set_position(
        [pos_veg.x0, pos_veg.y1 - veg_h, pos_veg.width, veg_h]
    )
    x0, width = pos_crop.x0, pos_crop.width
    fruit_top = pos_proc.y1
    mixed_top = fruit_top - fruit_h - gap
    add_cat(
        fig.add_axes([x0, fruit_top - fruit_h, width, fruit_h]),
        "Fruit waste",
        False,
    )
    add_cat(
        fig.add_axes([x0, mixed_top - mixed_h, width, mixed_h]),
        "Mixed residues",
        True,
    )

    ax_leg = fig.add_axes(
        [x0, 0.04, width, max(mixed_top - mixed_h - gap - 0.04, 0.08)]
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
