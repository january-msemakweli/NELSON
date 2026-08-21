"""
Plant-metal reduction in edible/shoot tissue.

Cleveland dot plot: one point per extraction x PTE with a numeric
percentage change. Positive values are reductions. Facets are target
PTEs. Point shape and color are the hierarchical feedstock group.
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
OUT = FIGDIR / "plant_metal_reduction"

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from plot_feedstock_composition import CATEGORY_ORDER, SPECIFIC_TO_CATEGORY
from plot_target_ptes import PTE_ORDER

LABEL_COLOR = "#1A1A1A"

NUM = r"(-?\d+(?:\.\d+)?)(?:\s*-\s*(-?\d+(?:\.\d+)?))?"
METAL_RE = r"(Cr\(VI\)|\bCd\b|\bPb\b|\bNi\b|\bAs\b|\bCu\b|\bZn\b|\bFe\b|\bMn\b|\bAl\b|\bHg\b|\bF\b|\bCr\b)"

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

FEEDSTOCK_MARKERS = {
    "Seed/husk/shell residues": "o",
    "Crop residue": "s",
    "Processing residue": "D",
    "Fruit waste": "^",
    "Vegetable residue": "v",
    "Mixed residues": "P",
}

FEEDSTOCK_COLORS = {
    "Seed/husk/shell residues": "#1B7F4E",
    "Crop residue": "#E07B39",
    "Processing residue": "#4C72B0",
    "Fruit waste": "#C44E52",
    "Vegetable residue": "#8172B3",
    "Mixed residues": "#937860",
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


def norm_metal(token: str) -> str:
    return METAL_NAMES.get(token.strip().upper(), token.strip())


def sole_metal(targets: str) -> str | None:
    metals = [p.strip() for p in targets.split(";") if p.strip()]
    if metals == ["Cr", "Cr(VI)"]:
        return "Cr"
    if len(metals) == 1:
        return metals[0]
    return None


def nearest_tissue(text: str, pos: int) -> int:
    matches = list(
        re.finditer(
            r"\b(edible(?:[- ]shoot)?|edible parts?|leaves?|shoot|fruit|aerial|stalks?|stems?|roots?)\b",
            text[:pos],
            flags=re.I,
        )
    )
    if not matches:
        return 3
    word = matches[-1].group(1).lower()
    if word.startswith("root"):
        return 1
    if word.startswith("stem") or word.startswith("stalk"):
        return 2
    return 4


def kind_weight(window: str) -> int:
    low = window.lower()
    if re.search(r"translocation", low):
        return 0
    if re.search(r"total \w+ uptake|total uptake|total \w+ accumulation", low):
        return 1
    if re.search(r"concentration", low):
        return 3
    return 2


def signed_pct(value: float, window: str) -> float:
    if value < 0:
        return value
    low = window.lower()
    increased = bool(re.search(r"\bincreased\b|\bmobilization\b", low))
    reduced = bool(
        re.search(
            r"\breduced\b|\bdecreased\b|\breduction\b|\blower\b|\bdeclined\b",
            low,
        )
    )
    if increased and not reduced:
        return -abs(value)
    return value


def skip_text(text: str) -> bool:
    if not text or text == "NR" or text.startswith("NR ") or text.startswith("Reported graphically"):
        return True
    if "%" not in text:
        return True
    if re.search(r"translocation", text, re.I) and not re.search(
        r"concentration|edible|shoot|leaf", text, re.I
    ):
        return True
    return False


def parse_effects(text: str, targets: str) -> list[tuple[str, float]]:
    text = str(text).strip()
    if skip_text(text):
        return []

    items: list[tuple[str, float, int, int]] = []
    used: list[tuple[int, int]] = []
    single = sole_metal(targets)

    def already(pos: int) -> bool:
        return any(a <= pos < b for a, b in used)

    def clause_at(pos: int, end: int) -> str:
        starts = [m.end() for m in re.finditer(r"[.;]\s", text[:pos])]
        start = starts[-1] if starts else 0
        nxt = re.search(r"[.;]\s|[.;]$", text[pos:])
        stop = pos + nxt.start() if nxt else len(text)
        return text[start:max(stop, end)]

    def add(metal: str, value: float, pos: int, end: int) -> None:
        if already(pos):
            return
        if re.match(r"\s*HNC\b", text[end:], flags=re.I) or re.search(r"\(\s*$", text[:pos]):
            return
        clause = clause_at(pos, end)
        local = text[max(0, pos - 40) : end + 40]
        if re.search(
            r"not reported separately|not the selected|exact value for|"
            r"correspond to \d+%",
            clause,
            re.I,
        ):
            return
        if re.search(r"overall reduction range|%\s+overall|\bTHM\b", clause, re.I):
            return
        if re.search(r"\bno consistent\b|\bNS\b|not significant", clause, re.I):
            return
        tw = nearest_tissue(text, end)
        kw = kind_weight(local)
        if kw == 0:
            return
        used.append((pos, end))
        items.append((metal, signed_pct(value, local), tw, kw))

    for m in re.finditer(
        rf"{METAL_RE}\s*:\s*((?:-?\d+(?:\.\d+)?%\s*,\s*)+-?\d+(?:\.\d+)?%)",
        text,
        flags=re.I,
    ):
        metal = norm_metal(m.group(1))
        for n in re.finditer(rf"{NUM}\s*%", m.group(2)):
            add(
                metal,
                midpoint(n.group(1), n.group(2)),
                m.start() + n.start(),
                m.start() + n.end(),
            )

    for m in re.finditer(
        rf"{METAL_RE}\s*:\s*.{{0,80}}?(?:reduced|decreased|increased)"
        rf"(?: by)?\s*(?:approximately\s+|~)?{NUM}\s*%",
        text,
        flags=re.I,
    ):
        add(norm_metal(m.group(1)), midpoint(m.group(2), m.group(3)), m.start(), m.end())

    for m in re.finditer(
        rf"{METAL_RE}\s+(?:concentration|accumulation|uptake|in [\w\s-]{{0,28}})?"
        rf"\s*(?:reduced|decreased|increased|lower|declined)"
        rf"(?: by)?\s*(?:approximately\s+|~)?{NUM}\s*%",
        text,
        flags=re.I,
    ):
        add(norm_metal(m.group(1)), midpoint(m.group(2), m.group(3)), m.start(), m.end())

    for m in re.finditer(
        rf"{METAL_RE}\s+reduction\s+(?:up to\s+|of\s+|=\s+)?{NUM}\s*%",
        text,
        flags=re.I,
    ):
        add(norm_metal(m.group(1)), midpoint(m.group(2), m.group(3)), m.start(), m.end())

    for m in re.finditer(rf"{METAL_RE}\s*:?\s+{NUM}\s*%", text, flags=re.I):
        add(norm_metal(m.group(1)), midpoint(m.group(2), m.group(3)), m.start(), m.end())

    for m in re.finditer(
        rf"(?:approximately\s+|~)?{NUM}\s*%"
        rf"(?:\s+reduction|\s+decrease)?\s+in\s+.{{0,40}}?{METAL_RE}",
        text,
        flags=re.I,
    ):
        add(norm_metal(m.group(3)), midpoint(m.group(1), m.group(2)), m.start(), m.end())

    for m in re.finditer(
        rf"(?:Leaf|Leaves|Shoot|Edible[- ]shoot|Edible parts?)\s*[:,]?\s*"
        rf"(?:reduced by\s+|decreased by\s+|~)?{NUM}\s*%",
        text,
        flags=re.I,
    ):
        if single:
            add(single, midpoint(m.group(1), m.group(2)), m.start(), m.end())

    for n in re.finditer(rf"{NUM}\s*%", text):
        if already(n.start()):
            continue
        before = text[max(0, n.start() - 90) : n.start()]
        metals = list(re.finditer(METAL_RE, before, flags=re.I))
        if not metals:
            continue
        add(
            norm_metal(metals[-1].group(1)),
            midpoint(n.group(1), n.group(2)),
            n.start(),
            n.end(),
        )

    if not items and single:
        nums = list(re.finditer(rf"{NUM}\s*%", text))
        if nums:
            values = [signed_pct(midpoint(n.group(1), n.group(2)), text) for n in nums]
            items.append((single, float(np.median(values)), 3, 2))

    collapsed: dict[str, list[tuple[float, int, int]]] = {}
    for metal, value, tw, kw in items:
        collapsed.setdefault(metal, []).append((value, tw, kw))

    out = []
    for metal, recs in collapsed.items():
        max_tw = max(tw for _v, tw, _kw in recs)
        if max_tw >= 3:
            recs = [r for r in recs if r[1] >= 3]
        else:
            continue
        max_kw = max(kw for _v, _tw, kw in recs)
        recs = [r for r in recs if r[2] == max_kw]
        vals = [v for v, _tw, _kw in recs]
        if vals:
            out.append((metal, float(np.median(vals))))
    return out


def load_effects() -> pd.DataFrame:
    df = pd.read_csv(DATA, dtype=str, keep_default_na=False)
    rows = []
    for _, rec in df.iterrows():
        for metal, value in parse_effects(rec["change_in_plant_metal_uptake"], rec["target_metals"]):
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
    counts = out.groupby(["metal", "citation"])["extraction_id"].nunique()
    out["y_label"] = out.apply(
        lambda r: r["citation"] if counts[(r["metal"], r["citation"])] == 1 else r["label"],
        axis=1,
    )
    return out


def draw_facet(ax, sub: pd.DataFrame, metal: str, show_xlabel: bool) -> None:
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
        grp = rec["feedstock_group"]
        ax.scatter(
            rec["pct_reduction"],
            ypos[rec["extraction_id"]],
            marker=FEEDSTOCK_MARKERS.get(grp, "o"),
            s=36,
            color=FEEDSTOCK_COLORS.get(grp, "#333333"),
            edgecolors="#1A1A1A",
            linewidths=0.35,
            zorder=3,
        )
    n = len(order)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=8.5 if n >= 20 else 9)
    ax.set_ylim(-0.7, n - 0.3)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.tick_params(axis="x", length=3.5, width=0.6, labelsize=8.5, colors="#333333")
    ax.tick_params(axis="y", length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color="#E4E4E4", linewidth=0.7)
    if show_xlabel:
        ax.set_xlabel("Reduction in edible-tissue metal concentration (%)", fontsize=11)
    ax.set_title(
        f"{metal}  (n = {n} treatments)",
        loc="left",
        fontsize=11,
        fontweight="bold",
        color=LABEL_COLOR,
        pad=6,
    )


def legend_handles(effects: pd.DataFrame) -> list[Line2D]:
    used = [g for g in CATEGORY_ORDER if g in set(effects["feedstock_group"])]
    return [
        Line2D(
            [0],
            [0],
            marker=FEEDSTOCK_MARKERS[g],
            color="none",
            markerfacecolor=FEEDSTOCK_COLORS[g],
            markeredgecolor="#1A1A1A",
            markeredgewidth=0.35,
            markersize=8.5,
            label=g,
        )
        for g in used
    ]


def plot_effects(effects: pd.DataFrame) -> None:
    present = [m for m in PTE_ORDER if m in set(effects["metal"])]
    n_by = {
        m: int(effects.loc[effects["metal"] == m, "extraction_id"].nunique())
        for m in present
    }
    left_metals = present[0::2]
    right_metals = present[1::2]

    inch_per = 0.175
    title_in = 0.30
    gap_in = 0.42
    xlab_in = 0.42
    top_m = 0.18
    bot_m = 0.16
    ylim_extra = 0.4

    def stack_in(metals: list[str], extra_in: float = 0.0) -> float:
        h = top_m + bot_m + extra_in
        for i, metal in enumerate(metals):
            h += title_in + (n_by[metal] + ylim_extra) * inch_per
            if i < len(metals) - 1:
                h += gap_in
        return h

    fig_w = 16.2
    fig_h = max(stack_in(left_metals, xlab_in), stack_in(right_metals, xlab_in + 1.35))
    fig = plt.figure(figsize=(fig_w, fig_h))

    xmin = min(-10.0, float(effects["pct_reduction"].min()) - 5)
    xmax = max(100.0, float(effects["pct_reduction"].max()) + 5)

    left_x, width = 0.195, 0.295
    right_x = 0.685

    def place_column(metals: list[str], x0: float) -> float:
        y = 1.0 - (top_m / fig_h)
        last_ax = None
        for i, metal in enumerate(metals):
            data_h = (n_by[metal] + ylim_extra) * inch_per / fig_h
            y -= title_in / fig_h
            ax = fig.add_axes([x0, y - data_h, width, data_h])
            sub = effects.loc[effects["metal"] == metal].copy()
            xlabel = i == len(metals) - 1
            draw_facet(ax, sub, metal, show_xlabel=xlabel)
            ax.set_xlim(xmin, xmax)
            last_ax = ax
            y -= data_h
            if i < len(metals) - 1:
                y -= gap_in / fig_h
        return y, last_ax

    y_left, _ = place_column(left_metals, left_x)
    y_right, _ = place_column(right_metals, right_x)

    ax_leg = fig.add_axes([right_x, 0.035, width, max(y_right - gap_in / fig_h - 0.035, 0.08)])
    ax_leg.set_axis_off()
    ax_leg.set_clip_on(False)
    handles = legend_handles(effects)
    n_cols = 2
    leg = ax_leg.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.42),
        ncol=n_cols,
        frameon=True,
        fancybox=False,
        edgecolor="#1A1A1A",
        facecolor="white",
        framealpha=1,
        fontsize=10,
        handletextpad=0.45,
        columnspacing=1.15,
        borderpad=0.8,
        labelspacing=0.75,
        borderaxespad=0.15,
        title="Feedstock group",
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
    print(effects["metal"].value_counts().reindex(PTE_ORDER).dropna().to_string())
    print(
        f"points={len(effects)} treatments={effects['extraction_id'].nunique()} "
        f"studies={effects['study_id'].nunique()}"
    )
    print(effects.groupby("feedstock_group").size().reindex(CATEGORY_ORDER).to_string())
    print(effects["pct_reduction"].describe().to_string())
    plot_effects(effects)


if __name__ == "__main__":
    main()
