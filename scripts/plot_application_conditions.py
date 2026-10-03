"""
Application-condition plots versus plant-metal reduction.

Three scatter panels, using the same edible-part plant-metal parser:
  A  Application rate (% w/w)
  B  Initial soil pH
  C  Initial soil PTE concentration (mg/kg)

A and B are treatment-level medians. C matches each metal's plant
reduction to that metal's parsed initial concentration. Multi-dose
lists, wastewater mg/L records, and extractable-only values that sit
beside a total are omitted from C.
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
OUT = FIGDIR / "application_conditions"

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from plot_feedstock_composition import (
    CATEGORY_ORDER,
    FEEDSTOCK_COLORS,
    FEEDSTOCK_MARKERS,
)
from plot_plant_metal_reduction import load_effects as load_plant
from plot_plant_metal_reduction import METAL_NAMES, METAL_RE, sole_metal

LABEL_COLOR = "#1A1A1A"
NUM = r"(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)"


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


def norm_metal(token: str) -> str:
    return METAL_NAMES.get(token.strip().upper(), token.strip())


def to_float(raw: str) -> float:
    return float(raw.replace(",", ""))


def skip_initial(text: str) -> bool:
    s = str(text).strip()
    if not s or s == "NR":
        return True
    if not re.search(r"\d", s):
        return True
    low = s.lower()
    if re.search(r"mg\s*/\s*l\b|mmol", low):
        return True
    if re.search(r"\bsg\s*:", low) and re.search(r"malir", low):
        return True
    if re.search(
        r"experimental (?:pb |cr |ni |cd )?(?:levels|additions|treatments)|"
        r"spiked at 0|cd spiked at|"
        r"supplied at|unamended cr-100|"
        r"0,\s*2\.5 and|"
        r"^\s*0\s*[,;]\s*\d",
        low,
    ):
        return True
    if re.fullmatch(r"\s*0\s*,\s*1\s*,\s*2\s*", s):
        return True
    return False


def rank_clause(clause: str) -> int:
    low = clause.lower()
    if re.search(r"available|dtpa|extractable|ab-dtpa", low):
        return 0
    if re.search(r"\btotal\b|legacy|initial soil total|after ageing|after aging", low):
        return 2
    return 1


def parse_initial(text: str, targets: str) -> list[tuple[str, float]]:
    if skip_initial(text):
        return []
    text = str(text)
    text = re.sub(r"(?:\u00b1|\ufffd)\s*" + NUM, "", text)
    text = re.sub(r"\s+", " ", text)
    items: list[tuple[str, float, int]] = []

    def add(metal: str, value: float, clause: str) -> None:
        if value <= 0:
            return
        items.append((metal, value, rank_clause(clause)))

    spike = re.search(
        rf"native soil {METAL_RE}\s*=\s*{NUM}.*?spiked with\s*{NUM}",
        text,
        flags=re.I,
    )
    if spike:
        metal = norm_metal(spike.group(1))
        add(metal, to_float(spike.group(2)) + to_float(spike.group(3)), spike.group(0))
        return [(metal, to_float(spike.group(2)) + to_float(spike.group(3)))]

    legacy = re.search(
        rf"legacy {METAL_RE}\s*=\s*{NUM}.*?newly introduced {METAL_RE}\s*=\s*{NUM}",
        text,
        flags=re.I,
    )
    if legacy:
        metal = norm_metal(legacy.group(1))
        add(metal, to_float(legacy.group(2)) + to_float(legacy.group(4)), legacy.group(0))
        return [(m, v) for m, v, _w in items]

    soil_total = re.search(
        rf"initial soil total {METAL_RE}\s*=\s*{NUM}",
        text,
        flags=re.I,
    )
    if soil_total and re.search(r"fresh bcb|sar-aged|abcb", text, flags=re.I):
        metal = norm_metal(soil_total.group(1))
        return [(metal, to_float(soil_total.group(2)))]

    for m in re.finditer(
        rf"(?:total\s+)?{METAL_RE}\s*[=:]\s*{NUM}(?:\s*-\s*{NUM})?",
        text,
        flags=re.I,
    ):
        lo = to_float(m.group(2))
        hi = to_float(m.group(3)) if m.group(3) else lo
        add(norm_metal(m.group(1)), (lo + hi) / 2.0, m.group(0))

    for m in re.finditer(
        rf"{METAL_RE}\s+{NUM}(?:\s*-\s*{NUM})?",
        text,
        flags=re.I,
    ):
        lo = to_float(m.group(2))
        hi = to_float(m.group(3)) if m.group(3) else lo
        add(norm_metal(m.group(1)), (lo + hi) / 2.0, m.group(0))

    for m in re.finditer(
        rf"{NUM}\s*mg\s*(?:/\s*kg)?\s*{METAL_RE}",
        text,
        flags=re.I,
    ):
        add(norm_metal(m.group(2)), to_float(m.group(1)), m.group(0))

    single = sole_metal(targets)
    if not items and single:
        nums = [to_float(n.group(1)) for n in re.finditer(NUM, text)]
        nums = [v for v in nums if v > 0]
        if len(nums) == 1:
            add(single, nums[0], text)
        elif len(nums) == 2 and re.search(r"\d\s*-\s*\d", text):
            add(single, (nums[0] + nums[1]) / 2.0, text)

    collapsed: dict[str, list[tuple[float, int]]] = {}
    for metal, value, weight in items:
        collapsed.setdefault(metal, []).append((value, weight))
    out = []
    for metal, recs in collapsed.items():
        best = max(w for _v, w in recs)
        vals = [v for v, w in recs if w == best]
        if vals:
            out.append((metal, float(np.median(vals))))
    return out


def load_meta() -> pd.DataFrame:
    df = pd.read_csv(DATA, dtype=str, keep_default_na=False)
    return df[
        [
            "extraction_id",
            "study_id",
            "target_metals",
            "initial_metal_concentration",
            "soil_ph",
            "application_rate_pct_ww",
            "feedstock_category",
            "specific_feedstock",
        ]
    ].copy()


def treatment_points(plant: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    mid = plant.groupby("extraction_id", as_index=False).agg(
        pct_reduction=("pct_reduction", "median"),
        feedstock_group=("feedstock_group", "first"),
        study_id=("study_id", "first"),
    )
    out = mid.merge(
        meta[
            [
                "extraction_id",
                "soil_ph",
                "application_rate_pct_ww",
            ]
        ],
        on="extraction_id",
        how="left",
    )
    out["soil_ph"] = pd.to_numeric(out["soil_ph"], errors="coerce")
    out["rate_ww"] = pd.to_numeric(out["application_rate_pct_ww"], errors="coerce")
    return out


def concentration_points(plant: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, rec in meta.iterrows():
        for metal, conc in parse_initial(
            rec["initial_metal_concentration"], rec["target_metals"]
        ):
            rows.append(
                {
                    "extraction_id": rec["extraction_id"],
                    "metal": metal,
                    "conc_mg_kg": conc,
                }
            )
    parsed = pd.DataFrame(rows)
    if parsed.empty:
        return parsed
    out = plant.merge(parsed, on=["extraction_id", "metal"], how="inner")
    return out


def draw_scatter(
    ax,
    pts: pd.DataFrame,
    xcol: str,
    title: str,
    xlabel: str,
    xlim: tuple[float, float],
    logx: bool = False,
) -> None:
    ax.axhline(0, color="#888888", linewidth=0.8, zorder=1)
    for _, rec in pts.iterrows():
        grp = rec["feedstock_group"]
        ax.scatter(
            rec[xcol],
            rec["pct_reduction"],
            s=42,
            marker=FEEDSTOCK_MARKERS.get(grp, "o"),
            color=FEEDSTOCK_COLORS.get(grp, "#333333"),
            edgecolors="#1A1A1A",
            linewidths=0.35,
            alpha=0.85,
            zorder=3,
        )
    if logx:
        ax.set_xscale("log")
        ax.set_xlim(xlim)
    else:
        ax.set_xlim(*xlim)
    ax.set_ylim(-5, 108)
    ax.set_yticks(range(0, 101, 20))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(10))
    ax.tick_params(axis="both", length=3.5, width=0.6, labelsize=8.5, colors="#333333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#E4E4E4", linewidth=0.7)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Reduction in edible-part PTE concentration (%)", fontsize=10)
    n = int(pts["extraction_id"].nunique())
    extra = f", {len(pts)} PTE points" if logx else ""
    ax.set_title(
        f"{title}  (n = {n} treatments{extra})",
        loc="left",
        fontsize=11,
        fontweight="bold",
        color=LABEL_COLOR,
        pad=8,
    )


def feedstock_handles(pts: pd.DataFrame) -> list[Line2D]:
    used = set(pts["feedstock_group"])
    return [
        Line2D(
            [0],
            [0],
            marker=FEEDSTOCK_MARKERS[g],
            color="none",
            markerfacecolor=FEEDSTOCK_COLORS[g],
            markeredgecolor="#1A1A1A",
            markeredgewidth=0.35,
            markersize=8.0,
            label=g,
        )
        for g in CATEGORY_ORDER
        if g in used
    ]


def plot_figure(treat: pd.DataFrame, conc: pd.DataFrame) -> None:
    rate = treat.dropna(subset=["rate_ww", "pct_reduction"]).copy()
    ph = treat.dropna(subset=["soil_ph", "pct_reduction"]).copy()

    fig = plt.figure(figsize=(13.4, 10.4))
    gs = GridSpec(
        3,
        2,
        height_ratios=[1.0, 1.05, 0.36],
        hspace=0.34,
        wspace=0.24,
        left=0.08,
        right=0.99,
        top=0.94,
        bottom=0.05,
        figure=fig,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])
    draw_scatter(
        ax_a,
        rate,
        "rate_ww",
        "A   Application rate",
        "Application rate (% w/w)",
        (-0.2, 10.8),
    )
    draw_scatter(
        ax_b,
        ph,
        "soil_ph",
        "B   Initial soil pH",
        "Initial soil pH",
        (3.6, 9.2),
    )
    xmax = max(4000.0, float(conc["conc_mg_kg"].max()) * 1.15) if len(conc) else 4000.0
    xmin = min(0.08, float(conc["conc_mg_kg"].min()) * 0.6) if len(conc) else 0.08
    draw_scatter(
        ax_c,
        conc,
        "conc_mg_kg",
        "C   Initial soil PTE concentration",
        "Initial soil PTE concentration (mg/kg)",
        (max(xmin, 0.05), xmax),
        logx=True,
    )

    ax_leg = fig.add_subplot(gs[2, :])
    ax_leg.set_axis_off()
    shown = pd.concat(
        [
            rate[["feedstock_group"]],
            ph[["feedstock_group"]],
            conc[["feedstock_group"]],
        ],
        ignore_index=True,
    )
    ax_leg.legend(
        handles=feedstock_handles(shown),
        loc="upper left",
        bbox_to_anchor=(0.00, 0.95),
        ncol=3,
        frameon=False,
        fontsize=8.5,
        handletextpad=0.40,
        columnspacing=1.15,
        labelspacing=0.55,
        title="Feedstock category",
        title_fontsize=10,
    )
    fig.savefig(OUT.with_suffix(".pdf"))
    fig.savefig(OUT.with_suffix(".png"))
    plt.close(fig)
    print(f"wrote {OUT.with_suffix('.pdf')}")
    print(f"wrote {OUT.with_suffix('.png')}")


def main() -> None:
    style()
    plant = load_plant()
    meta = load_meta()
    treat = treatment_points(plant, meta)
    conc = concentration_points(plant, meta)
    FIGDIR.mkdir(exist_ok=True)
    print(
        "rate",
        treat["rate_ww"].notna().sum(),
        "soil_ph",
        treat["soil_ph"].notna().sum(),
        "conc points",
        len(conc),
        "conc treatments",
        conc["extraction_id"].nunique() if len(conc) else 0,
    )
    if len(conc):
        print(conc.groupby("metal").size().to_string())
        print(conc["conc_mg_kg"].describe().to_string())
    plot_figure(treat, conc)


if __name__ == "__main__":
    main()
