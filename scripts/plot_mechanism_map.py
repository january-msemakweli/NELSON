"""
Mechanism evidence map.

Rows are the 13 immobilization mechanisms listed in the manuscript
outline. Panel A columns are standard feedstock classes. Panel B
columns are major PTEs. A cell is the number of unique studies that
report that mechanism for that feedstock or metal.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "ANALYSIS SET.csv"
FIGDIR = ROOT / "figures"
SCRIPTS = Path(__file__).resolve().parent
OUT = FIGDIR / "mechanism_evidence_map"

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from plot_feedstock_composition import CATEGORY_ORDER, standard_category

LABEL_COLOR = "#1A1A1A"

MECHANISMS = [
    ("Electrostatic adsorption", r"electrostatic"),
    ("Surface complexation", r"surface complexation"),
    ("Cation exchange", r"(?<!increased )(?:\bcation exchange(?! capacity)|\bion exchange\b)"),
    ("Precipitation", r"precipit"),
    ("Oxygen-containing functional-group complexation",
     r"oxygen-containing|o-containing|functional group|carboxyl|hydroxyl|phenolic"),
    ("Mineral precipitation",
     r"phosphate(?:-mediated)? precipitation|carbonate precipitation|pyromorphite|"
     r"silicate|ni\(oh\)2|mineral precipit|hydroxide precipitation|fe-cr co-precipitation|"
     r"fe/cr hydroxides"),
    ("pH-mediated immobilization", r"increased soil ph|ph increase|higher soil ph"),
    ("Increased CEC", r"increased (?:cec|cation exchange capacity)|\bcec\b"),
    ("Redox transformation",
     r"redox|cr\(vi\)|reduction of (?:toxic )?cr|reduc(?:ed|tion) of cr|"
     r"partial reduction of cr|nzvi|n-zvi|zero-valent"),
    ("Pore adsorption", r"pore filling|porous|pore adsorption|\bpores\b"),
    ("Transformation into less-labile fractions",
     r"less(?:-|\s)(?:labile|available)|residual fraction|stable (?:cd )?fractions?|"
     r"conversion of exchangeable|oxide-bound|organic matter-bound|speciation"),
    ("Reduced root uptake",
     r"reduced (?:root |plant )?uptake|reduced phytoavailability|"
     r"reduced (?:cd|cr|pb|as|ni|arsenic) uptake"),
    ("Reduced root-to-shoot translocation",
     r"translocat|transfer to (?:leaves|shoots|aerial)|reduced transfer of"),
]

PTE_ORDER = ["Cd", "Pb", "Cr", "Ni", "As", "Cu", "Zn"]

FEED_SHORT = {
    "Cereal field residue": "Cereal field",
    "Cereal processing residue": "Cereal processing",
    "Fruit and nut processing residue": "Fruit/nut",
    "Oilseed processing residue": "Oilseed",
    "Sugar-crop processing residue": "Sugar-crop",
    "Oil-palm processing residue": "Oil-palm",
    "Agricultural/root processing residue": "Ag/root",
    "Vegetable processing residue": "Vegetable",
    "Industrial/fiber crop residue": "Industrial/fiber",
    "Legume crop residue": "Legume",
    "Seed-processing residue": "Seed-processing",
    "Mixed postharvest agricultural residues": "Mixed postharvest",
    "Mixed eligible and non-eligible feedstocks": "Mixed eligible/non",
}

CMAP = LinearSegmentedColormap.from_list(
    "nelson_green",
    ["#F4F4F4", "#C6DCCB", "#7FB892", "#3A9A68", "#1B7F4E", "#145C39"],
)


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


def classify(text: str) -> list[str]:
    low = str(text).lower()
    if not low or low == "nr":
        return []
    hits = []
    for label, pat in MECHANISMS:
        if re.search(pat, low, flags=re.I):
            hits.append(label)
    return hits


def parse_ptes(targets: str) -> list[str]:
    out = []
    for part in str(targets).split(";"):
        metal = part.strip()
        if metal == "Cr(VI)":
            metal = "Cr"
        if metal in PTE_ORDER and metal not in out:
            out.append(metal)
    return out


def study_pairs(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    feed_pairs = set()
    pte_pairs = set()
    for _, rec in df.iterrows():
        mechs = classify(rec["immobilization_mechanism"])
        if not mechs:
            continue
        feed = standard_category(rec["specific_feedstock"], rec["feedstock_category"])
        ptes = parse_ptes(rec["target_metals"])
        study = rec["study_id"]
        for mech in mechs:
            feed_pairs.add((study, mech, feed))
            for metal in ptes:
                pte_pairs.add((study, mech, metal))

    mech_labels = [m for m, _p in MECHANISMS]
    feed = pd.DataFrame(0, index=mech_labels, columns=CATEGORY_ORDER, dtype=int)
    pte = pd.DataFrame(0, index=mech_labels, columns=PTE_ORDER, dtype=int)
    for _study, mech, cat in feed_pairs:
        if cat in feed.columns:
            feed.loc[mech, cat] += 1
    for _study, mech, metal in pte_pairs:
        pte.loc[mech, metal] += 1
    return feed, pte, int(df["study_id"].nunique())


def draw_heat(ax, mat: pd.DataFrame, title: str, col_labels: list[str], vmax: int, show_ylabels: bool) -> None:
    data = mat.to_numpy(dtype=float)
    im = ax.imshow(data, cmap=CMAP, aspect="auto", vmin=0, vmax=vmax, interpolation="nearest")
    n_r, n_c = data.shape
    ax.set_xticks(range(n_c))
    ax.set_xticklabels(col_labels, fontsize=8, rotation=40, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(n_r))
    if show_ylabels:
        ax.set_yticklabels(mat.index.tolist(), fontsize=8.5)
    else:
        ax.set_yticklabels([])
    ax.tick_params(axis="both", length=0)
    ax.set_xlim(-0.5, n_c - 0.5)
    ax.set_ylim(n_r - 0.5, -0.5)
    for y in range(n_r + 1):
        ax.axhline(y - 0.5, color="white", linewidth=0.8)
    for x in range(n_c + 1):
        ax.axvline(x - 0.5, color="white", linewidth=0.8)
    for i in range(n_r):
        for j in range(n_c):
            val = int(data[i, j])
            if val == 0:
                continue
            color = "white" if val >= 0.62 * vmax else LABEL_COLOR
            ax.text(j, i, str(val), ha="center", va="center", fontsize=8, color=color)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=LABEL_COLOR, pad=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def plot_maps(feed: pd.DataFrame, pte: pd.DataFrame, n_studies: int) -> None:
    vmax = int(max(feed.to_numpy().max(), pte.to_numpy().max(), 1))
    n_row = len(feed.index)
    n_feed = len(CATEGORY_ORDER)

    fig_w, fig_h = 13.8, 12.2
    left = 0.30
    top = 0.94
    row_h = 0.028
    col_w = 0.046
    gap = 0.12
    cbar_w = 0.018
    cbar_pad = 0.018

    a_h = n_row * row_h
    a_w = n_feed * col_w
    a_y = top - a_h
    b_y = a_y - gap - a_h

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax_a = fig.add_axes([left, a_y, a_w, a_h])
    ax_b = fig.add_axes([left, b_y, a_w, a_h])
    cax = fig.add_axes([left + a_w + cbar_pad, b_y, cbar_w, a_y + a_h - b_y])
    im = draw_heat(
        ax_a,
        feed,
        f"A   Feedstock category  (n = {n_studies} studies)",
        [FEED_SHORT[c] for c in CATEGORY_ORDER],
        vmax,
        show_ylabels=True,
    )
    draw_heat(
        ax_b,
        pte,
        "B   Major PTEs",
        PTE_ORDER,
        vmax,
        show_ylabels=True,
    )
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Number of studies", fontsize=10)
    cb.ax.tick_params(labelsize=8.5, length=3.2, width=0.6)
    cb.outline.set_visible(False)
    fig.savefig(OUT.with_suffix(".pdf"))
    fig.savefig(OUT.with_suffix(".png"))
    plt.close(fig)
    print(f"wrote {OUT.with_suffix('.pdf')}")
    print(f"wrote {OUT.with_suffix('.png')}")


def main() -> None:
    style()
    df = pd.read_csv(DATA, dtype=str, keep_default_na=False)
    feed, pte, n_studies = study_pairs(df)
    FIGDIR.mkdir(exist_ok=True)
    print(feed.sum(axis=1).to_string())
    print()
    print(pte.to_string())
    plot_maps(feed, pte, n_studies)


if __name__ == "__main__":
    main()
