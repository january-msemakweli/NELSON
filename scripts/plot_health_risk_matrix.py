"""
Study-level health-risk evidence matrix.

Rows are unique studies. Columns are HQ, HI, and CR. Cells are coded
from the extracted text as not reported, reported and decreased,
reported and below threshold, or reported but still above threshold.
Dietary exposure was not extracted as a separate field.

When a study has several treatments, the more conservative residual-risk
code is kept (above threshold over below, below over decreased).
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "ANALYSIS SET.csv"
FIGDIR = ROOT / "figures"
OUT = FIGDIR / "health_risk_evidence"

LABEL_COLOR = "#1A1A1A"

METRICS = [
    ("HQ", "hq"),
    ("HI", "hi"),
    ("CR", "cr"),
]

NOT_REPORTED = "Not reported"
DECREASED = "Reported, decreased"
BELOW = "Reported, below threshold"
ABOVE = "Reported, still above threshold"

RANK = {
    NOT_REPORTED: 0,
    DECREASED: 1,
    BELOW: 2,
    ABOVE: 3,
}

CODE_ORDER = [NOT_REPORTED, DECREASED, BELOW, ABOVE]
CODE_COLORS = {
    NOT_REPORTED: "#D0D0D0",
    DECREASED: "#1B7F4E",
    BELOW: "#4C72B0",
    ABOVE: "#C44E52",
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


def is_blank(text: str) -> bool:
    s = str(text).strip()
    return s == "" or s == "NR"


def extracted_numbers(text: str) -> list[float]:
    """Pull HQ/HI-scale values and scientific-notation CR values."""
    found: list[float] = []
    used: list[tuple[int, int]] = []
    for m in re.finditer(
        r"(?<![\w./])(\d+(?:\.\d+)?)\s*[x×]\s*10\s*[-−]?\s*(\d+)",
        text,
        flags=re.I,
    ):
        found.append(float(m.group(1)) * 10 ** (-int(m.group(2))))
        used.append((m.start(), m.end()))
    for m in re.finditer(r"(?<![\w./])(\d+\.\d+)(?!%)", text):
        if any(a <= m.start() < b for a, b in used):
            continue
        val = float(m.group(1))
        if val <= 100:
            found.append(val)
    return found


def classify_text(text: str, metric: str) -> str:
    if is_blank(text):
        return NOT_REPORTED
    low = text.lower()
    nums = extracted_numbers(text)

    no_arm = bool(
        re.search(
            r"not reported separately|no [a-z0-9 +-]+-specific numerical|"
            r"not explicitly (reported|tabulated)",
            low,
        )
    )
    says_decrease = bool(
        re.search(r"\breduced\b|\bdecreased\b|\blower than\b|\blowest among\b", low)
    )
    says_above = bool(
        re.search(
            r"remained above|still above|above threshold|above of 0\.001|"
            r"cd remained >\s*1|cd remained above|whereas cd remained|"
            r"children remained above|child hi remained|"
            r"above 1(?!\d)|hi >\s*1|>\s*1\.0",
            low,
        )
    )
    says_below = bool(re.search(r"below 1|<\s*1|hri < 1|values were below", low))

    if metric in {"hq", "hi"}:
        if any(v > 1.0 for v in nums if v <= 80):
            says_above = True
        elif nums and all(v < 1.0 for v in nums if v <= 80) and not says_above:
            says_below = True
    if metric == "cr":
        cr_vals = [v for v in nums if v < 1]
        if any(v >= 0.001 for v in cr_vals):
            says_above = True

    if says_above:
        return ABOVE
    if says_below:
        return BELOW
    if says_decrease:
        return DECREASED
    if no_arm:
        return NOT_REPORTED
    if re.search(r"evaluated|calculated|were also reported", low) and not nums:
        return NOT_REPORTED
    if nums:
        return DECREASED
    return NOT_REPORTED


def worse(codes: list[str]) -> str:
    return max(codes, key=lambda c: RANK[c])


def study_matrix(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for study_id, group in df.groupby("study_id", sort=False):
        rec = {
            "study_id": study_id,
            "citation": group["citation"].iloc[0],
            "year": int(group["year"].iloc[0]),
        }
        for label, col in METRICS:
            rec[label] = worse([classify_text(v, col) for v in group[col]])
        rec["any_reported"] = any(rec[label] != NOT_REPORTED for label, _col in METRICS)
        rows.append(rec)
    out = pd.DataFrame(rows)
    out = out.sort_values(
        ["any_reported", "year", "citation"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    return out


def display_matrix(studies: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    reported = studies.loc[studies["any_reported"]].copy()
    n_other = int((~studies["any_reported"]).sum())
    n_total = len(studies)
    other = {
        "study_id": "other",
        "citation": f"Others (n = {n_other})",
        "year": 9999,
        "any_reported": False,
    }
    for label, _col in METRICS:
        other[label] = NOT_REPORTED
    shown = pd.concat([reported, pd.DataFrame([other])], ignore_index=True)
    return shown, n_total, n_other


def plot_matrix(studies: pd.DataFrame) -> None:
    shown, n_total, n_other = display_matrix(studies)
    labels = [c for c, _ in METRICS]
    grid = shown[labels]
    codes = {name: i for i, name in enumerate(CODE_ORDER)}
    numeric = grid.map(lambda x: codes[x]).to_numpy(dtype=float)

    n = len(shown)
    fig_h = 5.6
    fig, ax = plt.subplots(figsize=(7.8, fig_h))
    cmap = ListedColormap([CODE_COLORS[c] for c in CODE_ORDER])
    ax.imshow(numeric, cmap=cmap, aspect="auto", vmin=0, vmax=3, interpolation="nearest")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax.xaxis.tick_top()
    ax.set_yticks(range(n))
    ax.set_yticklabels(shown["citation"], fontsize=9)
    ax.tick_params(axis="both", length=0)
    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.set_ylim(n - 0.5, -0.5)

    for y in range(n + 1):
        ax.axhline(y - 0.5, color="white", linewidth=0.8)
    for x in range(len(labels) + 1):
        ax.axvline(x - 0.5, color="white", linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_visible(False)

    n_rep = n_total - n_other
    ax.set_title(
        f"Health-risk evidence  (n = {n_total} studies, {n_rep} reporting HQ, HI, or CR)",
        loc="left",
        fontsize=11,
        fontweight="bold",
        color=LABEL_COLOR,
        pad=14,
    )
    handles = [
        Patch(facecolor=CODE_COLORS[c], edgecolor="#1A1A1A", linewidth=0.4, label=c)
        for c in CODE_ORDER
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.012),
        ncol=2,
        frameon=False,
        fontsize=8.5,
        handlelength=1.1,
        handleheight=1.1,
        columnspacing=1.4,
    )
    fig.subplots_adjust(left=0.44, right=0.98, top=0.86, bottom=0.16)
    fig.savefig(OUT.with_suffix(".pdf"))
    fig.savefig(OUT.with_suffix(".png"))
    plt.close(fig)
    print(f"wrote {OUT.with_suffix('.pdf')}")
    print(f"wrote {OUT.with_suffix('.png')}")


def main() -> None:
    style()
    df = pd.read_csv(DATA, dtype=str, keep_default_na=False)
    studies = study_matrix(df)
    FIGDIR.mkdir(exist_ok=True)
    reported = studies.loc[studies["any_reported"]]
    print(reported[["citation", "HQ", "HI", "CR"]].to_string(index=False))
    print(
        f"studies={len(studies)} reporting={int(studies['any_reported'].sum())}"
    )
    plot_matrix(studies)


if __name__ == "__main__":
    main()
