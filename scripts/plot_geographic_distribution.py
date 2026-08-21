"""
Geographic distribution: unique included studies by country.

World choropleth. Counts are study-level (one row per study_id).
A study listing more than one country is counted once in each listed country.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as patheffects
import pandas as pd
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "ANALYSIS SET.csv"
FIGDIR = ROOT / "figures"
CACHE = ROOT / "scripts" / "_cache"
OUT = FIGDIR / "geographic_distribution"

NE_URLS = [
    "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip",
    "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip",
]

ISO_TO_COUNTRY = {
    "CHN": "China",
    "PAK": "Pakistan",
    "IRN": "Iran",
    "KOR": "South Korea",
    "TUR": "Turkey",
    "POL": "Poland",
    "BRA": "Brazil",
    "IND": "India",
    "DNK": "Denmark",
    "GBR": "United Kingdom",
    "CAN": "Canada",
    "VNM": "Vietnam",
    "AUT": "Austria",
}

# lon, lat nudges for crowded labels (degrees, before projection)
LABEL_NUDGE = {
    "Denmark": (10, 6),
    "United Kingdom": (-16, 7),
    "Austria": (9, -5),
    "Poland": (8, 3),
    "Vietnam": (12, 0),
    "South Korea": (13, 1),
    "Turkey": (2, -1),
    "Iran": (0, 0),
    "Pakistan": (0, -1),
    "India": (2, 0),
    "China": (0, 0),
    "Brazil": (0, 0),
    "Canada": (0, 0),
}

EMPTY = "#E8EBE6"
OCEAN = "#7EB6D4"
BINS = [0.5, 1.5, 2.5, 4.5, 11.5, 50]
BIN_LABELS = ["1", "2", "3 to 4", "5 to 11", "12 or more"]
BIN_COLORS = ["#C8E0C4", "#8FCB8C", "#5BA86D", "#2E8B57", "#145C39"]
LABEL_COLOR = "#1A1A1A"


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.10,
        }
    )


def study_counts_by_country(df: pd.DataFrame) -> pd.DataFrame:
    studies = df.drop_duplicates(subset="study_id")[["study_id", "country"]].copy()
    rows = []
    for _, rec in studies.iterrows():
        for country in [c.strip() for c in rec["country"].split(";") if c.strip()]:
            rows.append({"study_id": rec["study_id"], "country": country})
    long = pd.DataFrame(rows)
    counts = long.groupby("country", as_index=False).size()
    counts = counts.rename(columns={"size": "n_studies"})
    return counts, int(studies["study_id"].nunique())


def load_world() -> gpd.GeoDataFrame:
    shp_dir = CACHE / "ne_110m_admin_0_countries"
    matches = list(shp_dir.glob("*.shp")) if shp_dir.exists() else []
    if matches:
        return gpd.read_file(matches[0])

    CACHE.mkdir(parents=True, exist_ok=True)
    last_err = None
    for url in NE_URLS:
        try:
            req = Request(url, headers={"User-Agent": "nelson-review-map/1.0"})
            with urlopen(req, timeout=60) as resp:
                raw = resp.read()
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                zf.extractall(shp_dir)
            matches = list(shp_dir.glob("*.shp"))
            if not matches:
                raise FileNotFoundError("shapefile missing after extract")
            return gpd.read_file(matches[0])
        except Exception as err:
            last_err = err
    raise RuntimeError(f"Could not download Natural Earth countries: {last_err}")


def prepare_geodata(world: gpd.GeoDataFrame, counts: pd.DataFrame) -> gpd.GeoDataFrame:
    iso_col = "ISO_A3" if "ISO_A3" in world.columns else "iso_a3"
    name_col = "NAME" if "NAME" in world.columns else "name"
    world = world.copy()
    world["iso"] = world[iso_col].astype(str)
    # Natural Earth codes France/Norway sometimes as -99; keep name fallback
    world["country"] = world["iso"].map(ISO_TO_COUNTRY)
    if world["country"].isna().any():
        name_map = {v: v for v in ISO_TO_COUNTRY.values()}
        name_map.update(
            {
                "Republic of Korea": "South Korea",
                "Viet Nam": "Vietnam",
                "Iran (Islamic Republic of)": "Iran",
            }
        )
        world.loc[world["country"].isna(), "country"] = world.loc[
            world["country"].isna(), name_col
        ].map(name_map)

    world = world.loc[world[name_col].astype(str) != "Antarctica"].copy()
    world = world.merge(counts, on="country", how="left")
    world["n_studies"] = world["n_studies"].fillna(0).astype(int)
    world = world.to_crs("+proj=robin +lon_0=0 +datum=WGS84 +units=m +no_defs")
    return world


def label_points(world: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    labeled = world.loc[world["n_studies"] > 0, ["country", "n_studies", "geometry"]].copy()
    labeled["geometry"] = labeled.geometry.representative_point()
    labeled = labeled.set_geometry("geometry").to_crs("EPSG:4326")
    xs, ys = [], []
    for _, rec in labeled.iterrows():
        dx, dy = LABEL_NUDGE.get(rec["country"], (0, 0))
        xs.append(float(rec.geometry.x) + dx)
        ys.append(float(rec.geometry.y) + dy)
    nudged = gpd.GeoDataFrame(
        labeled.drop(columns="geometry"),
        geometry=gpd.points_from_xy(xs, ys),
        crs="EPSG:4326",
    ).to_crs(world.crs)
    return nudged


def plot_map(world: gpd.GeoDataFrame, n_studies: int) -> None:
    fig, ax = plt.subplots(figsize=(10.6, 5.8))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor(OCEAN)
    ax.patch.set_visible(True)
    ax.patch.set_zorder(0)

    world.loc[world["n_studies"] == 0].plot(
        ax=ax,
        color=EMPTY,
        edgecolor="#FFFFFF",
        linewidth=0.25,
        zorder=1,
    )

    cmap = mcolors.ListedColormap(BIN_COLORS)
    norm = mcolors.BoundaryNorm(BINS, cmap.N)
    world.loc[world["n_studies"] > 0].plot(
        ax=ax,
        column="n_studies",
        cmap=cmap,
        norm=norm,
        edgecolor="#FFFFFF",
        linewidth=0.35,
        zorder=2,
    )

    labels = label_points(world)
    for _, rec in labels.iterrows():
        n = int(rec["n_studies"])
        on_dark = n >= 5
        ax.annotate(
            f"{rec['country']}\n{n}",
            xy=(rec.geometry.x, rec.geometry.y),
            ha="center",
            va="center",
            fontsize=6.6,
            fontweight="bold",
            color="#FFFFFF" if on_dark else LABEL_COLOR,
            linespacing=1.05,
            zorder=4,
            path_effects=[
                patheffects.withStroke(
                    linewidth=2.0,
                    foreground="#123D28" if on_dark else "#FFFFFF",
                )
            ],
        )

    handles = [Patch(facecolor=EMPTY, edgecolor="#D0D0D0", label="None")]
    handles += [
        Patch(facecolor=c, edgecolor="#FFFFFF", label=lab)
        for c, lab in zip(BIN_COLORS, BIN_LABELS)
    ]
    legend = ax.legend(
        handles=handles,
        title="Unique studies",
        loc="lower left",
        frameon=True,
        fontsize=8,
        title_fontsize=8.5,
        borderpad=0.45,
        labelspacing=0.35,
        fancybox=False,
        edgecolor="#D5D5D5",
        facecolor="#FFFFFF",
        framealpha=0.92,
    )
    legend.get_frame().set_linewidth(0.5)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor(OCEAN)
    ax.patch.set_visible(True)
    ax.set_title(
        f"Geographic distribution of included studies  (n = {n_studies})",
        loc="left",
        fontsize=12,
        fontweight="bold",
        color=LABEL_COLOR,
        pad=8,
    )
    ax.text(
        0.0,
        -0.04,
        "One study was conducted in Denmark and the United Kingdom and is counted in both countries.",
        transform=ax.transAxes,
        fontsize=7.2,
        color="#555555",
        ha="left",
        va="top",
    )

    minx, miny, maxx, maxy = world.total_bounds
    ax.set_xlim(minx - 2e5, maxx + 2e5)
    ax.set_ylim(miny - 1e5, maxy + 8e5)

    FIGDIR.mkdir(exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"))
    fig.savefig(OUT.with_suffix(".png"))
    plt.close(fig)
    print(f"wrote {OUT.with_suffix('.pdf')}")
    print(f"wrote {OUT.with_suffix('.png')}")


def main() -> None:
    style()
    df = pd.read_csv(DATA, dtype=str, keep_default_na=False)
    counts, n_studies = study_counts_by_country(df)
    world = prepare_geodata(load_world(), counts)
    plot_map(world, n_studies)
    print(counts.sort_values("n_studies", ascending=False).to_string(index=False))
    print(f"unique studies={n_studies}")
    print(f"country-study assignments={int(counts['n_studies'].sum())}")


if __name__ == "__main__":
    main()
