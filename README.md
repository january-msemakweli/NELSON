# Postharvest agricultural waste-derived biochar for immobilization of potentially toxic elements in contaminated agricultural soils: Implications for leafy vegetable safety and human health. A systematic review

Analysis files and figure scripts for this systematic review.

Analyst: January G. Msemakweli

Figures are generated from the standardized analysis set.

## Data

| File | Role |
|---|---|
| `ANALYSIS SET.csv` | Standardized analysis table (100 treatment-level rows) |
| `ANALYSIS SET codebook.csv` | Column names, types, and coding rules |

**Counts**

- 100 extracted treatment records
- 63 unique included studies (`study_id` and DOI)

Most study-level figures use one row per `study_id`, not one row per extraction. A study with several biochar treatments still counts as one study.

## Folder layout

```
NELSON/
  ANALYSIS SET.csv
  ANALYSIS SET codebook.csv
  README.md
  scripts/
    build_analysis_set.py
    plot_risk_of_bias.py
    plot_publication_trends.py
    plot_geographic_distribution.py
    plot_experimental_settings.py
    plot_contamination_sources.py
    plot_target_ptes.py
    plot_leafy_vegetables.py
    plot_feedstock_composition.py
  figures/
```

`scripts/_cache/` stores the Natural Earth country shapefile used by the map. It is created on first run and is gitignored.

`Main Manuscript.docx` is gitignored.

## Rebuild the analysis set

```bash
python scripts/build_analysis_set.py
```

This writes `ANALYSIS SET.csv` and `ANALYSIS SET codebook.csv`. Headers are cleaned, missing values are coded as `NR`, synonyms are harmonized, and numeric companion columns are parsed (temperature, pH, application rate, durations in days, and others). See the codebook for definitions.

## Figures

Run from the project root. Each script writes PDF and PNG files to `figures/`.

```bash
python scripts/plot_risk_of_bias.py
python scripts/plot_publication_trends.py
python scripts/plot_geographic_distribution.py
python scripts/plot_experimental_settings.py
python scripts/plot_contamination_sources.py
python scripts/plot_target_ptes.py
python scripts/plot_leafy_vegetables.py
python scripts/plot_feedstock_composition.py
```

| Script | Output | What it shows |
|---|---|---|
| `plot_risk_of_bias.py` | `risk_of_bias_traffic_light` (A+B), `risk_of_bias_A_summary`, `risk_of_bias_B_traffic_light` | Study-level risk of bias, plus standalone panels |
| `plot_publication_trends.py` | `publication_trends` | Unique studies by publication year |
| `plot_geographic_distribution.py` | `geographic_distribution` | Unique studies by country |
| `plot_experimental_settings.py` | `experimental_settings` | Greenhouse, pot, field, and other settings |
| `plot_contamination_sources.py` | `contamination_sources` | Contamination source categories |
| `plot_target_ptes.py` | `target_ptes` | Unique studies investigating each target PTE |
| `plot_leafy_vegetables.py` | `leafy_vegetables` | Unique study-species combinations |
| `plot_feedstock_composition.py` | `feedstock_composition` (A+B), `feedstock_composition_A_categories`, `feedstock_composition_B_specific` | Feedstock categories and specific wastes |

Risk-of-bias ratings are collapsed to the study. If a study has more than one extraction, the more conservative rating is kept.

The map counts a multi-country study once in each listed country (one paper in Denmark and the United Kingdom). Country totals therefore add to 64, while unique studies remain 63.

Greenhouse and pot are separated using the original extraction wording. The analysis-set `experiment_type` column had collapsed those two.

Contamination sources are mutually exclusive at the study level. Naturally contaminated soils with a stated origin are coded as mining, industrial, or wastewater irrigation. Studies with both spiked and natural arms are `Mixed/other`.

The PTE chart counts a study once for each metal it investigated, so the bars do not sum to 63.

The vegetable chart counts unique study-species combinations, not treatment rows. `Others` includes coriander, rapeseed, cabbage, crown daisy, and remaining crops not in the named list.

The feedstock figure uses unique study-category combinations in Panel A and unique study-specific feedstock combinations in Panel B. Maize stover and maize stalk are combined as corn stalk/stover.

## Requirements

Python 3, then:

```bash
pip install -r requirements.txt
```

geopandas is needed only for the map. The first map run downloads Natural Earth country boundaries.

Arial is used for figure text (available on Windows).

## License

All rights reserved. See `LICENSE`. This code is written for this review and its local files only. It may not be used for other studies or other purposes.
