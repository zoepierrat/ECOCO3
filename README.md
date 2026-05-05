# ECOCO3 Analysis Repository

This repository contains atmospheric and hydrological data analysis workflows comparing FLUXNET and ECOCO3 satellite data.

## Project Structure

```
ECOCO3/
├── .gitignore                          # Git configuration (ignores data/** files)
├── LICENSE
├── README.md                           # This file
├── plot_scripts.py                     # Visualization module (13 functions, 1297 lines)
│
├── notebooks/
│   ├── 00a_FLUXNET_datashuttle_preprocess.ipynb    # FLUXNET data prep
│   ├── 00b_ECOCO_preprocess_C1_V3.ipynb            # ECOCO3 data cleanup
│   ├── 01_Analysis_V5.ipynb                        # Main analysis & plots (uses plot_scripts)
│   ├── Access_ECOCO_GES_DISC.ipynb                 # GES DISC data access
│   └── Reformat_FLUXNET_Data_Shuttle.ipynb         # FLUXNET reformatting
│
└── data/
    ├── ECOCO3_cleaned/                 # Cleaned ECOCO3 datasets (in .gitignore)
    └── FLUXNET_Data_Shuttle/           # FLUXNET datasets (in .gitignore)
```

## Active Pipeline

The main analysis workflow is in **01_Analysis_V5.ipynb**, which uses 8 functions from `plot_scripts.py`:

1. `hour_to_timestamp()` — Convert hour float to HH:MM format
2. `plot_data_coverage_map()` — Map showing spatial + temporal data coverage
3. `plot_data_coverage_map_sites()` — FLUXNET site coverage summary
4. `plot_diurnal_cycles_spei()` — Diurnal cycles stratified by drought status
5. `plot_diurnal_cycles_spei_comparison()` — Compare FLUXNET vs ECOCO3 by drought
6. `plot_merged_diurnal_cycles()` — Side-by-side diurnal curves with bootstrap CI
7. `plot_seasonal_cycles_comparison()` — 3-panel seasonal comparison
8. `plot_violin_comparison_stacked()` — 4-panel distribution comparison

## Data Files

### In .gitignore (local only)

All data files are ignored by git and must be generated locally:

- `data/ECOCO3_cleaned/*.csv` — ECOCO3 processed datasets
- `data/FLUXNET_Data_Shuttle/*.csv` — FLUXNET data files

### Data Size Notes

The `.gitignore` was configured because:
- `WUE_combined_halfhourly_shuttle.csv` was **2.5 GB** (exceeds GitHub's 100 MB file limit)
- Other CSVs are **multi-hundred MB** (not suitable for version control)

Use the preprocessing notebooks to regenerate data locally from raw sources.

## Recent Changes

### Cleanup commit (2026-05-05)

Removed 531 lines of unused code:

- **Removed 4 unused functions**: `plot_seasonal_cycles_multi`, `plot_diurnal_wue`, `plot_merged_diurnal_wue`, `compute_centroids_from_hourly_avg`
- **Removed duplicates**: 121 duplicate function definitions
- **Final size**: 1,297 lines (was 1,828)
- **Functions retained**: 13 (8 actively used + 5 dependencies)
- **Removed from git**: 2 orphaned ECOCO3 cleaned CSVs not referenced by any notebook

## Dependencies

Key Python packages:
- pandas, numpy — Data manipulation
- matplotlib, seaborn — Plotting
- geopandas, cartopy — Geospatial visualization
- statsmodels — Statistical testing (ANOVA, Tukey HSD, LOWESS)
- shapely — Geometry operations

Install via:
```bash
pip install pandas numpy matplotlib seaborn geopandas cartopy statsmodels shapely
```

## Quick Start

1. **Prepare data**: Run preprocessing notebooks in order:
   - `00a_FLUXNET_datashuttle_preprocess.ipynb`
   - `00b_ECOCO_preprocess_C1_V3.ipynb`
   
2. **Analyze**: Open and run `01_Analysis_V5.ipynb`

3. **Visualizations**: The notebook automatically uses `plot_scripts.py` to generate figures

## Git Setup

```bash
# Verify remote
git remote -v

# Push changes
git push origin main

# Check status
git status
```

All data files are untracked (ignored by `.gitignore`). Commit only code and notebooks.
