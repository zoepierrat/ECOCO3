# ECOCO3 Analysis Repository

This repository contains the data analysis workflow comparing ECOCO3 satellite-derived carbon
flux estimates against FLUXNET tower observations, with a focus on diurnal and seasonal WUE
patterns under drought stress.

**This `publication` branch is pruned to what's needed to reproduce the manuscript's figures**
(plus `gpp_nt_partitioning_check.py`, kept for Supplementary Text S1) — old/superseded pipeline
versions, other one-off robustness-check scripts, and exploratory notebook cells that don't feed a
published figure or the supplementary text have been removed. See `version-2` (or later) for the
full working history.

## Project Structure

```
ECOCO3/
├── README.md
├── plot_scripts.py                         # Visualization module
├── analysis_functions.py                   # Shared analysis helpers (site matching, coverage stats, FLUX-vs-ECO plot)
├── drought_utils.py                        # SPEI/SPI drought-index extraction helpers
│
├── Reformat_FLUXNET_Data_Shuttle.ipynb     # Step 0 — reformat raw FLUXNET data shuttle
├── 00a_FLUXNET_datashuttle_preprocess.ipynb  # Step 1 — FLUXNET preprocessing & QC
├── 00b_ECOCO_preprocess_C2_V1.ipynb          # Step 2 — ECOCO3 C2/V2 preprocessing & QC
├── Analysis.ipynb                            # Step 3 — main analysis; writes every manuscript figure directly
├── Access_ECOCO_GES_DISC.ipynb               # GES DISC data access helper
├── gpp_nt_partitioning_check.py              # Standalone: NT vs DT GPP partitioning sensitivity (Supplementary Text S1)
│
├── data/
│   ├── Support/                            # External reference datasets (not in git)
│   │   ├── koppen_geiger_0p1.nc            #   Köppen-Geiger climate classification
│   │   └── spei01.nc                       #   SPEI drought index
│   ├── ECOCO3_cleaned/                     # Processed ECOCO3 data (not in git)
│   ├── ECOCO3_V2/                          # Raw ECOCO3 C2/V2 scenes (not in git)
│   └── FLUXNET_Data_Shuttle/               # Processed FLUXNET data (not in git)
│
├── figures/
│   ├── diurnal_cycles_drought/             # Per-group diurnal cycle PNGs (side effect of the tables
│   │                                        #   plot_drought_suppression_summary/plot_centroid_shift_summary
│   │                                        #   read from — regenerated whenever those cells re-run)
│   └── manuscript/                         # Every published figure, written directly under its final
│       └── supplementary/                  #   manuscript number — see mapping below
└── tables/                                 # CSV summary tables consumed by the figures above
```

## Manuscript figures

`Analysis.ipynb`'s figure-producing cells save straight to `figures/manuscript/` (main text) or
`figures/manuscript/supplementary/`, under the manuscript's own figure numbers — there's no separate
working directory or copy/rename step anymore. Re-run a cell and its output updates in place.

**Main text**

| Manuscript | File | Notebook cell |
|---|---|---|
| Figure 1 | `figures/manuscript/Figure1.png` | `plot_scripts.plot_data_coverage_map(...)` |
| Figure 2 | `figures/manuscript/Figure2.png` | `af.plot_flux_vs_eco(...)` |
| Figure 3 | `figures/manuscript/Figure3.png` | `plot_scripts.plot_merged_diurnal_cycles(...)` |
| Figure 4 | `figures/manuscript/Figure4.png` | `plot_scripts.plot_diurnal_cycles_spei_comparison(...)`, SPEI |
| Figure 5 | `figures/manuscript/Figure5.png` | `plot_scripts.plot_sample_size_power_curves(...)` |

**Supplementary**

| Manuscript | File (`figures/manuscript/supplementary/`) | Notebook cell |
|---|---|---|
| Figure S3 | `FigureS3.png` | `plot_scripts.plot_data_coverage_map_sites(...)` |
| Figure S4 | `FigureS4.png` | `af.plot_flux_vs_eco_by_site_map(...)` |
| Figure S5 | `FigureS5.png` | `af.plot_group_error_summary(...)` |
| Figure S6 | `FigureS6.png` | `plot_scripts.plot_seasonal_cycles_comparison(..., group_type='Veg')` |
| Figure S7 | `FigureS7.png` | `plot_scripts.plot_seasonal_cycles_comparison(..., group_type='kg_label')` |
| Figure S8 | `FigureS8.png` | `plot_scripts.plot_seasonal_offset_summary(metric='offset')` ⚠️ see note below |
| Figure S9 | `FigureS9.png` | `plot_scripts.plot_seasonal_offset_summary(metric='pct_amp_diff')` ⚠️ see note below |
| Figure S10 | `FigureS10.png` | `plot_scripts.plot_violin_comparison_split(...)` |
| Figure S11 | `FigureS11.png` | `plot_scripts.plot_violin_comparison_stacked(...)` |
| Figure S12 | `FigureS12.png` | `plot_scripts.plot_diurnal_veg_comparison(...)` |
| Figure S13 | `FigureS13.png` | `plot_scripts.plot_diurnal_cycles_spei_comparison(...)`, SPI |
| Figure S14 | `FigureS14.png` | `plot_scripts.plot_drought_suppression_summary(...)` |
| Figure S15 | `FigureS15.png` | `plot_scripts.plot_centroid_shift_summary(...)` |
| Figure S16 | `FigureS16.png` | inline cell, "Phase-Angle Dependence by Time of Day" |
| Figure S17 | `FigureS17.png` | inline cell, "Phase-Angle Dependence by Vegetation Type" |

> ⚠️ **Reconstructed — Figures S8/S9:** these read `tables/seasonal_cycle_metrics_bootstrap_CI.csv`
> and `tables/seasonal_cycle_metrics_aggregate.csv`. The *original* code that generated those tables
> was lost before this branch existed. `plot_scripts.compute_seasonal_cycle_metrics()` (called from
> the notebook cell immediately before the `plot_seasonal_offset_summary` calls) is a from-scratch
> reconstruction — same table schema, and methodology matched to conventions already established
> elsewhere in this codebase (LOWESS-smoothed seasonal cycle via `hemisphere_adjust_doy`, a
> cluster-bootstrap by FLUXNET site / ECOCO3 pixel location, matching the "cluster-bootstrap" language
> already in `plot_seasonal_offset_summary`'s docstring). **It is not verified to reproduce the
> original numbers bit-for-bit** — re-running it will very likely shift Figures S8/S9 slightly from
> the currently-committed versions. See the reconstruction note in `plot_scripts.py` above
> `compute_seasonal_cycle_metrics` for full detail.

## Pipeline

Run notebooks in order:

1. **`Reformat_FLUXNET_Data_Shuttle.ipynb`** — reads the raw FLUXNET data shuttle snapshot, attaches Köppen-Geiger climate class, and writes `complete_fluxnet_data_shuttle_metadata_table.csv`
2. **`00a_FLUXNET_datashuttle_preprocess.ipynb`** — QC, gap-filling, WUE computation for FLUXNET; writes half-hourly and daily CSVs + metadata
3. **`00b_ECOCO_preprocess_C2_V1.ipynb`** — QC and WUE computation for ECOCO3; attaches SPEI/SPI and climate class
4. **`Analysis.ipynb`** — loads both datasets, filters to valid vegetation/climate groups, produces every figure and table listed above. Imports shared logic from `analysis_functions.py`, `drought_utils.py`, and `plot_scripts.py`.

## plot_scripts.py — Function Reference

Every function below is called from `Analysis.ipynb` (directly or as a helper of another
listed function); nothing else remains in the module on this branch.

| Function | Description |
|---|---|
| `plot_data_coverage_map()` | Spatial + temporal coverage map for ECOCO3 (Figure 1) |
| `plot_data_coverage_map_sites()` | FLUXNET site coverage map + bar charts (Figure S3) |
| `plot_diurnal_cycles_spei()` | Diurnal cycles stratified by drought / non-drought — called per vegetation/climate group while building the suppression & centroid-shift tables |
| `plot_diurnal_cycles_spei_comparison()` | FLUXNET vs ECOCO3 diurnal comparison by drought (Figures 4, S13) |
| `plot_diurnal_veg_comparison()` | Diurnal WUE by vegetation type, FLUXNET vs ECOCO3 (Figure S12) |
| `plot_merged_diurnal_cycles()` | Side-by-side diurnal curves with bootstrap CI (Figure 3) |
| `plot_sample_size_power_curves()` | Subsampling power curves, pooled across vegetation types (Figure 5) |
| `plot_seasonal_cycles_comparison()` | 3-panel seasonal cycle — FLUXNET (solid) vs ECOCO3 (dashed) (Figures S6, S7) |
| `compute_seasonal_cycle_metrics()` | Builds the tables `plot_seasonal_offset_summary` reads (Figures S8, S9 — reconstruction, see note above) |
| `plot_seasonal_offset_summary()` | Forest plot of seasonal peak-timing offset / amplitude difference (Figures S8, S9) |
| `plot_violin_comparison_stacked()` | 4-panel WUE distribution comparison, ANOVA/Tukey letters (Figure S11) |
| `plot_violin_comparison_split()` | Split-violin FLUXNET vs ECOCO3 comparison (Figure S10) |
| `plot_drought_suppression_summary()` | Forest plots of midday Δ + 95% bootstrap CI by veg & climate (Figure S14) |
| `plot_centroid_shift_summary()` | Forest plots of diurnal centroid timing shift under drought (Figure S15) |

### Helpers / dependencies

`apply_lowess`, `compute_diurnal_centroid`, `get_stats`, `get_group_letters`, `hour_to_timestamp`,
`hemisphere_adjust_doy`, `plot_violin`, `plot_violin_split`, `_load_supp`, `_load_shift`, `_sig_star`,
`_bootstrap_diurnal_centroid`, `_bootstrap_delta`, `_power_curve_centroid`, `_power_curve_delta`,
`_extend_to_actual`, `_fast_centroid`, `_repeats_for`, `_seasonal_peak_trough`, `_seasonal_cluster_ids`,
`_bootstrap_seasonal_offset_amp`

## Kept for Supplementary Text, not a manuscript figure

`gpp_nt_partitioning_check.py` — standalone script checking whether using FLUXNET's nighttime-partitioned
GPP (`GPP_NT_VUT_USTAR50`) instead of the daytime-partitioned GPP used throughout the main analysis
changes the results. Not called from `Analysis.ipynb` and doesn't produce a numbered figure, but
is the source for the manuscript's "flux-partitioning approaches did not change our results (Supplementary
Text S1)" claim, so it's kept on this branch even though it's outside the figure-pruning scope of
everything above. Run standalone: `python gpp_nt_partitioning_check.py`.

## Setup

### Virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy matplotlib seaborn geopandas cartopy statsmodels shapely \
            scipy xarray astral netCDF4
```

### Required external data (not in git)

| File | Source |
|---|---|
| `data/Support/koppen_geiger_0p1.nc` | Beck et al. 2018 Köppen-Geiger map |
| `data/Support/spei01.nc` | SPEI Global Drought Monitor |
| `data/FLUXNET_Data_Shuttle/*.csv` | FLUXNET data shuttle download |
| `data/ECOCO3_cleaned/*.csv` | ECOCO3 GES DISC download (see `Access_ECOCO_GES_DISC.ipynb`) |

All data files are excluded from version control via `.gitignore`.

## Generated outputs

After running the full pipeline:

| Path | Contents |
|---|---|
| `tables/centroid_summary_V2.csv` / `_kg_V2.csv` | Diurnal centroid hour by veg / climate class × SPEI bin |
| `tables/suppression_summary_midday_V2.csv` / `_kg_V2.csv` | Midday drought suppression by veg / climate class |
| `tables/centroid_shift_summary_V2.csv` / `_kg_V2.csv` | Diurnal centroid timing shift under drought, by veg / climate class |
| `tables/figure2_error_stats_by_group.csv` | FLUXNET-vs-ECOCO3 error stats feeding Figure S5 |
| `tables/seasonal_cycle_metrics_*.csv` | Seasonal peak-timing/amplitude tables feeding Figures S8, S9 — see reconstruction note above |
| `tables/dt_vs_nt_*` | Written by `gpp_nt_partitioning_check.py` (Supplementary Text S1), not the main notebook |
| `figures/diurnal_cycles_drought/` | Per-group diurnal cycle PNGs (not individually published; a side effect of building the suppression/shift tables) |
| `figures/manuscript/` | Every published figure, written directly under its manuscript number — see mapping above |
