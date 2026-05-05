"""
Configuration and utilities for ECOCO3 analysis pipeline.

This module defines data paths, variable names, and configuration constants
used across the analysis notebooks and plotting functions.
"""

from pathlib import Path

# ====================================================================
# DATA PATHS
# ====================================================================

DATA_DIR = Path(__file__).parent / "data"
ECOCO3_CLEANED_DIR = DATA_DIR / "ECOCO3_cleaned"
FLUXNET_SHUTTLE_DIR = DATA_DIR / "FLUXNET_Data_Shuttle"

# Expected input data files (local, not tracked by git)
FLUXNET_HOURLY_CSV = FLUXNET_SHUTTLE_DIR / "WUE_combined_halfhourly_shuttle.csv"
FLUXNET_DAILY_CSV = FLUXNET_SHUTTLE_DIR / "WUE_combined_daily_shuttle.csv"
ECOCO3_FULLSET_CSV = ECOCO3_CLEANED_DIR / "ECOCO3_V1_df_wue_fullset.csv"
ECOCO3_DAILY_CSV = ECOCO3_CLEANED_DIR / "ECOCO3_V1_df_wue_daily_fullset.csv"

# ====================================================================
# VARIABLE DEFINITIONS
# ====================================================================

# Core measurement variables
VARIABLES = ['GPP', 'ET', 'WUE']

# Time aggregation levels
TIME_RESOLUTIONS = {
    'hourly': 'H',
    'daily': 'D',
    'seasonal': 'MS'
}

# Geographic/climate grouping columns
GROUPING_COLUMNS = {
    'vegetation': 'Veg',
    'climate': 'kg_label',
    'site': 'Site'
}

# ====================================================================
# ANALYSIS PARAMETERS
# ====================================================================

# Diurnal cycle analysis
MIDDAY_HOURS = (10, 14)  # 10:00-14:00 local time
BOOTSTRAP_N = 1000  # Number of bootstrap samples

# Seasonal smoothing
LOWESS_FRAC = 0.2  # Fraction of data for local regression

# Drought stratification (SPEI-based)
SPEI_THRESHOLDS = {
    'drought': -0.5,      # SPEI < -0.5
    'normal': 0.5,        # SPEI > 0.5
}

# ====================================================================
# NOTEBOOK PIPELINE ORDER
# ====================================================================

NOTEBOOK_PIPELINE = [
    ("00a_FLUXNET_datashuttle_preprocess.ipynb", "FLUXNET raw data → preprocessed CSV"),
    ("00b_ECOCO_preprocess_C1_V3.ipynb", "ECOCO3 raw data → cleaned CSV"),
    ("01_Analysis_V5.ipynb", "Main analysis: comparisons, plots, statistics"),
]

SUPPORTING_NOTEBOOKS = [
    ("Access_ECOCO_GES_DISC.ipynb", "Download ECOCO data from GES DISC"),
    ("Reformat_FLUXNET_Data_Shuttle.ipynb", "Reformat FLUXNET shuttle data"),
]

# ====================================================================
# PLOTTING MODULE
# ====================================================================

# 13 functions available in plot_scripts.py
PLOT_FUNCTIONS = {
    # Utilities
    'hour_to_timestamp': 'Convert hour (float) → HH:MM string',
    'get_stats': 'Compute mean & std dev by hour',
    'compute_diurnal_centroid': 'Calculate weighted time-of-day',
    
    # Comparison plots
    'plot_seasonal_cycles_comparison': '3-panel seasonal: FLUXNET vs ECOCO3',
    'plot_merged_diurnal_cycles': 'Diurnal cycles side-by-side with CI',
    'plot_diurnal_cycles_spei': 'Diurnal by drought stratification',
    'plot_diurnal_cycles_spei_comparison': 'Drought comparison: FLUXNET vs ECOCO3',
    'plot_violin_comparison_stacked': '4-panel distribution: Veg & climate',
    
    # Coverage maps
    'plot_data_coverage_map': 'Spatial + temporal coverage: scenes per location',
    'plot_data_coverage_map_sites': 'Site-level coverage summary',
    
    # Helpers (low-level)
    'plot_violin': 'Single violin plot with ANOVA significance letters',
    'get_group_letters': 'Statistical grouping from Tukey HSD',
    'apply_lowess': 'LOWESS smoothing by group',
}

# ====================================================================
# DATA QUALITY FILTERS
# ====================================================================

# Default valid values for grouping
VALID_VEGETATION = [
    'BSV', 'CRO', 'CSH', 'CVM', 'DBF', 'DNF', 'EBF', 'ENF', 
    'GRA', 'MF', 'OSH', 'SAV', 'SNO', 'URB', 'WAT', 'WET', 'WSA'
]

VALID_KOPPEN = [
    'Af', 'Am', 'Aw', 'BWh', 'BWk', 'BSh', 'BSk', 
    'Csa', 'Csb', 'Csc', 'Cwa', 'Cwb', 'Cwc', 'Cfa', 'Cfb', 'Cfc',
    'Dsa', 'Dsb', 'Dsc', 'Dsd', 'Dwa', 'Dwb', 'Dwc', 'Dwd',
    'Dfa', 'Dfb', 'Dfc', 'Dfd', 'ET', 'EF'
]

# ====================================================================
# GIT & VERSION CONTROL
# ====================================================================

GIT_IGNORE_PATTERNS = [
    "data/**/*.csv",      # All data CSVs (too large for git)
    "data/**/*.nc",       # NetCDF files
    "data/**/*.h5",       # HDF5 files
    ".DS_Store",          # macOS
    "__pycache__/",       # Python
    ".ipynb_checkpoints/", # Jupyter
]

# Last cleanup: 2026-05-05
# - Removed 531 lines of unused code from plot_scripts.py
# - Removed 2 orphaned ECOCO3_cleaned CSVs from git tracking
# - plot_scripts.py: 1828 → 1297 lines
