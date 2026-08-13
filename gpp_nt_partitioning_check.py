"""
Reviewer-response robustness check: do our drought-suppression and diurnal
centroid-shift conclusions hold if FLUXNET GPP is computed with the
nighttime (NT) partitioning method (Reichstein et al. 2005) instead of the
daytime (DT) method (Lasslop et al. 2010) used in the main analysis?

gpp_percentile_sensitivity.py found the DT-vs-NT method choice is a larger
effect (median 22% |GPP difference|) than the choice of u* threshold
percentile within DT (3.5-10%) — comparable in magnitude to our reported
WUE drought-suppression effects. This script re-runs the actual
midday-suppression and diurnal-centroid-shift analysis with NT-based
FLUXNET GPP/WUE, to check whether conclusions (direction and significance)
survive the switch, rather than just reasoning about it from a
percent-difference number.

ECOCO3 is untouched (GPP partitioning is a FLUXNET/eddy-covariance concept;
ECOCO3's GPP comes from OCO-3 SIF). Only the FLUXNET side is recomputed.

Approach: rather than re-deriving the full FLUXNET pipeline from raw data
(which would also require re-deriving Season/SPEI/SPI/site metadata), this
starts from the already-QC'd, already-merged DT-based half-hourly CSV
(WUE_combined_halfhourly_shuttle.csv) — every column in it except GPP/WUE
is independent of partitioning method — and merges in just GPP_NT_VUT_USTAR50
and TA_F (needed to recompute WUE) from the raw per-site files, joined on
(Site, TIMESTAMP). Rows where NT partitioning failed (-9999) even though DT
succeeded are dropped.

The category lists (valid_veg, valid_kg) and the ECOCO3-side data are kept
identical to the main DT-based analysis (01_Analysis_V6.ipynb) so this is a
clean single-variable (DT vs NT) comparison, not confounded by a different
category selection. Filtering matches 01_Analysis_V6.ipynb cell 8 (categories
with sufficient data in BOTH vegetation and climate dimensions).

Outputs (parallel to the main notebook's V2 tables, with an _NT suffix):
  tables/suppression_summary_midday_NT_V2.csv
  tables/suppression_summary_midday_kg_NT_V2.csv
  tables/centroid_shift_summary_NT_V2.csv
  tables/centroid_shift_summary_kg_NT_V2.csv
  tables/dt_vs_nt_comparison.csv    side-by-side FLUXNET % change, DT vs NT
  tables/dt_vs_nt_summary.txt       cross-category summary (also printed)
  tables/dt_vs_nt_comparison.csv   (side-by-side FLUXNET % change, DT vs NT)
"""

import glob
import os

import numpy as np
import pandas as pd

import analysis_functions as af
import plot_scripts

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_FLUXNET_DIR = (
    "/Users/zoepierratucsb/Library/Mobile Documents/com~apple~CloudDocs/"
    "Documents/UCSB/ECOCO/data/FLUXNET_Data_Shuttle"
)

MIN_SCENES = 500
MIN_SITE_YEARS = 3


def find_site_folder(site, raw_dir):
    for f in os.listdir(raw_dir):
        parts = f.split("_")
        if len(parts) >= 2 and parts[1] == site:
            return f
    return None


def build_nt_fluxnet_hh():
    """DT-based half-hourly FLUXNET frame, with GPP/WUE swapped to NT."""
    df = pd.read_csv("data/FLUXNET_Data_Shuttle/WUE_combined_halfhourly_shuttle.csv")
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])

    sites = sorted(df["Site"].unique())
    nt_parts = []
    n_missing = 0
    n_missing_cols = 0
    needed = ["TIMESTAMP_START", "GPP_NT_VUT_USTAR50", "TA_F"]
    for site in sites:
        folder = find_site_folder(site, RAW_FLUXNET_DIR)
        matches = glob.glob(os.path.join(RAW_FLUXNET_DIR, folder or "", "*FLUXMET_HH*.csv")) if folder else []
        if not matches:
            n_missing += 1
            continue
        header_cols = pd.read_csv(matches[0], nrows=0).columns
        if not all(c in header_cols for c in needed):
            n_missing_cols += 1
            continue
        raw = pd.read_csv(matches[0], usecols=needed, low_memory=False)
        raw["TIMESTAMP"] = pd.to_datetime(raw["TIMESTAMP_START"], format="%Y%m%d%H%M")
        raw["Site"] = site
        raw = raw[raw["GPP_NT_VUT_USTAR50"] != -9999]
        nt_parts.append(raw[["Site", "TIMESTAMP", "GPP_NT_VUT_USTAR50", "TA_F"]])

    print(f"  Sites missing raw HH file: {n_missing}/{len(sites)}")
    print(f"  Sites missing needed HH columns: {n_missing_cols}/{len(sites)}")
    nt = pd.concat(nt_parts, ignore_index=True)

    merged = df.merge(nt, on=["Site", "TIMESTAMP"], how="inner")
    print(f"  Rows before NT merge: {len(df):,}  |  after (NT partitioning available): {len(merged):,}")

    # Same WUE formula as 00a_FLUXNET_datashuttle_preprocess.ipynb, with GPP_NT in place of GPP_DT.
    merged["WUE"] = (
        merged["GPP_NT_VUT_USTAR50"] * 12.011 * 10**-6 * 1800
        / (merged["LE_CORR"] / ((2.501 - 0.00237 * merged["TA_F"]) * 10**6) * 1800)
    )
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna(subset=["WUE"])

    merged = merged.rename(columns={"GPP_NT_VUT_USTAR50": "GPP", "LE_CORR": "ET", "Long": "Lon"})
    merged["Hour"] = merged["TIMESTAMP"].dt.hour
    return merged


def load_ecoco3_hh():
    """Same loading/filtering as 01_Analysis_V6.ipynb cell 1 — unaffected by
    FLUXNET's GPP partitioning method, reproduced here since this script
    runs standalone rather than inside the notebook kernel."""
    df = pd.read_csv("data/ECOCO3_cleaned/ECOCO3_V2_df_wue_fullset.csv")
    df["LocalTime"] = pd.to_datetime(df["LocalTime"], format="%Y-%m-%d %H:%M:%S")
    df = df[df["IGBP_class_MODIS"] == df["IGBP_class_VIIRS"]]
    df = df.rename(columns={"IGBP_class_MODIS": "Veg"})
    df["WUE"] = df["WUE_gCkgH20"]
    df = df[df["ET"] > 50]
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df["TIMESTAMP"] = df["LocalTime"]
    df["Site"] = df["SiteName"]
    df["Hour"] = df["TIMESTAMP"].dt.hour
    return df


def run_drought_loop(df_ecoco_summer, df_flux_summer, categories, group_col, bins, label_suffix, index_name):
    """Mirrors cells 16/17 of 01_Analysis_V6.ipynb."""
    all_records, supp_records, shift_records = [], [], []

    for var in categories:
        df_ecoco_var = df_ecoco_summer[df_ecoco_summer[group_col] == var]
        df_flux_var = df_flux_summer[df_flux_summer[group_col] == var]
        if df_ecoco_var.empty or df_flux_var.empty:
            continue

        df_ecoco_centroids = plot_scripts.plot_diurnal_cycles_spei(
            df_ecoco_var, bins, title=f"Diurnal WUE by {group_col} (ECOCO) - {var} - {label_suffix}",
            index_name=index_name,
        )
        df_flux_centroids = plot_scripts.plot_diurnal_cycles_spei(
            df_flux_var, bins, title=f"Diurnal WUE by {group_col} (FLUXNET-NT) - {var} - {label_suffix}",
            index_name=index_name,
        )
        if len(df_ecoco_centroids) == 0 or len(df_flux_centroids) == 0:
            continue

        df_ecoco_centroids[group_col], df_ecoco_centroids["dataset"] = var, "ECOCO"
        df_flux_centroids[group_col], df_flux_centroids["dataset"] = var, "FLUXNET"
        all_records += [df_ecoco_centroids, df_flux_centroids]

        bin_col = f"{index_name} Bin"
        for df_c in (df_ecoco_centroids, df_flux_centroids):
            supp_records.append(df_c[df_c[bin_col] == "Midday drought - non-drought"].copy())
            shift_records.append(df_c[df_c[bin_col] == "Centroid shift (Drought - Non-Drought)"].copy())

    df_centroids = pd.concat(all_records, ignore_index=True)
    df_supp = pd.concat(supp_records, ignore_index=True)
    df_shift = pd.concat(shift_records, ignore_index=True)

    df_pivot_supp = df_supp.pivot_table(
        index=["Variable", group_col], columns="dataset",
        values=["Midday Δ", "CI 2.5%", "CI 97.5%", "% Change", "N drought days", "N non-drought days"],
        aggfunc="mean",
    ).reset_index().round(2)

    df_shift["dataset"] = pd.Categorical(df_shift["dataset"], categories=["ECOCO", "FLUXNET"])
    df_pivot_shift = df_shift.pivot_table(
        index=["Variable", group_col], columns="dataset",
        values=["Centroid Shift (hrs)", "Shift CI 2.5%", "Shift CI 97.5%"],
        aggfunc="mean", dropna=False, observed=False,
    ).reset_index().round(2)

    return df_pivot_supp, df_pivot_shift


def compare_dt_nt(dt_path, nt_supp, group_col):
    """Side-by-side FLUXNET % change, DT (already-published table) vs NT
    (just computed), plus whether 'CI excludes 0' significance flips."""
    dt = pd.read_csv(dt_path, header=[0, 1])
    dt.columns = [
        "Variable", group_col, "pct_ECOCO", "pct_FLUX", "ci25_ECOCO", "ci25_FLUX",
        "ci975_ECOCO", "ci975_FLUX", "delta_ECOCO", "delta_FLUX", "nd_ECOCO", "nd_FLUX",
        "nnd_ECOCO", "nnd_FLUX",
    ]
    dt_flux = dt[["Variable", group_col, "pct_FLUX", "ci25_FLUX", "ci975_FLUX"]].copy()
    dt_flux["sig_DT"] = (dt_flux["ci25_FLUX"] > 0) | (dt_flux["ci975_FLUX"] < 0)
    dt_flux = dt_flux.rename(columns={"pct_FLUX": "pct_change_DT"})[
        ["Variable", group_col, "pct_change_DT", "sig_DT"]
    ]

    nt = nt_supp.copy()
    nt.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c for c in nt.columns]
    nt_flux = nt[["Variable", group_col, "% Change_FLUXNET", "CI 2.5%_FLUXNET", "CI 97.5%_FLUXNET"]].copy()
    nt_flux["sig_NT"] = (nt_flux["CI 2.5%_FLUXNET"] > 0) | (nt_flux["CI 97.5%_FLUXNET"] < 0)
    nt_flux = nt_flux.rename(columns={"% Change_FLUXNET": "pct_change_NT"})[
        ["Variable", group_col, "pct_change_NT", "sig_NT"]
    ]

    merged = dt_flux.merge(nt_flux, on=["Variable", group_col], how="outer")
    merged["same_direction"] = np.sign(merged["pct_change_DT"]) == np.sign(merged["pct_change_NT"])
    merged["same_significance"] = merged["sig_DT"] == merged["sig_NT"]
    return merged


def main():
    print("Building NT-based FLUXNET half-hourly dataframe...")
    df_flux_nt = build_nt_fluxnet_hh()

    print("Loading ECOCO3 half-hourly data...")
    df_ecoco = load_ecoco3_hh()

    print("Recomputing valid_veg / valid_kg (same thresholds as the main notebook, DT-based FLUXNET for category eligibility)...")
    df_flux_dt = pd.read_csv("data/FLUXNET_Data_Shuttle/WUE_combined_halfhourly_shuttle.csv")
    df_flux_dt["TIMESTAMP"] = pd.to_datetime(df_flux_dt["TIMESTAMP"])
    eco_veg_stats = af.category_stats(df_ecoco, "Veg")
    flux_veg_stats = af.category_stats(df_flux_dt, "Veg")
    eco_kg_stats = af.category_stats(df_ecoco, "kg_label")
    flux_kg_stats = af.category_stats(df_flux_dt, "kg_label")
    valid_veg = af.filter_valid(eco_veg_stats, flux_veg_stats, MIN_SCENES, MIN_SITE_YEARS)
    valid_kg = af.filter_valid(eco_kg_stats, flux_kg_stats, MIN_SCENES, MIN_SITE_YEARS)
    print(f"  valid_veg ({len(valid_veg)}): {valid_veg}")
    print(f"  valid_kg  ({len(valid_kg)}): {valid_kg}")

    # Restrict to categories with sufficient data in BOTH dimensions, matching
    # the (now fixed) filtering in 01_Analysis_V6.ipynb cell 8.
    df_flux = df_flux_nt[df_flux_nt["Veg"].isin(valid_veg) & df_flux_nt["kg_label"].isin(valid_kg)]
    df_ecoco_f = df_ecoco[df_ecoco["Veg"].isin(valid_veg) & df_ecoco["kg_label"].isin(valid_kg)]

    df_flux_summer = df_flux[df_flux["Season"] == "Summer"]
    df_ecoco_summer = df_ecoco_f[df_ecoco_f["Season"] == "Summer"]

    spei_bins = [
        ("Drought", lambda df: df["SPEI"] < -1.5),
        ("Non-Drought", lambda df: df["SPEI"] > 0),
    ]

    print("\nRunning by-vegetation drought analysis (NT)...")
    supp_veg_nt, shift_veg_nt = run_drought_loop(
        df_ecoco_summer, df_flux_summer, valid_veg, "Veg", spei_bins, "NT", "SPEI"
    )
    supp_veg_nt.to_csv("tables/suppression_summary_midday_NT_V2.csv", index=False)
    shift_veg_nt.to_csv("tables/centroid_shift_summary_NT_V2.csv", index=False)

    print("Running by-climate drought analysis (NT)...")
    valid_kg_eco = af.sufficient_drought_days(df_ecoco_summer, "kg_label", spei_bins)
    supp_kg_nt, shift_kg_nt = run_drought_loop(
        df_ecoco_summer, df_flux_summer, valid_kg_eco, "kg_label", spei_bins, "NT", "SPEI"
    )
    supp_kg_nt.to_csv("tables/suppression_summary_midday_kg_NT_V2.csv", index=False)
    shift_kg_nt.to_csv("tables/centroid_shift_summary_kg_NT_V2.csv", index=False)

    print("\nComparing against the already-published DT tables...")
    cmp_veg = compare_dt_nt("tables/suppression_summary_midday_V2.csv", supp_veg_nt, "Veg")
    cmp_kg = compare_dt_nt("tables/suppression_summary_midday_kg_V2.csv", supp_kg_nt, "kg_label")
    cmp_veg["group_type"] = "Veg"
    cmp_kg["group_type"] = "kg_label"
    comparison = pd.concat([cmp_veg, cmp_kg.rename(columns={"kg_label": "Veg"})], ignore_index=True)
    comparison.to_csv("tables/dt_vs_nt_comparison.csv", index=False)

    wue_cmp = comparison[comparison["Variable"] == "WUE"].dropna(subset=["pct_change_DT", "pct_change_NT"])
    n_flip_dir = (~wue_cmp["same_direction"]).sum()
    n_flip_sig = (~wue_cmp["same_significance"]).sum()

    lines = []
    lines.append("=== DT vs NT: WUE midday drought-suppression % change ===")
    lines.append(
        wue_cmp[["group_type", "Veg", "pct_change_DT", "pct_change_NT", "same_direction", "same_significance"]]
        .to_string(index=False)
    )
    lines.append(f"\nCategories where WUE effect direction flips DT->NT: {n_flip_dir}/{len(wue_cmp)}")
    lines.append(f"Categories where significance (CI excludes 0) flips DT->NT: {n_flip_sig}/{len(wue_cmp)}")
    flipped = wue_cmp[~wue_cmp["same_direction"] | ~wue_cmp["same_significance"]]
    if len(flipped):
        lines.append("\nCategories that flip (direction and/or significance):")
        lines.append(
            flipped[["group_type", "Veg", "pct_change_DT", "pct_change_NT", "same_direction", "same_significance"]]
            .to_string(index=False)
        )
    lines.append(f"\nFull comparison (all variables, not just WUE): tables/dt_vs_nt_comparison.csv")

    summary = "\n".join(lines)
    print("\n" + summary)
    with open("tables/dt_vs_nt_summary.txt", "w") as f:
        f.write(summary + "\n")
    print("\nSummary written to tables/dt_vs_nt_summary.txt")


if __name__ == "__main__":
    main()
