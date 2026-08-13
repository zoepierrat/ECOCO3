"""
Reviewer-response analysis: how much does the choice of FLUXNET GPP product
affect our results?

FLUXNET2015/ONEFlux computes GPP under two partitioning methods (daytime,
DT — Lasslop et al. 2010; nighttime, NT — Reichstein et al. 2005), each at
seven percentiles of the bootstrapped u* threshold distribution (05, 16, 25,
50/USTAR50, 75, 84, 95). Our analysis uses GPP_DT_VUT_USTAR50 (the DT-method
median). This script quantifies, across every site actually used in the
paper, how much GPP would differ under the other percentiles and under NT
instead of DT — the two questions raised by the reviewer.

Since WUE is directly proportional to GPP in our calculation (all other
terms depend only on LE_CORR and TA_F), the percentage differences computed
here for GPP apply identically to WUE.

Requires the raw per-site FLUXNET FULLSET CSVs (the DD file's
GPP_DT/NT_VUT_* columns) — these were removed from the project's local
data/FLUXNET_Data_Shuttle/ after preprocessing (see config.GIT_IGNORE_PATTERNS)
but a full copy survives at RAW_FLUXNET_DIR below. Only the combined,
already-QC'd WUE_combined_*.csv files are tracked as pipeline outputs; this
script re-derives its own QC'd subset directly from the raw files to access
the percentile columns that don't survive into the combined CSVs.

Output:
  tables/gpp_percentile_sensitivity_by_site.csv    one row per site
  tables/gpp_percentile_sensitivity_summary.txt    cross-site summary (also printed)
"""

import glob
import os

import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_FLUXNET_DIR = (
    "/Users/zoepierratucsb/Library/Mobile Documents/com~apple~CloudDocs/"
    "Documents/UCSB/ECOCO/data/FLUXNET_Data_Shuttle"
)
METADATA_CSV = os.path.join(
    PROJECT_DIR, "data/FLUXNET_Data_Shuttle/FLUXNET_metadata_successful_daily_sites.csv"
)
OUTPUT_CSV = os.path.join(PROJECT_DIR, "tables/gpp_percentile_sensitivity_by_site.csv")
SUMMARY_TXT = os.path.join(PROJECT_DIR, "tables/gpp_percentile_sensitivity_summary.txt")

REF_COL = "GPP_DT_VUT_USTAR50"
NT_REF_COL = "GPP_NT_VUT_USTAR50"
DT_PCTS = ["05", "16", "25", "75", "84", "95"]
DT_COLS = [f"GPP_DT_VUT_{p}" for p in DT_PCTS]


def find_site_folder(site, raw_dir):
    """Raw folders are named e.g. AMF_<site>_FLUXNET_..., ICOS_<site>_..., etc."""
    for f in os.listdir(raw_dir):
        parts = f.split("_")
        if len(parts) >= 2 and parts[1] == site:
            return f
    return None


def site_sensitivity(fpath):
    """Per-site median (over QC'd, 2018+ daily rows) absolute % deviation of
    each GPP product from GPP_DT_VUT_USTAR50. Returns None if the file is
    missing required columns or has no valid rows after QC."""
    df = pd.read_csv(fpath, low_memory=False)

    qc_col = "NEE_VUT_REF_QC" if "NEE_VUT_REF_QC" in df.columns else "NEE_CUT_REF_QC"
    needed = [REF_COL, NT_REF_COL, *DT_COLS, "LE_F_MDS_QC", "LE_CORR", "TIMESTAMP", qc_col]
    if not all(c in df.columns for c in needed):
        return None

    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], format="%Y%m%d", errors="coerce")
    df = df[df["TIMESTAMP"].dt.year >= 2018]

    # Same QC criteria as 00a_FLUXNET_datashuttle_preprocess.ipynb's daily cell.
    df = df[
        (df[REF_COL] != -9999)
        & (df["LE_F_MDS_QC"] > 0.8)
        & (df[qc_col] > 0.8)
        & (df["LE_CORR"] > 5)
    ]
    for c in [REF_COL, NT_REF_COL, *DT_COLS]:
        df = df[df[c] != -9999]

    if df.empty:
        return None

    ref = df[REF_COL]
    rec = {"n_days": len(df)}
    for c, p in zip(DT_COLS, DT_PCTS):
        pct_diff = (df[c] - ref) / ref.abs() * 100
        rec[f"DT_{p}_pct_diff_median"] = pct_diff.median()
        rec[f"DT_{p}_pct_diff_abs_median"] = pct_diff.abs().median()
    nt_diff = (df[NT_REF_COL] - ref) / ref.abs() * 100
    rec["NT_vs_DT_pct_diff_median"] = nt_diff.median()
    rec["NT_vs_DT_pct_diff_abs_median"] = nt_diff.abs().median()
    return rec


def main():
    meta = pd.read_csv(METADATA_CSV)
    sites = sorted(meta["Site"].unique())

    rows = []
    n_missing_file = n_missing_cols = n_no_valid_rows = 0

    for site in sites:
        folder = find_site_folder(site, RAW_FLUXNET_DIR)
        matches = glob.glob(os.path.join(RAW_FLUXNET_DIR, folder or "", "*FLUXMET_DD*.csv")) if folder else []
        if not matches:
            n_missing_file += 1
            continue

        rec = site_sensitivity(matches[0])
        if rec is None:
            # Distinguish "missing columns" from "no valid rows" by a quick recheck.
            df = pd.read_csv(matches[0], nrows=1)
            if not all(c in df.columns for c in [REF_COL, NT_REF_COL, *DT_COLS]):
                n_missing_cols += 1
            else:
                n_no_valid_rows += 1
            continue

        rec["Site"] = site
        rows.append(rec)

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)

    lines = []
    lines.append(f"Sites in final analysis: {len(sites)}")
    lines.append(f"Sites processed OK: {len(rows)}")
    lines.append(f"Sites missing raw file: {n_missing_file}")
    lines.append(f"Sites missing needed columns: {n_missing_cols}")
    lines.append(f"Sites with no valid rows after QC: {n_no_valid_rows}")
    lines.append(f"\nPer-site results: {OUTPUT_CSV}")

    lines.append("\n=== Cross-site summary ===")
    lines.append("(median across sites of each site's median |% diff| from GPP_DT_VUT_USTAR50)")
    for p in DT_PCTS:
        col = f"DT_{p}_pct_diff_abs_median"
        lines.append(
            f"  DT_VUT_{p} vs DT_VUT_USTAR50:  median |%diff| = {out[col].median():.2f}%"
            f"   (IQR {out[col].quantile(.25):.2f}-{out[col].quantile(.75):.2f}%)"
        )

    nt_col = "NT_vs_DT_pct_diff_abs_median"
    lines.append(
        f"\n  GPP_NT_VUT_USTAR50 vs GPP_DT_VUT_USTAR50 (method effect): "
        f"median |%diff| = {out[nt_col].median():.2f}%"
        f"  (IQR {out[nt_col].quantile(.25):.2f}-{out[nt_col].quantile(.75):.2f}%)"
    )
    lines.append(
        "\nInterpretation: percentile choice within DT is a modest effect; the DT-vs-NT\n"
        "partitioning method choice is markedly larger and comparable in magnitude to our\n"
        "reported drought-suppression effect sizes (see gpp_nt_partitioning_check.py)."
    )

    summary = "\n".join(lines)
    print(summary)
    with open(SUMMARY_TXT, "w") as f:
        f.write(summary + "\n")
    print(f"\nSummary written to {SUMMARY_TXT}")


if __name__ == "__main__":
    main()
