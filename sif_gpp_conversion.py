"""
Empirical SIF -> GPP conversion factor, fit directly against coincident
FLUXNET GPP, as an alternative to the literature vegetation-specific
factors currently used in 00b_ECOCO_preprocess_C1_V3.ipynb /
00b_ECOCO_preprocess_C2_V1.ipynb:

    conversion_factors = {'ENF': 11.72, 'DBF': 12.66, ...}  # per-IGBP-class
    df['GPP'] = df['SIF'] * conversion_factors.get(row['IGBP_MODIS'], 13.63)

Uses the same tower-pixel matching as 01_Analysis_V6.ipynb (build_tower_pixel_matches,
radius_km=15, same vegetation) and make_coincident (now extended with an
extra_eco_cols param to carry ECOCO3's raw SIF through the match, alongside
the already-converted GPP_ECO and the independently-measured GPP_FLUX).

Fits GPP_FLUX ~ SIF three ways:
  - through-origin (matches the physical model GPP = k * SIF already in use)
  - with intercept (diagnostic only, to check whether forcing through zero
    costs meaningful fit quality)
  - saturating, GPP = (m * SIF) / (a + SIF) (diagnostic only, to check
    whether the SIF->GPP relationship is meaningfully nonlinear over the
    observed SIF range)
and reports both against the current per-vegetation-factor approach's
fit (GPP_ECO, already SIF * lookup-table factor, vs GPP_FLUX) as a benchmark
for what a single global number would gain or lose.

Output:
  tables/sif_gpp_conversion_summary.txt
  tables/sif_gpp_conversion_by_veg.csv   (per-vegetation empirical factors,
                                           for context -- not what's used
                                           downstream, just to show how much
                                           variation a single number averages over)
"""

import os

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import analysis_functions as af

RADIUS_KM = 15.0


def load_data():
    df_wue_hh_fluxnet = pd.read_csv("data/FLUXNET_Data_Shuttle/WUE_combined_halfhourly_shuttle.csv")
    df_wue_hh_fluxnet["TIMESTAMP"] = pd.to_datetime(df_wue_hh_fluxnet["Local_Time_30min"], format="%Y-%m-%d %H:%M:%S")
    df_wue_hh_fluxnet = df_wue_hh_fluxnet.rename(columns={"GPP_DT_VUT_USTAR50": "GPP", "LE_CORR": "ET", "Long": "Lon"})

    df_wue_hh_ecoco3 = pd.read_csv("data/ECOCO3_cleaned/ECOCO3_V2_df_wue_fullset.csv")
    df_wue_hh_ecoco3["LocalTime"] = pd.to_datetime(df_wue_hh_ecoco3["LocalTime"], format="%Y-%m-%d %H:%M:%S")
    df_wue_hh_ecoco3 = df_wue_hh_ecoco3[df_wue_hh_ecoco3["IGBP_class_MODIS"] == df_wue_hh_ecoco3["IGBP_class_VIIRS"]]
    df_wue_hh_ecoco3 = df_wue_hh_ecoco3.rename(columns={"IGBP_class_MODIS": "Veg"})
    df_wue_hh_ecoco3["WUE"] = df_wue_hh_ecoco3["WUE_gCkgH20"]
    df_wue_hh_ecoco3 = df_wue_hh_ecoco3[df_wue_hh_ecoco3["ET"] > 50]
    df_wue_hh_ecoco3 = df_wue_hh_ecoco3.replace([np.inf, -np.inf], np.nan).dropna(subset=["GPP", "ET", "WUE", "SIF", "Veg", "Lat", "Lon"])
    df_wue_hh_ecoco3["TIMESTAMP"] = df_wue_hh_ecoco3["LocalTime"]

    return df_wue_hh_fluxnet, df_wue_hh_ecoco3


def fit_through_origin(x, y):
    """Least-squares slope for y = k*x (no intercept), plus R^2 against that model."""
    k = np.sum(x * y) / np.sum(x**2)
    resid = y - k * x
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    return k, r2


def fit_with_intercept(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    return slope, intercept, r2


def fit_saturating(x, y):
    """Michaelis-Menten-style saturating fit: y = (m*x)/(a+x).
    m = asymptotic GPP as SIF -> infinity; a = SIF value at half of that
    asymptote. Initial slope near x=0 is m/a -- directly comparable to the
    through-origin linear k, since both describe the SIF->GPP relationship
    at low SIF."""
    def model(x, m, a):
        return (m * x) / (a + x)

    p0 = [y.max(), np.median(x)]
    popt, _ = curve_fit(model, x, y, p0=p0, maxfev=10000)
    m, a = popt
    pred = model(x, m, a)
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    rmse = np.sqrt(np.mean((y - pred) ** 2))
    return m, a, r2, rmse


def main():
    print("Loading data...")
    df_wue_hh_fluxnet, df_wue_hh_ecoco3 = load_data()

    print("Matching tower/pixel pairs...")
    pixel_matches = af.build_tower_pixel_matches(
        df_wue_hh_fluxnet, df_wue_hh_ecoco3, radius_km=RADIUS_KM, lat_col="Lat", lon_col="Lon"
    )

    print("Building coincident dataset (SIF carried through)...")
    df_coincident = af.make_coincident(
        df_wue_hh_fluxnet, df_wue_hh_ecoco3, pixel_matches,
        tolerance=pd.Timedelta("0.25h"), extra_eco_cols=["SIF"],
    )
    df_coincident = df_coincident.dropna(subset=["SIF", "GPP_FLUX", "GPP_ECO"])
    df_coincident = df_coincident[(df_coincident["SIF"] > 0) & (df_coincident["GPP_FLUX"] > 0)]
    n = len(df_coincident)
    print(f"  {n:,} coincident half-hourly points with valid SIF + GPP_FLUX + GPP_ECO")

    sif = df_coincident["SIF"].values
    gpp_flux = df_coincident["GPP_FLUX"].values
    gpp_eco_current = df_coincident["GPP_ECO"].values  # already SIF * per-veg literature factor

    k_global, r2_global = fit_through_origin(sif, gpp_flux)
    slope_int, intercept_int, r2_int = fit_with_intercept(sif, gpp_flux)
    m_sat, a_sat, r2_sat, rmse_sat = fit_saturating(sif, gpp_flux)

    # Benchmark: how well does the CURRENT per-vegetation-factor GPP already
    # agree with FLUXNET GPP at these same coincident points?
    ss_res_current = np.sum((gpp_flux - gpp_eco_current) ** 2)
    ss_tot = np.sum((gpp_flux - gpp_flux.mean()) ** 2)
    r2_current = 1 - ss_res_current / ss_tot
    rmse_current = np.sqrt(np.mean((gpp_flux - gpp_eco_current) ** 2))

    gpp_global = k_global * sif
    rmse_global = np.sqrt(np.mean((gpp_flux - gpp_global) ** 2))

    # Per-vegetation empirical factors, for context on how much a single
    # global number averages over (not used downstream -- diagnostic only).
    by_veg = []
    for veg, sub in df_coincident.groupby("Veg"):
        if len(sub) < 20:
            continue
        k_v, r2_v = fit_through_origin(sub["SIF"].values, sub["GPP_FLUX"].values)
        by_veg.append({"Veg": veg, "n": len(sub), "k_empirical": k_v, "r2": r2_v})
    by_veg_df = pd.DataFrame(by_veg).sort_values("Veg")
    os.makedirs("tables", exist_ok=True)
    by_veg_df.to_csv("tables/sif_gpp_conversion_by_veg.csv", index=False)

    lines = []
    lines.append(f"Coincident half-hourly points used: {n:,}  (tower-pixel match, radius={RADIUS_KM} km, ±15 min)")
    lines.append("")
    lines.append("=== Single global empirical factor (GPP_FLUX = k * SIF, all vegetation pooled) ===")
    lines.append(f"  k (through origin) = {k_global:.3f}    R^2 = {r2_global:.3f}    RMSE = {rmse_global:.2f} umol CO2 m-2 s-1")
    lines.append(f"  slope with intercept = {slope_int:.3f}, intercept = {intercept_int:.3f}    R^2 = {r2_int:.3f}")
    lines.append("")
    lines.append("=== Saturating fit: GPP_FLUX = (m * SIF) / (a + SIF) ===")
    lines.append(f"  m (asymptote) = {m_sat:.3f} umol CO2 m-2 s-1    a (half-saturation SIF) = {a_sat:.3f}")
    lines.append(f"  initial slope m/a = {m_sat / a_sat:.3f}  (compare to linear k = {k_global:.3f})")
    lines.append(f"  R^2 = {r2_sat:.3f}    RMSE = {rmse_sat:.2f} umol CO2 m-2 s-1")
    lines.append(
        f"  vs linear-through-origin: R^2 {'improves' if r2_sat > r2_global + 0.01 else 'is essentially the same'} "
        f"({r2_global:.3f} -> {r2_sat:.3f}), at the cost of a second free parameter -- "
        f"{'worth the added complexity' if r2_sat > r2_global + 0.03 else 'likely not worth the added complexity given the small sample'}"
    )
    lines.append(f"  SIF range in this sample: {sif.min():.3f} to {sif.max():.3f} (median {np.median(sif):.3f})")
    lines.append("")
    lines.append("=== Benchmark: current per-vegetation literature factors (already in use) ===")
    lines.append(f"  R^2 = {r2_current:.3f}    RMSE = {rmse_current:.2f} umol CO2 m-2 s-1")
    lines.append("")
    lines.append("=== Per-vegetation empirical factors (context only -- not a single number) ===")
    lines.append(by_veg_df.to_string(index=False))
    lines.append("")
    lines.append(
        f"Global R^2 ({r2_global:.3f}) vs current per-veg R^2 ({r2_current:.3f}): "
        f"{'comparable' if abs(r2_global - r2_current) < 0.02 else ('WORSE' if r2_global < r2_current else 'better')}"
    )

    summary = "\n".join(lines)
    print("\n" + summary)
    with open("tables/sif_gpp_conversion_summary.txt", "w") as f:
        f.write(summary + "\n")
    print("\nSummary written to tables/sif_gpp_conversion_summary.txt")
    print("Per-vegetation table written to tables/sif_gpp_conversion_by_veg.csv")


if __name__ == "__main__":
    main()
