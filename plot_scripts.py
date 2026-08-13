from matplotlib.colors import LogNorm
from shapely import Point
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
from scipy.stats import mannwhitneyu
from pathlib import Path
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D


# === Palettes ===
season_palette = {'Summer': '#679256', 'Winter': '#94B4BD', 'Other': 'gray'}
daytime_palette = {'Morning': '#ea6f1b', 'Midday': '#f9e740', 'Afternoon': '#1b6998', 'Other': 'black'}

veg_color_palette = {
    'BSV': '#B0C4DE', 'CRO': '#FFEC8B', 'CSH': '#AB82FF', 'CVM': '#8B814C', 'DBF': '#98FB98',
    'DNF': '#9ACD32', 'EBF': '#7FFF00', 'ENF': '#006400', 'GRA': '#FFA54F', 'MF': '#8FBC8F',
    'OSH': '#FFE4E1', 'SAV': '#FFD700', 'SNO': '#000000', 'URB': '#FF0000', 'WAT': '#98F5FF',
    'WET': '#4169E1', 'WSA': '#CDAA7D'
}

koppen_label_color_palette = {
    "Af":  "#0000FF",   # Tropical, rainforest
    "Am":  "#0078FF",   # Tropical, monsoon
    "Aw":  "#46AAFA",   # Tropical, savannah
    "BWh": "#FF0000",   # Arid, desert, hot
    "BWk": "#FF9696",   # Arid, desert, cold
    "BSh": "#F5A500",   # Arid, steppe, hot
    "BSk": "#FFDC64",   # Arid, steppe, cold
    "Csa": "#FFFF00",   # Temperate, dry summer, hot summer
    "Csb": "#C8C800",   # Temperate, dry summer, warm summer
    "Csc": "#969600",   # Temperate, dry summer, cold summer
    "Cwa": "#96FF96",   # Temperate, dry winter, hot summer
    "Cwb": "#64C864",   # Temperate, dry winter, warm summer
    "Cwc": "#329632",   # Temperate, dry winter, cold summer
    "Cfa": "#C8FF50",   # Temperate, no dry season, hot summer
    "Cfb": "#64FF50",   # Temperate, no dry season, warm summer
    "Cfc": "#32C800",   # Temperate, no dry season, cold summer
    "Dsa": "#FF00FF",   # Cold, dry summer, hot summer
    "Dsb": "#C800C8",   # Cold, dry summer, warm summer
    "Dsc": "#963296",   # Cold, dry summer, cold summer
    "Dsd": "#966496",   # Cold, dry summer, very cold winter
    "Dwa": "#AAAFDF",   # Cold, dry winter, hot summer
    "Dwb": "#5A78DC",   # Cold, dry winter, warm summer
    "Dwc": "#4B50B4",   # Cold, dry winter, cold summer
    "Dwd": "#320087",   # Cold, dry winter, very cold winter
    "Dfa": "#00FFFF",   # Cold, no dry season, hot summer
    "Dfb": "#37C8FF",   # Cold, no dry season, warm summer
    "Dfc": "#007D7D",   # Cold, no dry season, cold summer
    "Dfd": "#00465F",   # Cold, no dry season, very cold winter
    "ET":  "#B2B2B2",   # Polar, tundra
    "EF":  "#666666"    # Polar, frost
}


# =================================================================
# DEPENDENCIES FOR USED FUNCTIONS
# =================================================================

def hemisphere_adjust_doy(timestamps, lats):
    """Day-of-year, shifted by half a year for Southern Hemisphere rows so
    both hemispheres' summers land on the same nominal day (~day 180) and
    their winters land at the edges (~day 1 / day 365) -- instead of a
    Southern Hemisphere summer (real DOY ~330-60) being averaged directly
    against a Northern Hemisphere summer (real DOY ~150-210) at the same
    calendar day, which smears or cancels the true seasonal cycle for any
    category (vegetation type, climate class) that mixes hemispheres.
    """
    doy = timestamps.dt.dayofyear.to_numpy()
    days_in_year = np.where(timestamps.dt.is_leap_year.to_numpy(), 366, 365)
    is_south = (lats < 0).to_numpy()
    shift = days_in_year // 2
    adjusted = doy.copy()
    adjusted[is_south] = ((doy[is_south] + shift[is_south] - 1) % days_in_year[is_south]) + 1
    return pd.Series(adjusted, index=timestamps.index)


def apply_lowess(group, variable, frac=0.2):
    """Apply LOWESS smoothing to a variable per group, ensuring DOY is sorted."""
    group = group.sort_values('DOY')
    smoothed = sm.nonparametric.lowess(
        endog=group[variable],
        exog=group['DOY'],
        frac=frac,
        return_sorted=False
    )
    group[f'{variable}_smooth'] = smoothed
    return group


def get_group_letters(tukey_result):
    res_df = pd.DataFrame(
        tukey_result._results_table.data[1:],
        columns=tukey_result._results_table.data[0]
    )
    groups = sorted(list(set(res_df['group1']).union(res_df['group2'])))
    sig = {(row['group1'], row['group2']): row['reject']
           for _, row in res_df.iterrows()}
    sig.update({(b,a): v for (a,b),v in sig.items()})
    letters = {g: '' for g in groups}
    letter_sets = []
    for g in groups:
        placed = False
        for i, s in enumerate(letter_sets):
            if all(not sig.get((g, other), False) for other in s):
                s.add(g)
                letters[g] += chr(97+i)
                placed = True
        if not placed:
            letter_sets.append(set([g]))
            letters[g] += chr(97+len(letter_sets)-1)
    return letters


def plot_violin(df_wue_daily, grouping_category, ax, title):
    if grouping_category == 'Veg':
        color_map = veg_color_palette
    elif grouping_category == 'kg_label':
        color_map = koppen_label_color_palette
    formula = f'WUE ~ C({grouping_category})'
    model = ols(formula, data=df_wue_daily).fit()
    anova_result = sm.stats.anova_lm(model, typ=2)
    mc = MultiComparison(df_wue_daily['WUE'], df_wue_daily[grouping_category])
    tukey_result = mc.tukeyhsd()
    group_letters = get_group_letters(tukey_result)
    median_wue_order = (
        df_wue_daily.groupby(grouping_category)["WUE"]
        .median()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    sns.violinplot(x=grouping_category, y='WUE', data=df_wue_daily, palette=color_map,
                    inner='quartile', order=median_wue_order ,scale='area', cut=0, ax=ax)
    # Compact letter designations, placed above the axes (axes-fraction y,
    # data-space x) rather than below in the 0-to-negative margin -- this
    # keeps them clear of the violins regardless of how the y-limit below
    # is chosen, and off the x-tick labels underneath.
    for i, vegetation in enumerate(median_wue_order):
        letter = group_letters[vegetation]
        ax.text(i, 1.03, letter, transform=ax.get_xaxis_transform(),
                ha='center', va='bottom', fontsize=21)
    # pad=30 leaves room between the title and the letters row above the axes
    ax.set_title(title, fontsize=26, pad=30)
    ax.set_xlabel(grouping_category, fontsize=21)
    ax.set_ylabel('WUE \n [gC kg$^{-1}$H$_2$O]', fontsize=21)
    # WUE is heavily right-skewed (occasional very large values when ET is
    # near zero -- daily FLUXNET WUE ranges up to 80), so this is a deliberate
    # zoomed-in view onto the bulk of the distribution, not the full range;
    # see the truncation note added by the calling figure-level function.
    ax.set_ylim(0, 6)
    ax.tick_params(axis='x', rotation=45, labelsize=21)
    ax.tick_params(axis='y', labelsize=21)


def compute_diurnal_centroid(df, hour_col='Hour', weight_col='WUE'):
    weights = df[weight_col].values
    hours = df[hour_col].values
    if np.sum(weights) == 0:
        return np.nan
    return np.sum(hours * weights) / np.sum(weights)


def hour_to_timestamp(hour_float):
    if pd.isna(hour_float):
        return np.nan
    h = int(hour_float)
    m = int((hour_float - h) * 60)
    return f"{h:02d}:{m:02d}"


def get_stats(df, var):
    avg = df.groupby('Hour')[var].mean().reset_index()
    std = df.groupby('Hour')[var].std().reset_index()
    return pd.merge(avg, std, on='Hour', suffixes=('_avg', '_std'))


# =================================================================
# USED FUNCTIONS (8 functions)
# =================================================================

def plot_seasonal_cycles_comparison(
    df_fluxnet,
    df_ecoco3,
    variables,                     # list of 3 variables (GPP, ET, WUE)
    y_labels=None,
    group_type='Veg',
    veg_color_palette=veg_color_palette,
    koppen_label_color_palette=koppen_label_color_palette,
    frac=0.2,
    ylims=[[ -0.5, 13], [0, 5], [0, 5]],
    valid_veg=None,
    valid_kg=None,
    output_path=None
):

    assert len(variables) == 3, "Please provide exactly 3 variables"

    # Copy + add hemisphere-adjusted DOY (see hemisphere_adjust_doy) -- several
    # vegetation/climate categories mix Northern and Southern Hemisphere sites
    # at very different ratios between FLUXNET and ECOCO3, so a raw calendar
    # DOY would blend opposite-phase seasonal cycles and make cross-dataset
    # comparisons at the category level unreliable.
    df_ecoco3 = df_ecoco3.copy()
    df_fluxnet = df_fluxnet.copy()
    df_ecoco3['DOY'] = hemisphere_adjust_doy(df_ecoco3['TIMESTAMP'], df_ecoco3['Lat'])
    df_fluxnet['DOY'] = hemisphere_adjust_doy(df_fluxnet['TIMESTAMP'], df_fluxnet['Lat'])

    # -----------------------------------------
    # Select grouping + color palette
    # -----------------------------------------
    if group_type == 'Veg':
        group_col = 'Veg'
        palette = veg_color_palette
        valid_groups = valid_veg
        title_suffix = 'Vegetation'
    elif group_type == 'kg_label':
        group_col = 'kg_label'
        palette = koppen_label_color_palette
        valid_groups = valid_kg
        title_suffix = 'Climate'
    else:
        raise ValueError("group_type must be 'Veg' or 'kg_label'")

    # Filter
    if valid_groups is not None:
        df_ecoco3 = df_ecoco3[df_ecoco3[group_col].isin(valid_groups)]
        df_fluxnet = df_fluxnet[df_fluxnet[group_col].isin(valid_groups)]

    # -----------------------------------------
    # Create figure (3 rows × 2 columns), share y per row
    # -----------------------------------------
    fig, axes = plt.subplots(
        nrows=3, ncols=2, figsize=(16, 16),
        sharex=True, sharey='row'  # <- share y-axis for columns
    )
    fig.suptitle(
        f"Seasonal cycles by {title_suffix}: FLUXNET vs ECOCO3",
        fontsize=36
    )

    panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    datasets = [(df_fluxnet, "FLUXNET"), (df_ecoco3, "ECOCO3")]

    label_idx = 0
    for i, variable in enumerate(variables):
        for j, (df_plot, col_title) in enumerate(datasets):
            ax = axes[i, j]

            doy_group = (
                df_plot
                .groupby(['DOY', group_col])[variable]
                .mean()
                .reset_index()
                .groupby(group_col)
                .apply(apply_lowess, variable=variable, frac=frac)
            )

            for group, d in doy_group.groupby(group_col):
                ax.plot(
                    d['DOY'],
                    d[f'{variable}_smooth'],
                    color=palette.get(group, 'gray'),
                    linewidth=7,
                    label=group
                )

            # Panel label
            ax.text(
                0.02, 0.92, panel_labels[label_idx],
                transform=ax.transAxes,
                fontsize=18,
                fontweight='bold',
                va='top',
                ha='left'
            )
            label_idx += 1

            # Titles (top row only)
            if i == 0:
                ax.set_title(col_title, fontsize=31)

            # Y labels (left column only)
            if j == 0:
                ax.set_ylabel(y_labels[i] if y_labels else variable, fontsize=29)
                ax.tick_params(axis='y', labelsize=26)

            ax.set_xlim(1, 366)
            ax.set_ylim(ylims[i])

    # X label bottom row
    for ax in axes[-1, :]:
        ax.set_xlabel('Day of Year', fontsize=29)
        ax.tick_params(axis='x', labelsize=26)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    n_legend_rows = -(-len(handles) // 6)  # ceil(n / ncol)
    fig.legend(
        handles, labels, loc='lower center', ncol=6, fontsize=21,
        frameon=False, bbox_to_anchor=(0.5, -0.03)
    )

    # Bottom margin scales with legend row count (up to 15 groups -> 3 rows
    # for kg_label; only 6-8 -> 1-2 rows for Veg), top margin reserves room
    # for the large suptitle -- both previously fixed and too small, which
    # let the legend collide with the bottom row's x-axis and the suptitle
    # collide with the top row's column titles.
    bottom_margin = 0.03 + 0.028 * (n_legend_rows - 1)
    # h_pad gives the rows breathing room -- the two-line y-labels
    # ("GPP\n[gC m-2 day-1]" etc.) were tall enough to collide vertically
    # between adjacent rows without it.
    plt.tight_layout(rect=[0, bottom_margin, 1, 0.965], h_pad=1.5)

    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    plt.show()

def plot_merged_diurnal_cycles(
    df_flux,
    df_ecoco,
    title='Flux vs ECOCO Summer Diurnal Cycles',
    midday_hours=(10, 11, 12, 13, 14),
    n_boot=1000,
    random_state=42,
    output_path=None
):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(random_state)

    flux_color = '#4DAC26'
    rs_color   = '#7B2D8B'

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig.suptitle(title, fontsize=31)

    panel_labels = ['(a)', '(b)', '(c)']

    variables = ['GPP', 'ET', 'WUE']
    y_labels = [
        'GPP \n [µmol CO$_2$ m$^{-2}$ s$^{-1}$]',
        'ET \n [W m$^{-2}$]',
        'WUE \n [gC kg$^{-1}$ H$_2$O]'
    ]

    results = []

    for i, var in enumerate(variables):

        ax = axes[i]

        # Hourly stats for plotting + centroid
        flux_stats  = get_stats(df_flux, var)
        ecoco_stats = get_stats(df_ecoco, var)

        # ---- Centroids
        flux_centroid  = compute_diurnal_centroid(flux_stats, 'Hour', f'{var}_avg')
        ecoco_centroid = compute_diurnal_centroid(ecoco_stats, 'Hour', f'{var}_avg')

        flux_time  = hour_to_timestamp(flux_centroid)
        ecoco_time = hour_to_timestamp(ecoco_centroid)

        # ---- Plot FLUX
        sns.lineplot(
            x='Hour', y=f'{var}_avg',
            data=flux_stats,
            marker='o',
            color=flux_color,
            linewidth=2,
            ax=ax,
            label=f'FLUXNET ({flux_time})'
        )

        ax.fill_between(
            flux_stats['Hour'],
            flux_stats[f'{var}_avg'] - flux_stats[f'{var}_std'],
            flux_stats[f'{var}_avg'] + flux_stats[f'{var}_std'],
            color=flux_color,
            alpha=0.15
        )

        # ---- Plot ECOCO
        sns.lineplot(
            x='Hour', y=f'{var}_avg',
            data=ecoco_stats,
            marker='o',
            linestyle='--',
            color=rs_color,
            linewidth=2,
            ax=ax,
            label=f'ECOCO3 ({ecoco_time})'
        )

        ax.fill_between(
            ecoco_stats['Hour'],
            ecoco_stats[f'{var}_avg'] - ecoco_stats[f'{var}_std'],
            ecoco_stats[f'{var}_avg'] + ecoco_stats[f'{var}_std'],
            color=rs_color,
            alpha=0.15
        )

        # ---- Centroid markers
        flux_y  = np.interp(flux_centroid, flux_stats['Hour'], flux_stats[f'{var}_avg'])
        ecoco_y = np.interp(ecoco_centroid, ecoco_stats['Hour'], ecoco_stats[f'{var}_avg'])

        ax.scatter(flux_centroid, flux_y,
                   s=150, edgecolor='black',
                   facecolor=flux_color, marker='X')

        ax.scatter(ecoco_centroid, ecoco_y,
                   s=150, edgecolor='black',
                   facecolor=rs_color, marker='X')

        # ==========================================================
        # BOOTSTRAPPED MIDDAY DIFFERENCE
        # ==========================================================

        flux_mid  = df_flux[df_flux['Hour'].isin(midday_hours)][var].dropna().values
        ecoco_mid = df_ecoco[df_ecoco['Hour'].isin(midday_hours)][var].dropna().values

        observed_delta = ecoco_mid.mean() - flux_mid.mean()
        observed_pct   = (observed_delta / flux_mid.mean()) * 100 if flux_mid.mean() != 0 else np.nan
        
        boot_deltas = []
        for _ in range(n_boot):
            flux_sample  = rng.choice(flux_mid,  size=len(flux_mid),  replace=True)
            ecoco_sample = rng.choice(ecoco_mid, size=len(ecoco_mid), replace=True)
            boot_deltas.append(ecoco_sample.mean() - flux_sample.mean())

        ci_low  = np.percentile(boot_deltas, 2.5)
        ci_high = np.percentile(boot_deltas, 97.5)

        results.append({
            'Variable': var,
            'Flux Midday Mean': flux_mid.mean(),
            'ECOCO Midday Mean': ecoco_mid.mean(),
            'Midday Difference (ECOCO - FLUX)': observed_delta,
            '95% CI Lower (bootstrap)': ci_low,
            '95% CI Upper (bootstrap)': ci_high,
            'Flux Centroid (hr)': flux_centroid,
            'ECOCO Centroid (hr)': ecoco_centroid,
            'Flux Centroid (time)': flux_time,
            'ECOCO Centroid (time)': ecoco_time,
            'Centroid Difference (hr)': ecoco_centroid - flux_centroid
        })

        # ---- Panel formatting
        ax.text(0.02, 0.92, panel_labels[i],
                transform=ax.transAxes,
                fontsize=18,
                fontweight='bold',
                va='top',
                ha='left')
        
        # -------------------------
        ax.text(
            0.98, 0.05,
            f"Δ = {observed_delta:.2f}\n({observed_pct:.1f}%)",
            transform=ax.transAxes,
            ha='right',
            fontsize=21
        )


        ax.set_ylabel(y_labels[i], fontsize=24)
        ax.set_xlim(0, 24)
        ax.grid(True)
        ax.legend(frameon=False, fontsize=23)
        ax.tick_params(axis='both', labelsize=21)

    axes[-1].set_xlabel('Hour of Day', fontsize=26)

    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=300)

    plt.show()

    return pd.DataFrame(results)


def _bootstrap_diurnal_centroid(hours, values, n_boot, rng):
    """Stratified-by-hour bootstrap: resample each hour's values with
    replacement, recompute the hourly means, then the weighted diurnal
    centroid (hour and interpolated value) for each of n_boot draws."""
    unique_hours = np.unique(hours)
    idx_by_hour = [np.where(hours == h)[0] for h in unique_hours]

    boot_hour = np.empty(n_boot)
    boot_val = np.empty(n_boot)
    for b in range(n_boot):
        means = np.array([
            values[idxs[rng.integers(0, len(idxs), len(idxs))]].mean()
            for idxs in idx_by_hour
        ])
        boot_hour[b] = np.sum(unique_hours * means) / np.sum(means)
        boot_val[b] = np.interp(boot_hour[b], unique_hours, means)
    return boot_hour, boot_val


def plot_diurnal_veg_comparison(
    df_fluxnet,
    df_ecoco3,
    valid_veg,
    var='WUE',
    y_label='WUE [gC kg$^{-1}$ H$_2$O]',
    morning_hours=(6, 7, 8, 9),
    midday_hours=(11, 12, 13),
    n_boot=500,
    random_state=42,
    output_path=None
):
    """Full diurnal cycles by vegetation type — FLUXNET (a) vs ECOCO3 (b), each
    marked with its diurnal centroid — plus a bottom row of dumbbell
    comparisons: centroid WUE magnitude per veg (c), centroid hour per veg
    (d), and morning-minus-midday WUE Δ per veg (e). Pairs where a bootstrap's
    95% CI on the FLUXNET-ECOCO3 difference excludes zero are marked with '*'.

    Returns (centroids, comparisons): per-(veg, dataset) centroid estimates,
    and per-veg FLUXNET-ECOCO3 differences with bootstrap CIs and
    significance flags.
    """
    flux_color = '#4DAC26'
    rs_color   = '#7B2D8B'
    rng = np.random.default_rng(random_state)

    stats_cache = {}
    boot_cache = {}
    delta_boot_cache = {}
    records = []
    for veg in valid_veg:
        for label, df in [('FLUXNET', df_fluxnet), ('ECOCO3', df_ecoco3)]:
            sub = df[df['Veg'] == veg]
            if sub.empty:
                continue
            stats = get_stats(sub, var)
            stats_cache[(veg, label)] = stats
            centroid_hour = compute_diurnal_centroid(stats, 'Hour', f'{var}_avg')
            centroid_val = np.interp(centroid_hour, stats['Hour'], stats[f'{var}_avg'])
            records.append({'Veg': veg, 'Dataset': label, 'Hour': centroid_hour, 'Value': centroid_val})

            boot_cache[(veg, label)] = _bootstrap_diurnal_centroid(
                sub['Hour'].values, sub[var].values, n_boot, rng
            )

            morning_vals = sub.loc[sub['Hour'].isin(morning_hours), var].dropna().values
            midday_vals  = sub.loc[sub['Hour'].isin(midday_hours), var].dropna().values
            if len(morning_vals) > 0 and len(midday_vals) > 0:
                delta_boot_cache[(veg, label)] = _bootstrap_delta(morning_vals, midday_vals, n_boot, rng)
    centroids = pd.DataFrame(records)
    veg_order = [v for v in valid_veg if v in set(centroids['Veg'])]

    # ── Bootstrap significance of the FLUXNET vs ECOCO3 difference, per veg ─
    comparisons = []
    for veg in veg_order:
        row = {'Veg': veg}
        if (veg, 'FLUXNET') in boot_cache and (veg, 'ECOCO3') in boot_cache:
            fb_hour, fb_val = boot_cache[(veg, 'FLUXNET')]
            eb_hour, eb_val = boot_cache[(veg, 'ECOCO3')]

            diff_hour = fb_hour - eb_hour
            diff_val  = fb_val - eb_val
            ci_hour = np.percentile(diff_hour, [2.5, 97.5])
            ci_val  = np.percentile(diff_val, [2.5, 97.5])

            row.update({
                'Hour_diff': diff_hour.mean(),
                'Hour_CI_low': ci_hour[0], 'Hour_CI_high': ci_hour[1],
                'Hour_sig': not (ci_hour[0] <= 0 <= ci_hour[1]),
                'Value_diff': diff_val.mean(),
                'Value_CI_low': ci_val[0], 'Value_CI_high': ci_val[1],
                'Value_sig': not (ci_val[0] <= 0 <= ci_val[1]),
            })
        if (veg, 'FLUXNET') in delta_boot_cache and (veg, 'ECOCO3') in delta_boot_cache:
            diff_delta = delta_boot_cache[(veg, 'FLUXNET')] - delta_boot_cache[(veg, 'ECOCO3')]
            ci_delta = np.percentile(diff_delta, [2.5, 97.5])
            row.update({
                'Delta_diff': diff_delta.mean(),
                'Delta_CI_low': ci_delta[0], 'Delta_CI_high': ci_delta[1],
                'Delta_sig': not (ci_delta[0] <= 0 <= ci_delta[1]),
            })
        comparisons.append(row)
    comparisons = pd.DataFrame(comparisons).set_index('Veg')

    # Variable name and units on separate lines for panels (a)/(c). Panel (e)
    # builds its own label from the unsplit y_label (see below) since it
    # already carries its own "(Morning - Midday)" line -- splitting units
    # there too would make it three lines.
    y_label_nl = y_label.replace(' [', ' \n[', 1)

    fig = plt.figure(figsize=(38, 22))
    gs = fig.add_gridspec(2, 6, hspace=0.3, wspace=0.7)
    ax_flux  = fig.add_subplot(gs[0, 0:3])
    ax_eco   = fig.add_subplot(gs[0, 3:6])
    ax_value = fig.add_subplot(gs[1, 0:2])
    ax_time  = fig.add_subplot(gs[1, 2:4])
    ax_delta = fig.add_subplot(gs[1, 4:6])

    # ── Row 1: full diurnal curves, centroid marked with X ──────────────────
    for ax, label, panel in zip([ax_flux, ax_eco], ['FLUXNET', 'ECOCO3'], ['(a)', '(b)']):
        for veg in valid_veg:
            if (veg, label) not in stats_cache:
                continue
            stats = stats_cache[(veg, label)]
            color = veg_color_palette.get(veg, 'gray')
            ax.plot(stats['Hour'], stats[f'{var}_avg'], marker='o', color=color, label=veg)

            centroid_row = centroids[(centroids['Veg'] == veg) & (centroids['Dataset'] == label)]
            ax.scatter(centroid_row['Hour'], centroid_row['Value'], s=150, marker='X',
                       edgecolor='black', facecolor=color, zorder=5)

        ax.text(0.02, 0.97, panel, transform=ax.transAxes, fontsize=42,
                fontweight='bold', va='top', ha='left')
        ax.set_title(label, fontsize=42)
        ax.set_xlabel('Hour of Day', fontsize=34)
        ax.set_xlim(0, 24)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=28)

    ax_flux.set_ylabel(y_label_nl, fontsize=34)
    ax_eco.legend(title='Vegetation', bbox_to_anchor=(1.02, 1), loc='upper left',
                  fontsize=26, title_fontsize=30, frameon=False)

    # ── (c): centroid WUE magnitude, FLUXNET vs ECOCO3, per veg (dumbbell) ──
    for x, veg in enumerate(veg_order):
        flux_row = centroids[(centroids['Veg'] == veg) & (centroids['Dataset'] == 'FLUXNET')]
        eco_row  = centroids[(centroids['Veg'] == veg) & (centroids['Dataset'] == 'ECOCO3')]
        if flux_row.empty or eco_row.empty:
            continue
        fv = flux_row['Value'].values[0]
        ev = eco_row['Value'].values[0]

        ax_value.plot([x, x], [fv, ev], color='gray', lw=1.5, zorder=1)
        ax_value.scatter(x, fv, s=150, marker='X', color=flux_color, edgecolor='black', zorder=3,
                          label='FLUXNET' if x == 0 else None)
        ax_value.scatter(x, ev, s=150, marker='X', color=rs_color, edgecolor='black', zorder=3,
                          label='ECOCO3' if x == 0 else None)

        if veg in comparisons.index and comparisons.loc[veg, 'Value_sig']:
            offset = 0.06 * centroids['Value'].max()
            ax_value.text(x, max(fv, ev) + offset, '*', fontsize=38,
                          fontweight='bold', ha='center', va='bottom')

    ax_value.set_xticks(range(len(veg_order)))
    ax_value.set_xticklabels(veg_order, fontsize=28, rotation=45, ha='right')
    ax_value.set_ylabel(y_label_nl, fontsize=34)
    ax_value.set_title('Diurnal Centroid Magnitude', fontsize=38)
    ax_value.text(0.02, 0.97, '(c)', transform=ax_value.transAxes, fontsize=42,
                  fontweight='bold', va='top', ha='left')
    ax_value.text(0.98, 0.03, f'* : bootstrap 95% CI excludes 0 (n={n_boot})',
                  transform=ax_value.transAxes, fontsize=20, style='italic',
                  ha='right', va='bottom')
    ax_value.tick_params(axis='y', labelsize=26)
    ax_value.grid(alpha=0.3, axis='y')
    ax_value.legend(frameon=False, fontsize=26, loc='best')

    # (a), (b), (c) all show WUE magnitude — put them on the same y-scale;
    # add headroom above the data so the significance asterisks and panel
    # tag never collide with the plotted lines/points.
    y_min = min(ax_flux.get_ylim()[0], ax_eco.get_ylim()[0], ax_value.get_ylim()[0])
    y_max = max(ax_flux.get_ylim()[1], ax_eco.get_ylim()[1], ax_value.get_ylim()[1])
    y_max += 0.16 * (y_max - y_min)
    for ax in [ax_flux, ax_eco, ax_value]:
        ax.set_ylim(y_min, y_max)

    # ── (d): centroid hour, FLUXNET vs ECOCO3, per veg (dumbbell) ───────────
    for y, veg in enumerate(veg_order):
        flux_row = centroids[(centroids['Veg'] == veg) & (centroids['Dataset'] == 'FLUXNET')]
        eco_row  = centroids[(centroids['Veg'] == veg) & (centroids['Dataset'] == 'ECOCO3')]
        if flux_row.empty or eco_row.empty:
            continue
        fh = flux_row['Hour'].values[0]
        eh = eco_row['Hour'].values[0]

        ax_time.plot([fh, eh], [y, y], color='gray', lw=1.5, zorder=1)
        ax_time.scatter(fh, y, s=150, marker='X', color=flux_color, edgecolor='black', zorder=3,
                        label='FLUXNET' if y == 0 else None)
        ax_time.scatter(eh, y, s=150, marker='X', color=rs_color, edgecolor='black', zorder=3,
                        label='ECOCO3' if y == 0 else None)

        if veg in comparisons.index and comparisons.loc[veg, 'Hour_sig']:
            ax_time.text(max(fh, eh) + 0.5, y, '*', fontsize=38,
                         fontweight='bold', ha='left', va='center')

    ax_time.set_yticks(range(len(veg_order)))
    ax_time.set_yticklabels(veg_order, fontsize=28)
    # extra padding above/below the outer rows so significance asterisks and
    # the panel tag/caption never collide with the top/bottom dumbbells
    ax_time.set_ylim(-1.3, len(veg_order) - 0.2)
    ax_time.invert_yaxis()
    ax_time.set_xlim(0, 24)
    ax_time.set_title('Diurnal Centroid Time', fontsize=38)
    ax_time.set_xlabel('Diurnal Centroid (Hour of Day)', fontsize=34)
    ax_time.text(0.02, 0.97, '(d)', transform=ax_time.transAxes, fontsize=42,
                 fontweight='bold', va='top', ha='left')
    ax_time.text(0.98, 0.03, f'* : bootstrap 95% CI excludes 0 (n={n_boot})',
                 transform=ax_time.transAxes, fontsize=20, style='italic',
                 ha='right', va='bottom')
    ax_time.tick_params(axis='x', labelsize=26)
    ax_time.grid(alpha=0.3, axis='x')
    ax_time.legend(frameon=False, fontsize=26, loc='upper right')

    # ── (e): morning-minus-midday Δ, FLUXNET vs ECOCO3, per veg (dumbbell) ──
    for x, veg in enumerate(veg_order):
        if (veg, 'FLUXNET') not in delta_boot_cache or (veg, 'ECOCO3') not in delta_boot_cache:
            continue
        fd = delta_boot_cache[(veg, 'FLUXNET')].mean()
        ed = delta_boot_cache[(veg, 'ECOCO3')].mean()

        ax_delta.plot([x, x], [fd, ed], color='gray', lw=1.5, zorder=1)
        ax_delta.scatter(x, fd, s=150, marker='X', color=flux_color, edgecolor='black', zorder=3,
                          label='FLUXNET' if x == 0 else None)
        ax_delta.scatter(x, ed, s=150, marker='X', color=rs_color, edgecolor='black', zorder=3,
                          label='ECOCO3' if x == 0 else None)

        if veg in comparisons.index and comparisons.loc[veg, 'Delta_sig']:
            offset = 0.06 * (comparisons['Delta_diff'].abs().max() if 'Delta_diff' in comparisons else 1)
            ax_delta.text(x, max(fd, ed) + offset, '*', fontsize=38,
                          fontweight='bold', ha='center', va='bottom')

    ax_delta.axhline(0, color='black', lw=1, linestyle='--', alpha=0.5, zorder=1)
    ax_delta.set_xticks(range(len(veg_order)))
    ax_delta.set_xticklabels(veg_order, fontsize=28, rotation=45, ha='right')
    ax_delta.set_ylabel(f'Δ {y_label}\n(Morning − Midday)', fontsize=34)
    ax_delta.set_title('Morning − Midday Δ', fontsize=38)
    # headroom above the data so significance asterisks never collide with
    # the title
    y0, y1 = ax_delta.get_ylim()
    ax_delta.set_ylim(y0, y1 + 0.2 * (y1 - y0))
    ax_delta.text(0.02, 0.97, '(e)', transform=ax_delta.transAxes, fontsize=42,
                  fontweight='bold', va='top', ha='left')
    ax_delta.text(0.98, 0.03, f'* : bootstrap 95% CI excludes 0 (n={n_boot})',
                  transform=ax_delta.transAxes, fontsize=20, style='italic',
                  ha='right', va='bottom')
    ax_delta.tick_params(axis='y', labelsize=26)
    ax_delta.grid(alpha=0.3, axis='y')
    ax_delta.legend(frameon=False, fontsize=26, loc='best')

    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

    return centroids, comparisons


def _bootstrap_delta(morning_vals, midday_vals, n_boot, rng):
    """Bootstrap the difference in means (morning - midday) via independent
    resampling of the morning and midday observations."""
    n_m, n_d = len(morning_vals), len(midday_vals)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        morning_mean = morning_vals[rng.integers(0, n_m, n_m)].mean()
        midday_mean  = midday_vals[rng.integers(0, n_d, n_d)].mean()
        deltas[b] = morning_mean - midday_mean
    return deltas


def _fast_centroid(sub_hours, sub_values):
    """Diurnal centroid (hour, interpolated value) from raw (Hour, value)
    arrays — used inside the power-curve subsampling loop where pandas
    groupby overhead matters."""
    unique_hours = np.unique(sub_hours)
    if len(unique_hours) < 2:
        return np.nan, np.nan
    means = np.array([sub_values[sub_hours == h].mean() for h in unique_hours])
    if np.sum(means) == 0:
        return np.nan, np.nan
    centroid_hour = np.sum(unique_hours * means) / np.sum(means)
    centroid_val = np.interp(centroid_hour, unique_hours, means)
    return centroid_hour, centroid_val


def _extend_to_actual(sample_sizes, n_available):
    """Candidate sizes capped at n_available, with n_available itself
    appended as the final point — so the curve reaches the actual available
    sample size instead of stopping at the nearest candidate below it."""
    sizes = sorted(set(s for s in sample_sizes if s < n_available) | {n_available})
    return sizes


def _repeats_for(n, n_repeats):
    """Fewer repeats at very large n: each repeat is O(n) and the sampling
    distribution is already tight, so fewer draws still give a stable width."""
    return n_repeats if n <= 50_000 else max(30, n_repeats // 5)


def _power_curve_centroid(hours, values, sample_sizes, n_repeats, rng):
    """For each candidate sample size (extended to reach the actual n
    available), draw independent random subsamples and recompute the diurnal
    centroid — returns the 95% interval width of the resulting hour/value
    estimates at each n."""
    n_total = len(hours)
    rows = []
    for n in _extend_to_actual(sample_sizes, n_total):
        reps = _repeats_for(n, n_repeats)
        hour_ests, val_ests = [], []
        for _ in range(reps):
            idx = rng.integers(0, n_total, n)
            h, v = _fast_centroid(hours[idx], values[idx])
            if np.isfinite(h):
                hour_ests.append(h)
                val_ests.append(v)
        if len(hour_ests) < 10:
            continue
        hour_ests, val_ests = np.array(hour_ests), np.array(val_ests)
        h_lo, h_hi = np.percentile(hour_ests, [2.5, 97.5])
        v_lo, v_hi = np.percentile(val_ests, [2.5, 97.5])
        rows.append({'n': n, 'hour_ci_width': h_hi - h_lo, 'value_ci_width': v_hi - v_lo})
    return pd.DataFrame(rows)


def _power_curve_delta(morning_vals, midday_vals, sample_sizes, n_repeats, rng):
    """For each candidate per-window sample size n (extended to reach the
    actual n available in the smaller of the two windows), draw independent
    random subsamples of n morning and n midday observations and recompute
    Δ(mean) — returns the 95% interval width at each n."""
    n_morning, n_midday = len(morning_vals), len(midday_vals)
    n_available = min(n_morning, n_midday)
    rows = []
    for n in _extend_to_actual(sample_sizes, n_available):
        reps = _repeats_for(n, n_repeats)
        ests = []
        for _ in range(reps):
            m = morning_vals[rng.integers(0, n_morning, n)].mean()
            d = midday_vals[rng.integers(0, n_midday, n)].mean()
            ests.append(m - d)
        ests = np.array(ests)
        lo, hi = np.percentile(ests, [2.5, 97.5])
        rows.append({'n': n, 'delta_ci_width': hi - lo})
    return pd.DataFrame(rows)


def plot_sample_size_power_curves(
    df_fluxnet,
    df_ecoco3,
    var='WUE',
    var_units='gC kg$^{-1}$ H$_2$O',
    morning_hours=(6, 7, 8, 9),
    midday_hours=(11, 12, 13),
    sample_sizes=(5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000),
    n_repeats=200,
    random_state=42,
    output_path=None
):
    """How many samples are needed to separate signal from noise, pooled
    across all vegetation types: for the diurnal centroid time (a), centroid
    magnitude (b), and the morning-minus-midday Δ (c), repeatedly subsample
    each dataset at increasing sample sizes and track how the 95% interval
    width of the resulting estimate shrinks. A dashed vertical line marks
    each dataset's actual available sample size.

    Returns a DataFrame with columns [Statistic, Dataset, n, ci_width].
    """
    flux_color = '#4DAC26'
    rs_color   = '#7B2D8B'
    rng = np.random.default_rng(random_state)

    curves = {}
    n_actual = {}
    for label, df in [('FLUXNET', df_fluxnet), ('ECOCO3', df_ecoco3)]:
        hours = df['Hour'].values
        values = df[var].values
        n_actual[label] = len(df)

        centroid_curve = _power_curve_centroid(hours, values, sample_sizes, n_repeats, rng)
        curves[('Centroid Time', label)] = centroid_curve[['n', 'hour_ci_width']].rename(
            columns={'hour_ci_width': 'ci_width'})
        curves[('Centroid Magnitude', label)] = centroid_curve[['n', 'value_ci_width']].rename(
            columns={'value_ci_width': 'ci_width'})

        morning_vals = df.loc[df['Hour'].isin(morning_hours), var].dropna().values
        midday_vals  = df.loc[df['Hour'].isin(midday_hours), var].dropna().values
        delta_curve = _power_curve_delta(morning_vals, midday_vals, sample_sizes, n_repeats, rng)
        curves[('Morning-Midday Δ', label)] = delta_curve[['n', 'delta_ci_width']].rename(
            columns={'delta_ci_width': 'ci_width'})

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    panels = ['Centroid Time', 'Centroid Magnitude', 'Morning-Midday Δ']
    y_labels = [
        '95% interval width [hrs]',
        f'95% interval width [{var_units}]',
        f'95% interval width [{var_units}]',
    ]
    panel_labels = ['(a)', '(b)', '(c)']

    for ax, stat, y_lab, panel in zip(axes, panels, y_labels, panel_labels):
        for label, color in [('FLUXNET', flux_color), ('ECOCO3', rs_color)]:
            curve = curves[(stat, label)]
            ax.plot(curve['n'], curve['ci_width'], marker='o', color=color, label=label)
            ax.axvline(n_actual[label], color=color, lw=4, linestyle='--', alpha=0.5)

        ax.set_xscale('log')
        ax.set_xlabel('Sample size (n)', fontsize=21)
        ax.set_ylabel(y_lab, fontsize=21)
        ax.set_title(stat, fontsize=23)
        ax.text(0.02, 0.97, panel, transform=ax.transAxes, fontsize=23,
                fontweight='bold', va='top', ha='left')
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=17)
        ax.legend(frameon=False, fontsize=16)

    fig.suptitle('Sample Size Needed to Separate Signal from Noise (pooled across vegetation types)\n'
                 'Dashed lines mark each dataset\'s actual available sample size', fontsize=21)
    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

    records = []
    for (stat, label), curve in curves.items():
        for _, row in curve.iterrows():
            records.append({'Statistic': stat, 'Dataset': label, 'n': row['n'], 'ci_width': row['ci_width']})
    return pd.DataFrame(records)


def plot_diurnal_cycles_spei(
    df,
    spei_bins,
    title='Diurnal Cycles by SPEI',
    midday_hours=(10,14),
    n_boot=1000,
    random_state=42,
    index_name='SPEI',
    output_dir='figures/diurnal_cycles_drought'
):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd

    df = df.copy()
    df['Hour'] = df['TIMESTAMP'].dt.hour
    df['Date'] = df['TIMESTAMP'].dt.date

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig.suptitle(title, fontsize=31)

    panel_labels = ['(a)', '(b)', '(c)']
    variables = ['GPP', 'ET', 'WUE']
    y_labels = [
        'GPP [µmol CO$_2$ m$^{-2}$ s$^{-1}$]',
        'ET [W m$^{-2}$]',
        'WUE [gC kg$^{-1}$ H$_2$O]'
    ]

    colors = ['#D95F0E', '#2C7FB8']
    results_records = []
    rng = np.random.default_rng(random_state)

    drought_label, drought_condition = spei_bins[0]
    nondrought_label, nondrought_condition = spei_bins[1]
    
    for i, var in enumerate(variables):
        
        ax = axes[i]
        
        # -------------------------
        # DIURNAL CURVES + CENTROIDS
        # -------------------------
        bin_subsets = {}
        for j, (label, condition) in enumerate(spei_bins):

            subset = df[condition(df)].copy()
            if subset.empty:
                continue
            bin_subsets[label] = subset

            stats = get_stats(subset, var)  # <-- your original function
            
            color = colors[j]
            
            sns.lineplot(
                x='Hour', y=f'{var}_avg',
                data=stats,
                marker='o',
                linewidth=2,
                color=color,
                ax=ax,
                label=label
            )
            
            ax.fill_between(
                stats['Hour'],
                stats[f'{var}_avg'] - stats[f'{var}_std'],
                stats[f'{var}_avg'] + stats[f'{var}_std'],
                color=color,
                alpha=0.15
            )
            
            # ---- use centroid function
            centroid = compute_diurnal_centroid(
                stats,
                'Hour',
                f'{var}_avg'
            )
            
            centroid_time = hour_to_timestamp(centroid)
            
            y_val = np.interp(
                centroid,
                stats['Hour'],
                stats[f'{var}_avg']
            )
            
            ax.scatter(
                centroid,
                y_val,
                s=150,
                edgecolor='black',
                facecolor=color,
                marker='X'
            )
            
            # Save centroid row
            results_records.append({
                'Variable': var,
                f'{index_name} Bin': label,
                'Centroid Hour': centroid,
                'Centroid Time': centroid_time,
                'Midday Δ': np.nan,
                '% Change': np.nan,
                'CI 2.5%': np.nan,
                'CI 97.5%': np.nan,
                'N drought days': np.nan,
                'N non-drought days': np.nan,
                'Centroid Shift (hrs)': np.nan,
                'Shift CI 2.5%': np.nan,
                'Shift CI 97.5%': np.nan
            })

        # -------------------------
        # CENTROID SHIFT UNDER DROUGHT (bootstrap CI)
        # -------------------------
        drought_subset    = bin_subsets.get(drought_label)
        nondrought_subset = bin_subsets.get(nondrought_label)
        if drought_subset is not None and nondrought_subset is not None:
            n_drought_days    = drought_subset['Date'].nunique()
            n_nondrought_days = nondrought_subset['Date'].nunique()

            if n_drought_days > 5 and n_nondrought_days > 5:
                boot_hour_d, _  = _bootstrap_diurnal_centroid(
                    drought_subset['Hour'].values, drought_subset[var].values, n_boot, rng
                )
                boot_hour_nd, _ = _bootstrap_diurnal_centroid(
                    nondrought_subset['Hour'].values, nondrought_subset[var].values, n_boot, rng
                )
                shift_boot = boot_hour_d - boot_hour_nd
                shift_ci_low, shift_ci_high = np.percentile(shift_boot, [2.5, 97.5])

                drought_centroid    = compute_diurnal_centroid(get_stats(drought_subset, var), 'Hour', f'{var}_avg')
                nondrought_centroid = compute_diurnal_centroid(get_stats(nondrought_subset, var), 'Hour', f'{var}_avg')
                observed_shift = drought_centroid - nondrought_centroid

                results_records.append({
                    'Variable': var,
                    f'{index_name} Bin': 'Centroid shift (Drought - Non-Drought)',
                    'Centroid Hour': np.nan,
                    'Centroid Time': np.nan,
                    'Midday Δ': np.nan,
                    '% Change': np.nan,
                    'CI 2.5%': np.nan,
                    'CI 97.5%': np.nan,
                    'N drought days': n_drought_days,
                    'N non-drought days': n_nondrought_days,
                    'Centroid Shift (hrs)': observed_shift,
                    'Shift CI 2.5%': shift_ci_low,
                    'Shift CI 97.5%': shift_ci_high
                })

        # -------------------------
        # MIDDAY DROUGHT DIFFERENCE
        # -------------------------
        df_mid = df[
            (df['Hour'] >= midday_hours[0]) &
            (df['Hour'] <= midday_hours[1])
        ]
        
        df_d = df_mid[drought_condition(df_mid)]
        df_nd = df_mid[nondrought_condition(df_mid)]
        
        daily_d = df_d.groupby('Date')[var].mean().dropna()
        daily_nd = df_nd.groupby('Date')[var].mean().dropna()
        
        if len(daily_d) > 5 and len(daily_nd) > 5:
            
            obs_diff = daily_d.mean() - daily_nd.mean()
            obs_pct = (obs_diff / daily_nd.mean()) * 100
            
            boot_diffs = []
            for _ in range(n_boot):
                sample_d = np.random.choice(daily_d, len(daily_d), replace=True)
                sample_nd = np.random.choice(daily_nd, len(daily_nd), replace=True)
                boot_diffs.append(sample_d.mean() - sample_nd.mean())
            
            ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])
            
            ax.text(
                0.98, 0.05,
                f"Midday Δ = {obs_diff:.2f}\n({obs_pct:.1f}%)",
                transform=ax.transAxes,
                ha='right',
                fontsize=14
            )
            
            # Save midday row
            results_records.append({
                'Variable': var,
                f'{index_name} Bin': 'Midday drought - non-drought',
                'Centroid Hour': np.nan,
                'Centroid Time': np.nan,
                'Midday Δ': obs_diff,
                '% Change': obs_pct,
                'CI 2.5%': ci_low,
                'CI 97.5%': ci_high,
                'N drought days': len(daily_d),
                'N non-drought days': len(daily_nd),
                'Centroid Shift (hrs)': np.nan,
                'Shift CI 2.5%': np.nan,
                'Shift CI 97.5%': np.nan
            })

        ax.text(
            0.02, 0.92,
            panel_labels[i],
            transform=ax.transAxes,
            fontsize=23,
            fontweight='bold',
            va='top',
            ha='left'
        )
        
        ax.set_ylabel(y_labels[i], fontsize=26)
        ax.set_xlim(0, 24)
        ax.grid(True)
        ax.legend(frameon=False, fontsize=23)
    
    axes[-1].set_xlabel('Hour of Day', fontsize=26)
    plt.tight_layout()
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    out = f'{output_dir}/{title}.png'
    plt.savefig(out, dpi=300)
    plt.close()
    
    return pd.DataFrame(results_records)

def plot_data_coverage_map(df,valid_veg=None, valid_kg=None,output_path=None):

    # ======================================================
    # MAP DATA: SCENES PER LOCATION
    # ======================================================
    loc_counts = (
        df.groupby(['Lat', 'Lon'])['Timestamp']
        .nunique()
        .reset_index(name='Count')
    )
        
    #loc_counts = loc_counts[loc_counts['Count'] >= MIN_SCENES]
    
    geometry = [Point(xy) for xy in zip(loc_counts['Lon'], loc_counts['Lat'])]
    gdf = gpd.GeoDataFrame(loc_counts, geometry=geometry, crs="EPSG:4326")

    # ======================================================
    # BAR DATA WITH RICHNESS
    # ======================================================
    df_veg = df[df['Veg'].isin(valid_veg)] if valid_veg is not None else df
    df_kg  = df[df['kg_label'].isin(valid_kg)] if valid_kg is not None else df

    # Scenes per vegetation
    veg_bar = (
        df_veg.groupby('Veg')['Timestamp']
        .nunique()
        .reset_index(name='Scene_Count')
    )

    # Scenes per climate
    kop_bar = (
        df_kg.groupby('kg_label')['Timestamp']
        .nunique()
        .reset_index(name='Scene_Count')
    )

    # Climate diversity per vegetation
    kg_per_veg = (
        df_veg.groupby('Veg')['kg_label']
        .nunique()
        .reset_index(name='Num_Climate_Classes')
    )

    # Vegetation diversity per climate
    veg_per_kg = (
        df_kg.groupby('kg_label')['Veg']
        .nunique()
        .reset_index(name='Num_Veg_Classes')
    )

    # Merge summaries
    veg_summary = veg_bar.merge(kg_per_veg, on='Veg')
    kop_summary = kop_bar.merge(veg_per_kg, on='kg_label')

    veg_summary = veg_summary.sort_values('Scene_Count')
    kop_summary = kop_summary.sort_values('Scene_Count')

    # TOD and Season
    tod_bar = (
        df.groupby('Hour')['Timestamp']
        .nunique()
        .sort_values()
    )

    season_order = ['Winter', 'Spring', 'Summer', 'Fall']
    season_bar = (
        df.groupby('Season')['Timestamp']
        .nunique()
        .reindex(season_order)
    )

    # ======================================================
    # FIGURE LAYOUT
    # ======================================================
    fig = plt.figure(figsize=(10, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[3, 1.5, 1.5])

    ax_map = fig.add_subplot(gs[0, :], projection=ccrs.PlateCarree())
    ax_veg = fig.add_subplot(gs[1, 0])
    ax_kop = fig.add_subplot(gs[1, 1])
    ax_tod = fig.add_subplot(gs[2, 0])
    ax_season = fig.add_subplot(gs[2, 1])

    # ======================================================
    # MAP: NUMBER OF SCENES PER LOCATION
    # ======================================================
    ax_map.set_global()
    ax_map.add_feature(cfeature.LAND, color='lightgray')
    ax_map.add_feature(cfeature.OCEAN, color='lightblue')
    ax_map.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax_map.add_feature(cfeature.BORDERS, linestyle=':')

    ax_map.axhline(51.6, color='blue', linestyle='--', linewidth=1)
    ax_map.axhline(-51.6, color='blue', linestyle='--', linewidth=1)

    sc = ax_map.scatter(
        gdf['Lon'],
        gdf['Lat'],
        s=50,
        c=gdf['Count'],
        cmap='plasma',
        alpha=0.7,
        edgecolors='none',
        vmin=1,
        vmax=30,
        transform=ccrs.PlateCarree()
    )

    cbar = plt.colorbar(
        sc,
        ax=ax_map,
        orientation='horizontal',
        fraction=0.03,
        pad=0.04,
        extend='max',
        shrink=0.5
    )

    cbar.set_label('Number of Scenes', fontsize=18)
    cbar.ax.tick_params(labelsize=16)

    ax_map.set_title('Number of Scenes per Location', fontsize=23)

    # ======================================================
    # BAR — VEGETATION (WITH CLIMATE BREADTH)
    # ======================================================
    veg_colors = [veg_color_palette.get(v, 'gray') for v in veg_summary['Veg']]

    ax_veg.barh(
        veg_summary['Veg'],
        veg_summary['Scene_Count'],
        color=veg_colors
    )

    ax_veg.set_title('Scenes per Vegetation', fontsize=18)
    ax_veg.set_xlabel('Scene Count', fontsize=16)

    # for i, (count, kg_n) in enumerate(zip(veg_summary['Scene_Count'], veg_summary['Num_Climate_Classes'])):
    #     ax_veg.text(
    #         count + 1,
    #         i,
    #         f'{kg_n} kgs',
    #         va='center',
    #         fontsize=16
    #     )

    # ======================================================
    # BAR — KÖPPEN (WITH VEGETATION BREADTH)
    # ======================================================
    kop_colors = [koppen_label_color_palette.get(k, 'gray') for k in kop_summary['kg_label']]

    ax_kop.barh(
        kop_summary['kg_label'],
        kop_summary['Scene_Count'],
        color=kop_colors
    )

    ax_kop.set_title('Scenes per KG Climate Class', fontsize=18 )
    ax_kop.set_xlabel('Scene Count', fontsize=16)

    # for i, (count, veg_n) in enumerate(zip(kop_summary['Scene_Count'], kop_summary['Num_Veg_Classes'])):
    #     ax_kop.text(
    #         count + 1,
    #         i,
    #         f'{veg_n} veg',
    #         va='center',
    #         fontsize=16
    #     )

    # ======================================================
    # BAR — TIME OF DAY
    # ======================================================
    ax_tod.barh(tod_bar.index, tod_bar.values, color='steelblue')
    ax_tod.invert_yaxis()  # Optional: invert y-axis to have morning at top
    ax_tod.set_yticks(tod_bar.index)
    ax_tod.set_yticklabels([f'{hour}:00' for hour in tod_bar.index])
    ax_tod.set_title('Scenes per Time of Day', fontsize=18)
    ax_tod.set_xlabel('Scene Count', fontsize=16)

    # ======================================================
    # BAR — SEASON
    # ======================================================
    ax_season.barh(season_bar.index, season_bar.values, color='darkorange')
    ax_season.invert_yaxis()  # Optional: invert y-axis to have winter at top
    ax_season.set_title('Scenes per Season', fontsize=18)
    ax_season.set_xlabel('Scene Count', fontsize=16)

    # ======================================================
    # PANEL LABELS
    # ======================================================
    # Placed above and to the left of each axes (not inside, top-left) so
    # they never sit on top of that panel's own title or data.
    label_kwargs = dict(
        x=-0.06, y=1.02,
        fontsize=18,
        fontweight='bold',
        va='bottom',
        ha='left'
    )

    ax_map.text(s='(a)', transform=ax_map.transAxes, **label_kwargs)
    ax_veg.text(s='(b)', transform=ax_veg.transAxes, **label_kwargs)
    ax_kop.text(s='(c)', transform=ax_kop.transAxes, **label_kwargs)
    ax_tod.text(s='(d)', transform=ax_tod.transAxes, **label_kwargs)
    ax_season.text(s='(e)', transform=ax_season.transAxes, **label_kwargs)
    # ======================================================
    # FINALIZE
    # ======================================================
    plt.suptitle('ECOCO3 Version 2 Coverage Summary', fontsize=26)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300)
    else:
        plt.savefig('figures/data_coverage_map.png', dpi=300)
    plt.show()


def plot_violin_split(df_fluxnet, df_ecoco3, grouping_category, ax, title, valid_categories=None, show_legend=True):
    """One panel: FLUXNET and ECOCO3 as a split violin per category (left
    half FLUXNET, right half ECOCO3), with a star above any category where
    the two distributions differ (Mann-Whitney U, p < 0.05) — the direct
    FLUXNET-vs-ECOCO3 comparison, as opposed to plot_violin's ANOVA/Tukey
    letters, which instead compare categories against each other within a
    single dataset.
    """
    df_fluxnet = df_fluxnet.copy()
    df_ecoco3 = df_ecoco3.copy()
    if valid_categories is not None:
        df_fluxnet = df_fluxnet[df_fluxnet[grouping_category].isin(valid_categories)]
        df_ecoco3 = df_ecoco3[df_ecoco3[grouping_category].isin(valid_categories)]

    df_fluxnet = df_fluxnet.assign(Dataset='FLUXNET')
    df_ecoco3 = df_ecoco3.assign(Dataset='ECOCO3')
    combined = pd.concat([df_fluxnet, df_ecoco3], ignore_index=True)

    order = (
        combined.groupby(grouping_category)['WUE']
        .median()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    # Lightened versions of the FLUXNET/ECOCO3 color pair used elsewhere in
    # this module (e.g. plot_morning_midday_violin) -- full saturation made
    # the black quartile lines hard to see against the fill.
    flux_color = '#9DD187'
    rs_color   = '#B68BBF'
    sns.violinplot(x=grouping_category, y='WUE', hue='Dataset', data=combined,
                    hue_order=['FLUXNET', 'ECOCO3'], palette={'FLUXNET': flux_color, 'ECOCO3': rs_color},
                    split=True, inner='quartile', inner_kws=dict(linewidth=2.2),
                    order=order, cut=0, ax=ax)

    for i, category in enumerate(order):
        flux_vals = df_fluxnet.loc[df_fluxnet[grouping_category] == category, 'WUE'].dropna()
        eco_vals = df_ecoco3.loc[df_ecoco3[grouping_category] == category, 'WUE'].dropna()
        if len(flux_vals) < 1 or len(eco_vals) < 1:
            continue
        _, p = mannwhitneyu(flux_vals, eco_vals, alternative='two-sided')
        if p < 0.05:
            ax.text(i, 5.3, '*', ha='center', va='center', fontsize=40, fontweight='bold',
                    color='black', path_effects=[pe.withStroke(linewidth=2, foreground='white')])

    ax.set_title(title, fontsize=32, pad=16)
    ax.set_xlabel(grouping_category, fontsize=28)
    ax.set_ylabel('WUE \n [gC kg$^{-1}$H$_2$O]', fontsize=28)
    ax.set_ylim(0, 6)
    ax.tick_params(axis='x', rotation=45, labelsize=27)
    ax.tick_params(axis='y', labelsize=27)
    # Back inside the axes, upper right, but low enough to clear the star
    # row (~y=5-5.7) above it. Only one legend needed across both panels.
    if show_legend:
        ax.legend(fontsize=24, loc='upper right', bbox_to_anchor=(0.99, 0.80), framealpha=0.9)
    else:
        ax.legend_.remove()


def plot_violin_comparison_stacked(
    df_ecoco3,
    df_fluxnet,
    grouping_category='Veg',
    second_grouping_category='kg_label',
    valid_veg=None,
    valid_kg=None,
    output_path=None
):

    fig, axes = plt.subplots(4, 1, figsize=(8, 11), sharey=True)

    # Top: FLUXNET
    plot_violin(
        df_wue_daily=df_fluxnet[df_fluxnet[grouping_category].isin(valid_veg) if valid_veg is not None else df_fluxnet[grouping_category]],
        grouping_category=grouping_category,
        ax=axes[0],
        title='FLUXNET by Vegetation'
    )

    # Bottom: ECOCO3
    plot_violin(
        df_wue_daily=df_ecoco3[df_ecoco3[grouping_category].isin(valid_veg) if valid_veg is not None else df_ecoco3[grouping_category]],
        grouping_category=grouping_category,
        ax=axes[1],
        title='ECOCO3 by Vegetation'
    )

    plot_violin(
        df_wue_daily=df_fluxnet[df_fluxnet[second_grouping_category].isin(valid_kg) if valid_kg is not None else df_fluxnet[second_grouping_category]],
        grouping_category=second_grouping_category,
        ax=axes[2],
        title='FLUXNET by Climate'
    )

    plot_violin(
        df_wue_daily=df_ecoco3[df_ecoco3[second_grouping_category].isin(valid_kg) if valid_kg is not None else df_ecoco3[second_grouping_category]],
        grouping_category=second_grouping_category,
        ax=axes[3],
        title='ECOCO3 by Climate'
    )


    # -----------------------------------------
    # Clean up
    # -----------------------------------------
    axes[0].set_xlabel('')  # remove duplicate x label
    axes[2].set_xlabel('')  # remove duplicate x label
    axes[1].set_xlabel('')  # remove duplicate x label
    axes[3].set_xlabel('')  # remove duplicate x label

    # Panel labels — placed above the axes (not inside, top-left) so they
    # can't overlap the violins or the significance annotations beneath them.
    # y=1.22 clears both the title (pad=30) and the CLD letters row
    # (y=1.03, axes-fraction) that plot_violin now draws above its own axes.
    panel_labels = ['(a)', '(b)', '(c)', '(d)']
    for i, ax in enumerate(axes):
        ax.text(
            0.0, 1.22, panel_labels[i],
            transform=ax.transAxes,
            fontsize=21,
            fontweight='bold',
            va='bottom',
            ha='left'
        )

    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_violin_comparison_split(
    df_ecoco3,
    df_fluxnet,
    grouping_category='Veg',
    second_grouping_category='kg_label',
    valid_veg=None,
    valid_kg=None,
    output_path=None
):
    """FLUXNET vs ECOCO3 WUE distributions, one split-violin panel per
    grouping (vegetation, climate) — see plot_violin_split. The star above
    a category marks p < 0.05 on a Mann-Whitney U test between FLUXNET and
    ECOCO3 for that category specifically. Companion to (not a replacement
    for) plot_violin_comparison_stacked's 4-panel Figure 3.
    """
    fig, axes = plt.subplots(2, 1, figsize=(18, 12))

    plot_violin_split(
        df_fluxnet=df_fluxnet, df_ecoco3=df_ecoco3,
        grouping_category=grouping_category, ax=axes[0],
        title='By Vegetation', valid_categories=valid_veg
    )

    plot_violin_split(
        df_fluxnet=df_fluxnet, df_ecoco3=df_ecoco3,
        grouping_category=second_grouping_category, ax=axes[1],
        title='By Climate', valid_categories=valid_kg,
        show_legend=False
    )

    axes[0].set_xlabel('')  # remove duplicate x label
    axes[1].set_xlabel('')  # 'kg_label' is a column name, not a reader-facing label

    # Panel labels — placed above the axes (not inside, top-left) since a
    # star can appear over the first category there for either panel. Pulled
    # further left/up than a bare (0, 1.02) so they clear the y-axis's own
    # top tick label ("6") instead of sitting right on top of it.
    panel_labels = ['(a)', '(b)']
    for i, ax in enumerate(axes):
        ax.text(
            -0.06, 1.04, panel_labels[i],
            transform=ax.transAxes,
            fontsize=30,
            fontweight='bold',
            va='bottom',
            ha='left'
        )

    plt.tight_layout(h_pad=6)

    if output_path is not None:
        # bbox_inches='tight' so the externally-anchored legends aren't clipped
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    plt.show()

def plot_diurnal_cycles_spei_comparison(
    df_ecoco3,
    df_fluxnet,
    spei_bins,
    title='Diurnal Cycles by SPEI',
    midday_hours=(10,14),
    n_boot=1000,
    output_path=None,
    index_name='SPEI'
):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd

    datasets = [(df_fluxnet.copy(), 'FLUXNET'),
                (df_ecoco3.copy(), 'ECOCO3')]

    variables = ['GPP', 'ET', 'WUE']
    y_labels = [
        'GPP \n[µmol CO$_2$ m$^{-2}$ s$^{-1}$]',
        'ET \n[W m$^{-2}$]',
        'WUE \n[gC kg$^{-1}$ H$_2$O]'
    ]

    colors = ['#D95F0E', '#2C7FB8']
    panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

    fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True, sharey='row')
    fig.suptitle(title + ' (FLUXNET vs ECOCO3)', fontsize=31)

    results_records = []
    label_idx = 0

    drought_label, drought_condition = spei_bins[0]
    nondrought_label, nondrought_condition = spei_bins[1]

    # -------------------------------------------------
    # LOOP: variables (rows) × datasets (cols)
    # -------------------------------------------------
    for i, var in enumerate(variables):
        for j, (df, dataset_name) in enumerate(datasets):

            df['Hour'] = df['TIMESTAMP'].dt.hour
            df['Date'] = df['TIMESTAMP'].dt.date

            ax = axes[i, j]

            # -------------------------
            # DIURNAL CURVES
            # -------------------------
            for k, (label, condition) in enumerate(spei_bins):

                subset = df[condition(df)].copy()
                if subset.empty:
                    continue

                stats = get_stats(subset, var)
                color = colors[k]
                
                # Centroid
                centroid = compute_diurnal_centroid(stats, 'Hour', f'{var}_avg')
                y_val = np.interp(centroid, stats['Hour'], stats[f'{var}_avg'])

                sns.lineplot(
                    x='Hour', y=f'{var}_avg',
                    data=stats,
                    marker='o',
                    linewidth=2,
                    color=color,
                    ax=ax,
                    label=f'{label} ({hour_to_timestamp(centroid)})'
                )
                ax.fill_between(
                    stats['Hour'],
                    stats[f'{var}_avg'] - stats[f'{var}_std'],
                    stats[f'{var}_avg'] + stats[f'{var}_std'],
                    color=color,
                    alpha=0.15
                )

                ax.scatter(
                    centroid,
                    y_val,
                    s=120,
                    edgecolor='black',
                    facecolor=color,
                    marker='X'
                )

                results_records.append({
                    'Dataset': dataset_name,
                    'Variable': var,
                    f'{index_name} Bin': label,
                    'Centroid Hour': centroid,
                    'Centroid Time': hour_to_timestamp(centroid),
                    'Midday Δ': np.nan,
                    '% Change': np.nan,
                    'CI 2.5%': np.nan,
                    'CI 97.5%': np.nan,
                    'N drought days': np.nan,
                    'N non-drought days': np.nan,
                })

            # -------------------------
            # MIDDAY DIFFERENCE
            # -------------------------
            df_mid = df[
                (df['Hour'] >= midday_hours[0]) &
                (df['Hour'] <= midday_hours[1])
            ]

            df_d = df_mid[drought_condition(df_mid)]
            df_nd = df_mid[nondrought_condition(df_mid)]

            daily_d = df_d.groupby('Date')[var].mean().dropna()
            daily_nd = df_nd.groupby('Date')[var].mean().dropna()

            if len(daily_d) > 5 and len(daily_nd) > 5:
                obs_diff = daily_d.mean() - daily_nd.mean()
                obs_pct = (obs_diff / daily_nd.mean()) * 100

                boot_diffs = []
                for _ in range(n_boot):
                    sample_d  = np.random.choice(daily_d,  len(daily_d),  replace=True)
                    sample_nd = np.random.choice(daily_nd, len(daily_nd), replace=True)
                    boot_diffs.append(sample_d.mean() - sample_nd.mean())
                ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])

                ax.text(
                    0.98, 0.05,
                    f"Midday Δ = {obs_diff:.2f}\n({obs_pct:.1f}%)",
                    transform=ax.transAxes,
                    ha='right',
                    fontsize=21
                )

                results_records.append({
                    'Dataset': dataset_name,
                    'Variable': var,
                    f'{index_name} Bin': 'Midday drought - non-drought',
                    'Centroid Hour': np.nan,
                    'Centroid Time': np.nan,
                    'Midday Δ': obs_diff,
                    '% Change': obs_pct,
                    'CI 2.5%': ci_low,
                    'CI 97.5%': ci_high,
                    'N drought days': len(daily_d),
                    'N non-drought days': len(daily_nd),
                })

            # -------------------------
            # Formatting
            # -------------------------
            ax.text(
                0.02, 0.92,
                panel_labels[label_idx],
                transform=ax.transAxes,
                fontsize=23,
                fontweight='bold',
                va='top'
            )
            label_idx += 1

            if i == 0:
                ax.set_title(dataset_name, fontsize=23)

            if j == 0:
                ax.set_ylabel(y_labels[i], fontsize=23)

            ax.set_xlim(0, 24)
            ax.grid(True)
            ax.legend(frameon=False, fontsize=23, loc='upper right')
            ax.tick_params(axis='both', labelsize=21)

    for ax in axes[-1, :]:
        ax.set_xlabel('Hour of Day', fontsize=23)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()

    return pd.DataFrame(results_records)

def plot_data_coverage_map_sites(df, valid_veg=None, valid_kg=None, output_path=None):

    # ======================================================
    # CLEAN YEARS
    # ======================================================
    df['first_year'] = pd.to_numeric(df['first_year'], errors='coerce')
    df['first_year'] = df['first_year'].clip(lower=2018)  # only keep records starting in 2018 or later
    df['last_year']  = pd.to_numeric(df['last_year'], errors='coerce')

    df = df.dropna(subset=['first_year', 'last_year'])

    # Number of years per record post- 2018
    df['Num_Years'] = (df['last_year'] - df['first_year']) + 1

    # ======================================================
    # MAP DATA: YEARS PER LOCATION
    # ======================================================
    loc_counts = (
        df.groupby(['Lat', 'Long'])['Num_Years']
        .max()  # use max duration per site (not sum!)
        .reset_index(name='Years')
    )

    geometry = [Point(xy) for xy in zip(loc_counts['Long'], loc_counts['Lat'])]
    gdf = gpd.GeoDataFrame(loc_counts, geometry=geometry, crs="EPSG:4326")

    # ======================================================
    # FILTERED DATA
    # ======================================================
    df_veg = df[df['Veg'].isin(valid_veg)] if valid_veg is not None else df
    df_kg  = df[df['kg_label'].isin(valid_kg)] if valid_kg is not None else df

    # ======================================================
    # BAR DATA
    # ======================================================

    # Vegetation distribution
    veg_bar = (
        df_veg.groupby('Veg')['Num_Years']
        .size()
        .reset_index(name='Site Years')
    )

    # Köppen climate distribution
    kop_bar = (
        df_kg.groupby('kg_label')['Num_Years']
        .size()
        .reset_index(name='Site Years')
    )

    # Climate diversity per vegetation
    kg_per_veg = (
        df_veg.groupby('Veg')['kg_label']
        .nunique()
        .reset_index(name='Num_Climate_Classes')
    )

    # Vegetation diversity per climate
    veg_per_kg = (
        df_kg.groupby('kg_label')['Veg']
        .nunique()
        .reset_index(name='Num_Veg_Classes')
    )

    veg_summary = veg_bar.merge(kg_per_veg, on='Veg')
    kop_summary = kop_bar.merge(veg_per_kg, on='kg_label')

    veg_summary = veg_summary.sort_values('Site Years')
    kop_summary = kop_summary.sort_values('Site Years')

    # ======================================================
    # FIGURE LAYOUT
    # ======================================================
    fig = plt.figure(figsize=(10, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[3, 1.5, 1.5])

    ax_map = fig.add_subplot(gs[0, :], projection=ccrs.PlateCarree())
    ax_veg = fig.add_subplot(gs[1, 0])
    ax_kop = fig.add_subplot(gs[1, 1])

    # ======================================================
    # MAP
    # ======================================================
    ax_map.set_global()
    ax_map.add_feature(cfeature.LAND, color='lightgray')
    ax_map.add_feature(cfeature.OCEAN, color='lightblue')
    ax_map.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax_map.add_feature(cfeature.BORDERS, linestyle=':')

    ax_map.axhline(51.6, color='blue', linestyle='--', linewidth=1)
    ax_map.axhline(-51.6, color='blue', linestyle='--', linewidth=1)

    sc = ax_map.scatter(
        gdf['Long'],
        gdf['Lat'],
        s=50,
        c=gdf['Years'],
        cmap='plasma',
        alpha=0.7,
        edgecolors='none',
        transform=ccrs.PlateCarree()
    )

    cbar = plt.colorbar(sc, 
        ax=ax_map, 
        orientation='horizontal', 
        fraction=0.03,
        pad=0.04, 
        extend='max',
        shrink=0.5
    )
    
    cbar.set_label('Number of Years per Site', fontsize=18)
    cbar.ax.tick_params(labelsize=16)

    # ======================================================
    # VEGETATION BAR
    # ======================================================
    veg_colors = [veg_color_palette.get(v, 'gray') for v in veg_summary['Veg']]
    ax_veg.barh(veg_summary['Veg'], veg_summary['Site Years'], color=veg_colors)
    ax_veg.set_title('Site Years per Vegetation Type', fontsize=18)
    ax_veg.set_xlabel('Site Years', fontsize=16)

    # ======================================================
    # KÖPPEN BAR
    # ======================================================
    kop_colors = [koppen_label_color_palette.get(k, 'gray') for k in kop_summary['kg_label']]

    ax_kop.barh(kop_summary['kg_label'], kop_summary['Site Years'], color=kop_colors)
    ax_kop.set_title('Site Years per KG Climate Class', fontsize=18)
    ax_kop.set_xlabel('Site Years', fontsize=16)

    # ======================================================
    # PANEL LABELS
    # ======================================================
    # Placed above and to the left of each axes (not inside, top-left) so
    # they never sit on top of that panel's own title or data.
    label_kwargs = dict(
        x=-0.14, y=1.02,
        fontsize=18,
        fontweight='bold',
        va='bottom',
        ha='left'
    )

    ax_map.text(s='(a)', transform=ax_map.transAxes, **label_kwargs)
    ax_veg.text(s='(b)', transform=ax_veg.transAxes, **label_kwargs)
    ax_kop.text(s='(c)', transform=ax_kop.transAxes, **label_kwargs)

    # ======================================================
    # FINALIZE
    # ======================================================
    plt.suptitle('FLUXNET Site Coverage Summary', fontsize=26)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# Helpers shared by the two summary figures below
# ──────────────────────────────────────────────────────────────────────────────

def _load_supp(path):
    df = pd.read_csv(path, header=[0, 1])
    df.columns = ['Variable', 'Group',
                  'pct_ECOCO',   'pct_FLUX',
                  'ci25_ECOCO',  'ci25_FLUX',
                  'ci975_ECOCO', 'ci975_FLUX',
                  'delta_ECOCO', 'delta_FLUX',
                  'nd_ECOCO',    'nd_FLUX',
                  'nnd_ECOCO',   'nnd_FLUX']
    return df

def _load_shift(path):
    df = pd.read_csv(path, header=[0, 1])
    df.columns = ['Variable', 'Group',
                  'shift_ECOCO',  'shift_FLUX',
                  'ci25_ECOCO',   'ci25_FLUX',
                  'ci975_ECOCO',  'ci975_FLUX']
    return df

_ECO_COLOR   = '#4DAC26'
_FLUX_COLOR  = '#7B2D8B'
_VARIABLES   = ['GPP', 'ET', 'WUE']
_UNITS       = {'GPP': 'µmol CO₂ m⁻² s⁻¹', 'ET': 'W m⁻²', 'WUE': 'gC kg⁻¹ H₂O'}
_MARKER_SIZE = 14
_CAP_SIZE    = 6
_LINE_WIDTH  = 3.0
_ELINE_WIDTH = 2.8
_VLINE_WIDTH = 2.8
_STAR_SIZE   = 57

def _sig_star(lo, hi):
    return '*' if (lo > 0 or hi < 0) else ''

_RCPARAMS = {
    'font.size':         57,
    'axes.titlesize':    65,
    'axes.labelsize':    57,
    'xtick.labelsize':   52,
    'ytick.labelsize':   52,
    'legend.fontsize':   55,
    'axes.linewidth':    2.5,
    'xtick.major.width': 2.2,
    'ytick.major.width': 2.2,
    'font.weight':       'normal',
}


# ──────────────────────────────────────────────────────────────────────────────
# Figure: Midday Δ forest plots (suppression magnitude + 95 % CI)
# ──────────────────────────────────────────────────────────────────────────────

def plot_drought_suppression_summary(
    sup_veg_path='tables/suppression_summary_midday.csv',
    sup_kg_path='tables/suppression_summary_midday_kg.csv',
    output_path='figures/WUE_analysis/drought_suppression_summary.png',
    index_name='SPEI',
    drought_threshold=-1.5,
    nondrought_threshold=0
):
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    sup_veg = _load_supp(sup_veg_path)
    sup_kg  = _load_supp(sup_kg_path)

    GROUPINGS = [
        ('By Vegetation Type', sup_veg),
        ('By Climate Class',   sup_kg),
    ]

    # Size each row's height to how many categories it actually has to fit,
    # instead of a fixed ratio — otherwise a row with more groups than the
    # other (e.g. more climate classes than vegetation types) gets squeezed
    # into too little vertical space and its y-tick labels overlap.
    row_n = [max(df.groupby('Variable').size().max() if len(df) else 1, 1)
             for _, df in GROUPINGS]

    with plt.rc_context(_RCPARAMS):
        fig = plt.figure(figsize=(30, 30))
        gs  = fig.add_gridspec(2, 3, hspace=0.6, wspace=0.75, height_ratios=row_n)
        panel_labels = list('abcdef')
        label_idx = 0

        for row, (row_title, df) in enumerate(GROUPINGS):
            for col, var in enumerate(_VARIABLES):
                ax  = fig.add_subplot(gs[row, col])
                sub = df[df['Variable'] == var].copy().sort_values('delta_FLUX', ascending=True)
                n   = len(sub)
                if n == 0:
                    ax.set_visible(False)
                    label_idx += 1
                    continue

                y_flux = np.arange(n) * 1.8
                y_eco  = y_flux + 0.7

                for i, (_, r) in enumerate(sub.iterrows()):
                    xerr_f = [[max(r.delta_FLUX  - r.ci25_FLUX,  0)],
                              [max(r.ci975_FLUX  - r.delta_FLUX,  0)]]
                    ax.errorbar(r.delta_FLUX, y_flux[i], xerr=xerr_f,
                                fmt='s', color=_FLUX_COLOR, markersize=_MARKER_SIZE,
                                capsize=_CAP_SIZE, lw=_LINE_WIDTH,
                                elinewidth=_ELINE_WIDTH, capthick=_ELINE_WIDTH, zorder=3)
                    s = _sig_star(r.ci25_FLUX, r.ci975_FLUX)
                    if s:
                        ax.text(r.delta_FLUX, y_flux[i] + 0.22, s,
                                ha='center', fontsize=_STAR_SIZE, color=_FLUX_COLOR, zorder=4)

                    xerr_e = [[max(r.delta_ECOCO - r.ci25_ECOCO,  0)],
                              [max(r.ci975_ECOCO - r.delta_ECOCO, 0)]]
                    ax.errorbar(r.delta_ECOCO, y_eco[i], xerr=xerr_e,
                                fmt='o', color=_ECO_COLOR, markersize=_MARKER_SIZE,
                                capsize=_CAP_SIZE, lw=_LINE_WIDTH,
                                elinewidth=_ELINE_WIDTH, capthick=_ELINE_WIDTH, zorder=3)
                    s = _sig_star(r.ci25_ECOCO, r.ci975_ECOCO)
                    if s:
                        ax.text(r.delta_ECOCO, y_eco[i] + 0.22, s,
                                ha='center', fontsize=_STAR_SIZE, color=_ECO_COLOR, zorder=4)

                ax.axvline(0, color='black', lw=_VLINE_WIDTH, ls='--', alpha=0.5)
                ax.set_yticks(y_flux + 0.35)
                ax.set_yticklabels(sub['Group'].tolist())
                ax.set_xlabel(f'Midday Δ \n [{_UNITS[var]}]')
                ax.set_title(var)
                ax.grid(axis='x', alpha=0.3, lw=1.5)
                ax.spines[['top', 'right']].set_visible(False)
                ax.tick_params(axis='both', width=2.2, length=6)
                ax.tick_params(axis='y', labelsize=max(14, min(32, 380 / n)))
                ax.text(-0.15, 1.03, f'({panel_labels[label_idx]})',
                        transform=ax.transAxes, fontsize=70, fontweight='bold')
                label_idx += 1
                if col == 0:
                    ax.text(-0.50, 0.5, row_title, transform=ax.transAxes,
                            fontsize=52, rotation=90, va='center', ha='center')

        fig.subplots_adjust(top=0.83, bottom=0.1)

        eco_p  = mpatches.Patch(color=_ECO_COLOR,  label='ECOCO3')
        flux_p = mpatches.Patch(color=_FLUX_COLOR, label='FLUXNET')
        sig_p  = plt.Line2D([], [], color='gray', marker='$*$', linestyle='None',
                            markersize=14, label='* CI excludes 0')
        fig.legend(handles=[eco_p, flux_p, sig_p], loc='upper right',
                   frameon=False, bbox_to_anchor=(0.99, 0.92), fontsize=28)
        fig.suptitle(
            'Midday Drought Suppression (95 % Bootstrap CI)\n'
            f'Drought: {index_name} < {drought_threshold}  |  Non-Drought: {index_name} > {nondrought_threshold}',
            fontsize=57, y=0.995)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# Figure: Centroid timing shift under drought (Drought − Non-Drought, hours)
# ──────────────────────────────────────────────────────────────────────────────

def plot_centroid_shift_summary(
    shift_veg_path='tables/centroid_shift_summary.csv',
    shift_kg_path='tables/centroid_shift_summary_kg.csv',
    output_path='figures/WUE_analysis/centroid_shift_summary.png'
):
    """Bootstrap 95% CI on each dataset's own drought-vs-non-drought centroid
    timing shift; '*' marks bars whose CI excludes zero."""
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    shift_veg = _load_shift(shift_veg_path)
    shift_kg  = _load_shift(shift_kg_path)

    SHIFT_GROUPINGS = [
        ('By Vegetation Type', shift_veg),
        ('By Climate Class',   shift_kg),
    ]

    # Size each row's height to how many categories it actually has to fit,
    # instead of a fixed ratio — otherwise a row with more groups than the
    # other (e.g. more climate classes than vegetation types) gets squeezed
    # into too little vertical space and its y-tick labels overlap.
    row_n = [max(df.groupby('Variable').size().max() if len(df) else 1, 1)
             for _, df in SHIFT_GROUPINGS]

    with plt.rc_context(_RCPARAMS):
        fig = plt.figure(figsize=(30, 30))
        gs  = fig.add_gridspec(2, 3, hspace=0.6, wspace=0.75, height_ratios=row_n)
        panel_labels = list('abcdef')
        label_idx = 0

        for row, (row_title, df) in enumerate(SHIFT_GROUPINGS):
            for col, var in enumerate(_VARIABLES):
                ax  = fig.add_subplot(gs[row, col])
                sub = df[df['Variable'] == var].copy().sort_values('shift_FLUX', ascending=True)
                n   = len(sub)
                if n == 0:
                    ax.set_visible(False)
                    label_idx += 1
                    continue

                y_flux = np.arange(n) * 1.8
                y_eco  = y_flux + 0.7

                for i, (_, r) in enumerate(sub.iterrows()):
                    xerr_f = [[max(r.shift_FLUX - r.ci25_FLUX, 0)],
                              [max(r.ci975_FLUX - r.shift_FLUX, 0)]]
                    ax.errorbar(r.shift_FLUX, y_flux[i], xerr=xerr_f,
                                fmt='s', color=_FLUX_COLOR, markersize=_MARKER_SIZE,
                                capsize=_CAP_SIZE, lw=_LINE_WIDTH,
                                elinewidth=_ELINE_WIDTH, capthick=_ELINE_WIDTH, zorder=3)
                    s = _sig_star(r.ci25_FLUX, r.ci975_FLUX)
                    if s:
                        ax.text(r.shift_FLUX, y_flux[i] + 0.22, s,
                                ha='center', fontsize=_STAR_SIZE, color=_FLUX_COLOR, zorder=4)

                    xerr_e = [[max(r.shift_ECOCO - r.ci25_ECOCO, 0)],
                              [max(r.ci975_ECOCO - r.shift_ECOCO, 0)]]
                    ax.errorbar(r.shift_ECOCO, y_eco[i], xerr=xerr_e,
                                fmt='o', color=_ECO_COLOR, markersize=_MARKER_SIZE,
                                capsize=_CAP_SIZE, lw=_LINE_WIDTH,
                                elinewidth=_ELINE_WIDTH, capthick=_ELINE_WIDTH, zorder=3)
                    s = _sig_star(r.ci25_ECOCO, r.ci975_ECOCO)
                    if s:
                        ax.text(r.shift_ECOCO, y_eco[i] + 0.22, s,
                                ha='center', fontsize=_STAR_SIZE, color=_ECO_COLOR, zorder=4)

                ax.axvline(0, color='black', lw=_VLINE_WIDTH, ls='--', alpha=0.5)
                ax.set_yticks(y_flux + 0.35)
                ax.set_yticklabels(sub['Group'].tolist())
                ax.set_xlabel('Centroid shift \n [hrs]')
                ax.set_title(var)
                ax.grid(axis='x', alpha=0.3, lw=1.5)
                ax.spines[['top', 'right']].set_visible(False)
                ax.tick_params(axis='both', width=2.2, length=6)
                ax.tick_params(axis='y', labelsize=max(14, min(32, 380 / n)))
                ax.text(-0.15, 1.03, f'({panel_labels[label_idx]})',
                        transform=ax.transAxes, fontsize=70, fontweight='bold')
                label_idx += 1
                if col == 0:
                    ax.text(-0.50, 0.5, row_title, transform=ax.transAxes,
                            fontsize=52, rotation=90, va='center', ha='center')

        fig.text(0.5, 0.025, '(Drought − Non-Drought)', ha='center', fontsize=40)

        fig.subplots_adjust(top=0.83, bottom=0.15)

        eco_p  = mpatches.Patch(color=_ECO_COLOR,  label='ECOCO3')
        flux_p = mpatches.Patch(color=_FLUX_COLOR, label='FLUXNET')
        sig_p  = plt.Line2D([], [], color='gray', marker='$*$', linestyle='None',
                            markersize=14, label='* CI excludes 0')
        fig.legend(handles=[eco_p, flux_p, sig_p], loc='upper right',
                   frameon=False, bbox_to_anchor=(0.99, 0.92), fontsize=28)
        fig.suptitle(
            'Diurnal Centroid Timing Shift Under Drought (95% Bootstrap CI)\n'
            'Positive = later peak under drought  |  Negative = earlier peak',
            fontsize=57, y=0.995)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()


def plot_seasonal_offset_summary(
    per_group_path='tables/seasonal_cycle_metrics_bootstrap_CI.csv',
    aggregate_path='tables/seasonal_cycle_metrics_aggregate.csv',
    metric='offset',
    output_path=None
):
    """Forest plot of the per-group seasonal peak-timing offset (metric=
    'offset') or amplitude difference (metric='pct_amp_diff') between
    FLUXNET and ECOCO3, with 95% bootstrap CIs -- one point per group (this
    metric is already a FLUXNET-vs-ECOCO3 comparison, unlike
    plot_centroid_shift_summary's two-marker-per-group layout). An
    'ALL GROUPS' row shows the across-group bootstrap mean (see the
    two-level bootstrap in the seasonal-metrics scripts), separated from the
    per-group rows by a gap.
    """
    info = _SEASONAL_METRIC_INFO[metric]
    per_group = pd.read_csv(per_group_path)
    aggregate = pd.read_csv(aggregate_path).set_index('Variable')

    GROUPINGS = [('By Vegetation Type', 'Veg'), ('By Climate Class', 'kg_label')]
    row_n = [max(per_group[per_group['Grouping'] == g]['Group'].nunique(), 1) + 2
             for _, g in GROUPINGS]

    with plt.rc_context(_RCPARAMS):
        fig = plt.figure(figsize=(30, 32))
        gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.75, height_ratios=row_n)
        panel_labels = list('abcdef')
        label_idx = 0

        for row, (row_title, grouping) in enumerate(GROUPINGS):
            for col, var in enumerate(_VARIABLES):
                ax = fig.add_subplot(gs[row, col])
                sub = per_group[(per_group['Grouping'] == grouping) &
                                 (per_group['Variable'] == var)].copy()
                sub = sub.sort_values(info['mean_col'], ascending=True)
                n = len(sub)
                if n == 0:
                    ax.set_visible(False)
                    label_idx += 1
                    continue

                y_groups = np.arange(n)
                y_agg = n + 1.8  # gap above the per-group rows

                for i, (_, r) in enumerate(sub.iterrows()):
                    mean, lo, hi = r[info['mean_col']], r[info['lo_col']], r[info['hi_col']]
                    xerr = [[max(mean - lo, 0)], [max(hi - mean, 0)]]
                    ax.errorbar(mean, y_groups[i], xerr=xerr, fmt='o', color=_SEASONAL_POINT_COLOR,
                                markersize=_MARKER_SIZE, capsize=_CAP_SIZE, lw=_LINE_WIDTH,
                                elinewidth=_ELINE_WIDTH, capthick=_ELINE_WIDTH, zorder=3)
                    s = _sig_star(lo - info['ref'], hi - info['ref'])
                    if s:
                        ax.text(mean, y_groups[i] + 0.35, s, ha='center',
                                fontsize=_STAR_SIZE, color=_SEASONAL_POINT_COLOR, zorder=4)

                if var in aggregate.index:
                    a = aggregate.loc[var]
                    amean, alo, ahi = a[info['mean_col']], a[info['lo_col']], a[info['hi_col']]
                    xerr = [[max(amean - alo, 0)], [max(ahi - amean, 0)]]
                    ax.errorbar(amean, y_agg, xerr=xerr, fmt='D', color=_SEASONAL_AGG_COLOR,
                                markersize=_MARKER_SIZE * 1.15, capsize=_CAP_SIZE, lw=_LINE_WIDTH,
                                elinewidth=_ELINE_WIDTH, capthick=_ELINE_WIDTH, zorder=5)
                    s = _sig_star(alo - info['ref'], ahi - info['ref'])
                    if s:
                        ax.text(amean, y_agg + 0.35, s, ha='center',
                                fontsize=_STAR_SIZE, color=_SEASONAL_AGG_COLOR, zorder=6)

                ax.axhline(n + 0.5, color='gray', lw=1.5, ls=':', alpha=0.6)
                ax.axvline(info['ref'], color='black', lw=_VLINE_WIDTH, ls='--', alpha=0.5)
                ax.set_ylim(-1, y_agg + 1.1)  # headroom above the aggregate point/star, clear of the title
                ax.set_yticks(list(y_groups) + [y_agg])
                ax.set_yticklabels(sub['Group'].tolist() + ['ALL GROUPS'])
                ax.set_xlabel(info['xlabel'], fontsize=30)
                ax.set_title(var)
                ax.grid(axis='x', alpha=0.3, lw=1.5)
                ax.spines[['top', 'right']].set_visible(False)
                ax.tick_params(axis='both', width=2.2, length=6)
                ax.tick_params(axis='x', labelsize=28)
                ax.tick_params(axis='y', labelsize=max(14, min(32, 380 / n)))
                ax.text(-0.15, 1.03, f'({panel_labels[label_idx]})',
                        transform=ax.transAxes, fontsize=70, fontweight='bold')
                label_idx += 1
                if col == 0:
                    ax.text(-0.62, 0.5, row_title, transform=ax.transAxes,
                            fontsize=52, rotation=90, va='center', ha='center')

        fig.subplots_adjust(top=0.85, bottom=0.1)

        group_p = plt.Line2D([], [], color=_SEASONAL_POINT_COLOR, marker='o', linestyle='None',
                             markersize=18, label='Individual group')
        agg_p = plt.Line2D([], [], color=_SEASONAL_AGG_COLOR, marker='D', linestyle='None',
                           markersize=18, label='All groups (pooled)')
        sig_p = plt.Line2D([], [], color='gray', marker='$*$', linestyle='None',
                           markersize=14, label='* CI excludes reference')
        fig.legend(handles=[group_p, agg_p, sig_p], loc='upper right',
                   frameon=False, bbox_to_anchor=(0.99, 0.94), fontsize=28)
        fig.suptitle(info['suptitle'], fontsize=57, y=1.0)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()