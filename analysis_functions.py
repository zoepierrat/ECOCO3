"""
Shared analysis helpers used by Analysis.ipynb: coverage stats, FLUXNET/ECOCO3
site matching, and the FLUXNET-vs-ECOCO3 scatter comparison figure.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ====================================================================
# COVERAGE STATS
# ====================================================================

def category_stats(df, group_col):
    """Return scenes and site-years counts per category."""
    year = df['TIMESTAMP'].dt.year
    scenes = df.groupby(group_col).size().rename('n_scenes')
    site_years = (
        df.assign(_year=year)
          .drop_duplicates(subset=[group_col, 'Site', '_year'])
          .groupby(group_col).size()
          .rename('n_site_years')
    )
    return pd.concat([scenes, site_years], axis=1).fillna(0).astype(int)


def filter_valid(eco_stats, flux_stats, min_scenes, min_site_years):
    eco_ok = eco_stats[(eco_stats['n_scenes'] >= min_scenes) &
                        (eco_stats['n_site_years'] >= min_site_years)].index
    flux_ok = flux_stats[(flux_stats['n_scenes'] >= min_scenes) &
                          (flux_stats['n_site_years'] >= min_site_years)].index
    return sorted(set(eco_ok) & set(flux_ok))


def sufficient_drought_days(df, group_col, bins, min_days=5, date_col='TIMESTAMP'):
    """Categories (values in group_col) where every bin in `bins` has more
    than `min_days` unique dates within that category.

    `bins` is the same (label, condition) list passed to
    plot_scripts.plot_diurnal_cycles_spei — condition(df) returns a boolean
    mask. Matches that function's own >5-day gate on its bootstrap CIs, so a
    category failing here can't produce a stable drought-vs-non-drought
    estimate there either — filter a dataset's category list through this
    before looping over it to drop categories that dataset can't support,
    even if another dataset (e.g. FLUXNET) has enough for the same category.
    """
    df = df.copy()
    df['_date'] = pd.to_datetime(df[date_col]).dt.date
    keep = []
    for cat, sub in df.groupby(group_col):
        ok = all(sub.loc[condition(sub), '_date'].nunique() > min_days for _, condition in bins)
        if ok:
            keep.append(cat)
    return sorted(keep)


def print_summary(eco_stats, flux_stats, valid_list, label, min_scenes, min_site_years):
    all_cats = eco_stats.index.union(flux_stats.index)
    print(f"\n{label}  (MIN_SCENES={min_scenes}, MIN_SITE_YEARS={min_site_years})")
    print(f"  {'':8s}  {'ECOCO3':>16s}          {'FLUXNET':>16s}")
    print(f"  {'':8s}  {'scenes':>8s} {'site-yr':>8s}   {'scenes':>8s} {'site-yr':>8s}   kept")
    print("  " + "-" * 62)
    for cat in sorted(all_cats):
        es  = eco_stats.loc[cat,  'n_scenes']     if cat in eco_stats.index  else 0
        esy = eco_stats.loc[cat,  'n_site_years'] if cat in eco_stats.index  else 0
        fs  = flux_stats.loc[cat, 'n_scenes']     if cat in flux_stats.index else 0
        fsy = flux_stats.loc[cat, 'n_site_years'] if cat in flux_stats.index else 0
        mark = 'Y' if cat in valid_list else '-'
        print(f"  {cat:<8s}  {es:>8,d} {esy:>8d}   {fs:>8,d} {fsy:>8d}   {mark}")

# ====================================================================
# SITE MATCHING / COINCIDENCE
# ====================================================================

def latlon_to_xyz(lat, lon):
    lat_r, lon_r = np.radians(lat), np.radians(lon)
    return np.column_stack([
        np.cos(lat_r) * np.cos(lon_r),
        np.cos(lat_r) * np.sin(lon_r),
        np.sin(lat_r),
    ])


def build_tower_pixel_matches(df_flux, df_eco, radius_km, lat_col='Lat', lon_col='Lon'):
    """Every (tower, ECOCO3 pixel-location) pair where the pixel's IGBP-derived
    vegetation class matches the tower's designated vegetation type and the
    pixel falls within radius_km of the tower.

    One row per qualifying pair — a pixel location can match more than one
    tower if towers happen to be clustered within radius_km of each other.
    """
    EARTH_RADIUS_KM = 6371.0
    flux_sites = df_flux[[lat_col, lon_col, 'Veg']].drop_duplicates().reset_index(drop=True)
    eco_sites  = df_eco[[lat_col, lon_col, 'Veg']].drop_duplicates().reset_index(drop=True)

    flux_tree = cKDTree(latlon_to_xyz(flux_sites[lat_col].values, flux_sites[lon_col].values))
    eco_tree  = cKDTree(latlon_to_xyz(eco_sites[lat_col].values, eco_sites[lon_col].values))
    max_chord = radius_km / EARTH_RADIUS_KM

    nearby = flux_tree.query_ball_tree(eco_tree, r=max_chord)

    rows = []
    for fi, eco_idxs in enumerate(nearby):
        tower_lat, tower_lon, tower_veg = flux_sites.loc[fi, [lat_col, lon_col, 'Veg']]
        for ei in eco_idxs:
            if eco_sites.loc[ei, 'Veg'] == tower_veg:
                rows.append((tower_lat, tower_lon, eco_sites.loc[ei, lat_col], eco_sites.loc[ei, lon_col], tower_veg))

    return pd.DataFrame(rows, columns=['site_lat', 'site_lon', lat_col, lon_col, 'Veg'])


def make_coincident(df_flux, df_eco, pixel_matches, tolerance, normalize_dates=False,
                     lat_col='Lat', lon_col='Lon', extra_eco_cols=None):
    """Restrict df_eco to pixels in pixel_matches (same vegetation as their
    matched tower, within radius_km — see build_tower_pixel_matches), tag each
    with that tower's canonical site_lat/site_lon, then attach the
    nearest-in-time FLUX reading (± tolerance) to every matched ECO pixel.

    ECO is kept as the base (left) frame so every matching pixel is preserved
    as its own row — a tower with several nearby same-vegetation pixels ends
    up compared against all of them, not just the closest one.

    normalize_dates: floor both timestamps to midnight before merging.
    Use for daily data — ECOCO3 daily timestamps carry the overpass time
    (~13:30), so without normalization merge_asof can match FLUX day N+1
    (00:00, 10.5 h away) instead of the correct ECO day N+1 (13:30, 13.5 h).
    Pair with tolerance=pd.Timedelta('12h') to enforce same-calendar-day only.

    extra_eco_cols: additional ECOCO3-side columns to carry through unsuffixed
    (e.g. ['SIF'] to pair raw SIF against the matched FLUXNET GPP for fitting
    an empirical SIF->GPP conversion — FLUXNET has no equivalent column so
    these aren't part of the merge key and don't get an _ECO/_FLUX suffix).
    """
    eco = df_eco.merge(pixel_matches, on=[lat_col, lon_col, 'Veg'], how='inner')
    flux = df_flux.copy()
    flux['site_lat'] = flux[lat_col]
    flux['site_lon'] = flux[lon_col]

    flux['TIMESTAMP'] = pd.to_datetime(flux['TIMESTAMP'])
    eco['TIMESTAMP']  = pd.to_datetime(eco['TIMESTAMP'])

    if normalize_dates:
        flux['TIMESTAMP'] = flux['TIMESTAMP'].dt.normalize()
        eco['TIMESTAMP']  = eco['TIMESTAMP'].dt.normalize()

    extra_eco_cols = extra_eco_cols or []
    flux = flux[['site_lat', 'site_lon', 'TIMESTAMP', 'GPP', 'ET', 'WUE', 'Veg']].sort_values('TIMESTAMP')
    eco  = eco[['site_lat', 'site_lon', 'TIMESTAMP', 'GPP', 'ET', 'WUE', 'Veg'] + extra_eco_cols].sort_values('TIMESTAMP')

    return pd.merge_asof(
        eco, flux,
        on='TIMESTAMP', by=['site_lat', 'site_lon', 'Veg'],
        tolerance=tolerance, direction='nearest',
        suffixes=('_ECO', '_FLUX'),
    ).dropna()


def assign_time_of_day(ts):
    hour = ts.hour
    if   6  <= hour < 10: return 'Morning'
    elif 10 <= hour < 14: return 'Midday'
    elif 14 <= hour < 18: return 'Afternoon'
    else: return np.nan

# ====================================================================
# FLUXNET vs ECOCO3 COMPARISON PLOT
# ====================================================================

UNITS = {
    'hh': {
        'GPP': 'µmol CO$_2$ m$^{-2}$ s$^{-1}$',
        'ET':  'W m$^{-2}$',
        'WUE': 'gC kg$^{-1}$ H$_2$O',
    },
    'daily': {
        'GPP': 'gC m$^{-2}$ day$^{-1}$',
        'ET':  'mm day$^{-1}$',
        'WUE': 'gC kg$^{-1}$ H$_2$O',
    },
}


def plot_flux_vs_eco(df_hh, df_daily=None, output_path=None):
    datasets = [('Half-hourly', df_hh, 'hh')]
    if df_daily is not None and len(df_daily) > 0:
        datasets.append(('Daily', df_daily, 'daily'))

    nrows = len(datasets)
    variables = ['GPP', 'ET', 'WUE']
    panel_labels = 'abcdefghi'

    fig, axes = plt.subplots(nrows, 3, figsize=(16, 8 * nrows), squeeze=False,
                              gridspec_kw={'hspace': 0.9})

    for row, (label, df, res) in enumerate(datasets):
        color_vals  = df['TIMESTAMP'].dt.hour  if res == 'hh' else df['TIMESTAMP'].dt.month
        color_label = 'Hour of Day'            if res == 'hh' else 'Month'
        # Both hour-of-day and month are cyclic, so use matplotlib's cyclic
        # colormaps rather than a sequential one like viridis: twilight_shifted
        # for the diurnal (hour) cycle, twilight for the seasonal (month) cycle.
        cmap_name   = 'twilight_shifted'       if res == 'hh' else 'twilight'

        # Position colorbar to the right of each row
        row_bottom = (nrows - 1 - row) / nrows + 0.01 / nrows
        row_height = 1 / nrows * 0.70
        cbar_ax = fig.add_axes([0.92, row_bottom, 0.015, row_height])

        for i, var in enumerate(variables):
            ax = axes[row, i]
            x = df[f'{var}_FLUX']
            y = df[f'{var}_ECO']
            mask = np.isfinite(x) & np.isfinite(y)
            x, y, c = x[mask], y[mask], color_vals[mask]

            sc = ax.scatter(x, y, c=c, cmap=cmap_name, alpha=0.6, s=30)

            lim = [min(x.min(), y.min()), max(x.max(), y.max())]
            ax.plot(lim, lim, '--', lw=2, color='black', label='1:1')

            if len(x) > 5:
                r2    = np.corrcoef(x, y)[0, 1] ** 2
                rmse  = np.sqrt(np.mean((y - x) ** 2))
                bias  = np.mean(y - x)
                slope, intercept = np.polyfit(x, y, 1)
                lim_arr = np.array(lim)
                ax.plot(lim_arr, slope * lim_arr + intercept, '-', lw=2,
                        color='gray', alpha=0.8, label='Best fit')
                ax.legend(fontsize=12, loc='lower right', framealpha=0.8)
                ax.text(0.5, -0.3,
                        rf"R$^2$ = {r2:.2f}" "\n"
                        rf"RMSE = {rmse:.2f} {UNITS[res][var]}" "\n"
                        rf"Bias = {bias:.2f} {UNITS[res][var]}" "\n"
                        rf"Slope = {slope:.2f}",
                        transform=ax.transAxes, fontsize=21,
                        va='top', ha='center')

            ax.set_xlabel('FLUXNET', fontsize=21)
            ax.set_ylabel('ECOCO3', fontsize=21)
            ax.set_title(f'{label} — {var} \n [{UNITS[res][var]}]', fontsize=23)
            ax.tick_params(axis='both', labelsize=18)
            ax.set_aspect('equal', adjustable='box')
            ax.text(0.02, 0.98, f'({panel_labels[row * 3 + i]})',
                    transform=ax.transAxes, fontsize=26,
                    fontweight='bold', va='top', ha='left',
                    bbox=dict(facecolor='white', alpha=1.0, edgecolor='none', pad=4))

        cbar = fig.colorbar(sc, cax=cbar_ax, shrink=0.5)
        cbar.set_label(color_label, fontsize=21)
        cbar.ax.tick_params(labelsize=18)
        cbar.ax.invert_yaxis()  # small values on top, large on bottom — reads top-to-bottom as small -> large

    plt.subplots_adjust(right=0.90, wspace=0.27, hspace=0)
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

# ====================================================================
# PER-SITE MAP + COLOR-BY-SITE VARIANT
# ====================================================================
# Sites in a coincident df carry only site_lat/site_lon (the tower's own,
# un-modified Lat/Long from the FLUXNET metadata), so a straight lat/lon
# merge against the metadata table recovers the Site ID exactly.

def attach_site_id(df, metadata, lat_col='site_lat', lon_col='site_lon'):
    """Tag each row with its FLUXNET Site ID via exact lat/lon match."""
    lut = metadata[['Site', 'Lat', 'Long']].drop_duplicates(subset=['Lat', 'Long'])
    out = df.merge(lut, left_on=[lat_col, lon_col], right_on=['Lat', 'Long'], how='left')
    return out.drop(columns=['Lat', 'Long'])


def assign_site_colors(site_ids, seed=0):
    """One visually distinct color per site, drawn from a high-resolution
    colormap in shuffled order so spatially clustered sites (e.g. the
    several Wisconsin towers) don't get similar hues just because they're
    adjacent in the input list."""
    site_ids = sorted(set(site_ids))
    n = len(site_ids)
    cmap = plt.get_cmap('tab20')
    positions = np.linspace(0.02, 0.98, n)
    np.random.RandomState(seed).shuffle(positions)
    return {site: cmap(pos) for site, pos in zip(site_ids, positions)}


def plot_flux_vs_eco_by_site_map(df_hh, df_daily=None, metadata=None, output_path=None):
    """Combines plot_coincident_site_map and plot_flux_vs_eco_by_site into a
    single figure: a world map of the FLUXNET tower sites on top (doubling
    as the color key), and the FLUXNET-vs-ECOCO3 comparison scatter grid
    below it, colored by that same site -> color mapping -- no more needing
    two separate figures to interpret the site colors in one.
    """
    datasets = [('Half-hourly', df_hh, 'hh')]
    if df_daily is not None and len(df_daily) > 0:
        datasets.append(('Daily', df_daily, 'daily'))

    tagged = {res: attach_site_id(df, metadata) for _, df, res in datasets}
    all_sites = pd.concat([d['Site'] for d in tagged.values()]).dropna()
    site_colors = assign_site_colors(all_sites)

    nrows = len(datasets)
    variables = ['GPP', 'ET', 'WUE']
    panel_labels = 'abcdefghi'

    fig = plt.figure(figsize=(16, 9 + 8.5 * nrows))
    gs = fig.add_gridspec(nrows + 1, 3, height_ratios=[9] + [8.5] * nrows, hspace=0.12, wspace=0.27)

    # --- Map (top row, spans all columns) ---
    ax_map = fig.add_subplot(gs[0, :], projection=ccrs.PlateCarree())
    site_locs = pd.concat([df[['site_lat', 'site_lon']] for _, df, _ in datasets]).drop_duplicates()
    sites = attach_site_id(site_locs, metadata)
    map_colors = [site_colors.get(s, 'lightgray') for s in sites['Site']]

    ax_map.set_global()
    ax_map.add_feature(cfeature.LAND, color='lightgray')
    ax_map.add_feature(cfeature.OCEAN, color='lightblue')
    ax_map.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax_map.add_feature(cfeature.BORDERS, linestyle=':')
    ax_map.axhline(51.6, color='blue', linestyle='--', linewidth=1)
    ax_map.axhline(-51.6, color='blue', linestyle='--', linewidth=1)
    ax_map.scatter(sites['site_lon'], sites['site_lat'], s=120, c=map_colors,
                    edgecolor='black', linewidth=0.8, zorder=5, transform=ccrs.PlateCarree())
    ax_map.set_title(f'FLUXNET Sites in the ECOCO3 Validation Comparison (n={len(sites)})\n'
                      'color key for the panels below', fontsize=20)
    ax_map.text(-0.05, 1, '(a)', transform=ax_map.transAxes, fontsize=24,
                fontweight='bold', va='top', ha='left')

    # --- Scatter comparison grid (below) ---
    label_idx = 1  # (a) used by the map above
    for row, (label, _, res) in enumerate(datasets):
        df = tagged[res]
        for i, var in enumerate(variables):
            ax = fig.add_subplot(gs[row + 1, i])
            x = df[f'{var}_FLUX']
            y = df[f'{var}_ECO']
            mask = np.isfinite(x) & np.isfinite(y) & df['Site'].notna()
            x, y, site = x[mask], y[mask], df['Site'][mask]
            colors = [site_colors[s] for s in site]

            ax.scatter(x, y, c=colors, alpha=0.7, s=30, edgecolor='none')

            lim = [min(x.min(), y.min()), max(x.max(), y.max())]
            ax.plot(lim, lim, '--', lw=2, color='black', label='1:1')

            if len(x) > 5:
                r2 = np.corrcoef(x, y)[0, 1] ** 2
                rmse = np.sqrt(np.mean((y - x) ** 2))
                bias = np.mean(y - x)
                slope, intercept = np.polyfit(x, y, 1)
                lim_arr = np.array(lim)
                ax.plot(lim_arr, slope * lim_arr + intercept, '-', lw=2,
                        color='gray', alpha=0.8, label='Best fit')
                ax.legend(fontsize=12, loc='lower right', framealpha=0.8)
                ax.text(0.5, -0.3,
                        rf"R$^2$ = {r2:.2f}" "\n"
                        rf"RMSE = {rmse:.2f} {UNITS[res][var]}" "\n"
                        rf"Bias = {bias:.2f} {UNITS[res][var]}" "\n"
                        rf"Slope = {slope:.2f}",
                        transform=ax.transAxes, fontsize=19,
                        va='top', ha='center')

            ax.set_xlabel('FLUXNET', fontsize=21)
            ax.set_ylabel('ECOCO3', fontsize=21)
            ax.set_title(f'{label} — {var} \n [{UNITS[res][var]}]', fontsize=23)
            ax.tick_params(axis='both', labelsize=18)
            ax.set_aspect('equal', adjustable='box')
            ax.text(0.02, 0.98, f'({panel_labels[label_idx]})',
                    transform=ax.transAxes, fontsize=26,
                    fontweight='bold', va='top', ha='left',
                    bbox=dict(facecolor='white', alpha=1.0, edgecolor='none', pad=4))
            label_idx += 1

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

    return sites.assign(color=map_colors), site_colors


def compute_group_error_stats(df_hh, df_daily=None, metadata=None,
                               group_cols=('Site', 'kg_label', 'Veg', 'TOD'), min_n=5):
    """Same R2/RMSE/Bias/Slope metrics shown in Figure 2, broken out per
    category instead of pooled across all sites. One row per
    (resolution, grouping, group value, variable).

    Site and kg_label (FLUXNET metadata's Köppen climate class) aren't
    carried through make_coincident, so both are attached here from
    `metadata` via an exact Lat/Long match (same join attach_site_id already
    uses for Site). Veg rides along in the coincident data already; TOD
    (time-of-day) only exists on the half-hourly frame — daily data has one
    value per day, so TOD is skipped for the 'Daily' resolution.

    Categories with <= min_n coincident points are dropped (matches the
    same >5 gate Figure 2's own panels use before quoting a metric).
    """
    datasets = [('Half-hourly', df_hh, 'hh')]
    if df_daily is not None and len(df_daily) > 0:
        datasets.append(('Daily', df_daily, 'daily'))

    variables = ['GPP', 'ET', 'WUE']
    kg_lut = metadata[['Lat', 'Long', 'kg_label']].drop_duplicates(subset=['Lat', 'Long'])

    rows = []
    for label, df, res in datasets:
        df = attach_site_id(df, metadata)
        df = df.merge(kg_lut, left_on=['site_lat', 'site_lon'], right_on=['Lat', 'Long'], how='left')
        df = df.drop(columns=['Lat', 'Long'])

        for group_col in group_cols:
            if group_col == 'TOD' and res != 'hh':
                continue
            if group_col not in df.columns:
                continue
            for group_val, sub in df.groupby(group_col):
                for var in variables:
                    x = sub[f'{var}_FLUX']
                    y = sub[f'{var}_ECO']
                    mask = np.isfinite(x) & np.isfinite(y)
                    x, y = x[mask], y[mask]
                    if len(x) <= min_n:
                        continue
                    rows.append({
                        'Resolution': label, 'Grouping': group_col, 'Group': group_val,
                        'Variable': var, 'n': len(x),
                        'R2':    np.corrcoef(x, y)[0, 1] ** 2,
                        'RMSE':  np.sqrt(np.mean((y - x) ** 2)),
                        'Bias':  np.mean(y - x),
                        'Slope': np.polyfit(x, y, 1)[0],
                    })

    return pd.DataFrame(rows)


def _contrast_text_color(rgba):
    r, g, b = rgba[:3]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return 'black' if luminance > 0.6 else 'white'


def plot_group_error_summary(group_stats, resolution='Half-hourly', metric='R2', output_path=None):
    """Heatmap summary of compute_group_error_stats: one panel per grouping
    (Site, climate, vegetation, time-of-day), columns = GPP/ET/WUE, color =
    `metric`. R2 is the default because it's unit-free and so is the only
    one of the four metrics comparable across GPP/ET/WUE on one color scale
    (RMSE/Bias carry each variable's own units).

    Rows are sorted by mean value across variables, descending — except
    TOD, which keeps its natural Morning/Midday/Afternoon order since that
    sequence is itself meaningful. Each cell is annotated with the metric
    value and the underlying n, since several categories here have small
    sample sizes (e.g. n<10) that a color alone would hide.
    """
    groupings = ['Site', 'kg_label', 'Veg', 'TOD']
    titles = {'Site': 'By Site', 'kg_label': 'By Climate',
              'Veg': 'By Vegetation', 'TOD': 'By Time of Day'}
    panel_labels = 'abcd'
    variables = ['GPP', 'ET', 'WUE']
    tod_order = ['Morning', 'Midday', 'Afternoon']

    df = group_stats[group_stats['Resolution'] == resolution]
    vmin, vmax = (0, 1) if metric == 'R2' else (df[metric].min(), df[metric].max())
    cmap = plt.get_cmap('viridis')

    fig, axes = plt.subplots(2, 2, figsize=(20, 24))
    axes = axes.flatten()

    for idx, (ax, g) in enumerate(zip(axes, groupings)):
        sub = df[df['Grouping'] == g]
        pivot = sub.pivot(index='Group', columns='Variable', values=metric).reindex(columns=variables)
        n_pivot = sub.pivot(index='Group', columns='Variable', values='n').reindex(columns=variables)

        if g == 'TOD':
            pivot, n_pivot = pivot.reindex(tod_order), n_pivot.reindex(tod_order)
        else:
            order = pivot.mean(axis=1).sort_values(ascending=False).index
            pivot, n_pivot = pivot.loc[order], n_pivot.loc[order]

        im = ax.imshow(pivot.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_xticks(range(len(variables)))
        ax.set_xticklabels(variables, fontsize=30)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=27)
        ax.set_title(titles[g], fontsize=35, pad=18)
        ax.tick_params(length=0)
        ax.text(-0.10, 1.07, f'({panel_labels[idx]})', transform=ax.transAxes,
                fontsize=38, fontweight='bold', va='bottom', ha='left')

        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val, n = pivot.values[i, j], n_pivot.values[i, j]
                if np.isnan(val):
                    continue
                color = _contrast_text_color(cmap((val - vmin) / (vmax - vmin)))
                ax.text(j, i, f"{val:.2f}\n(n={int(n)})", ha='center', va='center',
                        color=color, fontsize=23)

        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label(metric if metric != 'R2' else r'R$^2$', fontsize=26)
        cbar.ax.tick_params(labelsize=21)

    fig.suptitle('Error stats by category for coincident observations',
                 fontsize=37, y=1.02)
    plt.tight_layout(h_pad=6, w_pad=4)
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()


# ====================================================================
# PHASE-ANGLE PLOT HELPERS
# ====================================================================

