from matplotlib.colors import LogNorm
from shapely import Point
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
from pathlib import Path
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as mpatches


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


# ======================================================
# APPLY LOWESS SMOOTHING (GENERIC)
# ======================================================
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


def plot_seasonal_cycles(
    df_plot,
    variable,
    veg_color_palette=veg_color_palette,
    koppen_label_color_palette=koppen_label_color_palette,
    output_base=None,
    frac=0.2, 
    valid_veg=None,
    valid_kg=None
):

    df_plot = df_plot.copy()
    df_plot['DOY'] = df_plot['TIMESTAMP'].dt.dayofyear

    # ======================================================
    # DAILY MEANS PER SITE
    # ======================================================
    df_plot = df_plot[
    df_plot['Veg'].isin(valid_veg) &
    df_plot['kg_label'].isin(valid_kg)]

    daily_site = (
        df_plot
        .groupby(['DOY', 'Site', 'Veg', 'kg_label'])[variable]
        .mean()
        .reset_index()
    )

    # ======================================================
    # SEASONAL MEANS (VEGETATION & KÖPPEN)
    # ======================================================

    veg_doy = (
        df_plot[df_plot['Veg'].isin(valid_veg)]
        .groupby(['DOY', 'Veg'])[variable]
        .mean()
        .reset_index()
        .groupby('Veg')
        .apply(apply_lowess, variable=variable, frac=frac)
    )

    kg_doy = (
        df_plot[df_plot['kg_label'].isin(valid_kg)]
        .groupby(['DOY', 'kg_label'])[variable]
        .mean()
        .reset_index()
        .groupby('kg_label')
        .apply(apply_lowess, variable=variable, frac=frac)
    )

    # ======================================================
    # TWO-PANEL SEASONAL CYCLE FIGURE
    # ======================================================
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # (a) Vegetation
    for veg, d in veg_doy.groupby('Veg'):
        axes[0].plot(
            d['DOY'],
            d[f'{variable}_smooth'],
            color=veg_color_palette.get(veg, 'gray'),
            label=veg,
            linewidth=3
        )
    axes[0].set_title(f'(a) Seasonal cycle of {variable} by vegetation')
    axes[0].set_ylabel(variable)
    axes[0].set_xlim(1, 366)
    axes[0].legend(ncol=2, fontsize=8, frameon=False)

    # (b) Köppen climate
    for kg, d in kg_doy.groupby('kg_label'):
        axes[1].plot(
            d['DOY'],
            d[f'{variable}_smooth'],
            color=koppen_label_color_palette.get(kg, 'gray'),
            label=kg,
            linewidth=3
        )
    axes[1].set_title(f'(b) Seasonal cycle of {variable} by climate')
    axes[1].set_xlabel('Day of Year')
    axes[1].set_ylabel(variable)
    axes[1].set_xlim(1, 366)
    axes[1].legend(ncol=2, fontsize=8, frameon=False)

    plt.tight_layout()
    plt.savefig(output_base / f"seasonal_{variable}_all.png", dpi=300)
    plt.show()

    # ======================================================
    # SITE-LEVEL SEASONAL CYCLES BY VEGETATION
    # ======================================================
    veg_out = output_base / "veg" / variable
    veg_out.mkdir(parents=True, exist_ok=True)

    for veg, dveg in daily_site.groupby('Veg'):
        fig, ax = plt.subplots(figsize=(10, 5))

        for site, dsite in dveg.groupby('Site'):
            ax.scatter(
                dsite['DOY'],
                dsite[variable],
                color=veg_color_palette.get(veg, 'gray'),
                alpha=0.4,
                s=15
            )

            mean_doy = (
                dveg.groupby('DOY')[variable]
                .mean()
                .reset_index()
                .sort_values('DOY')
            )

            mean_smooth = sm.nonparametric.lowess(
                endog=mean_doy[variable],
                exog=mean_doy['DOY'],
                frac=0.2,
                return_sorted=False
            )

            ax.plot(
                mean_doy['DOY'],
                mean_smooth,
                color='black',
                linewidth=3,
            )


        ax.set_title(f"Seasonal cycle of {variable} – {veg}")
        ax.set_xlabel("Day of Year")
        ax.set_ylabel(variable)
        ax.set_xlim(1, 366)
        ax.legend(frameon=False)

        plt.tight_layout()
        plt.savefig(veg_out / f"seasonal_{variable}_Veg_{veg}.png", dpi=300)
        plt.close()

    # ======================================================
    # SITE-LEVEL SEASONAL CYCLES BY KÖPPEN CLIMATE
    # ======================================================
    kg_out = output_base / "koppen" / variable
    kg_out.mkdir(parents=True, exist_ok=True)

    for kg, dkg in daily_site.groupby('kg_label'):
        fig, ax = plt.subplots(figsize=(10, 5))

        for site, dsite in dkg.groupby('Site'):
            ax.scatter(
                dsite['DOY'],
                dsite[variable],   # ✅ fixed
                color=koppen_label_color_palette.get(kg, 'gray'),
                alpha=0.4,
                s=15
            )

        mean_doy = (
            dkg.groupby('DOY')[variable]
            .mean()
            .reset_index()
            .sort_values('DOY')
        )   
        mean_smooth = sm.nonparametric.lowess(
            endog=mean_doy[variable],
            exog=mean_doy['DOY'],
            frac=0.2,
            return_sorted=False
        )   

        ax.plot(
            mean_doy['DOY'],
            mean_smooth,
            color='black',
            linewidth=3,
        )

        ax.set_title(f"Seasonal cycle of {variable} – {kg}")
        ax.set_xlabel("Day of Year")
        ax.set_ylabel(variable)
        ax.set_xlim(1, 366)
        ax.legend(frameon=False)

        plt.tight_layout()
        plt.savefig(kg_out / f"seasonal_{variable}_KG_{kg}.png", dpi=300)
        plt.close()

def get_group_letters(tukey_result):
    res_df = pd.DataFrame(
        tukey_result._results_table.data[1:],
        columns=tukey_result._results_table.data[0]
    )

    groups = sorted(list(set(res_df['group1']).union(res_df['group2'])))

    # Build significance matrix
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

# Function to handle the ANOVA, Tukey HSD, and plotting
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

    # --- Group Letters ---
    group_letters = get_group_letters(tukey_result)

    # --- Sorting by Median WUE ---
    median_wue_order = (
        df_wue_daily.groupby(grouping_category)["WUE"]
        .median()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    # --- Violin plot for the given dataset ---
    sns.violinplot(x=grouping_category, y='WUE', data=df_wue_daily, palette=color_map, 
                    inner='quartile', order=median_wue_order ,scale='area', cut=0, ax=ax)

    # Add significance letters
    for i, vegetation in enumerate(median_wue_order):
        y = -0.5  # Adjust this for spacing
        letter = group_letters[vegetation]
        ax.text(i+0.2, y, letter, ha='center', va='bottom', fontsize=16, fontweight='bold')

    ax.set_title(title, fontsize=20)
    ax.set_xlabel(grouping_category, fontsize=16)
    ax.set_ylabel('WUE [gC kg$^{-1}$H$_2$O]', fontsize=16)
    ax.set_ylim(0, 6)  # Adjust as needed
    ax.tick_params(axis='x', rotation=45, labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

# ======================================================
# --- Centroid Calculation Function ---
def compute_diurnal_centroid(df, hour_col='Hour', weight_col='WUE'):
    weights = df[weight_col].values
    hours = df[hour_col].values
    if np.sum(weights) == 0:
        return np.nan
    return np.sum(hours * weights) / np.sum(weights)

def compute_centroids_from_hourly_avg(df, group_col):
    hourly_avg = df.groupby(['Hour', group_col])['WUE'].mean().reset_index()
    centroids = (
        hourly_avg.groupby(group_col)
                  .apply(lambda d: compute_diurnal_centroid(d, hour_col='Hour', weight_col='WUE'))
                  .reset_index(name='Diurnal_Centroid')
    )
    return centroids, hourly_avg

def plot_diurnal_wue(df, group_col=None, palette=None, title='ECOCO Summer WUE Diurnal Profile', legend_title=None):
    if group_col == 'Veg':
        palette = veg_color_palette
    elif group_col == 'kg_label':
        palette = koppen_label_color_palette

    if group_col:
        hourly_avg = df.groupby(['Hour', group_col])['WUE'].mean().reset_index()
        hourly_std = df.groupby(['Hour', group_col])['WUE'].std().reset_index()
        stats = pd.merge(hourly_avg, hourly_std, on=['Hour', group_col], suffixes=('_avg', '_std'))

        # Compute centroids
        centroids = (
            hourly_avg.groupby(group_col)
                      .apply(lambda d: compute_diurnal_centroid(d, hour_col='Hour', weight_col='WUE'))
                      .reset_index(name='Diurnal_Centroid')
        )

    else:
        hourly_avg = df.groupby('Hour')['WUE'].mean().reset_index()
        hourly_std = df.groupby('Hour')['WUE'].std().reset_index()
        stats = pd.merge(hourly_avg, hourly_std, on='Hour', suffixes=('_avg', '_std'))
        centroids = pd.DataFrame({'Diurnal_Centroid': [compute_diurnal_centroid(df)]})

    plt.figure(figsize=(12, 6))

    if group_col:
        sns.lineplot(
            x='Hour',
            y='WUE_avg',
            hue=group_col,
            data=stats,
            marker='o',
            linewidth=2,
            palette=palette
        )
        # Plot centroids for each group without legend entries
        for _, row in centroids.iterrows():
            group = row[group_col]
            centroid_hour = row['Diurnal_Centroid']
            sub = stats[stats[group_col] == group]
            wue_at_centroid = np.interp(centroid_hour, sub['Hour'], sub['WUE_avg'])
            plt.scatter(centroid_hour, wue_at_centroid,
                        color=palette.get(group, 'black'),
                        s=150,
                        edgecolor='black',
                        zorder=10,
                        marker='X',
                        label='_nolegend_')  # omit from legend

        # Add one black 'X' marker for centroid legend
        plt.scatter([], [], color='black', marker='X', s=150, edgecolor='black', label='Diurnal centroid')

    else:
        sns.lineplot(
            x='Hour',
            y='WUE_avg',
            data=stats,
            marker='o',
            color='blue',
            label='Average WUE'
        )
        plt.fill_between(
            stats['Hour'],
            stats['WUE_avg'] - stats['WUE_std'],
            stats['WUE_avg'] + stats['WUE_std'],
            color='blue',
            alpha=0.2,
            label='WUE ± Std Dev'
        )
        # Plot centroid marker with legend
        centroid_hour = centroids['Diurnal_Centroid'].iloc[0]
        wue_at_centroid = np.interp(centroid_hour, stats['Hour'], stats['WUE_avg'])
        plt.scatter(centroid_hour, wue_at_centroid,
                    color='black',
                    s=150,
                    edgecolor='black',
                    marker='X',
                    label='Diurnal centroid')

    plt.title(title, fontsize=24)
    plt.xlabel('Hour of Day', fontsize=20)
    plt.ylabel('Average WUE [gC per kg H₂O]', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlim(0, 24)
    plt.grid(True)
    if group_col:
        plt.legend(title=legend_title, fontsize=12, loc='best')
    else:
        plt.legend(fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_seasonal_cycles_multi(
    df_plot,
    variables,                     # list of 3 variables
    y_labels=None,                 # optional list of 3 y-axis labels
    data_type=None,     # for title only
    group_type='Veg',              # 'Veg' OR 'kg_label'
    veg_color_palette=veg_color_palette,
    koppen_label_color_palette=koppen_label_color_palette,
    frac=0.2,
    ylims = [[-0.5,13],[0,4.5],[0,4.5]],
    valid_veg=None,
    valid_kg=None,
    output_path=None
):

    assert len(variables) == 3, "Please provide exactly 3 variables"

    df_plot = df_plot.copy()
    df_plot['DOY'] = df_plot['TIMESTAMP'].dt.dayofyear

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

    # Filter if needed
    if valid_groups is not None:
        df_plot = df_plot[df_plot[group_col].isin(valid_groups)]

    # -----------------------------------------
    # Create figure
    # -----------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig.suptitle(
        f"Seasonal cycles of {', '.join(variables)} by {title_suffix} for {data_type}",
        fontsize=20
    )

    panel_labels = ['(a)', '(b)', '(c)']

    for i, variable in enumerate(variables):

        doy_group = (
            df_plot
            .groupby(['DOY', group_col])[variable]
            .mean()
            .reset_index()
            .groupby(group_col)
            .apply(apply_lowess, variable=variable, frac=frac)
        )

        ax = axes[i]

        for group, d in doy_group.groupby(group_col):
            ax.plot(
                d['DOY'],
                d[f'{variable}_smooth'],
                color=palette.get(group, 'gray'),
                label=group,
                linewidth=3
            )
        
        # Panel label
        ax.text(
            0.02, 0.92, panel_labels[i],
            transform=ax.transAxes,
            fontsize=14,
            fontweight='bold',
            va='top',
            ha='left'
        )

        ax.set_ylabel(y_labels[i] if y_labels else variable, fontsize=16)
        ax.set_xlim(1, 366)
        ax.set_ylim(ylims[i])

        if i == 0:
            ax.legend(ncol=2, fontsize=16, frameon=False)

    axes[-1].set_xlabel('Day of Year', fontsize=16)

    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=300)

    plt.show()

def plot_merged_diurnal_wue(
    df_flux,
    df_ecoco,
    title='Flux vs ECOCO Summer WUE Diurnal Profile'
):

    plt.figure(figsize=(12, 6))

    # =====================
    # FLUX
    # =====================
    flux_avg = df_flux.groupby('Hour')['WUE'].mean().reset_index()
    flux_std = df_flux.groupby('Hour')['WUE'].std().reset_index()
    flux_stats = pd.merge(flux_avg, flux_std, on='Hour', suffixes=('_avg', '_std'))

    flux_centroid = compute_diurnal_centroid(
        flux_stats,
        hour_col='Hour',
        weight_col='WUE_avg'
    )

    sns.lineplot(
        x='Hour',
        y='WUE_avg',
        data=flux_stats,
        marker='o',
        color = '#2C7FB8',
        linewidth=2,
        label='FLUXNET'
    )

    plt.fill_between(
        flux_stats['Hour'],
        flux_stats['WUE_avg'] - flux_stats['WUE_std'],
        flux_stats['WUE_avg'] + flux_stats['WUE_std'],
        color = '#2C7FB8',
        alpha=0.15
    )

    flux_wue_centroid = np.interp(
        flux_centroid,
        flux_stats['Hour'],
        flux_stats['WUE_avg']
    )

    plt.scatter(
        flux_centroid,
        flux_wue_centroid,
        s=150,
        edgecolor='black',
        facecolor='#2C7FB8',
        marker='X',
        label='FLUXNET centroid'
    )

    # =====================
    # ECOCO
    # =====================
    ecoco_avg = df_ecoco.groupby('Hour')['WUE'].mean().reset_index()
    ecoco_std = df_ecoco.groupby('Hour')['WUE'].std().reset_index()
    ecoco_stats = pd.merge(ecoco_avg, ecoco_std, on='Hour', suffixes=('_avg', '_std'))

    ecoco_centroid = compute_diurnal_centroid(
        ecoco_stats,
        hour_col='Hour',
        weight_col='WUE_avg'
    )

    sns.lineplot(
        x='Hour',
        y='WUE_avg',
        data=ecoco_stats,
        marker='o',
        linewidth=2,
        linestyle='--',
        color = '#D95F0E',
        label='ECOCO3'
    )

    plt.fill_between(
        ecoco_stats['Hour'],
        ecoco_stats['WUE_avg'] - ecoco_stats['WUE_std'],
        ecoco_stats['WUE_avg'] + ecoco_stats['WUE_std'],
        color = '#D95F0E',
        alpha=0.15
    )

    ecoco_wue_centroid = np.interp(
        ecoco_centroid,
        ecoco_stats['Hour'],
        ecoco_stats['WUE_avg']
    )

    plt.scatter(
        ecoco_centroid,
        ecoco_wue_centroid,
        s=150,
        edgecolor='black',
        marker='X',
        facecolor = '#D95F0E',
        label='ECOCO3 centroid'
    )

    # =====================
    # Formatting
    # =====================
    plt.title(title, fontsize=24)
    plt.xlabel('Hour of Day', fontsize=20)
    plt.ylabel('Average WUE [gC per kg H₂O]', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlim(0, 24)
    plt.grid(True)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.show()

def hour_to_timestamp(hour_float):
    h = int(hour_float)
    m = int((hour_float - h) * 60)
    return f"{h:02d}:{m:02d}"

def get_stats(df, var):
    avg = df.groupby('Hour')[var].mean().reset_index()
    std = df.groupby('Hour')[var].std().reset_index()
    return pd.merge(avg, std, on='Hour', suffixes=('_avg', '_std'))

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
    fig.suptitle(title, fontsize=24)

    panel_labels = ['(a)', '(b)', '(c)']

    variables = ['GPP', 'ET', 'WUE']
    y_labels = [
        'GPP [µmol CO$_2$ m$^{-2}$ s$^{-1}$]',
        'ET [W m$^{-2}$]',
        'WUE [gC kg$^{-1}$ H$_2$O]'
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
                fontsize=14,
                fontweight='bold',
                va='top',
                ha='left')
        
        # -------------------------
        ax.text(
            0.98, 0.05,
            f"Δ = {observed_delta:.2f}\n({observed_pct:.1f}%)",
            transform=ax.transAxes,
            ha='right',
            fontsize=16
        )


        ax.set_ylabel(y_labels[i], fontsize=20)
        ax.set_xlim(0, 24)
        ax.grid(True)
        ax.legend(frameon=False, fontsize=18)
        ax.tick_params(axis='both', labelsize=16)

    axes[-1].set_xlabel('Hour of Day', fontsize=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)

    plt.show()

    return pd.DataFrame(results)

def plot_diurnal_cycles_spei(
    df,
    spei_bins,
    title='Diurnal Cycles by SPEI',
    midday_hours=(10,14),
    n_boot=1000
):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
    
    df = df.copy()
    df['Hour'] = df['TIMESTAMP'].dt.hour
    df['Date'] = df['TIMESTAMP'].dt.date
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig.suptitle(title, fontsize=24)
    
    panel_labels = ['(a)', '(b)', '(c)']
    variables = ['GPP', 'ET', 'WUE']
    y_labels = [
        'GPP [µmol CO$_2$ m$^{-2}$ s$^{-1}$]',
        'ET [W m$^{-2}$]',
        'WUE [gC kg$^{-1}$ H$_2$O]'
    ]
    
    colors = ['#D95F0E', '#2C7FB8']
    results_records = []
    
    drought_label, drought_condition = spei_bins[0]
    nondrought_label, nondrought_condition = spei_bins[1]
    
    for i, var in enumerate(variables):
        
        ax = axes[i]
        
        # -------------------------
        # DIURNAL CURVES + CENTROIDS
        # -------------------------
        for j, (label, condition) in enumerate(spei_bins):
            
            subset = df[condition(df)].copy()
            if subset.empty:
                continue
            
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
                'SPEI Bin': label,
                'Centroid Hour': centroid,
                'Centroid Time': centroid_time,
                'Midday Δ': np.nan,
                '% Change': np.nan,
                'CI 2.5%': np.nan,
                'CI 97.5%': np.nan,
                'N drought days': np.nan,
                'N non-drought days': np.nan
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
                fontsize=11
            )
            
            # Save midday row
            results_records.append({
                'Variable': var,
                'SPEI Bin': 'Midday drought - non-drought',
                'Centroid Hour': np.nan,
                'Centroid Time': np.nan,
                'Midday Δ': obs_diff,
                '% Change': obs_pct,
                'CI 2.5%': ci_low,
                'CI 97.5%': ci_high,
                'N drought days': len(daily_d),
                'N non-drought days': len(daily_nd)
            })
        
        ax.text(
            0.02, 0.92,
            panel_labels[i],
            transform=ax.transAxes,
            fontsize=18,
            fontweight='bold',
            va='top',
            ha='left'
        )
        
        ax.set_ylabel(y_labels[i], fontsize=20)
        ax.set_xlim(0, 24)
        ax.grid(True)
        ax.legend(frameon=False, fontsize=18)
    
    axes[-1].set_xlabel('Hour of Day', fontsize=20)
    plt.tight_layout()
    
    out = f'figures/diurnal_cycles_drought/{title}.png'
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

    cbar.set_label('Number of Scenes', fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    ax_map.set_title('Number of Scenes per Location', fontsize=18)

    # ======================================================
    # BAR — VEGETATION (WITH CLIMATE BREADTH)
    # ======================================================
    veg_colors = [veg_color_palette.get(v, 'gray') for v in veg_summary['Veg']]

    ax_veg.barh(
        veg_summary['Veg'],
        veg_summary['Scene_Count'],
        color=veg_colors
    )

    ax_veg.set_title('Scenes per Vegetation', fontsize=14)
    ax_veg.set_xlabel('Scene Count', fontsize=12)

    # for i, (count, kg_n) in enumerate(zip(veg_summary['Scene_Count'], veg_summary['Num_Climate_Classes'])):
    #     ax_veg.text(
    #         count + 1,
    #         i,
    #         f'{kg_n} kgs',
    #         va='center',
    #         fontsize=12
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

    ax_kop.set_title('Scenes per KG Climate Class', fontsize=14 )
    ax_kop.set_xlabel('Scene Count', fontsize=12)

    # for i, (count, veg_n) in enumerate(zip(kop_summary['Scene_Count'], kop_summary['Num_Veg_Classes'])):
    #     ax_kop.text(
    #         count + 1,
    #         i,
    #         f'{veg_n} veg',
    #         va='center',
    #         fontsize=12
    #     )

    # ======================================================
    # BAR — TIME OF DAY
    # ======================================================
    ax_tod.barh(tod_bar.index, tod_bar.values, color='steelblue')
    ax_tod.invert_yaxis()  # Optional: invert y-axis to have morning at top
    ax_tod.set_yticks(tod_bar.index)
    ax_tod.set_yticklabels([f'{hour}:00' for hour in tod_bar.index])
    ax_tod.set_title('Scenes per Time of Day', fontsize=14)
    ax_tod.set_xlabel('Scene Count', fontsize=12)

    # ======================================================
    # BAR — SEASON
    # ======================================================
    ax_season.barh(season_bar.index, season_bar.values, color='darkorange')
    ax_season.invert_yaxis()  # Optional: invert y-axis to have winter at top
    ax_season.set_title('Scenes per Season', fontsize=14)
    ax_season.set_xlabel('Scene Count', fontsize=12)

    # ======================================================
    # PANEL LABELS
    # ======================================================
    label_kwargs = dict(
        x=0.00, y=1.08,
        fontsize=14,
        fontweight='bold',
        va='top',
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
    plt.suptitle('ECOCO3 Version 1 Coverage Summary', fontsize=20)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300)
    else:
        plt.savefig('figures/data_coverage_map.png', dpi=300)
    plt.show()


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

    # Copy + add DOY
    df_ecoco3 = df_ecoco3.copy()
    df_fluxnet = df_fluxnet.copy()
    df_ecoco3['DOY'] = df_ecoco3['TIMESTAMP'].dt.dayofyear
    df_fluxnet['DOY'] = df_fluxnet['TIMESTAMP'].dt.dayofyear

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
        nrows=3, ncols=2, figsize=(16, 12), 
        sharex=True, sharey='row'  # <- share y-axis for columns
    )
    fig.suptitle(
        f"Seasonal cycles by {title_suffix}: FLUXNET vs ECOCO3",
        fontsize=28
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
                fontsize=14,
                fontweight='bold',
                va='top',
                ha='left'
            )
            label_idx += 1

            # Titles (top row only)
            if i == 0:
                ax.set_title(col_title, fontsize=24)

            # Y labels (left column only)
            if j == 0:
                ax.set_ylabel(y_labels[i] if y_labels else variable, fontsize=22)
                ax.tick_params(axis='y', labelsize=20)

            ax.set_xlim(1, 366)
            ax.set_ylim(ylims[i])

            # Legend only once
            if i == 0 and j == 0:
                ax.legend(ncol=2, fontsize=16, frameon=False)

    # X label bottom row
    for ax in axes[-1, :]:
        ax.set_xlabel('Day of Year', fontsize=22)
        ax.tick_params(axis='x', labelsize=20)

    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=300)

    plt.show()

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
        title='FLUXNET by Köppen-Geiger Climate'
    )

    plot_violin(
        df_wue_daily=df_ecoco3[df_ecoco3[second_grouping_category].isin(valid_kg) if valid_kg is not None else df_ecoco3[second_grouping_category]],
        grouping_category=second_grouping_category,
        ax=axes[3],
        title='ECOCO3 by Köppen-Geiger Climate'
    )


    # -----------------------------------------
    # Clean up
    # -----------------------------------------
    axes[0].set_xlabel('')  # remove duplicate x label
    axes[2].set_xlabel('')  # remove duplicate x label
    axes[1].set_xlabel('')  # remove duplicate x label
    axes[3].set_xlabel('')  # remove duplicate x label

    # Panel labels
    panel_labels = ['(a)', '(b)', '(c)', '(d)']
    for i, ax in enumerate(axes):
        ax.text(
            0.02, 0.95, panel_labels[i],
            transform=ax.transAxes,
            fontsize=16,
            fontweight='bold',
            va='top',
            ha='left'
        )

    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=300)

    plt.show()

def plot_diurnal_cycles_spei_comparison(
    df_ecoco3,
    df_fluxnet,
    spei_bins,
    title='Diurnal Cycles by SPEI',
    midday_hours=(10,14),
    n_boot=1000,
    output_path=None
):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd

    datasets = [(df_fluxnet.copy(), 'FLUXNET'),
                (df_ecoco3.copy(), 'ECOCO3')]

    variables = ['GPP', 'ET', 'WUE']
    y_labels = [
        'GPP [µmol CO$_2$ m$^{-2}$ s$^{-1}$]',
        'ET [W m$^{-2}$]',
        'WUE [gC kg$^{-1}$ H$_2$O]'
    ]

    colors = ['#D95F0E', '#2C7FB8']
    panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

    fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True, sharey='row')
    fig.suptitle(title + ' (FLUXNET vs ECOCO3)', fontsize=24)

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
                    'SPEI Bin': label,
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
                    fontsize=16
                )

                results_records.append({
                    'Dataset': dataset_name,
                    'Variable': var,
                    'SPEI Bin': 'Midday drought - non-drought',
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
                fontsize=18,
                fontweight='bold',
                va='top'
            )
            label_idx += 1

            if i == 0:
                ax.set_title(dataset_name, fontsize=18)

            if j == 0:
                ax.set_ylabel(y_labels[i], fontsize=18)

            ax.set_xlim(0, 24)
            ax.grid(True)
            ax.legend(frameon=False, fontsize=18, loc='upper right')
            ax.tick_params(axis='both', labelsize=16)

    for ax in axes[-1, :]:
        ax.set_xlabel('Hour of Day', fontsize=18)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()

    return pd.DataFrame(results_records)

def plot_data_coverage_map_sites(df, valid_veg=None, valid_kg=None):

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
    
    cbar.set_label('Number of Years per Site', fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    ax_map.set_title('Spatial Coverage of ECOCO3 / Flux Sites', fontsize=18)

    # ======================================================
    # VEGETATION BAR
    # ======================================================
    veg_colors = [veg_color_palette.get(v, 'gray') for v in veg_summary['Veg']]
    ax_veg.barh(veg_summary['Veg'], veg_summary['Site Years'], color=veg_colors)
    ax_veg.set_title('Site Years per Vegetation Type', fontsize=14)
    ax_veg.set_xlabel('Site Years', fontsize=12)

    # ======================================================
    # KÖPPEN BAR
    # ======================================================
    kop_colors = [koppen_label_color_palette.get(k, 'gray') for k in kop_summary['kg_label']]

    ax_kop.barh(kop_summary['kg_label'], kop_summary['Site Years'], color=kop_colors)
    ax_kop.set_title('Site Years per Köppen Climate Class', fontsize=14)
    ax_kop.set_xlabel('Site Years', fontsize=12)

    # ======================================================
    # FINALIZE
    # ======================================================
    plt.suptitle('FLUXNET Site Coverage Summary', fontsize=20)
    plt.tight_layout()
    plt.show()