# =========================================================
# Author: Leonardo Gabriel Mourao Thiel
# Project: Master Thesis – System Inertia in the Energy System of the Future: Model-Based Cost Optimization to Secure Inertia Requirements
# Topic: Generation of Plots for Analysis of 2024 System inertia
# Date: 27.04.2026

# =========================================================# =========================================================
# IMPORTS & GLOBAL SETTINGS
# =========================================================

import sys
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Base data path
path = "..//Data"

# ---------------------------------------------------------
# Matplotlib configuration (publication-ready)
# ---------------------------------------------------------
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "legend.fontsize": 16,
    "figure.dpi": 100,   # Display resolution
    "savefig.dpi": 300   # Export resolution (high-quality)
})


# =========================================================
# TECHNOLOGY CLASSIFICATION
# =========================================================

# ---------------------------------------------------------
# Conventional generation technologies
# ---------------------------------------------------------
conventional = [
    "Biomass",
    "Fossil Brown coal/Lignite",
    "Fossil Coal-derived gas",
    "Fossil Gas",
    "Fossil Hard coal",
    "Fossil Oil",
    "Fossil Oil shale",
    "Fossil Peat",
    "Nuclear",
    "Waste",
    "Hydro Run-of-river and poundage",
    "Hydro Run-of-river and pondage",
    "Hydro Pumped Storage",
    "Hydro Water Reservoir"
]

# ---------------------------------------------------------
# Non-conventional (renewable & flexible) technologies
# ---------------------------------------------------------
non_conventional = [
    "Solar",
    "Wind Offshore",
    "Wind Onshore",
    "Marine",
    "Other renewable",
    "Energy storage",
    "Geothermal",
    "Other"
]
def save_figure(fig, name, folder="../Results/2024", subfolder=None, dpi=300):
    """
    Save a matplotlib figure in publication-quality formats (PDF and PNG).

    This function exports a figure both as a vector graphic (PDF) and a
    high-resolution raster image (PNG), ensuring compatibility with
    academic publications and presentations.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure object to be saved.

    name : str
        Base filename (without extension).

    folder : str, optional
        Base output directory. Default is "../Results/2024".

    subfolder : str or None, optional
        Optional subdirectory within the base folder.

    dpi : int, optional
        Resolution for PNG export (dots per inch). Default is 300.

    Returns
    -------
    None
        Files are saved to disk.

    Notes
    -----
    - PDF is saved as a vector graphic (ideal for publications).
    - PNG is saved as a raster image (useful for slides/reports).
    - Bounding box is tightened to remove excess whitespace.
    """

    import os

    # =====================================================
    # BUILD OUTPUT PATH
    # =====================================================
    if subfolder:
        path = os.path.join(folder, subfolder)
    else:
        path = folder

    os.makedirs(path, exist_ok=True)

    # =====================================================
    # FILE PATHS
    # =====================================================
    pdf_path = os.path.join(path, f"{name}.pdf")
    png_path = os.path.join(path, f"{name}.png")

    # =====================================================
    # SAVE FIGURE
    # =====================================================
    fig.savefig(
        pdf_path,
        bbox_inches="tight"
    )

    fig.savefig(
        png_path,
        dpi=dpi,
        bbox_inches="tight"
    )

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")
def median_plot(summary_sorted):
    """
    Plot mean system inertia per country with standard deviation and median.

    This function visualizes the distribution of system inertia (H_sys)
    across countries using:
    - bars for mean values
    - error bars for standard deviation
    - points for median values

    Parameters
    ----------
    summary_sorted : pandas.DataFrame
        DataFrame indexed by country, containing:
        - "mean"   : average H_sys
        - "std"    : standard deviation of H_sys
        - "median" : median H_sys

    Returns
    -------
    None
        The function saves the figure and closes it.

    Notes
    -----
    - Mean ± std highlights variability.
    - Median provides robustness against skewed distributions.
    - Suitable for cross-country comparison of system stability.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines

    # =====================================================
    # X-AXIS POSITIONS
    # =====================================================
    x = np.arange(len(summary_sorted))

    fig, ax = plt.subplots(figsize=(14, 6))

    # =====================================================
    # MEAN + STANDARD DEVIATION (BARS)
    # =====================================================
    ax.bar(
        x,
        summary_sorted["mean"],
        yerr=summary_sorted["std"],
        capsize=5,
        color="#4C72B0"
    )

    # =====================================================
    # MEDIAN (POINTS)
    # =====================================================
    ax.scatter(
        x,
        summary_sorted["median"],
        color="black",
        zorder=3
    )

    # =====================================================
    # AXES (GERMAN)
    # =====================================================
    ax.set_xticks(x)
    ax.set_xticklabels(
        summary_sorted.index,
        rotation=45,
        ha="right"
    )

    ax.set_ylabel("Systemträgheit $H_{sys}$")
    ax.set_xlabel("Land")

    ax.set_title("Systemträgheit nach Ländern")

    # =====================================================
    # LEGEND (GERMAN)
    # =====================================================
    bar_handle = mpatches.Patch(
        color="#4C72B0",
        label="Mittelwert ± Standardabweichung"
    )

    median_handle = mlines.Line2D(
        [],
        [],
        color="black",
        marker="o",
        linestyle="None",
        label="Median"
    )

    ax.legend(
        handles=[bar_handle, median_handle],
        frameon=False
    )

    plt.tight_layout()

    # =====================================================
    # SAVE
    # =====================================================
    save_figure(fig, "mean_median_errorbars_per_country")

    plt.close(fig)
def share_below_2(summary_sorted):
    """
    Plot the share of critical hours (H_sys < 2) per country.

    Countries are sorted by descending share of critical hours.
    High-risk countries are visually highlighted.

    Parameters
    ----------
    summary_sorted : pandas.DataFrame
        DataFrame indexed by country with column:
        - "share_<2_%" : percentage of hours with H_sys < 2

    Returns
    -------
    None
        The function saves the plot.

    Notes
    -----
    - Threshold of 10% is used to highlight critical countries.
    - Bar labels display exact percentage values.
    """

    import matplotlib.pyplot as plt
    import numpy as np

    # =====================================================
    # SORT COUNTRIES BY RISK
    # =====================================================
    summary_sorted = summary_sorted.sort_values(
        "share_<2_%",
        ascending=False
    )

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(summary_sorted))
    values = summary_sorted["share_<2_%"].values

    # =====================================================
    # COLOR CODING (RISK LEVEL)
    # =====================================================
    colors = [
        "#d62728" if v > 10 else "#1f77b4"
        for v in values
    ]

    bars = ax.bar(x, values, color=colors)

    # =====================================================
    # AXES (GERMAN)
    # =====================================================
    ax.set_xticks(x)
    ax.set_xticklabels(summary_sorted.index, rotation=45)

    ax.set_ylabel("Anteil kritischer Stunden (%)")
    ax.set_xlabel("Land")

    ax.set_title("Anteil von Stunden mit niedriger Systemträgheit ($H_{sys} < 2$)")

  

    # =====================================================
    # GRID
    # =====================================================
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    # =====================================================
    # SAVE
    # =====================================================
    save_figure(fig, "share_below_2_per_country")

    plt.close(fig)
def daytime_share_below_2(summary_sorted, daytime_share):
    """
    Plot share of critical hours (H_sys < 2) by daytime and country.

    Parameters
    ----------
    summary_sorted : pandas.DataFrame
        DataFrame indexed by country (defines plot order).

    daytime_share : pandas.DataFrame
        Index: daytime categories (e.g. Nacht, Morgen, Nachmittag, Abend)
        Columns: countries
        Values: share of critical hours (%)

    Returns
    -------
    None
        The function saves the plot.

    Notes
    -----
    - Maintains consistent country order across plots.
    - Enables comparison of temporal patterns of system weakness.
    """

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))

    # =====================================================
    # ENSURE CONSISTENT COUNTRY ORDER
    # =====================================================
    country_order = summary_sorted.index

    # Reindex safely (avoids KeyError if missing countries)
    daytime_sorted = daytime_share.reindex(columns=country_order, fill_value=0)

    # =====================================================
    # PLOT
    # =====================================================
    daytime_sorted.T.plot(
        kind="bar",
        ax=ax
    )

    # =====================================================
    # AXES (GERMAN)
    # =====================================================
    ax.set_ylabel("Anteil kritischer Stunden (%)")
    ax.set_xlabel("Land")

    ax.set_title(
        "Anteil kritischer Stunden nach Tageszeit ($H_{sys} < 2$)"
    )

    ax.tick_params(axis="x", rotation=45)

    # Optional: improve readability
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()

    # =====================================================
    # SAVE
    # =====================================================
    save_figure(fig, "daytime_share_below_2")

    plt.close(fig)
def hourly_mean(hourly_mean):
    """
    Plot average system inertia over the day for each country.

    Parameters
    ----------
    hourly_mean : pandas.DataFrame
        Index: hour of day
        Columns: countries
        Values: average H_sys

    Returns
    -------
    None
        The function saves the plot.

    Notes
    -----
    - Shows temporal patterns of inertia availability.
    - Useful for identifying recurring weak periods (e.g. night vs peak).
    """

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))

    # =====================================================
    # PLOT LINES PER COUNTRY
    # =====================================================
    for country in hourly_mean.columns:
        ax.plot(
            hourly_mean.index,
            hourly_mean[country],
            label=country,
            linewidth=1.5
        )

    # =====================================================
    # AXES (GERMAN)
    # =====================================================
    ax.set_xlabel("Stunde")
    ax.set_ylabel("Durchschnittliche Systemträgheit ($H_{sys}$)")

    ax.set_title("Durchschnittliches Tagesprofil der Systemträgheit")

    # Improve readability
    ax.grid(True, linestyle="--", alpha=0.3)

    # Better legend layout
    ax.legend(
        ncol=3,
        fontsize=8,
        frameon=False
    )

    # Optional: clean x-axis (important for thesis)
    ax.set_xlim(hourly_mean.index.min(), hourly_mean.index.max())

    plt.tight_layout()

    # =====================================================
    # SAVE
    # =====================================================
    save_figure(fig, "hourly_mean_by_country")

    plt.close(fig)   
def boxplot_per_country(df_filtered, country_columns):
    """
    Plot distribution of system inertia (H_sys) per country using boxplots.

    Parameters
    ----------
    df_filtered : pandas.DataFrame
        DataFrame containing H_sys values for multiple countries.

    country_columns : list of str
        List of column names corresponding to countries.

    Returns
    -------
    None
        The function saves the plot.

    Notes
    -----
    - Boxplots show median, quartiles, and outliers.
    - Useful for identifying variability and extreme values.
    """

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))

    # =====================================================
    # BOXPLOT
    # =====================================================
    df_filtered[country_columns].plot(
        kind="box",
        ax=ax
    )

    # =====================================================
    # AXES (GERMAN)
    # =====================================================
    ax.set_ylabel("Systemträgheit ($H_{sys}$)")
    ax.set_xlabel("Land")

    ax.set_title("Verteilung der Systemträgheit nach Ländern")

    ax.tick_params(axis="x", rotation=45)

    # Improve readability
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()

    # =====================================================
    # SAVE
    # =====================================================
    save_figure(fig, "boxplot_per_country")

    plt.close(fig)
def hourly_profiles_lines(hourly_mean):
    """
    Plot hourly system inertia profiles across countries.

    Each line represents one hour-of-day, showing how inertia
    is distributed across countries.

    Parameters
    ----------
    hourly_mean : pandas.DataFrame
        Index: hour of day
        Columns: countries
        Values: mean H_sys

    Returns
    -------
    None
        The function saves the plot.

    Notes
    -----
    - Highlights spatial differences per hour.
    - Can become cluttered if many hours are plotted.
    """

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 6))

    # =====================================================
    # PLOT LINES (ONE PER HOUR)
    # =====================================================
    for hour in hourly_mean.index:
        ax.plot(
            hourly_mean.columns,
            hourly_mean.loc[hour],
            alpha=0.4,
            linewidth=1
        )

    # =====================================================
    # AXES (GERMAN)
    # =====================================================
    ax.set_xticks(range(len(hourly_mean.columns)))
    ax.set_xticklabels(hourly_mean.columns, rotation=90)

    ax.set_ylabel("Systemträgheit ($H_{sys}$)")
    ax.set_xlabel("Land")

    ax.set_title("Stündliche Profile der Systemträgheit über Länder")

    # Improve readability
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()

    # =====================================================
    # SAVE
    # =====================================================
    save_figure(fig, "hourly_profiles_lines")

    plt.close(fig)
def get_genearation_data(valid_countries, start_date, end_date):
    """
    Load and filter hourly generation data per country.

    Parameters
    ----------
    valid_countries : list of str
        List of sheet names (countries) in the Excel file.

    start_date : str or datetime
        Start of the filtering period.

    end_date : str or datetime
        End of the filtering period.

    Returns
    -------
    dict
        Dictionary mapping country → filtered DataFrame.

    Notes
    -----
    - Assumes Excel file with one sheet per country.
    - Datetime column must be named "Datetime".
    - Timezone information is removed if present.
    """

    import pandas as pd

    generation_data = pd.read_excel(
        path + "//2024//country_hourly_all.xlsx",
        sheet_name=valid_countries
    )

    for country, df in generation_data.items():

        # Ensure datetime format
        df["Datetime"] = pd.to_datetime(
            df["Datetime"],
            errors="coerce"
        )

        # Remove timezone if present
        if df["Datetime"].dt.tz is not None:
            df["Datetime"] = df["Datetime"].dt.tz_localize(None)

        # Apply date filter
        df_filtered = df[
            (df["Datetime"] >= start_date) &
            (df["Datetime"] <= end_date)
        ].copy()

        generation_data[country] = df_filtered


    return generation_data    
def generation_share_konventionell_vs_nicht_konventionell(
    valid_countries, start_date, end_date
):
    """
    Compute and plot the share of conventional vs. non-conventional
    electricity generation per country.

    Parameters
    ----------
    valid_countries : list of str
        List of country names (must match Excel sheet names).

    start_date : str or datetime
        Start of analysis period.

    end_date : str or datetime
        End of analysis period.

    Returns
    -------
    None
        The function saves the resulting plot.

    Notes
    -----
    - Uses predefined technology groupings:
        * conventional
        * non_conventional
    - Aggregates generation over the full time horizon.
    - Shares are expressed as percentage of total generation.
    """

    import pandas as pd
    import matplotlib.pyplot as plt

    generation_share = []

    # =====================================================
    # LOAD DATA
    # =====================================================
    generation_data = get_genearation_data(
        valid_countries,
        start_date,
        end_date
    )

    # =====================================================
    # LOOP COUNTRIES
    # =====================================================
    for country, df in generation_data.items():

        # Keep numeric columns only
        df_num = df.drop(columns=["Datetime"]).apply(
            pd.to_numeric,
            errors="coerce"
        )

        # -------------------------------------------------
        # TOTAL GENERATION
        # -------------------------------------------------
        total_generation = df_num["Total_Generation"].sum()

        if total_generation == 0:
            continue  # avoid division by zero

        # -------------------------------------------------
        # TECHNOLOGY GROUPS
        # -------------------------------------------------
        conv_cols = [
            c for c in conventional
            if c in df_num.columns
        ]

        nonconv_cols = [
            c for c in non_conventional
            if c in df_num.columns
        ]

        # -------------------------------------------------
        # AGGREGATION
        # -------------------------------------------------
        conventional_gen = df_num[conv_cols].sum().sum()
        non_conventional_gen = df_num[nonconv_cols].sum().sum()

        # -------------------------------------------------
        # SHARES
        # -------------------------------------------------
        conventional_pct = conventional_gen / total_generation * 100
        non_conventional_pct = non_conventional_gen / total_generation * 100

        generation_share.append({
            "country": country,
            "conventional_%": conventional_pct,
            "non_conventional_%": non_conventional_pct
        })

    # =====================================================
    # DATAFRAME
    # =====================================================
    generation_share = (
        pd.DataFrame(generation_share)
        .set_index("country")
    )

    # Sort for better comparison
    generation_share_sorted = generation_share.sort_values(
        "conventional_%"
    )

    # =====================================================
    # PLOT
    # =====================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    generation_share_sorted.plot(
        kind="bar",
        stacked=True,
        ax=ax
    )

    # =====================================================
    # AXES (GERMAN)
    # =====================================================
    ax.set_ylabel("Erzeugungsanteil (%)")
    ax.set_xlabel("Land")

    ax.set_title(
        "Anteil konventioneller vs. nicht-konventioneller Stromerzeugung"
    )

    # Legend (German)
    ax.legend(
        ["Konventionell", "Nicht-konventionell"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=2,
        frameon=False
    )

    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    # =====================================================
    # SAVE
    # =====================================================
    save_figure(
        fig,
        "generation_share_konventionell_vs_nicht_konventionell"
    )

    plt.close(fig)