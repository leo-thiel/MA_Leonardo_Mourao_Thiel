# =========================================================
# Author: Leonardo Gabriel Mourao Thiel
# Project: Master Thesis – System Inertia in the Energy System of the Future: Model-Based Cost Optimization to Secure Inertia Requirements
# Topic: Generation of Plots for Analysis of 2040 System inertia
# Date: 27.04.2026
# =========================================================


# Plotting and data handling libraries
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

# Additional plotting utilities for custom legends
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

# ---------------------------------------------------------
# Global plotting style configuration
# ---------------------------------------------------------
# Ensures consistent appearance across all figures
# (important for thesis-quality plots)

plt.rcParams.update({
    "font.size": 22,        # default text size
    "axes.titlesize": 26,   # plot titles
    "axes.labelsize": 22,   # axis labels
    "legend.fontsize": 22,  # legend text
    "figure.dpi": 100,      # resolution for display
    "savefig.dpi": 600      # high resolution for export (publication quality)
})

# ---------------------------------------------------------
# Scenario labeling (for plots and legends)
# ---------------------------------------------------------
# Maps internal scenario names to human-readable labels

scenario_labels = {
    "no_inertia": "Ohne Trägheitrestriktion",
    "thermal_only": "Nur konventionelle Trägheit",
    "thermal_plus_virtual": "konventionelle + virtuelle Trägheit"
}

# ---------------------------------------------------------
# Fuel type mapping (standardization of labels)
# ---------------------------------------------------------
# Used to aggregate technologies into consistent categories

FUEL_MAP = {
    "Gas": "Gas",
    "Hard coal": "Steinkohle",
    "Lignite": "Braunkohle",
    "Heavy oil": "Schweröl",
    "Light oil": "Leichtöl",
    "Nuclear": "Kernenergie",
    "Hydrogen": "Wasserstoff",
    "Oil shale": "Ölschiefer",

    "Biomass": "Biomasse",
    "Waste": "Müllverbrennung",

    "Other": "sonstige RES",   # fallback category for renewables
}

# =========================================================
# COLOR MAP (CONSISTENT ACROSS ALL PLOTS)
# =========================================================

# Defines a fixed color for each technology
# → ensures visual consistency across all figures
# → critical for comparability between scenarios

COLOR_MAP = {
    "Gas": "#1f77b4",
    "Kohle": "#4d4d4d",
    "Braunkohle": "#8c564b",
    "Schweröl": "#9467bd",
    "Leichtöl": "#c5b0d5",
    "Ölschiefer": "#7f7f7f",

    "Kernenergie": "#636efa",
    "Wasserstoff": "#00cc96",

    "Biomasse": "#2ca02c",
    "Müllverbrennung": "#bcbd22",

    "Hydro": "#1f9ed6",
    "Battery": "#ff7f0e",

    "Sonstige RES": "#98df8a",
    "Sonstige nicht-RES": "#d62728",

    # Highlight category (e.g. for aggregated non-conventional sources)
    "Nicht-konventionell": "#ff4d4d"
}

# ---------------------------------------------------------
# Output directory for results
# ---------------------------------------------------------
# Ensures that all generated plots and processed files
# are stored in a structured location

output = "../Results/2040"

# Create directory if it does not exist
os.makedirs(output, exist_ok=True)
# =========================================================
# SOLUTION FILE PARSING AND PREPROCESSING
# =========================================================

def read_sol_file(path: str):
    """
    Reads a Gurobi .sol file and converts it into a structured DataFrame.

    The .sol format contains lines of the form:
        variable_name   value

    Parameters
    ----------
    path : str
        Path to the .sol file.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - var   : variable name (string)
        - value : variable value (float)
    """

    # Container for parsed variable-value pairs
    data = []

    # Open solution file in read mode
    with open(path, "r") as f:

        # Iterate over each line in the file
        for line in f:

            # Skip comment lines (starting with "#")
            # and empty lines
            if line.startswith("#") or line.strip() == "":
                continue

            # Split line into tokens (variable name + value)
            parts = line.split()

            # Only process valid lines with exactly 2 elements
            if len(parts) == 2:
                var, val = parts

                # Convert value to float and store tuple
                data.append((var, float(val)))

    # Convert list of tuples into a pandas DataFrame
    return pd.DataFrame(data, columns=["var", "value"])


# =========================================================
# VARIABLE NAME PARSING
# =========================================================

# The following sections prepare parsing of Gurobi variable names.
# Since variable names encode multiple dimensions (e.g. unit, hour),
# we need structured extraction logic.

# Example variable names:
#   p_unit1_5
#   flow_DE_FR_12
#   battery_soc_BAT1_24
#
# → typically contain:
#   - variable type
#   - unit or country identifier (UID)
#   - time index (hour)

# =========================================================
# ROBUST VARIABLE PARSING WITH UID EXTRACTION
# =========================================================

import re

# Regular expressions (regex) are used to:
# - extract structured information from variable names
# - make parsing robust against different naming conventions
#
# Example:
#   pattern = r"p_(.+)_(\d+)"
#   → group(1) = unit ID
#   → group(2) = hour

# =========================================================
# PARSE VARIABLES + ADD COUNTRY
# =========================================================

# Goal:
# Extend parsed DataFrame by extracting:
#   - variable type (e.g. generation, flow, SOC)
#   - unit ID (UID)
#   - time index (hour)
#   - country (derived from UID)

# This enables:
#   - aggregation per country
#   - time series reconstruction
#   - plotting and analysis

# =========================================================
# PARSE VARIABLES + COUNTRY VIA STRING MATCHING
# =========================================================

# In many cases, country information is embedded in the UID:
# Example:
#   "Gas_DE_2040_15" → country = "DE"
#
# Strategy:
#   - split UID string by "_"
#   - identify country position
#   - or match against known country codes
#
# Alternative (more robust):
#   - use predefined mapping of units → countries
#   - avoids reliance on naming conventions

def parse_variables(df, countries):
    """
    Parses Gurobi variable names and assigns country via string matching.

    Method
    ------
    - Extract hour (last numeric token)
    - Extract variable type (first token)
    - Extract unit identifier (middle part)
    - Detect country by matching tokens against a predefined country list

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing columns:
        - var   : variable name (string)
        - value : variable value (float)

    countries : list
        List of country codes used in the model (e.g. ["DE", "FR", "IT"]).

    Returns
    -------
    pd.DataFrame
        Extended DataFrame with additional columns:
        - type    : variable type (e.g. p, flow, soc)
        - unit    : unit or identifier string
        - hour    : time index (int)
        - country : detected country code
    """

    # Containers for parsed components
    types = []
    units = []
    hours = []
    country_out = []

    # Convert country list to set for faster lookup (O(1))
    country_set = set(countries)

    # Iterate over all variable names
    for var in df["var"]:

        # -------------------------------------------------
        # 1) Extract hour (last numeric token)
        # -------------------------------------------------
        # Pattern: underscore followed by digits at end of string
        # Example: "p_DE_12" → hour = 12
        match_hour = re.search(r"_(\d+)$", var)

        if match_hour:
            hour = int(match_hour.group(1))

            # Remove hour part from string for further parsing
            left = var[:match_hour.start()]
        else:
            # No hour found → static variable (e.g. percentages)
            hour = None
            left = var

        # -------------------------------------------------
        # 2) Split variable name into tokens
        # -------------------------------------------------
        tokens = left.split("_")

        # Variable type = first token (e.g. p, flow, soc)
        var_type = tokens[0]

        # Unit identifier = everything except the first token
        # Example:
        #   "p_Gas_DE_1" → unit = "Gas_DE"
        unit = "_".join(tokens[1:]) if len(tokens) > 1 else None

        # -------------------------------------------------
        # 3) Detect country via exact token match
        # -------------------------------------------------
        # Assumption:
        # Country codes appear as separate tokens in variable name

        country = None

        for t in tokens:
            if t in country_set:
                country = t
                break  # stop at first match

        # -------------------------------------------------
        # Store parsed results
        # -------------------------------------------------
        types.append(var_type)
        units.append(unit)
        hours.append(hour)
        country_out.append(country)

    # ---------------------------------------------------------
    # Add parsed columns to DataFrame
    # ---------------------------------------------------------
    df["type"] = types
    df["unit"] = units
    df["hour"] = hours
    df["country"] = country_out

    return df

# =========================================================
# LOAD ALL SCENARIOS
# =========================================================

def load_all_solutions(sol_paths: dict, inputs) -> dict:
    """
    Loads and parses all scenario solution files.

    Workflow
    --------
    1. Read raw Gurobi .sol file
    2. Parse variable names into structured components
    3. Store results per scenario

    Parameters
    ----------
    sol_paths : dict
        Mapping {scenario_name: path_to_sol_file}

    inputs : object
        Model input container (must include country list, e.g. inputs.countryList)

    Returns
    -------
    dict
        Mapping {scenario_name: parsed DataFrame}
    """

    # Container for all scenario results
    solutions = {}

    # Iterate over all scenarios
    for name, path in sol_paths.items():

        # -------------------------------------------------
        # 1) Read raw solution file
        # -------------------------------------------------
        df = read_sol_file(path)

        # -------------------------------------------------
        # 2) Parse variable names into structured format
        # -------------------------------------------------
        # IMPORTANT: parse_variables expects a country list
        df = parse_variables(df, inputs.countryList)

        # -------------------------------------------------
        # 3) Store result under scenario name
        # -------------------------------------------------
        solutions[name] = df

    return solutions
# =========================================================
# BUILD UNIT → COUNTRY MAPPING
# =========================================================

def build_unit_country_map(inputs):
    """
    Creates a mapping from unit ID to country.

    Purpose
    -------
    Provides a robust alternative to string-based country detection.
    Instead of extracting country from variable names,
    this mapping directly links each unit to its country.

    Parameters
    ----------
    inputs : ModelInputs
        Input container with unit objects (thermal, RES, battery, etc.)

    Returns
    -------
    dict
        Mapping {unit_id: country}
    """

    # Dictionary: UID → country
    mapping = {}

    # -------------------------------------------------
    # Thermal units
    # -------------------------------------------------
    for u in inputs.thermal_units:
        mapping[str(u.uid)] = u.country

    # -------------------------------------------------
    # Other RES units (biomass, waste, etc.)
    # -------------------------------------------------
    for u in inputs.other_res_units:
        mapping[str(u.uid)] = u.country

    # -------------------------------------------------
    # Non-RES units (time-series based)
    # -------------------------------------------------
    for u in inputs.non_res_units:
        mapping[str(u.uid)] = u.country

    # -------------------------------------------------
    # Battery units
    # -------------------------------------------------
    for b in inputs.battery_units:
        mapping[str(b.uid)] = b.country

    return mapping
# =========================================================
# SAVE FIGURE (WITH SUBFOLDER SUPPORT)
# =========================================================

def save_figure(fig, name, folder="../Results/2040", subfolder=None, dpi=600):
    """
    Save a matplotlib figure in high quality (PDF + PNG).

    Purpose
    -------
    Ensures consistent and publication-ready export of figures.
    Saves both:
    - PDF (vector format → ideal for papers/thesis)
    - PNG (raster format → quick preview / presentations)

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure object to be saved.

    name : str
        Base filename (without extension).

    folder : str
        Root output directory.

    subfolder : str or None
        Optional subfolder for structured result storage
        (e.g. "generation", "inertia", "flows").

    dpi : int
        Resolution for PNG export.
    """

    import os

    # ---------------------------------------------------------
    # 1) Build output path
    # ---------------------------------------------------------
    # Allows hierarchical folder structure:
    # e.g. ../Results/2040/generation/

    if subfolder:
        path = os.path.join(folder, subfolder)
    else:
        path = folder

    # Create directory if it does not exist
    os.makedirs(path, exist_ok=True)

    # ---------------------------------------------------------
    # 2) Define file paths
    # ---------------------------------------------------------
    pdf_path = os.path.join(path, f"{name}.pdf")
    png_path = os.path.join(path, f"{name}.png")

    # ---------------------------------------------------------
    # 3) Save figure
    # ---------------------------------------------------------

    # PDF: vector format → scalable, no quality loss (best for thesis)
    fig.savefig(
        pdf_path,
        bbox_inches="tight"
    )

    # PNG: raster format → fixed resolution (useful for quick viewing)
    fig.savefig(
        png_path,
        dpi=dpi,
        bbox_inches="tight"
    )

    # ---------------------------------------------------------
    # 4) Logging
    # ---------------------------------------------------------
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")
# =========================================================
# H_sys CALCULATION PER COUNTRY (FINAL FILTERED VERSION)
# =========================================================

def extract_H_sys_per_country(df: pd.DataFrame,
                             inputs,
                             countries: list) -> pd.DataFrame:
    """
    Computes system inertia ratio H_sys per country and hour.

    Definition
    ----------
    H_sys = inertia_total / demand

    Notes
    -----
    - Uses ONLY total system inertia (inertia_total)
    - Filters out invalid entries (missing values, zero demand)
    - Returns a tidy DataFrame for plotting or analysis

    Parameters
    ----------
    df : pd.DataFrame
        Parsed solution DataFrame (from .sol file)

    inputs : ModelInputs
        Contains demand time series (df_load)

    countries : list
        List of model countries

    Returns
    -------
    pd.DataFrame
        Columns:
        - country
        - hour
        - H_sys (dimensionless inertia ratio)
    """

    # =====================================================
    # 1) Extract ONLY total inertia
    # =====================================================
    # Filter for variables of type "inertia" AND unit "total_*"
    # → ensures we only use aggregated system inertia

    inertia = df[
        (df["type"] == "inertia") &
        (df["unit"].str.startswith("total_")) &  # critical filter
        (df["hour"].notna()) &
        (df["country"].notna())
    ].copy()

    # Ensure hour is integer (important for merging)
    inertia["hour"] = inertia["hour"].astype(int)

    # Rename for clarity
    inertia = inertia.rename(columns={"value": "inertia"})

    # =====================================================
    # 2) Prepare demand data
    # =====================================================
    # Convert wide format (country columns) → long format

    df_load = inputs.df_load.copy()

    # Create consistent hour index (must match optimization model)
    df_load["hour"] = range(1, len(df_load) + 1)

    demand = df_load.melt(
        id_vars="hour",
        var_name="country",
        value_name="demand"
    )

    # Optional: restrict to modeled countries
    demand = demand[demand["country"].isin(countries)]

    # =====================================================
    # 3) Merge inertia and demand
    # =====================================================
    merged = pd.merge(
        inertia,
        demand,
        on=["country", "hour"],
        how="inner"
    )

    # Remove invalid or zero demand values
    # → avoids division errors and meaningless ratios
    merged = merged[merged["demand"] > 0]

    # =====================================================
    # 4) Compute system inertia ratio
    # =====================================================
    merged["H_sys"] = merged["inertia"] / merged["demand"]

    # =====================================================
    # 5) Return clean result
    # =====================================================
    return merged[["country", "hour", "H_sys"]]
# =========================================================
# STATISTICAL ANALYSIS OF H_sys
# =========================================================

def compute_stats(H_df: pd.DataFrame,
                  critical_threshold: float = 2.0) -> pd.DataFrame:
    """
    Computes descriptive statistics of system inertia (H_sys)
    per country.

    In addition to standard statistics, the function computes:
        - share of critical hours (H_sys < threshold)

    Parameters
    ----------
    H_df : pd.DataFrame
        Long-format DataFrame with columns:
        - country
        - hour
        - H_sys

    critical_threshold : float, optional
        Threshold below which inertia is considered critical.
        Default = 2.0

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per country and columns:
        - country
        - mean
        - median
        - std
        - min
        - max
        - count
        - share_critical (%)
    """

    # =====================================================
    # 1) Input validation
    # =====================================================
    # Ensure required columns are present
    required_cols = {"country", "H_sys"}
    missing = required_cols - set(H_df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Remove invalid or missing inertia values
    # → ensures robust statistics
    H_df = H_df.dropna(subset=["H_sys"])

    # =====================================================
    # 2) Compute descriptive statistics per country
    # =====================================================
    stats = (
        H_df
        .groupby("country")["H_sys"]
        .agg(
            # Central tendency
            mean="mean",
            median="median",

            # Dispersion
            std="std",

            # Extremes
            min="min",
            max="max",

            # Sample size
            count="count",

            # Share of critical hours (%)
            # → fraction of hours where inertia is below threshold
            share_critical=lambda x: (x < critical_threshold).mean() * 100
        )
        .reset_index()
    )

    # =====================================================
    # 3) Sorting (useful for plotting / ranking)
    # =====================================================
    # Countries with lowest inertia first
    stats = stats.sort_values("mean")

    return stats
def plot_mean_median_std(stats: pd.DataFrame, scenario: str):
    """
    Plot mean, median, and standard deviation of H_sys per country.

    Visualization design:
    - Bars represent mean H_sys
    - Error bars represent ± standard deviation
    - Black dots indicate median values

    Parameters
    ----------
    stats : pd.DataFrame
        Output from compute_stats(), containing:
        country, mean, median, std, ...

    scenario : str
        Scenario key used for labeling and file naming
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines

    # =====================================================
    # SCENARIO LABEL (HUMAN-READABLE)
    # =====================================================
    label = scenario_labels.get(scenario, scenario)

    # X positions for countries
    x = np.arange(len(stats))

    # =====================================================
    # CREATE FIGURE
    # =====================================================
    fig, ax = plt.subplots(figsize=(14, 6))

    # =====================================================
    # MEAN + STANDARD DEVIATION (BARS)
    # =====================================================
    ax.bar(
        x,
        stats["mean"],
        yerr=stats["std"],
        capsize=5,
        color="#4C72B0"
    )

    # =====================================================
    # MEDIAN (POINTS)
    # =====================================================
    ax.scatter(
        x,
        stats["median"],
        color="black",
        zorder=3
    )

    # =====================================================
    # AXES (GERMAN)
    # =====================================================
    ax.set_xticks(x)
    ax.set_xticklabels(stats["country"], rotation=45, ha="right")

    ax.set_ylabel("Systemträgheit ($H_{sys}$)")
    ax.set_xlabel("Land")

    ax.set_title(f"Systemträgheit nach Ländern ({label})")

    # Improve readability
    ax.grid(axis="y", linestyle="--", alpha=0.4)

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

    # =====================================================
    # LAYOUT + SAVE
    # =====================================================
    plt.tight_layout()

    save_figure(
        fig,
        scenario + "_mean_median_std",
        subfolder="inertia"
    )

    plt.close(fig)
def plot_critical_hours(stats: pd.DataFrame, scenario: str):
    """
    Plot share of critical inertia hours (H_sys < threshold) per country.

    Visualization design:
    - Bar chart sorted by critical share (descending)
    - Red bars highlight high-risk countries
    - Values annotated above bars for readability

    Parameters
    ----------
    stats : pd.DataFrame
        Output from compute_stats(), must include:
        - country
        - share_critical

    scenario : str
        Scenario key used for labeling and file naming
    """

    import matplotlib.pyplot as plt
    import numpy as np

    # =====================================================
    # SCENARIO LABEL (HUMAN-READABLE)
    # =====================================================
    label = scenario_labels.get(scenario, scenario)

    # =====================================================
    # SORT COUNTRIES BY RISK
    # =====================================================
    stats_sorted = stats.sort_values("share_critical", ascending=False)


    x = np.arange(len(stats_sorted))
    values = stats_sorted["share_critical"].values

    # =====================================================
    # COLOR CODING (RISK HIGHLIGHTING)
    # =====================================================
    colors = ["#d62728" if v > 10 else "#1f77b4" for v in values]

    # =====================================================
    # CREATE PLOT
    # =====================================================
    fig, ax = plt.subplots(figsize=(18, 6))

    bars = ax.bar(x, values, color=colors)

    # =====================================================
    # AXES (GERMAN)
    # =====================================================
    ax.set_xticks(x)
    ax.set_xticklabels(stats_sorted["country"], rotation=45, ha="right")

    ax.set_ylabel("Anteil Stunden mit $H_{sys} < 2$ (%)")
    ax.set_xlabel("Land")

    ax.set_title(f"Kritische Stunden der Systemträgheit ({label})")

    # Threshold reference line (10%)
    ax.axhline(10, linestyle="--", color="red", alpha=0.6)

    # =====================================================
    # VALUE LABELS (ROBUST POSITIONING)
    # =====================================================
    ymax = max(values) if len(values) > 0 else 1

   

    # =====================================================
    # READABILITY
    # =====================================================
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    # =====================================================
    # SAVE
    # =====================================================
    save_figure(
        fig,
        scenario + "_critical_hours_pct",
        subfolder="inertia"
    )

    plt.close(fig)
# =========================================================
# PREPARE UNIT ID SETS
# =========================================================

def get_unit_sets(inputs):
    """
    Extract sets of unit identifiers for different generation categories.

    This function processes the provided input data structure and groups
    unit IDs into predefined categories (e.g., thermal and non-reservoir units).
    The IDs are converted to strings to ensure consistent downstream usage
    (e.g., for dictionary keys, comparisons, or serialization).

    Parameters
    ----------
    inputs : object
        Container object holding unit collections. Expected to provide:
        - inputs.thermal_units : iterable of unit objects
        - inputs.non_res_units : iterable of unit objects
        Each unit object must have a unique identifier accessible via `uid`.

    Returns
    -------
    dict
        Dictionary containing sets of unit IDs (as strings) for each category:
        - "thermal" : set of str
            IDs of all thermal generation units.
        - "non_res" : set of str
            IDs of all non-reservoir generation units.

    Notes
    -----
    - Converting IDs to strings avoids type inconsistencies if `uid`
      is not uniformly typed (e.g., int vs. str).
    - The use of sets ensures uniqueness and enables efficient membership checks (O(1)).

    Raises
    ------
    AttributeError
        If `inputs` does not contain the expected attributes.
    """

    # Create a set of thermal unit IDs (converted to string for consistency)
    thermal_ids = set(str(u.uid) for u in inputs.thermal_units)

    # Create a set of non-reservoir unit IDs (also normalized to string)
    non_res_ids = set(str(u.uid) for u in inputs.non_res_units)

    # Return categorized unit ID sets in a dictionary structure
    return {
        "thermal": thermal_ids,
        "non_res": non_res_ids
    }  
def compute_generation_split_balance(df, inputs, countries):
    """
    Compute the generation split between conventional and non-conventional sources.

    This function aggregates generation and demand data to derive a balance between
    different generation categories. It classifies generation into thermal, hydro,
    other renewable sources (RES), and non-RES, and computes the residual demand
    covered by non-conventional sources.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing model results or dispatch variables with at least:
        - "type" : str
            Variable type (e.g., "p", "p_non_res", "charge")
        - "var" : str
            Variable name containing unit identifiers
        - "value" : float
            Numerical value (e.g., generation or charging)
        - "hour" : optional
            Timestamp or indicator for time-dependent variables

    inputs : object
        Container object holding system input data. Expected attributes:
        - inputs.df_load : pandas.DataFrame
            Load time series per country
        - inputs.thermal_units : iterable
        - inputs.other_res_units : iterable
        - inputs.df_hydro : pandas.DataFrame or dict of DataFrames
            Hydro generation time series

    countries : list of str
        List of country codes to include in the aggregation.

    Returns
    -------
    dict
        Dictionary with aggregated generation values:
        - "conventional" : float
            Total conventional generation (thermal + hydro + other RES + non-RES)
        - "non_conventional" : float
            Residual demand not covered by explicitly modeled conventional sources
        - "thermal" : float
            Total thermal generation
        - "other_res" : float
            Generation from other renewable sources
        - "non_res" : float
            Non-renewable generation from aggregated variable

    Notes
    -----
    - Unit identification is performed via substring matching in `df["var"]`,
      which assumes that unit IDs are embedded in variable names.
    - Residual ("non_conventional") is computed as a balancing term and may
      include imports, storage discharge, or unmodeled generation sources.
    - Battery charging is computed but not included in the final balance;
      this may be intentional or require further integration depending on model design.
    """

    # =========================================================
    # DEMAND AGGREGATION
    # =========================================================
    # Total electricity demand across selected countries and all time steps
    demand = inputs.df_load[countries].sum().sum()

    # =========================================================
    # BATTERY CHARGING (not used in final balance)
    # =========================================================
    # Sum of all charging actions (e.g., storage consumption)
    battery_charge = df[
        (df["type"] == "charge") &
        (df["hour"].notna())  # ensure time-dependent entries only
    ]["value"].sum()

    # =========================================================
    # THERMAL GENERATION
    # =========================================================
    # Extract thermal unit IDs and match them against variable names
    thermal_ids = set(str(u.uid) for u in inputs.thermal_units)

    thermal_mask = (
        (df["type"] == "p") &  # production variables
        (df["var"].apply(lambda x: any(uid in x for uid in thermal_ids)))
    )

    thermal = df[thermal_mask]["value"].sum()

    # =========================================================
    # OTHER RENEWABLE GENERATION (RES)
    # =========================================================
    # Identify generation from other renewable units (excluding hydro if separate)
    other_res_ids = set(str(u.uid) for u in inputs.other_res_units)

    other_mask = (
        (df["type"] == "p") &
        (df["var"].apply(lambda x: any(uid in x for uid in other_res_ids)))
    )

    other_res = df[other_mask]["value"].sum()

    # =========================================================
    # NON-RENEWABLE (AGGREGATED VARIABLE)
    # =========================================================
    # Direct aggregation of non-renewable generation variable
    non_res = df[df["type"] == "p_non_res"]["value"].sum()

    # =========================================================
    # HYDRO GENERATION
    # =========================================================
    # Handle both single DataFrame and dictionary of DataFrames (e.g., scenarios)
    df_hydro = inputs.df_hydro

    if isinstance(df_hydro, dict):
        # Sum across all hydro datasets (e.g., multiple reservoirs or scenarios)
        hydro = sum(df_h[countries].sum().sum() for df_h in df_hydro.values())
    else:
        hydro = df_hydro[countries].sum().sum()

    # =========================================================
    # BALANCE CALCULATION
    # =========================================================
    # Residual demand not covered by explicitly modeled sources
    non_conventional = (
        demand
        - thermal
        - hydro
        - other_res
        - non_res
    )

    # Total conventional generation
    conventional = thermal + hydro + other_res + non_res

    # =========================================================
    # OUTPUT
    # =========================================================
    return {
        "conventional": conventional,
        "non_conventional": non_conventional,
        "thermal": thermal,
        "other_res": other_res,
        "non_res": non_res
    }

# =========================================================
# PLOT: GENERATION SPLIT (BALANCE-BASED)
# =========================================================
def plot_generation_balance(gen_split, scenario_name="scenario"):
    """
    Plot the generation split between conventional and non-conventional sources.

    This function creates a bar chart comparing aggregated conventional
    and non-conventional generation values and saves the figure to disk.

    Parameters
    ----------
    gen_split : dict
        Dictionary containing generation values with keys:
        - "conventional"
        - "non_conventional"

    scenario_name : str, optional
        Identifier used for file naming and labeling.

    Returns
    -------
    None
    """

    import matplotlib.pyplot as plt

    # =====================================================
    # SCENARIO LABEL
    # =====================================================
    label_sc = scenario_labels.get(scenario_name, scenario_name)

    # =====================================================
    # DATA
    # =====================================================
    labels = ["Konventionell", "Nicht-konventionell"]
    values = [
        gen_split.get("conventional", 0),
        gen_split.get("non_conventional", 0)
    ]

    # =====================================================
    # PLOT
    # =====================================================
    fig, ax = plt.subplots(figsize=(6, 6))

    colors = ["#4C72B0", "#55A868"]

    bars = ax.bar(labels, values, color=colors)

    # =====================================================
    # VALUE LABELS
    # =====================================================
    ymax = max(values) if len(values) > 0 else 1

    for bar in bars:
        height = bar.get_height()

        if height > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + ymax * 0.02,
                f"{height:.0f}",
                ha="center",
                va="bottom",
                fontsize=11
            )

    # =====================================================
    # AXES (GERMAN)
    # =====================================================
    ax.set_ylabel("Energie (MWh)")
    ax.set_title(f"Erzeugungsstruktur ({label_sc})")

    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()

    # =====================================================
    # SAVE
    # =====================================================
    save_figure(
        fig,
        f"generation_balance_{scenario_name}",
        subfolder="generation"
    )

    plt.close(fig)

# =========================================================
# NET IMPORTS PER COUNTRY
# =========================================================

def compute_net_imports(df):
    """
    Compute net electricity imports per country.

    Net imports are defined as total imports minus total exports for each country,
    based on directional flow variables.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing flow variables with at least the following columns:
        - "type" : str
            Variable type (must include "flow" entries)
        - "var" : str
            Encoded flow variable name (e.g., "flow_DE_FR_1")
        - "value" : float
            Flow magnitude (assumed ≥ 0 for active flows)

    Returns
    -------
    dict
        Dictionary mapping country codes to net imports (float):
        - positive value → net importer
        - negative value → net exporter

    Notes
    -----
    - Variable names are expected to follow the pattern:
      "flow_<source>_<destination>_<index>"
    - Only positive flow values are considered (i.e., `max(value, 0)`),
      assuming that directionality is encoded in the variable name rather
      than the sign of `value`.
    - Countries appearing only in imports or exports are still included.
    """

    # Filter for flow variables only
    flows = df[df["type"] == "flow"].copy()

    # Dictionaries to accumulate imports and exports per country
    imports = {}
    exports = {}

    # =========================================================
    # PARSE FLOWS AND AGGREGATE
    # =========================================================
    for _, row in flows.iterrows():

        var = row["var"]      # e.g., "flow_DE_FR_1"
        value = row["value"]  # flow magnitude

        # Split variable name into components
        parts = var.split("_")

        # Ensure expected structure: flow_<src>_<dest>_<...>
        if len(parts) < 4:
            continue  # skip malformed entries

        _, src, dest, _ = parts

        # -----------------------------------------------------
        # EXPORTS: from source country to destination
        # -----------------------------------------------------
        exports[src] = exports.get(src, 0) + max(value, 0)

        # -----------------------------------------------------
        # IMPORTS: into destination country
        # -----------------------------------------------------
        imports[dest] = imports.get(dest, 0) + max(value, 0)

    # =========================================================
    # NET IMPORT CALCULATION
    # =========================================================
    # Combine all countries appearing in either imports or exports
    net = {}
    all_countries = set(imports.keys()) | set(exports.keys())

    for c in all_countries:
        net[c] = imports.get(c, 0) - exports.get(c, 0)

    return net 
# =========================================================
# GENERATION SPLIT (PHYSICALLY CONSISTENT)
# =========================================================

def compute_generation_shares(df, inputs, countries):
    """
    Compute generation shares based on physically utilized energy per country.

    This function derives the share of conventional and non-conventional
    generation using a physically consistent accounting framework. It ensures
    that only effectively used renewable energy is counted (i.e., after
    curtailment and deloading) and that no negative contributions occur.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing model results with at least:
        - "type" : str
            Variable type (e.g., "p", "p_non_res", "curtailement", "discharge")
        - "value" : float
            Numerical value of the variable
        - "country" : str
            Country identifier
        - "unit" : str, optional
            Unit identifier (required for thermal/RES classification)

    inputs : object
        Container with system input data. Expected attributes:
        - inputs.df_total_renewable : pandas.DataFrame
            Renewable generation potential per country
        - inputs.df_hydro : pandas.DataFrame or dict of DataFrames
            Hydro generation time series
        - inputs.thermal_units : iterable
        - inputs.other_res_units : iterable

    countries : list of str
        List of country codes to evaluate.

    Returns
    -------
    pandas.DataFrame
        DataFrame with generation shares per country:
        - "country" : str
        - "conventional_%" : float
        - "non_conventional_%" : float

    Guarantees
    ----------
    - No negative renewable contributions (clipped at zero)
    - Shares sum to 100% (if total generation > 0)
    - Consistency with model structure (curtailment and deloading accounted for)

    Notes
    -----
    - Non-conventional generation includes:
        * renewable feed-in (after curtailment and deloading)
        * battery discharge
    - Conventional generation includes:
        * thermal
        * hydro
        * other RES (if classified as dispatchable)
        * non-RES aggregate
    - The function assumes that all required country columns exist in the input data.
    """

    results = []

    # =====================================================
    # 1) RENEWABLE INPUT (POTENTIAL GENERATION)
    # =====================================================
    # Total available renewable energy per country (before curtailment)
    renewable = inputs.df_total_renewable[countries].sum()

    # Curtailment: unused renewable energy due to system constraints
    curtail = df[df["type"] == "curtailement"] \
        .groupby("country")["value"].sum()

    # Deloading: partial reduction of renewable output (if modeled)
    deload = df[df["type"].str.contains("deload", na=False)] \
        .groupby("country")["value"].sum()

    # =====================================================
    # 2) ACTUAL RENEWABLE FEED-IN
    # =====================================================
    renewable_used = {}

    for c in countries:
        # Net renewable generation after losses
        val = (
            renewable.get(c, 0)
            - curtail.get(c, 0)
            - deload.get(c, 0)
        )

        # Enforce physical constraint: no negative generation
        renewable_used[c] = max(val, 0)

    # =====================================================
    # 3) BATTERY DISCHARGE
    # =====================================================
    # Storage output contributing to supply
    discharge = df[df["type"] == "discharge"] \
        .groupby("country")["value"].sum()

    # =====================================================
    # 4) CONVENTIONAL GENERATION COMPONENTS
    # =====================================================
    # Prepare unit ID sets for classification
    thermal_ids = set(str(u.uid) for u in inputs.thermal_units)
    other_ids = set(str(u.uid) for u in inputs.other_res_units)

    # Thermal generation
    thermal = df[
        (df["type"] == "p") &
        (df["unit"].isin(thermal_ids))
    ].groupby("country")["value"].sum()

    # Other renewable (dispatchable or separately classified)
    other_res = df[
        (df["type"] == "p") &
        (df["unit"].isin(other_ids))
    ].groupby("country")["value"].sum()

    # Aggregated non-renewable generation
    non_res = df[df["type"] == "p_non_res"] \
        .groupby("country")["value"].sum()

    # Hydro generation (supports multiple data structures)
    df_hydro = inputs.df_hydro

    if isinstance(df_hydro, dict):
        hydro = {
            c: sum(
                df_h[c].sum()
                for df_h in df_hydro.values()
                if c in df_h.columns
            )
            for c in countries
        }
    else:
        hydro = df_hydro[countries].sum().to_dict()

    # =====================================================
    # 5) SHARE CALCULATION
    # =====================================================
    for c in countries:

        # Non-conventional supply (renewables + storage discharge)
        non_conv = renewable_used.get(c, 0) + discharge.get(c, 0)

        # Conventional supply (dispatchable / controllable sources)
        conv = (
            thermal.get(c, 0)
            + other_res.get(c, 0)
            + non_res.get(c, 0)
            + hydro.get(c, 0)
        )

        total = conv + non_conv

        # Compute percentage shares (avoid division by zero)
        if total > 0:
            share_conv = conv / total * 100
            share_non = non_conv / total * 100
        else:
            share_conv = 0
            share_non = 0

        results.append({
            "country": c,
            "conventional_%": share_conv,
            "non_conventional_%": share_non
        })

    return pd.DataFrame(results)
def compute_generation_shares_detailed(df, inputs, countries):
    """
    Compute detailed generation shares per technology (in %).

    This function disaggregates generation by fuel/technology and computes
    percentage shares per country. It is based on physically used renewable
    energy (after curtailment and deloading) and explicitly excludes battery
    discharge from the accounting.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing dispatch results with at least:
        - "type" : str
            Variable type (e.g., "p", "curtailement", "deload")
        - "value" : float
            Generation value
        - "country" : str
            Country identifier
        - "unit" : str
            Unit identifier

    inputs : object
        Container with system input data. Expected attributes:
        - inputs.df_total_renewable : pandas.DataFrame
            Renewable generation potential per country
        - inputs.df_hydro : pandas.DataFrame or dict of DataFrames
            Hydro generation
        - inputs.thermal_units : iterable
        - inputs.other_res_units : iterable
        - inputs.non_res_units : iterable

    countries : list of str
        List of country codes.

    Returns
    -------
    pandas.DataFrame
        DataFrame with percentage shares per technology and country.
        Columns correspond to technologies (fuel types) plus:
        - "Hydro"
        - "Nicht-konventionell"
        - "country"

    Notes
    -----
    - Renewable generation is included as "Nicht-konventionell" after
      subtracting curtailment and deloading.
    - Battery discharge is intentionally excluded from this breakdown.
    - Fuel mapping is based on `FUEL_MAP`; unmapped types default to "Sonstige".
    - Shares are only computed for positive contributions.
    """

    import pandas as pd

    results = []

    # =====================================================
    # 1) RENEWABLE INPUT (POTENTIAL → USED)
    # =====================================================
    # Total renewable potential per country
    renewable = inputs.df_total_renewable[countries].sum()

    # Curtailment (unused renewable energy)
    curtail = df[df["type"] == "curtailement"] \
        .groupby("country")["value"].sum()

    # Deloading (partial reduction of output)
    deload = df[df["type"].str.contains("deload", na=False)] \
        .groupby("country")["value"].sum()

    # Compute physically used renewable energy (non-negative constraint)
    renewable_used = {}
    for c in countries:
        val = (
            renewable.get(c, 0)
            - curtail.get(c, 0)
            - deload.get(c, 0)
        )
        renewable_used[c] = max(val, 0)

    # =====================================================
    # 2) UNIT → FUEL MAPPING
    # =====================================================
    # Map each unit ID to a fuel/technology category
    unit_fuel = {}

    # Thermal units (typically fossil or dispatchable)
    for u in inputs.thermal_units:
        unit_fuel[str(u.uid)] = FUEL_MAP.get(u.fuel_type, "Sonstige")

    # Other RES units (may use different attribute naming)
    for u in inputs.other_res_units:
        unit_fuel[str(u.uid)] = FUEL_MAP.get(u.fueltype, "Sonstige")

    # Non-renewable units grouped into a single category
    for u in inputs.non_res_units:
        unit_fuel[str(u.uid)] = "Sonstige nicht-RES"

    # =====================================================
    # 3) GENERATION BY TECHNOLOGY
    # =====================================================
    # Filter dispatch variables and assign fuel categories
    df_p = df[df["type"] == "p"].copy()
    df_p["fuel"] = df_p["unit"].map(unit_fuel)

    # Aggregate generation per country and fuel type
    gen_by_fuel = (
        df_p
        .groupby(["country", "fuel"])["value"]
        .sum()
        .unstack(fill_value=0)
    )

    # -----------------------------------------------------
    # HYDRO GENERATION (handled separately)
    # -----------------------------------------------------
    df_hydro = inputs.df_hydro

    if isinstance(df_hydro, dict):
        hydro = {
            c: sum(
                df_h[c].sum()
                for df_h in df_hydro.values()
                if c in df_h.columns
            )
            for c in countries
        }
    else:
        hydro = df_hydro[countries].sum().to_dict()

    # =====================================================
    # 4) SHARE CALCULATION
    # =====================================================
    for c in countries:

        tech_values = {}

        # Add dispatch-based technologies
        if c in gen_by_fuel.index:
            tech_values.update(gen_by_fuel.loc[c].to_dict())

        # Add hydro as separate technology
        tech_values["Hydro"] = hydro.get(c, 0)

        # Add non-conventional renewable generation (excl. storage)
        tech_values["Nicht-konventionell"] = renewable_used.get(c, 0)

        # NOTE: Battery discharge intentionally excluded

        total = sum(tech_values.values())

        # Skip countries with no generation
        if total == 0:
            continue

        # Compute percentage shares (only for positive contributions)
        shares = {
            tech: val / total * 100
            for tech, val in tech_values.items()
            if val > 0
        }

        # Add country identifier
        shares["country"] = c

        results.append(shares)

    # Combine results into DataFrame (fill missing technologies with 0)
    df_result = pd.DataFrame(results).fillna(0)

    return df_result
def export_generation_shares_all_scenarios(solutions, inputs, countries):
    """
    Export detailed generation shares for multiple scenarios to an Excel file.

    This function iterates over a predefined set of scenarios, computes detailed
    generation shares per country and technology, and writes the results into
    separate sheets of a single Excel workbook.

    Parameters
    ----------
    solutions : dict
        Dictionary mapping scenario names (str) to DataFrames containing
        model results (dispatch variables). Each entry is passed to
        `compute_generation_shares_detailed`.

    inputs : object
        Container with system input data required for share computation.

    countries : list of str
        List of country codes to include in the analysis.

    Returns
    -------
    None
        The function writes results to an Excel file and prints progress messages.

    Notes
    -----
    - Output file is written to "../Results/2040/generation_shares_detailed.xlsx".
    - Each scenario is stored in a separate Excel sheet.
    - Sheet names are derived from `scenario_labels` (if available) and truncated
      to 31 characters (Excel limitation).
    - Missing scenarios in `solutions` are skipped with a warning.
    - Values are rounded to two decimal places for readability.
    """

    import pandas as pd
    import os

    # =====================================================
    # SCENARIO ORDER DEFINITION
    # =====================================================
    # Ensures consistent ordering of sheets in the Excel output
    scenario_order = ["no_inertia", "thermal_only", "thermal_plus_virtual"]

    # =====================================================
    # OUTPUT PATH SETUP
    # =====================================================
    # Create output directory if it does not exist
    folder = os.path.join("output", "generation")
    os.makedirs(folder, exist_ok=True)

    # Define Excel file path (note: independent of 'folder' above)
    file_path = os.path.join("../Results/2040", "generation_shares_detailed.xlsx")

    # =====================================================
    # EXCEL EXPORT
    # =====================================================
    with pd.ExcelWriter(file_path, engine="openpyxl") as writer:

        for scen in scenario_order:

            # Skip scenarios not present in the solutions dictionary
            if scen not in solutions:
                print(f"⚠️ Scenario missing: {scen}")
                continue

            print(f"Processing: {scen}")

            # -------------------------------------------------
            # COMPUTE GENERATION SHARES
            # -------------------------------------------------
            df_shares = compute_generation_shares_detailed(
                solutions[scen],
                inputs,
                countries
            )

            # Skip empty results (e.g., no generation data)
            if df_shares.empty:
                continue

            # -------------------------------------------------
            # FORMATTING
            # -------------------------------------------------
            # Round values for cleaner presentation in Excel
            df_shares = df_shares.round(2)

            # Sort columns alphabetically (keeping "country" first)
            cols = ["country"] + sorted(
                [c for c in df_shares.columns if c != "country"]
            )
            df_shares = df_shares[cols]

            # -------------------------------------------------
            # SHEET NAME HANDLING
            # -------------------------------------------------
            # Map scenario to readable label (fallback: raw name)
            sheet_name = scenario_labels.get(scen, scen)

            # Excel constraint: maximum 31 characters per sheet name
            sheet_name = sheet_name[:31]

            # -------------------------------------------------
            # WRITE TO EXCEL
            # -------------------------------------------------
            df_shares.to_excel(
                writer,
                sheet_name=sheet_name,
                index=False
            )

    # =====================================================
    # FINAL LOG MESSAGE
    # =====================================================
    print(f"Saved Excel: {file_path}")
# =========================================================
# PLOT: GENERATION SHARE PER COUNTRY
# =========================================================

def plot_generation_shares(shares, scenario):
    """
    Plot generation shares (conventional vs. non-conventional) per country.

    This function creates a stacked bar chart showing the percentage split
    between conventional and non-conventional generation for each country
    within a given scenario.

    Parameters
    ----------
    shares : pandas.DataFrame
        DataFrame containing generation shares per country, as produced by
        `compute_generation_shares`. Expected columns:
        - "country" : str
        - "conventional_%" : float
        - "non_conventional_%" : float

    scenario : str
        Scenario identifier used for labeling and file naming.

    Returns
    -------
    None
        The function saves the plot to disk and closes the figure.

    Notes
    -----
    - The plot is sorted by increasing non-conventional share to improve readability.
    - The function assumes that `scenario_labels` and `save_figure` are defined globally.
    - Shares are expected to sum to approximately 100% per country.
    """

    import matplotlib.pyplot as plt
    import numpy as np

    # Map scenario identifier to human-readable label (fallback: raw name)
    label = scenario_labels.get(scenario, scenario)

    # =====================================================
    # DATA PREPARATION
    # =====================================================
    # Sort countries by non-conventional share (ascending)
    shares = shares.sort_values("non_conventional_%")

    # Create x-axis positions
    x = np.arange(len(shares))

    # =====================================================
    # PLOTTING
    # =====================================================
    fig, ax = plt.subplots(figsize=(14, 6))

    # Base layer: conventional generation share
    ax.bar(x, shares["conventional_%"])

    # Stacked layer: non-conventional share on top
    ax.bar(
        x,
        shares["non_conventional_%"],
        bottom=shares["conventional_%"]
    )

    # =====================================================
    # AXIS FORMATTING
    # =====================================================
    ax.set_xticks(x)
    ax.set_xticklabels(shares["country"], rotation=45)

    ax.set_ylabel("Anteil (%)")
    ax.set_title(f"Erzeugungsmix pro Land ({label})")

    # Legend order matches stacking order
    ax.legend(["Konventionell", "Nicht-Konventionell"])

    # =====================================================
    # SAVE OUTPUT
    # =====================================================
    save_figure(
        fig,
        scenario + "_generation_share",
        subfolder="generation"
    )

    # Adjust layout to prevent label overlap
    plt.tight_layout()

    # Close figure to free memory
    plt.close()
# =========================================================
# SCENARIO COMPARISON (RELATIVE CHANGE VS BASE)
# =========================================================

def compare_generation_shares(solutions, inputs, countries, base_scenario="no_inertia"):
    """
    Compute changes in generation shares relative to a base scenario.

    This function compares the percentage shares of conventional and
    non-conventional generation across multiple scenarios against a
    specified base scenario. The result represents absolute differences
    in percentage points (not relative percent change).

    Parameters
    ----------
    solutions : dict
        Dictionary mapping scenario names (str) to DataFrames containing
        model results.

    inputs : object
        Container with system input data required for share computation.

    countries : list of str
        List of country codes included in the analysis.

    base_scenario : str, optional
        Scenario used as reference for comparison. Default is "no_inertia".

    Returns
    -------
    pandas.DataFrame
        DataFrame containing differences in generation shares per country
        and scenario. Columns include:
        - "country" : str
        - "conventional_%" : float (difference in percentage points)
        - "non_conventional_%" : float (difference in percentage points)
        - "scenario" : str

    Notes
    -----
    - Differences are computed as:
        Δ = scenario_share − base_share
      and therefore represent percentage point deviations.
    - Positive values indicate an increase relative to the base scenario.
    - The function assumes consistent country coverage across scenarios.
    """

    import pandas as pd

    # =====================================================
    # BASE SCENARIO COMPUTATION
    # =====================================================
    # Extract and compute shares for the reference scenario
    base_df = solutions[base_scenario]
    base_shares = compute_generation_shares(base_df, inputs, countries)

    # Set country as index for alignment in subtraction
    base_shares = base_shares.set_index("country")

    results = []

    # =====================================================
    # SCENARIO COMPARISON LOOP
    # =====================================================
    for scen, df in solutions.items():

        # Skip base scenario (no comparison needed)
        if scen == base_scenario:
            continue

        # Compute shares for current scenario
        scen_shares = compute_generation_shares(df, inputs, countries)
        scen_shares = scen_shares.set_index("country")

        # -------------------------------------------------
        # DELTA COMPUTATION
        # -------------------------------------------------
        # Align on country index and compute differences
        delta = scen_shares - base_shares

        # Convert back to flat structure and add scenario label
        temp = delta.reset_index()
        temp["scenario"] = scen

        results.append(temp)

    # =====================================================
    # OUTPUT
    # =====================================================
    return pd.concat(results, ignore_index=True)
# =========================================================
# PLOT: CHANGE VS BASE SCENARIO
# =========================================================

def plot_generation_change(delta_df):
    """
    Plot changes in non-conventional generation shares relative to a base scenario.

    This function visualizes the difference in non-conventional generation shares
    (in percentage points) across multiple scenarios and countries using a grouped
    bar chart.

    Parameters
    ----------
    delta_df : pandas.DataFrame
        DataFrame as returned by `compare_generation_shares`, containing:
        - "country" : str
        - "scenario" : str
        - "non_conventional_%" : float
            Difference in percentage points relative to the base scenario

    Returns
    -------
    None
        The function saves the plot to disk and closes the figure.

    Notes
    -----
    - Values represent absolute differences (percentage points), not relative percent change.
    - A horizontal reference line at zero highlights increases vs. decreases.
    - The function assumes that `scenario_labels` and `save_figure` are defined globally.
    """

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # =====================================================
    # DATA PREPARATION
    # =====================================================
    # Reshape data into pivot table:
    # rows = countries, columns = scenarios, values = delta shares
    pivot = delta_df.pivot(
        index="country",
        columns="scenario",
        values="non_conventional_%"
    ).fillna(0)

    countries = pivot.index
    scenarios = pivot.columns

    # X positions for grouped bars
    x = np.arange(len(countries))

    # Dynamic bar width depending on number of scenarios
    width = 0.8 / len(scenarios)

    # =====================================================
    # PLOTTING
    # =====================================================
    fig, ax = plt.subplots(figsize=(14, 6))

    for i, scen in enumerate(scenarios):

        # Map scenario to readable label
        label = scenario_labels.get(scen, scen)

        # Plot bars with horizontal offset per scenario
        ax.bar(
            x + i * width,
            pivot[scen],
            width,
            label=label
        )

    # =====================================================
    # AXIS FORMATTING
    # =====================================================
    # Center x-ticks under grouped bars
    ax.set_xticks(x + width * (len(scenarios) - 1) / 2)
    ax.set_xticklabels(countries, rotation=45)

    # Reference line to distinguish positive/negative changes
    ax.axhline(0, color="black", linewidth=1)

    ax.set_ylabel("Änderung des Anteils nicht-konventioneller Erzeugung (%)")
    ax.set_title("Änderung gegenüber Szenario ohne Trägheit")

    # Display legend for scenarios
    ax.legend()

    # Improve layout (avoid label overlap)
    plt.tight_layout()

    # =====================================================
    # SAVE OUTPUT
    # =====================================================
    save_figure(
        fig,
        "generation_change_vs_no_inertia",
        subfolder="generation"
    )

    # Close figure to free memory
    plt.close()

# =========================================================
# CURTAILMENT SHARE OF TOTAL GENERATION
# =========================================================

def compute_curtailment(df, inputs, countries):
    """
    Compute curtailment as a share of total electricity generation per country.

    This function quantifies the amount of curtailed renewable energy and relates
    it to total electricity generation (including conventional generation and
    storage discharge). The result is expressed in percentage terms.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing model results with at least:
        - "type" : str
            Variable type (e.g., "curtailement", "p", "p_non_res", "discharge")
        - "value" : float
            Energy value (MWh)
        - "country" : str
            Country identifier
        - "unit" : str, optional
            Unit identifier for classification

    inputs : object
        Container with system input data. Expected attributes:
        - inputs.df_total_renewable : pandas.DataFrame
            Renewable generation potential per country
        - inputs.df_hydro : pandas.DataFrame or dict of DataFrames
            Hydro generation
        - inputs.thermal_units : iterable
        - inputs.other_res_units : iterable

    countries : list of str
        List of country codes.

    Returns
    -------
    pandas.DataFrame
        DataFrame with curtailment metrics per country:
        - "country" : str
        - "curtailment_MWh" : float
        - "total_generation_MWh" : float
        - "curtailment_%" : float

    Notes
    -----
    - Curtailment is defined as unused renewable energy ("curtailement").
    - Total generation includes:
        * renewable generation (after curtailment and deloading)
        * battery discharge
        * thermal generation
        * other RES (if dispatch-based)
        * non-RES aggregate
        * hydro
    - The resulting percentage represents the share of curtailed energy
      relative to total supplied energy (not relative to renewable potential).
    """

    import pandas as pd

    results = []

    # =====================================================
    # 1) CURTAILMENT
    # =====================================================
    # Total curtailed renewable energy per country
    curtail = df[df["type"] == "curtailement"] \
        .groupby("country")["value"].sum()

    # =====================================================
    # 2) RENEWABLE GENERATION (AVAILABLE → USED)
    # =====================================================
    # Total renewable potential
    renewable = inputs.df_total_renewable[countries].sum()

    # Additional reduction via deloading
    deload = df[df["type"].str.contains("deload", na=False)] \
        .groupby("country")["value"].sum()

    # Actual renewable generation after curtailment and deloading
    renewable_used = {
        c: max(
            renewable.get(c, 0)
            - curtail.get(c, 0)
            - deload.get(c, 0),
            0
        )
        for c in countries
    }

    # =====================================================
    # 3) BATTERY DISCHARGE
    # =====================================================
    # Storage output contributing to total supply
    discharge = df[df["type"] == "discharge"] \
        .groupby("country")["value"].sum()

    # =====================================================
    # 4) CONVENTIONAL GENERATION
    # =====================================================
    # Prepare unit ID sets
    thermal_ids = set(str(u.uid) for u in inputs.thermal_units)
    other_ids = set(str(u.uid) for u in inputs.other_res_units)

    # Thermal generation
    thermal = df[
        (df["type"] == "p") &
        (df["unit"].isin(thermal_ids))
    ].groupby("country")["value"].sum()

    # Other RES (dispatch-based classification)
    other_res = df[
        (df["type"] == "p") &
        (df["unit"].isin(other_ids))
    ].groupby("country")["value"].sum()

    # Aggregated non-renewable generation
    non_res = df[df["type"] == "p_non_res"] \
        .groupby("country")["value"].sum()

    # =====================================================
    # 5) HYDRO GENERATION
    # =====================================================
    df_hydro = inputs.df_hydro

    if isinstance(df_hydro, dict):
        hydro = {
            c: sum(
                df_h[c].sum()
                for df_h in df_hydro.values()
                if c in df_h.columns
            )
            for c in countries
        }
    else:
        hydro = df_hydro[countries].sum().to_dict()

    # =====================================================
    # 6) FINAL CALCULATION PER COUNTRY
    # =====================================================
    for c in countries:

        # Curtailed energy
        cur = curtail.get(c, 0)

        # Total generation (physically supplied energy)
        total_gen = (
            renewable_used.get(c, 0)
            + discharge.get(c, 0)
            + thermal.get(c, 0)
            + other_res.get(c, 0)
            + non_res.get(c, 0)
            + hydro.get(c, 0)
        )

        # Curtailment share relative to total generation
        if total_gen > 0:
            share = cur / total_gen * 100
        else:
            share = 0

        results.append({
            "country": c,
            "curtailment_MWh": cur,
            "total_generation_MWh": total_gen,
            "curtailment_%": share
        })

    return pd.DataFrame(results) 
def plot_curtailment(curt_df, scenario):
    """
    Plot curtailment share of total generation per country.

    Visualizes the percentage of curtailed renewable energy
    relative to total electricity generation.

    Parameters
    ----------
    curt_df : pandas.DataFrame
        Must contain:
        - "country"
        - "curtailment_%"

    scenario : str
        Scenario identifier.

    Returns
    -------
    None
    """

    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    # =====================================================
    # SCENARIO LABEL
    # =====================================================
    label = scenario_labels.get(scenario, scenario)

    # =====================================================
    # SAFETY CHECKS
    # =====================================================
    if curt_df is None or curt_df.empty:
        print(f"[INFO] No curtailment data for {scenario}")
        return

    required_cols = ["country", "curtailment_%"]
    for col in required_cols:
        if col not in curt_df.columns:
            raise ValueError(f"Missing required column: {col}")

    # =====================================================
    # DATA CLEANING
    # =====================================================
    df = curt_df.copy()

    df["curtailment_%"] = pd.to_numeric(
        df["curtailment_%"],
        errors="coerce"
    ).fillna(0)

    df["curtailment_%"] = df["curtailment_%"].clip(lower=0)

    # =====================================================
    # SORTING
    # =====================================================
    df = df.sort_values(
        "curtailment_%",
        ascending=False
    ).reset_index(drop=True)

    x = np.arange(len(df))
    countries = df["country"].astype(str).values
    values = df["curtailment_%"].values

    # =====================================================
    # COLOR CODING (IMPORTANT)
    # =====================================================
    # Highlight high curtailment (>5%) in red
    colors = [
        "#d62728" if v > 5 else "#4C72B0"
        for v in values
    ]

    # =====================================================
    # PLOT
    # =====================================================
    fig, ax = plt.subplots(figsize=(14, 6))

    bars = ax.bar(x, values, color=colors)

    # -----------------------------------------------------
    # VALUE LABELS
    # -----------------------------------------------------
    ymax = max(values) if len(values) > 0 else 1
    offset = ymax * 0.02 if ymax > 0 else 0.5

    for i, v in enumerate(values):
        if v > 0:
            ax.text(
                i,
                v + offset,
                f"{v:.2f}%",
                ha="center",
                va="bottom",
                fontsize=10
            )

    # =====================================================
    # AXES
    # =====================================================
    ax.set_xticks(x)
    ax.set_xticklabels(countries, rotation=45, ha="right")

    ax.set_ylabel("Abregelung (% der Gesamterzeugung)")
    ax.set_title(f"Abregelungsanteil nach Ländern ({label})")

    # Reference line (policy-relevant threshold)
    ax.axhline(5, linestyle="--", color="red", alpha=0.6)

    # Dynamic y-limit
    ax.set_ylim(0, ymax * 1.2 if ymax > 0 else 1)

    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()

    # =====================================================
    # SAVE
    # =====================================================
    save_figure(
        fig,
        f"curtailment_{scenario}",
        subfolder="generation"
    )

    plt.close(fig)
# =========================================================
# CURTAILMENT CHANGE VS BASE SCENARIO
# =========================================================

def compare_curtailment_vs_base(solutions, inputs, countries, base="no_inertia"):
    """
    Compute changes in curtailment share relative to a base scenario.

    This function evaluates how the share of curtailed energy (relative to total
    generation) changes across scenarios compared to a reference scenario.

    Parameters
    ----------
    solutions : dict
        Dictionary mapping scenario names (str) to DataFrames containing
        model results.

    inputs : object
        Container with system input data required for curtailment computation.

    countries : list of str
        List of country codes included in the analysis.

    base : str, optional
        Base scenario used for comparison. Default is "no_inertia".

    Returns
    -------
    pandas.DataFrame
        DataFrame containing curtailment differences per country and scenario:
        - "country" : str
        - "delta_curtailment_%" : float (difference in percentage points)
        - "scenario" : str

    Notes
    -----
    - Differences are computed as:
        Δ = scenario − base
      and represent **percentage point changes**, not relative percentages.
    - Positive values indicate increased curtailment relative to the base scenario.
    """

    import pandas as pd

    # =====================================================
    # BASE SCENARIO
    # =====================================================
    base_df = solutions[base]
    base_curt = compute_curtailment(base_df, inputs, countries)
    base_curt = base_curt.set_index("country")

    results = []

    # =====================================================
    # SCENARIO LOOP
    # =====================================================
    for scen, df in solutions.items():

        if scen == base:
            continue

        # Compute curtailment for current scenario
        scen_curt = compute_curtailment(df, inputs, countries)
        scen_curt = scen_curt.set_index("country")

        # -------------------------------------------------
        # DELTA (percentage point difference)
        # -------------------------------------------------
        delta = scen_curt["curtailment_%"] - base_curt["curtailment_%"]

        temp = delta.reset_index()
        temp.columns = ["country", "delta_curtailment_%"]
        temp["scenario"] = scen

        results.append(temp)

    return pd.concat(results, ignore_index=True)
# =========================================================
# PLOT: CURTAILMENT CHANGE VS BASE SCENARIO
# =========================================================



    # =====================================================
    # SAVE OUTPUT
    # =====================================================
    save_figure(
        fig,
        "curtailment_change_vs_no_inertia",
        subfolder="generation"
    )

    plt.close()
def plot_curtailment_change(delta_df):
    """
    Plot changes in curtailment share relative to a base scenario.

    Values are expressed in percentage points (Δ%).
    """

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    scenario_colors = {
        "thermal_only": "#1f77b4",           # blau
        "thermal_plus_virtual": "#ff7f0e"    # orange
    }
    # =====================================================
    # PIVOT (CRITICAL FIX → ensures correct alignment)
    # =====================================================
    pivot = delta_df.pivot(
        index="country",
        columns="scenario",
        values="delta_curtailment_%"
    ).fillna(0)

    countries = pivot.index
    scenarios = pivot.columns

    x = np.arange(len(countries))
    width = 0.8 / len(scenarios)

    # =====================================================
    # PLOT
    # =====================================================
    fig, ax = plt.subplots(figsize=(20, 8))

    for i, scen in enumerate(scenarios):

        values = pivot[scen].values
        label = scenario_labels.get(scen, scen)

        # Color: increase = red, decrease = green
        base_color = scenario_colors.get(scen, "#333333")
        ax.bar(
            x + i * width,
            values,
            width,
            label=label,
            color=base_color
        )

    # =====================================================
    # AXES
    # =====================================================
    ax.axhline(0, linewidth=1, color="black")

    ax.set_xticks(x + width * (len(scenarios) - 1) / 2)
    ax.set_xticklabels(countries, rotation=45, ha="right")

    ax.set_ylabel("Änderung der Abregelung (%)")
    ax.set_title("Änderung der Abregelung gegenüber Referenzszenario")

    ax.grid(axis="y", linestyle="--", alpha=0.3)

    # =====================================================
    # LEGEND
    # =====================================================
    ax.legend(
        title="Szenarien",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        frameon=False
    )

    plt.tight_layout()

    # =====================================================
    # SAVE
    # =====================================================
    save_figure(
        fig,
        "curtailment_change_vs_base",
        subfolder="generation"
    )

    plt.close(fig)
def compare_generation_split(solutions, inputs):
    """
    Compare detailed generation split across multiple scenarios.

    This function computes the total generation per fuel/technology for each
    scenario and aggregates the results into a single DataFrame.

    Parameters
    ----------
    solutions : dict
        Dictionary mapping scenario names (str) to DataFrames containing
        model results.

    inputs : object
        Container with system input data required for generation classification.

    Returns
    -------
    pandas.DataFrame
        DataFrame with scenarios as index and generation values per
        fuel/technology as columns.

    Notes
    -----
    - Missing values (technologies not present in a scenario) are filled with zero.
    - Results represent absolute generation values (e.g., MWh), not shares.
    """

    import pandas as pd

    rows = []

    # =====================================================
    # LOOP OVER SCENARIOS
    # =====================================================
    for scen, df in solutions.items():

        # Compute generation split by fuel
        split = compute_generation_split_by_fuel(df, inputs)

        # Add scenario identifier for DataFrame construction
        split["scenario"] = scen

        rows.append(split)

    # Combine into DataFrame
    df_compare = pd.DataFrame(rows)

    # Replace missing values (technologies absent in some scenarios)
    df_compare = df_compare.fillna(0)

    # Use scenario as index for cleaner representation
    df_compare = df_compare.set_index("scenario")

    return df_compare
# =========================================================
# DETAILED GENERATION SPLIT
# =========================================================

def compute_generation_split_by_fuel(df, inputs):
    """
    Compute total generation per fuel/technology.

    This function aggregates generation values across all units and assigns
    them to fuel categories based on predefined mappings. It also includes
    hydro generation and non-conventional generation as separate categories.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing dispatch results with at least:
        - "type" : str (e.g., "p")
        - "unit" : str
        - "value" : float

    inputs : object
        Container with system input data. Expected attributes:
        - inputs.thermal_units
        - inputs.other_res_units
        - inputs.non_res_units
        - inputs.df_hydro

    Returns
    -------
    dict
        Dictionary mapping fuel/technology names to total generation (float).

    Notes
    -----
    - Unit-to-fuel mapping is defined via `FUEL_MAP`.
    - Hydro generation is added independently from dispatch.
    - Non-conventional generation is computed via `compute_non_conventional`.
    """

    # =====================================================
    # UNIT → FUEL MAPPING
    # =====================================================
    unit_fuel = {}

    # Thermal units
    for u in inputs.thermal_units:
        unit_fuel[str(u.uid)] = FUEL_MAP.get(u.fuel_type, "Sonstige RES")

    # Other renewable units
    for u in inputs.other_res_units:
        unit_fuel[str(u.uid)] = FUEL_MAP.get(u.fueltype, "Sonstige RES")

    # Non-renewable units
    for u in inputs.non_res_units:
        unit_fuel[str(u.uid)] = "Sonstige nicht-RES"

    split = {}

    # =====================================================
    # DISPATCH GENERATION (type = "p")
    # =====================================================
    gen = df[df["type"] == "p"]

    # Aggregate generation per unit
    grouped = gen.groupby("unit")["value"].sum()

    for unit, val in grouped.items():
        # Map unit to fuel category
        fuel = match_unit_to_fuel(unit, unit_fuel)

        # Accumulate generation per fuel
        split[fuel] = split.get(fuel, 0) + val

    # =====================================================
    # HYDRO GENERATION
    # =====================================================
    if isinstance(inputs.df_hydro, dict):
        hydro = sum(
            df_h.select_dtypes(include="number").sum().sum()
            for df_h in inputs.df_hydro.values()
        )
    else:
        hydro = inputs.df_hydro.select_dtypes(include="number").sum().sum()

    if hydro > 0:
        split["Hydro"] = hydro

    # =====================================================
    # NON-CONVENTIONAL GENERATION
    # =====================================================
    # Computed separately (e.g., RES after curtailment, storage effects)
    non_conv = compute_non_conventional(df, inputs)

    if non_conv > 0:
        split["Nicht-konventionell"] = non_conv

    return split
def compute_non_conventional(df, inputs):
    """
    Compute non-conventional generation as a residual of the system balance.

    This function derives non-conventional generation implicitly using a
    system-wide energy balance. It ensures consistency with total demand,
    generation, and storage operation.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing model results with at least:
        - "type" : str (e.g., "p", "charge", "discharge")
        - "value" : float

    inputs : object
        Container with system input data. Expected:
        - inputs.df_load : pandas.DataFrame
            Electricity demand time series

    Returns
    -------
    float
        Non-conventional generation (MWh), constrained to be non-negative.

    Notes
    -----
    - The residual is computed as:
        non_conv = demand + (charge − discharge) − conventional_generation
    - This represents generation not explicitly captured by dispatch variables,
      e.g., variable renewables or imports.
    - The result is clipped at zero to avoid negative artifacts from numerical
      inconsistencies.
    """

    # =====================================================
    # DEMAND
    # =====================================================
    demand = inputs.df_load.select_dtypes(include="number").sum().sum()

    # =====================================================
    # EXPLICIT GENERATION (dispatch variables)
    # =====================================================
    gen = df[df["type"] == "p"]["value"].sum()

    # =====================================================
    # STORAGE OPERATION
    # =====================================================
    battery_charge = df[df["type"] == "charge"]["value"].sum()
    battery_discharge = df[df["type"] == "discharge"]["value"].sum()

    # =====================================================
    # RESIDUAL CALCULATION
    # =====================================================
    non_conv = (
        demand
        + (battery_charge - battery_discharge)
        - gen
    )

    # Enforce non-negative result (physical constraint)
    return max(non_conv, 0)
def plot_generation_pie_detailed(split, scenario):

    import matplotlib.pyplot as plt
    import numpy as np

    label = scenario_labels.get(scenario, scenario)

    # -----------------------------
    # 🔹 Clean + Filter
    # -----------------------------
    filtered = {k: v for k, v in split.items() if v > 1e-6}

    if not filtered:
        print(f"Keine Daten für {scenario}")
        return

    order = [
        "Braunkohle", "Kohle", "Gas",
        "Kernenergie",
        "Biomasse", "Müllverbrennung",
        "Wasserstoff",
        "Hydro",
        "Nicht-konventionell",
        "Sonstige RES",
        "Sonstige nicht-RES",
    ]

    data = {k: filtered[k] for k in order if k in filtered}
    if not data:
        data = filtered

    labels = list(data.keys())
    values = np.array(list(data.values()))
    colors = [COLOR_MAP.get(l, "#cccccc") for l in labels]

    # -----------------------------
    # 🔹 Anteile berechnen
    # -----------------------------
    shares = values / values.sum() * 100

    # -----------------------------
    # 🔹 Kleine Anteile bündeln
    # -----------------------------
    threshold = 1  # %
    main_mask = shares >= threshold

    labels_main = [l for l, m in zip(labels, main_mask) if m]
    values_main = [v for v, m in zip(values, main_mask) if m]
    colors_main = [c for c, m in zip(colors, main_mask) if m]

    other_value = sum(v for v, m in zip(values, main_mask) if not m)

    if other_value > 0:
        labels_main.append("Sonstige")
        values_main.append(other_value)
        colors_main.append("#bbbbbb")

    labels = labels_main
    values = np.array(values_main)
    colors = colors_main

    # neu berechnen
    shares = values / values.sum() * 100

    # -----------------------------
    # 🔹 Sortieren
    # -----------------------------
    idx = np.argsort(-shares)
    labels = [labels[i] for i in idx]
    values = values[idx]
    colors = [colors[i] for i in idx]
    shares = shares[idx]

    # -----------------------------
    # 🔹 Plot
    # -----------------------------
    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1])

    ax = fig.add_subplot(gs[0])
    ax_leg = fig.add_subplot(gs[1])
    ax_leg.axis("off")

    wedges, texts, autotexts = ax.pie(
        values,
        colors=colors,
        startangle=90,
        wedgeprops=dict(edgecolor="white"),
        autopct=lambda p: f"{p:.1f}%" if p > 5 else "",  # 🔥 nur große Anteile
        pctdistance=0.7
    )

    # -----------------------------
    # 🔹 Prozent-Labels schöner machen
    # -----------------------------
    for t in autotexts:
        t.set_color("white")
        t.set_fontsize(16)
        t.set_weight("bold")

    # -----------------------------
    # 🔹 Legende (ohne %)
    # -----------------------------
    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=c, markersize=10)
        for c in colors
    ]

    ax_leg.legend(
        legend_handles,
        labels,
        loc="center left",
        frameon=False,
        title="Technologien"
    )

    # -----------------------------
    # 🔹 Titel
    # -----------------------------
    ax.set_title(f"Erzeugungsmix – {label}")

    plt.tight_layout()

    save_figure(fig, scenario + "_generation_pie", subfolder="generation")

    plt.close()
# =========================================================
# GENERATION SPLIT PER COUNTRY
# =========================================================

def compute_generation_split_per_country(df, inputs, countries):
    """
    Compute generation breakdown per country using a balance-based approach.

    This function aggregates generation by technology for each country and
    derives non-conventional generation as a residual of the country-level
    energy balance.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing model results with at least:
        - "type" : str (e.g., "p", "p_non_res", "charge")
        - "value" : float
        - "country" : str
        - "unit" : str

    inputs : object
        Container with system input data. Expected attributes:
        - inputs.df_load : pandas.DataFrame
            Electricity demand per country
        - inputs.df_hydro : pandas.DataFrame or dict of DataFrames
            Hydro generation
        - inputs.thermal_units : iterable
        - inputs.other_res_units : iterable

    countries : list of str
        List of country codes.

    Returns
    -------
    pandas.DataFrame
        DataFrame with generation components per country:
        - "country"
        - "Thermal"
        - "Hydro"
        - "Other RES"
        - "Non-RES"
        - "Non-Conventional"

    Notes
    -----
    - Non-conventional generation is computed as a residual:
        non_conv = demand + battery_charge − conventional_generation
    - This implicitly captures generation not explicitly modeled in dispatch
      variables (e.g., variable renewables, imports).
    - Results may be sensitive to inconsistencies in the underlying data
      (e.g., missing flows or storage terms).
    """

    import pandas as pd

    results = []

    # =====================================================
    # UNIT ID SETS FOR CLASSIFICATION
    # =====================================================
    thermal_ids = set(str(u.uid) for u in inputs.thermal_units)
    other_res_ids = set(str(u.uid) for u in inputs.other_res_units)

    # =====================================================
    # LOOP OVER COUNTRIES
    # =====================================================
    for c in countries:

        # -------------------------------------------------
        # DEMAND
        # -------------------------------------------------
        demand = inputs.df_load[c].sum()

        # -------------------------------------------------
        # BATTERY CHARGE (energy absorbed from system)
        # -------------------------------------------------
        battery_charge = df[
            (df["type"] == "charge") &
            (df["country"] == c)
        ]["value"].sum()

        # -------------------------------------------------
        # THERMAL GENERATION
        # -------------------------------------------------
        thermal = df[
            (df["type"] == "p") &
            (df["unit"].isin(thermal_ids)) &
            (df["country"] == c)
        ]["value"].sum()

        # -------------------------------------------------
        # OTHER RENEWABLE GENERATION
        # -------------------------------------------------
        other_res = df[
            (df["type"] == "p") &
            (df["unit"].isin(other_res_ids)) &
            (df["country"] == c)
        ]["value"].sum()

        # -------------------------------------------------
        # NON-RENEWABLE (AGGREGATED VARIABLE)
        # -------------------------------------------------
        non_res = df[
            (df["type"] == "p_non_res") &
            (df["country"] == c)
        ]["value"].sum()

        # -------------------------------------------------
        # HYDRO GENERATION
        # -------------------------------------------------
        df_hydro = inputs.df_hydro

        if isinstance(df_hydro, dict):
            hydro = sum(
                df_h[c].sum()
                for df_h in df_hydro.values()
                if c in df_h.columns
            )
        else:
            hydro = df_hydro[c].sum()

        # -------------------------------------------------
        # RESIDUAL (NON-CONVENTIONAL GENERATION)
        # -------------------------------------------------
        non_conventional = (
            demand
            + battery_charge
            - (thermal + hydro + other_res + non_res)
        )

        results.append({
            "country": c,
            "Thermal": thermal,
            "Hydro": hydro,
            "Other RES": other_res,
            "Non-RES": non_res,
            "Non-Conventional": non_conventional
        })

    return pd.DataFrame(results)
# =========================================================
# PLOT PER COUNTRY (ALL SCENARIOS)
# =========================================================

def plot_generation_per_country_all_scenarios(solutions, inputs, countries):
    """
    Plot generation mix per country across multiple scenarios (stacked bars).
    """

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # =====================================================
    # FIXED SCENARIO ORDER (IMPORTANT)
    # =====================================================
    scenario_order = ["no_inertia", "thermal_only", "thermal_plus_virtual"]
    scenarios = [s for s in scenario_order if s in solutions]

    # =====================================================
    # TECHNOLOGY ORDER (CONSISTENT)
    # =====================================================
    tech_order = ["Thermal", "Hydro", "Other RES", "Non-RES", "Non-Conventional"]

    for c in countries:

        fig, ax = plt.subplots(figsize=(8, 5))

        data = []

        # =====================================================
        # COLLECT DATA
        # =====================================================
        for s in scenarios:
            df = solutions[s]

            split_df = compute_generation_split_per_country(
                df, inputs, [c]
            )

            data.append(split_df.iloc[0])

        data = pd.DataFrame(data, index=scenarios)

        # =====================================================
        # STACKED BAR
        # =====================================================
        bottom = np.zeros(len(scenarios))

        for tech in tech_order:

            values = data[tech].fillna(0).values
            color = COLOR_MAP.get(tech, "#cccccc")

            ax.bar(
                scenarios,
                values,
                bottom=bottom,
                label=tech,
                color=color
            )

            bottom += values

        # =====================================================
        # LABELS (GERMAN + SCENARIO LABELS)
        # =====================================================
        ax.set_xticklabels(
            [scenario_labels.get(s, s) for s in scenarios],
            rotation=20
        )

        ax.set_title(f"Erzeugungsmix – {c}")
        ax.set_ylabel("Energie (MWh)")

        ax.grid(axis="y", linestyle="--", alpha=0.3)

        # =====================================================
        # LEGEND (OUTSIDE → CLEAN)
        # =====================================================
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            frameon=False
        )

        plt.tight_layout()

        # =====================================================
        # SAVE
        # =====================================================
        save_figure(
            fig,
            f"generation_{c}_all_scenarios",
            subfolder="generation"
        )

        plt.close(fig)
# =========================================================
# NET IMPORT CALCULATION PER COUNTRY
# =========================================================

def compute_net_imports(df, countries):
    """
    Compute total net electricity imports per country.

    Net imports are calculated as:
        net_import = inflow − outflow + slack_import

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing flow and slack variables with at least:
        - "type" : str ("flow", "slack")
        - "unit" : str (encoded flow direction)
        - "value" : float
        - "country" : str (for slack variables)

    countries : list of str
        List of country codes.

    Returns
    -------
    pandas.DataFrame
        DataFrame with:
        - "country" : str
        - "net_import" : float

    Notes
    -----
    - Flow direction is inferred from the "unit" string:
        * "..._<country>" → inflow
        * "<country>_..." → outflow
    - Slack variables represent external imports (e.g., unmet demand or boundary flows).
    """

    import pandas as pd

    results = []

    for c in countries:

        # =====================================================
        # INFLOW (imports into country)
        # =====================================================
        inflow = df[
            (df["type"] == "flow") &
            (df["unit"].str.endswith(f"_{c}"))
        ]["value"].sum()

        # =====================================================
        # OUTFLOW (exports from country)
        # =====================================================
        outflow = df[
            (df["type"] == "flow") &
            (df["unit"].str.startswith(f"{c}_"))
        ]["value"].sum()

        # =====================================================
        # SLACK IMPORTS (external balancing)
        # =====================================================
        slack_import = df[
            (df["type"] == "slack") &
            (df["country"] == c)
        ]["value"].sum()

        # =====================================================
        # NET IMPORT
        # =====================================================
        net_import = inflow - outflow + slack_import

        results.append({
            "country": c,
            "net_import": net_import
        })

    return pd.DataFrame(results)        

# =========================================================
# PLOT: NET IMPORTS COMPARISON
# =========================================================

def plot_net_imports(solutions, countries):
    """
    Plot net electricity imports per country across multiple scenarios.

    This function compares net imports (imports − exports + slack) for each
    country and scenario using a grouped bar chart.

    Parameters
    ----------
    solutions : dict
        Dictionary mapping scenario names (str) to DataFrames containing
        model results.

    countries : list of str
        List of country codes.

    Returns
    -------
    None
        The function saves the plot to disk and closes the figure.

    Notes
    -----
    - Net imports are computed via `compute_net_imports`.
    - Positive values indicate net imports; negative values indicate net exports.
    - A horizontal zero line is included for visual reference.
    """

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd  # 🔥 missing import in your original code

    scenarios = list(solutions.keys())

    # =====================================================
    # DATA COLLECTION
    # =====================================================
    all_data = []

    for s in scenarios:
        df = solutions[s]

        # Compute net imports per country
        net_df = compute_net_imports(df, countries)
        net_df["scenario"] = s

        all_data.append(net_df)

    # Combine all scenarios into one DataFrame
    data = pd.concat(all_data)

    # =====================================================
    # DATA TRANSFORMATION
    # =====================================================
    # Pivot: rows = countries, columns = scenarios
    pivot = data.pivot(
        index="country",
        columns="scenario",
        values="net_import"
    ).fillna(0)

    # =====================================================
    # PLOTTING
    # =====================================================
    x = np.arange(len(pivot.index))
    width = 0.8 / len(scenarios)  # dynamic width (scales better)

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, s in enumerate(scenarios):
        ax.bar(
            x + i * width,
            pivot[s],
            width,
            label=s
        )

    # =====================================================
    # AXIS FORMATTING
    # =====================================================
    ax.set_xticks(x + width * (len(scenarios) - 1) / 2)
    ax.set_xticklabels(pivot.index, rotation=45)

    ax.set_ylabel("Net imports (MWh)")
    ax.set_title("Net imports by country")

    # Reference line: distinguishes imports vs exports
    ax.axhline(0)

    ax.legend()

    plt.tight_layout()

    # =====================================================
    # SAVE OUTPUT
    # =====================================================
    save_figure(fig, "net_imports", subfolder="generation")

    plt.close()
    
# =========================================================
# TOTAL CROSS-BORDER FLOW
# =========================================================

def compute_total_cross_border_flow(df):
    """
    Compute total absolute cross-border electricity flows.

    This function aggregates all cross-border flows by summing the absolute
    values of flow variables across all time steps.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing model results with at least:
        - "type" : str (must include "flow")
        - "value" : float
        - "hour" : optional (used to filter time-dependent entries)

    Returns
    -------
    float
        Total absolute cross-border flow (MWh).

    Notes
    -----
    - Absolute values are used to account for bidirectional flows.
    - This metric reflects total system utilization of interconnectors,
      not net exchanges.
    """

    # Filter relevant flow variables
    flows = df[
        (df["type"] == "flow") &
        (df["hour"].notna())
    ].copy()

    # Convert to absolute values (captures total volume regardless of direction)
    flows["abs_flow"] = flows["value"].abs()

    # Aggregate total flow
    total_flow = flows["abs_flow"].sum()

    return total_flow 
# =========================================================
# COMPARE TOTAL FLOWS
# =========================================================

def compare_total_flows(solutions):
    """
    Compare total cross-border flows across scenarios.

    Parameters
    ----------
    solutions : dict
        Dictionary mapping scenario names (str) to DataFrames.

    Returns
    -------
    pandas.DataFrame
        DataFrame with:
        - "scenario" : str
        - "total_flow" : float
    """

    import pandas as pd

    results = []

    for s, df in solutions.items():
        total = compute_total_cross_border_flow(df)

        results.append({
            "scenario": s,
            "total_flow": total
        })

    return pd.DataFrame(results)
# =========================================================
# PLOT TOTAL FLOWS
# =========================================================
def plot_total_flows(results):
    """
    Plot total cross-border flows per scenario.

    Parameters
    ----------
    results : pandas.DataFrame
        Output of `compare_total_flows`, containing:
        - scenario
        - total_flow
    """

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # =====================================================
    # FIXED SCENARIO ORDER
    # =====================================================
    scenario_order = ["no_inertia", "thermal_only", "thermal_plus_virtual"]

    df = results.copy()
    df = df.set_index("scenario").reindex(scenario_order).reset_index()

    # =====================================================
    # LABELS
    # =====================================================
    labels = [scenario_labels.get(s, s) for s in df["scenario"]]
    values = df["total_flow"].values

    x = np.arange(len(labels))

    # =====================================================
    # COLORS (optional but useful)
    # =====================================================
    colors = ["#4C72B0", "#55A868", "#C44E52"][:len(labels)]

    # =====================================================
    # PLOT
    # =====================================================
    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(x, values, color=colors)

    # -----------------------------------------------------
    # VALUE LABELS
    # -----------------------------------------------------
    ymax = max(values) if len(values) > 0 else 1
    offset = ymax * 0.02 if ymax > 0 else 1

    for i, v in enumerate(values):
        if v > 0:
            ax.text(
                i,
                v + offset,
                f"{v:.0f}",
                ha="center",
                va="bottom",
                fontsize=10
            )

    # =====================================================
    # AXES
    # =====================================================
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)

    ax.set_ylabel("Grenzüberschreitende Flüsse (MWh)")
    ax.set_title("Gesamte grenzüberschreitende Flüsse")

    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()

    # =====================================================
    # SAVE
    # =====================================================
    save_figure(fig, "total_cross_border_flow", subfolder="flow")

    plt.close(fig)

# =========================================================
# COMPARE FLOWS WITH BASE SCENARIO
# =========================================================

def compare_total_flows_with_base(solutions, base="no_inertia"):
    """
    Compute total cross-border flows and deviations relative to a base scenario.

    Parameters
    ----------
    solutions : dict
        Dictionary mapping scenario names to DataFrames.

    base : str, optional
        Base scenario for comparison.

    Returns
    -------
    pandas.DataFrame
        DataFrame with:
        - "scenario"
        - "total_flow"
        - "delta" : absolute difference (MWh)
        - "percent_change" : relative change (%)

    Notes
    -----
    - Percent change is computed relative to the base scenario:
        (scenario − base) / base × 100
    - If base flow is zero, percent change is set to zero to avoid division errors.
    """

    import pandas as pd

    results = []

    # =====================================================
    # BASE SCENARIO
    # =====================================================
    base_flow = compute_total_cross_border_flow(solutions[base])

    # =====================================================
    # LOOP OVER SCENARIOS
    # =====================================================
    for s, df in solutions.items():

        total = compute_total_cross_border_flow(df)

        delta = total - base_flow

        percent = (delta / base_flow * 100) if base_flow != 0 else 0

        results.append({
            "scenario": s,
            "total_flow": total,
            "delta": delta,
            "percent_change": percent
        })

    return pd.DataFrame(results)    
# =========================================================
# PLOT DELTA FLOWS (RELATIVE TO BASE)
# =========================================================
def plot_flow_changes(results, base="no_inertia"):
    """
    Plot absolute changes in cross-border flows relative to a base scenario.
    """

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # =====================================================
    # FIXED SCENARIO ORDER
    # =====================================================
    scenario_order = ["no_inertia", "thermal_only", "thermal_plus_virtual"]

    df = results.copy()
    df = df[df["scenario"] != base]
    df = df.set_index("scenario").reindex(scenario_order).dropna().reset_index()

    # =====================================================
    # LABELS
    # =====================================================
    base_label = scenario_labels.get(base, base)

    labels = [scenario_labels.get(s, s) for s in df["scenario"]]
    values = df["delta"].values

    x = np.arange(len(labels))

    # =====================================================
    # COLOR LOGIC (VERY IMPORTANT)
    # =====================================================
    # Increase = red, decrease = green
    colors = ["#d62728" if v > 0 else "#2ca02c" for v in values]

    # =====================================================
    # PLOT
    # =====================================================
    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(x, values, color=colors)

    # -----------------------------------------------------
    # VALUE LABELS
    # -----------------------------------------------------
    ymax = max(abs(values)) if len(values) > 0 else 1
    offset = ymax * 0.05

    for i, v in enumerate(values):
        ax.text(
            i,
            v + (offset if v >= 0 else -offset),
            f"{v:.0f}",
            ha="center",
            va="bottom" if v >= 0 else "top",
            fontsize=10
        )

    # =====================================================
    # AXES
    # =====================================================
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)

    ax.set_ylabel("Änderung der Flüsse (MWh)")
    ax.set_title(f"Änderung der grenzüberschreitenden Flüsse (vs. {base_label})")

    # Reference line
    ax.axhline(0, color="black", linewidth=1)

    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()

    # =====================================================
    # SAVE
    # =====================================================
    save_figure(fig, "flow_change_vs_base", subfolder="flow")

    plt.close(fig)


def plot_flow_percent(results, base="no_inertia"):
    """
    Plot relative changes in cross-border flows (% vs base scenario).
    """

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # =====================================================
    # FIXED SCENARIO ORDER
    # =====================================================
    scenario_order = ["no_inertia", "thermal_only", "thermal_plus_virtual"]

    df = results.copy()
    df = df[df["scenario"] != base]
    df = df.set_index("scenario").reindex(scenario_order).dropna().reset_index()

    # =====================================================
    # LABELS
    # =====================================================
    base_label = scenario_labels.get(base, base)

    labels = [scenario_labels.get(s, s) for s in df["scenario"]]
    values = df["percent_change"].values

    x = np.arange(len(labels))

    # =====================================================
    # COLOR LOGIC
    # =====================================================
    # Increase = red, decrease = green
    colors = ["#d62728" if v > 0 else "#2ca02c" for v in values]

    # =====================================================
    # PLOT
    # =====================================================
    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(x, values, color=colors)

    # -----------------------------------------------------
    # VALUE LABELS
    # -----------------------------------------------------
    ymax = max(abs(values)) if len(values) > 0 else 1
    offset = ymax * 0.05


    # =====================================================
    # AXES
    # =====================================================
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)

    ax.set_ylabel("Relative Änderung (%)")
    ax.set_title(f"Relative Änderung der grenzüberschreitenden Flüsse (vs. {base_label})")

    # Reference line
    ax.axhline(0, color="black", linewidth=1)

    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()

    # =====================================================
    # SAVE
    # =====================================================
    save_figure(fig, "flow_percent_vs_base", subfolder="flow")

    plt.close(fig)
def plot_H_sys_heatmap(H_df, scenario, vmin=0, vmax=10):
    """
    Plot a heatmap of system inertia (H_sys) with consistent scaling
    and visual emphasis on critical thresholds.
    """

    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd

    # =====================================================
    # SCENARIO LABEL
    # =====================================================
    label = scenario_labels.get(scenario, scenario)

    # =====================================================
    # DATA CLEANING
    # =====================================================
    df = H_df.copy()

    df["H_sys"] = pd.to_numeric(df["H_sys"], errors="coerce")
    df = df.dropna(subset=["H_sys"])

    # =====================================================
    # PIVOT
    # =====================================================
    pivot = df.pivot(
        index="country",
        columns="hour",
        values="H_sys"
    ).astype(float)

    # Optional: sort countries by mean inertia (very useful!)
    pivot["__mean__"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("__mean__").drop(columns="__mean__")

    # =====================================================
    # PLOT
    # =====================================================
    fig, ax = plt.subplots(figsize=(14, max(6, len(pivot) * 0.4)))

    sns.heatmap(
        pivot,
        cmap="coolwarm",     
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        cbar_kws={"label": "System inertia H_sys"},
        mask=pivot.isna()
    )


    # =====================================================
    # AXIS FORMATTING
    # =====================================================
    hours = pivot.columns
    n_hours = len(hours)

    step = max(1, n_hours // 10)
    xticks = list(range(0, n_hours, step))

    ax.set_xticks(xticks)
    ax.set_xticklabels([int(hours[i]) for i in xticks], rotation=0)

    ax.set_ylabel("Country")
    ax.set_xlabel("Hour")

    ax.set_title(f"System inertia over time ({label})")

    plt.tight_layout()

    # =====================================================
    # SAVE
    # =====================================================
    save_figure(
        fig,
        f"{scenario}_Hsys_heatmap",
        subfolder="inertia"
    )

    plt.close(fig)

def plot_curtailment(solutions, countries):
    """
    Plot total curtailed energy per scenario with improved readability
    and interpretability.
    """

    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    # =====================================================
    # FIXED SCENARIO ORDER
    # =====================================================
    scenario_order = ["no_inertia", "thermal_only", "thermal_plus_virtual"]

    results = []

    # =====================================================
    # DATA AGGREGATION
    # =====================================================
    for s in scenario_order:

        if s not in solutions:
            continue

        df = solutions[s]

        curtail = df[df["type"] == "curtailement"]["value"].sum()

        results.append({
            "scenario": s,
            "curtailment": curtail
        })

    df_res = pd.DataFrame(results)

    # =====================================================
    # LABELS
    # =====================================================
    labels = [scenario_labels.get(s, s) for s in df_res["scenario"]]
    values = df_res["curtailment"].values

    x = np.arange(len(labels))

    # =====================================================
    # COLOR LOGIC (important!)
    # =====================================================
    # Lower curtailment = better → green
    base_value = values[0] if len(values) > 0 else 0

    colors = [
        "#2ca02c" if v <= base_value else "#d62728"
        for v in values
    ]

    # =====================================================
    # PLOT
    # =====================================================
    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(x, values, color=colors)

    # -----------------------------------------------------
    # VALUE LABELS
    # -----------------------------------------------------
    ymax = max(values) if len(values) > 0 else 1

    for i, v in enumerate(values):
        ax.text(
            i,
            v + ymax * 0.02,
            f"{v:,.0f}",
            ha="center",
            fontsize=11
        )

    # =====================================================
    # AXES
    # =====================================================
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)

    ax.set_ylabel("Abregelung (MWh)")
    ax.set_title("Gesamte Abregelung je Szenario")

    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()

    # =====================================================
    # SAVE
    # =====================================================
    save_figure(fig, "curtailment", subfolder="generation")

    plt.close(fig)
def plot_top_flow_changes(solutions, base="no_inertia"):
    """
    Plot percentage changes in key transmission flows (top lines).
    """

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.ticker import ScalarFormatter

    # =====================================================
    # FIXED SCENARIO ORDER
    # =====================================================
    scenario_order = ["no_inertia", "thermal_only", "thermal_plus_virtual"]

    # =====================================================
    # BASE FLOWS
    # =====================================================
    base_df = solutions[base]

    base_flows = (
        base_df[base_df["type"] == "flow"]
        .groupby("unit")["value"]
        .sum()
    )

    # Top 5 by absolute flow
    top_units = base_flows.abs().sort_values(ascending=False).head(5).index

    # =====================================================
    # LOOP SCENARIOS
    # =====================================================
    for s in scenario_order:

        if s not in solutions or s == base:
            continue

        df = solutions[s]

        flows = (
            df[df["type"] == "flow"]
            .groupby("unit")["value"]
            .sum()
        )

        # -------------------------------------------------
        # % CHANGE
        # -------------------------------------------------
        diff_pct = (
            (flows - base_flows)
            / (base_flows.abs() + 1e-6)
            * 100
        ).replace([np.inf, -np.inf], np.nan)

        top = diff_pct.reindex(top_units).dropna()

        # -------------------------------------------------
        # SORT
        # -------------------------------------------------
        top = top.sort_values()

        # -------------------------------------------------
        # LABELS (AFTER SORTING!)
        # -------------------------------------------------
        labels = []
        for u in top.index:
            parts = u.split("_")
            if len(parts) >= 3:
                labels.append(f"{parts[-2]} → {parts[-1]}")
            else:
                labels.append(u)

        # -------------------------------------------------
        # COLOR LOGIC
        # -------------------------------------------------
        colors = ["#d62728" if v > 0 else "#2ca02c" for v in top]

        # =====================================================
        # PLOT
        # =====================================================
        fig, ax = plt.subplots(figsize=(9, 5))

        ax.barh(labels, top, color=colors)

        ax.axvline(0, color="black", linewidth=1)

        # -------------------------------------------------
        # DYNAMIC SYMLOG
        # -------------------------------------------------
        max_val = np.nanmax(np.abs(top.values))
        linthresh = max(10, max_val * 0.05)

        ax.set_xscale("symlog", linthresh=linthresh)
        ax.xaxis.set_major_formatter(ScalarFormatter())

        # -------------------------------------------------
        # VALUE LABELS
        # -------------------------------------------------
        for i, v in enumerate(top):
            ax.annotate(
                f"{v:.1f}%",
                xy=(v, i),
                xytext=(6 if v > 0 else -6, 0),
                textcoords="offset points",
                va="center",
                ha="left" if v > 0 else "right",
                fontsize=11
            )

        # =====================================================
        # FORMATTING
        # =====================================================
        ax.set_xlabel("Änderung der Flüsse (%)")
        ax.set_ylabel("Leitung")

        ax.set_title(
            f"Top 5 Leitungen (nach Fluss)\n"
            f"Änderung relativ zum Basisszenario\n"
            f"({scenario_labels.get(s, s)} vs {scenario_labels.get(base, base)})"
        )

        ax.grid(axis="x", linestyle="--", alpha=0.4)

        plt.tight_layout()

        # =====================================================
        # SAVE
        # =====================================================
        save_figure(fig, f"flows_percent_top5_{s}", subfolder="flows")

        plt.close(fig)

# =========================================================
# INERTIA SHARE PER COUNTRY
# =========================================================

def compute_inertia_share_per_country(df, countries):
    """
    Compute the composition of system inertia per country.

    This function calculates the relative contribution of non-virtual
    (conventional) and virtual inertia to total system inertia for each country.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing inertia variables with at least:
        - "type" : str (must include "inertia")
        - "unit" : str (e.g., "non_virtual_*", "virtual_*", "total_*")
        - "value" : float
        - "country" : str

    countries : list of str
        List of country codes.

    Returns
    -------
    pandas.DataFrame
        DataFrame with:
        - "country"
        - "Non-Virtual (%)"
        - "Virtual (%)"

    Notes
    -----
    - Shares are computed relative to total inertia per country.
    - Countries with zero total inertia are excluded.
    - The classification relies on naming conventions in `unit`.
    """

    import pandas as pd

    results = []

    for c in countries:

        # =====================================================
        # INERTIA COMPONENTS
        # =====================================================
        non_virtual = df[
            (df["type"] == "inertia") &
            (df["unit"].str.startswith("non_virtual_")) &
            (df["country"] == c)
        ]["value"].sum()

        virtual = df[
            (df["type"] == "inertia") &
            (df["unit"].str.startswith("virtual_")) &
            (df["country"] == c)
        ]["value"].sum()

        total = df[
            (df["type"] == "inertia") &
            (df["unit"].str.startswith("total_")) &
            (df["country"] == c)
        ]["value"].sum()

        # =====================================================
        # SAFETY CHECK
        # =====================================================
        if total == 0:
            continue

        results.append({
            "country": c,
            "Non-Virtual (%)": non_virtual / total * 100,
            "Virtual (%)": virtual / total * 100
        })

    return pd.DataFrame(results)
def plot_inertia_share_per_country(df_share, scenario):
    """
    Plot and export inertia composition per country.
    """

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import os

  

    label = scenario_labels.get(scenario, scenario)

    # -----------------------------
    # Safety
    # -----------------------------
    if df_share is None or df_share.empty:
        print("Keine Daten")
        return

    df = df_share.copy()

    # -----------------------------
    # Clean
    # -----------------------------
    df["Non-Virtual (%)"] = pd.to_numeric(df["Non-Virtual (%)"], errors="coerce").fillna(0)
    df["Virtual (%)"] = pd.to_numeric(df["Virtual (%)"], errors="coerce").fillna(0)

    # -----------------------------
    # Sortieren (wie im Plot!)
    # -----------------------------
    df = df.sort_values("Virtual (%)").reset_index(drop=True)

    # -------------------------------------
    # 🔹 Excel-Daten vorbereiten
    # -------------------------------------
    df_export = df.copy()
    df_export = df_export.rename(columns={
        "Non-Virtual (%)": "Konventionell (%)",
        "Virtual (%)": "Virtuell (%)"
    })

    df_export = df_export.round(2)

    # -----------------------------
    # Plot
    # -----------------------------
    x = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(12, 6))

    non_virtual = df["Non-Virtual (%)"].values
    virtual = df["Virtual (%)"].values

    ax.bar(x, non_virtual, label="Konventionell", color="#4C72B0")
    ax.bar(x, virtual, bottom=non_virtual, label="Virtuell", color="#DD8452")

    ax.set_xticks(x)
    ax.set_xticklabels(df["country"], rotation=45, ha="right")

    ax.set_ylabel("Anteil (%)")
    ax.set_ylim(0, 100)

    ax.set_title(f"Systemträgheit nach Ländern ({label})")

    ax.legend()

    plt.tight_layout()

    save_figure(fig, f"{scenario}_inertia_share", subfolder="inertia")
    plt.close()

    # -------------------------------------
    # 🔹 Excel Export
    # -------------------------------------
    folder = os.path.join("output", "inertia")
    os.makedirs(folder, exist_ok=True)

    excel_path = os.path.join("../Results/2040", f"inertia_share_{scenario}.xlsx")

    df_export.to_excel(excel_path, index=False)

    print(f"Saved Excel: {excel_path}")
    

def plot_inertia_daily_profile(df, scenario, threshold=5):
    """
    Plot average daily inertia composition with improved filtering
    and ordering.
    """

    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    label = scenario_labels.get(scenario, scenario)

    # =====================================================
    # CLEAN
    # =====================================================
    df = df.copy()

    df["hour"] = pd.to_numeric(df["hour"], errors="coerce")
    df = df.dropna(subset=["hour"])
    df["hour"] = df["hour"].astype(int)

    df["Non-Virtual (%)"] = pd.to_numeric(
        df["Non-Virtual (%)"], errors="coerce"
    ).fillna(0)

    df["Virtual (%)"] = pd.to_numeric(
        df["Virtual (%)"], errors="coerce"
    ).fillna(0)

    df["hour_of_day"] = ((df["hour"] - 1) % 24) + 1

    # =====================================================
    # DAILY PROFILE
    # =====================================================
    grouped = (
        df.groupby(["country", "hour_of_day"])[
            ["Non-Virtual (%)", "Virtual (%)"]
        ]
        .mean()
        .reset_index()
    )

    # =====================================================
    # BETTER FILTERING
    # =====================================================
    # Use mean instead of max (more robust!)
    relevance = grouped.groupby("country")["Virtual (%)"].mean()
    relevant = relevance[relevance > threshold].index

    grouped = grouped[grouped["country"].isin(relevant)]

    if grouped.empty:
        print("Keine relevanten Länder")
        return

    # =====================================================
    # SORT COUNTRIES (important!)
    # =====================================================
    order = relevance.sort_values(ascending=False).index
    countries = [c for c in order if c in grouped["country"].unique()]

    # =====================================================
    # PLOT
    # =====================================================
    n = len(countries)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=True)

    if n == 1:
        axes = [axes]

    for ax, country in zip(axes, countries):

        sub = grouped[grouped["country"] == country] \
            .sort_values("hour_of_day")

        h = sub["hour_of_day"].values
        non_virtual = sub["Non-Virtual (%)"].values
        virtual = sub["Virtual (%)"].values

        ax.stackplot(
            h,
            non_virtual,
            virtual,
            labels=["Konventionell", "Virtuell"],
            colors=["#4C72B0", "#DD8452"]
        )

        # 🔥 highlight high virtual periods
        ax.fill_between(
            h,
            0,
            virtual,
            where=(virtual > 50),
            color="#DD8452",
            alpha=0.3
        )

        ax.set_title(country)
        ax.set_ylabel("Anteil (%)")
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Stunde (typischer Tag)")

    fig.suptitle(f"Tagesprofil der Systemträgheit ({label})")

    axes[0].legend(loc="upper right")

    plt.tight_layout()

    save_figure(
        fig,
        f"inertia_daily_profile_{scenario}",
        subfolder="inertia"
    )

    plt.close()    
# =========================================================
# EXTRACT PERCENTAGE FACTORS (FILTER ZERO COUNTRIES)
# =========================================================

def compute_virtual_percentages(df, countries):
    """
    Extract percentage contributions of different virtual inertia sources per country.

    This function reads percentage-type variables from the model output and maps
    them to predefined technology categories. Countries with no contribution
    (all zeros) are excluded.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing model results with at least:
        - "type" : str (must include "percentage")
        - "unit" : str (technology-specific identifiers)
        - "value" : float
        - "country" : str

    countries : list of str
        List of country codes.

    Returns
    -------
    pandas.DataFrame
        DataFrame with percentage contributions per country and technology.

    Notes
    -----
    - Only the first matching entry per prefix is used.
    - Countries with zero total contribution are filtered out.
    - Results depend on consistent naming conventions in `unit`.
    """

    import pandas as pd

    results = []

    for c in countries:

        def get_val(prefix):
            subset = df[
                (df["type"] == "percentage") &
                (df["unit"].str.startswith(prefix)) &
                (df["country"] == c)
            ]["value"]

            return float(subset.iloc[0]) if not subset.empty else 0.0

        row = {
            "country": c,
            "Battery": get_val("battery_"),
            "Onshore Wind": get_val("onshore_"),
            "Offshore Wind": get_val("offshore_"),
            "PV Deload": get_val("pv_deloading_"),
            "PV Battery": get_val("pv_battery_"),
            "Roof Deload": get_val("rooftop_deloading_"),
            "Roof Battery": get_val("rooftop_battery_"),
        }

        # -------------------------------------------------
        # 🔥 FILTER: nur behalten, wenn nicht alles 0 ist
        # -------------------------------------------------
        total = sum(v for k, v in row.items() if k != "country")

        if total > 0:
            results.append(row)

    return pd.DataFrame(results)
# =========================================================
# FORMAT TABLE FOR DISPLAY
# =========================================================

def format_percentage_table(df_perc):
    """
    Format virtual inertia percentage table for reporting.

    This function sorts and rounds percentage values to improve readability.

    Parameters
    ----------
    df_perc : pandas.DataFrame
        Output of `compute_virtual_percentages`.

    Returns
    -------
    pandas.DataFrame
        Formatted DataFrame sorted by battery contribution.
    """

    df = df_perc.copy()

    # Sort by dominant technology (battery)
    df = df.sort_values("Battery", ascending=False)

    # Round numerical columns
    numeric_cols = df.columns.drop("country")
    df[numeric_cols] = df[numeric_cols].round(2)

    return df
# =========================================================
# VIRTUAL INERTIA BREAKDOWN (FILTER ZERO COUNTRIES)
# =========================================================

def compute_virtual_inertia_breakdown(df, countries):
    """
    Compute absolute contributions of virtual inertia technologies per country.

    This function aggregates inertia contributions by technology, distinguishing
    between deloading and battery-based provision where applicable.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing inertia variables with:
        - "type" : str (must include "inertia")
        - "unit" : str (technology identifiers)
        - "value" : float
        - "country" : str

    countries : list of str
        List of country codes.

    Returns
    -------
    pandas.DataFrame
        DataFrame with inertia contributions (absolute values) per technology.

    Notes
    -----
    - PV and rooftop contributions are split into deloading and battery components.
    - Countries with very low total virtual inertia are excluded (threshold = 10).
    - Results depend on naming conventions in `unit`.
    """

    import pandas as pd

    results = []

    for c in countries:

        # Helper: sum all inertia contributions for a prefix
        def get_sum(prefix):
            return df[
                (df["type"] == "inertia") &
                (df["unit"].str.startswith(prefix)) &
                (df["country"] == c)
            ]["value"].sum()

        row = {
            "country": c,

            "Battery": get_sum("battery_"),

            # PV split into deloading and battery contribution
            "PV (deloading)": get_sum("pv_") - get_sum("pv_batt_"),
            "PV (battery)": get_sum("pv_batt_"),

            # Rooftop split
            "Rooftop (deloading)": get_sum("roof_") - get_sum("roof_batt_"),
            "Rooftop (battery)": get_sum("roof_batt_"),

            "Onshore Wind": get_sum("onshore_"),
            "Offshore Wind": get_sum("offshore_")
        }

        # Filter out countries with negligible virtual inertia
        total_virtual = sum(row[k] for k in row if k != "country")

        if total_virtual > 10:
            results.append(row)

    return pd.DataFrame(results)
# =========================================================
# PLOT VIRTUAL INERTIA BREAKDOWN
# =========================================================


def plot_virtual_inertia_breakdown(df_breakdown):
    """
    Plot and export virtual inertia composition with improved sorting
    and interpretability.
    """

    import matplotlib.pyplot as plt
    import os
    import pandas as pd

    # -------------------------------------
    # 🔹 Index
    # -------------------------------------
    df = df_breakdown.set_index("country")

    # -------------------------------------
    # 🔹 Normalize
    # -------------------------------------
    df_pct_full = df.div(df.sum(axis=1), axis=0) * 100
    df_pct_full = df_pct_full.fillna(0)

    # -------------------------------------
    # 🔹 Copy
    # -------------------------------------
    df_pct_plot = df_pct_full.copy()

    # -------------------------------------
    # 🔹 SAFE column access helper
    # -------------------------------------
    def col(df, name):
        return df[name] if name in df.columns else 0

    # -------------------------------------
    # 🔹 PV + Rooftop
    # -------------------------------------
    df_pct_plot["PV/Rooftop (battery)"] = (
        col(df_pct_plot, "PV (battery)") +
        col(df_pct_plot, "Rooftop (battery)")
    )

    df_pct_plot["PV/Rooftop (deloading)"] = (
        col(df_pct_plot, "PV (deloading)") +
        col(df_pct_plot, "Rooftop (deloading)")
    )

    # -------------------------------------
    # 🔹 Wind
    # -------------------------------------
    df_pct_plot["Wind"] = (
        col(df_pct_plot, "Onshore Wind") +
        col(df_pct_plot, "Offshore Wind")
    )

    # -------------------------------------
    # 🔹 Nur relevante Spalten behalten
    # -------------------------------------
    order = [
        "Battery",
        "PV/Rooftop (deloading)",
        "PV/Rooftop (battery)",
        "Wind"
    ]

    df_pct_plot = df_pct_plot[[c for c in order if c in df_pct_plot.columns]]

    # -------------------------------------
    # 🔹 Drop leere Spalten (wichtig!)
    # -------------------------------------
    df_pct_plot = df_pct_plot.loc[:, (df_pct_plot.sum(axis=0) > 0)]

    if df_pct_plot.empty:
        print("⚠️ Keine Daten für Plot")
        return

    # -------------------------------------
    # 🔹 Sortieren
    # -------------------------------------
    if "Battery" in df_pct_plot.columns:
        df_pct_plot = df_pct_plot.sort_values("Battery")

    # -------------------------------------
    # 🔹 Farben (robust!)
    # -------------------------------------
    color_map = {
        "Battery": "#4C72B0",
        "PV/Rooftop (deloading)": "#F28E2B",
        "PV/Rooftop (battery)": "#59A14F",
        "Wind": "#9C755F"
    }

    colors = [color_map.get(c, "#cccccc") for c in df_pct_plot.columns]

    # -------------------------------------
    # 🔹 Plot
    # -------------------------------------
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 0.6])

    ax = fig.add_subplot(gs[0])
    ax_leg = fig.add_subplot(gs[1])
    ax_leg.axis("off")

    df_pct_plot.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=colors,
        width=0.85,
        edgecolor="white",
        linewidth=0.5
    )

    ax.set_ylabel("Anteil (%)")
    ax.set_title("Virtuelle Trägheitskomposition")

    ax.tick_params(axis='x', rotation=45)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    # -------------------------------------
    # 🔹 Legend
    # -------------------------------------
    handles, labels = ax.get_legend_handles_labels()
    ax.get_legend().remove()

    ax_leg.legend(
        handles,
        labels,
        loc="center left",
        frameon=False,
        title="Technologien"
    )

    plt.tight_layout()

    save_figure(fig, "virtual_inertia_breakdown", subfolder="inertia")
    plt.close(fig)

    # -------------------------------------
    # 🔹 Excel Export
    # -------------------------------------
    folder = os.path.join("../Results/2040")
    os.makedirs(folder, exist_ok=True)

    excel_path = os.path.join(folder, "virtual_inertia_breakdown.xlsx")

    with pd.ExcelWriter(excel_path) as writer:
        df_pct_full.round(2).to_excel(writer, sheet_name="Detail")
        df_pct_plot.round(2).to_excel(writer, sheet_name="Aggregiert")

    print(f"Saved Excel: {excel_path}")
    

def plot_Hsys_vs_demand(H_df, inputs, scenario):
    """
    Scatter plot of system inertia vs demand with trend and correlation.
    """

    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    label = scenario_labels.get(scenario, scenario)

    # =====================================================
    # PREPARE DEMAND
    # =====================================================
    df_load = inputs.df_load.copy()
    df_load["hour"] = range(1, len(df_load) + 1)

    df_load_long = df_load.melt(
        id_vars="hour",
        var_name="country",
        value_name="demand"
    )

    # =====================================================
    # MERGE
    # =====================================================
    merged = H_df.merge(
        df_load_long,
        on=["country", "hour"],
        how="inner"
    )

    # Optional: normalize demand per country
    merged["demand_norm"] = merged.groupby("country")["demand"] \
        .transform(lambda x: x / x.max())
    # =====================================================
    # CLEAN DATA
    # =====================================================
    merged = merged.copy()

    merged["demand"] = pd.to_numeric(merged["demand"], errors="coerce")
    merged["H_sys"] = pd.to_numeric(merged["H_sys"], errors="coerce")

    merged = merged.dropna(subset=["demand", "H_sys"])

    # Safety check
    if len(merged) < 2:
        print(f"⚠️ Not enough data for correlation in {scenario}")
        return

    x = merged["demand"].values
    y = merged["H_sys"].values
   
    # =====================================================
    # CORRELATION
    # =====================================================
    corr = np.corrcoef(x, y)[0, 1]

    # =====================================================
    # TREND LINE (linear fit)
    # =====================================================
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)

    # =====================================================
    # PLOT
    # =====================================================
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(
        x,
        y,
        alpha=0.4,
        s=10
    )

    # Trend line
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, p(x_line), linewidth=2)

    # =====================================================
    # LABELS
    # =====================================================
    ax.set_xlabel("Normalized demand")
    ax.set_ylabel("System inertia (H_sys)")
    ax.set_title(f"H_sys vs demand ({label})")

    # Correlation annotation
    ax.text(
        0.05,
        0.95,
        f"Correlation: {corr:.2f}",
        transform=ax.transAxes,
        va="top"
    )

    ax.grid(alpha=0.3)

    plt.tight_layout()

    save_figure(fig, f"{scenario}_Hsys_vs_demand", subfolder="system")
    plt.close()    
# =========================================================
# DELTA H_SYS HEATMAP (FIXED)
# =========================================================

def plot_delta_Hsys(
    solutions,
    inputs,
    countries,
    base="no_inertia",
    targets=("thermal_only", "thermal_plus_virtual"),
    fillna=False,
    enforce_all_countries=True
):
    """
    Plot changes in system inertia (H_sys) relative to a base scenario.

    This function computes the difference in inertia (ΔH_sys) between each
    target scenario and a base scenario and visualizes the results as heatmaps.

    Parameters
    ----------
    solutions : dict
        Dictionary mapping scenario names to DataFrames.

    inputs : object
        Input container required for H_sys extraction.

    countries : list of str
        List of country codes.

    base : str, optional
        Base scenario used for comparison.

    targets : tuple of str, optional
        Target scenarios to compare against the base.

    fillna : bool, optional
        If True, missing values are filled with zero. Otherwise shown as NaN.

    enforce_all_countries : bool, optional
        If True, ensures all countries appear in the heatmap.

    Returns
    -------
    None
        The function saves one heatmap per target scenario.

    Notes
    -----
    - A shared color scale is used across all plots for comparability.
    - The color scale is based on the 99th percentile to reduce outlier effects.
    - Missing values are optionally visualized as gaps.
    """

    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    # -----------------------------------------------------
    # Scenario label (human-readable)
    # -----------------------------------------------------
    base_label = scenario_labels.get(base, base)

    # -----------------------------------------------------
    # Compute base scenario inertia
    # -----------------------------------------------------
    H_base = extract_H_sys_per_country(
        solutions[base], inputs, countries
    )

    all_pivots = {}
    all_values = []

    # -----------------------------------------------------
    # Process target scenarios
    # -----------------------------------------------------
    for target in targets:

        if target not in solutions:
            print(f"Szenario fehlt: {target}")
            continue

        H_target = extract_H_sys_per_country(
            solutions[target], inputs, countries
        )

        # -------------------------------------------------
        # Merge base and target
        # -------------------------------------------------
        merged = H_target.merge(
            H_base,
            on=["country", "hour"],
            how="outer",
            suffixes=("_target", "_base")
        )

        # Ensure numeric values
        merged["H_sys_target"] = pd.to_numeric(
            merged["H_sys_target"], errors="coerce"
        )
        merged["H_sys_base"] = pd.to_numeric(
            merged["H_sys_base"], errors="coerce"
        )

        # Compute delta
        merged["delta"] = (
            merged["H_sys_target"] - merged["H_sys_base"]
        )

        # -------------------------------------------------
        # Pivot (country × time)
        # -------------------------------------------------
        pivot = merged.pivot(
            index="country",
            columns="hour",
            values="delta"
        ).apply(pd.to_numeric, errors="coerce")

        # Ensure all countries are included
        if enforce_all_countries:
            pivot = pivot.reindex(countries)

        # Optionally fill missing values
        if fillna:
            pivot = pivot.fillna(0)

        # Sort countries by average delta
        pivot["__mean__"] = pivot.mean(axis=1, skipna=True)
        pivot = pivot.sort_values("__mean__").drop(columns="__mean__")

        all_pivots[target] = pivot
        all_values.append(pivot.values.flatten())

    # -----------------------------------------------------
    # Shared color scale (robust against outliers)
    # -----------------------------------------------------
    all_values = np.concatenate(all_values)

    vmax = np.nanpercentile(np.abs(all_values), 99)

    if vmax == 0 or np.isnan(vmax):
        vmax = 1

    # -----------------------------------------------------
    # Plot per scenario
    # -----------------------------------------------------
    for target, pivot in all_pivots.items():

        target_label = scenario_labels.get(target, target)

        fig_height = max(6, pivot.shape[0] * 0.4)

        fig, ax = plt.subplots(figsize=(14, fig_height))

        sns.heatmap(
            pivot,
            cmap="coolwarm",
            center=0,
            vmin=-vmax,
            vmax=vmax,
            mask=pivot.isna(),
            ax=ax,
            cbar_kws={"label": "Δ Systemträgheit $H_{sys}$"}
        )

        # -------------------------------------------------
        # X-axis formatting (reduce tick density)
        # -------------------------------------------------
        hours = pivot.columns
        step = max(1, len(hours) // 10)

        xticks = list(range(0, len(hours), step))
        ax.set_xticks(xticks)
        ax.set_xticklabels(
            [int(hours[i]) for i in xticks],
            rotation=0
        )

        # -------------------------------------------------
        # Labels (German for publication)
        # -------------------------------------------------
        ax.set_title(
            f"Änderung der Systemträgheit\n"
            f"({target_label} vs. {base_label})"
        )
        ax.set_xlabel("Stunde")
        ax.set_ylabel("Land")

        plt.tight_layout()

        # -------------------------------------------------
        # Save figure
        # -------------------------------------------------
        save_figure(fig, f"delta_Hsys_{target}", subfolder="delta")

        plt.close()
# =========================================================
# INERTIA SLACK (TIME SERIES)
# =========================================================

def plot_inertia_slack(df, scenario):
    """
    Plot the time series of inertia slack (unmet inertia requirement).

    This function extracts slack variables associated with inertia constraints
    and visualizes their temporal evolution.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing model results with at least:
        - "type" : str (must include "slack")
        - "unit" : str (must include "inertia_*")
        - "value" : float
        - "hour" : int

    scenario : str
        Scenario identifier used for labeling and file naming.

    Returns
    -------
    None
        The function saves the plot and closes it.

    Notes
    -----
    - Slack represents unmet inertia requirements in the model.
    - Values are aggregated across all countries per time step.
    """

    import matplotlib.pyplot as plt
    import pandas as pd

    # -----------------------------------------------------
    # Scenario label (human-readable)
    # -----------------------------------------------------
    label = scenario_labels.get(scenario, scenario)

    # -----------------------------------------------------
    # Filter inertia slack variables
    # -----------------------------------------------------
    slack = df[
        (df["type"] == "slack") &
        (df["unit"].str.startswith("inertia"))
    ]

    if slack.empty:
        print(f"[INFO] No inertia slack in {scenario}")
        return

    # -----------------------------------------------------
    # Aggregate over time
    # -----------------------------------------------------
    slack_ts = (
        slack.groupby("hour")["value"]
        .sum()
        .sort_index()
    )

    # -----------------------------------------------------
    # Basic statistics (useful for interpretation)
    # -----------------------------------------------------
    total_slack = slack_ts.sum()
    max_slack = slack_ts.max()

    # -----------------------------------------------------
    # Plot
    # -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(slack_ts.index, slack_ts.values)

    # Highlight critical periods (optional but powerful)
    ax.fill_between(
        slack_ts.index,
        slack_ts.values,
        where=(slack_ts.values > 0),
        alpha=0.3
    )

    # -----------------------------------------------------
    # Labels (German)
    # -----------------------------------------------------
    ax.set_title(f"Trägheitslücke (Slack) – {label}")
    ax.set_ylabel("Trägheitslücke [MW·s]")
    ax.set_xlabel("Stunde")

    # -----------------------------------------------------
    # Optional annotation (very useful in thesis)
    # -----------------------------------------------------
    ax.text(
        0.01, 0.95,
        f"Summe: {total_slack:.0f} MW·s\nMax: {max_slack:.0f} MW·s",
        transform=ax.transAxes,
        va="top"
    )

    ax.grid(alpha=0.3)

    plt.tight_layout()

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------
    save_figure(
        fig,
        f"{scenario}_inertia_slack",
        subfolder="inertia_slack"
    )

    plt.close()
# =========================================================
# CROSS-COUNTRY CORRELATION (H_sys)
# =========================================================

def plot_Hsys_correlation(H_df):
    """
    Plot cross-country correlation of system inertia (H_sys).

    This function computes pairwise correlation coefficients between countries
    based on their hourly inertia time series and visualizes the result as a heatmap.

    Parameters
    ----------
    H_df : pandas.DataFrame
        DataFrame containing:
        - "hour" : int
        - "country" : str
        - "H_sys" : float

    Returns
    -------
    None
        The function saves the correlation heatmap and closes the figure.

    Notes
    -----
    - Correlation is computed using Pearson correlation (default in pandas).
    - Missing values are handled automatically by pairwise deletion.
    """

    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    # -----------------------------------------------------
    # Pivot (time × country)
    # -----------------------------------------------------
    pivot = H_df.pivot(
        index="hour",
        columns="country",
        values="H_sys"
    )

    # Ensure numeric values
    pivot = pivot.apply(pd.to_numeric, errors="coerce")

    # -----------------------------------------------------
    # Correlation matrix
    # -----------------------------------------------------
    corr = pivot.corr()

    # Optional: sort countries by average correlation
    mean_corr = corr.mean().sort_values()
    corr = corr.loc[mean_corr.index, mean_corr.index]

    # -----------------------------------------------------
    # Plot
    # -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
        cbar_kws={"label": "Korrelationskoeffizient"}
    )

    # -----------------------------------------------------
    # Labels (German)
    # -----------------------------------------------------
    ax.set_title("Korrelation der Systemträgheit $H_{sys}$ zwischen Ländern")
    ax.set_xlabel("Land")
    ax.set_ylabel("Land")

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------
    save_figure(fig, "Hsys_korrelation", subfolder="correlation")

    plt.close()
    
    # =========================================================
# CRITICAL HOURS BY TIME OF DAY
# =========================================================

def plot_critical_hours_by_time_and_country(H_df, scenario, threshold=2):
    """
    Plot distribution of critical low-inertia hours by country and time of day.

    This function identifies time steps where system inertia (H_sys) falls
    below a defined threshold and analyzes their distribution across countries
    and daytime categories.

    Parameters
    ----------
    H_df : pandas.DataFrame
        DataFrame containing:
        - "country" : str
        - "hour" : int
        - "H_sys" : float

    scenario : str
        Scenario identifier used for labeling and file naming.

    threshold : float, optional
        Critical inertia threshold. Hours with H_sys below this value are
        classified as critical. Default is 2.

    Returns
    -------
    None
        The function saves the plot and closes the figure.

    Notes
    -----
    - Daytime categories:
        * Night (0–5)
        * Morning (6–11)
        * Afternoon (12–17)
        * Evening (18–23)
    - Shares are computed relative to total hours per country.
    """

    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    # -----------------------------------------------------
    # Scenario label
    # -----------------------------------------------------
    label = scenario_labels.get(scenario, scenario)

    # -----------------------------------------------------
    # Filter critical hours
    # -----------------------------------------------------
    crit = H_df[H_df["H_sys"] < threshold].copy()

    if crit.empty:
        print(f"[INFO] No critical hours in {scenario}")
        return

    # -----------------------------------------------------
    # Convert to hour of day
    # -----------------------------------------------------
    crit["hour_of_day"] = (crit["hour"] - 1) % 24

    # Map hour → daytime category (German labels!)
    def hour_to_daytime(hour):
        if 0 <= hour <= 5:
            return "Nacht"
        elif 6 <= hour <= 11:
            return "Morgen"
        elif 12 <= hour <= 17:
            return "Nachmittag"
        else:
            return "Abend"

    crit["Tageszeit"] = crit["hour_of_day"].apply(hour_to_daytime)

    order = ["Nacht", "Morgen", "Nachmittag", "Abend"]

    # -----------------------------------------------------
    # Group by country and daytime
    # -----------------------------------------------------
    counts = (
        crit.groupby(["country", "Tageszeit"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=order, fill_value=0)
    )

    # -----------------------------------------------------
    # Normalize (share of total hours per country)
    # -----------------------------------------------------
    total_hours = H_df.groupby("country").size()
    shares = counts.div(total_hours, axis=0) * 100

    # -----------------------------------------------------
    # Sort countries by total risk (very important!)
    # -----------------------------------------------------
    shares["__sum__"] = shares.sum(axis=1)
    shares = shares.sort_values("__sum__", ascending=False)
    shares = shares.drop(columns="__sum__")

    # -----------------------------------------------------
    # Plot
    # -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))

    shares.plot(
        kind="bar",
        ax=ax,
        stacked=True
    )

    # -----------------------------------------------------
    # Labels (German)
    # -----------------------------------------------------
    ax.set_ylabel(f"Anteil kritischer Stunden (H_sys < {threshold}) [%]")
    ax.set_xlabel("Land")
    ax.set_title(f"Kritische Stunden nach Tageszeit ({label})")

    ax.tick_params(axis='x', rotation=45)

    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()

    save_figure(fig, f"{scenario}_daytime_country", subfolder="system")

    plt.close(fig)

def compute_critical_high_demand(H_df, inputs, countries, threshold=2):
    """
    Compute the share of critical low-inertia hours during periods of high demand.

    This function identifies hours with high electricity demand (top 25%)
    for each country and calculates the fraction of those hours where system
    inertia (H_sys) falls below a given threshold.

    Parameters
    ----------
    H_df : pandas.DataFrame
        DataFrame containing:
        - "country" : str
        - "hour" : int
        - "H_sys" : float

    inputs : object
        Input container with:
        - inputs.df_load : pandas.DataFrame (timestamp + demand per country)

    countries : list of str
        List of country codes.

    threshold : float, optional
        Critical inertia threshold (default: 2).

    Returns
    -------
    pandas.DataFrame
        DataFrame with:
        - "country"
        - "high_demand_threshold" : float (75th percentile of demand)
        - "critical_share_high_demand_%" : float
        - "n_hours" : int (number of high-demand hours)

    Notes
    -----
    - High-demand periods are defined as the top 25% of demand values.
    - The metric captures the overlap between system stress (low inertia)
      and peak demand conditions.
    """

    import pandas as pd

    # =====================================================
    # PREPARE DEMAND DATA
    # =====================================================
    demand_df = inputs.df_load.copy()

    ts_col = demand_df.columns[0]
    demand_df[ts_col] = pd.to_datetime(demand_df[ts_col])

    start_ts = demand_df[ts_col].min()

    # Convert timestamps → hour index
    demand_df["hour"] = (
        (demand_df[ts_col] - start_ts) / pd.Timedelta(hours=1)
    ).astype(int) + 1

    # Convert to long format
    demand_long = demand_df.melt(
        id_vars=["hour"],
        value_vars=countries,
        var_name="country",
        value_name="demand"
    )

    # =====================================================
    # MERGE WITH INERTIA DATA
    # =====================================================
    merged = H_df.merge(demand_long, on=["country", "hour"])

    # Identify critical hours
    merged["critical"] = merged["H_sys"] < threshold

    results = []

    # =====================================================
    # ANALYSIS PER COUNTRY
    # =====================================================
    for c in countries:

        sub = merged[merged["country"] == c]

        if sub.empty:
            continue

        # Threshold for high demand (75th percentile)
        q75 = sub["demand"].quantile(0.75)

        high_demand = sub[sub["demand"] >= q75]

        if len(high_demand) == 0:
            continue

        # Share of critical hours within high-demand periods
        share_critical = high_demand["critical"].mean() * 100

        results.append({
            "country": c,
            "high_demand_threshold": q75,
            "critical_share_high_demand_%": share_critical,
            "n_hours": len(high_demand)
        })

    return pd.DataFrame(results)
def plot_critical_high_demand(df_res, scenario):
    """
    Plot the share of critical low-inertia hours during high-demand periods.

    Parameters
    ----------
    df_res : pandas.DataFrame
        Output of `compute_critical_high_demand`.

    scenario : str
        Scenario identifier.

    Returns
    -------
    None
    """

    # -----------------------------------------------------
    # Scenario label (human-readable)
    # -----------------------------------------------------
    label = scenario_labels.get(scenario, scenario)

    # -----------------------------------------------------
    # Sort countries by risk level (descending)
    # → highlights most critical systems first
    # -----------------------------------------------------
    df_res = df_res.sort_values(
        "critical_share_high_demand_%",
        ascending=False
    ).reset_index(drop=True)

    # -----------------------------------------------------
    # X positions (robust against categorical issues)
    # -----------------------------------------------------
    x = np.arange(len(df_res))

    fig, ax = plt.subplots(figsize=(14, 6))

    # -----------------------------------------------------
    # Bar plot: share of critical hours under high demand
    # -----------------------------------------------------
    bars = ax.bar(
        x,
        df_res["critical_share_high_demand_%"]
    )

    # -----------------------------------------------------
    # Optional: highlight high-risk countries (>20%)
    # → improves interpretability immediately
    # -----------------------------------------------------
    for bar, val in zip(bars, df_res["critical_share_high_demand_%"]):
        if val > 20:
            bar.set_color("#d62728")  # red = critical

    # -----------------------------------------------------
    # Axis formatting (German for publication)
    # -----------------------------------------------------
    ax.set_ylabel("Anteil kritischer Stunden (%)")
    ax.set_xlabel("Land")

    ax.set_title(
        f"Kritische Stunden bei hoher Nachfrage (oberes Quartil)\n({label})"
    )

    # -----------------------------------------------------
    # Safe tick handling (explicit positions)
    # -----------------------------------------------------
    ax.set_xticks(x)
    ax.set_xticklabels(
        df_res["country"],
        rotation=45,
        ha="right"
    )

    # -----------------------------------------------------
    # Annotate values (important for interpretation)
    # -----------------------------------------------------
    for i, v in enumerate(df_res["critical_share_high_demand_%"]):
        if v > 0:
            ax.text(
                i,
                v + 0.5,
                f"{v:.1f}",
                ha="center",
                va="bottom",
                fontsize=10
            )

    # -----------------------------------------------------
    # Reference line (interpretation threshold)
    # -----------------------------------------------------
    ax.axhline(20, linestyle="--", alpha=0.6)

    # -----------------------------------------------------
    # Improve readability
    # -----------------------------------------------------
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()

    # -----------------------------------------------------
    # Save figure
    # -----------------------------------------------------
    save_figure(
        fig,
        f"critical_high_demand_{scenario}",
        subfolder="inertia"
    )

    plt.close()
def plot_battery_charge_all_countries(solutions, countries):
    """
    Plot total battery charging per country across multiple scenarios.
    """

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # -----------------------------------------------------
    # Data collection (aggregate charging per country)
    # -----------------------------------------------------
    data = []

    for scen, df in solutions.items():

        for c in countries:

            charge = df[
                (df["type"] == "charge") &
                (df["country"] == c)
            ]["value"].sum()

            data.append({
                "country": c,
                "scenario": scen,
                "charge": charge
            })

    df_plot = pd.DataFrame(data)

    # -----------------------------------------------------
    # Pivot (country × scenario)
    # -----------------------------------------------------
    pivot = df_plot.pivot(
        index="country",
        columns="scenario",
        values="charge"
    ).fillna(0)

    # -----------------------------------------------------
    # Sort countries by total charging (important!)
    # -----------------------------------------------------
    pivot["__total__"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("__total__", ascending=False)
    pivot = pivot.drop(columns="__total__")

    # -----------------------------------------------------
    # Plot setup
    # -----------------------------------------------------
    x = np.arange(len(pivot.index))
    scenarios = pivot.columns

    width = 0.8 / len(scenarios)

    fig, ax = plt.subplots(figsize=(14, 6))

    # -----------------------------------------------------
    # Bars per scenario
    # -----------------------------------------------------
    for i, scen in enumerate(scenarios):

        label = scenario_labels.get(scen, scen)

        ax.bar(
            x + i * width,
            pivot[scen],
            width,
            label=label
        )

    # -----------------------------------------------------
    # Axis formatting (German)
    # -----------------------------------------------------
    ax.set_xticks(x + width * (len(scenarios) - 1) / 2)
    ax.set_xticklabels(
        pivot.index,
        rotation=45,
        ha="right"
    )

    ax.set_ylabel("Gesamte Ladeenergie (MWh)")
    ax.set_xlabel("Land")

    ax.set_title("Batterieladung nach Ländern und Szenarien")

    # -----------------------------------------------------
    # Grid improves readability for magnitude comparison
    # -----------------------------------------------------
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    ax.legend()

    plt.tight_layout()

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------
    save_figure(
        fig,
        "battery_charge_all_countries",
        subfolder="battery"
    )

    plt.close()

# =========================================================
# GENERATION MIX PER COUNTRY (CONSISTENT COLORS & ORDER)
# =========================================================

def plot_generation_mix_per_country(df_mix, country):
    """
    Plot generation mix for a single country across multiple scenarios.

    This function creates a stacked bar chart showing the contribution of
    different generation technologies for each scenario. Technology order
    and color mapping are fixed to ensure consistency across plots.

    Parameters
    ----------
    df_mix : pandas.DataFrame
        DataFrame containing generation values with at least:
        - "country" : str
        - "scenario" : str
        - technology columns (e.g., "Gas", "Hydro", etc.)

    country : str
        Country code to filter the data.

    Returns
    -------
    None
        The function saves the figure and closes it.

    Notes
    -----
    - Missing technologies are handled automatically.
    - Technology order is fixed for comparability.
    - Colors are assigned via a global COLOR_MAP.
    """

    import matplotlib.pyplot as plt
    import numpy as np

    # =====================================================
    # 1) Filter data
    # =====================================================
    df = df_mix[df_mix["country"] == country].copy()
    df = df.set_index("scenario")

    # Optional: feste Szenario-Reihenfolge
    scenario_order = ["no_inertia", "thermal_only", "thermal_plus_virtual"]
    df = df.reindex(scenario_order)

    # =====================================================
    # 2) FIXED TECHNOLOGY ORDER (WICHTIG!)
    # =====================================================
    tech_order = [
        "Kernenergie",
        "Braunkohle",
        "Kohle",
        "Gas",
        "Schweröl",
        "Leichtöl",
        "Ölschiefer",
        "Wasserstoff",
        "Biomasse",
        "Müllverbrennung",
        "Hydro",
        "Battery",
        "Sonstige RES",
        "Sonstige nicht-RES"
    ]

    # Nur vorhandene Spalten nehmen
    tech_cols = [t for t in tech_order if t in df.columns]

    # =====================================================
    # 3) Plot vorbereiten
    # =====================================================
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(10, 6))

    bottom = np.zeros(len(df))

    # =====================================================
    # 4) Stacked bars
    # =====================================================
    for tech in tech_cols:

        values = df[tech].fillna(0).values
        color = COLOR_MAP.get(tech, "#cccccc")

        ax.bar(
            x,
            values,
            bottom=bottom,
            label=tech,
            color=color
        )

        bottom += values

    # =====================================================
    # 5) Layout
    # =====================================================
    ax.set_ylabel("Erzeugung MWh")  # ⚠️ wichtig korrigiert!
    ax.set_title(f"Generation Mix – {country}")

    ax.set_xticks(range(len(df.index)))
    ax.set_xticklabels(df.index, rotation=20)

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
        ncol=1
    )
    # =====================================================
    # 6) Save
    # =====================================================
    plt.tight_layout(rect=[0, 0, 0.8, 1])
    save_figure(fig, f"{country}_generation_mix_scenarios",subfolder="generation_mix")

    plt.close()
# =========================================================
# GENERATION MIX PER COUNTRY (MULTI-PANEL)
# =========================================================

def plot_generation_mix_all_countries(df_mix, countries):
    """
    Plot generation mix for multiple countries across scenarios.

    This function creates a multi-panel figure where each subplot represents
    one country. Within each subplot, stacked bars show the generation mix
    across scenarios.

    Parameters
    ----------
    df_mix : pandas.DataFrame
        DataFrame containing:
        - "country" : str
        - "scenario" : str
        - technology columns (e.g., "Gas", "Hydro", etc.)

    countries : list of str
        List of country codes to include in the plot.

    Returns
    -------
    None
        The function saves the combined figure and closes it.

    Notes
    -----
    - A fixed technology order and color scheme is used for consistency.
    - All subplots share the same y-axis for comparability.
    - Missing technologies are handled automatically.
    """

    import matplotlib.pyplot as plt
    import numpy as np
    import math

    # =====================================================
    # LAYOUT
    # =====================================================
    n = len(countries)
    ncols = 3
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5 * ncols, 4 * nrows),
        sharey=True
    )

    axes = axes.flatten()

    # =====================================================
    # SCENARIO ORDER
    # =====================================================
    scenario_order = ["no_inertia", "thermal_only", "thermal_plus_virtual"]

    # =====================================================
    # TECHNOLOGY ORDER
    # =====================================================
    tech_order = [
        "Kernenergie",
        "Braunkohle",
        "Kohle",
        "Gas",
        "Schweröl",
        "Leichtöl",
        "Ölschiefer",
        "Wasserstoff",
        "Biomasse",
        "Müllverbrennung",
        "Hydro",
        "Battery",
        "Sonstige RES",
        "Sonstige nicht-RES"
    ]

    # =====================================================
    # LOOP OVER COUNTRIES
    # =====================================================
    for i, country in enumerate(countries):

        ax = axes[i]

        df = df_mix[df_mix["country"] == country].copy()
        df = df.set_index("scenario").reindex(scenario_order)

        tech_cols = [t for t in tech_order if t in df.columns]

        # -------------------------------------------------
        # 🔥 NORMALIZATION (CRUCIAL)
        # -------------------------------------------------
        df_norm = df[tech_cols].div(df[tech_cols].sum(axis=1), axis=0) * 100

        x = np.arange(len(df_norm))
        bottom = np.zeros(len(df_norm))

        # -------------------------------------------------
        # STACKED BARS
        # -------------------------------------------------
        for tech in tech_cols:

            values = df_norm[tech].fillna(0).values
            color = COLOR_MAP.get(tech, "#cccccc")

            ax.bar(
                x,
                values,
                bottom=bottom,
                color=color,
                linewidth=0
            )

            bottom += values

        # -------------------------------------------------
        # FORMATTING (DEUTSCH)
        # -------------------------------------------------
        ax.set_title(country)

        ax.set_xticks(x)
        ax.set_xticklabels(
            [scenario_labels.get(s, s) for s in df_norm.index],
            rotation=20
        )

        ax.set_ylim(0, 100)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    # =====================================================
    # REMOVE UNUSED SUBPLOTS
    # =====================================================
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # =====================================================
    # Y-AXIS LABEL
    # =====================================================
    axes[0].set_ylabel("Erzeugungsanteil (%)")

    # =====================================================
    # GLOBAL LEGEND (clean)
    # =====================================================
    handles = []
    labels = []

    for tech in tech_order:
        if tech in df_mix.columns:
            handles.append(
                plt.Rectangle((0, 0), 1, 1,
                              color=COLOR_MAP.get(tech, "#ccc"))
            )
            labels.append(tech)

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=6,
        title="Technologien",
        frameon=False
    )

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    # =====================================================
    # SAVE
    # =====================================================
    save_figure(
        fig,
        "generation_mix_all_countries",
        subfolder="generation_mix"
    )

    plt.close()
# =========================================================
# GENERATION MIX PER COUNTRY AND SCENARIO (FINAL)
# =========================================================

def compute_generation_mix_per_country_scenarios(solutions, inputs, countries):
    """
    Compute generation mix per country and scenario (absolute values).

    This function aggregates generation by technology for each country and
    scenario using unit-level dispatch data. Technologies are identified via
    a predefined unit-to-fuel mapping.

    Parameters
    ----------
    solutions : dict
        Dictionary mapping scenario names (str) to DataFrames containing
        model results.

    inputs : object
        Input container with:
        - thermal_units
        - other_res_units
        - non_res_units
        - df_hydro (optional hydro generation data)

    countries : list of str
        List of country codes.

    Returns
    -------
    pandas.DataFrame
        DataFrame with:
        - "country"
        - "scenario"
        - technology columns (absolute generation values, MWh)

    Notes
    -----
    - Generation is aggregated from dispatch variable "p".
    - Non-RES generation ("p_non_res") is added separately.
    - Hydro and battery discharge are currently computed but NOT included
      in the final output (see remarks below).
    - Missing values are filled with zero.
    """

    import pandas as pd

    results = []

    # =====================================================
    # 1) BUILD UNIT → FUEL MAPPING
    # =====================================================
    unit_fuel = {}

    for u in inputs.thermal_units:
        unit_fuel[str(u.uid)] = FUEL_MAP.get(u.fuel_type, "Sonstige")

    for u in inputs.other_res_units:
        unit_fuel[str(u.uid)] = FUEL_MAP.get(u.fueltype, "Sonstige")

    for u in inputs.non_res_units:
        unit_fuel[str(u.uid)] = "Sonstige nicht-RES"

    # =====================================================
    # 2) LOOP OVER SCENARIOS
    # =====================================================
    for scen, df in solutions.items():

        # -------------------------------------------------
        # PRECOMPUTE HYDRO (currently unused!)
        # -------------------------------------------------
        df_hydro = inputs.df_hydro
        if isinstance(df_hydro, dict):
            hydro_total = {
                c: sum(
                    df_h[c].sum()
                    for df_h in df_hydro.values()
                    if c in df_h.columns
                )
                for c in countries
            }
        else:
            hydro_total = df_hydro[countries].sum().to_dict()

        # -------------------------------------------------
        # BATTERY DISCHARGE (currently unused!)
        # -------------------------------------------------
        discharge = df[df["type"] == "discharge"] \
            .groupby("country")["value"].sum()

        # -------------------------------------------------
        # NON-RES GENERATION
        # -------------------------------------------------
        non_res = df[df["type"] == "p_non_res"] \
            .groupby("country")["value"].sum()

        # =================================================
        # LOOP OVER COUNTRIES
        # =================================================
        for c in countries:

            tech_values = {}

            # ---------------------------------------------
            # THERMAL + OTHER RES (dispatch variable "p")
            # ---------------------------------------------
            subset = df[
                (df["type"] == "p") &
                (df["country"] == c)
            ]

            grouped = subset.groupby("unit")["value"].sum()

            for unit, val in grouped.items():
                tech = unit_fuel.get(unit, "Sonstige")
                tech_values[tech] = tech_values.get(tech, 0) + val

            # ---------------------------------------------
            # ADD NON-RES
            # ---------------------------------------------
            tech_values["Sonstige nicht-RES"] = (
                tech_values.get("Sonstige nicht-RES", 0)
                + non_res.get(c, 0)
            )

            # ---------------------------------------------
            # METADATA
            # ---------------------------------------------
            tech_values["country"] = c
            tech_values["scenario"] = scen

            results.append(tech_values)

    return pd.DataFrame(results).fillna(0)
def clean_unit(unit):
    """
    Remove trailing time index from unit identifier.

    Example
    -------
    CCGT_old_2_AL_2040.0_1_1 → CCGT_old_2_AL_2040.0_1

    Parameters
    ----------
    unit : str
        Full unit identifier string (including time index).

    Returns
    -------
    str
        Cleaned unit identifier without the final time component.

    Notes
    -----
    - Assumes the last underscore-separated element represents time.
    - Required for consistent matching with unit IDs.
    """

    parts = unit.split("_")

    # Remove only the last element (time index)
    return "_".join(parts[:-1])


# =====================================================
# MATCH UNIT TO FUEL (ROBUST PREFIX MATCHING)
# =====================================================

def match_unit_to_fuel(unit, unit_fuel):
    """
    Map a unit string to a fuel/technology category using prefix matching.

    Parameters
    ----------
    unit : str
        Unit identifier from model output.

    unit_fuel : dict
        Mapping from unit UID → fuel/technology.

    Returns
    -------
    str
        Fuel/technology category.

    Notes
    -----
    - Matching is based on string prefix.
    - If no match is found, "Sonstige" is returned.
    - Sensitive to naming consistency.
    """

    for uid in unit_fuel:
        if unit.startswith(uid):
            return unit_fuel[uid]

    return "Sonstige"


# =====================================================
# EXTRACT OBJECTIVE VALUE FROM .sol FILE
# =====================================================

def extract_objective_from_sol(path):
    """
    Extract objective value from a solver output (.sol) file.

    Parameters
    ----------
    path : str
        Path to the .sol file.

    Returns
    -------
    float or None
        Objective value if found, otherwise None.

    Notes
    -----
    - Assumes the objective value is located in the first line.
    - Format expected: "Objective value = ..."
    """

    with open(path, "r") as f:
        first_line = f.readline()

    if "Objective value" in first_line:
        return float(first_line.split("=")[1].strip())

    return None


# =====================================================
# SCENARIO COMPARISON SUMMARY
# =====================================================

def compare_scenarios(solutions, sol_paths, inputs, countries):
    """
    Compute key performance indicators (KPIs) for scenario comparison.

    This function aggregates system-level metrics including generation,
    curtailment, inertia indicators, and total system cost.

    Parameters
    ----------
    solutions : dict
        Dictionary mapping scenario names to result DataFrames.

    sol_paths : dict
        Dictionary mapping scenario names to .sol file paths.

    inputs : object
        Input container used for extracting H_sys.

    countries : list of str
        List of country codes.

    Returns
    -------
    pandas.DataFrame
        Summary table with one row per scenario.

    Metrics
    -------
    - generation : total generation (MWh)
    - curtailment : curtailed energy (MWh)
    - avg_Hsys : average system inertia
    - min_Hsys : minimum system inertia
    - critical_hours : number of hours below threshold (H_sys < 2)
    - critical_share_% : share of critical hours
    - inertia_slack : unmet inertia requirement
    - total_cost : objective function value

    Notes
    -----
    - H_sys is aggregated across all countries and hours.
    - Critical threshold is fixed at H_sys < 2.
    """

    import pandas as pd

    rows = []

    for name, df in solutions.items():

        # =================================================
        # BASIC METRICS
        # =================================================
        generation = df[df["type"] == "p"]["value"].sum()
        commitments = df[df["type"] == "y"]["value"].sum()  # currently unused
        curtailment = df[df["type"] == "curtailement"]["value"].sum()

        # =================================================
        # INERTIA (H_sys)
        # =================================================
        H_df = extract_H_sys_per_country(df, inputs, countries)

        if H_df.empty:
            print(f"⚠️ No H_sys data for {name}")
            continue

        avg_Hsys = H_df["H_sys"].mean()
        min_Hsys = H_df["H_sys"].min()

        # Critical hours (threshold = 2)
        critical_mask = H_df["H_sys"] < 2
        critical_hours = critical_mask.sum()
        critical_share = critical_mask.mean() * 100

        # =================================================
        # INERTIA SLACK
        # =================================================
        inertia_slack = df[
            df["var"].str.startswith("slack_inertia", na=False)
        ]["value"].sum()

        # =================================================
        # OBJECTIVE VALUE
        # =================================================
        obj = extract_objective_from_sol(sol_paths[name])

        # =================================================
        # COLLECT RESULTS
        # =================================================
        rows.append({
            "scenario": name,
            "generation": generation,
            "curtailment": curtailment,
            "avg_Hsys": avg_Hsys,
            "min_Hsys": min_Hsys,
            "critical_hours": critical_hours,
            "critical_share_%": critical_share,
            "inertia_slack": inertia_slack,
            "total_cost": obj
        })

    return pd.DataFrame(rows)
def inertia_split(df):
    """
    Compute total, virtual, and non-virtual inertia.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing inertia-related variables with:
        - "var"
        - "value"

    Returns
    -------
    tuple
        (total_inertia, virtual_inertia, non_virtual_inertia)

    Notes
    -----
    - Requires a column "type_full". If not present, it is created.
    - Classification is based on string patterns in variable names.
    """

    if "type_full" not in df.columns:
        df = enrich_types(df)

    total = df[df["type_full"] == "inertia_total"]["value"].sum()
    virtual = df[df["type_full"] == "inertia_virtual"]["value"].sum()
    non_virtual = df[df["type_full"] == "inertia_non_virtual"]["value"].sum()

    return total, virtual, non_virtual


# =====================================================
# TYPE ENRICHMENT
# =====================================================

def enrich_types(df):
    """
    Extract structured type information from variable names.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with column "var".

    Returns
    -------
    pandas.DataFrame
        Updated DataFrame with:
        - "type_full" : full type string
        - "type" : base type (prefix)

    Notes
    -----
    - Uses regex to extract prefix before first numeric/index part.
    - Assumes naming convention: type_subtype_...
    """

    df["type_full"] = df["var"].str.extract(r"^([a-zA-Z_]+)")
    df["type"] = df["type_full"].str.split("_").str[0]

    return df


# =====================================================
# PREPARE DISPATCH DATA
# =====================================================

def prepare_dispatch(df):
    """
    Extract dispatch data and assign fuel types.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing model results.

    Returns
    -------
    pandas.DataFrame
        Filtered dispatch DataFrame with additional column:
        - "fuel"
    """

    dispatch = df[df["type"] == "p"].copy()

    # Assign fuel using external mapping function
    dispatch["fuel"] = dispatch["unit"].apply(extract_fuel)

    return dispatch


# =====================================================
# ADD HOUR OF DAY
# =====================================================

def add_hour_of_day(df):
    """
    Add hour-of-day column (1–24) based on hourly index.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing column "hour".

    Returns
    -------
    pandas.DataFrame
        Updated DataFrame with column "hour_of_day".
    """

    df["hour"] = df["hour"].astype(int)
    df["hour_of_day"] = ((df["hour"] - 1) % 24) + 1

    return df
# =====================================================
# DISPATCH DAILY PROFILE (SCENARIOS)
# =====================================================

def compute_dispatch_daily_profile_scenarios(solutions, inputs):
    """
    Compute average daily dispatch profiles per technology and scenario.

    This function aggregates dispatch data into a typical daily profile
    (hour-of-day representation) for each scenario and technology.

    Parameters
    ----------
    solutions : dict
        Dictionary mapping scenario names to DataFrames.

    inputs : object
        Input container with unit definitions:
        - thermal_units
        - other_res_units
        - non_res_units

    Returns
    -------
    pandas.DataFrame
        DataFrame with:
        - "hour_of_day"
        - "fuel"
        - "value" (average dispatch)
        - "scenario"

    Notes
    -----
    - Dispatch is based on variable type "p".
    - Values are averaged over all days (not summed).
    - Battery discharge is added as a separate technology.
    """

    import pandas as pd

    results = []

    # =====================================================
    # UNIT → FUEL MAPPING
    # =====================================================
    unit_fuel = {}

    for u in inputs.thermal_units:
        unit_fuel[str(u.uid)] = FUEL_MAP.get(u.fuel_type, "Sonstige RES")

    for u in inputs.other_res_units:
        unit_fuel[str(u.uid)] = FUEL_MAP.get(u.fueltype, "Sonstige RES")

    for u in inputs.non_res_units:
        unit_fuel[str(u.uid)] = "Sonstige nicht-RES"

    # =====================================================
    # LOOP OVER SCENARIOS
    # =====================================================
    for scen, df in solutions.items():

        df = df.copy()

        # Remove invalid time entries
        df = df[df["hour"].notna()]
        df["hour"] = df["hour"].astype(int)

        # Convert to hour-of-day (1–24)
        df["hour_of_day"] = ((df["hour"] - 1) % 24) + 1

        # -------------------------------------------------
        # DISPATCH (p)
        # -------------------------------------------------
        dispatch = df[df["type"] == "p"].copy()

        dispatch["fuel"] = dispatch["unit"].map(unit_fuel)
        dispatch["fuel"] = dispatch["fuel"].fillna("Sonstige RES")

        grouped = (
            dispatch.groupby(["hour_of_day", "fuel"])["value"]
            .mean()
            .reset_index()
        )

        # -------------------------------------------------
        # BATTERY DISCHARGE
        # -------------------------------------------------
        battery = df[df["type"] == "discharge"]

        if not battery.empty:
            battery_grouped = (
                battery.groupby("hour_of_day")["value"]
                .mean()
                .reset_index()
            )
            battery_grouped["fuel"] = "Battery"

            grouped = pd.concat(
                [grouped, battery_grouped],
                ignore_index=True
            )

        # -------------------------------------------------
        # METADATA
        # -------------------------------------------------
        grouped["scenario"] = scen

        results.append(grouped)

    return pd.concat(results, ignore_index=True)
def plot_dispatch_by_scenario(df, inputs, scenario):
    """
    Plot hourly dispatch time series by technology for a given scenario.

    This function aggregates dispatch (generation) across all units and
    visualizes the temporal evolution of generation by fuel type using a
    stacked area chart.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing model results with:
        - "type" : str (must include "p" for dispatch)
        - "unit" : str
        - "value" : float
        - "hour" : int or float

    inputs : object
        Input container with:
        - thermal_units
        - other_res_units
        - non_res_units

    scenario : str
        Scenario identifier used for labeling and file naming.

    Returns
    -------
    None
        The function saves the plot and closes the figure.

    Notes
    -----
    - Dispatch is aggregated across all countries.
    - Values represent instantaneous power (MW), not energy.
    - Technology mapping is based on unit identifiers.
    """

    import pandas as pd
    import matplotlib.pyplot as plt

    df = df.copy()

    # -----------------------------------------------------
    # FILTER DISPATCH
    # -----------------------------------------------------
    df = df[df["type"] == "p"]

    if "hour" not in df.columns:
        print(f"⚠️ {scenario}: missing 'hour'")
        return

    df = df[df["hour"].notna()].copy()

    if df.empty:
        print(f"⚠️ {scenario}: empty after filtering")
        return

    df["hour"] = df["hour"].astype(int)

    # -----------------------------------------------------
    # UNIT → FUEL MAPPING
    # -----------------------------------------------------
    unit_fuel = {}

    for u in inputs.thermal_units:
        unit_fuel[str(u.uid)] = FUEL_MAP.get(u.fuel_type, "Sonstige")

    for u in inputs.other_res_units:
        unit_fuel[str(u.uid)] = FUEL_MAP.get(u.fueltype, "Sonstige")

    for u in inputs.non_res_units:
        unit_fuel[str(u.uid)] = "Sonstige nicht-RES"

    df["fuel"] = df["unit"].apply(
        lambda u: match_unit_to_fuel(u, unit_fuel)
    )

    # -----------------------------------------------------
    # AGGREGATION
    # -----------------------------------------------------
    grouped = (
        df.groupby(["hour", "fuel"])["value"]
        .sum()
        .reset_index()
    )

    pivot = grouped.pivot(
        index="hour",
        columns="fuel",
        values="value"
    ).fillna(0)

    # -----------------------------------------------------
    # 🔥 BESSERE REIHENFOLGE (Interpretierbarkeit)
    # -----------------------------------------------------
    order = [
        # stabile Grundlast unten
        "Kernenergie",
        "Braunkohle",
        "Kohle",

        # flexibel
        "Gas",

        # mittel
        "Biomasse", "Müllverbrennung",

        # neue Technologien
        "Wasserstoff",
        "Sonstige nicht-RES",

        # volatile ganz oben
        "Sonstige RES",
    ]

    fuels = [f for f in order if f in pivot.columns]
    pivot = pivot[fuels]

    colors = [COLOR_MAP.get(f, "#cccccc") for f in fuels]

    # -----------------------------------------------------
    # 🔥 OPTIONAL: SMOOTHING (für Lesbarkeit)
    # -----------------------------------------------------
    # pivot = pivot.rolling(2, min_periods=1).mean()

    # -----------------------------------------------------
    # PLOT
    # -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(16, 4))

    pivot.plot(
        kind="area",
        stacked=True,
        ax=ax,
        color=colors,
        linewidth=0
    )

    # -----------------------------------------------------
    # DEUTSCHE BESCHRIFTUNG
    # -----------------------------------------------------
    ax.set_title(f"Erzeugungsfahrplan – {scenario_labels.get(scenario, scenario)}")

    ax.set_ylabel("Leistung (MW)")
    ax.set_xlabel("Stunde")

    # bessere x-Achse
    ax.set_xlim(pivot.index.min(), pivot.index.max())
    ax.set_xticks(range(0, int(pivot.index.max()) + 1, 24))

    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # -----------------------------------------------------
    # SAVE
    # -----------------------------------------------------
    save_figure(
        fig,
        scenario + "_dispatch_time_series",
        subfolder="dispatch"
    )

    plt.close()
# =========================================================
# DISPATCH DAILY AVERAGE (ALL SCENARIOS)
# =========================================================

def plot_dispatch_daily_avg_all(solutions, inputs):
    """
    Plot average daily dispatch profiles for multiple scenarios.

    This function computes a representative daily profile (hour-of-day)
    of generation by technology and visualizes it using stacked area plots
    in vertically aligned subplots.

    Parameters
    ----------
    solutions : dict
        Dictionary mapping scenario names to result DataFrames.

    inputs : object
        Input container with:
        - thermal_units
        - other_res_units
        - non_res_units

    Returns
    -------
    None
        Saves the figure to disk.

    Notes
    -----
    - Dispatch is based on variable type "p".
    - Values represent average power (MW), not energy.
    - Small technologies (<2% of total generation) are aggregated.
    - All scenarios use identical colors and ordering.
    """

    import pandas as pd
    import matplotlib.pyplot as plt

  

    scenario_order = ["no_inertia", "thermal_only", "thermal_plus_virtual"]

   

    # -----------------------------------------
    # Fuel Mapping
    # -----------------------------------------
    unit_fuel = {}

    for u in inputs.thermal_units:
        unit_fuel[str(u.uid)] = FUEL_MAP.get(u.fuel_type, "Sonstige")

    for u in inputs.other_res_units:
        unit_fuel[str(u.uid)] = FUEL_MAP.get(u.fueltype, "Sonstige")

    for u in inputs.non_res_units:
        unit_fuel[str(u.uid)] = "Sonstige nicht-RES"

    # -----------------------------------------
    # Subplots (VERTIKAL → wichtig!)
    # -----------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    all_fuels = set()
    pivots = {}

    # -----------------------------------------
    # Daten vorbereiten
    # -----------------------------------------
    for scen in scenario_order:

        if scen not in solutions:
            continue

        df = solutions[scen].copy()
        df = df[df["type"] == "p"]

        if "hour" not in df.columns:
            continue

        df = df[df["hour"].notna()]
        df["hour"] = df["hour"].astype(int)
        df["hour_of_day"] = ((df["hour"] - 1) % 24)

        df["fuel"] = df["unit"].apply(lambda u: match_unit_to_fuel(u, unit_fuel))

        grouped = df.groupby(["hour_of_day", "fuel"])["value"].mean().reset_index()

        pivot = grouped.pivot(
            index="hour_of_day",
            columns="fuel",
            values="value"
        ).fillna(0)

        pivot = pivot.reindex(range(24), fill_value=0)

        # -----------------------------------------
        # Kleine Technologien zusammenfassen 🔥
        # -----------------------------------------
        total = pivot.sum().sum()
        threshold = 0.02 * total  # 2 %

        small = pivot.sum() < threshold
        if small.any():
            pivot["Sonstige"] = pivot.loc[:, small].sum(axis=1)
            pivot = pivot.loc[:, ~small]

        pivots[scen] = pivot
        all_fuels.update(pivot.columns)

    # -----------------------------------------
    # Einheitliche Reihenfolge
    # -----------------------------------------
    order = [
        "Braunkohle", "Kohle", "Gas",
        "Kernenergie",
        "Biomasse", "Müllverbrennung",
        "Wasserstoff",
        "Sonstige RES",
        "Sonstige nicht-RES",
        "Sonstige"
    ]

    fuels = [f for f in order if f in all_fuels]
    colors = [COLOR_MAP.get(f, "#cccccc") for f in fuels]

    # -----------------------------------------
    # Plotten
    # -----------------------------------------
    for i, scen in enumerate(scenario_order):

        if scen not in pivots:
            continue

        pivot = pivots[scen].reindex(columns=fuels, fill_value=0)
        pivot.plot(
            kind="area",
            stacked=True,
            ax=axes[i],
            color=colors,
            linewidth=0,
            alpha=0.9,
            legend=False
        )

        axes[i].set_title(scenario_labels.get(scen, scen))
        axes[i].set_xlim(0, 23)
        axes[i].set_xticks([0, 6, 12, 18, 23])
        axes[i].grid(True, alpha=0.2)

    axes[0].set_ylabel("Leistung [MW]")
    axes[1].set_ylabel("Leistung [MW]")
    axes[2].set_ylabel("Leistung [MW]")
    axes[2].set_xlabel("Stunde")

    # -----------------------------------------
    # Legende unten (viel besser!)
    # -----------------------------------------
    handles = [
        plt.Line2D([0], [0], color=c, lw=6)
        for c in colors
    ]

    fig.legend(
        handles,
        fuels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=4,
        frameon=False
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    save_figure(fig, "dispatch_daily_avg_all", subfolder="dispatch")

    plt.close()
def compute_battery_profile_scenarios(solutions):
    """
    Compute average daily battery charging and discharging profiles
    for multiple scenarios.

    This function aggregates battery operation into a representative
    daily cycle (hour-of-day) and distinguishes between charging and
    discharging flows.

    Parameters
    ----------
    solutions : dict
        Dictionary mapping scenario names to DataFrames.

    Returns
    -------
    pandas.DataFrame
        DataFrame with:
        - "hour_of_day" : int (1–24)
        - "value" : float (average power, MW)
        - "flow" : str ("Charge" or "Discharge")
        - "scenario" : str

    Notes
    -----
    - Charging values are stored as negative values for visualization.
    - Values represent average power, not energy.
    - Aggregation is performed via mean over all days.
    """

    import pandas as pd

    results = []

    for scen, df in solutions.items():

        df = df.copy()

        # =================================================
        # CLEAN TIME INDEX
        # =================================================
        df["hour"] = pd.to_numeric(df["hour"], errors="coerce")
        df = df.dropna(subset=["hour"])
        df["hour"] = df["hour"].astype(int)

        df["hour_of_day"] = ((df["hour"] - 1) % 24) + 1

        # =================================================
        # DISCHARGE (positive values)
        # =================================================
        discharge = df[df["type"] == "discharge"].copy()

        discharge = (
            discharge.groupby(["hour"])["value"]
            .sum()
            .reset_index()
        )

        # Stunde des Tages neu berechnen
        discharge["hour_of_day"] = ((discharge["hour"] - 1) % 24) + 1

        # dann erst mitteln
        discharge = (
            discharge.groupby("hour_of_day")["value"]
            .mean()
            .reset_index()
        )

        discharge["flow"] = "Discharge"

        # =================================================
        # CHARGE (stored as negative values)
        # =================================================
        charge = df[df["type"] == "charge"].copy()

        charge = (
            charge.groupby(["hour"])["value"]
            .sum()
            .reset_index()
        )

        charge["hour_of_day"] = ((charge["hour"] - 1) % 24) + 1

        charge = (
            charge.groupby("hour_of_day")["value"]
            .mean()
            .reset_index()
        )

        charge["value"] = -charge["value"]
        charge["flow"] = "Charge"

        # =================================================
        # COMBINE
        # =================================================
        battery = pd.concat([discharge, charge], ignore_index=True)
        battery["scenario"] = scen

        results.append(battery)

    return pd.concat(results, ignore_index=True)

def plot_battery_net_per_country(solutions, countries):
    """
    Plots the average daily net battery profile for selected countries.

    For each country, one figure is created showing all scenarios.
    The net battery profile is defined as:

        Net = Discharge - Charge

    Interpretation:
    - Positive values → battery discharges (injects power into grid)
    - Negative values → battery charges (absorbs power)

    Parameters
    ----------
    solutions : dict
        Dictionary mapping scenario names to DataFrames.
        Each DataFrame must contain:
        - "var", "value", "hour", "country"

    countries : list
        List of country codes (e.g. ["DE", "FR", "PL"])

    Notes
    -----
    - Battery variables are identified via substring "batt" in `var`.
    - Since the model does not explicitly separate charge/discharge,
      sign convention is inferred from variable names.
    - Profiles represent average power (MW), not energy.
    """

    import matplotlib.pyplot as plt
    import pandas as pd

    if isinstance(countries, str):
        countries = [countries]

    SCENARIO_COLORS = {
        "no_inertia": "#4d4d4d",
        "thermal_only": "#1f77b4",
        "thermal_plus_virtual": "#2ca02c"
    }

    for country in countries:

        fig, ax = plt.subplots(figsize=(7, 4))

        for scenario, df in solutions.items():

            df = df.copy()

            # -----------------------------------------
            # 🔹 Zeit
            # -----------------------------------------
            df["hour"] = pd.to_numeric(df["hour"], errors="coerce")
            df = df.dropna(subset=["hour"])
            df["hour"] = df["hour"].astype(int)

            df["hour_of_day"] = ((df["hour"] - 1) % 24) + 1

            # -----------------------------------------
            # 🔹 Batterie filtern (JETZT KORREKT!)
            # -----------------------------------------
            batt = df[
                (df["country"] == country) &
                (df["type"].isin(["charge", "discharge"]))
            ].copy()

            if batt.empty:
                print(f"{country} - {scenario}: keine Batterie-Daten")
                continue

            # -----------------------------------------
            # 🔹 Vorzeichen
            # -----------------------------------------
            batt["signed"] = batt["value"]

            batt.loc[batt["type"] == "charge", "signed"] *= -1
            # discharge bleibt positiv

            # -----------------------------------------
            # 🔹 Aggregation (WICHTIG!)
            # -----------------------------------------
            hourly = (
                batt.groupby("hour")["signed"]
                .sum()   # ✅ jetzt korrekt!
                .reset_index()
            )

            hourly["hour_of_day"] = ((hourly["hour"] - 1) % 24) + 1

            hourly = (
                hourly.groupby("hour_of_day")["signed"]
                .mean()
                .reindex(range(1, 25), fill_value=0)
            )

            # -----------------------------------------
            # 🔹 Plot
            # -----------------------------------------
            ax.plot(
                hourly.index,
                hourly.values,
                label=scenario_labels.get(scenario, scenario),
                color=SCENARIO_COLORS.get(scenario, "#999999"),
                linewidth=2
            )

        # -----------------------------------------
        # 🔹 Layout
        # -----------------------------------------
        ax.axhline(0, color="black", linewidth=1)

        ax.set_title(f"Batterie-Netto-Profil ({country})")
        ax.set_xlabel("Stunde des Tages")
        ax.set_ylabel("Nettoleistung (MW)")

        ax.set_xticks(range(1, 25, 3))
        ax.grid(True, alpha=0.3)

        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            frameon=False,
            title="Szenario"
        )

        plt.tight_layout(rect=[0, 0, 0.85, 1])
        save_figure(fig, f"battery_net_{country}", subfolder="battery")

        plt.close()
def plot_battery_by_scenario(solutions):
    """
    Plot battery charging and discharging profiles per scenario.

    This function visualizes the average daily battery operation using
    bar charts, with charging shown as negative values and discharging
    as positive values.

    Parameters
    ----------
    solutions : dict
        Dictionary mapping scenario names to DataFrames.

    Returns
    -------
    None
        The function saves the figure and closes it.

    Notes
    -----
    - Charging is displayed as negative values to distinguish direction.
    - All scenarios share a common y-axis for comparability.
    """
    import matplotlib.pyplot as plt

    df_batt = compute_battery_profile_scenarios(solutions)

    # =====================================================
    # SCENARIO ORDER
    # =====================================================
    scenario_order = [
        "no_inertia",
        "thermal_only",
        "thermal_plus_virtual"
    ]

    scenarios = [
        s for s in scenario_order
        if s in df_batt["scenario"].unique()
    ]

    # =====================================================
    # SUBPLOTS
    # =====================================================
    fig, axes = plt.subplots(
        1, len(scenarios),
        figsize=(20, 8),
        sharey=True
    )

    if len(scenarios) == 1:
        axes = [axes]

    # =====================================================
    # COLORS
    # =====================================================
    color_discharge = COLOR_MAP.get("Battery", "#ff7f0e")
    color_charge = "#999999"
    color_net = "#000000"

    # =====================================================
    # PLOTTING
    # =====================================================
    for i, scen in enumerate(scenarios):

        subset = df_batt[df_batt["scenario"] == scen].copy()

        pivot = subset.pivot(
            index="hour_of_day",
            columns="flow",
            values="value"
        ).fillna(0)

        # Ensure columns exist
        discharge = pivot.get("Discharge", 0)
        charge = pivot.get("Charge", 0)

        net = discharge + charge  # charge already negative

        ax = axes[i]

        # -------------------------------------------------
        # AREA (flows)
        # -------------------------------------------------
        ax.fill_between(
            pivot.index,
            0,
            discharge,
            color=color_discharge,
            alpha=0.7,
            label="Entladen"
        )

        ax.fill_between(
            pivot.index,
            0,
            charge,
            color=color_charge,
            alpha=0.7,
            label="Laden"
        )

        # -------------------------------------------------
        # NET LINE (🔥 sehr wichtig)
        # -------------------------------------------------
        ax.plot(
            pivot.index,
            net,
            color=color_net,
            linewidth=2,
            label="Netto"
        )

        # -------------------------------------------------
        # FORMATTING
        ax.set_title(
           scenario_labels.get(scen, scen)
        )

        ax.set_xlabel("Stunde")
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="black", linewidth=1)

        ax.set_xticks([0, 6, 12, 18, 23])

    axes[0].set_ylabel("Leistung (MW)")

    # =====================================================
    # LEGEND
    # =====================================================
    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        title="Batteriebetrieb"
    )

    plt.tight_layout(rect=[0, 0, 0.95, 1])

    save_figure(fig, "battery_dispatch_all", subfolder="battery")

    plt.close()


def compute_dsr_utilization(solutions, inputs, countries):
    """
    Compute Demand-Side Response (DSR) utilization per country and scenario.

    This function compares the total utilized DSR from the optimization model
    with the available DSR potential derived from input schedules.

    Parameters
    ----------
    solutions : dict
        Dictionary mapping scenario names to result DataFrames.

    inputs : object
        Input container with attribute:
        - dsr_units : iterable of DSR units with:
            - country
            - schedule (time series of available flexibility)

    countries : list of str
        List of country codes.

    Returns
    -------
    pandas.DataFrame
        DataFrame with:
        - "scenario"
        - "country"
        - "dsr_used" (MWh or MW depending on model definition)
        - "dsr_potential" (sum of available flexibility)
        - "dsr_utilization_%" (percentage)

    Notes
    -----
    - Potential is computed as the sum over the entire time horizon.
    - Utilization is aggregated over all DSR units per country.
    - Assumes consistency between model output units and input schedules.
    """

    import pandas as pd

    results = []

    # =====================================================
    # 1) DSR POTENTIAL (INPUT-BASED)
    # =====================================================
    dsr_potential = {c: 0 for c in countries}

    for d in inputs.dsr_units:
        c = d.country

        # Total available flexibility over time horizon
        total_potential = d.schedule.sum()

        dsr_potential[c] += total_potential

    # =====================================================
    # 2) DSR UTILIZATION (MODEL OUTPUT)
    # =====================================================
    for scen, df in solutions.items():

        dsr_used = (
            df[df["type"] == "dsr"]
            .groupby("country")["value"]
            .sum()
        )

        for c in countries:

            used = dsr_used.get(c, 0)
            potential = dsr_potential.get(c, 0)

            utilization = (
                used / potential * 100
                if potential > 0 else 0
            )

            results.append({
                "scenario": scen,
                "country": c,
                "dsr_used": used,
                "dsr_potential": potential,
                "dsr_utilization_%": utilization
            })

    return pd.DataFrame(results)
def plot_dsr_utilization_per_scenario(df_util):
    """
    Plot DSR utilization per country for each scenario.

    This function visualizes how much of the available DSR potential
    is actually used in each scenario.

    Parameters
    ----------
    df_util : pandas.DataFrame
        Output of compute_dsr_utilization().

    Returns
    -------
    None
        The function saves one plot per scenario.

    Notes
    -----
    - Countries with zero DSR potential are excluded.
    - Values are shown as percentage of total potential.
    """
    import matplotlib.pyplot as plt

    scenarios = df_util["scenario"].unique()

    for scen in scenarios:

        subset = df_util[df_util["scenario"] == scen].copy()

        # -------------------------------------------------
        # FILTER + SORT
        # -------------------------------------------------
        subset = subset[subset["dsr_potential"] > 0]

        if subset.empty:
            print(f"⚠️ Kein DSR-Potenzial für {scen}")
            continue

        subset = subset.sort_values(
            "dsr_utilization_%",
            ascending=False
        )

        # -------------------------------------------------
        # COLORS (highlight high usage)
        # -------------------------------------------------
        values = subset["dsr_utilization_%"].values

        colors = [
            "#d62728" if v > 50 else "#4C72B0"
            for v in values
        ]

        # -------------------------------------------------
        # PLOT
        # -------------------------------------------------
        fig, ax = plt.subplots(figsize=(14, 6))

        bars = ax.bar(
            subset["country"],
            values,
            color=colors
        )

        # -------------------------------------------------
        # LABELS (DEUTSCH)
        # -------------------------------------------------
        ax.set_ylabel("DSR-Nutzung (% des Potenzials)")
        ax.set_xlabel("Land")

        ax.set_title(
            f"DSR-Nutzung nach Land\n"
            f"({scenario_labels.get(scen, scen)})"
        )

        ax.set_ylim(0, 100)

        # -------------------------------------------------
        # VALUE LABELS
        # -------------------------------------------------
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width()/2,
                    height + 1,
                    f"{height:.1f}%",
                    ha="center",
                    fontsize=9
                )

        # -------------------------------------------------
        # GRID + AXIS
        # -------------------------------------------------
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        plt.xticks(rotation=45)

        plt.tight_layout()

        # -------------------------------------------------
        # SAVE
        # -------------------------------------------------
        save_figure(
            fig,
            f"dsr_utilization_{scen}",
            subfolder="flexibility"
        )

        plt.close()
    