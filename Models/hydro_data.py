# =========================================================
# Author: Leonardo Gabriel Mourao Thiel
# Project: Master Thesis – System Inertia in the Energy System of the Future: Model-Based Cost Optimization to Secure Inertia Requirements
# Topic: Parameterization of Hydro 
# Date: 27.04.2026
# =========================================================
import os
import pandas as pd
import numpy as np

# Hydro technology categories (as defined in PEMMDB datasets)
categories = [
    "Run of River - MW",
    "Reservoir - MW",
    "PS Open (turbine) - MW",
    "PS Closed (turbine) - MW",
    "Pondage - MW"
]

# ---------------------------------------------------------
# 1) LOAD AND FILTER HYDRO GENERATION PROFILES
# ---------------------------------------------------------

def load_hydro_profiles(pfad):
    """
    Loads hydro generation time series for different technology types.

    Parameters
    ----------
    pfad : str
        Base directory path containing hydro generation Excel files.

    Returns
    -------
    dict of pandas.DataFrame
        Dictionary mapping hydro technology types to their corresponding
        time series DataFrames. Each DataFrame contains:
        - 'Timestamp'
        - Country columns with generation values (MW)

    Notes
    -----
    - Each file corresponds to a specific hydro technology.
    - No filtering is applied at this stage.
    """

    # Load generation profiles for each hydro technology
    df_run_of_river = pd.read_excel(f"{pfad}/2040/Generation_2040/Run_of_River.xlsx")
    df_ps_open      = pd.read_excel(f"{pfad}/2040/Generation_2040/PS_Open.xlsx")
    df_ps_closed    = pd.read_excel(f"{pfad}/2040/Generation_2040/PS_Closed.xlsx")
    df_reservoir    = pd.read_excel(f"{pfad}/2040/Generation_2040/Reservoir.xlsx")
    df_pondage      = pd.read_excel(f"{pfad}/2040/Generation_2040/Pondage.xlsx")

    # Reset indices to ensure consistent indexing
    for df in [
        df_run_of_river, df_ps_open, df_ps_closed, df_reservoir, df_pondage
    ]:
        df.reset_index(drop=True, inplace=True)

    # Return structured dictionary
    return {
        "Run-of-River": df_run_of_river,
        "PS Open": df_ps_open,
        "PS Closed": df_ps_closed,
        "Reservoir": df_reservoir,
        "Pondage": df_pondage
    }


def filter_hydro_profiles(df_dict, country_list, start_date, end_date):
    """
    Filters hydro generation profiles by time horizon and selected countries.

    Parameters
    ----------
    df_dict : dict of pandas.DataFrame
        Hydro generation profiles by technology.
    country_list : list of str
        Countries to include.
    start_date : pandas.Timestamp
        Start of time window (inclusive).
    end_date : pandas.Timestamp
        End of time window (exclusive).

    Returns
    -------
    dict of pandas.DataFrame
        Filtered hydro generation profiles.

    Notes
    -----
    - Missing values are replaced with zero (assumes no generation).
    - Only countries present in each dataset are retained.
    """

    df_hydro = {}

    for name, df in df_dict.items():

        # Ensure valid datetime format
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df = df.dropna(subset=["Timestamp"])

        # Apply time filter
        df = df.loc[(df["Timestamp"] >= start_date) & (df["Timestamp"] < end_date)].copy()

        # Filter available country columns
        df = df[["Timestamp"] + [c for c in country_list if c in df.columns]]

        # Replace missing values with 0 (no generation assumption)
        df_hydro[name] = df.fillna(0)

        # Reset index for consistency
        df.reset_index(drop=True, inplace=True)

    return df_hydro


# ---------------------------------------------------------
# 2) LOAD HYDRO CAPACITIES FROM PEMMDB MarketNodeInfo
# ---------------------------------------------------------

def load_hydro_capacities(pfad, country_list, year=2040):
    """
    Loads installed hydro capacities from PEMMDB MarketNodeInfo sheets.

    Parameters
    ----------
    pfad : str
        Base directory path.
    country_list : list of str
        Countries to include.
    year : int, optional
        Target year (default: 2040).

    Returns
    -------
    pandas.DataFrame
        Capacity table with:
        - Rows: hydro technology categories
        - Columns: countries
        - Values: installed capacity (MW)

    Notes
    -----
    - Multiple file suffixes are checked due to naming inconsistencies.
    - Missing files are reported but not fatal.
    """

    suffixes = ["00", "G1", "CA"]  # possible file suffixes

    # Initialize empty capacity table
    capacity_table = pd.DataFrame(index=categories, columns=country_list)
    capacity_table[:] = 0

    for country in country_list:
        found = False

        for suffix in suffixes:
            file_path = os.path.join(
                pfad, "2040", "res_2040",
                f"PEMMDB_{country}{suffix}_Hydro_Inflows_{year}.xlsx"
            )

            if os.path.exists(file_path):
                # Read relevant sheet (skip metadata rows)
                df = pd.read_excel(
                    file_path,
                    sheet_name="MarketNodeInfo",
                    skiprows=2
                )

                # Extract capacity values per category
                for cat in categories:
                    row = df[df.iloc[:, 0] == cat]
                    if len(row) == 1:
                        capacity_table.loc[cat, country] = row.iloc[0, 1]

                found = True
                break


    return capacity_table


def load_hydro_inertia(pfad, country_list, inertia_mapping, df_hydro, year=2040):
    """
    Computes the aggregated kinetic energy (inertia contribution) of hydro units.

    The inertia is calculated based on generation levels, installed capacity,
    and technology-specific inertia constants.

    Parameters
    ----------
    pfad : str
        Base directory path.
    country_list : list of str
        Countries to include.
    inertia_mapping : dict
        Mapping: hydro technology → inertia constant H (in seconds).
    df_hydro : dict of pandas.DataFrame
        Filtered hydro generation profiles.
    year : int, optional
        Target year (default: 2040).

    Returns
    -------
    pandas.DataFrame
        Time series of total hydro inertia per country (unit: MW·s or equivalent).

    Notes
    -----
    - Inertia is approximated as proportional to generation and inertia constant.
    - A load factor (LF) is used to normalize the contribution.
    - The inertia contribution is capped by installed capacity.
    """

    categories = ['Run-of-River', 'Pondage', 'Reservoir', 'PS Open', 'PS Closed']

    # Load installed capacities
    capacity_table = load_hydro_capacities(pfad, country_list, year=year)

    # Mapping between profile keys and capacity table categories
    category_map = {
        'Run-of-River': 'Run of River - MW',
        'Reservoir': 'Reservoir - MW',
        'Pondage': 'Pondage - MW',
        'PS Open': 'PS Open (turbine) - MW',
        'PS Closed': 'PS Closed (turbine) - MW'
    }

    # Assumed load factors per technology
    LF_mapping = {
        "Run-of-River": 0.61,
        "Pondage": 0.61,
        "Reservoir": 0.56,
        "PS Open": 0.46,
        "PS Closed": 0.46
    }

    # Initialize inertia time series (per country)
    df_hydro_inertia = pd.DataFrame(
        0.0,
        index=range(len(df_hydro['Run-of-River'])),
        columns=country_list
    )

    # Compute inertia contribution
    for country in country_list:

        for cat in categories:
            H = inertia_mapping[cat]   # inertia constant
            LF = LF_mapping[cat]       # load factor

            # Hourly generation (MW)
            P = df_hydro[cat][country].astype(float)

            # Compute kinetic energy contribution:
            # min(operational inertia, maximum inertia from installed capacity)
            # formula from ENTSO-E
            Ekin_cat = pd.Series(
                np.minimum(
                    P * H / LF,
                    H * capacity_table.loc[category_map.get(cat), country]
                ),
                index=P.index
            )

            # Align index with output DataFrame
            Ekin_cat.index = df_hydro_inertia.index

            # Aggregate contributions across technologies
            df_hydro_inertia[country] += Ekin_cat

    return df_hydro_inertia