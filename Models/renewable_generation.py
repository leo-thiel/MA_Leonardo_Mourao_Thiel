# =========================================================
# Author: Leonardo Gabriel Mourao Thiel
# Project: Master Thesis – System Inertia in the Energy System of the Future: Model-Based Cost Optimization to Secure Inertia Requirements
# Topic: Parameterization of renewable generation
# Date: 27.04.2026
# =========================================================

import pandas as pd


# Mapping of countries to their corresponding market nodes
# Used to aggregate nodal capacities to country level
country_nodes_map = {
    "AL": ["AL00"],
    "AT": ["AT00"],
    "BA": ["BA00"],
    "BE": ["BE00", "BEOF"],
    "BG": ["BG00"],
    "CH": ["CH00"],
    "CZ": ["CZ00"],
    "DE": ["DE00", "DEKF"],
    "DK": ["DKBH", "DKE1", "DKKF", "DKNS", "DKW1"],
    "ES": ["ES00"],
    "FR": ["FR00"],
    "GR": ["GR00", "GR03"],
    "HR": ["HR00"],
    "HU": ["HU00"],
    "IT": ["ITA00", "ITCA", "ITCN", "ITCS", "ITN1", "ITS1", "ITSA", "ITSI"],
    "LU": ["LUB1", "LUF1", "LUG1", "LUV1"],
    "MK": ["MK00"],
    "ME": ["ME00"],
    "NL": ["NL00", "NL60", "NL6H", "NLA0", "NLBH", "NLLL"],
    "PL": ["PL00"],
    "PT": ["PT00"],
    "RO": ["RO00"],
    "RS": ["RS00"],
    "SI": ["SI00"],
    "SK": ["SK00"]
}


def load_generation_profiles(pfad):
    """
    Loads renewable generation profiles (solar and wind) from Excel files.

    Parameters
    ----------
    pfad : str
        Base directory containing generation data.

    Returns
    -------
    dict of pandas.DataFrame
        Dictionary containing generation profiles for:
        - solar (utility-scale PV)
        - rooftop PV
        - onshore wind
        - offshore wind

    Notes
    -----
    - Each DataFrame contains a 'Timestamp' column and country columns.
    - No filtering or cleaning is applied at this stage.
    """

    df_dict = {
        "solar": pd.read_excel(f"{pfad}/2040/Generation_2040/SolarPV_2040.xlsx"),
        "rooftop": pd.read_excel(f"{pfad}/2040/Generation_2040/SolarRooftop_2040.xlsx"),
        "onshore": pd.read_excel(f"{pfad}/2040/Generation_2040/Wind_Onshore_2040.xlsx"),
        "offshore": pd.read_excel(f"{pfad}/2040/Generation_2040/Wind_Offshore_2040.xlsx"),
    }

    # Ensure consistent indexing
    for df in df_dict.values():
        df.reset_index(drop=True, inplace=True)

    return df_dict


def load_load_profiles(pfad):
    """
    Loads hourly demand data with parsed timestamps.

    Parameters
    ----------
    pfad : str
        Base directory containing demand data.

    Returns
    -------
    pandas.DataFrame
        DataFrame with timestamped demand per country.
    """

    return pd.read_csv(
        f"{pfad}/2040/load_hourly_countries_2040.csv",
        sep=";",
        parse_dates=["Timestamp"],
        dayfirst=True
    )


def clean_and_align_timestamps(df_dict):
    """
    Standardizes timestamp columns across all renewable datasets.

    Parameters
    ----------
    df_dict : dict of pandas.DataFrame
        Dictionary of generation profiles.

    Returns
    -------
    dict of pandas.DataFrame
        Cleaned DataFrames with consistent 'Timestamp' column.

    Notes
    -----
    - Renames 'time' column to 'Timestamp' if necessary.
    - Removes invalid timestamps.
    """

    for key, df in df_dict.items():

        # Harmonize column naming
        if "time" in df.columns:
            df.rename(columns={"time": "Timestamp"}, inplace=True)

        # Convert to datetime and drop invalid entries
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df.dropna(subset=["Timestamp"], inplace=True)

    return df_dict


def compute_total_renewables(df_dict, country_list, start_date, end_date):
    """
    Computes total renewable generation per country by summing all RES sources.

    Parameters
    ----------
    df_dict : dict of pandas.DataFrame
        Renewable generation datasets.
    country_list : list of str
        Countries to include.
    start_date : datetime-like
        Start of time window (inclusive).
    end_date : datetime-like
        End of time window (exclusive).

    Returns
    -------
    tuple of pandas.DataFrame
        - Total renewable generation
        - Rooftop PV generation
        - Utility-scale PV generation

    Notes
    -----
    - Missing country columns are treated as zero generation.
    - Assumes all datasets share the same temporal resolution.
    """

    df_solar = df_dict["solar"]
    df_roof = df_dict["rooftop"]
    df_onshore = df_dict["onshore"]
    df_offshore = df_dict["offshore"]

    # Initialize result DataFrame with timestamps
    df_total = pd.DataFrame()
    df_total["Timestamp"] = df_solar["Timestamp"]

    # Aggregate generation per country
    for land in country_list:
        solar = df_solar[land] if land in df_solar.columns else 0
        roof = df_roof[land] if land in df_roof.columns else 0
        onshore = df_onshore[land] if land in df_onshore.columns else 0
        offshore = df_offshore[land] if land in df_offshore.columns else 0

        df_total[land] = solar + roof + onshore + offshore

    # Apply time filtering
    df_total = df_total[
        (df_total["Timestamp"] >= start_date) &
        (df_total["Timestamp"] < end_date)
    ]
    df_total.reset_index(drop=True, inplace=True)

    # Filter individual components for consistency
    df_solar = df_solar[
        (df_solar["Timestamp"] >= start_date) &
        (df_solar["Timestamp"] < end_date)
    ].reset_index(drop=True)

    df_roof = df_roof[
        (df_roof["Timestamp"] >= start_date) &
        (df_roof["Timestamp"] < end_date)
    ].reset_index(drop=True)

    return df_total, df_roof, df_solar


def aggregate_by_country(capacity_map, country_nodes_map):
    """
    Aggregates nodal capacities to country level.

    Parameters
    ----------
    capacity_map : dict
        Mapping: market_node → capacity.
    country_nodes_map : dict
        Mapping: country → list of market nodes.

    Returns
    -------
    dict
        Mapping: country → aggregated capacity.
    """

    return {
        country: sum(capacity_map.get(node, 0) for node in nodes)
        for country, nodes in country_nodes_map.items()
    }


def get_capacity(pfad):
    """
    Loads installed capacities and aggregates them from node level to country level.

    Parameters
    ----------
    pfad : str
        Base directory containing capacity datasets.

    Returns
    -------
    tuple of dict
        - Rooftop PV capacity (MW)
        - Utility-scale PV capacity (MW)
        - Onshore wind capacity (MW)
        - Offshore wind capacity (MW)

    Notes
    -----
    - Input data may be in GW → converted to MW.
    - Aggregation is based on predefined country-node mapping.
    """

    # Load datasets
    df_solar_capacity = pd.read_csv(
        pfad + "/2040/capacities_2040/solar_capacity_2040.csv",
        sep=";"
    )

    df_wind_capacity = pd.read_csv(
        pfad + "/2040/capacities_2040/wind_cleaned.csv",
        sep=";"
    )

    # --- Solar capacities ---
    capacity_map_rooftop = (
        df_solar_capacity[df_solar_capacity["solar_type"] == "solar_rooftop"]
        .set_index("market_node")["capacity"]
        .to_dict()
    )

    capacity_map_pv = (
        df_solar_capacity[df_solar_capacity["solar_type"] == "solar_pv"]
        .set_index("market_node")["capacity"]
        .to_dict()
    )

    # --- Wind capacities ---
    capacity_map_onshore = (
        df_wind_capacity[df_wind_capacity["wind_type"] == "onshore"]
        .set_index("market_node")["installed_capacity_GW"]
        .to_dict()
    )

    capacity_map_offshore = (
        df_wind_capacity[df_wind_capacity["wind_type"] == "offshore"]
        .set_index("market_node")["installed_capacity_GW"]
        .to_dict()
    )

    # Aggregate to country level
    country_capacity_rooftop = aggregate_by_country(capacity_map_rooftop, country_nodes_map)
    country_capacity_pv = aggregate_by_country(capacity_map_pv, country_nodes_map)
    country_capacity_onshore = aggregate_by_country(capacity_map_onshore, country_nodes_map)
    country_capacity_offshore = aggregate_by_country(capacity_map_offshore, country_nodes_map)

    # Convert from GW to MW (if applicable)
    country_capacity_rooftop = {k: float(v) * 1000.0 for k, v in country_capacity_rooftop.items()}
    country_capacity_pv = {k: float(v) * 1000.0 for k, v in country_capacity_pv.items()}
    country_capacity_onshore = {k: float(v) * 1000.0 for k, v in country_capacity_onshore.items()}
    country_capacity_offshore = {k: float(v) * 1000.0 for k, v in country_capacity_offshore.items()}

    return (
        country_capacity_rooftop,
        country_capacity_pv,
        country_capacity_onshore,
        country_capacity_offshore
    )