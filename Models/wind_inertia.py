# =========================================================
# Author: Leonardo Gabriel Mourao Thiel
# Project: Master Thesis – System Inertia in the Energy System of the Future: Model-Based Cost Optimization to Secure Inertia Requirements
# Topic: Parameterization of Inertia profiles for wind turbines
# Date: 27.04.2026
# =========================================================

import pandas as pd


def inertia_profiles(pfad, startdate, enddate):
    """
    Loads precomputed wind inertia profiles (kinetic energy contribution)
    and filters them to a specified time range.

    Parameters
    ----------
    pfad : str
        Base directory containing inertia datasets.
    startdate : str or pandas.Timestamp
        Start of time window (inclusive).
    enddate : str or pandas.Timestamp
        End of time window (exclusive).
    time_col : str, optional
        Name of the time column in the input files (default: "time").

    Returns
    -------
    dict of pandas.DataFrame
        Dictionary containing:
        - "onshore": inertia time series for onshore wind
        - "offshore": inertia time series for offshore wind

    Notes
    -----
    - Inertia values are assumed to be precomputed externally.
    - Time filtering ensures consistency with the model simulation horizon.
    """

    # Load inertia datasets
    df_onshore_inertia = pd.read_excel(
        f"{pfad}/2040/Generation_2040/Wind_Onshore_Inertia_2040.xlsx"
    )
    df_offshore_inertia = pd.read_excel(
        f"{pfad}/2040/Generation_2040/Wind_Offshore_Inertia_2040.xlsx"
    )

    # Convert time column to datetime
    df_onshore_inertia["time"] = pd.to_datetime(df_onshore_inertia["time"])
    df_offshore_inertia["time"] = pd.to_datetime(df_offshore_inertia["time"])

    # Convert filtering bounds
    startdate = pd.to_datetime(startdate)
    enddate = pd.to_datetime(enddate)

    # Apply time filtering
    df_onshore_inertia = df_onshore_inertia[
        (df_onshore_inertia["time"] >= startdate) &
        (df_onshore_inertia["time"] < enddate)
    ].reset_index(drop=True)

    df_offshore_inertia = df_offshore_inertia[
        (df_offshore_inertia["time"] >= startdate) &
        (df_offshore_inertia["time"] < enddate)
    ].reset_index(drop=True)

    return {
        "onshore": df_onshore_inertia,
        "offshore": df_offshore_inertia,
    }


def load_load_profiles(pfad):
    """
    Loads hourly electricity demand profiles.

    Parameters
    ----------
    pfad : str
        Base directory containing demand data.

    Returns
    -------
    pandas.DataFrame
        DataFrame with:
        - Timestamp column (datetime)
        - Country-level demand values

    Notes
    -----
    - Assumes European date format (day-first).
    """

    return pd.read_csv(
        f"{pfad}/2040/load_hourly_countries_2040.csv",
        sep=";",
        parse_dates=["Timestamp"],
        dayfirst=True
    )


def clean_and_align_timestamps(df_dict):
    """
    Standardizes timestamp columns across multiple datasets.

    Parameters
    ----------
    df_dict : dict of pandas.DataFrame
        Dictionary containing time series datasets.

    Returns
    -------
    dict of pandas.DataFrame
        Cleaned datasets with consistent timestamp format.

    Notes
    -----
    - Renames 'time' column to 'Timestamp' if necessary.
    - Removes invalid or missing timestamps.
    - Ensures compatibility across different data sources.
    """

    for key, df in df_dict.items():

        # Rename column if necessary
        if "time" in df.columns:
            df.rename(columns={"time": "Timestamp"}, inplace=True)

        # Convert to datetime and remove invalid entries
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df.dropna(subset=["Timestamp"], inplace=True)

    return df_dict