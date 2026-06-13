# =========================================================
# Author: Leonardo Gabriel Mourao Thiel
# Project: Master Thesis – System Inertia in the Energy System of the Future: Model-Based Cost Optimization to Secure Inertia Requirements
# Topic: Parameterization of demand profile
# Date: 27.04.2026
# =========================================================
import pandas as pd
import os


def load_hourly_demand(base_path, country_list, start_date, end_date, filename="load_hourly_countries_2040.csv"):
    """
    Loads hourly electricity demand data for selected countries, using 2016 as a
    reference year and mapping timestamps to the year 2040.

    The function filters the dataset to a specified time window and retains only
    the relevant country columns.

    Parameters
    ----------
    base_path : str
        Base directory containing the data (e.g., ".../Masterarbeit/Daten").
    country_list : list of str
        List of country identifiers (column names) to include in the output.
    start_date : str or pandas.Timestamp
        Start of the time window (inclusive).
    end_date : str or pandas.Timestamp
        End of the time window (exclusive).
    filename : str, optional
        Name of the CSV file (default: "load_hourly_countries_2040.csv").

    Returns
    -------
    pandas.DataFrame
        DataFrame containing:
        - 'Timestamp' column (shifted to year 2040)
        - Demand values for selected countries

    Notes
    -----
    - The year 2016 is used as a historical reference profile (e.g., realistic load patterns).
    - Timestamps are shifted to 2040 while preserving intra-year temporal structure
      (month, day, hour).
    - The function assumes that country names correspond to column headers in the dataset.
    """

    # Construct full file path to the dataset
    file_path = os.path.join(base_path, "2040", filename)

    # Load CSV file with timestamp parsing (European date format assumed)
    df = pd.read_csv(
        file_path,
        sep=";",
        parse_dates=["Timestamp"],
        dayfirst=True
    )

    # Identify rows corresponding to the reference year (2016)
    mask_2016 = df['Timestamp'].dt.year == 2016

    # Shift timestamps from 2016 to 2040 while preserving month/day/hour structure
    df.loc[mask_2016, "Timestamp"] = df.loc[mask_2016, "Timestamp"].apply(
        lambda ts: ts.replace(year=2040)
    )

    # Convert input date boundaries to pandas datetime objects
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Filter dataset to the specified time interval [start_date, end_date)
    df_filtered = df[
        (df["Timestamp"] >= start_date) &
        (df["Timestamp"] < end_date)
    ]

    # Select only relevant columns:
    # - Timestamp
    # - Countries that exist in the dataset
    cols_to_keep = ["Timestamp"] + [
        c for c in country_list if c in df.columns
    ]
    df_filtered = df_filtered[cols_to_keep]

    # Reset index for clean downstream processing
    df_filtered.reset_index(drop=True, inplace=True)

    return df_filtered