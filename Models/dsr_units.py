# =========================================================
# Author: Leonardo Gabriel Mourao Thiel
# Project: Master Thesis – System Inertia in the Energy System of the Future: Model-Based Cost Optimization to Secure Inertia Requirements
# Topic: Parameterization of Demand Side Resposne Units
# Date: 27.04.2026
# =========================================================
import pandas as pd


class DSRUnit:
    """
    Represents a Demand Side Response (DSR) unit characterized by a cost parameter
    and an availability schedule.

    Attributes
    ----------
    uid : str
        Unique identifier of the DSR unit (typically country + band ID).
    country : str
        Country in which the DSR resource is located.
    costs : float
        Activation cost in EUR/MWh.
    schedule : pandas.Series
        Time-indexed availability profile (Timestamp -> available MW).
    """

    def __init__(self, unit_id, country, costs, schedule):
        """
        Initializes a DSRUnit instance.

        Parameters
        ----------
        unit_id : str
            Unique identifier for the DSR unit.
        country : str
            Country of operation.
        costs : float
            Marginal activation cost in EUR/MWh.
        schedule : pandas.Series
            Time series of available capacity (MW), indexed by timestamps.
        """
        self.uid = unit_id
        self.country = country
        self.costs = costs
        self.schedule = schedule  # Mapping: Timestamp -> MW

    def display(self):
        """
        Returns a compact string representation of the DSR unit.

        Returns
        -------
        str
            Summary including ID, country, cost, and number of active hours.
        """
        return (
            f"DSRUnit: {self.uid}, Country: {self.country}, "
            f"Costs: {self.costs} EUR/MWh, Hours: {len(self.schedule)}"
        )


def load_dsr_units(pfad, countryList, start_date, end_date):
    """
    Loads and processes Demand Side Response (DSR) units from CSV files.

    The function constructs time-dependent availability schedules for each
    (country, band_id) combination, assigns cost parameters, and filters
    the data to a specified time horizon.

    Parameters
    ----------
    pfad : str
        Base directory path containing the input CSV files.
    countryList : list of str
        List of countries to include.
    start_date : str or pandas.Timestamp
        Start of the time window (inclusive).
    end_date : str or pandas.Timestamp
        End of the time window (inclusive).

    Returns
    -------
    list of DSRUnit
        List of DSRUnit objects with time-resolved availability profiles.

    Notes
    -----
    - Each (country, band_id) combination is treated as one aggregated DSR unit.
    - Availability is derived from hourly time series data.
    - Units with missing or invalid cost data are excluded.
    """

    # Load time series data (hourly availability per band)
    df_timeseries = pd.read_csv(
        f"{pfad}/2040/capacities_2040/dsr_timeseries.csv",
        sep=";",
        dtype={
            "year": float,
            "country": str,
            "band_id": float,
            "hour": float,
            "available_MW": float,
            "date": str
        },
        low_memory=False
    )
    df_timeseries.reset_index(drop=True, inplace=True)

    # Remove rows with missing temporal information
    df_timeseries = df_timeseries.dropna(subset=["year", "hour"])

    # Ensure correct data types for temporal fields
    df_timeseries["year"] = df_timeseries["year"].astype(int)
    df_timeseries["hour"] = df_timeseries["hour"].astype(int)

    # Load cost and capacity summary data
    df_capacity = pd.read_csv(
        f"{pfad}/2040/capacities_2040/dsr_band_summary.csv",
        sep=";"
    )

    # Filter both datasets for selected countries and target year (2040)
    df_timeseries = df_timeseries[
        (df_timeseries["year"] == 2040) &
        (df_timeseries["country"].isin(countryList))
    ]
    df_capacity = df_capacity[
        (df_capacity["year"] == 2040) &
        (df_capacity["country"].isin(countryList))
    ]

    # Create lookup table for marginal costs:
    # Key: (country, band_id) → Value: price_EUR_MWh
    cost_lookup = df_capacity.set_index(
        ["country", "band_id"]
    )["price_EUR_MWh"].to_dict()

    # NOTE: This lookup is currently redundant (same as cost_lookup)
    n_unit_lookup = df_capacity.set_index(
        ["country", "band_id"]
    )["price_EUR_MWh"].to_dict()

    # Convert time boundaries to pandas datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    dsr_units_objects = []

    # Group time series by (country, band_id) → one DSR unit per group
    for (country, band_id), group in df_timeseries.groupby(["country", "band_id"]):

        # Generate unique unit identifier
        unit_id = f"{country}_band{int(band_id)}"

        group = group.copy()

        # Construct timestamp:
        # - Original date string (day/month)
        # - Year explicitly set to 2040
        # - Hour offset applied (hour-1 because hours are 1-based)
        group["Timestamp"] = pd.to_datetime(
            group["date"] + "2040",
            format="%d.%m.%Y",
            errors="coerce"
        ) + pd.to_timedelta(group["hour"] - 1, unit="h")

        # Filter to requested time horizon
        mask = (
            (group["Timestamp"] >= start_date) &
            (group["Timestamp"] <= end_date)
        )
        group = group.loc[mask].copy()
        group.reset_index(drop=True, inplace=True)

        # Ensure numeric availability values and replace invalid entries with 0
        group["available_MW"] = pd.to_numeric(
            group["available_MW"],
            errors="coerce"
        ).fillna(0)

        # Create time-indexed availability schedule
        schedule = group.set_index("Timestamp")["available_MW"]

        # Retrieve marginal cost from lookup table
        costs = cost_lookup.get((country, band_id), 0.0)

        # Skip units with invalid cost values (e.g., NaN)
        if str(costs) == "nan":
            continue

        # Only include units with at least one positive availability value
        # no empty DSR schedule should be used to avoid unnecessery model comlexity
        if (schedule > 0).any():
            dsr_unit = DSRUnit(
                unit_id=unit_id,
                country=country,
                costs=float(costs),
                schedule=schedule
            )
            dsr_units_objects.append(dsr_unit)

    return dsr_units_objects