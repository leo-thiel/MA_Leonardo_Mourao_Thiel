# =========================================================
# Author: Leonardo Gabriel Mourao Thiel
# Project: Master Thesis – System Inertia in the Energy System of the Future: Model-Based Cost Optimization to Secure Inertia Requirements
# Topic: Parameterization of NonResUnits
# Date: 27.04.2026
# =========================================================
import pandas as pd


class NonResUnit:
    """
    Represents a non-renewable residual capacity unit with time-dependent availability.

    Attributes
    ----------
    uid : str
        Unique identifier of the unit (typically country-level aggregation).
    country : str
        Country where the unit is located.
    costs : float
        Effective marginal cost in EUR/MWh, including efficiency losses and CO₂ pricing.
    schedule : dict
        Time-dependent available capacity (Timestamp -> MW).
    """

    def __init__(self, unit_id, country, costs, schedule):
        """
        Initializes a NonResUnit instance.

        Parameters
        ----------
        unit_id : str
            Unique identifier of the unit.
        country : str
            Country of operation.
        costs : float
            Effective marginal cost (EUR/MWh).
        schedule : dict
            Mapping of timestamps to available capacity (MW).
        """
        self.uid = unit_id
        self.country = country
        self.costs = costs
        self.schedule = schedule  # Mapping: Timestamp -> MW

    def display(self):
        """
        Returns a concise string representation of the unit.

        Returns
        -------
        str
            Summary including ID, country, cost, and number of time steps.
        """
        return (
            f"NonResUnit: {self.uid}, Country: {self.country}, "
            f"Costs: {self.costs:.2f} EUR/MW, Hours: {len(self.schedule)}"
        )


def load_nonres_units(
    pfad,
    countryList,
    start_date,
    end_date,
    timeseries_file="other_nonres_timeseries.csv",
    capacity_file="other_nonres_capacity.csv",
    co2_price=147
):
    """
    Loads non-renewable residual capacity units and constructs time-dependent schedules.

    The function combines time series availability data with static capacity and cost
    parameters, applies temporal filtering, and computes effective marginal costs.

    Parameters
    ----------
    pfad : str
        Base directory path containing input CSV files.
    countryList : list of str
        Countries to include.
    start_date : str or pandas.Timestamp
        Start of the time window (inclusive).
    end_date : str or pandas.Timestamp
        End of the time window (inclusive).
    timeseries_file : str, optional
        Filename of the availability time series dataset.
    capacity_file : str, optional
        Filename of the capacity and cost dataset.
    co2_price : float, optional
        CO₂ price in EUR/tCO₂ used to compute emission-related costs.

    Returns
    -------
    list of NonResUnit
        List of aggregated non-renewable units per country.

    Notes
    -----
    - Each country is represented as a single aggregated unit.
    - Availability is scaled by the number of units.
    - Costs include:
        * fuel/marginal cost adjusted by efficiency
        * CO₂ cost component
    """

    # Load input datasets
    df_ts_raw = pd.read_csv(
        f"{pfad}/2040/capacities_2040/{timeseries_file}",
        sep=";",
        low_memory=False
    )
    df_cap = pd.read_csv(
        f"{pfad}/2040/capacities_2040/{capacity_file}",
        sep=";",
        low_memory=False
    )

    # Filter datasets for target year and selected countries
    df_ts = df_ts_raw[
        (df_ts_raw["year"] == 2040) &
        (df_ts_raw["country"].isin(countryList))
    ].copy()

    df_cap = df_cap[
        (df_cap["year"] == 2040) &
        (df_cap["country"].isin(countryList))
    ].copy()

    # Create lookup tables for cost and technical parameters
    cost_lookup = dict(zip(df_cap["country"], df_cap["price_EUR_MWh"]))
    eff_lookup  = dict(zip(df_cap["country"], df_cap["efficiency"]))
    co2_lookup  = dict(zip(df_cap["country"], df_cap["co2_factor_t_per_MWh"]))
    n_units_lookup = dict(zip(df_cap["country"], df_cap["units"]))

    # Convert time boundaries
    start_date = pd.to_datetime(start_date)
    end_date   = pd.to_datetime(end_date)

    nonres_units_objects = []

    # Group time series by country (one aggregated unit per country)
    for country, group in df_ts.groupby("country"):

        group = group.copy()

        # Construct full timestamp:
        # - date string (day/month)
        # - fixed year (2040)
        # - hourly offset (hour index starts at 1)
        group["date_full"] = group["date"].str.strip() + "2040"
        group["Timestamp"] = pd.to_datetime(
            group["date_full"],
            format="%d.%m.%Y",
            errors="coerce"
        ) + pd.to_timedelta(group["hour"] - 1, unit="h")

        # Apply time filter
        group = group[
            (group["Timestamp"] >= start_date) &
            (group["Timestamp"] <= end_date)
        ]

        # Ensure valid numeric capacity values
        group["available_capacity_MW"] = pd.to_numeric(
            group["available_capacity_MW"],
            errors="coerce"
        ).fillna(0)

        # Create availability schedule (Timestamp -> MW)
        schedule = dict(zip(
            group["Timestamp"],
            group["available_capacity_MW"]
        ))

        # Retrieve number of units for scaling
        n_units = n_units_lookup.get(country, 0)

        # Compute effective marginal costs:
        # - Fuel cost adjusted by efficiency
        # - Add CO₂ cost component
        eff = max(eff_lookup.get(country, 1), 1)  # avoid division by <1 or zero
        costs = (
            cost_lookup.get(country) / eff +
            co2_lookup.get(country) * co2_price
        )

        # Only include units with non-empty schedules
        if schedule:
            unit_id = f"{country}"

            # Scale availability by number of units
            scaled_schedule = {
                ts: cap * n_units for ts, cap in schedule.items()
            }

            nonres_units_objects.append(
                NonResUnit(
                    unit_id,
                    country,
                    costs,
                    scaled_schedule
                )
            )

    return nonres_units_objects