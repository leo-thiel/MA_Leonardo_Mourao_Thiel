# =========================================================
# Author: Leonardo Gabriel Mourao Thiel
# Project: Master Thesis – System Inertia in the Energy System of the Future: Model-Based Cost Optimization to Secure Inertia Requirements
# Topic: Parameterization of OtherRESUnits
# Date: 27.04.2026
# =========================================================
import pandas as pd
import math


class OtherResUnit:
    """
    Represents a dispatchable renewable energy (RES) unit such as biomass or waste.

    These units are modeled as thermal generators with renewable fuel input,
    including operational constraints such as minimum stable load, startup costs,
    and inertia contribution.

    Attributes
    ----------
    uid : str
        Unique identifier of the unit.
    country : str
        Country of the unit.
    market_node : str
        Market node or bidding zone.
    year : int
        Scenario year.
    scenario : str
        Scenario label.
    technology : str
        Original technology name from dataset.
    capacity_MW : float
        Installed capacity of the individual unit.
    min_stable_power_MW : float
        Minimum stable generation level.
    fueltype : str
        Internal fuel classification.
    marginal_cost : float
        Variable generation cost (EUR/MWh).
    startup_costs : float
        Startup cost per unit.
    inertia : float
        Inertia contribution (scaled by capacity).
    """

    def __init__(self, country, market_node, year, scenario, technology,
                 capacity_MW, min_stable_power_MW, fuel, marginal_cost,
                 startup_costs, inertia, uid):

        self.country = country
        self.market_node = market_node
        self.year = year
        self.scenario = scenario

        # Original technology label (as provided in dataset)
        self.technology = technology

        # Installed capacity of THIS individual unit (after disaggregation)
        self.capacity_MW = capacity_MW

        # Minimum stable generation level (technical constraint)
        self.min_stable_power_MW = min_stable_power_MW

        # Internal fuel classification (used for cost & inertia mapping)
        self.fueltype = fuel

        # Economic parameters
        self.marginal_cost = marginal_cost
        self.startup_costs = startup_costs

        # Physical system parameter
        self.inertia = inertia

        # Unique identifier
        self.uid = uid

    def display(self):
        """
        Returns a human-readable summary of the unit.

        Returns
        -------
        str
            Formatted string with key technical and economic parameters.
        """
        return (
            f"UID: {self.uid}, Country: {self.country}, Node: {self.market_node}, "
            f"Year: {self.year}, Scenario: {self.scenario}, Technology: {self.technology}, "
            f"Fuel: {self.fueltype}, Capacity: {self.capacity_MW} MW, "
            f"Min Stable Power: {self.min_stable_power_MW} MW, "
            f"Startup Cost: {self.startup_costs}, Marginal Cost: {self.marginal_cost}, "
            f"Inertia: {self.inertia}"
        )


def detect_fueltype(technology):
    """
    Classifies fuel type based on technology string.

    Parameters
    ----------
    technology : str
        Technology name from dataset.

    Returns
    -------
    str
        Simplified fuel classification.

    Notes
    -----
    - Used to harmonize inconsistent naming conventions.
    - Enables mapping to cost and inertia parameters.
    """

    tech = str(technology).lower()

    if "biomass" in tech:
        return "Biomass"
    elif "waste" in tech:
        return "Waste"
    else:
        return "Other"


def load_other_res_units(
    path,
    countryList,
    startup_costs_mapping,
    marginal_costs,
    inertia_mapping,
    unit_size_mapping=None,
    minimum_plant_mapping=None
):
    """
    Loads and disaggregates 'Other RES' units from aggregated capacity data.

    The dataset typically provides total installed capacity per country and technology.
    This function splits aggregated capacity into multiple representative units to
    better approximate unit commitment behavior (e.g., startup costs, minimum load).

    Parameters
    ----------
    path : str
        Root directory of the dataset.
    countryList : list of str
        Countries to include.
    startup_costs_mapping : dict
        Startup cost per fuel type.
    marginal_costs : dict
        Marginal cost per fuel type.
    inertia_mapping : dict
        Inertia constant per fuel type.
    unit_size_mapping : dict, optional
        Typical plant size (MW) used for disaggregation.
    minimum_plant_mapping : dict, optional
        Minimum load as percentage of capacity.

    Returns
    -------
    list of OtherResUnit
        List of disaggregated generation units.

    Notes
    -----
    - Disaggregation is required to represent non-linear operational constraints.
    - Larger plants → fewer units; smaller plants → more units.
    """

    # ---------------------------------------------------------
    # Default assumptions for plant size and minimum load
    # ---------------------------------------------------------

    if unit_size_mapping is None:
        unit_size_mapping = {
            "Biomass": 50,
            "Waste": 62.5,
            "Other": 50
        }

    if minimum_plant_mapping is None:
        minimum_plant_mapping = {
            "Biomass": 50,
            "Waste": 40,
            "Other": 50
        }

    # ---------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------

    df = pd.read_csv(
        f"{path}/2040/capacities_2040/other_res_capacity.csv",
        sep=";",
        encoding="latin1"
    )

    # Filter relevant subset
    df = df[df["country"].isin(countryList)]
    df = df[df["year"] == 2040]

    # Remove incomplete entries
    df = df.dropna(subset=["country", "technology", "installed_capacity_MW", "year"])

    # Convert capacity column (handle comma decimal separator)
    df["installed_capacity_MW"] = (
        df["installed_capacity_MW"]
        .astype(str)
        .str.replace(",", ".")
    )

    df["installed_capacity_MW"] = pd.to_numeric(
        df["installed_capacity_MW"],
        errors="coerce"
    )

    # Remove invalid capacities
    df = df[df["installed_capacity_MW"] > 0]

    units = []
    global_counter = 1  # ensures unique IDs

    # ---------------------------------------------------------
    # Iterate through dataset (efficient iteration)
    # ---------------------------------------------------------

    for row in df.itertuples(index=False):

        installed_capacity = float(row.installed_capacity_MW)

        if installed_capacity <= 0:
            continue

        # Classify fuel type
        fueltype = detect_fueltype(row.technology)

        # Determine representative unit size
        unit_size = unit_size_mapping.get(fueltype, 10)

        # Compute number of units (ceil ensures full coverage)
        num_units = max(1, math.ceil(installed_capacity / unit_size))

        # Capacity per modeled unit
        capacity_per_unit = installed_capacity / num_units

        # Minimum stable generation constraint
        min_stable_power_MW = (
            minimum_plant_mapping.get(fueltype, 50) / 100
            * capacity_per_unit
        )

        # Economic parameters
        startup_cost = startup_costs_mapping.get(fueltype, 250) * capacity_per_unit
        marginal_cost = marginal_costs.get(fueltype, 250)

        # ---------------------------------------------------------
        # Create individual unit representations
        # ---------------------------------------------------------

        for _ in range(num_units):

            uid = f"{row.country}_{fueltype}_{int(row.year)}_{global_counter}"
            global_counter += 1

            # Inertia scales linearly with capacity
            inertia = inertia_mapping.get(fueltype, 0) * capacity_per_unit

            unit = OtherResUnit(
                country=row.country,
                market_node=getattr(row, "market_node", "Unknown"),
                year=int(row.year),
                scenario=getattr(row, "scenario", "Base"),
                technology=row.technology,
                capacity_MW=capacity_per_unit,
                min_stable_power_MW=min_stable_power_MW,
                fuel=fueltype,
                marginal_cost=marginal_cost,
                startup_costs=startup_cost,
                inertia=inertia,
                uid=uid
            )

            units.append(unit)

    # ---------------------------------------------------------
    # Ensure UID uniqueness
    # ---------------------------------------------------------

    uids = [u.uid for u in units]
    assert len(uids) == len(set(uids)), "Duplicate Other RES Unit IDs detected!"

    return units