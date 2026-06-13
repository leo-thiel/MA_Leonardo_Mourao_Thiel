# =========================================================
# Author: Leonardo Gabriel Mourao Thiel
# Project: Master Thesis – System Inertia in the Energy System of the Future: Model-Based Cost Optimization to Secure Inertia Requirements
# Topic: Parameterization of ThermalUnits
# Date: 27.04.2026
# =========================================================
import pandas as pd


# =========================================================
# Fuel category mapping
# =========================================================
# Maps detailed plant types from dataset to aggregated fuel categories
# used in the model (for cost and inertia parameters)
fuel_cat_mapping = {
    'CCGT new': 'Gas CC',
    'CCGT old 1': 'Gas CC',
    'CCGT old 2': 'Gas CC',
    'CCGT present 1': 'Gas CC',
    'CCGT present 2': 'Gas CC',
    'Conventional old 1': 'Gas CT',
    'Conventional old 2': 'Gas CT',
    'OCGT new': 'Gas CT',
    'OCGT old': 'Gas CT'
}


# =========================================================
# Thermal unit class
# =========================================================

class ThermalUnit:
    """
    Represents a thermal generation unit used in the dispatch model.

    Attributes
    ----------
    uid : str
        Unique identifier of the unit.
    fuel_type : str
        Original fuel type from dataset.
    fuel_category : str
        Aggregated fuel category (used for cost/inertia mapping).
    capacity_MW : float
        Installed capacity of the unit.
    min_stable_power_MW : float
        Minimum stable generation level.
    startup_costs : float
        Startup cost (scaled by unit capacity).
    plant_type : str
        Original plant classification.
    country : str
        Country of the unit.
    inertia : float
        Inertia contribution (scaled with capacity).
    marginal_cost : float
        Marginal generation cost (EUR/MWh).
    """

    def __init__(self, uid, fuel_type, capacity_MW, min_stable_power_MW=0,
                 ramp_up_rate_MWh_per_h=0, ramp_down_rate_MWh_per_h=0,
                 fuel_category=None, startup_costs=0, plant_type=None,
                 country=None, inertia=0, marginal_cost=1000):

        self.uid = uid
        self.fuel_type = fuel_type
        self.capacity_MW = capacity_MW

        # Operational constraints
        self.min_stable_power_MW = min_stable_power_MW

        # Fuel classification
        self.fuel_category = fuel_category

        # Economic parameters
        self.startup_costs = startup_costs
        self.marginal_cost = marginal_cost

        # Metadata
        self.plant_type = plant_type
        self.country = country

        # Physical parameter
        self.inertia = inertia

    def display(self):
        """
        Returns a readable summary of the unit.
        """
        return (
            f"ID: {self.uid}, Country: {self.country}, Plant: {self.plant_type}, "
            f"Fuel: {self.fuel_type}, Category: {self.fuel_category}, "
            f"Capacity MW: {self.capacity_MW}, MinStable MW: {self.min_stable_power_MW}, "
            f"Startup Costs: {self.startup_costs}, Inertia: {self.inertia}, "
            f"Marginal Costs: {self.marginal_cost}"
        )


# =========================================================
# Loading function
# =========================================================

def load_thermal_units(pfad, countryList, startup_costs_mapping, marginal_costs, inertia_mapping):
    """
    Loads and disaggregates thermal generation units from dataset.

    The dataset typically provides aggregated capacities with a number of units.
    This function splits them into individual units to enable modeling of
    unit commitment constraints (e.g., minimum load, startup costs).

    Parameters
    ----------
    pfad : str
        Base directory containing input data.
    countryList : list of str
        Countries to include.
    startup_costs_mapping : dict
        Startup cost per fuel category.
    marginal_costs : dict
        Marginal cost per fuel category.
    inertia_mapping : dict
        Inertia constant per fuel category.

    Returns
    -------
    list of ThermalUnit
        List of individual thermal units.

    Notes
    -----
    - Capacity is equally distributed across units.
    - Inertia scales linearly with capacity.
    - Startup costs are proportional to unit size.
    """

    file_path = f"{pfad}/2040/capacities_2040/thermal_cleaned.csv"
    df = pd.read_csv(file_path, sep=";", encoding='latin1')

    # ---------------------------------------------------------
    # Data cleaning and preprocessing
    # ---------------------------------------------------------

    df.columns = df.columns.str.strip().str.lower()

    df['plant_type'] = df['plant_type'].astype(str).fillna("Unknown")
    df['fuel_type'] = df['fuel_type'].astype(str).fillna("Unknown")

    df['number_of_units'] = df['number_of_units'].fillna(1).astype(int)
    df['net_capacity_mw'] = df['net_capacity_mw'].fillna(0).astype(float)

    df['min_stable_power_pct'] = df.get('min_stable_power_pct', 0).fillna(0).astype(float)

    # Optional ramp rates (currently not used in unit creation)
    df['ramp_up_rate_mwh_per_h'] = df.get('ramp_up_rate_mwh_per_h', 0).fillna(0).astype(float)
    df['ramp_down_rate_mwh_per_h'] = df.get('ramp_down_rate_mwh_per_h', 0).fillna(0).astype(float)

    # Filter relevant countries
    df = df[df['country'].isin(countryList)].reset_index(drop=True)

    unitsList = []
    global_counter = 1  # ensures unique IDs

    # ---------------------------------------------------------
    # Iterate through dataset
    # ---------------------------------------------------------

    for _, row in df.iterrows():

        n_units = int(row["number_of_units"])
        if n_units == 0:
            continue

        # Default minimum load if missing
        min_pct = float(row.get("min_stable_power_pct") or 20)

        # Split capacity equally across units
        unit_capacity = float(row["net_capacity_mw"]) / n_units

        for unit_id in range(1, n_units + 1):

            # Fallback if plant_type is invalid
            if row['plant_type'] == "nan":
                row['plant_type'] = row["fuel_type"]

            # Generate unique ID
            uid = f"{str(row['plant_type']).replace(' ', '_')}_{row['country']}_{row['year']}_{global_counter}"
            global_counter += 1

            plant_type_str = str(row["plant_type"])
            original_fuel = str(row["fuel_type"])

            # Map to aggregated fuel category
            fuel_cat = fuel_cat_mapping.get(plant_type_str, original_fuel)

            # Economic parameters
            startup_cost = startup_costs_mapping.get(fuel_cat, 250) * unit_capacity
            marginal_cost = marginal_costs.get(fuel_cat, 0)

            # Physical parameter (scaled with capacity)
            inertia = inertia_mapping.get(fuel_cat, 0) * unit_capacity

            # Create unit object
            unit = ThermalUnit(
                uid=uid,
                fuel_type=original_fuel,
                fuel_category=fuel_cat,
                startup_costs=startup_cost,
                plant_type=plant_type_str,
                country=row["country"],
                capacity_MW=unit_capacity,
                min_stable_power_MW=unit_capacity * (min_pct / 100),
                inertia=inertia,
                marginal_cost=marginal_cost
            )

            unitsList.append(unit)

    return unitsList