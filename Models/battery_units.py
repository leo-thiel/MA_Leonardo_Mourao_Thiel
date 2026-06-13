# =========================================================
# Author: Leonardo Gabriel Mourao Thiel
# Project: Master Thesis – System Inertia in the Energy System of the Future: Model-Based Cost Optimization to Secure Inertia Requirements
# Topic: Parameterization of Battery Storage Units
# Date: 27.04.2026
# =========================================================

import pandas as pd


class BatteryUnit:
    """
    Represents a battery energy storage system (BESS) unit with its core technical attributes.

    Attributes
    ----------
    uid : str or int
        Unique identifier of the battery unit.
    type : str
        Technology or classification of the battery (e.g., Li-ion, NaS).
    country : str
        Country where the battery unit is located.
    storage_capacity_MWh : float
        Energy storage capacity in megawatt-hours (MWh).
    power_capacity_MW : float
        Maximum charge/discharge power in megawatts (MW).
    efficiency : float
        Round-trip efficiency of the battery (typically between 0 and 1).
    """

    def __init__(self, uid, type_, country, storage_capacity_MWh, power_capacity_MW, efficiency):
        """
        Initializes a BatteryUnit instance with the provided parameters.

        Parameters
        ----------
        uid : str or int
            Unique identifier of the battery unit.
        type_ : str
            Battery technology type (named `type_` to avoid conflict with Python keyword `type`).
        country : str
            Country of installation.
        storage_capacity_MWh : float
            Total energy storage capacity in MWh.
        power_capacity_MW : float
            Maximum power output/input in MW.
        efficiency : float
            Round-trip efficiency (0 <= efficiency <= 1).
        """
        # Assign unique identifier
        self.uid = uid

        # Store battery type (technology classification)
        self.type = type_

        # Store geographic location
        self.country = country

        # Energy capacity of the battery system
        self.storage_capacity_MWh = storage_capacity_MWh

        # Power capacity (charge/discharge limit)
        self.power_capacity_MW = power_capacity_MW

        # Round-trip efficiency of the system
        self.efficiency = efficiency

    def display(self):
        """
        Returns a human-readable string representation of the battery unit.

        Returns
        -------
        str
            Formatted string summarizing key attributes of the battery unit.
        """
        return (
            f"ID: {self.uid}, Type: {self.type}, Country: {self.country}, "
            f"Storage MWh: {self.storage_capacity_MWh}, Power MW: {self.power_capacity_MW}, "
            f"Efficiency: {self.efficiency}"
        )

# === Loading function with global counter for unique IDs ===
def load_battery_units(pfad, countryList):
    """
    Loads and preprocesses battery unit data from a CSV file and converts it into
    a list of BatteryUnit objects.

    The function performs data cleaning, filtering by country, and assigns a unique
    identifier (UID) to each battery unit using a global counter.

    Parameters
    ----------
    pfad : str
        Base directory path where the battery dataset is stored.
    countryList : list of str
        List of country codes/names used to filter the dataset.

    Returns
    -------
    list of BatteryUnit
        List containing initialized BatteryUnit objects.

    Notes
    -----
    - Missing values are handled with domain-specific defaults.
    - Units with zero efficiency are excluded from the dataset.
    - A global counter is used to ensure unique identifiers across all units.
    """

    # Load CSV file containing battery data
    df_Battery = pd.read_csv(
        pfad + "/2040/capacities_2040/battery_cleaned.csv",
        sep=";",
        encoding='latin1'
    )

    # Standardize column names (remove whitespace and convert to lowercase)
    df_Battery.columns = df_Battery.columns.str.strip().str.lower()

    # Data cleaning and type casting with fallback defaults
    df_Battery['type'] = df_Battery['type'].astype(str).fillna("Battery")

    # Energy capacity in MWh (default = 0 if missing)
    df_Battery['storage_capacity'] = df_Battery['storage_capacity'].fillna(0).astype(float)

    # Power capacity in MW (default = 0 if missing)
    df_Battery['net_maximum_capacity_generation_perspective'] = (
        df_Battery['net_maximum_capacity_generation_perspective']
        .fillna(0)
        .astype(float)
    )

    # Country information (fallback = "Unknown")
    df_Battery['country'] = df_Battery['country'].astype(str).fillna("Unknown")

    # Efficiency (default = 1.0 → assumes ideal system if missing)
    df_Battery['average_efficiency'] = df_Battery['average_efficiency'].fillna(1.0).astype(float)

    # Filter dataset to include only specified countries
    df_Battery = df_Battery[df_Battery['country'].isin(countryList)].reset_index(drop=True)

    # Initialize output list and global UID counter
    unitsList = []
    global_counter = 1  # Ensures unique IDs across all generated units

    # Iterate over each row in the dataset
    for t, row in df_Battery.iterrows():

        # Extract relevant parameters
        storage_capacity_total = float(row["storage_capacity"])
        power_capacity_total = float(row["net_maximum_capacity_generation_perspective"])
        efficiency = float(row["average_efficiency"])

        # Skip invalid entries (efficiency = 0 is physically meaningless)
        if efficiency == 0:
            continue

        # Assign full capacity to a single unit
        storage_per_unit = storage_capacity_total
        power_per_unit = power_capacity_total

        # Generate unique identifier:
        # Format: <type>_<country>_<running index>
        uid = f"{row['type'].replace(' ', '_')}_{row['country']}_{global_counter}"
        global_counter += 1

        # Create BatteryUnit instance
        unit = BatteryUnit(
            uid=uid,
            type_=row["type"],
            country=row["country"],
            storage_capacity_MWh=storage_per_unit,
            power_capacity_MW=power_per_unit,
            efficiency=efficiency
        )

        # Append to result list
        unitsList.append(unit)

    # Validate uniqueness of all generated UIDs
    uids = [u.uid for u in unitsList]
    assert len(uids) == len(set(uids)), "Duplicate Battery Unit IDs detected!"

    return unitsList