# =========================================================
# Author: Leonardo Gabriel Mourao Thiel
# Project: Master Thesis – System Inertia in the Energy System of the Future: Model-Based Cost Optimization to Secure Inertia Requirements
# Topic: InputModeling for optimization model
# Date: 27.04.2026
# =========================================================
"""
Input loading module for the energy system model.

Design goals
------------
- Clear separation between configuration, loading logic, and data container.
- Reproducibility: all model inputs are deterministically derived from InputConfig.
- Maintainability: avoids ambiguous tuple-based data passing.
"""

# =========================================================
# Standard library imports
# =========================================================

from dataclasses import dataclass
from typing import Dict, List
import pandas as pd

# =========================================================
# Project imports
# (Top-level imports improve testability and traceability)
# =========================================================

from wind_inertia import inertia_profiles
from NTC import NTCDatabase

from renewable_generation import (
    load_generation_profiles,
    clean_and_align_timestamps,
    compute_total_renewables,
    get_capacity,
)

from hydro_data import (
    load_hydro_profiles,
    filter_hydro_profiles,
    load_hydro_inertia,
)

from demand import load_hourly_demand
from battery_units import load_battery_units
from thermal_units import load_thermal_units
from dsr_units import load_dsr_units
from other_res_units import load_other_res_units
from non_res_units import load_nonres_units


# =========================================================
# Configuration container
# =========================================================

@dataclass(frozen=True)
class InputConfig:
    """
    Immutable configuration object defining all parameters required
    to generate model inputs.

    Design rationale
    ----------------
    - Strong typing improves readability and reduces runtime errors.
    - Centralized configuration enables reproducibility.
    - Immutability (frozen=True) prevents unintended side effects.

    Attributes
    ----------
    data_path : str
        Root directory containing all input datasets.
    start_date : str
        Start timestamp for time series filtering.
    end_date : str
        End timestamp for time series filtering.
    countries : list of str
        Countries included in the model scope.
    year : int, optional
        Scenario year (default: 2040).
    startup_costs_mapping : dict
        Mapping of technology → startup costs (EUR).
    marginal_costs : dict
        Mapping of technology → marginal costs (EUR/MWh).
    inertia_mapping : dict
        Mapping of technology → inertia constant H (seconds).
    """

    data_path: str
    start_date: str
    end_date: str
    countries: List[str]

    year: int = 2040

    startup_costs_mapping: Dict[str, float] = None
    marginal_costs: Dict[str, float] = None
    inertia_mapping: Dict[str, float] = None

    def __post_init__(self):
        """
        Post-initialization hook for setting default mappings.

        Because the dataclass is immutable, default dictionaries are
        assigned using object.__setattr__ only if not provided.

        This avoids shared mutable defaults across instances.
        """

        if self.startup_costs_mapping is None:
            object.__setattr__(self, "startup_costs_mapping", {
                "Gas CC": 132.98,
                "Gas CT": 63.89,
                "Nuclear": 152.20,
                "Lignite": 214.70,
                "Hard coal": 198.86,
                "Heavy oil": 233.32,
                "Light oil": 254.49,
                "Oil shale": 199.39,
                "Hydrogen": 209.21,
                "Biomass": 177.96,
                "Waste": 286.20
            })

        if self.marginal_costs is None:
            object.__setattr__(self, "marginal_costs", {
                "Gas CC": 68.49,
                "Gas CT": 94.93,
                "Lignite": 146.12,
                "Hard coal": 129.08,
                "Hydrogen": 95.37,
                "Nuclear": 18.55,
                "Heavy oil": 177.99,
                "Light oil": 195.99,
                "Oil shale": 149.14,
                "Biomass": 102.86,
                "Waste": 372.08
            })

        if self.inertia_mapping is None:
            object.__setattr__(self, "inertia_mapping", {
                "Gas CT": 4.2,
                "Gas CC": 4.2,
                "Nuclear": 5.9,
                "Lignite": 3.8,
                "Hard coal": 4.2,
                "Heavy oil": 4.3,
                "Light oil": 4.3,
                "Oil shale": 4.3,
                "Hydrogen": 4.2,
                "Biomass": 3.3,
                "Waste": 3.8,
                "Run-of-River": 2.7,
                "Pondage": 2.7,
                "Reservoir": 3.7,
                "PS Open": 3.5,
                "PS Closed": 3.5,
                "Other": 3.3
            })


# =========================================================
# Data container for model inputs
# =========================================================

@dataclass
class ModelInputs:
    """
    Structured container for all input data required by the optimization model.

    Design rationale
    ----------------
    - Eliminates ambiguous tuple unpacking.
    - Improves readability via named attributes.
    - Simplifies debugging and model extension.

    Notes
    -----
    - All attributes are expected to be fully preprocessed and aligned.
    - Time series should share a consistent temporal index.
    """

    countryList: List

    df_inertia_onshore: pd.DataFrame
    df_inertia_offshore: pd.DataFrame
    ntc: NTCDatabase

    df_total_renewable: pd.DataFrame
    df_roof: pd.DataFrame
    df_pv: pd.DataFrame

    capacity_roof: float
    capacity_pv: float
    capacity_onshore: float
    capacity_offshore: float

    df_hydro: pd.DataFrame
    hydro_inertia: pd.DataFrame

    df_load: pd.DataFrame

    battery_units: pd.DataFrame
    thermal_units: pd.DataFrame
    dsr_units: pd.DataFrame
    other_res_units: pd.DataFrame
    non_res_units: pd.DataFrame


# =========================================================
# Input loading logic
# =========================================================

class InputLoader:
    """
    Central orchestrator for assembling all model inputs.

    Design rationale
    ----------------
    - Implemented as a stateless loader (static method).
    - Ensures deterministic behaviour (pure function style).
    - Facilitates unit testing and reproducibility.
    """

    @staticmethod
    def load(cfg: InputConfig) -> ModelInputs:
        """
        Loads and assembles all input datasets based on the provided configuration.

        Parameters
        ----------
        cfg : InputConfig
            Configuration object defining model scope and parameters.

        Returns
        -------
        ModelInputs
            Fully populated input container for the energy system model.
        """

        # -------------------------------------------------
        # 1) Wind inertia profiles
        # -------------------------------------------------

        df_wind_inertia = inertia_profiles(
            cfg.data_path,
            cfg.start_date,
            cfg.end_date
        )

        df_inertia_onshore = df_wind_inertia["onshore"]
        df_inertia_offshore = df_wind_inertia["offshore"]

        # -------------------------------------------------
        # 2) Net Transfer Capacities (NTC)
        # -------------------------------------------------

        ntc_path = f"{cfg.data_path}/{cfg.year}/ntcs_countries_eraa2023.csv"
        ntc = NTCDatabase(ntc_path)

        # -------------------------------------------------
        # 3) Renewable generation
        # -------------------------------------------------

        df_dict = load_generation_profiles(cfg.data_path)

        # Ensure temporal consistency across all renewable datasets
        df_dict = clean_and_align_timestamps(df_dict)

        df_total_renewable, df_roof, df_pv = compute_total_renewables(
            df_dict,
            cfg.countries,
            cfg.start_date,
            cfg.end_date
        )

        # Installed capacities for normalization or scaling
        capacity_roof, capacity_pv, capacity_onshore, capacity_offshore = \
            get_capacity(cfg.data_path)

        # -------------------------------------------------
        # 4) Hydro generation and inertia
        # -------------------------------------------------

        df_hydro_raw = load_hydro_profiles(cfg.data_path)

        df_hydro = filter_hydro_profiles(
            df_hydro_raw,
            cfg.countries,
            cfg.start_date,
            cfg.end_date
        )

        hydro_inertia = load_hydro_inertia(
            cfg.data_path,
            cfg.countries,
            cfg.inertia_mapping,
            df_hydro,
            year=cfg.year
        )

        # -------------------------------------------------
        # 5) Demand and flexibility resources
        # -------------------------------------------------

        df_load = load_hourly_demand(
            cfg.data_path,
            cfg.countries,
            cfg.start_date,
            cfg.end_date
        )

        battery_units = load_battery_units(cfg.data_path, cfg.countries)

        thermal_units = load_thermal_units(
            cfg.data_path,
            cfg.countries,
            cfg.startup_costs_mapping,
            cfg.marginal_costs,
            cfg.inertia_mapping
        )

        dsr_units = load_dsr_units(
            cfg.data_path,
            cfg.countries,
            cfg.start_date,
            cfg.end_date
        )

        other_res_units = load_other_res_units(
            cfg.data_path,
            cfg.countries,
            cfg.startup_costs_mapping,
            cfg.marginal_costs,
            cfg.inertia_mapping
        )

        non_res_units = load_nonres_units(
            cfg.data_path,
            cfg.countries,
            cfg.start_date,
            cfg.end_date
        )

        # -------------------------------------------------
        # Return structured input object
        # -------------------------------------------------

        return ModelInputs(
            countryList=cfg.countries,
            df_inertia_onshore=df_inertia_onshore,
            df_inertia_offshore=df_inertia_offshore,
            ntc=ntc,
            df_total_renewable=df_total_renewable,
            df_roof=df_roof,
            df_pv=df_pv,
            capacity_roof=capacity_roof,
            capacity_pv=capacity_pv,
            capacity_onshore=capacity_onshore,
            capacity_offshore=capacity_offshore,
            df_hydro=df_hydro,
            hydro_inertia=hydro_inertia,
            df_load=df_load,
            battery_units=battery_units,
            thermal_units=thermal_units,
            dsr_units=dsr_units,
            other_res_units=other_res_units,
            non_res_units=non_res_units
        )