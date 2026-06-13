# =========================================================
# Author: Leonardo Gabriel Mourao Thiel
# Project: Master Thesis – System Inertia in the Energy System of the Future: Model-Based Cost Optimization to Secure Inertia Requirements
# Topic: Parameterization of NTC Database
# Date: 27.04.2026
# =========================================================
import pandas as pd


class NTCDatabase:
    """
    Represents a Net Transfer Capacity (NTC) database for cross-border transmission.

    The class reads bilateral transfer capacities from a CSV file and stores them
    in adjacency-like dictionaries for fast lookup.

    Attributes
    ----------
    df : pandas.DataFrame
        Raw input data containing NTC values and metadata.
    ntc : dict
        Nested dictionary storing transfer capacities:
        ntc[country_a][country_b] = capacity (MW)
    line_type : dict
        Nested dictionary storing connection types:
        line_type[country_a][country_b] = type (e.g., AC, DC)
    """

    def __init__(self, csv_path: str):
        """
        Initializes the NTCDatabase by loading and processing the CSV file.

        Parameters
        ----------
        csv_path : str
            Path to the CSV file containing NTC data.
        """
        # Load raw dataset (semicolon-separated format assumed)
        self.df = pd.read_csv(csv_path, sep=";")

        # Initialize adjacency structures
        self.ntc = {}
        self.line_type = {}

        # Build internal data structures
        self._build()

    def _build(self):
        """
        Constructs adjacency dictionaries for NTC values and line types.

        Notes
        -----
        - Connections are treated as bidirectional and symmetric.
        - Each country pair is stored twice (A→B and B→A).
        - Assumes NTC values are identical in both directions.
        """

        for _, row in self.df.iterrows():

            # Extract connection endpoints and attributes
            a = row["country_from"]
            b = row["country_to"]
            t = row["type"]
            cap = int(row["ntc"])

            # Ensure dictionary entries exist
            self.ntc.setdefault(a, {})
            self.ntc.setdefault(b, {})
            self.line_type.setdefault(a, {})
            self.line_type.setdefault(b, {})

            # Store bidirectional NTC values
            self.ntc[a][b] = cap
            self.ntc[b][a] = cap

            # Store line type (e.g., AC or DC)
            self.line_type[a][b] = t
            self.line_type[b][a] = t

    def get_ntc(self, a: str, b: str) -> int | None:
        """
        Retrieves the Net Transfer Capacity between two countries.

        Parameters
        ----------
        a : str
            Origin country.
        b : str
            Destination country.

        Returns
        -------
        int or None
            NTC value in MW if available, otherwise None.
        """
        return self.ntc.get(a, {}).get(b)

   
    def countries(self):
        """
        Returns all countries present in the NTC database.

        Returns
        -------
        list of str
            Sorted list of country identifiers.
        """
        return sorted(self.ntc.keys())

    def connections(self):
        """
        Returns all bilateral connections with their NTC values.

        Returns
        -------
        dict
            Dictionary mapping (country_a, country_b) → NTC (MW).

        Notes
        -----
        - Includes both directions (A→B and B→A).
        - May contain duplicate logical edges due to bidirectional storage.
        """
        return {
            (a, b): self.ntc[a][b]
            for a in self.ntc
            for b in self.ntc[a]
        }