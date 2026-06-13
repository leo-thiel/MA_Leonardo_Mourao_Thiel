# Master Thesis – System Inertia Analysis

This repository contains models, and results related to the
analysis of system inertia for the years 2024 and 2040.


---

## Author

Leonardo Gabriel Mourao Thiel (2396158)  
Karlsruhe Institute of Technology (KIT)  
Institute for Industrial Production (IIP)  
May 2026  

---
## Purpose

This repository enables:

- Reproduction of all model results  
- Analysis of system inertia under different scenarios  
- Comparison between present (2024) and future (2040) power systems  

## Repository Structure

### Thesis Document

- `latex/` – LaTeX source files (as .zip)  
- `pdf/` – Final thesis document  

---

### Data

Contains all input data required for the analysis.

Includes:

- generation capacity data  
- renewable generation profiles  
- demand time series  
- hydropower inflows  
- Net Transfer Capacities (NTC, based on TYNDP 2024)  

---

### Models

Python scripts and notebooks used for data processing and optimization.

#### Core Model

- `model/build_model.py` – Full optimization model (unit commitment + inertia)

#### Supporting Modules

- `inputs` – Input configuration and preprocessing  
- `thermal_units` – Thermal generation modeling  
- `battery_units` – Battery parameterization  
- `renewable_generation` – Renewable profiles  
- `wind_inertia` – Synthetic inertia estimation  
- `hydro_data` – Hydro inflows and inertia  
- `NTC` – Cross-border transmission capacities  
- `demand` – Demand profiles  

#### Execution

- `run_model.py` – Main script to run optimization scenarios  

#### Notebooks

- `2024_inertia` – Analysis for 2024  
- `2040_Model_inertia` – Optimization scenarios  
- `2040_plots` – Visualization of results  
- `generate_RE_production` – Renewable generation modeling  
- `capacity_table_24_40` – Capacity comparison  

---

### Results

Contains all output data and model results.

- `capacity_comparison.xlsx` – Capacity comparison (2024 vs 2040)

Subfolders:

- `2024/` – Results for 2024  
- `2040/` – Results for 2040  
- `gurobi_out/` – Optimization outputs (.sol, .log, .ilp)

#### Scenario outputs

Each scenario is solved separately:

- `no_inertia`  
- `thermal_only`  
- `thermal_plus_virtual`  

#### Time Horizon Runs

- `gurobi_out/2040_summer_01-07_to_31_08/`  
  → Results for full summer simulation (July–August 2040)

---
## Data Generation

Some input datasets (e.g. renewable generation profiles) were originally
generated using Jupyter notebooks such as:

- `generate_RE_production`

These notebooks are **not required to run the model**, as all processed
input data is already included in the repository (`/data`).

They are provided for transparency and reproducibility of the data
generation process.

## How to Run the Model

1. Install dependencies:
```bash
pip install -r requirements.txt
2. Ensure a valid Gurobi license is available
3. Run:python run_model.py

## Notes

- Optimization is performed using Gurobi.
- User must provide their own Gurobi license
- A valid Gurobi license (e.g. Web License Service or local license)
  must be configured via environment variables or a license file.
- The optimization model can be memory-intensive depending on the
  scenario size and solver configuration. In case of memory limitations,
  it is recommended to adjust solver parameters (e.g. presolve, node files,
  number of threads) or reduce the simulation horizon.
