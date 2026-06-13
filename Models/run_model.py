# =========================================================
# Author: Leonardo Gabriel Mourao Thiel
# Master Thesis – Energy System Inertia Optimization
# =========================================================

# Import Gurobi library
from gurobipy import Model, GRB, quicksum
from inputs import InputConfig, InputLoader
import os
import gurobipy as gp
from build_model import build_full_model
import pandas as pd

# =========================
# Gurobi Environment
# =========================

def create_gurobi_env():
    access_id = os.getenv("GUROBI_ACCESS_ID")

    if access_id:
        return gp.Env(params={
            "WLSACCESSID": access_id,
            "WLSSECRET": os.getenv("GUROBI_SECRET"),
            "LICENSEID": int(os.getenv("GUROBI_LICENSE_ID")),
        })
    else:
        return gp.Env()


# =========================
# Solver parameters
# =========================

def set_parameters(model, log_path):

    model.setParam("Threads", 8)
    model.setParam("MIPGap", 0.01)
    model.setParam("MIPFocus", 1)
    model.setParam("Presolve", 2)
    model.setParam("Heuristics", 0.2)
    model.setParam("NumericFocus", 1)

    model.setParam("Method", 0)
    model.setParam("NodeMethod", 1)

    model.setParam("LogFile", log_path)

    return model


# =========================
# Save results
# =========================

def save_results(model, out_path):
    if model.SolCount > 0:
        model.write(out_path + ".sol")
    elif model.status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
        model.computeIIS()
        model.write(out_path + ".ilp")


# =========================
# Main
# =========================

def main():
   


    # =========================
    # Scenario definitions
    # =========================

    SCENARIOS = [
        {"name": "thermal_plus_virtual", "calculate_inertia": True, "calculate_virtual_inertia": True},
        {"name": "no_inertia", "calculate_inertia": False, "calculate_virtual_inertia": False},
        {"name": "thermal_only", "calculate_inertia": True, "calculate_virtual_inertia": False},
    ]
    # -------------------------
    # PATH 
    # -------------------------
    path = "../data"

    # -------------------------
    # Countries
    # -------------------------
    countries = [
        "AL","AT","BA","BE","BG","CH","CZ","DE","DK","ES","FR","GR",
        "HR","HU","IT","LU","MK","ME","NL","PL","PT","RO","RS","SI","SK"
    ]

    # -------------------------
    # Time horizon
    # -------------------------
    start_date = "2040-07-01"
    end_date   = "2040-09-01"
   
    duration = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days  # Tage zwischen den Daten (approx)

    # -------------------------
    # Costs
    # -------------------------
    inertia_costs_battery = 2230 * (duration / 365)
    inertia_costs_solar_battery = 3220 * (duration / 365)
    inertia_costs_wind = 1670 * (duration / 365)
    inertia_costs_solar = 0

    # -------------------------
    # Load data
    # -------------------------
    cfg = InputConfig(path, start_date, end_date, countries)
    inputs = InputLoader.load(cfg)

    # -------------------------
    # Gurobi Env
    # -------------------------
    env = create_gurobi_env()

    # -------------------------
    # Output folder
    # -------------------------
    out_dir = "./results"
    os.makedirs(out_dir, exist_ok=True)

    # =========================
    # RUN SCENARIOS
    # =========================

    for sc in SCENARIOS:

        name = sc["name"]
        print(f"\n=== Running {name} ===")

        model = gp.Model(env=env)
        model = set_parameters(model, os.path.join(out_dir, f"{name}.log"))

        model = build_full_model(
            model,
            inputs,
            countries,
            inertia_costs_battery=inertia_costs_battery,
            inertia_costs_solar_battery=inertia_costs_solar_battery,
            inertia_costs_solar=inertia_costs_solar,
            inertia_costs_wind=inertia_costs_wind,
            calculate_inertia=sc["calculate_inertia"],
            calculate_virtual_inertia=sc["calculate_virtual_inertia"],
        )

        model.optimize()

        save_results(model, os.path.join(out_dir, name))

        model.dispose()


if __name__ == "__main__":
    main()