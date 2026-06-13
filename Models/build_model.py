# =========================================================
# Author: Leonardo Gabriel Mourao Thiel
# Project: Master Thesis – System Inertia in the Energy System of the Future: Model-Based Cost Optimization to Secure Inertia Requirements
# Topic: build Optimization model for future energy system
# Date: 27.04.2026
# =========================================================
import gurobipy as gp
from gurobipy import GRB
import pandas as pd


def build_full_model(
    m,
    inputs,
    countries,
    inertia_costs_battery=0.0,
    inertia_costs_solar_battery=0.0,
    inertia_costs_solar=0.0,
    inertia_costs_wind=0.0,
    calculate_virtual_inertia=True,
    calculate_inertia=True,
    RoCoFmax=1,
    fn=50,
    t_krit=1,
    slack_costs=400,
    slack_costs_inertia=500
):
    """
    Builds a full mixed-integer optimization model (unit commitment + dispatch)
    with optional inertia constraints using Gurobi.

    The model integrates multiple system components:
    - Thermal generation (with unit commitment)
    - Renewable generation (exogenous profiles)
    - Battery storage (with SOC dynamics)
    - Demand-side response (DSR)
    - Cross-border transmission (NTC-based)
    - System inertia (physical and virtual)

    Parameters
    ----------
    m : gurobipy.Model
        Pre-initialized Gurobi model instance.

    inputs : object
        Container holding all model input data (units, time series, parameters).

    countries : list of str
        Set of countries included in the model (nodes of the system).

    inertia_costs_* : float
        Cost coefficients for providing virtual inertia via different technologies
        (battery, solar, wind). Used in objective function.

    calculate_virtual_inertia : bool
        If True, enables endogenous decision variables for virtual inertia provision.

    calculate_inertia : bool
        If True, enforces system inertia constraints (frequency stability proxy).

    RoCoFmax : float
        Maximum allowed Rate of Change of Frequency (currently not explicitly used,
        but conceptually linked to inertia requirement).

    fn : float
        Nominal system frequency (Hz), typically 50 Hz in European systems.

    t_krit : float
        Critical time parameter (seconds), used for inertia approximation.

    slack_costs : float
        Penalty cost for violating NTC constraints (external exchanges).

    slack_costs_inertia : float
        Penalty cost for violating inertia constraints (feasibility slack).

    Returns
    -------
    gurobipy.Model
        Fully constructed optimization model.

    Notes
    -----
    - Curtailment: involuntary reduction (no inertia contribution)
    - Deloading: intentional reserve provision (used for inertia)
    - The model is formulated as a MIQP (.
    - Unit commitment decisions are binary (on/off).
    - Inertia constraints approximate frequency stability requirements.
    - Slack variables ensure feasibility but are penalized in the objective.
    """
     # =========================================================
    # 0) Time axis / hour index from df_load
    # =========================================================

    df_load = inputs.df_load.copy()
    ts_col = df_load.columns[0]

    # Ensure timestamps are real datetimes
    df_load[ts_col] = pd.to_datetime(df_load[ts_col], errors="coerce")
    df_load = df_load.dropna(subset=[ts_col]).reset_index(drop=True)

    # Define horizon directly from df_load 
    start_time = df_load[ts_col].min()
    end_time = df_load[ts_col].max()

    # Create hour_id starting at 1 
    df_load["hour_id"] = ((df_load[ts_col] - start_time) / pd.Timedelta(hours=1)).astype(int) + 1
    hours = sorted(df_load["hour_id"].unique().tolist())




    # Map hour_id -> timestamp (nice for post-processing)
    hour_to_ts = dict(zip(df_load["hour_id"], df_load[ts_col]))

  

    # =========================================================
    # 1) Unit lists (objects) and grouping by country
    # =========================================================

    thermal_units = list(getattr(inputs, "thermal_units", []) or [])
    battery_units = list(getattr(inputs, "battery_units", []) or [])
    other_res_units = list(getattr(inputs, "other_res_units", []) or [])
    dsr_units = list(getattr(inputs, "dsr_units", []) or [])
    non_res_units = list(getattr(inputs, "non_res_units", []) or [])
    # Grouping speeds up building constraints (avoid scanning full list repeatedly)
    thermal_by_country = {c: [] for c in countries}
    battery_by_country = {c: [] for c in countries}
    other_res_by_country = {c: [] for c in countries}
    dsr_by_country = {c: [] for c in countries}
    non_res_by_country = {c: [] for c in countries}
    for u in thermal_units:
        if getattr(u, "country", None) in thermal_by_country:
            thermal_by_country[u.country].append(u)
    for b in battery_units:
        if getattr(b, "country", None) in battery_by_country:
            battery_by_country[b.country].append(b)

    for r in other_res_units:
        if getattr(r, "country", None) in other_res_by_country:
            other_res_by_country[r.country].append(r)

    for d in dsr_units:
        if getattr(d, "country", None) in dsr_by_country:
            dsr_by_country[d.country].append(d)
    for u in non_res_units:
        if u.country in non_res_by_country:
            non_res_by_country[u.country].append(u)
    # =========================================================
    # 2) Create model and decision variables
    # =========================================================

    # Main generation variable for *all* dispatchable units
    p = {}

    # Commitment + start for thermal units
    y = {}
    start = {}

    # ---------------------------
    # 2.1 Thermal variables
    # ---------------------------
    for u in thermal_units:
        uid = str(u.uid)
        cap = float(getattr(u, "capacity_MW", 0.0) or 0.0)

        for h in hours:
            p[(uid, h)] = m.addVar(lb=0.0, ub=cap, vtype=GRB.CONTINUOUS, name=f"p_{uid}_{h}")
            y[(uid, h)] = m.addVar(vtype=GRB.BINARY, name=f"y_{uid}_{h}")
            start[(uid, h)] = m.addVar(vtype=GRB.BINARY, name=f"start_{uid}_{h}")

    # ---------------------------
    # 2.2 Other RES variables (dispatch limited by installed capacity)
    # ---------------------------
    
    for r in other_res_units:
        uid = str(r.uid)
        cap = float(getattr(r, "capacity_MW", 0.0) or 0.0)
        for h in hours:
            p[(uid, h)] = m.addVar(lb=0.0, ub=cap, vtype=GRB.CONTINUOUS, name=f"p_{uid}_{h}")
            y[(uid, h)] = m.addVar(vtype=GRB.BINARY, name=f"y_{uid}_{h}")
            start[(uid, h)] = m.addVar(vtype=GRB.BINARY, name=f"start_{uid}_{h}")
    # ---------------------------
    # 2.3 Battery variables
    # ---------------------------
    battery_charge = {}
    battery_discharge = {}
    battery_soc = {}

    for b in battery_units:
        bid = str(b.uid)
        p_cap = float(getattr(b, "power_capacity_MW", 0.0) or 0.0)
        e_cap = float(getattr(b, "storage_capacity_MWh", 0.0) or 0.0)

        for h in hours:
            battery_charge[(bid, h)] = m.addVar(lb=0.0, ub=p_cap, vtype=GRB.CONTINUOUS, name=f"charge_{bid}_{h}")
            battery_discharge[(bid, h)] = m.addVar(lb=0.0, ub=p_cap, vtype=GRB.CONTINUOUS, name=f"discharge_{bid}_{h}")
            battery_soc[(bid, h)] = m.addVar(lb=0.0, ub=e_cap, vtype=GRB.CONTINUOUS, name=f"soc_{bid}_{h}")

    # ---------------------------
    # 2.4 DSR variables 
    # ---------------------------
    dsr = {}
    for d in dsr_units:
        did = str(d.uid)
        # simplest assumption: constant cap
        for h in hours:
            cap = d.schedule.iloc[h]
            dsr[(did, h)] = m.addVar(lb=0.0, ub=cap, vtype=GRB.CONTINUOUS, name=f"dsr_{did}_{h}")

    m.update()
    # ---------------------------
    # 2.4 non_res variables
    # ---------------------------
    
    p_non_res = {}

    for u in non_res_units:
        uid = str(u.uid)

        for h in hours:
            ts = hour_to_ts[h]

            cap = u.schedule.get(ts, 0.0)

            p_non_res[(uid,h)] = m.addVar(
                lb=0,
                ub=cap,
                vtype=GRB.CONTINUOUS,
                name=f"p_non_res_{uid}_{h}"
        )
    # =========================================================
    # 3) Thermal constraints: start + minimum stable power
    # =========================================================

    # 3.1 start logic:
    # start[h] >= y[h] - y[h-1]
    # start[1] == y[1] (assume unit was off initially)
    for u in thermal_units:
        uid = str(u.uid)
        for idx, h in enumerate(hours):
            if idx == 0:
                m.addConstr(start[(uid, h)] == y[(uid, h)], name=f"start_init_{uid}_{h}")
            else:
                m.addConstr(start[(uid, h)] >= y[(uid, h)] - y[(uid, hours[idx - 1])], name=f"start_min_{uid}_{h}")
                m.addConstr(start[(uid, h)] <= (y[(uid, h)]), name=f"start_constr_{uid}_{h}")


    # 3.2 Minimum stable power:
    # if y=1 => p >= Pmin, if y=0 => p >= 0
    for u in thermal_units:
        uid = str(u.uid)
        pmin = float(getattr(u, "min_stable_power_MW", 0.0) or 0.0)
        cap=float(getattr(u, "capacity_MW", 0.0) or 0.0)
        for h in hours:
            m.addConstr(p[(uid, h)] >= pmin * y[(uid, h)], name=f"p_min_{uid}_{h}")
            m.addConstr(p[(uid,h)] <= cap * y[(uid,h)],name=f"p_cap_{uid}_{h}")
    # =========================================================
    # 4) Other_RES constraints: start + minimum stable power
    # =========================================================        
    # 4.1 start logic:
    # start[h] >= y[h] - y[h-1]
    # start[1] == y[1] (assume unit was off initially)
    for u in other_res_units:
        uid = str(u.uid)
        for idx, h in enumerate(hours):
            if idx == 0:
                m.addConstr(start[(uid, h)] == y[(uid, h)], name=f"start_init_{uid}_{h}")
            else:
                h_prev = hours[idx - 1]
                m.addConstr(start[(uid, h)] >= y[(uid, h)] - y[(uid, h_prev)], name=f"start_min_{uid}_{h}")
                m.addConstr(start[(uid, h)] <= (y[(uid, h)]), name=f"start_constr_{uid}_{h}")


    # 4.2 Minimum stable power:
    # if y=1 => p >= Pmin, if y=0 => p >= 0
    for u in other_res_units:
        uid = str(u.uid)
        cap=float(getattr(u, "capacity_MW", 0.0) or 0.0)
        pmin =float(getattr(u, "min_stable_power_MW", 0.0) or 0.0)
        for h in hours:
            m.addConstr(p[(uid, h)] >= pmin * y[(uid, h)], name=f"p_min_{uid}_{h}")
            m.addConstr(p[(uid,h)] <= cap * y[(uid,h)],name=f"p_cap_{uid}_{h}")
    # =========================================================
    # 5) Battery SOC constraints
    # =========================================================
    b_mode = {}

    for b in battery_units:
        bid = str(b.uid)
        eta = float(getattr(b, "efficiency", 1.0) or 1.0)
        p_cap = float(getattr(b, "power_capacity_MW", 0.0) or 0.0)

        for idx, h in enumerate(hours):

            b_mode[(bid, h)] = m.addVar(vtype=GRB.BINARY)
    
            m.addConstr(battery_charge[(bid, h)] <= p_cap * b_mode[(bid, h)])
            m.addConstr(battery_discharge[(bid, h)] <= p_cap * (1 - b_mode[(bid, h)]))
            if idx == 0:
                # initial SOC assumed 0 (extendable)
                m.addConstr(
                    battery_soc[(bid, h)] == eta * battery_charge[(bid, h)] - battery_discharge[(bid, h)] / eta,
                    name=f"soc_init_{bid}_{h}"
                )
            else:
                h_prev = hours[idx - 1]
                m.addConstr(
                    battery_soc[(bid, h)] == battery_soc[(bid, h_prev)] + eta * battery_charge[(bid, h)] - battery_discharge[(bid, h)] / eta,
                    name=f"soc_{bid}_{h}"
                )

    # =========================================================
    # 6) NTC flows (internal) + external slack flows
    # =========================================================

    db_ntc = inputs.ntc

    flow = {}
    slack_ntc = {}
    handled = set()

    for a in db_ntc.countries():
        for b in db_ntc.ntc[a]:
            ntc_ab = float(db_ntc.ntc[a][b])

            # internal line: a and b are in the modeled region
            if a in countries and b in countries:
                pair = tuple(sorted([a, b]))
                if pair in handled:
                    continue
                handled.add(pair)

                for h in hours:
                    flow[(a, b, h)] = m.addVar(lb=-ntc_ab, ub=ntc_ab, vtype=GRB.CONTINUOUS, name=f"flow_{a}_{b}_{h}")
                continue

            # export to outside (a in model, b outside)
            if a in countries and b not in countries:
                pair = tuple(sorted([a, b]))
                if pair in handled:
                    continue
                handled.add(pair)
                for h in hours:
                    slack_ntc[(a, b, h)] = m.addVar(lb=-ntc_ab, ub=0.0, vtype=GRB.CONTINUOUS, name=f"slack_ntc_{a}_{b}_{h}")
                continue

            # import from outside (a outside, b in model)
            if a not in countries and b in countries:
                pair = tuple(sorted([a, b]))
                if pair in handled:
                    continue
                handled.add(pair)
                for h in hours:
                    slack_ntc[(a, b, h)] = m.addVar(lb=0.0, ub=ntc_ab, vtype=GRB.CONTINUOUS, name=f"slack_ntc_{a}_{b}_{h}")
                continue

    # Build inflow/outflow lists for each country/hour
    inflows = {(c, h): [] for c in countries for h in hours}
    outflows = {(c, h): [] for c in countries for h in hours}

    for (src, dest, h), var in flow.items():
        if dest in countries:
            inflows[(dest, h)].append(var)
        if src in countries:
            outflows[(src, h)].append(var)
    m.update()   

    # Curtailment variable:
    # represents involuntary reduction of renewable generation
    # used to ensure feasibility if supply exceeds demand
    # not technology-specific → aggregate curtailment proxy
    # NOT contributing to inertia (unlike deloading)    
    curtailement = {}
    for c in countries:
        for h in hours:
            curtailement[(c, h)] = m.addVar(lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"curtailement_{c}_{h}")

    # =========================================================
    # 7) Inertia variables (optional)
    # =========================================================
    inertia_total = {}
    if calculate_inertia:

        perct_roof_battery = {}
        perct_pv_battery = {}
        perct_pv_deloading = {}
        perct_roof_deloading = {}

        perct_battery = {}
        perct_onshore = {}
        perct_offshore = {}

        inertia_battery = {}
        inertia_pv = {}
        inertia_pv_battery = {}
        inertia_roof = {}
        inertia_roof_battery = {}
        inertia_onshore = {}
        inertia_offshore = {}
        inertia_non_virtual = {}
        inertia_virtual = {}
        inertia_total = {}
        slack_inertia= {}
        deloading_roof = {}
        deloading_pv = {}
        deloading_roof_battery = {}
        deloading_pv_battery = {}
    for c in countries:

        # country-level shares (0..1)
        if calculate_virtual_inertia:

            perct_roof_battery[c] = m.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"percentage_rooftop_battery_{c}")
            perct_pv_battery[c] = m.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"percentage_pv_battery_{c}")
            perct_pv_deloading[c] = m.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"percentage_pv_deloading_{c}")
            perct_roof_deloading[c] = m.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"percentage_rooftop_deloading_{c}")
            perct_battery[c] = m.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"percentage_battery_{c}")
            perct_onshore[c] = m.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"percentage_onshore_{c}")
            perct_offshore[c] = m.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"percentage_offshore_{c}")
            m.addConstr(perct_pv_deloading[c]<=(1-perct_pv_battery[c]))
            m.addConstr(perct_roof_deloading[c]<=(1-perct_roof_battery[c]))
        for h in hours:
            inertia_total[(c, h)] = m.addVar(lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"inertia_total_{c}_{h}")
            if calculate_inertia:
                inertia_non_virtual[(c, h)] = m.addVar(lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"inertia_non_virtual_{c}_{h}")
                slack_inertia[(c, h)] = m.addVar(lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"slack_inertia_{c}_{h}")

                if calculate_virtual_inertia:

                    # hourly inertia values from different virtual sources
                    inertia_virtual[(c, h)] = m.addVar(lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"inertia_virtual_{c}_{h}")
                    inertia_battery[(c, h)] = m.addVar(lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"inertia_battery_{c}_{h}")
                    inertia_pv[(c, h)] = m.addVar(lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"inertia_pv_{c}_{h}")
                    inertia_pv_battery[(c, h)] = m.addVar(lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"inertia_pv_batt_{c}_{h}")
                    inertia_roof[(c, h)] = m.addVar(lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"inertia_roof_{c}_{h}")
                    inertia_roof_battery[(c, h)] = m.addVar(lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"inertia_roof_batt_{c}_{h}")
                    inertia_onshore[(c, h)] = m.addVar(lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"inertia_onshore_{c}_{h}")
                    inertia_offshore[(c, h)] = m.addVar(lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"inertia_offshore_{c}_{h}")

                    # deloading factors 0..1 
                    deloading_roof[(c, h)] = m.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"deload_roof_{c}_{h}")
                    deloading_pv[(c, h)] = m.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"deload_pv_{c}_{h}")
                    deloading_roof_battery[(c, h)] = m.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"deload_roof_batt_{c}_{h}")
                    deloading_pv_battery[(c, h)] = m.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"deload_pv_batt_{c}_{h}")


        m.update()

    # =========================================================
    # 8) Demand balance + inertia constraints
    # =========================================================

    df_hydro = getattr(inputs, "df_hydro", None)
    hydro_inertia_df = inputs.hydro_inertia
    
    for _, row in df_load.iterrows():
        h = int(row["hour_id"])
        ts = row[ts_col]

        for c in countries:
            # 8.1 Demand (MW)
            total_demand = float(row[c])

            # 8.2 Thermal generation sum + other_RES
            gen_thermal = gp.quicksum(p[(str(u.uid), h)] for u in thermal_by_country[c])+gp.quicksum(p[(str(r.uid), h)] for r in other_res_by_country[c])

        
            
            # 8.4 Non RES dispatch sum

            gen_non_res = gp.quicksum(
                    p_non_res[(str(u.uid),h)]
                    for u in non_res_by_country[c]
            )
            # 8.4 Battery net injection
            batt_net = gp.quicksum(
                battery_discharge[(str(b.uid), h)] - battery_charge[(str(b.uid), h)]
                for b in battery_by_country[c]
            )
            batt_charge=gp.quicksum(
                battery_charge[(str(b.uid), h)]
                for b in battery_by_country[c]
            )
            # 8.5 DSR demand reduction
            dsr_red = gp.quicksum(
                dsr[(str(d.uid), h)]
                for d in dsr_by_country[c]
            )        
            # 8.6 NTC net imports
            inflow = gp.quicksum(inflows[(c, h)])
            outflow = gp.quicksum(outflows[(c, h)])

            # 8.7 External slack imports/exports
            slack_import = gp.quicksum(
                slack_ntc[(src, dest, h)]
                for (src, dest, _h) in slack_ntc
                if dest == c and _h == h
            ) - gp.quicksum(
                slack_ntc[(src, dest, h)]
                for (src, dest, _h) in slack_ntc
                if src == c and _h == h
            )

            # 8.8 Hydro generation (supports dict-of-dfs or single df)
            total_hydro = 0.0
            if isinstance(df_hydro, dict):
                for _, df_h in df_hydro.items():
                    if c in df_h.columns:
                        total_hydro += float(df_h.fillna(0.0).iloc[h - 1][c])
            elif isinstance(df_hydro, pd.DataFrame):
                if c in df_hydro.columns:
                    total_hydro = float(df_hydro.fillna(0.0).iloc[h - 1][c])

            # 8.9 Renewable generation + hydro
            total_renewable = float(inputs.df_total_renewable.loc[h-1, c])
            deload_solar=0
            # Deloading of RES:
            # fraction of available renewable generation intentionally not injected
            # creates upward reserve (headroom) for virtual inertia provision            
            if calculate_virtual_inertia:
                pv_gen = float(inputs.df_pv.iloc[h - 1][c]) if c in inputs.df_pv.columns else 0.0
                roof_gen = float(inputs.df_roof.iloc[h - 1][c]) if c in inputs.df_roof.columns else 0.0
                if pv_gen != 0:
                    deload_solar = pv_gen*deloading_pv[(c, h)]* perct_pv_deloading[c]+pv_gen* deloading_pv_battery[(c, h)]*perct_pv_battery[c]
                if roof_gen != 0:
                    deload_solar += roof_gen*deloading_roof[(c, h)]* perct_roof_deloading[c]+roof_gen* deloading_roof_battery[(c, h)]*perct_roof_battery[c]


            # Power balance:
            # Supply (generation + imports + storage + DSR)
            # minus withheld renewable energy (deloading)
            # minus curtailed energy
            # must equal demand in each country and hour
            #
            # Note:
            # - Deloading = intentional withholding for inertia provision
            # - Curtailment = forced reduction due to system constraints            
            m.addConstr(
                gen_thermal  + gen_non_res + total_hydro
                + total_renewable
                + batt_net + dsr_red
                + inflow - outflow + slack_import
                - deload_solar-curtailement[(c, h)]
                == total_demand,
                name=f"demand_balance_{c}_{h}"
               )

            # =====================================================
            # Inertia constraints
            # =====================================================

            virtual_inertia=0
            # Grid-related inertia proxy (NBM):
            # approximates contribution of network elements
            # proportional to system load (constant H = 0.23 s)            
            inertia_vink=total_demand*0.23
            inertia_thermal = gp.quicksum(
                    float(u.inertia) * y[(str(u.uid), h)]
                    for u in thermal_by_country[c]
                )
             # Hydro inertia from table
            inertia_hydro = float(hydro_inertia_df.iloc[h - 1][c]) if c in hydro_inertia_df.columns else 0.0
            # Other RES inertia:
            inertia_other = gp.quicksum(
                float(getattr(r, "inertia", 0.0) or 0.0)
                * y[(str(r.uid), h)]
                for r in other_res_by_country[c])

            if calculate_virtual_inertia:

                # Battery headroom approximation:
                # available upward flexibility = unused discharge capacity
                # + potential reduction of charging
                # assumes sufficient state of charge (no energy limitation modeled)                
                cap_batt = sum(float(getattr(b, "power_capacity_MW", 0.0) or 0.0) for b in battery_by_country[c])
                dis_sum = gp.quicksum(battery_discharge[(str(b.uid), h)] for b in battery_by_country[c])
                ch_sum = gp.quicksum(battery_charge[(str(b.uid), h)] for b in battery_by_country[c])

                headroom_up_batt = cap_batt - dis_sum + ch_sum
                m.addConstr(
                    inertia_battery[(c, h)] == headroom_up_batt * t_krit * perct_battery[c],
                    name=f"link_inertia_batt_{c}_{h}"
                )

                # PV and rooftop inertia 
                cap_pv = inputs.capacity_pv.get(c)
                cap_roof = inputs.capacity_roof.get(c)
                headroom_pv_batt = cap_pv 
                # PV-battery headroom approximation:
                # = installed PV capacity minus effective generation after deloading
                # represents controllable upward flexibility
                #
                # Important modeling assumption:
                # Curtailment is NOT included in headroom,
                # i.e. only actively controllable reserve is considered
                # → conservative estimate of available inertia
                if pv_gen != 0:
                    # Headroom from PV deloading:
                    # proportional to current generation level
                    # represents immediately activatable reserve
                    headroom_pv = pv_gen * deloading_pv[(c, h)]
                    m.addConstr(inertia_pv[(c, h)] == headroom_pv * t_krit * perct_pv_deloading[c], name=f"link_inertia_pv_{c}_{h}")
                    headroom_pv_batt = cap_pv - (1 - deloading_pv_battery[(c, h)]) * pv_gen

                m.addConstr(inertia_pv_battery[(c, h)] == headroom_pv_batt * t_krit * perct_pv_battery[c], name=f"link_inertia_pv_batt_{c}_{h}")
                headroom_roof_batt=cap_roof
                if roof_gen != 0:
                    headroom_roof = roof_gen * deloading_roof[(c, h)]
                    m.addConstr(inertia_roof[(c, h)] == headroom_roof * t_krit * perct_roof_deloading[c], name=f"link_inertia_roof_{c}_{h}")
                    headroom_roof_batt = cap_roof - (1 - deloading_roof_battery[(c, h)]) * roof_gen



                m.addConstr(inertia_roof_battery[(c, h)] == headroom_roof_batt * t_krit * perct_roof_battery[c], name=f"link_inertia_roof_batt_{c}_{h}")
                # Wind inertia potentials from inputs
                onshore_val = float(inputs.df_inertia_onshore.iloc[h - 1][c]) if c in inputs.df_inertia_onshore.columns else 0.0
                offshore_val = float(inputs.df_inertia_offshore.iloc[h - 1][c]) if c in inputs.df_inertia_offshore.columns else 0.0
                # Wind inertia is exogenously estimated (from preprocessing)
                # scaled by decision variable representing activation share    
                m.addConstr(inertia_onshore[(c, h)] == perct_onshore[c] * onshore_val, name=f"link_inertia_onshore_{c}_{h}")
                m.addConstr(inertia_offshore[(c, h)] == perct_offshore[c] * offshore_val, name=f"link_inertia_offshore_{c}_{h}")

                # Total virtual inertia:
                # sum of contributions from all technologies
                # (battery, PV, PV-battery, rooftop, wind)                
                virtual_inertia = gp.quicksum([
                    inertia_battery[(c, h)],
                    inertia_pv[(c, h)],
                    inertia_pv_battery[(c, h)],
                    inertia_roof[(c, h)],
                    inertia_roof_battery[(c, h)],
                    inertia_onshore[(c, h)],
                    inertia_offshore[(c, h)],
                ])




            if calculate_inertia:
                  # calculate non_virtual distribution (thermal+other_RES+Hydro+vink)


                m.addConstr(
                    inertia_non_virtual[(c, h)] == inertia_thermal + inertia_other+inertia_hydro+inertia_vink,
                    name=f"link_inertia_non_virtual_{c}_{h}"
                )
                # Minimum inertia requirement:
                # proportional to demand (H = 2 s)
                # based on ENTSO-E guidelines for frequency stability
                inertia_required = 2.0 * total_demand
          # virtual inertia contribution calculated above

                if calculate_virtual_inertia:
                    m.addConstr(
                        inertia_virtual[(c, h)] ==  virtual_inertia,
                        name=f"virtual_inertia_total_{c}_{h}"
                    )
                # total inertia non_virtual + (virtual if calcualted)

                m.addConstr(
                    inertia_total[(c, h)] == inertia_non_virtual[(c, h)] + virtual_inertia,
                    name=f"link_inertia_total_{c}_{h}"
                )
                # Inertia constraint with slack:
                # ensures feasibility even if requirement is violated
                # slack is heavily penalized in objective
                m.addConstr(
                    inertia_total[(c, h)]+slack_inertia[(c, h)] >= inertia_required,
                    name=f"inertia_req_{c}_{h}"
                )
            else:
                #(calcualte inertia just for comparion without requirements)
                 m.addConstr(
                        inertia_total[(c, h)] == inertia_thermal + inertia_other+inertia_hydro+inertia_vink,
                        name=f"inertia_req_{c}_{h}"
                    )

                  
                    
    # =========================================================
    # 9) Objective function
    # =========================================================

    # 9.1 Fuel costs (thermal + other RES if they have marginal_cost)
    fuel_costs = gp.quicksum(
        float(getattr(u, "marginal_cost", 0.0) or 0.0) * p[(str(u.uid), h)]
        for u in (thermal_units + other_res_units)
        for h in hours
        if (str(u.uid), h) in p
    )
    non_res_costs = gp.quicksum(
        u.costs * p_non_res[(str(u.uid),h)]
        for u in non_res_units
        for h in hours
    )
    # 9.2 start costs (thermal + other_RES)
    start_costs = gp.quicksum(
        float(getattr(u, "start_costs", 0.0) or 0.0) * start[(str(u.uid), h)]
        for u in (thermal_units + other_res_units)
        for h in hours
    )

    # 9.3 DSR costs (optional)
    dsr_costs = gp.quicksum(
        float(getattr(d, "costs", 0.0) or 0.0) * dsr[(str(d.uid), h)]
        for d in dsr_units
        for h in hours
    ) 
    # 9.4 Slack NTC costs (penalize absolute slack)
    slack_ntc_costs = gp.quicksum(
        slack_costs * (slack_ntc[(a, b, h)] if slack_ntc[(a, b, h)].LB >= 0 else -slack_ntc[(a, b, h)])
        for (a, b, h) in slack_ntc
    )

    # 9.5 Slack inertia costs (only if inertia is active)
    if calculate_inertia:
        slack_inertia_costs = gp.quicksum(
            slack_costs_inertia * slack_inertia[(c, h)]
            for c in countries
            for h in hours
        )
    else:
        slack_inertia_costs = 0.0

    # 9.6 Virtual inertia “investment” costs
    if calculate_virtual_inertia:
        virtual_inertia_costs = gp.quicksum(
            perct_roof_battery[c] * inputs.capacity_roof.get(c) * inertia_costs_solar_battery
            + perct_pv_battery[c] * inputs.capacity_pv.get(c)* inertia_costs_solar_battery
            + perct_roof_deloading[c] * inputs.capacity_roof.get(c) * inertia_costs_solar
            + perct_pv_deloading[c] * inputs.capacity_pv.get(c)* inertia_costs_solar
            + perct_battery[c] * sum(float(getattr(b, "power_capacity_MW") or 0.0) for b in battery_by_country[c]) * inertia_costs_battery
            + perct_onshore[c] * inputs.capacity_onshore.get(c) * inertia_costs_wind
            + perct_offshore[c] * inputs.capacity_offshore.get(c) * inertia_costs_wind
            for c in countries
        )
    else:
        virtual_inertia_costs = 0.0
    # 9.7 final costs (minimize)

    final_cost_target = (
        fuel_costs
        + start_costs
        + dsr_costs
        + slack_ntc_costs
        + slack_inertia_costs
        + virtual_inertia_costs
        +non_res_costs
    )

    m.setObjective(final_cost_target, GRB.MINIMIZE)

    return m
