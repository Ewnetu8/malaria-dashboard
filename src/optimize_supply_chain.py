"""
Supply-chain optimization: allocate a limited AL-course budget across woredas.

Usage:
    python src/optimize_supply_chain.py

Reads:  data/model_frame.csv
Writes: outputs/woreda_need_vs_allocation_<year>.csv
        outputs/tier_need_vs_allocation_<year>.csv
        outputs/supply_chain_coverage.html
"""
import os

import pandas as pd
import plotly.express as px
import pulp

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_YEAR = 2024
BUDGET_FRACTION_OF_NEED = 0.70          # CHANGE ME
FLOOR_BY_TIER = {"very high": 0.50, "high": 0.30, "moderate": 0.10, "low": 0.0}  # CHANGE ME
TIER_WEIGHT = {"very high": 4, "high": 3, "moderate": 2, "low": 1}               # CHANGE ME
TIME_TIEBREAK_WEIGHT = 0.02             # CHANGE ME


def main():
    model_frame = pd.read_csv(os.path.join(DATA_DIR, "model_frame.csv"), parse_dates=["date"])

    annual = (model_frame[model_frame["year"] == TARGET_YEAR]
              .groupby("ADM3_PCODE")
              .agg(ADM3_EN=("ADM3_EN", "first"), ADM1_EN=("ADM1_EN", "first"),
                   annual_need_al=("need_al_courses", "sum"),
                   annual_need_art=("need_artesunate", "sum"),
                   time_hr=("time_hr", "first"), burden_tier=("burden_tier", "first"),
                   population=("population", "mean"))
              .reset_index().dropna(subset=["annual_need_al"]))

    annual["weight"] = annual["burden_tier"].map(TIER_WEIGHT)
    annual["floor"] = annual["burden_tier"].map(FLOOR_BY_TIER) * annual["annual_need_al"]

    total_need = annual["annual_need_al"].sum()
    budget = BUDGET_FRACTION_OF_NEED * total_need
    t_min, t_max = annual["time_hr"].min(), annual["time_hr"].max()
    annual["time_norm"] = (annual["time_hr"] - t_min) / (t_max - t_min)

    prob = pulp.LpProblem("AL_allocation", pulp.LpMaximize)
    alloc_vars = {
        r.ADM3_PCODE: pulp.LpVariable(f"alloc_{r.ADM3_PCODE}", lowBound=r.floor, upBound=r.annual_need_al)
        for r in annual.itertuples()
    }
    obj_terms = []
    for r in annual.itertuples():
        need = max(r.annual_need_al, 1)
        coef = (r.weight - TIME_TIEBREAK_WEIGHT * r.time_norm) / need
        obj_terms.append(coef * alloc_vars[r.ADM3_PCODE])
    prob += pulp.lpSum(obj_terms)
    prob += pulp.lpSum(alloc_vars.values()) <= budget

    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    print("Solve status:", pulp.LpStatus[status])

    annual["al_allocation"] = annual["ADM3_PCODE"].map(lambda w: alloc_vars[w].value())
    annual["al_coverage_pct"] = (annual["al_allocation"] / annual["annual_need_al"] * 100).round(1)

    annual.to_csv(os.path.join(OUT_DIR, f"woreda_need_vs_allocation_{TARGET_YEAR}.csv"), index=False)

    tier_summary = (annual.groupby("burden_tier")
                     .agg(n_woredas=("ADM3_PCODE", "count"),
                          total_need_al_courses=("annual_need_al", "sum"),
                          total_allocated_al_courses=("al_allocation", "sum"))
                     .reindex(["very high", "high", "moderate", "low"]))
    tier_summary["gap_courses"] = tier_summary["total_need_al_courses"] - tier_summary["total_allocated_al_courses"]
    tier_summary["coverage_pct"] = (tier_summary["total_allocated_al_courses"] /
                                     tier_summary["total_need_al_courses"] * 100).round(1)
    tier_summary.reset_index().to_csv(
        os.path.join(OUT_DIR, f"tier_need_vs_allocation_{TARGET_YEAR}.csv"), index=False)

    print(tier_summary)

    fig = px.box(annual, x="burden_tier", y="al_coverage_pct", color="burden_tier",
                 category_orders={"burden_tier": ["very high", "high", "moderate", "low"]},
                 title=f"Optimized AL coverage % by burden tier ({TARGET_YEAR})", points="all")
    fig.write_html(os.path.join(OUT_DIR, "supply_chain_coverage.html"), include_plotlyjs='cdn')

    print("Saved allocation CSVs + plot.")


if __name__ == "__main__":
    main()
