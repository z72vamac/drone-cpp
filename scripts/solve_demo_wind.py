#!/usr/bin/env python
"""Solve the demo instance (central + 4 satellites) with the NEW wind-aware
model (real joules, per-heading airspeed, power-law wind) and plot it with
the wind direction arrow.

Steps: calibrate E_min under true wind-aware accounting, solve K=1 at a
tight and a loose level, solve K=2 warm-started at the tight level.
Saves solutions + plots (with wind arrow) to demo_wind/.

Usage:
    python scripts/solve_demo_wind.py
"""
from __future__ import annotations
import sys, os, json, time, logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

from drone_cpp.data_structures import Instance, DroneParams
from drone_cpp.model import build_model
from drone_cpp.visualization import CPPVis
from demo_interruptions import build_demo_instance, count_central_visits
from validate_wind import estimate_e_min_true

logger = logging.getLogger("drone_cpp.demo_wind")

OUT_DIR = "demo_wind"


def with_endurance(inst: Instance, E: float) -> Instance:
    drone = DroneParams(front_area=inst.drone.front_area,
                        drag_coef=inst.drone.drag_coef, max_endurance=float(E),
                        cruise_speed=inst.drone.cruise_speed,
                        vertical_speed=inst.drone.vertical_speed)
    return Instance(regions=inst.regions, depot=inst.depot, drone=drone,
                    wind=inst.wind, num_operations=inst.num_operations)


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    os.makedirs(OUT_DIR, exist_ok=True)
    base = build_demo_instance(300.0, num_ops=4)  # endurance replaced below
    logger.info("Demo wind: %.1f m/s dir=%s",
                base.wind.speed_at_10m, base.wind.direction)

    e_min, per_r = estimate_e_min_true(base)
    logger.info("E_min_true = %.2f J per region: %s", e_min, per_r)
    e_tight = float(np.ceil(e_min * 1.5))
    e_loose = float(np.ceil(e_min * 3.0))
    logger.info("Levels: tight=%.0f J loose=%.0f J", e_tight, e_loose)

    rows = []
    sols = {}
    for E, lvl in [(e_tight, "tight"), (e_loose, "loose")]:
        inst = with_endurance(base, E)
        model = build_model(inst, "rings", verbose=False,
                            num_split_points=1, wind_aware=True)
        model.model.setParam("MIPGap", 0.02)
        t0 = time.time()
        sol = model.optimize(tl=180.0)
        el = time.time() - t0
        if sol is None:
            logger.info("WA K=1 %s E=%.0f -> INFEASIBLE", lvl, E)
            rows.append({"tag": f"WA K=1 {lvl}", "E": E, "objective": None,
                         "status": "INFEASIBLE"})
            continue
        vv, tot = count_central_visits(sol, 0)
        logger.info("WA K=1 %s E=%.0f -> obj=%.1f m ops=%d visits=%s gap=%.1f%% time=%.0fs",
                    lvl, E, sol.objective_value, len(sol.operations), vv,
                    model.model.MIPGap * 100, el)
        rows.append({"tag": f"WA K=1 {lvl}", "E": E,
                     "objective": round(sol.objective_value, 1),
                     "n_ops": len(sol.operations), "visits": vv,
                     "gap_pct": round(model.model.MIPGap * 100, 1),
                     "time_s": round(el, 1), "status": sol.status})
        sols[lvl] = (sol, inst)
        sol.save(os.path.join(OUT_DIR, f"demoW_E{int(E):04d}_K1.json"))
        fig = CPPVis.plot_solution_2d(
            inst, sol, figsize=(10, 8),
            title=(f"Demo wind-aware K=1 {lvl} E={E:.0f}J | {len(sol.operations)} "
                   f"op(s), {sol.objective_value:.1f} m"))
        fig.savefig(os.path.join(OUT_DIR, f"demoW_E{int(E):04d}_K1.png"), dpi=150)
        plt.close(fig)

    # K=2 warm-started at the tight level
    if "tight" in sols:
        sol1, inst1 = sols["tight"]
        model = build_model(inst1, "rings", verbose=False,
                            num_split_points=2, wind_aware=True)
        model.set_warm_start_from_k1(sol1)
        model.model.setParam("MIPGap", 0.02)
        t0 = time.time()
        sol = model.optimize(tl=300.0)
        el = time.time() - t0
        if sol is None:
            logger.info("WA K=2 tight -> INFEASIBLE")
            rows.append({"tag": "WA K=2 tight", "E": e_tight,
                         "objective": None, "status": "INFEASIBLE"})
        else:
            vv, tot = count_central_visits(sol, 0)
            logger.info("WA K=2 tight E=%.0f -> obj=%.1f m ops=%d visits=%s gap=%.1f%% time=%.0fs",
                        e_tight, sol.objective_value, len(sol.operations), vv,
                        model.model.MIPGap * 100, el)
            rows.append({"tag": "WA K=2 tight", "E": e_tight,
                         "objective": round(sol.objective_value, 1),
                         "n_ops": len(sol.operations), "visits": vv,
                         "gap_pct": round(model.model.MIPGap * 100, 1),
                         "time_s": round(el, 1), "status": sol.status})
            sol.save(os.path.join(OUT_DIR, f"demoW_E{int(e_tight):04d}_K2.json"))
            fig = CPPVis.plot_solution_2d(
                inst1, sol, figsize=(10, 8),
                title=(f"Demo wind-aware K=2 tight E={e_tight:.0f}J | "
                       f"{len(sol.operations)} op(s), {sol.objective_value:.1f} m, "
                       f"{tot} central visits"))
            fig.savefig(os.path.join(OUT_DIR, f"demoW_E{int(e_tight):04d}_K2.png"),
                        dpi=150)
            plt.close(fig)

    fig0 = CPPVis.plot_instance(with_endurance(base, e_tight),
                                chain_selection=None)
    fig0.savefig(os.path.join(OUT_DIR, "demoW_instance.png"), dpi=150)
    plt.close(fig0)

    json.dump(rows, open(os.path.join(OUT_DIR, "demoW_summary.json"), "w"), indent=2)
    print("\nDEMO WIND-AWARE SUMMARY")
    for r in rows:
        print(" ", r)


if __name__ == "__main__":
    main()