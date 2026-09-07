#!/usr/bin/env python
"""Solve the 3-region showcase instance (seed 42, heights 15/40/65 m) with the
CURRENT implementation (RingsModel: variable entry/exit points anywhere on the
rings) for endurance 60..120 J, and write solution JSONs + static snapshots.

Outputs:
  sweep_results_new/end_XXX_sol.json     (solution used for animation frames)
  sweep_results_new/endurance_XXX.png    (static snapshot, 2D)
  prints a summary row per endurance.
"""
from __future__ import annotations
import sys, os, json, time, logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")

from drone_cpp.data_structures import Instance, DroneParams
from drone_cpp.model import build_model
from drone_cpp.visualization import CPPVis
from sweep_endurance import build_base

ENDURANCES = [150, 200, 250, 300, 400, 600]
OUT_DIR = "sweep_results_new"
TIME_LIMIT = 45.0          # quality solutions for the animation
MIP_GAP = 0.02             # small gap tolerance; recorded in summary
NUM_OPS = 6                # enough operations for tight endurance levels


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    os.makedirs(OUT_DIR, exist_ok=True)
    inst = build_base(42)
    print(f"Instance: {inst.num_regions} regions, depot {inst.depot}")
    for r in inst.regions:
        for c in r.chains:
            print(f"  R{r.id} chain {c.idx}: h={c.height:.0f} m, "
                  f"{len(c.rings)} rings, {c.total_length:.1f} m")

    rows = []
    for end in ENDURANCES:
        t0 = time.time()
        drone = DroneParams(
            front_area=inst.drone.front_area,
            drag_coef=inst.drone.drag_coef,
            max_endurance=float(end),
            cruise_speed=inst.drone.cruise_speed,
            vertical_speed=inst.drone.vertical_speed,
        )
        inst_run = Instance(regions=inst.regions, depot=inst.depot,
                            drone=drone, wind=inst.wind,
                            num_operations=NUM_OPS)
        model = build_model(inst_run, "rings", verbose=False)
        model.model.setParam("MIPGap", MIP_GAP)
        sol = model.optimize(tl=TIME_LIMIT)
        elapsed = time.time() - t0
        if sol is None:
            gap = model.model.MIPGap if model.model.SolCount > 0 else float("nan")
            print(f"E={end}J  NO SOLUTION  gap={gap*100:.1f}%  status={model.model.Status}")
            rows.append({"endurance": end, "obj": "-", "n_ops": "-", "gap_pct": "-",
                         "time_s": f"{elapsed:.1f}", "status": "NONE"})
            continue
        gap = model.model.MIPGap
        print(f"E={end}J  obj={sol.objective_value:.1f} m  ops={len(sol.operations)}  "
              f"gap={gap*100:.1f}%  time={elapsed:.1f}s  status={sol.status}")
        sol.save(f"{OUT_DIR}/end_{end:03d}_sol.json")

        fig = CPPVis.plot_solution_2d(
            inst_run, sol, figsize=(9, 7),
            title=f"Endurance={end}J  |  {len(sol.operations)} op(s), {sol.objective_value:.1f} m")
        fig.savefig(f"{OUT_DIR}/endurance_{end:03d}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        # variable entry/exit check
        frac = [v for v in sol.vertex_lambdas.values() if abs(v - round(v)) > 1e-6]
        rows.append({"endurance": end, "obj": f"{sol.objective_value:.1f}",
                     "n_ops": len(sol.operations), "gap_pct": f"{gap*100:.1f}",
                     "time_s": f"{elapsed:.1f}", "status": sol.status,
                     "frac_lambdas": len(frac)})

    print("\nSUMMARY")
    for r in rows:
        print(r)
    json.dump(rows, open(f"{OUT_DIR}/summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()