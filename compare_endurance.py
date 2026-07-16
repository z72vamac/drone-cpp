"""Solve the same instance at two endurance levels and compare the resulting
solutions side by side. Useful to illustrate how the number of operations
scales with the energy budget.

Usage:
    python compare_endurance.py --endurance-high 3000 --endurance-limited 62
"""
from __future__ import annotations
import sys
import argparse
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")

from drone_cpp.instance_generator import InstanceGenerator
from drone_cpp.data_structures import Instance, DroneParams
from drone_cpp.model import CPPModel
from drone_cpp.visualization import CPPVis
from drone_cpp.config import (
    DEFAULT_SEED, DEFAULT_AREA_BOUNDS, DEFAULT_REGION_SIZES,
    DEFAULT_NUM_REGIONS, DEFAULT_SPACING, DEFAULT_NUM_HEIGHTS,
    DEFAULT_HEIGHT_RANGE, DEFAULT_NUM_OPS, DEFAULT_TIME_LIMIT, DEFAULT_MIP_GAP,
)

logger = logging.getLogger("drone_cpp.compare")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--endurance-high", type=float, default=3000.0,
                   help="High endurance value (J). Usually yields 1 operation.")
    p.add_argument("--endurance-limited", type=float, default=62.0,
                   help="Limited endurance value (J). Usually forces 2 operations.")
    p.add_argument("--time-limit", type=float, default=DEFAULT_TIME_LIMIT,
                   help=f"Gurobi time limit per run (default: {DEFAULT_TIME_LIMIT}s)")
    p.add_argument("--mip-gap", type=float, default=DEFAULT_MIP_GAP)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--out-high", type=str, default="solution_high.json")
    p.add_argument("--out-limited", type=str, default="solution_limited.json")
    p.add_argument("--plot-high", type=str, default="solution_high.png")
    p.add_argument("--plot-limited", type=str, default="solution_limited.png")
    return p.parse_args()


def build_base(seed: int) -> Instance:
    base_inst = InstanceGenerator.generate(
        num_regions=DEFAULT_NUM_REGIONS, area_bounds=DEFAULT_AREA_BOUNDS,
        min_region_size=DEFAULT_REGION_SIZES[0], max_region_size=DEFAULT_REGION_SIZES[1],
        spacing=DEFAULT_SPACING, num_heights=DEFAULT_NUM_HEIGHTS,
        min_height=DEFAULT_HEIGHT_RANGE[0], max_height=DEFAULT_HEIGHT_RANGE[1],
        num_ops=DEFAULT_NUM_OPS, seed=seed,
    )
    return InstanceGenerator.rebuild_instance(base_inst)


def solve_with_endurance(inst: Instance, endurance: float,
                         time_limit: float, mip_gap: float,
                         label: str):
    logger.info("[%s] endurance=%.1f J, solving (tl=%.1fs)...", label, endurance, time_limit)
    drone = DroneParams(
        front_area=inst.drone.front_area, drag_coef=inst.drone.drag_coef,
        max_endurance=endurance, cruise_speed=inst.drone.cruise_speed,
        vertical_speed=inst.drone.vertical_speed,
    )
    inst_run = Instance(regions=inst.regions, depot=inst.depot, drone=drone,
                        wind=inst.wind, num_operations=inst.num_operations)
    model = CPPModel(inst_run, verbose=False)
    model.model.setParam("MIPGap", mip_gap)
    solution = model.optimize(tl=time_limit)
    if solution is None:
        logger.error("[%s] No feasible solution at endurance=%.1f", label, endurance)
        return None
    logger.info("[%s] objective=%.2f m | %d ops | gap=%s%% | status=%s | %.1fs",
                label, solution.objective_value, len(solution.operations),
                f"{solution.mip_gap*100:.2f}" if solution.mip_gap is not None else "-",
                solution.status or "?", solution.solve_time or 0.0)
    return inst_run, solution


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    inst = build_base(args.seed)

    result_high = solve_with_endurance(inst, args.endurance_high,
                                       args.time_limit, args.mip_gap, "HIGH")
    if result_high is None:
        sys.exit(1)
    inst_high, sol_high = result_high
    sol_high.save(args.out_high)
    logger.info("Saved %s", args.out_high)
    fig_high = CPPVis.plot_solution(inst_high, sol_high)
    fig_high.suptitle(
        f"High endurance ({args.endurance_high:.0f} J): "
        f"{sol_high.objective_value:.1f} m, {len(sol_high.operations)} op(s)",
        fontsize=12,
    )
    fig_high.savefig(args.plot_high, dpi=150)
    plt.close(fig_high)
    logger.info("Saved %s", args.plot_high)

    result_lim = solve_with_endurance(inst, args.endurance_limited,
                                      args.time_limit, args.mip_gap, "LIMITED")
    if result_lim is None:
        sys.exit(1)
    inst_lim, sol_lim = result_lim
    sol_lim.save(args.out_limited)
    logger.info("Saved %s", args.out_limited)
    fig_lim = CPPVis.plot_solution(inst_lim, sol_lim)
    fig_lim.suptitle(
        f"Limited endurance ({args.endurance_limited:.0f} J): "
        f"{sol_lim.objective_value:.1f} m, {len(sol_lim.operations)} op(s)",
        fontsize=12,
    )
    fig_lim.savefig(args.plot_limited, dpi=150)
    plt.close(fig_lim)
    logger.info("Saved %s", args.plot_limited)

    logger.info("Done! Both solutions saved.")


if __name__ == "__main__":
    main()