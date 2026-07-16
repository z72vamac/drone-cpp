"""Solve a single drone CPP instance and plot the solution.

Entry point for the pipe: generate a random instance (configurable seed),
apply the cone-effect chain reconstruction, replace the drone endurance with
the caller's value, solve the MIQP with Gurobi and persist both the solution
JSON (with metadata) and the 2D instance/solution PNGs.

Usage:
    python solve_and_plot.py --endurance 100 --time-limit 120 --seed 42
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
    DEFAULT_HEIGHT_RANGE, DEFAULT_NUM_OPS,
    DEFAULT_ENDURANCE, DEFAULT_TIME_LIMIT, DEFAULT_MIP_GAP,
)

logger = logging.getLogger("drone_cpp.solve")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--endurance", type=float, default=DEFAULT_ENDURANCE,
                   help=f"Drone max endurance in Joules (default: {DEFAULT_ENDURANCE})")
    p.add_argument("--time-limit", type=float, default=DEFAULT_TIME_LIMIT,
                   help=f"Gurobi time limit in seconds (default: {DEFAULT_TIME_LIMIT})")
    p.add_argument("--mip-gap", type=float, default=DEFAULT_MIP_GAP,
                   help=f"Gurobi MIPGap fraction (default: {DEFAULT_MIP_GAP})")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help=f"Random seed (default: {DEFAULT_SEED})")
    p.add_argument("--out", type=str, default="cone_solution.json",
                   help="Output JSON path for the solution")
    p.add_argument("--plot-prefix", type=str, default="cone_solution",
                   help="Output prefix for plot PNGs (instance + solution)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def build_instance(endurance: float, seed: int) -> Instance:
    logger.info("Generating base instance (seed=%d, %d regions, heights %.1f-%.1f)",
                seed, DEFAULT_NUM_REGIONS, *DEFAULT_HEIGHT_RANGE)
    base_inst = InstanceGenerator.generate(
        num_regions=DEFAULT_NUM_REGIONS,
        area_bounds=DEFAULT_AREA_BOUNDS,
        min_region_size=DEFAULT_REGION_SIZES[0],
        max_region_size=DEFAULT_REGION_SIZES[1],
        spacing=DEFAULT_SPACING,
        num_heights=DEFAULT_NUM_HEIGHTS,
        min_height=DEFAULT_HEIGHT_RANGE[0],
        max_height=DEFAULT_HEIGHT_RANGE[1],
        num_ops=DEFAULT_NUM_OPS,
        seed=seed,
    )
    inst = InstanceGenerator.rebuild_instance(base_inst)
    logger.info("Replacing drone endurance with %.2f J", endurance)
    limited_drone = DroneParams(
        front_area=base_inst.drone.front_area,
        drag_coef=base_inst.drone.drag_coef,
        max_endurance=endurance,
        cruise_speed=base_inst.drone.cruise_speed,
        vertical_speed=base_inst.drone.vertical_speed,
    )
    return Instance(regions=inst.regions, depot=base_inst.depot, drone=limited_drone,
                    wind=base_inst.wind, num_operations=base_inst.num_operations)


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    inst = build_instance(args.endurance, args.seed)

    logger.info("Instance with cone-effect chains: %d regions", inst.num_regions)
    for r in inst.regions:
        for ch in r.chains:
            logger.info("    R%d h=%.0fm: %d segments, length=%.1fm",
                        r.id, ch.height, len(ch.segments), ch.total_length)
    logger.info("Depot: (%.1f, %.1f) | Wind: %.1f m/s @ 10m",
                inst.depot.x, inst.depot.y, inst.wind.speed_at_10m)

    logger.info("Building Gurobi MIQP model...")
    model = CPPModel(inst, verbose=True)
    logger.info("Variables: %d  |  Constraints: %d",
                model.model.NumVars, model.model.NumConstrs)
    model.model.setParam("MIPGap", args.mip_gap)

    logger.info("Solving (time limit=%.1fs, MIPGap=%.1f%%)...",
                args.time_limit, args.mip_gap * 100)
    solution = model.optimize(tl=args.time_limit)

    if solution is None:
        logger.error("No feasible solution found within the time limit.")
        sys.exit(1)

    logger.info("Solution found! objective=%.2f m | %d operations | gap=%s%% | status=%s | %.1fs",
                solution.objective_value, len(solution.operations),
                f"{solution.mip_gap*100:.2f}" if solution.mip_gap is not None else "-",
                solution.status or "?", solution.solve_time or 0.0)
    for oi, op in enumerate(solution.operations):
        logger.info("    Op %d: %d edges", oi, len(op.edges))
    for r_id, t_idx in solution.chain_selection.items():
        ch = inst.regions[r_id].chains[t_idx]
        logger.info("    R%d -> chain %d (h=%.0fm, %d segments)",
                    r_id, t_idx, ch.height, len(ch.segments))

    solution.save(args.out)
    logger.info("Solution saved to %s", args.out)

    logger.info("Generating plots...")
    fig1 = CPPVis.plot_instance(inst, show_chains=True)
    instance_png = f"{args.plot_prefix}_instance.png"
    fig1.savefig(instance_png, dpi=150)
    plt.close(fig1)
    logger.info("Saved %s", instance_png)

    fig2 = CPPVis.plot_solution(inst, solution)
    solution_png = f"{args.plot_prefix}.png"
    fig2.savefig(solution_png, dpi=150)
    plt.close(fig2)
    logger.info("Saved %s", solution_png)


if __name__ == "__main__":
    main()