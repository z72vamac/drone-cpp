"""Sweep drone endurance values and record objective, gap, plots per level.

For each endurance value a fresh CPP model is built, solved with Gurobi, and
the resulting solution JSON, 2D PNG and 3D PNG are written to sweep_results/.
A summary CSV is also produced at the end.

Usage:
    python sweep_endurance.py --time-limit 600 --mip-gap 0.1
    python sweep_endurance.py --start 50 --stop 130 --step 5
"""
from __future__ import annotations
import sys, os, csv, json, time, argparse, logging

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
    DEFAULT_HEIGHT_RANGE, DEFAULT_NUM_OPS, DEFAULT_MIP_GAP,
    SWEEP_ENDURANCE_RANGE,
)

logger = logging.getLogger("drone_cpp.sweep")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    start, stop, step = SWEEP_ENDURANCE_RANGE
    p.add_argument("--start", type=int, default=start,
                   help=f"Start endurance (J), default {start}")
    p.add_argument("--stop", type=int, default=stop,
                   help=f"Stop endurance (J), exclusive, default {stop}")
    p.add_argument("--step", type=int, default=step,
                   help=f"Endurance step (J), default {step}")
    p.add_argument("--time-limit", type=float, default=600.0,
                   help="Gurobi time limit per instance (default 600s)")
    p.add_argument("--mip-gap", type=float, default=DEFAULT_MIP_GAP,
                   help=f"Gurobi MIPGap (default {DEFAULT_MIP_GAP})")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help=f"Instance seed (default {DEFAULT_SEED})")
    p.add_argument("--out-dir", type=str, default="sweep_results",
                   help="Output directory for JSON+PNGs (default sweep_results)")
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


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    endurance_values = list(range(args.start, args.stop, args.step))
    logger.info("Sweep endurance values: %s J", endurance_values)

    inst = build_base(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    summary_rows = []

    for end in endurance_values:
        logger.info("=" * 60)
        logger.info("ENDURANCE = %d J", end)
        logger.info("=" * 60)

        limited_drone = DroneParams(
            front_area=inst.drone.front_area,
            drag_coef=inst.drone.drag_coef,
            max_endurance=float(end),
            cruise_speed=inst.drone.cruise_speed,
            vertical_speed=inst.drone.vertical_speed,
        )
        inst_run = Instance(regions=inst.regions, depot=inst.depot,
                            drone=limited_drone, wind=inst.wind,
                            num_operations=inst.num_operations)

        model = CPPModel(inst_run, verbose=True)
        model.model.setParam("MIPGap", args.mip_gap)
        t0 = time.time()
        solution = model.optimize(tl=args.time_limit)
        elapsed = time.time() - t0
        gap = model.model.MIPGap if model.model.SolCount > 0 else float("nan")
        node_count = model.model.NodeCount
        status = model.model.Status

        row = {"endurance": end, "elapsed_s": f"{elapsed:.1f}",
               "gap_pct": f"{gap*100:.2f}", "nodes": node_count, "status": status}

        if solution is None:
            logger.warning("No feasible solution for endurance=%d", end)
            row.update({"obj": "-", "n_ops": "-", "ops_detail": "-",
                        "chain_sel": "-", "total_chain_len": "-"})
            summary_rows.append(row)
            json.dump(row, open(f"{args.out_dir}/end_{end:03d}_data.json", "w"), indent=2)
            continue

        obj = solution.objective_value
        n_ops = len(solution.operations)
        ops_detail = [(oi, len(op.edges)) for oi, op in enumerate(solution.operations)]
        chain_sel = {f"R{r_id}": f"ch{ti}" for r_id, ti in solution.chain_selection.items()}
        total_chain_len = sum(
            inst_run.regions[r_id].chains[ti].total_length
            for r_id, ti in solution.chain_selection.items()
        )

        logger.info("Objective=%.2f m | Ops=%d | Gap=%.2f%% | Time=%.1fs | Status=%s",
                    obj, n_ops, gap * 100, elapsed, status)

        solution.save(f"{args.out_dir}/end_{end:03d}_sol.json")
        logger.info("Saved %s/end_%03d_sol.json", args.out_dir, end)

        fig2d = CPPVis.plot_solution_2d(
            inst_run, solution, figsize=(9, 7),
            title=f"Endurance={end}J  |  {n_ops} op(s), {obj:.1f}m",
        )
        fig2d.savefig(f"{args.out_dir}/endurance_{end:03d}.png", dpi=300, bbox_inches="tight")
        plt.close(fig2d)
        logger.info("Saved %s/endurance_%03d.png", args.out_dir, end)

        fig3d = CPPVis.plot_solution_3d(
            inst_run, solution, figsize=(10, 8),
            title=f"3D - Endurance={end}J  |  {n_ops} op(s), {obj:.1f}m",
        )
        fig3d.savefig(f"{args.out_dir}/endurance_{end:03d}_3d.png", dpi=300, bbox_inches="tight")
        plt.close(fig3d)
        logger.info("Saved %s/endurance_%03d_3d.png", args.out_dir, end)

        row.update({"obj": f"{obj:.1f}", "n_ops": n_ops,
                    "ops_detail": json.dumps(ops_detail),
                    "chain_sel": json.dumps(chain_sel),
                    "total_chain_len": f"{total_chain_len:.1f}"})
        summary_rows.append(row)
        data = {**row,
                "edges_per_op": [[str(e) for e in op.edges] for op in solution.operations],
                "total_chain_len_m": total_chain_len}
        json.dump(data, open(f"{args.out_dir}/end_{end:03d}_data.json", "w"), indent=2)

    summary_path = f"{args.out_dir}/summary.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["endurance", "n_ops", "obj", "gap_pct",
                                         "elapsed_s", "nodes", "status",
                                         "ops_detail", "chain_sel", "total_chain_len"])
        w.writeheader(); w.writerows(summary_rows)
    logger.info("Saved %s", summary_path)
    logger.info("Done! Results in %s/", args.out_dir)


if __name__ == "__main__":
    main()