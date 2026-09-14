#!/usr/bin/env python
"""Long run: wind-aware demo, K=2, tight level, warm-started from best known.

Chains each operation topologically (depot first) so MTZ order Starts are
valid, then solves with a long time limit.

Usage:
    python scripts/demo_wind_long.py --time-limit 7200
"""
from __future__ import annotations
import sys, os, json, time, argparse, logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

from drone_cpp.data_structures import (Point3D, Vertex, VertexType, Operation,
                                       Solution)
from drone_cpp.model import build_model
from drone_cpp.visualization import CPPVis
from demo_interruptions import build_demo_instance
from make_animation_frames import load_solution

logger = logging.getLogger("drone_cpp.demo_long")

OUT_DIR = "demo_wind"


def chain_op(edges, depot):
    by_u = {}
    for e in edges:
        by_u.setdefault(e[0], []).append(e)
    start = next((e for e in edges if e[0] == depot), None)
    if start is None:
        return list(edges)
    ordered, used, cur = [start], {start}, start[1]
    while True:
        nx = [e for e in by_u.get(cur, []) if e not in used]
        if not nx:
            break
        ordered.append(nx[0])
        used.add(nx[0])
        cur = nx[0][1]
    ordered += [e for e in edges if e not in used]
    return ordered


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--time-limit", type=float, default=7200.0)
    p.add_argument("--mip-gap", type=float, default=0.02)
    p.add_argument("--endurance", type=float, default=173.0)
    p.add_argument("--warm", type=str, default="demo_wind/demoW_E0173_K2.json")
    p.add_argument("--tag", type=str, default="demoW_E0173_K2_2h")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    inst = build_demo_instance(args.endurance, num_ops=4)
    sol0 = load_solution(args.warm)
    logger.info("Warm reference obj=%.2f ops=%d",
                sol0.objective_value, len(sol0.operations))
    # chain ops for valid MTZ order
    dep = inst.depot_vertex
    sol0.operations = [Operation(edges=chain_op(list(op.edges), dep))
                       for op in sol0.operations]

    model = build_model(inst, "rings", verbose=False,
                        num_split_points=2, wind_aware=True)
    model.set_warm_start(sol0)
    model.model.setParam("MIPGap", args.mip_gap)
    t0 = time.time()
    sol = model.optimize(tl=args.time_limit)
    el = time.time() - t0
    if sol is None:
        logger.error("No solution")
        return
    logger.info("LONG RUN -> obj=%.2f m ops=%d gap=%.2f%% time=%.0fs status=%s "
                "(first=%.2f at %.1fs)",
                sol.objective_value, len(sol.operations),
                model.model.MIPGap * 100, el, sol.status,
                sol.first_incumbent_obj or -1, sol.first_incumbent_time or -1)
    sol.save(os.path.join(OUT_DIR, f"{args.tag}.json"))
    fig = CPPVis.plot_solution_2d(
        inst, sol, figsize=(10, 8),
        title=(f"Demo wind-aware K=2 E={args.endurance:.0f}J 2h | "
               f"{len(sol.operations)} op(s), {sol.objective_value:.1f} m"))
    fig.savefig(os.path.join(OUT_DIR, f"{args.tag}.png"), dpi=150)
    plt.close(fig)
    logger.info("Saved %s/%s.*", OUT_DIR, args.tag)
    print("RESULT", {"obj": round(sol.objective_value, 2),
                     "ops": len(sol.operations),
                     "gap": round(model.model.MIPGap * 100, 2),
                     "time": round(el, 1), "status": sol.status,
                     "first": sol.first_incumbent_obj})


if __name__ == "__main__":
    main()