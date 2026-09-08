#!/usr/bin/env python
"""Sweep the number of continuous split points per ring (K) and compare.

For each K, the RingsModel partitions every ring into K arcs in series
(anchored at the ring start) that can be covered in different operations.
K=1 reproduces the original full-loop behaviour.

Runs cold (no warm start) with identical solver settings so the objectives
are comparable, and records objective / gap / time / status / model size.
Expected: the objective improves with K up to a knee K*, beyond which extra
split points give nothing (empirical number of points needed for optimality).

Usage:
    python scripts/sweep_split_points.py --endurance 150 --ks 1,2,3,4
"""
from __future__ import annotations
import sys, os, json, time, argparse, logging

sys.path.insert(0, ".")

from drone_cpp.data_structures import Instance, DroneParams
from drone_cpp.model import build_model
from sweep_endurance import build_base

logger = logging.getLogger("drone_cpp.split_sweep")

OUT_DIR = "split_points"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--endurance", type=float, default=150.0,
                   help="Drone max endurance in Joules (default: 150)")
    p.add_argument("--ks", type=str, default="1,2,3,4",
                   help="Comma-separated K values (default: 1,2,3,4)")
    p.add_argument("--num-ops", type=int, default=6,
                   help="Number of operations (default: 6)")
    p.add_argument("--time-limit", type=float, default=300.0,
                   help="Gurobi time limit per K in seconds (default 300)")
    p.add_argument("--mip-gap", type=float, default=0.02,
                   help="Gurobi MIPGap fraction (default 0.02)")
    p.add_argument("--seed", type=int, default=42,
                   help="Instance seed (default: 42)")
    p.add_argument("--k1-sol", type=str, default=None,
                   help="Path to a K=1 solution JSON used to warm-start K>=2 "
                        "(default: solve K=1 first when included in --ks)")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    ks = [int(k) for k in args.ks.split(",")]
    os.makedirs(OUT_DIR, exist_ok=True)

    inst = build_base(args.seed)
    drone = DroneParams(
        front_area=inst.drone.front_area,
        drag_coef=inst.drone.drag_coef,
        max_endurance=float(args.endurance),
        cruise_speed=inst.drone.cruise_speed,
        vertical_speed=inst.drone.vertical_speed,
    )
    inst_run = Instance(regions=inst.regions, depot=inst.depot, drone=drone,
                        wind=inst.wind, num_operations=args.num_ops)

    rows = []
    sol_k1 = None
    if args.k1_sol:
        import json as _json
        from drone_cpp.data_structures import (Point3D, Vertex, VertexType,
                                               Operation, Solution)
        _d = _json.load(open(args.k1_sol, encoding="utf-8"))
        _vt = {e.name: e for e in VertexType}
        _ops = []
        for _op in _d["operations"]:
            _ops.append(Operation(edges=[
                (Vertex(e["r"], e["i"], _vt[e["t"]]),
                 Vertex(e["r2"], e["i2"], _vt[e["t2"]])) for e in _op]))
        _vp, _vl, _vr = {}, {}, {}
        for _k, _p in _d["vertex_positions"].items():
            _r, _i, _t = _k.split(",")
            _vp[Vertex(int(_r), int(_i), _vt[_t])] = Point3D(_p["x"], _p["y"], _p["z"])
        for _k, _lam in _d["vertex_lambdas"].items():
            _r, _i, _t = _k.split(",")
            _vl[Vertex(int(_r), int(_i), _vt[_t])] = _lam
        for _k, _ri in _d.get("vertex_rings", {}).items():
            _r, _i, _t = _k.split(",")
            _vr[Vertex(int(_r), int(_i), _vt[_t])] = _ri
        sol_k1 = Solution(
            operations=_ops, objective_value=_d["objective_value"],
            vertex_positions=_vp,
            chain_selection={int(k): v for k, v in _d["chain_selection"].items()},
            vertex_lambdas=_vl, vertex_rings=_vr)
        logger.info("Loaded K=1 reference from %s (obj=%.2f)",
                    args.k1_sol, sol_k1.objective_value)
    for K in ks:
        logger.info("=" * 60)
        logger.info("K = %d split points per ring (E=%.0f J)", K, args.endurance)
        model = build_model(inst_run, "rings", verbose=False, num_split_points=K)
        summ = model.variable_summary()
        logger.info("Model size: %s", summ)
        model.model.setParam("MIPGap", args.mip_gap)
        if K > 1 and sol_k1 is not None:
            # exact mapping of the K=1 optimum: K starts at <= its value
            model.set_warm_start_from_k1(sol_k1)
            logger.info("Warm-started K=%d from K=1 optimum (%.2f)", K,
                        sol_k1.objective_value)
        t0 = time.time()
        sol = model.optimize(tl=args.time_limit)
        elapsed = time.time() - t0
        gap = model.model.MIPGap if model.model.SolCount > 0 else float("nan")
        row = {"K": K, "endurance": args.endurance,
               "objective": round(sol.objective_value, 2) if sol else None,
               "n_ops": len(sol.operations) if sol else None,
               "gap_pct": round(gap * 100, 2),
               "time_s": round(elapsed, 1),
               "status": sol.status if sol else "NONE",
               "n_vars": summ["binary"] + summ["continuous"] + summ["integer"],
               "n_constrs": summ["constraints"]}
        logger.info("K=%d -> obj=%s ops=%s gap=%.1f%% time=%.1fs status=%s",
                    K, row["objective"], row["n_ops"], row["gap_pct"],
                    row["time_s"], row["status"])
        rows.append(row)
        if K == 1 and sol is not None:
            sol_k1 = sol
        if sol is not None:
            sol.save(os.path.join(OUT_DIR, f"sol_E{int(args.endurance):04d}_K{K}.json"))
            logger.info("Saved solution %s", os.path.join(
                OUT_DIR, f"sol_E{int(args.endurance):04d}_K{K}.json"))

    out = os.path.join(OUT_DIR,
                       f"results_E{int(args.endurance):04d}.json")
    json.dump(rows, open(out, "w"), indent=2)
    logger.info("Saved %s", out)
    print("\nK-SWEEP SUMMARY (E=%.0f J)" % args.endurance)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()