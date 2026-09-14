#!/usr/bin/env python
"""K-sweep on the wind-aware demo (E=173 tight): K=1..4 split points per ring.

Shows how tour structure (fragmentation, weird moves) changes with the number
of allowed split points. K>=2 warm-started from the K=1 optimum via the exact
K=1 -> K mapping.

Usage:
    python scripts/sweep_demo_wind_k.py --ks 1,2,3,4 --time-limit 1200
"""
from __future__ import annotations
import sys, os, json, time, argparse, logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

from drone_cpp.model import build_model
from drone_cpp.visualization import CPPVis
from demo_interruptions import build_demo_instance, count_central_visits
from make_animation_frames import load_solution

logger = logging.getLogger("drone_cpp.demo_wind_k")

OUT_DIR = "demo_wind"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ks", type=str, default="1,2,3,4")
    p.add_argument("--endurance", type=float, default=173.0)
    p.add_argument("--num-ops", type=int, default=4)
    p.add_argument("--time-limit", type=float, default=1200.0)
    p.add_argument("--mip-gap", type=float, default=0.02)
    p.add_argument("--threads", type=int, default=0,
                   help="Gurobi threads (0 = all available cores)")
    p.add_argument("--warm-from", type=str, default=None,
                   help="Solution JSON for identity warm start (same K/structure). "
                        "Takes precedence over the K=1 mapping when given.")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    ks = [int(k) for k in args.ks.split(",")]
    os.makedirs(OUT_DIR, exist_ok=True)
    inst = build_demo_instance(args.endurance, num_ops=args.num_ops)

    rows = []
    sol_k1 = None
    warm_same = None
    if args.warm_from:
        from drone_cpp.data_structures import Operation as _Op
        warm_same = load_solution(args.warm_from)
        logger.info("Loaded identity warm start %s (obj=%.2f)",
                    args.warm_from, warm_same.objective_value)
        # chain each op from the depot so MTZ order Starts are valid
        try:
            from demo_interruptions import build_demo_instance as _bdi
            _dep = _bdi(args.endurance, num_ops=args.num_ops).depot_vertex
            _chained = []
            for _op in warm_same.operations:
                _by_u = {}
                for _e in _op.edges:
                    _by_u.setdefault(_e[0], []).append(_e)
                _st = next((_e for _e in _op.edges if _e[0] == _dep), None)
                if _st is None:
                    _chained.append(_op)
                    continue
                _ord, _used, _cur = [_st], {_st}, _st[1]
                while True:
                    _nx = [_e for _e in _by_u.get(_cur, []) if _e not in _used]
                    if not _nx:
                        break
                    _ord.append(_nx[0])
                    _used.add(_nx[0])
                    _cur = _nx[0][1]
                _ord += [_e for _e in _op.edges if _e not in _used]
                _chained.append(_Op(edges=_ord))
            warm_same.operations = _chained
        except Exception as ex:
            logger.warning("Chaining warm solution failed: %s", ex)
        # chain each op from the depot so MTZ order Starts are valid
        dep = inst.depot_vertex
        chained = []
        for op in warm_same.operations:
            by_u = {}
            for e in op.edges:
                by_u.setdefault(e[0], []).append(e)
            start = next((e for e in op.edges if e[0] == dep), None)
            if start is None:
                chained.append(op)
                continue
            ordered, used, cur = [start], {start}, start[1]
            while True:
                nx = [e for e in by_u.get(cur, []) if e not in used]
                if not nx:
                    break
                ordered.append(nx[0])
                used.add(nx[0])
                cur = nx[0][1]
            ordered += [e for e in op.edges if e not in used]
            chained.append(_Op(edges=ordered))
        warm_same.operations = chained
    for K in ks:
        model = build_model(inst, "rings", verbose=False,
                            num_split_points=K, wind_aware=True)
        summ = model.variable_summary()
        logger.info("K=%d model size: %s", K, summ)
        if warm_same is not None:
            try:
                # identity warm start: solution comes from the same model
                # structure, so lambdas are already in segment-index units
                model.set_warm_start(warm_same, lambda_in_meters=False)
                logger.info("Identity warm-started K=%d from %s", K, args.warm_from)
            except Exception as ex:
                logger.warning("Identity warm start failed: %s", ex)
        elif K > 1 and sol_k1 is not None:
            model.set_warm_start_from_k1(sol_k1)
            logger.info("Warm-started K=%d from K=1 (%.2f)", K,
                        sol_k1.objective_value)
        model.model.setParam("MIPGap", args.mip_gap)
        if args.threads > 0:
            model.model.setParam("Threads", args.threads)
        t0 = time.time()
        sol = model.optimize(tl=args.time_limit)
        el = time.time() - t0
        if sol is None:
            logger.info("K=%d -> NO SOLUTION", K)
            rows.append({"K": K, "objective": None, "status": "NONE"})
            continue
        vv, tot = count_central_visits(sol, 0)
        # count non-degenerate arcs (length > 1e-6 in dfs units)
        logger.info("K=%d -> obj=%.2f m ops=%d visits=%s gap=%.2f%% time=%.0fs status=%s "
                    "first=%.2f", K, sol.objective_value, len(sol.operations),
                    vv, model.model.MIPGap * 100, el, sol.status,
                    sol.first_incumbent_obj or -1)
        rows.append({"K": K, "objective": round(sol.objective_value, 2),
                     "n_ops": len(sol.operations), "visits": vv,
                     "gap_pct": round(model.model.MIPGap * 100, 2),
                     "time_s": round(el, 1), "status": sol.status,
                     "first": sol.first_incumbent_obj,
                     "n_vars": summ["binary"] + summ["continuous"] + summ["integer"]})
        sol.save(os.path.join(OUT_DIR, f"demoW_E{int(args.endurance):04d}_K{K}.json"))
        fig = CPPVis.plot_solution_2d(
            inst, sol, figsize=(10, 8),
            title=(f"Demo wind-aware K={K} E={args.endurance:.0f}J | "
                   f"{len(sol.operations)} op(s), {sol.objective_value:.1f} m"))
        fig.savefig(os.path.join(OUT_DIR, f"demoW_E{int(args.endurance):04d}_K{K}.png"),
                    dpi=150)
        plt.close(fig)
        if K == 1:
            sol_k1 = sol

    summary_path = os.path.join(
        OUT_DIR, f"demoW_Ksweep_E{int(args.endurance):04d}.json")
    merged = {}
    if os.path.exists(summary_path):
        try:
            for r in json.load(open(summary_path, encoding="utf-8")):
                merged[r["K"]] = r
        except Exception:
            pass
    for r in rows:
        merged[r["K"]] = r
    rows_all = [merged[k] for k in sorted(merged)]
    json.dump(rows_all, open(summary_path, "w"), indent=2)
    print("\nK-SWEEP SUMMARY")
    for r in rows_all:
        print(" ", r)


if __name__ == "__main__":
    main()