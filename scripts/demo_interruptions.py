#!/usr/bin/env python
"""Demo: one large central region + 4 small satellite regions around it.

The layout forces the drone to interrupt the coverage of the central region
to visit the satellites. With K=1 (single full loop per ring) this is
impossible; with K>=2 (split points per ring) the solver can split the
central rings and interleave satellite visits.

Usage:
    python scripts/demo_interruptions.py --endurance 250 --split-points 3
    python scripts/demo_interruptions.py --sweep-endurance   # try several E with K=1
"""
from __future__ import annotations
import sys, os, json, time, argparse, logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, ".")

from drone_cpp.data_structures import (
    Point3D, Segment, Ring, PolygonalChain, Region, Instance,
    DroneParams, WindParams, Vertex, VertexType,
)
from drone_cpp.instance_generator import InstanceGenerator
from drone_cpp.model import build_model
from drone_cpp.visualization import CPPVis

logger = logging.getLogger("drone_cpp.demo")

OUT_DIR = "demo_interruptions"


def _square(cx: float, cy: float, half: float) -> list:
    return [Point3D(cx - half, cy - half), Point3D(cx + half, cy - half),
            Point3D(cx + half, cy + half), Point3D(cx - half, cy + half)]


def build_demo_instance(endurance: float, num_ops: int = 4,
                        height: float = 30.0) -> Instance:
    """Central 40x40 square + 4 satellite 9x9 squares (N/S/E/W)."""
    rng = np.random.RandomState(7)
    regions = []

    # R0: central square, 2 rings via spiral generator (spacing 15 m)
    central = _square(0.0, 0.0, 20.0)
    ch0 = InstanceGenerator._generate_spiral_chain(
        central, height, 0, 0, rng, spacing=15.0)
    assert ch0 is not None and len(ch0.rings) >= 2, "central chain failed"
    regions.append(Region(id=0, boundary=central, chains=[ch0],
                          num_interruption_points=3))

    # R1..R4: satellites with a single ring each (their own boundary)
    sats = [(0.0, 32.0), (0.0, -32.0), (32.0, 0.0), (-32.0, 0.0)]
    for i, (cx, cy) in enumerate(sats, start=1):
        bpts = _square(cx, cy, 4.5)
        segs = [Segment(bpts[j], bpts[(j + 1) % 4]) for j in range(4)]
        ring = Ring(segments=segs, scale=1.0, height=height)
        chain = PolygonalChain(segments=list(segs), height=height,
                               region_id=i, idx=0, rings=[ring])
        regions.append(Region(id=i, boundary=bpts, chains=[chain],
                              num_interruption_points=1))

    drone = DroneParams(front_area=0.1, drag_coef=0.3,
                        max_endurance=float(endurance),
                        cruise_speed=15.0, vertical_speed=5.0)
    wind = WindParams(direction=np.array([1.0, 0.0, 0.0]),
                      speed_at_10m=4.0, hellmann_exponent=0.2)
    return Instance(regions=regions, depot=Point3D(-55.0, -55.0, 0.0),
                    drone=drone, wind=wind, num_operations=num_ops)


def _chain(order_edges, depot):
    by_u = {}
    for e in order_edges:
        by_u.setdefault(e[0], []).append(e)
    start = next((e for e in order_edges if e[0] == depot), None)
    if start is None:
        return list(order_edges)
    ordered, used, cur = [start], {start}, start[1]
    while True:
        nxt = [e for e in by_u.get(cur, []) if e not in used]
        if not nxt:
            break
        ordered.append(nxt[0])
        used.add(nxt[0])
        cur = nxt[0][1]
    ordered += [e for e in order_edges if e not in used]
    return ordered


def count_central_visits(solution, central_id: int = 0):
    """Maximal contiguous blocks of central-ring ARC edges per operation.

    Returns (visits_per_op, total). A visit = one maximal run of central
    coverage; visits - 1 (per op, chained through depot) ~ interruptions.
    """
    visits = []
    for op in solution.operations:
        # chain the operation from its depot leg
        by_u = {}
        for e in op.edges:
            by_u.setdefault(e[0], []).append(e)
        dep = [e for e in op.edges if e[0].region_id == -1]
        if not dep:
            ordered = list(op.edges)
        else:
            ordered, used, cur = [dep[0]], {dep[0]}, dep[0][1]
            while True:
                nx = [e for e in by_u.get(cur, []) if e not in used]
                if not nx:
                    break
                ordered.append(nx[0])
                used.add(nx[0])
                cur = nx[0][1]
            ordered += [e for e in op.edges if e not in used]

        def _is_central_arc(a, b):
            return (a.region_id == central_id and a.region_id == b.region_id
                    and a.vtype == VertexType.LAUNCH and b.vtype == VertexType.RETRIEVE
                    and abs(a.idx - b.idx) == 1 and a.idx % 2 == 0)

        n_visits, in_run = 0, False
        for (a, b) in ordered:
            if _is_central_arc(a, b):
                if not in_run:
                    n_visits += 1
                    in_run = True
            else:
                # leaving central coverage (transit/depot/connector to elsewhere
                # or a zero-length connector inside central keeps the run)
                if a.region_id == central_id and b.region_id == central_id:
                    continue  # intra-central link (connector): same visit
                in_run = False
        visits.append(n_visits)
    return visits, sum(visits)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--endurance", type=float, default=250.0)
    p.add_argument("--split-points", type=int, default=3)
    p.add_argument("--num-ops", type=int, default=4)
    p.add_argument("--time-limit", type=float, default=300.0)
    p.add_argument("--mip-gap", type=float, default=0.02)
    p.add_argument("--sweep-endurance", action="store_true", default=False,
                   help="Try E in {200,250,300} with K=1 (fast) and report ops")
    p.add_argument("--k1-sol", type=str, default=None,
                   help="K=1 solution JSON to warm-start K>=2")
    p.add_argument("--sweep", action="store_true", default=False,
                   help="Sweep endurance levels with chained warm starts "
                        "(each level starts from the previous solution)")
    p.add_argument("--levels", type=str, default="170,180,200,220,250,300,350,400",
                   help="Comma-separated endurance levels for --sweep")
    p.add_argument("--warm-from", type=str, default=None,
                   help="Solution JSON to warm-start the first sweep level "
                        "(same K required)")
    return p.parse_args()


def run_sweep(args: argparse.Namespace) -> None:
    """Solve the demo instance at increasing endurance levels (K fixed).

    Each level is warm-started from the previous level's solution (feasible
    since endurance only grows), so later levels start from a good incumbent.
    Records objective, ops, central visits, gap and time per level, plus one
    plot per level and a summary figure + JSON.
    """
    levels = [float(e) for e in args.levels.split(",")]
    prev_sol = None
    prev_E = None
    if args.warm_from:
        prev_sol = load_solution_json(args.warm_from)
        logger.info("Loaded warm-start %s (obj=%.2f)",
                    args.warm_from, prev_sol.objective_value)
    rows = []
    for E in levels:
        inst = build_demo_instance(E, num_ops=args.num_ops)
        model = build_model(inst, "rings", verbose=False,
                            num_split_points=args.split_points)
        if prev_sol is not None:
            try:
                model.set_warm_start(prev_sol)
                if prev_E is None:
                    logger.info("Warm-started E=%.0f from %s", E, args.warm_from)
                else:
                    logger.info("Warm-started E=%.0f from E=%.0f solution", E, prev_E)
            except Exception as ex:
                logger.warning("Chained warm start failed: %s", ex)
        model.model.setParam("MIPGap", args.mip_gap)
        t0 = time.time()
        sol = model.optimize(tl=args.time_limit)
        el = time.time() - t0
        gap = model.model.MIPGap if model.model.SolCount > 0 else float("nan")
        if sol is None:
            logger.info("E=%.0f K=%d -> INFEASIBLE (%.1fs)",
                        E, args.split_points, el)
            rows.append({"endurance": E, "objective": None, "n_ops": None,
                         "visits": None, "gap_pct": None, "time_s": round(el, 1),
                         "status": "INFEASIBLE"})
            continue
        vv, tot = count_central_visits(sol, 0)
        logger.info("E=%.0f K=%d -> obj=%.1f ops=%d visits=%s gap=%.1f%% time=%.1fs",
                    E, args.split_points, sol.objective_value,
                    len(sol.operations), vv, gap * 100, el)
        rows.append({"endurance": E, "objective": round(sol.objective_value, 1),
                     "n_ops": len(sol.operations), "visits": vv,
                     "total_visits": tot, "gap_pct": round(gap * 100, 1),
                     "time_s": round(el, 1), "status": sol.status})
        sol.save(os.path.join(OUT_DIR, f"sweep_E{int(E):04d}_K{args.split_points}.json"))
        fig = CPPVis.plot_solution_2d(
            inst, sol, figsize=(10, 8),
            title=(f"Demo sweep E={E:.0f}J K={args.split_points} | "
                   f"{len(sol.operations)} op(s), {sol.objective_value:.1f} m, "
                   f"{tot} central visits"))
        fig.savefig(os.path.join(OUT_DIR, f"sweep_E{int(E):04d}_K{args.split_points}.png"),
                    dpi=150)
        plt.close(fig)
        prev_sol = sol
        prev_E = E

    summary_path = os.path.join(
        OUT_DIR, f"sweep_summary_K{args.split_points}.json")
    merged = {}
    if os.path.exists(summary_path):
        try:
            for r in json.load(open(summary_path, encoding="utf-8")):
                merged[r["endurance"]] = r
        except Exception:
            pass
    for r in rows:
        merged[r["endurance"]] = r
    rows_all = [merged[E] for E in sorted(merged)]
    json.dump(rows_all, open(summary_path, "w"), indent=2)

    # summary figure: objective / ops / central visits vs endurance
    feas = [r for r in rows_all if r["objective"] is not None]
    if feas:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
        Es = [r["endurance"] for r in feas]
        axes[0].plot(Es, [r["objective"] for r in feas], "o-", color="#003366")
        axes[0].set_xlabel("Endurance (J)")
        axes[0].set_ylabel("Objective (m)")
        axes[0].grid(True, linestyle="--", alpha=0.4)
        axes[1].plot(Es, [r["n_ops"] for r in feas], "s-", color="#BA0C2F")
        axes[1].set_xlabel("Endurance (J)")
        axes[1].set_ylabel("# operations")
        axes[1].set_ylim(bottom=0)
        axes[1].grid(True, linestyle="--", alpha=0.4)
        axes[2].plot(Es, [r["total_visits"] for r in feas], "^-", color="#C4A35A")
        axes[2].set_xlabel("Endurance (J)")
        axes[2].set_ylabel("Central visits (interruptions)")
        axes[2].set_ylim(bottom=0)
        axes[2].grid(True, linestyle="--", alpha=0.4)
        fig.suptitle(f"Demo sweep K={args.split_points}: how solutions change with endurance")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f"sweep_summary_K{args.split_points}.png"),
                    dpi=150)
        plt.close(fig)
    logger.info("Sweep done. Results in %s/", OUT_DIR)
    print("\nSWEEP SUMMARY")
    for r in rows:
        print(r)


def load_solution_json(path: str):
    """Load a Solution from a solution JSON file (for warm starts)."""
    from drone_cpp.data_structures import Operation as _Op
    from drone_cpp.data_structures import Solution as _Sol
    import json as _json
    d = _json.load(open(path, encoding="utf-8"))
    vt = {e.name: e for e in VertexType}
    ops = []
    for op in d["operations"]:
        ops.append(_Op(edges=[(Vertex(e["r"], e["i"], vt[e["t"]]),
                              Vertex(e["r2"], e["i2"], vt[e["t2"]]))
                             for e in op]))
    vp, vl, vr = {}, {}, {}
    for k, pp in d["vertex_positions"].items():
        r, i, t = k.split(",")
        vp[Vertex(int(r), int(i), vt[t])] = Point3D(pp["x"], pp["y"], pp["z"])
    for k, lam in d["vertex_lambdas"].items():
        r, i, t = k.split(",")
        vl[Vertex(int(r), int(i), vt[t])] = lam
    for k, ri in d.get("vertex_rings", {}).items():
        r, i, t = k.split(",")
        vr[Vertex(int(r), int(i), vt[t])] = ri
    return _Sol(operations=ops, objective_value=d["objective_value"],
                vertex_positions=vp,
                chain_selection={int(k): v for k, v in d["chain_selection"].items()},
                vertex_lambdas=vl, vertex_rings=vr)


def solve(inst, K, tl, gap, warm=None):
    model = build_model(inst, "rings", verbose=False, num_split_points=K)
    try:
        import json as _json
        if warm is not None and K > 1:
            sol1 = load_solution_json(warm)
            model.set_warm_start_from_k1(sol1)
            logger.info("Warm-started K=%d from %s", K, warm)
    except Exception as ex:
        logger.warning("Warm start failed: %s", ex)
    model.model.setParam("MIPGap", gap)
    t0 = time.time()
    sol = model.optimize(tl=tl)
    return sol, model, time.time() - t0


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    if args.sweep:
        run_sweep(args)
        return

    if args.sweep_endurance:
        for E in [200.0, 250.0, 300.0]:
            inst = build_demo_instance(E, num_ops=args.num_ops)
            sol, model, el = solve(inst, 1, 60.0, 0.02)
            vv, tot = count_central_visits(sol, 0) if sol else ([], 0)
            logger.info("E=%.0f K=1 -> %s", E,
                        ("obj=%.1f ops=%d visits=%s" % (
                            sol.objective_value, len(sol.operations), vv)
                         if sol else "INFEASIBLE"))
        return

    inst = build_demo_instance(args.endurance, num_ops=args.num_ops)
    info = [(r.id, len(r.chains[0].rings),
             round(r.chains[0].rings[0].perimeter, 1)) for r in inst.regions]
    logger.info("Instance rings: %s (id, n_rings, outer perim)", info)

    sol, model, el = solve(inst, args.split_points, args.time_limit,
                           args.mip_gap, warm=args.k1_sol)
    if sol is None:
        logger.error("No solution found")
        return
    vv, tot = count_central_visits(sol, 0)
    logger.info("K=%d E=%.0f -> obj=%.1f m ops=%d gap=%.1f%% time=%.1fs visits=%s (total %d)",
                args.split_points, args.endurance, sol.objective_value,
                len(sol.operations), model.model.MIPGap * 100, el, vv, tot)
    sol.save(os.path.join(OUT_DIR,
                          f"demo_E{int(args.endurance):04d}_K{args.split_points}.json"))

    fig1 = CPPVis.plot_instance(inst, chain_selection=sol.chain_selection)
    fig1.savefig(os.path.join(OUT_DIR, "demo_instance.png"), dpi=150)
    plt.close(fig1)
    fig2 = CPPVis.plot_solution_2d(
        inst, sol, figsize=(10, 8),
        title=(f"Demo interruptions — K={args.split_points}, E={args.endurance:.0f}J | "
               f"{len(sol.operations)} op(s), {sol.objective_value:.1f} m, "
               f"{tot} central visits"))
    fig2.savefig(os.path.join(OUT_DIR, "demo_solution.png"), dpi=150)
    plt.close(fig2)
    logger.info("Saved plots + solution in %s/", OUT_DIR)
    print(f"CENTRAL VISITS per op: {vv} (total {tot})")


if __name__ == "__main__":
    main()