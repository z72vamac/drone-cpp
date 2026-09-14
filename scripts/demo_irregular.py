#!/usr/bin/env python
"""Demo with IRREGULAR polygons: one large central + 4 satellites.

Same protocol as the square demo but with irregular (non-rectangular)
boundaries, to see which entry/exit (cut) points the model chooses.
K sweep 1..5 with long time limit, wind-aware model.

Usage:
    python scripts/demo_irregular.py --ks 1,2,3,4,5 --time-limit 7200
    python scripts/demo_irregular.py --endurance 200 --ks 2 (skip calibration)
"""
from __future__ import annotations
import sys, os, json, time, argparse, logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

from drone_cpp.data_structures import (
    Point3D, Segment, Ring, PolygonalChain, Region, Instance,
    DroneParams, WindParams, Vertex, VertexType,
)
from drone_cpp.instance_generator import InstanceGenerator
from drone_cpp.model import build_model
from drone_cpp.visualization import CPPVis
from demo_interruptions import count_central_visits
from make_animation_frames import load_solution

logger = logging.getLogger("drone_cpp.demo_irr")

OUT_DIR = "demo_irregular"


def _is_convex(pts) -> bool:
    n = len(pts)
    signs = []
    for i in range(n):
        a = pts[i]
        b = pts[(i + 1) % n]
        c = pts[(i + 2) % n]
        cr = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if abs(cr) > 1e-9:
            signs.append(np.sign(cr))
    return len(set(signs)) <= 1


def _poly(cx, cy, radii, angles):
    pts = [(cx + r * np.cos(a), cy + r * np.sin(a))
           for r, a in zip(radii, angles)]
    assert _is_convex(pts), f"non-convex polygon at {(cx, cy)}"
    return [Point3D(float(x), float(y)) for x, y in pts]


def _single_ring_chain(boundary, height, region_id):
    segs = [Segment(boundary[j], boundary[(j + 1) % len(boundary)])
            for j in range(len(boundary))]
    ring = Ring(segments=segs, scale=1.0, height=height)
    return PolygonalChain(segments=list(segs), height=height,
                          region_id=region_id, idx=0, rings=[ring])


def build_irregular_instance(endurance: float, num_ops: int = 4,
                             height: float = 30.0) -> Instance:
    import numpy as _np
    rng = _np.random.RandomState(11)
    regions = []

    # R0: irregular convex heptagon around (0,0)
    c_ang = [0.0, 0.9, 1.8, 2.7, 3.6, 4.5, 5.5]
    c_rad = [22.0, 18.0, 24.0, 19.0, 23.0, 17.0, 21.0]
    central = _poly(0.0, 0.0, c_rad, c_ang)
    ch0 = InstanceGenerator._generate_spiral_chain(
        central, height, 0, 0, rng, spacing=12.0)
    assert ch0 is not None and len(ch0.rings) >= 2, \
        f"central chain failed: {None if ch0 is None else len(ch0.rings)} rings"
    regions.append(Region(id=0, boundary=central, chains=[ch0],
                          num_interruption_points=3))

    # R1..R4: irregular pentagons (single ring each)
    sats = [
        (2.0, 33.0, [5.5, 4.2, 6.0, 4.8, 5.2], [0.2, 1.4, 2.6, 3.8, 5.0]),
        (-3.0, -33.0, [4.8, 5.8, 4.4, 5.5, 5.0], [0.5, 1.7, 2.9, 4.1, 5.3]),
        (33.0, -2.0, [5.2, 4.5, 5.9, 4.7, 5.4], [0.0, 1.3, 2.5, 3.7, 5.1]),
        (-33.0, 3.0, [4.6, 5.6, 4.9, 5.3, 4.3], [0.4, 1.6, 2.8, 4.0, 5.2]),
    ]
    for i, (cx, cy, radii, angles) in enumerate(sats, start=1):
        bpts = _poly(cx, cy, radii, angles)
        chain = _single_ring_chain(bpts, height, i)
        regions.append(Region(id=i, boundary=bpts, chains=[chain],
                              num_interruption_points=1))

    drone = DroneParams(front_area=0.1, drag_coef=0.3,
                        max_endurance=float(endurance),
                        cruise_speed=15.0, vertical_speed=5.0)
    wind = WindParams(direction=np.array([1.0, 0.0, 0.0]),
                      speed_at_10m=4.0, hellmann_exponent=0.2)
    return Instance(regions=regions, depot=Point3D(-55.0, -55.0, 0.0),
                    drone=drone, wind=wind, num_operations=num_ops)


def with_endurance(inst: Instance, E: float) -> Instance:
    drone = DroneParams(front_area=inst.drone.front_area,
                        drag_coef=inst.drone.drag_coef, max_endurance=float(E),
                        cruise_speed=inst.drone.cruise_speed,
                        vertical_speed=inst.drone.vertical_speed)
    return Instance(regions=inst.regions, depot=inst.depot, drone=drone,
                    wind=inst.wind, num_operations=inst.num_operations)


def cut_report(sol, inst, K: int) -> dict:
    """Per ring: arc intervals [lau, ret] and covering op (to see cut points)."""
    from collections import defaultdict
    vl = sol.vertex_lambdas
    # map (region, arc_idx) -> (lau_lambda, ret_lambda, op)
    arcs = defaultdict(dict)
    for oi, op in enumerate(sol.operations):
        for (u, v) in op.edges:
            if (u.region_id >= 0 and u.region_id == v.region_id
                    and u.vtype == VertexType.LAUNCH
                    and v.vtype == VertexType.RETRIEVE
                    and abs(u.idx - v.idx) == 1 and u.idx % 2 == 0):
                ri = (u.idx // 2) // K
                k = (u.idx // 2) % K
                arcs[(u.region_id, ri)][k] = (
                    round(vl.get(u, 0.0), 3), round(vl.get(v, 0.0), 3), oi)
    return {f"R{r}ring{ri}": {f"arc{k}": {"lau": a[0], "ret": a[1], "op": a[2]}
                              for k, a in sorted(v.items())}
            for (r, ri), v in sorted(arcs.items())}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ks", type=str, default="1,2,3,4,5")
    p.add_argument("--endurance", type=float, default=None,
                   help="Fixed endurance (default: auto-calibrate 1.5x E_min)")
    p.add_argument("--num-ops", type=int, default=4)
    p.add_argument("--time-limit", type=float, default=7200.0)
    p.add_argument("--mip-gap", type=float, default=0.02)
    p.add_argument("--threads", type=int, default=0,
                   help="Gurobi threads (0 = all available cores)")
    p.add_argument("--warm-from", type=str, default=None,
                   help="Solution JSON for identity warm start (same K/structure), "
                        "chained topologically for valid MTZ order")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    ks = [int(k) for k in args.ks.split(",")]
    os.makedirs(OUT_DIR, exist_ok=True)
    base = build_demo_instance_ref = build_irregular_instance(300.0,
                                                              num_ops=args.num_ops)
    info = [(r.id, len(r.chains[0].rings),
             round(r.chains[0].rings[0].perimeter, 1)) for r in base.regions]
    logger.info("Irregular instance rings: %s (id, n_rings, outer perim)", info)

    if args.endurance is None:
        from validate_wind import estimate_e_min_true
        e_min, per_r = estimate_e_min_true(base)
        E = float(np.ceil(e_min * 1.5))
        logger.info("E_min_true=%.2f %s -> tight E=%.0f J", e_min, per_r, E)
    else:
        E = float(args.endurance)
        logger.info("Fixed endurance E=%.0f J", E)

    rows = []
    sol_k1 = None
    warm_same = None
    if args.warm_from:
        from drone_cpp.data_structures import Operation as _Op
        warm_same = load_solution(args.warm_from)
        logger.info("Loaded identity warm start %s (obj=%.2f)",
                    args.warm_from, warm_same.objective_value)
        dep = base.depot_vertex
        _chained = []
        for _op in warm_same.operations:
            _by_u = {}
            for _e in _op.edges:
                _by_u.setdefault(_e[0], []).append(_e)
            _st = next((_e for _e in _op.edges if _e[0] == dep), None)
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
    for K in ks:
        inst = with_endurance(base, E)
        model = build_model(inst, "rings", verbose=False,
                            num_split_points=K, wind_aware=True)
        summ = model.variable_summary()
        logger.info("K=%d model size: %s", K, summ)
        if warm_same is not None:
            try:
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
            rows.append({"K": K, "E": E, "objective": None, "status": "NONE"})
            continue
        vv, tot = count_central_visits(sol, 0)
        cuts = cut_report(sol, inst, K)
        logger.info("K=%d -> obj=%.2f m ops=%d visits=%s gap=%.2f%% time=%.0fs status=%s "
                    "first=%.2f", K, sol.objective_value, len(sol.operations),
                    vv, model.model.MIPGap * 100, el, sol.status,
                    sol.first_incumbent_obj or -1)
        rows.append({"K": K, "E": E, "objective": round(sol.objective_value, 2),
                     "n_ops": len(sol.operations), "visits": vv,
                     "gap_pct": round(model.model.MIPGap * 100, 2),
                     "time_s": round(el, 1), "status": sol.status,
                     "first": sol.first_incumbent_obj,
                     "n_vars": summ["binary"] + summ["continuous"] + summ["integer"],
                     "cuts": cuts})
        sol.save(os.path.join(OUT_DIR, f"irr_E{int(E):04d}_K{K}.json"))
        fig = CPPVis.plot_solution_2d(
            inst, sol, figsize=(10, 8),
            title=(f"Irregular demo wind-aware K={K} E={E:.0f}J | "
                   f"{len(sol.operations)} op(s), {sol.objective_value:.1f} m"))
        fig.savefig(os.path.join(OUT_DIR, f"irr_E{int(E):04d}_K{K}.png"), dpi=150)
        plt.close(fig)
        if K == 1:
            sol_k1 = sol

    # merge into summary file
    summary_path = os.path.join(OUT_DIR, "irr_summary.json")
    merged = {}
    if os.path.exists(summary_path):
        try:
            for r in json.load(open(summary_path, encoding="utf-8")):
                merged[(r["E"], r["K"])] = r
        except Exception:
            pass
    for r in rows:
        merged[(r["E"], r["K"])] = r
    rows_all = [merged[k] for k in sorted(merged)]
    json.dump(rows_all, open(summary_path, "w"), indent=2)
    print("\nIRREGULAR K-SWEEP SUMMARY")
    for r in rows_all:
        print(" ", {k: v for k, v in r.items() if k != "cuts"})


if __name__ == "__main__":
    main()