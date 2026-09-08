#!/usr/bin/env python
"""Validate the wind-aware RingsModel on the showcase instance.

1. Calibrate E_min under TRUE wind-aware accounting (cheapest single-region
   tour per region, then max) using the model's own precomputed coefficients.
2. Solve wind-aware K=1 at a tight and a loose level, with strong wind
   (8 m/s east) and calm (0 m/s): does wind change tour STRUCTURE?
3. Solve wind-aware K=2 at the tight level (splits still work?).
4. Reference: unaware K=1 at equivalent relative tightness.

Usage:
    python scripts/validate_wind.py
"""
from __future__ import annotations
import sys, os, json, time, logging

import numpy as np

sys.path.insert(0, ".")

from drone_cpp.data_structures import Instance, DroneParams, WindParams
from drone_cpp.model import build_model
from sweep_endurance import build_base

logger = logging.getLogger("drone_cpp.validate_wind")

OUT_DIR = "wind_validation"


def estimate_e_min_true(inst) -> tuple:
    """Cheapest single-region tour under TRUE wind-aware accounting.

    Uses a throwaway wind-aware K=1 model for its precomputed coefficients:
    depot legs with (F_out/F_ret, Vz), ring loops with min(fwd,bwd) energy,
    inter-ring gaps with the depot-direction proxy factor.
    Returns (E_min, per_region dict).
    """
    probe = build_model(inst, "rings", verbose=False, num_split_points=1,
                        wind_aware=True)
    dep = inst.depot
    worst, per_r = 0.0, {}
    for r in inst.regions:
        centroid = np.mean([[p.x, p.y] for p in r.boundary], axis=0)
        dxy = float(np.linalg.norm(centroid - np.array([dep.x, dep.y])))
        ti = 0
        ch = r.chains[ti]
        fout, fret, vz = probe._depot_F[r.id][ti]
        e = fout * dxy + vz * ch.height  # depot -> region (ascent)
        rings = ch.rings
        for ri, ring in enumerate(rings):
            ef = sum(probe._seg_C[(r.id, ti, ri, si)][0] * ring.segments[si].length
                     for si in range(len(ring.segments)))
            eb = sum(probe._seg_C[(r.id, ti, ri, si)][1] * ring.segments[si].length
                     for si in range(len(ring.segments)))
            e += min(ef, eb)
        # inter-ring gaps (proxy factor, like compare_methods)
        pts = np.array([[p.x, p.y] for p in r.boundary])
        max_radius = float(max(np.linalg.norm(p - centroid) for p in pts))
        for i in range(len(rings) - 1):
            s1 = rings[i].scale if hasattr(rings[i], "scale") else 1.0
            s2 = rings[i + 1].scale if hasattr(rings[i + 1], "scale") else 1.0
            e += fout * abs(s1 - s2) * max_radius
        e += fret * dxy  # region -> depot (descent, no vertical)
        per_r[r.id] = round(e, 2)
        worst = max(worst, e)
    return worst, per_r


def with_endurance(inst, E: float) -> Instance:
    drone = DroneParams(front_area=inst.drone.front_area,
                        drag_coef=inst.drone.drag_coef, max_endurance=float(E),
                        cruise_speed=inst.drone.cruise_speed,
                        vertical_speed=inst.drone.vertical_speed)
    return Instance(regions=inst.regions, depot=inst.depot, drone=drone,
                    wind=inst.wind, num_operations=inst.num_operations)


def with_wind(inst, speed: float, direction) -> Instance:
    wind = WindParams(direction=np.array(direction, dtype=float),
                      speed_at_10m=float(speed), hellmann_exponent=0.2)
    return Instance(regions=inst.regions, depot=inst.depot, drone=inst.drone,
                    wind=wind, num_operations=inst.num_operations)


def run(inst, E: float, K: int, wa: bool, tl: float, gap: float,
        warm=None, tag: str = ""):
    inst_run = with_endurance(inst, E)
    model = build_model(inst_run, "rings", verbose=False,
                        num_split_points=K, wind_aware=wa)
    summ = model.variable_summary()
    if warm is not None and K > 1:
        model.set_warm_start_from_k1(warm)
    model.model.setParam("MIPGap", gap)
    t0 = time.time()
    sol = model.optimize(tl=tl)
    el = time.time() - t0
    res = {"tag": tag, "E": E, "K": K, "wind_aware": wa,
           "objective": round(sol.objective_value, 2) if sol else None,
           "n_ops": len(sol.operations) if sol else None,
           "gap_pct": round(model.model.MIPGap * 100, 2) if sol else None,
           "time_s": round(el, 1),
           "status": sol.status if sol else "NONE",
           "n_vars": summ["binary"] + summ["continuous"] + summ["integer"]}
    logger.info("%s -> obj=%s ops=%s gap=%s%% time=%.0fs status=%s", tag,
                res["objective"], res["n_ops"], res["gap_pct"], el, res["status"])
    return sol, model, res


def tour_structure(sol):
    """Compact per-op ring assignment for comparison."""
    out = []
    for op in sol.operations:
        rings = sorted({(u.region_id, sol.vertex_rings.get(u, -1))
                        for (u, v) in op.edges if u.region_id >= 0}
                       | {(v.region_id, sol.vertex_rings.get(v, -1))
                          for (u, v) in op.edges if v.region_id >= 0})
        out.append(rings)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    os.makedirs(OUT_DIR, exist_ok=True)
    base = build_base(42)

    # controlled wind scenarios (direction east, constant)
    windy = with_wind(base, 8.0, [1.0, 0.0, 0.0])
    calm = with_wind(base, 0.0, [1.0, 0.0, 0.0])

    e_min, per_r = estimate_e_min_true(windy)
    logger.info("E_min_true (wind 8 m/s east) = %.2f J  per region: %s",
                e_min, per_r)
    e_tight = float(np.ceil(e_min * 1.5))
    e_loose = float(np.ceil(e_min * 3.0))
    logger.info("Levels: tight=%.0f J loose=%.0f J", e_tight, e_loose)

    rows = []
    sols = {}
    # wind-aware K=1: tight + loose, windy vs calm
    for E, lvl in [(e_tight, "tight"), (e_loose, "loose")]:
        for inst, wtag in [(windy, "wind8"), (calm, "calm")]:
            sol, model, res = run(inst, E, 1, True, 120.0, 0.02,
                                  tag=f"WA K=1 {lvl} {wtag}")
            res["tour"] = tour_structure(sol) if sol else None
            rows.append(res)
            sols[(lvl, wtag)] = sol

    # wind-aware K=2 at tight level (windy), warm-started from K=1
    sol, model, res = run(windy, e_tight, 2, True, 180.0, 0.02,
                          warm=sols[("tight", "wind8")],
                          tag="WA K=2 tight wind8")
    res["tour"] = tour_structure(sol) if sol else None
    rows.append(res)

    # unaware K=1 reference at same RELATIVE tightness (pseudo units):
    # pseudo E_min ~ 153 J (known) -> 1.5x = 230 J
    sol, model, res = run(calm, 230.0, 1, False, 120.0, 0.02,
                          tag="unaware K=1 pseudo-tight")
    res["tour"] = tour_structure(sol) if sol else None
    rows.append(res)

    json.dump(rows, open(os.path.join(OUT_DIR, "wind_validation.json"), "w"),
              indent=2, default=str)
    print("\nVALIDATION SUMMARY")
    for r in rows:
        print(f"  {r['tag']}: E={r['E']} obj={r['objective']} ops={r['n_ops']} "
              f"gap={r['gap_pct']}% time={r['time_s']}s tour={r['tour']}")
    print(f"E_min_true={e_min:.2f} J per_region={per_r}")


if __name__ == "__main__":
    main()