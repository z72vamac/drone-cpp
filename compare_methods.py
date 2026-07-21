"""Compare solver variants across instance sizes and endurance levels.

Runs three methods on every instance:
  1. heuristic   - greedy + 2-opt standalone solver
  2. rings       - Gurobi MIQP without warm start
  3. rings_ws    - Gurobi MIQP warm-started with the heuristic solution

Phase 0 (--calibrate): estimate the global minimum endurance needed to visit
all rings of any region (across all configurations and seeds) without Gurobi.
The sweep uses endurance levels derived from this global minimum.

Phase 1 (default): sweep all (regions x heights x seeds x endurance) combos,
run all three methods, and log results incrementally to results.csv.
Supports --resume to skip already-completed runs.

Usage:
    python compare_methods.py --calibrate          # Phase 0 only (~2 min)
    python compare_methods.py                      # Full sweep
    python compare_methods.py --resume             # Skip completed runs
    python compare_methods.py --time-limit 600     # Shorter Gurobi limit
"""
from __future__ import annotations
import sys, os, csv, json, time, argparse, logging

import numpy as np
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, ".")

from drone_cpp.instance_generator import InstanceGenerator
from drone_cpp.data_structures import (
    Instance, DroneParams, Solution, AtmosphereParams,
)
from drone_cpp.model import build_model
from drone_cpp.heuristic import HeuristicSolver
from drone_cpp.config import (
    DEFAULT_AREA_BOUNDS, DEFAULT_REGION_SIZES, DEFAULT_SPACING,
)

logger = logging.getLogger("drone_cpp.compare_methods")

REGION_COUNTS = [1, 2, 3, 5, 8, 10]
HEIGHT_LEVELS = [1, 2, 3]
SEEDS = [0, 1, 2, 3, 4]
ENDURANCE_FACTORS = [1.0, 1.5, 2.0, 3.0, 5.0, 8.0]

HEIGHTS_BY_LEVEL = {
    1: [30.0],
    2: [20.0, 50.0],
    3: [15.0, 35.0, 60.0],
}

OUT_DIR = "compare_results"
CSV_PATH = os.path.join(OUT_DIR, "results.csv")
DONE_PATH = os.path.join(OUT_DIR, "completed_runs.txt")
ENDURANCE_JSON = os.path.join(OUT_DIR, "endurance_levels.json")
SOLUTIONS_DIR = os.path.join(OUT_DIR, "solutions")

CSV_FIELDS = [
    "num_regions", "num_heights", "seed", "endurance", "method",
    "objective", "solve_time", "mip_gap",
    "first_incumbent_obj", "first_incumbent_time",
    "heuristic_obj", "num_ops", "status", "n_vars", "n_constrs",
]


def build_instance(num_regions, num_heights, seed, endurance=1e9):
    """Build an instance with num_ops = num_regions and constant heights.

    The area side is scaled by sqrt(num_regions / 3) and, if placement of
    non-overlapping regions still fails, progressively expanded until the
    generator succeeds (large regions of radius up to 25 with margin 10 can
    occasionally collide for some seeds).
    """
    side = 100.0 * np.sqrt(num_regions / 3.0)
    base = None
    for attempt in range(50):
        area = (0.0, 0.0, side, side)
        try:
            base = InstanceGenerator.generate(
                num_regions=num_regions,
                area_bounds=area,
                min_region_size=DEFAULT_REGION_SIZES[0],
                max_region_size=DEFAULT_REGION_SIZES[1],
                spacing=DEFAULT_SPACING,
                num_heights=1,
                min_height=15.0,
                max_height=65.0,
                num_ops=num_regions,
                seed=seed,
            )
            break
        except RuntimeError:
            side *= 1.25
    if base is None:
        raise RuntimeError(
            "Could not place %d regions even after expanding area (seed=%d)"
            % (num_regions, seed))
    heights = HEIGHTS_BY_LEVEL[num_heights]
    region_heights = {r.id: heights for r in base.regions}
    inst = InstanceGenerator.rebuild_instance(base, region_heights=region_heights)
    drone = DroneParams(
        front_area=base.drone.front_area,
        drag_coef=base.drone.drag_coef,
        max_endurance=endurance,
        cruise_speed=base.drone.cruise_speed,
        vertical_speed=base.drone.vertical_speed,
    )
    return Instance(regions=inst.regions, depot=inst.depot, drone=drone,
                    wind=inst.wind, num_operations=num_regions)


def _estimate_region_energy(inst, region, chain):
    """Energy (model units) for one operation covering all rings of *chain*.

    Matches the model's endurance accounting:
      depot->region :  En_xy * dxy + En_z * height   (ascent)
      ring traversal:  perimeter                      (model intra-ring energy)
      inter-ring    :  dxy only, no En_xy             (model inter-ring energy)
      region->depot :  En_xy * dxy                    (descent, no En_z)
    """
    depot = inst.depot
    drone = inst.drone
    wind = inst.wind
    rho0 = AtmosphereParams.air_density(0.0)

    centroid = np.mean([[p.x, p.y] for p in region.boundary], axis=0)
    dep_vec = np.array([centroid[0] - depot.x, centroid[1] - depot.y])
    nrm = float(np.linalg.norm(dep_vec))
    dir_unit = dep_vec / nrm if nrm > 0 else np.array([1.0, 0.0])
    ws0 = wind.speed_at_height(0.0)
    wdir = wind.direction[:2]
    if np.linalg.norm(wdir) > 0:
        wdir = wdir / np.linalg.norm(wdir)
    else:
        wdir = np.array([1.0, 0.0])
    nu_d_xy = float(np.linalg.norm(drone.cruise_speed * dir_unit - ws0 * wdir))
    En_xy = drone.E_xy * rho0 * nu_d_xy
    En_z = 0.5 * drone.E_z * drone.vertical_speed * rho0

    dxy_dep = nrm
    energy = En_xy * dxy_dep + En_z * chain.height

    rings = chain.rings
    for ring in rings:
        energy += ring.perimeter

    boundary_pts = np.array([[p.x, p.y] for p in region.boundary])
    max_radius = float(max(np.linalg.norm(p - centroid) for p in boundary_pts))
    for i in range(len(rings) - 1):
        energy += abs(rings[i].scale - rings[i + 1].scale) * max_radius

    energy += En_xy * dxy_dep
    return energy


def estimate_instance_e_min(inst):
    """Min endurance so each region fits in one operation (cheapest chain)."""
    worst = 0.0
    for region in inst.regions:
        best = float("inf")
        for chain in region.chains:
            if not chain.rings:
                continue
            e = _estimate_region_energy(inst, region, chain)
            if e < best:
                best = e
        if best < float("inf"):
            worst = max(worst, best)
    return worst


def calibrate(out_dir=OUT_DIR):
    """Phase 0: compute global E_min across all configs and seeds."""
    os.makedirs(out_dir, exist_ok=True)
    logger.info("=" * 60)
    logger.info("PHASE 0: Calibrating global minimum endurance (no Gurobi)")
    logger.info("=" * 60)
    global_e_min = 0.0
    per_config = {}
    for nr in REGION_COUNTS:
        for nh in HEIGHT_LEVELS:
            worst = 0.0
            for seed in SEEDS:
                inst = build_instance(nr, nh, seed)
                e = estimate_instance_e_min(inst)
                if e > worst:
                    worst = e
            key = "r%02d_h%d" % (nr, nh)
            per_config[key] = worst
            logger.info("  %s  E_min = %10.2f", key, worst)
            if worst > global_e_min:
                global_e_min = worst
    rounded = int(np.ceil(global_e_min * 1.05 / 50.0) * 50)
    values = [int(np.ceil(rounded * f)) for f in ENDURANCE_FACTORS]
    logger.info("-" * 60)
    logger.info("Global E_min (raw)     = %.2f", global_e_min)
    logger.info("Global E_min (rounded) = %d", rounded)
    logger.info("Endurance levels       = %s", values)
    data = {
        "global_e_min_raw": global_e_min,
        "global_e_min_rounded": rounded,
        "values": values,
        "per_config": per_config,
    }
    with open(os.path.join(out_dir, "endurance_levels.json"), "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Saved %s", os.path.join(out_dir, "endurance_levels.json"))
    return values


def load_or_calibrate(force=False):
    if not force and os.path.exists(ENDURANCE_JSON):
        with open(ENDURANCE_JSON) as f:
            data = json.load(f)
        logger.info("Loaded endurance levels from %s: %s",
                    ENDURANCE_JSON, data["values"])
        return data["values"]
    return calibrate()


def _sol_path(nr, nh, seed, E, method):
    name = "r%02d_h%d_s%d_e%04d_%s.json" % (nr, nh, seed, E, method)
    return os.path.join(SOLUTIONS_DIR, name)


def _row(nr, nh, seed, E, method, **kw):
    row = dict.fromkeys(CSV_FIELDS, "")
    row.update({"num_regions": nr, "num_heights": nh, "seed": seed,
                "endurance": E, "method": method})
    row.update(kw)
    return row


def run_sweep(endurance_values, resume=False, time_limit=1800.0, mip_gap=0.1):
    """Phase 1: run all 3 methods on every (nr, nh, seed, E) combo."""
    os.makedirs(SOLUTIONS_DIR, exist_ok=True)

    done = set()
    if resume and os.path.exists(DONE_PATH):
        with open(DONE_PATH) as f:
            done = set(line.strip() for line in f if line.strip())
        logger.info("Resume mode: %d runs already completed", len(done))

    write_header = not os.path.exists(CSV_PATH)
    csv_f = open(CSV_PATH, "a", newline="")
    writer = csv.DictWriter(csv_f, fieldnames=CSV_FIELDS)
    if write_header:
        writer.writeheader()
    done_f = open(DONE_PATH, "a")

    def mark_done(key):
        done_f.write(key + "\n")
        done_f.flush()
        done.add(key)

    def emit(row):
        writer.writerow(row)
        csv_f.flush()

    total = len(REGION_COUNTS) * len(HEIGHT_LEVELS) * len(SEEDS) \
        * len(endurance_values) * 3
    idx = 0
    for nr in REGION_COUNTS:
        for nh in HEIGHT_LEVELS:
            for seed in SEEDS:
                base_inst = build_instance(nr, nh, seed)
                for E in endurance_values:
                    drone = DroneParams(
                        front_area=base_inst.drone.front_area,
                        drag_coef=base_inst.drone.drag_coef,
                        max_endurance=float(E),
                        cruise_speed=base_inst.drone.cruise_speed,
                        vertical_speed=base_inst.drone.vertical_speed,
                    )
                    inst = Instance(regions=base_inst.regions,
                                    depot=base_inst.depot, drone=drone,
                                    wind=base_inst.wind,
                                    num_operations=base_inst.num_operations)

                    # ---- heuristic (always recompute; needed for warm start) ----
                    sol_h = None
                    t0 = time.time()
                    try:
                        sol_h = HeuristicSolver(inst).solve()
                    except Exception as ex:
                        logger.error("  heuristic FAILED %s: %s",
                                     (nr, nh, seed, E), ex)
                    heur_time = time.time() - t0

                    key_h = "%d,%d,%d,%d,heuristic" % (nr, nh, seed, E)
                    idx += 1
                    if key_h not in done:
                        logger.info("[%d/%d] r%d h%d s%d e%d  heuristic",
                                    idx, total, nr, nh, seed, E)
                        if sol_h is not None:
                            sol_h.save(_sol_path(nr, nh, seed, E, "heuristic"))
                            emit(_row(nr, nh, seed, E, "heuristic",
                                      objective="%.4f" % sol_h.objective_value,
                                      solve_time="%.2f" % heur_time,
                                      num_ops=len(sol_h.operations),
                                      status="HEURISTIC"))
                        else:
                            emit(_row(nr, nh, seed, E, "heuristic",
                                      solve_time="%.2f" % heur_time,
                                      status="FAILED"))
                        mark_done(key_h)

                    # ---- rings (no warm start) ----
                    key_r = "%d,%d,%d,%d,rings" % (nr, nh, seed, E)
                    idx += 1
                    if key_r not in done:
                        logger.info("[%d/%d] r%d h%d s%d e%d  rings",
                                    idx, total, nr, nh, seed, E)
                        try:
                            model = build_model(inst, "rings", verbose=False)
                            model.model.setParam("MIPGap", mip_gap)
                            nv = model.model.NumVars
                            nc = model.model.NumConstrs
                            t0 = time.time()
                            sol_r = model.optimize(tl=time_limit)
                            elapsed = time.time() - t0
                            if sol_r is not None:
                                sol_r.save(_sol_path(nr, nh, seed, E, "rings"))
                                emit(_row(
                                    nr, nh, seed, E, "rings",
                                    objective="%.4f" % sol_r.objective_value,
                                    solve_time="%.2f" % elapsed,
                                    mip_gap=("%.4f" % sol_r.mip_gap
                                             if sol_r.mip_gap is not None else ""),
                                    first_incumbent_obj=(
                                        "%.4f" % sol_r.first_incumbent_obj
                                        if sol_r.first_incumbent_obj is not None else ""),
                                    first_incumbent_time=(
                                        "%.2f" % sol_r.first_incumbent_time
                                        if sol_r.first_incumbent_time is not None else ""),
                                    num_ops=len(sol_r.operations),
                                    status=sol_r.status or "?",
                                    n_vars=nv, n_constrs=nc))
                            else:
                                emit(_row(nr, nh, seed, E, "rings",
                                          solve_time="%.2f" % elapsed,
                                          status="INFEASIBLE",
                                          n_vars=nv, n_constrs=nc))
                        except Exception as ex:
                            logger.error("  rings FAILED %s: %s",
                                         (nr, nh, seed, E), ex)
                            emit(_row(nr, nh, seed, E, "rings", status="ERROR"))
                        mark_done(key_r)

                    # ---- rings_ws (warm start with heuristic) ----
                    key_w = "%d,%d,%d,%d,rings_ws" % (nr, nh, seed, E)
                    idx += 1
                    if key_w not in done:
                        logger.info("[%d/%d] r%d h%d s%d e%d  rings_ws",
                                    idx, total, nr, nh, seed, E)
                        if sol_h is None:
                            emit(_row(nr, nh, seed, E, "rings_ws",
                                      status="SKIPPED_NO_HEURISTIC"))
                        else:
                            try:
                                model = build_model(inst, "rings", verbose=False)
                                model.model.setParam("MIPGap", mip_gap)
                                nv = model.model.NumVars
                                nc = model.model.NumConstrs
                                model.set_warm_start(sol_h)
                                t0 = time.time()
                                sol_w = model.optimize(tl=time_limit)
                                elapsed = time.time() - t0
                                if sol_w is not None:
                                    sol_w.save(_sol_path(nr, nh, seed, E, "rings_ws"))
                                    emit(_row(
                                        nr, nh, seed, E, "rings_ws",
                                        objective="%.4f" % sol_w.objective_value,
                                        solve_time="%.2f" % elapsed,
                                        mip_gap=("%.4f" % sol_w.mip_gap
                                                 if sol_w.mip_gap is not None else ""),
                                        first_incumbent_obj=(
                                            "%.4f" % sol_w.first_incumbent_obj
                                            if sol_w.first_incumbent_obj is not None else ""),
                                        first_incumbent_time=(
                                            "%.2f" % sol_w.first_incumbent_time
                                            if sol_w.first_incumbent_time is not None else ""),
                                        heuristic_obj="%.4f" % sol_h.objective_value,
                                        num_ops=len(sol_w.operations),
                                        status=sol_w.status or "?",
                                        n_vars=nv, n_constrs=nc))
                                else:
                                    emit(_row(nr, nh, seed, E, "rings_ws",
                                              solve_time="%.2f" % elapsed,
                                              status="INFEASIBLE",
                                              heuristic_obj="%.4f" % sol_h.objective_value,
                                              n_vars=nv, n_constrs=nc))
                            except Exception as ex:
                                logger.error("  rings_ws FAILED %s: %s",
                                             (nr, nh, seed, E), ex)
                                emit(_row(nr, nh, seed, E, "rings_ws",
                                          status="ERROR",
                                          heuristic_obj="%.4f" % sol_h.objective_value))
                        mark_done(key_w)

    csv_f.close()
    done_f.close()
    logger.info("Sweep complete. Results in %s", CSV_PATH)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--calibrate", action="store_true", default=False,
                   help="Run Phase 0 calibration only (recompute endurance levels)")
    p.add_argument("--resume", action="store_true", default=False,
                   help="Skip runs already recorded in completed_runs.txt")
    p.add_argument("--time-limit", type=float, default=1800.0,
                   help="Gurobi time limit per run in seconds (default 1800)")
    p.add_argument("--mip-gap", type=float, default=0.0,
                   help="Gurobi MIPGap fraction (default 0.0 = stop only at proven optimality)")
    return p.parse_args()


def main():
    args = parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    log_path = os.path.join(OUT_DIR, "run.log")
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    handlers = [logging.StreamHandler(sys.stdout),
                logging.FileHandler(log_path)]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)

    if args.calibrate:
        calibrate()
        return

    values = load_or_calibrate()
    run_sweep(values, resume=args.resume,
              time_limit=args.time_limit, mip_gap=args.mip_gap)


if __name__ == "__main__":
    main()
