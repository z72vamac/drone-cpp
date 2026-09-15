"""K-split battery: same testbed as compare_methods.py (Vertex-Approach),
but with num_split_points K >= 2.

For every (num_regions, num_heights, seed, endurance) config of the vertex
testbed and every K in --ks, runs two methods:
  1. ringsK{K}       - Gurobi MIQP with K arcs per ring, cold start
  2. ringsK{K}_wsk1  - same model warm-started from the K=1 (vertex) solution
                       stored in compare_results/solutions/ (loaded from disk,
                       no K=1 re-solve needed)

The heuristic is NOT re-run: heuristic rows already exist in
compare_results/results.csv and are K-agnostic.

Designed for multi-instance servers:
  - deterministic sharding (--shard-index / --num-shards) over the sorted
    config list; each shard writes its own CSV + completed file, so shards
    never write to the same file (safe for SLURM arrays);
  - --resume skips runs recorded in the shard's completed file;
  - --dry-run prints run counts without solving (capacity planning);
  - --threads caps Gurobi threads per run (set to cores-per-task on server).

Endurance levels are loaded from the vertex calibration JSON so both
batteries share identical E values (default compare_results/endurance_levels.json).

Recommended tiers (see server/README.md):
  Tier 1 (core): --ks 2,3 --nr 1,2,3            (full E x seeds: 540 runs/K... )
  Tier 2 (large): --ks 2 --nr 5,8,10 --E 300,2400 (tight+loose only)

Usage:
    python compare_methods_k.py --dry-run --ks 2,3 --nr 1,2,3
    python compare_methods_k.py --ks 2,3 --nr 1,2,3 --num-shards 8 --shard-index 0 --threads 4 --time-limit 900
    python compare_methods_k.py --ks 2 --nr 5,8,10 --E 300,2400 --resume
"""
from __future__ import annotations
import sys, os, csv, json, time, argparse, logging

sys.path.insert(0, ".")

from drone_cpp.data_structures import DroneParams, Instance, Solution
from drone_cpp.model import build_model
import compare_methods as cm

logger = logging.getLogger("drone_cpp.compare_methods_k")

OUT_DIR = "compare_results_k"

CSV_FIELDS = cm.CSV_FIELDS + ["k"]


def k1_sol_path(k1_dir, nr, nh, seed, E):
    name = "r%02d_h%d_s%d_e%04d_rings.json" % (nr, nh, seed, E)
    return os.path.join(k1_dir, name)


def _sol_path(out_solutions, nr, nh, seed, E, K, method):
    name = "r%02d_h%d_s%d_e%04d_ringsK%d_%s.json" % (
        nr, nh, seed, E, K, method)
    return os.path.join(out_solutions, name)


def _row(nr, nh, seed, E, K, method, **kw):
    row = dict.fromkeys(CSV_FIELDS, "")
    row.update({"num_regions": nr, "num_heights": nh, "seed": seed,
                "endurance": E, "method": method, "k": K})
    row.update(kw)
    return row


def build_grid(nr_sel, nh_sel, seed_sel, e_values, ks):
    grid = []
    for nr in nr_sel:
        for nh in nh_sel:
            for seed in seed_sel:
                for E in e_values:
                    for K in ks:
                        grid.append((nr, nh, seed, E, K))
    return sorted(grid)


def _parse_int_list(s, allowed):
    if s is None:
        return list(allowed)
    vals = [int(x) for x in s.split(",")]
    bad = [v for v in vals if v not in allowed]
    if bad:
        raise ValueError("Values %s not in allowed %s" % (bad, list(allowed)))
    return vals


def run_config(nr, nh, seed, E, K, args, emit, mark_done, done,
               out_solutions, idx, total):
    base_key = "%d,%d,%d,%d,K%d" % (nr, nh, seed, E, K)
    try:
        base_inst = cm.build_instance(nr, nh, seed)
    except Exception as ex:
        logger.error("  build FAILED %s: %s", (nr, nh, seed, E, K), ex)
        for method in ("cold", "wsk1"):
            key = base_key + "," + method
            if key not in done:
                emit(_row(nr, nh, seed, E, K, "ringsK%d_%s" % (K, method),
                           status="BUILD_ERROR"))
                mark_done(key)
        return
    drone = DroneParams(
        front_area=base_inst.drone.front_area,
        drag_coef=base_inst.drone.drag_coef,
        max_endurance=float(E),
        cruise_speed=base_inst.drone.cruise_speed,
        vertical_speed=base_inst.drone.vertical_speed,
    )
    inst = Instance(regions=base_inst.regions, depot=base_inst.depot,
                    drone=drone, wind=base_inst.wind,
                    num_operations=base_inst.num_operations)

    # ---- warm-start reference: K=1 vertex solution from disk ----
    sol_k1 = None
    k1_path = k1_sol_path(args.k1_solutions_dir, nr, nh, seed, E)
    if os.path.exists(k1_path):
        try:
            sol_k1 = Solution.load(k1_path)
        except Exception as ex:
            logger.warning("  K=1 load FAILED %s: %s", k1_path, ex)
            sol_k1 = None

    for method in ("cold", "wsk1"):
        key = base_key + "," + method
        idx[0] += 1
        if key in done:
            continue
        mname = "ringsK%d_%s" % (K, method)
        logger.info("[%d/%d] r%d h%d s%d e%d K%d %s",
                    idx[0], total, nr, nh, seed, E, K, method)
        if method == "wsk1" and sol_k1 is None:
            emit(_row(nr, nh, seed, E, K, mname, status="SKIPPED_NO_K1"))
            mark_done(key)
            continue
        try:
            model = build_model(inst, "rings", verbose=False,
                                num_split_points=K)
            model.model.setParam("MIPGap", args.mip_gap)
            model.model.setParam("Threads", args.threads)
            nv = model.model.NumVars
            nc = model.model.NumConstrs
            if method == "wsk1":
                model.set_warm_start_from_k1(sol_k1)
            t0 = time.time()
            sol = model.optimize(tl=args.time_limit)
            elapsed = time.time() - t0
            if sol is not None:
                sol.save(_sol_path(out_solutions, nr, nh, seed, E, K, method))
                emit(_row(
                    nr, nh, seed, E, K, mname,
                    objective="%.4f" % sol.objective_value,
                    solve_time="%.2f" % elapsed,
                    mip_gap=("%.4f" % sol.mip_gap
                             if sol.mip_gap is not None else ""),
                    first_incumbent_obj=(
                        "%.4f" % sol.first_incumbent_obj
                        if sol.first_incumbent_obj is not None else ""),
                    first_incumbent_time=(
                        "%.2f" % sol.first_incumbent_time
                        if sol.first_incumbent_time is not None else ""),
                    num_ops=len(sol.operations),
                    status=sol.status or "?",
                    n_vars=nv, n_constrs=nc))
            else:
                g_st = model.model.Status
                label = {3: "INFEASIBLE", 4: "INF_OR_UNBD",
                         8: "TIME_LIMIT_NO_SOL"}.get(
                             g_st, "GSTATUS_%d" % g_st)
                emit(_row(nr, nh, seed, E, K, mname,
                           solve_time="%.2f" % elapsed,
                           status=label, n_vars=nv, n_constrs=nc))
        except Exception as ex:
            logger.error("  %s FAILED %s: %s", mname, (nr, nh, seed, E, K), ex)
            emit(_row(nr, nh, seed, E, K, mname, status="ERROR"))
        mark_done(key)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__)
    p.add_argument("--ks", type=str, default="2,3",
                   help="Comma-separated K values (default '2,3')")
    p.add_argument("--nr", type=str, default=None,
                   help="Subset of region counts, e.g. '1,2,3' (default: all)")
    p.add_argument("--nh", type=str, default=None,
                   help="Subset of height levels, e.g. '1,2' (default: all)")
    p.add_argument("--seeds", type=str, default=None,
                   help="Subset of seeds, e.g. '0,1,2' (default: all)")
    p.add_argument("--E", type=str, default=None,
                   help="Subset of endurance values, e.g. '300,2400' "
                        "(default: all calibrated levels)")
    p.add_argument("--endurance-json", type=str,
                   default=os.path.join("compare_results",
                                        "endurance_levels.json"),
                   help="Calibration JSON with endurance levels")
    p.add_argument("--k1-solutions-dir", type=str,
                   default=os.path.join("compare_results", "solutions"),
                   help="Directory with K=1 vertex solutions for warm start")
    p.add_argument("--out-dir", type=str, default=OUT_DIR,
                   help="Output directory (default compare_results_k)")
    p.add_argument("--shard-index", type=int, default=0,
                   help="Shard to run (default 0)")
    p.add_argument("--num-shards", type=int, default=1,
                   help="Total number of shards (default 1)")
    p.add_argument("--resume", action="store_true", default=False,
                   help="Skip runs recorded in the shard completed file")
    p.add_argument("--dry-run", action="store_true", default=False,
                   help="Print run counts without solving")
    p.add_argument("--time-limit", type=float, default=900.0,
                   help="Gurobi time limit per run in seconds (default 900; "
                        "vertex battery used 1800)")
    p.add_argument("--mip-gap", type=float, default=0.02,
                   help="Gurobi MIPGap fraction (default 0.02; vertex used 0.0)")
    p.add_argument("--threads", type=int, default=0,
                   help="Gurobi Threads per run (default 0 = automatic; set to "
                        "cores-per-task when sharing a server node)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir,
                            "run_shard%03d.log" % args.shard_index)
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    handlers = [logging.StreamHandler(sys.stdout)]
    if not args.dry_run:
        handlers.append(logging.FileHandler(log_path))
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)

    with open(args.endurance_json) as f:
        e_all = json.load(f)["values"]
    ks = [int(k) for k in args.ks.split(",")]
    if any(k < 2 for k in ks):
        raise ValueError("This battery targets K>=2 (K=1 is the vertex "
                         "battery in compare_methods.py)")
    nr_sel = _parse_int_list(args.nr, cm.REGION_COUNTS)
    nh_sel = _parse_int_list(args.nh, cm.HEIGHT_LEVELS)
    seed_sel = _parse_int_list(args.seeds, cm.SEEDS)
    e_sel = args.E.split(",") if args.E else None
    e_values = [int(e) for e in e_sel] if e_sel else list(e_all)
    bad_e = [e for e in e_values if e not in e_all]
    if bad_e:
        raise ValueError("Endurance values %s not in calibrated %s"
                         % (bad_e, e_all))

    grid = build_grid(nr_sel, nh_sel, seed_sel, e_values, ks)
    shard = sorted(grid)[args.shard_index::args.num_shards]
    n_runs = len(shard) * 2  # cold + wsk1 per (config, K)
    logger.info("K-battery: %d configs x %d methods = %d runs "
                "(shard %d/%d, ks=%s, nr=%s, E=%s, tl=%.0f, gap=%.2f)",
                len(shard), 2, n_runs, args.shard_index, args.num_shards,
                ks, nr_sel, e_values, args.time_limit, args.mip_gap)
    if args.dry_run:
        per_k = {}
        for (_, _, _, _, K) in shard:
            per_k[K] = per_k.get(K, 0) + 2
        for K in sorted(per_k):
            logger.info("  K=%d: %d runs", K, per_k[K])
        return

    csv_path = os.path.join(args.out_dir,
                            "results_shard%03d.csv" % args.shard_index)
    done_path = os.path.join(args.out_dir,
                             "completed_shard%03d.txt" % args.shard_index)
    out_solutions = os.path.join(args.out_dir, "solutions")
    os.makedirs(out_solutions, exist_ok=True)

    done = set()
    if args.resume and os.path.exists(done_path):
        with open(done_path) as f:
            done = set(line.strip() for line in f if line.strip())
        logger.info("Resume mode: %d runs already completed", len(done))

    write_header = not os.path.exists(csv_path)
    csv_f = open(csv_path, "a", newline="")
    writer = csv.DictWriter(csv_f, fieldnames=CSV_FIELDS)
    if write_header:
        writer.writeheader()
    done_f = open(done_path, "a")

    def mark_done(key):
        done_f.write(key + "\n")
        done_f.flush()
        done.add(key)

    def emit(row):
        writer.writerow(row)
        csv_f.flush()

    idx = [0]
    for (nr, nh, seed, E, K) in shard:
        run_config(nr, nh, seed, E, K, args, emit, mark_done, done,
                   out_solutions, idx, n_runs)

    csv_f.close()
    done_f.close()
    logger.info("Shard %d complete. Results in %s",
                args.shard_index, csv_path)


if __name__ == "__main__":
    main()
