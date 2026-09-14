#!/usr/bin/env python
"""Animation frames (+ GIF) for the demo-interruptions sweep.

For every demo solution (demo_interruptions/sweep_E*_K2.json) renders one PNG
per path segment, chaining each operation topologically from the depot so the
replay shows exactly which path each operation takes (as in the showcase
anim_frames_*). Also assembles one GIF per level for quick viewing.

Output: demo_anim/E0180_K2/frame_*.png + demo_anim/E0180_K2.gif, ...

Usage:
    python scripts/make_demo_animation.py
    python scripts/make_demo_animation.py --levels 180,250,350
"""
from __future__ import annotations
import sys, os, json, argparse, logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

from drone_cpp.visualization import CPPVis
from demo_interruptions import build_demo_instance
from make_animation_frames import (
    load_solution, edge_segments, chain_edges,
)

logger = logging.getLogger("drone_cpp.demo_anim")

OUT_DIR = "demo_anim"
LEVELS = [170, 180, 200, 220, 250, 300, 350, 400]


def make_frames(end, inst, sol, out_dir: str, tag: str = None) -> int:
    os.makedirs(out_dir, exist_ok=True)
    colors, op_colors, sel_chain, ring_map, cum_map, get_cum = \
        CPPVis._solution_setup(inst, sol)
    title_base = tag or f"Demo E={end}J"

    plan = []  # (op_idx, seg)
    for op_idx, op in enumerate(sol.operations):
        for (u, v) in chain_edges(op.edges, inst.depot_vertex):
            for s in edge_segments(inst, sol, op_idx, u, v,
                                   sel_chain, get_cum, ring_map):
                plan.append((op_idx, s))
    n = len(plan) + 1  # frame 1 = base plot, then one frame per segment

    fig, ax = plt.subplots(figsize=(9, 7))
    CPPVis._draw_regions_2d(ax, inst, sol, colors, sel_chain)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title(f"{title_base} | {len(sol.operations)} ops, "
                 f"{sol.objective_value:.1f} m — base")
    fig.savefig(f"{out_dir}/frame_{1:03d}.png", dpi=100)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 7))
    CPPVis._draw_regions_2d(ax, inst, sol, colors, sel_chain)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    for k, (op_idx, (x1, y1, x2, y2)) in enumerate(plan, start=2):
        CPPVis._draw_arrow_2d(ax, x1, y1, x2, y2, op_colors[op_idx],
                              lw=1.5, alpha=0.85)
        ax.set_title(f"{title_base} | op {op_idx} | step {k - 1}/{n - 1}")
        fig.savefig(f"{out_dir}/frame_{k:03d}.png", dpi=100)
    plt.close(fig)
    return n


def make_gif(out_dir: str, n: int) -> None:
    from PIL import Image
    level = os.path.basename(out_dir)
    frames = [Image.open(f"{out_dir}/frame_{k:03d}.png") for k in range(1, n + 1)]
    gif_path = f"{out_dir}.gif"
    frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                   duration=450, loop=0)
    logger.info("Saved %s (%d frames)", gif_path, n)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--levels", type=str, default=",".join(map(str, LEVELS)),
                   help="Comma-separated endurance levels")
    p.add_argument("--split-points", type=int, default=2)
    p.add_argument("--num-ops", type=int, default=4)
    p.add_argument("--sol", type=str, default=None,
                   help="Single solution JSON to animate (overrides --levels). "
                        "Use with --outdir and --tag.")
    p.add_argument("--outdir", type=str, default=None,
                   help="Output directory for --sol mode")
    p.add_argument("--tag", type=str, default=None,
                   help="Title tag for --sol mode (e.g. 'Demo wind-aware K=2 E=173J')")
    p.add_argument("--endurance", type=float, default=None,
                   help="Endurance for rebuilding the instance in --sol mode "
                        "(defaults to parsing EXXXX from the filename)")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    if args.sol:
        # single-solution mode (e.g. wind-aware demo solutions)
        import re
        sol_path = args.sol
        if not os.path.exists(sol_path):
            raise SystemExit(f"missing {sol_path}")
        if args.endurance is None:
            m = re.search(r"E(\d+)", os.path.basename(sol_path))
            end = int(m.group(1)) if m else 0
        else:
            end = int(args.endurance)
        out_dir = args.outdir or os.path.join(
            OUT_DIR, os.path.splitext(os.path.basename(sol_path))[0])
        tag = args.tag or os.path.splitext(os.path.basename(sol_path))[0]
        inst = build_demo_instance(float(end), num_ops=args.num_ops)
        sol = load_solution(sol_path)
        n = make_frames(end, inst, sol, out_dir, tag=tag)
        make_gif(out_dir, n)
        print("FRAME_COUNTS", json.dumps({out_dir: n}))
        return
    counts = {}
    for end in [int(e) for e in args.levels.split(",")]:
        sol_path = (f"demo_interruptions/sweep_E{end:04d}_"
                    f"K{args.split_points}.json")
        if not os.path.exists(sol_path):
            logger.warning("MISSING %s — skipping", sol_path)
            continue
        inst = build_demo_instance(float(end), num_ops=args.num_ops)
        sol = load_solution(sol_path)
        out_dir = os.path.join(OUT_DIR, f"E{end:04d}_K{args.split_points}")
        n = make_frames(end, inst, sol, out_dir)
        make_gif(out_dir, n)
        counts[end] = n
        logger.info("E=%d: %d frames", end, n)
    print("FRAME_COUNTS", json.dumps(counts))


if __name__ == "__main__":
    main()