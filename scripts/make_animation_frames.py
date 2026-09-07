#!/usr/bin/env python
"""Generate animation frames for the endurance showcase (60-120 J).

Reads the pre-solved solutions in sweep_results_new/end_XXX_sol.json (seed 42,
3 regions, vertex-based model) and renders one PNG per path segment, so the
beamer can replay the drone path with \\animategraphics exactly as the
original articulo/presentation.tex did.

Output: anim_frames_060/frame_001.png ... anim_frames_120/frame_XXX.png
Prints, per endurance, the number of frames written.
"""
from __future__ import annotations
import sys, os, json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")

from drone_cpp.data_structures import Point3D, Vertex, VertexType, Operation, Solution
from drone_cpp.visualization import CPPVis
from sweep_endurance import build_base

ENDURANCES = [150, 200, 250, 300, 400, 600]


def load_solution(path: str) -> Solution:
    d = json.load(open(path, encoding="utf-8"))
    vt = {e.name: e for e in VertexType}
    ops = []
    for op in d["operations"]:
        edges = [(Vertex(e["r"], e["i"], vt[e["t"]]),
                  Vertex(e["r2"], e["i2"], vt[e["t2"]])) for e in op]
        ops.append(Operation(edges=edges))
    vp, vl, vr = {}, {}, {}
    for k, p in d["vertex_positions"].items():
        r, i, t = k.split(",")
        vp[Vertex(int(r), int(i), vt[t])] = Point3D(p["x"], p["y"], p["z"])
    for k, lam in d["vertex_lambdas"].items():
        r, i, t = k.split(",")
        vl[Vertex(int(r), int(i), vt[t])] = lam
    for k, ri in d.get("vertex_rings", {}).items():
        r, i, t = k.split(",")
        vr[Vertex(int(r), int(i), vt[t])] = ri
    return Solution(
        operations=ops,
        objective_value=d["objective_value"],
        vertex_positions=vp,
        chain_selection={int(k): v for k, v in d["chain_selection"].items()},
        vertex_lambdas=vl,
        vertex_rings=vr,
    )




def chain_edges(edges, depot_vertex):
    """Reorder the edges of one operation so that consecutive edges connect
    (v of edge k == u of edge k+1), starting from the depot. The solver stores
    the edges of an operation without a chronological order, which makes the
    incremental animation look disconnected unless we chain them here."""
    by_u = {}
    for (u, v) in edges:
        by_u.setdefault(u, []).append((u, v))
    # start from the outgoing depot leg
    start = None
    for (u, v) in edges:
        if u == depot_vertex or u.region_id == -1:
            start = (u, v)
            break
    if start is None and edges:
        start = edges[0]
    if start is None:
        return []
    ordered = [start]
    cur = start[1]
    used = {(start[0], start[1])}
    while True:
        nxt = None
        for (u, v) in by_u.get(cur, []):
            if (u, v) not in used:
                nxt = (u, v)
                break
        if nxt is None:
            break
        ordered.append(nxt)
        used.add(nxt)
        cur = nxt[1]
    if len(ordered) < len(edges):
        # leftover edges (should not happen with flow conservation): append rest
        rest = [e for e in edges if e not in used]
        ordered += rest
    return ordered

def edge_segments(instance, solution, op_idx, u, v, sel_chain, get_cum, ring_map):
    """List of (x1, y1, x2, y2) segments drawn for edge (u, v) — mirrors
    CPPVis._draw_edges_2d so the animation matches the static plot."""
    segs = []
    is_inter = CPPVis.is_inter_edge(u, v)
    is_rl = (not is_inter and CPPVis.is_rl_edge(u, v) and u.region_id in sel_chain)
    if is_rl:
        chain = sel_chain[u.region_id]
        pu = solution.vertex_positions.get(u)
        pv = solution.vertex_positions.get(v)
        if pu is None or pv is None:
            return segs
        cu, cv = get_cum(u), get_cum(v)
        is_same_ring = (u in ring_map and v in ring_map and
                        solution.vertex_rings.get(u) == solution.vertex_rings.get(v))
        has_ring = u in ring_map or v in ring_map
        if cu is not None and cv is not None and not has_ring:
            forward = cv >= cu
            path = CPPVis.get_chain_path(chain, cu, cv)
            if not forward:
                path = [(x2, y2, z2, x1, y1, z1)
                        for (x1, y1, z1, x2, y2, z2) in reversed(path)]
            segs = [(x1, y1, x2, y2) for (x1, y1, z1, x2, y2, z2) in path]
        elif cu is not None and cv is not None and is_same_ring:
            if abs(cu - cv) < 1e-9:
                path = CPPVis.ring_full_loop(ring_map[u], cu)
            else:
                path = CPPVis._get_ring_path(ring_map[u], cu, cv)
            segs = [(x1, y1, x2, y2) for (x1, y1, z1, x2, y2, z2) in path]
        else:
            segs = [(pu.x, pu.y, pv.x, pv.y)]
    else:
        pu = solution.vertex_positions.get(u)
        pv = solution.vertex_positions.get(v)
        if u == instance.depot_vertex or u.region_id == -1:
            pu = instance.depot
        if v == instance.depot_vertex or v.region_id == -1:
            pv = instance.depot
        if pu is None or pv is None:
            return segs
        same_chain = (u.region_id >= 0 and u.region_id == v.region_id
                      and u.region_id in sel_chain)
        if same_chain:
            chain = sel_chain[u.region_id]
            cu = CPPVis._vertex_chain_cum(u, sel_chain, solution, ring_map)
            cv = CPPVis._vertex_chain_cum(v, sel_chain, solution, ring_map)
            if cu is not None and cv is not None:
                ru = solution.vertex_rings.get(u, 0)
                rv = solution.vertex_rings.get(v, 0)
                if ru == rv:
                    forward = cv >= cu
                    path = CPPVis.get_chain_path(chain, cu, cv)
                    if not forward:
                        path = [(x2, y2, z2, x1, y1, z1)
                                for (x1, y1, z1, x2, y2, z2) in reversed(path)]
                    return [(x1, y1, x2, y2) for (x1, y1, z1, x2, y2, z2) in path]
        segs = [(pu.x, pu.y, pv.x, pv.y)]
    return segs


def make_frames(end: int, inst, sol: Solution, out_dir: str) -> int:
    os.makedirs(out_dir, exist_ok=True)
    colors, op_colors, sel_chain, ring_map, cum_map, get_cum = \
        CPPVis._solution_setup(inst, sol)

    plan = []
    for op_idx, op in enumerate(sol.operations):
        for (u, v) in chain_edges(op.edges, inst.depot_vertex):
            for s in edge_segments(inst, sol, op_idx, u, v, sel_chain, get_cum, ring_map):
                plan.append((op_idx, s))
    n = len(plan) + 1  # frame 1 = base plot, then one frame per segment

    fig, ax = plt.subplots(figsize=(9, 7))
    CPPVis._draw_regions_2d(ax, inst, sol, colors, sel_chain)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Endurance={end} J | {len(sol.operations)} op(s), {sol.objective_value:.1f} m")
    fig.savefig(f"{out_dir}/frame_{1:03d}.png", dpi=100)
    print(f"  {out_dir}/frame_{1:03d}.png (base)")

    for k, (op_idx, (x1, y1, x2, y2)) in enumerate(plan, start=2):
        CPPVis._draw_arrow_2d(ax, x1, y1, x2, y2, op_colors[op_idx],
                              lw=1.5, alpha=0.85)
        fig.savefig(f"{out_dir}/frame_{k:03d}.png", dpi=100)
    plt.close(fig)
    return n


def main() -> None:
    inst = build_base(42)
    counts = {}
    for end in ENDURANCES:
        sol_path = f"sweep_results_new/end_{end:03d}_sol.json"
        if not os.path.exists(sol_path):
            print(f"MISSING {sol_path} — skipping")
            continue
        sol = load_solution(sol_path)
        out_dir = f"anim_frames_{end:03d}"
        n = make_frames(end, inst, sol, out_dir)
        counts[end] = n
        print(f"ENDURANCE {end}: {n} frames -> {out_dir}/")
    print("FRAME_COUNTS", json.dumps(counts))


if __name__ == "__main__":
    main()