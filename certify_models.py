"""Certify equivalence between RingsModel and EdgesModel.

Subcommands:
  verify  Independently re-check a saved Solution JSON against both
          formulations' constraints (arcs, degrees, coverage, connectivity,
          endurance, objective, positions).
  cross   Warm-start one formulation with a solution claimed optimal by the
          other and see whether Gurobi accepts/improves/rejects it.
  fresh   Fresh solves of both models on one instance with Gurobi logs for
          performance comparison.
"""
from __future__ import annotations
import argparse, sys, time
from collections import defaultdict

import numpy as np

sys.path.insert(0, ".")

from drone_cpp.data_structures import Instance, Solution, Vertex, VertexType, AtmosphereParams
from drone_cpp.model import build_model
from compare_methods_edges import build_instance

DEPOT = Vertex(-1, 0, VertexType.START)


def ring_slots(inst: Instance):
    slots = {}
    for r in inst.regions:
        max_r = max((len(c.rings) if c.rings else 0) for c in r.chains)
        for ri in range(max_r):
            slots[(r.id, ri)] = [
                (ci, ch.rings[ri]) for ci, ch in enumerate(r.chains)
                if ch.rings and ri < len(ch.rings)
            ]
    return slots


def position_on_chain(inst, region_id, ri, lam, chain_idx):
    r = next(rg for rg in inst.regions if rg.id == region_id)
    ring = r.chains[chain_idx].rings[ri]
    n = len(ring.segments)
    si = max(0, min(n - 1, int(np.floor(lam))))
    g = lam - si
    seg = ring.segments[si]
    a = np.array([seg.start.x, seg.start.y, seg.start.z])
    b = np.array([seg.end.x, seg.end.y, seg.end.z])
    return a + (b - a) * g


def _pos2(sol, w):
    p = sol.vertex_positions[w]
    return np.array([p.x, p.y])


def _intra_pair(u, v):
    return u.region_id == v.region_id and u.idx // 2 == v.idx // 2 \
        and u != DEPOT and v != DEPOT


def arc_dist(inst, sol, u, v, slots):
    if u == DEPOT or v == DEPOT:
        w = v if u == DEPOT else u
        dxy = float(np.linalg.norm(_pos2(sol, w)
                                   - np.array([inst.depot.x, inst.depot.y])))
        dz = abs(sol.vertex_positions[w].z)
        return dxy + dz
    if _intra_pair(u, v):
        key = (u.region_id, u.idx // 2)
        cs = sol.chain_selection.get(u.region_id, 0)
        _, ring = slots[key][min(cs, len(slots[key]) - 1)]
        return ring.perimeter
    pu, pv = sol.vertex_positions[u], sol.vertex_positions[v]
    return float(np.hypot(pu.x - pv.x, pu.y - pv.y)) + abs(pu.z - pv.z)


def depot_en_xy(inst, region_id):
    r = next(rg for rg in inst.regions if rg.id == region_id)
    centroid = np.mean([[p.x, p.y] for p in r.boundary], axis=0)
    vec = np.array([centroid[0] - inst.depot.x, centroid[1] - inst.depot.y])
    nrm = float(np.linalg.norm(vec))
    unit = vec / nrm if nrm > 0 else np.array([1.0, 0.0])
    ws0 = inst.wind.speed_at_height(0.0)
    wdir = np.asarray(inst.wind.direction, dtype=float)[:2]
    nw = float(np.linalg.norm(wdir))
    if nw > 0:
        wdir = wdir / nw
    nu = float(np.linalg.norm(inst.drone.cruise_speed * unit - ws0 * wdir))
    return inst.drone.E_xy * AtmosphereParams.air_density(0.0) * nu


def op_energies(inst, sol):
    rho0 = AtmosphereParams.air_density(0.0)
    en_z_up = 0.5 * inst.drone.E_z * inst.drone.vertical_speed * rho0
    out = []
    for op in sol.operations:
        e = 0.0
        for u, v in op.edges:
            if u == DEPOT:
                pz = sol.vertex_positions[v].z
                dxy = float(np.linalg.norm(
                    _pos2(sol, v) - np.array([inst.depot.x, inst.depot.y])))
                e += depot_en_xy(inst, v.region_id) * dxy + en_z_up * pz
            elif v == DEPOT:
                dxy = float(np.linalg.norm(
                    _pos2(sol, u) - np.array([inst.depot.x, inst.depot.y])))
                e += depot_en_xy(inst, u.region_id) * dxy
            else:
                e += arc_dist(inst, sol, u, v, ring_slots(inst))
        out.append(e)
    return out


def verify_solution(inst: Instance, sol: Solution, formulation: str):
    errs, warns = [], []
    slots = ring_slots(inst)
    allow_reverse_intra = formulation == "rings"

    deg_out, deg_in = defaultdict(int), defaultdict(int)
    for o, op in enumerate(sol.operations):
        dout, din = defaultdict(int), defaultdict(int)
        nodes_o = set()
        adj = defaultdict(list)
        for u, v in op.edges:
            dout[u] += 1
            din[v] += 1
            nodes_o.add(u)
            nodes_o.add(v)
            adj[u].append(v)
            if u != DEPOT and v != DEPOT and _intra_pair(u, v):
                if u.vtype != VertexType.LAUNCH and not allow_reverse_intra:
                    errs.append(f"[{formulation}] op{o}: arco intra inverso "
                                f"{u}->{v} no existe en el modelo de aristas")
            elif u != DEPOT and v != DEPOT and u.region_id == v.region_id:
                warns.append(f"[{formulation}] op{o}: arco inter-anillo intra-"
                             f"region {u}->{v}")
        for w in nodes_o:
            deg_out[w] += dout[w]
            deg_in[w] += din[w]
        if DEPOT not in nodes_o:
            errs.append(f"op{o}: no contiene el depot")
        elif dout[DEPOT] != 1 or din[DEPOT] != 1:
            errs.append(f"op{o}: grado depot ({dout[DEPOT]},{din[DEPOT]}) != (1,1)")
        seen, stack = {DEPOT}, [DEPOT]
        while stack:
            n = stack.pop()
            for nb in adj.get(n, []):
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        if nodes_o - seen:
            errs.append(f"op{o}: SUBTOUR, no alcanzables desde depot: "
                        f"{sorted(nodes_o - seen)}")

    for w in set(deg_out) | set(deg_in):
        if w == DEPOT:
            continue
        if deg_out.get(w, 0) != 1 or deg_in.get(w, 0) != 1:
            errs.append(f"vertice {w}: grados totales (out={deg_out.get(w, 0)}, "
                        f"in={deg_in.get(w, 0)}) != (1,1)")

    covered = defaultdict(int)
    for op in sol.operations:
        for u, v in op.edges:
            if u != DEPOT and v != DEPOT and _intra_pair(u, v):
                covered[(u.region_id, u.idx // 2)] += 1
    for key in slots:
        if covered[key] != 1:
            errs.append(f"slot anillo {key} cubierto {covered[key]} veces (!=1)")
    for key in covered:
        if key not in slots:
            errs.append(f"slot anillo {key} inexistente (region sin ese anillo)")

    energies = op_energies(inst, sol)
    for o, e in enumerate(energies):
        if e > inst.drone.max_endurance + 1e-4:
            errs.append(f"op{o}: energia {e:.4f} > endurance "
                        f"{inst.drone.max_endurance:.1f}")

    obj = sum(arc_dist(inst, sol, u, v, slots)
              for op in sol.operations for u, v in op.edges)
    if abs(obj - sol.objective_value) > 1e-3 * max(1.0, abs(obj)):
        errs.append(f"objetivo recalculado {obj:.4f} != reportado "
                    f"{sol.objective_value:.4f}")

    for w, lam in sol.vertex_lambdas.items():
        cs = sol.chain_selection.get(w.region_id)
        ri = w.idx // 2
        if cs is None or (w.region_id, ri) not in slots:
            continue
        expected = position_on_chain(inst, w.region_id, ri, lam, cs)
        p = sol.vertex_positions.get(w)
        if p is None:
            continue
        actual = np.array([p.x, p.y, p.z])
        if float(np.linalg.norm(expected - actual)) > 1e-4:
            errs.append(f"posicion inconsistente {w}: lambda={lam:.4f} "
                        f"implica {np.round(expected, 3)}, guardado "
                        f"{np.round(actual, 3)}")

    if len(sol.operations) > inst.num_operations:
        errs.append(f"num_ops {len(sol.operations)} > O={inst.num_operations}")

    return errs, warns, dict(objective_recalc=obj, energies=energies)


def cmd_verify(args):
    inst = build_instance(args.nr, args.nh, args.seed, endurance=args.E)
    sol = Solution.load(args.solution)
    print(f"Instancia r{args.nr}_h{args.nh}_s{args.seed}_e{args.E:g} | "
          f"{args.solution}")
    print(f"  declarado: obj={sol.objective_value:.4f} [{sol.status}] "
          f"ops={len(sol.operations)} chains={sol.chain_selection}")
    ok = True
    for f in ("rings", "edges"):
        errs, warns, info = verify_solution(inst, sol, f)
        tag = "OK  " if not errs else "FAIL"
        print(f"  [{tag}] {f:5s}: obj_recalc={info['objective_recalc']:.4f} "
              f"E_ops={[round(e, 1) for e in info['energies']]}")
        for e in errs:
            print("         ERROR:", e)
        for w in warns:
            print("         aviso:", w)
        ok &= not errs
    return 0 if ok else 1


def cmd_cross(args):
    inst = build_instance(args.nr, args.nh, args.seed, endurance=args.E)
    sol = Solution.load(args.solution)
    target = args.target
    errs, _, info = verify_solution(inst, sol, target)
    print(f"Pre-verificacion de la solucion en la formulacion '{target}': "
          f"{'LIMPIA' if not errs else 'FALLA'}")
    for e in errs:
        print("   ", e)
    model = build_model(inst, target, verbose=False)
    model.model.setParam("MIPGap", 0.0)
    model.set_warm_start(sol)
    t0 = time.time()
    sol2 = model.optimize(tl=args.tl)
    el = time.time() - t0
    if sol2 is None:
        print(f"CROSS -> {target}: SIN solucion tras {el:.1f}s "
              f"(gurobi status={model.model.Status})")
        return 1
    print(f"CROSS -> {target}: start={sol.objective_value:.4f} "
          f"final={sol2.objective_value:.4f} [{sol2.status}] "
          f"gap={sol2.mip_gap} bound={model.model.ObjBound:.4f} "
          f"t={el:.1f}s delta={sol2.objective_value - sol.objective_value:+.4f}")
    return 0


def cmd_fresh(args):
    inst = build_instance(args.nr, args.nh, args.seed, endurance=args.E)
    tag = f"r{args.nr}h{args.nh}s{args.seed}e{args.E:g}"
    for mtype in ("rings", "edges"):
        model = build_model(inst, mtype, verbose=False)
        model.model.setParam("MIPGap", 0.0)
        model.model.setParam("LogFile", f"certify_{mtype}_{tag}.log")
        t0 = time.time()
        sol = model.optimize(tl=args.tl)
        el = time.time() - t0
        if sol is None:
            print(f"{mtype:5s}: SIN SOLUCION tras {el:.1f}s "
                  f"(gurobi status={model.model.Status})")
        else:
            print(f"{mtype:5s}: obj={sol.objective_value:.4f} [{sol.status}] "
                  f"gap={sol.mip_gap} bound={model.model.ObjBound:.4f} "
                  f"first_inc={sol.first_incumbent_obj} "
                  f"@t={sol.first_incumbent_time:.2f}s t={el:.1f}s "
                  f"nvars={model.model.NumVars} nconstrs={model.model.NumConstrs}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("nr", type=int)
        sp.add_argument("nh", type=int)
        sp.add_argument("seed", type=int)
        sp.add_argument("E", type=float)

    pv = sub.add_parser("verify")
    common(pv)
    pv.add_argument("solution")
    pv.set_defaults(fn=cmd_verify)

    pc = sub.add_parser("cross")
    common(pc)
    pc.add_argument("solution")
    pc.add_argument("--target", choices=["rings", "edges"], required=True)
    pc.add_argument("--tl", type=float, default=300.0)
    pc.set_defaults(fn=cmd_cross)

    pf = sub.add_parser("fresh")
    common(pf)
    pf.add_argument("--tl", type=float, default=600.0)
    pf.set_defaults(fn=cmd_fresh)

    args = p.parse_args()
    sys.exit(args.fn(args))


if __name__ == "__main__":
    main()
