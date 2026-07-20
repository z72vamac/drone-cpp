from __future__ import annotations
from typing import List, Dict, Tuple, Optional
import numpy as np
from .data_structures import (
    Instance, Solution, Operation, Vertex, VertexType, Edge, EdgeType,
    Point3D, Ring, PolygonalChain, Region, DroneParams, WindParams, AtmosphereParams
)


class HeuristicSolver:

    def __init__(self, instance: Instance):
        self.inst = instance
        self.depot = instance.depot
        self.depot_v = Vertex(-1, 0, VertexType.START)
        self.drone = instance.drone
        self.wind = instance.wind
        self.E = 0.5 * self.drone.drag_coef * self.drone.front_area
        self.rho0 = AtmosphereParams.air_density(0)
        self.v_vert = self.drone.vertical_speed

        self._per_region_en = {}
        for r in instance.regions:
            pts = np.array([[p.x, p.y] for p in r.boundary])
            centroid = pts.mean(axis=0)
            d = centroid - np.array([self.depot.x, self.depot.y])
            nrm = np.linalg.norm(d)
            du = d / nrm if nrm > 1e-6 else np.array([1.0, 0.0])
            wdir = self.wind.direction[:2]
            wdir = wdir / np.linalg.norm(wdir) if np.linalg.norm(wdir) > 1e-6 else np.array([1.0, 0.0])
            ws = self.wind.speed_at_10m
            nu = np.linalg.norm(self.drone.cruise_speed * du - ws * wdir)
            self._per_region_en[r.id] = {
                'En_xy': self.E * self.rho0 * nu,
                'En_z': 0.5 * self.E * self.v_vert * self.rho0,
            }

        self._max_rings = {}
        for r in self.inst.regions:
            mx = 0
            for ch in r.chains:
                if ch.rings:
                    mx = max(mx, len(ch.rings))
            self._max_rings[r.id] = mx

    def solve(self) -> Solution:
        chain_sel = self._select_chains()
        rings_info = self._build_ring_info(chain_sel)
        giant_tour = self._nearest_neighbor_tsp(rings_info)
        self._two_opt_on_order(giant_tour, rings_info)
        ops = self._split_into_operations(giant_tour, rings_info, chain_sel)
        self._optimize_all_routes(ops, rings_info)
        self._local_search(ops, rings_info, chain_sel)
        return self._build_solution(ops, rings_info, chain_sel)

    # ------------------------------------------------------------------
    # Chain selection
    # ------------------------------------------------------------------
    def _select_chains(self) -> Dict[int, int]:
        sel = {}
        for r in self.inst.regions:
            best_t, best_cost = 0, float('inf')
            pts = np.array([[p.x, p.y] for p in r.boundary])
            centroid = pts.mean(axis=0)
            for t, ch in enumerate(r.chains):
                if not ch.rings or len(ch.rings) < self._max_rings[r.id]:
                    continue
                total_perim = sum(rr.perimeter for rr in ch.rings)
                dxy = np.linalg.norm(centroid - [self.depot.x, self.depot.y])
                dz = ch.height
                cost = total_perim + 0.5 * (dxy + dz)
                if cost < best_cost:
                    best_cost, best_t = cost, t
            sel[r.id] = best_t
        return sel

    # ------------------------------------------------------------------
    # Ring info (list of dicts)
    # ------------------------------------------------------------------
    def _build_ring_info(self, chain_sel: Dict[int, int]) -> List[dict]:
        rings = []
        for r in self.inst.regions:
            ch = r.chains[chain_sel[r.id]]
            for ri, ring in enumerate(ch.rings):
                pts_3d = [(s.start.x, s.start.y, s.start.z) for s in ring.segments]
                cx = np.mean([p[0] for p in pts_3d])
                cy = np.mean([p[1] for p in pts_3d])
                cz = ch.height
                cum_lens = []
                acc = 0.0
                for s in ring.segments:
                    cum_lens.append(acc)
                    acc += s.length
                cum_lens.append(acc)

                rings.append(dict(
                    idx=len(rings),
                    region_id=r.id,
                    ring_idx=ri,
                    height=ch.height,
                    perimeter=ring.perimeter,
                    segments=ring.segments,
                    cum_lens=cum_lens,
                    centroid=(cx, cy, cz),
                    entry=(cx, cy, cz),
                    entry_lambda=0.0,
                    num_segs=len(ring.segments),
                ))
        return rings

    @staticmethod
    def _sample_point(segments: List, cum_lens: List[float], lam: float) -> Point3D:
        total = cum_lens[-1]
        if total <= 0:
            s = segments[0]
            return Point3D(s.start.x, s.start.y, s.start.z)
        lam = max(0.0, min(total, lam))
        for i in range(len(segments)):
            if cum_lens[i] <= lam < cum_lens[i + 1] or i == len(segments) - 1:
                seg = segments[i]
                t = (lam - cum_lens[i]) / seg.length if seg.length > 0 else 0.0
                return Point3D(
                    seg.start.x + t * (seg.end.x - seg.start.x),
                    seg.start.y + t * (seg.end.y - seg.start.y),
                    seg.start.z + t * (seg.end.z - seg.start.z))
        s = segments[-1]
        return Point3D(s.end.x, s.end.y, s.end.z)

    @staticmethod
    def _closest_on_ring(segments, cum_lens, target: Point3D, num_samples=60) -> Tuple[Point3D, float]:
        total = cum_lens[-1]
        best_pt = None; best_lam = 0.0; best_d = float('inf')
        for k in range(num_samples):
            lam = total * k / num_samples
            pt = HeuristicSolver._sample_point(segments, cum_lens, lam)
            d = np.linalg.norm([pt.x - target.x, pt.y - target.y, pt.z - target.z])
            if d < best_d:
                best_d = d; best_pt = pt; best_lam = lam
        return best_pt, best_lam

    @staticmethod
    def _closest_on_ring_2target(segments, cum_lens, t1: Point3D, t2: Point3D,
                                  num_samples=60) -> Tuple[Point3D, float]:
        total = cum_lens[-1]
        best_pt = None; best_lam = 0.0; best_d = float('inf')
        for k in range(num_samples):
            lam = total * k / num_samples
            pt = HeuristicSolver._sample_point(segments, cum_lens, lam)
            d = (np.linalg.norm([pt.x - t1.x, pt.y - t1.y, pt.z - t1.z]) +
                 np.linalg.norm([pt.x - t2.x, pt.y - t2.y, pt.z - t2.z]))
            if d < best_d:
                best_d = d; best_pt = pt; best_lam = lam
        return best_pt, best_lam

    # ------------------------------------------------------------------
    # Distance / energy helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _dist_3d(a: Point3D, b: Point3D) -> float:
        return np.linalg.norm([a.x - b.x, a.y - b.y, a.z - b.z])

    @staticmethod
    def _dz(a: Point3D, b: Point3D) -> float:
        return abs(a.z - b.z)

    @staticmethod
    def _dxy(a: Point3D, b: Point3D) -> float:
        return np.linalg.norm([a.x - b.x, a.y - b.y])

    def _inter_energy(self, a: Point3D, b: Point3D) -> float:
        return self._dxy(a, b) + self._dz(a, b)

    def _depot_to_energy(self, reg_id: int, p: Point3D) -> float:
        en = self._per_region_en[reg_id]
        return en['En_xy'] * self._dxy(self.depot, p) + en['En_z'] * p.z

    def _ring_to_depot_energy(self, reg_id: int, p: Point3D) -> float:
        en = self._per_region_en[reg_id]
        return en['En_xy'] * self._dxy(p, self.depot)

    def _route_energy(self, op_rings: List[int], rings_info: List[dict],
                      with_return=True) -> float:
        if not op_rings:
            return 0.0
        e = 0.0
        first = rings_info[op_rings[0]]
        dep = Point3D(self.depot.x, self.depot.y, 0)
        e += self._depot_to_energy(first['region_id'],
                                    Point3D(*first['entry']))
        for i, rid in enumerate(op_rings):
            ring = rings_info[rid]
            e += ring['perimeter']
            if i < len(op_rings) - 1:
                nxt = rings_info[op_rings[i + 1]]
                e += self._inter_energy(Point3D(*ring['entry']),
                                         Point3D(*nxt['entry']))
        if with_return:
            last = rings_info[op_rings[-1]]
            e += self._ring_to_depot_energy(last['region_id'],
                                             Point3D(*last['entry']))
        return e

    # ------------------------------------------------------------------
    # TSP: nearest-neighbour + 2-opt
    # ------------------------------------------------------------------
    def _nearest_neighbor_tsp(self, rings_info: List[dict]) -> List[int]:
        n = len(rings_info)
        visited = [False] * n
        order = []
        cur = np.array([self.depot.x, self.depot.y, 0])
        for _ in range(n):
            best_i, best_d = -1, float('inf')
            for i in range(n):
                if visited[i]: continue
                c = np.array(rings_info[i]['centroid'])
                d = np.linalg.norm(cur - c)
                if d < best_d:
                    best_d, best_i = d, i
            order.append(best_i)
            visited[best_i] = True
            cur = np.array(rings_info[best_i]['centroid'])
        return order

    def _two_opt_on_order(self, order: List[int], rings_info: List[dict],
                           max_passes=20) -> bool:
        improved = True
        n = len(order)
        for _ in range(max_passes):
            improved = False
            for i in range(n - 1):
                for j in range(i + 1, n):
                    a = rings_info[order[i - 1]]['centroid'] if i > 0 else (
                        self.depot.x, self.depot.y, 0)
                    b = rings_info[order[i]]['centroid']
                    c = rings_info[order[j]]['centroid']
                    d = rings_info[order[(j + 1) % n]]['centroid'] if j < n - 1 else (
                        self.depot.x, self.depot.y, 0)
                    old = (np.linalg.norm(np.array(a) - np.array(b)) +
                           np.linalg.norm(np.array(c) - np.array(d)))
                    new = (np.linalg.norm(np.array(a) - np.array(c)) +
                           np.linalg.norm(np.array(b) - np.array(d)))
                    if new + 1e-9 < old:
                        order[i:j + 1] = reversed(order[i:j + 1])
                        improved = True
            if not improved:
                break
        return True

    # ------------------------------------------------------------------
    # Split giant tour into operations respecting endurance
    # ------------------------------------------------------------------
    def _split_into_operations(self, order: List[int], rings_info: List[dict],
                                chain_sel: Dict[int, int]) -> List[List[int]]:
        E_max = self.drone.max_endurance
        ops = []
        cur = []
        cur_e = 0.0
        dep = Point3D(self.depot.x, self.depot.y, 0)

        for rid in order:
            ring = rings_info[rid]
            entry_pt = Point3D(*ring['entry'])
            ring_e = ring['perimeter']

            if not cur:
                to_here = self._depot_to_energy(ring['region_id'], entry_pt)
            else:
                last = rings_info[cur[-1]]
                to_here = self._inter_energy(Point3D(*last['entry']), entry_pt)

            back_e = self._ring_to_depot_energy(ring['region_id'], entry_pt)

            if cur_e + to_here + ring_e + back_e <= E_max + 1e-6:
                cur.append(rid)
                cur_e += to_here + ring_e
            else:
                if cur:
                    ops.append(cur)
                cur = [rid]
                cur_e = self._depot_to_energy(ring['region_id'], entry_pt) + ring_e

        if cur:
            ops.append(cur)
        return ops

    # ------------------------------------------------------------------
    # Optimize entry points (coordinate descent)
    # ------------------------------------------------------------------
    def _optimize_all_routes(self, ops: List[List[int]], rings_info: List[dict],
                              iterations=8):
        dep = Point3D(self.depot.x, self.depot.y, 0)
        for _ in range(iterations):
            for op in ops:
                for idx, rid in enumerate(op):
                    ring = rings_info[rid]
                    prev = dep if idx == 0 else Point3D(*rings_info[op[idx - 1]]['entry'])
                    nxt = dep if idx == len(op) - 1 else Point3D(*rings_info[op[idx + 1]]['entry'])
                    pt, lam = self._closest_on_ring_2target(
                        ring['segments'], ring['cum_lens'], prev, nxt)
                    ring['entry'] = (pt.x, pt.y, pt.z)
                    ring['entry_lambda'] = lam

    # ------------------------------------------------------------------
    # Local search: 2-opt per operation + or-opt between operations
    # ------------------------------------------------------------------
    def _local_search(self, ops: List[List[int]], rings_info: List[dict],
                       chain_sel: Dict[int, int]):
        self._two_opt_per_op(ops, rings_info)
        self._or_opt(ops, rings_info)

    def _two_opt_per_op(self, ops: List[List[int]], rings_info: List[dict],
                         max_passes=10):
        dep = Point3D(self.depot.x, self.depot.y, 0)
        for op in ops:
            improved = True
            for _ in range(max_passes):
                improved = False
                n = len(op)
                for i in range(n - 1):
                    for j in range(i + 1, n):
                        def cost_of(seq):
                            if not seq: return 0.0
                            c = self._depot_to_energy(
                                rings_info[seq[0]]['region_id'],
                                Point3D(*rings_info[seq[0]]['entry']))
                            for k in range(len(seq)):
                                r = rings_info[seq[k]]
                                c += r['perimeter']
                                if k < len(seq) - 1:
                                    nk = rings_info[seq[k + 1]]
                                    c += self._inter_energy(
                                        Point3D(*r['entry']),
                                        Point3D(*nk['entry']))
                            last = rings_info[seq[-1]]
                            c += self._ring_to_depot_energy(
                                last['region_id'], Point3D(*last['entry']))
                            return c
                        old_cost = cost_of(op)
                        new_seq = op[:i] + op[i:j + 1][::-1] + op[j + 1:]
                        new_cost = cost_of(new_seq)
                        if new_cost + 1e-9 < old_cost:
                            op[:] = new_seq
                            improved = True
                if not improved:
                    break

    def _or_opt(self, ops: List[List[int]], rings_info: List[dict]):
        E_max = self.drone.max_endurance
        improved = True
        while improved:
            improved = False
            for src_i in range(len(ops)):
                for dst_i in range(len(ops)):
                    if src_i == dst_i:
                        continue
                    for ri in range(len(ops[src_i])):
                        rid = ops[src_i][ri]
                        for pos in range(len(ops[dst_i]) + 1):
                            src_new = ops[src_i][:ri] + ops[src_i][ri + 1:]
                            dst_new = ops[dst_i][:pos] + [rid] + ops[dst_i][pos:]
                            e_src = self._route_energy(src_new, rings_info)
                            e_dst = self._route_energy(dst_new, rings_info)
                            if e_src <= E_max + 1e-6 and e_dst <= E_max + 1e-6:
                                obj_old = (self._route_energy(ops[src_i], rings_info) +
                                           self._route_energy(ops[dst_i], rings_info))
                                obj_new = e_src + e_dst
                                if obj_new + 1e-9 < obj_old:
                                    ops[src_i] = src_new
                                    ops[dst_i] = dst_new
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                self._optimize_all_routes(ops, rings_info, iterations=3)

    # ------------------------------------------------------------------
    # Build Solution object
    # ------------------------------------------------------------------
    def _build_solution(self, ops: List[List[int]], rings_info: List[dict],
                         chain_sel: Dict[int, int]) -> Solution:
        vertex_positions = {}
        vertex_lambdas = {}
        vertex_rings = {}
        solution_ops = []
        total_obj = 0.0

        for op_rings in ops:
            edges = []
            prev_v = self.depot_v
            prev_pos = Point3D(self.depot.x, self.depot.y, 0)

            for rid in op_rings:
                ring = rings_info[rid]
                r_id = ring['region_id']
                ri = ring['ring_idx']
                entry_pt = Point3D(*ring['entry'])

                v_launch = Vertex(r_id, 2 * ri, VertexType.LAUNCH)
                v_retrieve = Vertex(r_id, 2 * ri + 1, VertexType.RETRIEVE)

                vertex_positions[v_launch] = entry_pt
                vertex_positions[v_retrieve] = entry_pt
                vertex_lambdas[v_launch] = ring['entry_lambda']
                vertex_lambdas[v_retrieve] = ring['entry_lambda']
                vertex_rings[v_launch] = ri
                vertex_rings[v_retrieve] = ri

                edges.append((prev_v, v_launch))
                total_obj += self._inter_energy(prev_pos, entry_pt) if prev_v != self.depot_v else \
                    self._dxy(prev_pos, entry_pt) + self._dz(prev_pos, entry_pt)

                edges.append((v_launch, v_retrieve))
                total_obj += ring['perimeter']

                prev_v = v_retrieve
                prev_pos = entry_pt

            edges.append((prev_v, self.depot_v))
            total_obj += self._inter_energy(prev_pos, Point3D(self.depot.x, self.depot.y, 0))
            solution_ops.append(Operation(edges=edges))

        return Solution(
            operations=solution_ops,
            objective_value=total_obj,
            vertex_positions=vertex_positions,
            chain_selection=chain_sel,
            vertex_lambdas=vertex_lambdas,
            vertex_rings=vertex_rings,
        )
