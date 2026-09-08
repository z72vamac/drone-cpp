from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np
import gurobipy as gp
from gurobipy import GRB

from ..data_structures import (
    Point3D, Segment, Ring, PolygonalChain, Region, Vertex,
    VertexType, Edge, EdgeType, AtmosphereParams, Instance, Operation, Solution
)
from ..config import DEFAULT_MIP_FOCUS, DEFAULT_HEURISTICS
from .base import BaseModel


class RingsModel(BaseModel):
    """Each ring of the selected chain is an independent traversal.

    ``num_split_points`` (K) is the number of continuous split points allowed
    per ring.  K=1 reproduces the original behaviour (a single coincident
    launch/retrieve pair covering the whole ring as one full loop).  For K>=2
    each ring is partitioned into K arcs in series
    (launch_1 @ 0, retrieve_j == launch_{j+1}, retrieve_K @ ring end), so arcs
    can be covered in different operations.  Zero-length arcs are allowed, so K
    is an upper bound on the number of pieces.
    """

    def __init__(self, instance: Instance, verbose: bool = True,
                 num_split_points: int = 1, wind_aware: bool = False):
        super().__init__(instance)
        self.inst = instance
        self.R = instance.num_regions
        self.O = instance.num_operations
        self.depot_v = instance.depot_vertex
        self.K = max(1, int(num_split_points))
        self.wind_aware = bool(wind_aware)

        self._ring_info: Dict[Tuple[int, int, int], dict] = {}
        self._max_rings_by_region: Dict[int, int] = {}
        for r in instance.regions:
            max_r = 0
            for ci, ch in enumerate(r.chains):
                rings = ch.rings if ch.rings else []
                max_r = max(max_r, len(rings))
                for ri, ring in enumerate(rings):
                    self._ring_info[(r.id, ci, ri)] = {
                        "perimeter": ring.perimeter,
                        "num_seg": len(ring.segments),
                        "seg_lens": [s.length for s in ring.segments],
                        "cum_before": [sum(s.length for s in ring.segments[:j])
                                       for j in range(len(ring.segments))],
                        "height": ch.height,
                    }
            self._max_rings_by_region[r.id] = max_r

        if self.wind_aware:
            self._precompute_wind()

        self.verts: List[Vertex] = []
        self._ring_rl: List[Tuple[Vertex, Vertex]] = []
        self._ring_index: Dict[Vertex, int] = {}
        self._ring_arcs: Dict[Tuple[int, int], List[Tuple[Vertex, Vertex]]] = {}
        for r in instance.regions:
            for ri in range(self._max_rings_by_region[r.id]):
                arcs = []
                for k in range(self.K):
                    v_e = Vertex(r.id, (ri * self.K + k) * 2, VertexType.LAUNCH)
                    v_x = Vertex(r.id, (ri * self.K + k) * 2 + 1, VertexType.RETRIEVE)
                    self.verts.extend([v_e, v_x])
                    self._ring_rl.append((v_e, v_x))
                    self._ring_index[v_e] = ri
                    self._ring_index[v_x] = ri
                    arcs.append((v_e, v_x))
                self._ring_arcs[(r.id, ri)] = arcs

        # Zero-length connectors between consecutive coincident split points
        # (retrieve_j == launch_{j+1}), in both directions, so consecutive arcs
        # of the same ring can be covered back-to-back inside one operation.
        # The wrap connectors (retrieve_K <-> launch_1, same point) make the
        # anchored K-model a strict generalisation of K=1: any K=1 full loop
        # maps exactly (same entry/exit points, same cost) onto the K arcs.
        self._connectors: List[Tuple[Vertex, Vertex]] = []
        if self.K > 1:
            for arcs in self._ring_arcs.values():
                for j in range(len(arcs) - 1):
                    ret_j = arcs[j][1]
                    lau_j1 = arcs[j + 1][0]
                    self._connectors.append((ret_j, lau_j1))
                    self._connectors.append((lau_j1, ret_j))
                ret_k = arcs[-1][1]
                lau_1 = arcs[0][0]
                self._connectors.append((ret_k, lau_1))
                self._connectors.append((lau_1, ret_k))

        self.all_nodes = [self.depot_v] + self.verts

        self._classify_edges()

        self.model = gp.Model("RingsCPP")
        if not verbose:
            self.model.setParam("OutputFlag", 0)
        self._big_M = 1e5

        self.mu = {}
        self.gamma = {}
        self.alpha = {}
        self.lambd = {}
        self.P_x = {}
        self.P_y = {}
        self.P_z = {}
        self.rho_sel = {}

        self.x = {}
        self.y = {}
        self.zeta = {}
        self.k = {}

        self.edge_dist = {}
        self.edge_energy = {}

        # Cumulative distance from the ring start (paper's dfs): only needed
        # for K>1, where arc lengths depend on the split positions.
        self.dfs = {}
        if self.K > 1:
            max_per = max((info["perimeter"] for info in self._ring_info.values()),
                          default=1.0)
            for v in self.verts:
                self.dfs[v] = self.model.addVar(
                    lb=0., ub=max_per, vtype=GRB.CONTINUOUS, name=f"dfs_{v}")

        self._eta = {}
        self._lin_vars = []

        self._create_variables()
        self._add_location_constraints()
        self._add_drone_path_constraints()
        self._add_valid_inequalities()
        if self.K > 1:
            self._add_dfs_constraints()
        self._add_intra_ring_distance_energy()
        self._add_inter_distance_energy()
        self._add_depot_distance_energy()
        self._add_endurance_constraints()
        self._set_objective()
        self.model.update()

    def _precompute_wind(self):
        """Wind-aware energy coefficients (real joules, precomputed).

        Wind blows in the single constant direction ``wind.direction``; its
        speed follows the power law ``w(h)``.  The drone holds the required
        ground track at cruise speed, so the airspeed on a leg with unit
        direction ``t`` at height ``h`` is ``cruise*t - w(h)`` and the drag
        energy per metre is ``E_xy*rho(h)*|airspeed|``.
        - per ring segment, both headings: ``_seg_C[(r,ti,ri,si)]``.
        - per ordered region pair (centroid direction, paper approximation):
          ``_pair_F[(r1,r2)][ti]`` horizontal/vertical coefficients assuming
          flight at the departure region's selected chain altitude.
        - depot legs: ``_depot_F[r][ti]`` with the fixed depot->centroid
          direction of each region.
        """
        E = self.inst.drone
        w = self.inst.wind
        wdir = np.array([w.direction[0], w.direction[1]], dtype=float)
        nrm = float(np.linalg.norm(wdir))
        wdir = wdir / nrm if nrm > 0 else np.array([1., 0.])
        cruise = E.cruise_speed

        self._seg_C = {}
        for r in self.inst.regions:
            for ti, ch in enumerate(r.chains):
                rho = AtmosphereParams.air_density(ch.height)
                wv = w.speed_at_height(ch.height) * wdir
                rings = ch.rings if ch.rings else []
                for ri, ring in enumerate(rings):
                    for si, seg in enumerate(ring.segments):
                        L = seg.length
                        if L > 0:
                            t = np.array([seg.end.x - seg.start.x,
                                          seg.end.y - seg.start.y]) / L
                        else:
                            t = np.array([1., 0.])
                        c_f = E.E_xy * rho * float(np.linalg.norm(cruise * t - wv))
                        c_b = E.E_xy * rho * float(np.linalg.norm(-cruise * t - wv))
                        self._seg_C[(r.id, ti, ri, si)] = (c_f, c_b)

        self._centroid = {}
        for r in self.inst.regions:
            pts = np.array([[p.x, p.y] for p in r.boundary])
            self._centroid[r.id] = pts.mean(axis=0)

        self._pair_F = {}
        ids = [r.id for r in self.inst.regions]
        for r1 in ids:
            for r2 in ids:
                if r1 == r2:
                    continue
                d = self._centroid[r2] - self._centroid[r1]
                nrm = float(np.linalg.norm(d))
                d = d / nrm if nrm > 0 else np.array([1., 0.])
                ent = {}
                for ti, ch in enumerate(self.inst.regions[r1].chains):
                    rho = AtmosphereParams.air_density(ch.height)
                    wv = w.speed_at_height(ch.height) * wdir
                    f_xy = E.E_xy * rho * float(np.linalg.norm(cruise * d - wv))
                    v_z = 0.5 * E.E_z * E.vertical_speed * rho
                    ent[ti] = (f_xy, v_z)
                self._pair_F[(r1, r2)] = ent

        self._depot_F = {}
        dep = self.inst.depot
        for r in self.inst.regions:
            d = self._centroid[r.id] - np.array([dep.x, dep.y])
            nrm = float(np.linalg.norm(d))
            d = d / nrm if nrm > 0 else np.array([1., 0.])
            ent = {}
            for ti, ch in enumerate(r.chains):
                rho = AtmosphereParams.air_density(ch.height)
                wv = w.speed_at_height(ch.height) * wdir
                f_out = E.E_xy * rho * float(np.linalg.norm(cruise * d - wv))
                f_ret = E.E_xy * rho * float(np.linalg.norm(-cruise * d - wv))
                v_z = 0.5 * E.E_z * E.vertical_speed * rho
                ent[ti] = (f_out, f_ret, v_z)
            self._depot_F[r.id] = ent

    @property
    def name(self) -> str:
        return "Rings"

    def variable_summary(self) -> Dict[str, int]:
        n_bin = sum(1 for v in self.model.getVars() if v.VType == GRB.BINARY)
        n_cont = sum(1 for v in self.model.getVars() if v.VType == GRB.CONTINUOUS)
        n_int = sum(1 for v in self.model.getVars() if v.VType == GRB.INTEGER)
        return {"binary": n_bin, "continuous": n_cont, "integer": n_int,
                "constraints": self.model.NumConstrs}

    def _classify_edges(self):
        self._intra_rl = list(self._ring_rl)
        self._inter_edges: List[Tuple[Vertex, Vertex, int, int]] = []
        for u in self.verts:
            for v in self.verts:
                if u is v: continue
                if u.region_id != v.region_id:
                    self._inter_edges.append((u, v, u.region_id, v.region_id))
                elif self._ring_index[u] != self._ring_index[v]:
                    self._inter_edges.append((u, v, u.region_id, v.region_id))

    def _lin_bb(self, b1, b2, name):
        b = self.model.addVar(vtype=GRB.BINARY, name=f"linbb_{name}")
        self.model.addConstr(b <= b1, name=f"bb_{name}_1")
        self.model.addConstr(b <= b2, name=f"bb_{name}_2")
        self.model.addConstr(b >= b1 + b2 - 1, name=f"bb_{name}_3")
        self.model.addConstr(b >= 0, name=f"bb_{name}_4")
        self._lin_vars.append(b); return b

    def _lin_bc(self, b, c, lo, hi, name, M=None):
        Mv = M if M else hi
        e = self.model.addVar(lb=min(0., lo), ub=abs(hi), vtype=GRB.CONTINUOUS,
                              name=f"linbc_{name}")
        self.model.addConstr(e <= Mv * b, name=f"bc_{name}_1")
        self.model.addConstr(e <= c - lo * (1 - b), name=f"bc_{name}_2")
        self.model.addConstr(e >= c - Mv * (1 - b), name=f"bc_{name}_3")
        self.model.addConstr(e >= lo * b, name=f"bc_{name}_4")
        self._lin_vars.append(e); return e

    def _lin_bcs(self, b, c, name):
        e = self.model.addVar(lb=0., ub=1., vtype=GRB.CONTINUOUS, name=f"linbcs_{name}")
        self.model.addConstr(e <= b, name=f"bcs_{name}_1")
        self.model.addConstr(e <= c, name=f"bcs_{name}_2")
        self.model.addConstr(e >= c - (1 - b), name=f"bcs_{name}_3")
        self.model.addConstr(e >= 0, name=f"bcs_{name}_4")
        self._lin_vars.append(e); return e

    def _create_variables(self):
        for v in self.verts:
            r = self.inst.regions[v.region_id]
            ri = self._ring_index[v]
            for ti, ch in enumerate(r.chains):
                rings = ch.rings if ch.rings else []
                if ri < len(rings):
                    ns = len(rings[ri].segments)
                    for si in range(ns):
                        k = (v.region_id, v.idx, ti, si, ri)
                        self.mu[k] = self.model.addVar(
                            vtype=GRB.BINARY, name=f"mu_r{v.region_id}_v{v.idx}_t{ti}_s{si}_ri{ri}")

        for v in self.verts:
            self.gamma[v] = self.model.addVar(
                lb=0., ub=1., vtype=GRB.CONTINUOUS, name=f"gm_r{v.region_id}_v{v.idx}")

        for r in self.inst.regions:
            for ti in range(len(r.chains)):
                self.alpha[(r.id, ti)] = self.model.addVar(
                    vtype=GRB.BINARY, name=f"al_r{r.id}_t{ti}")

        for v in self.verts:
            ri = self._ring_index[v]
            mx = 0
            for c in self.inst.regions[v.region_id].chains:
                rings = c.rings if c.rings else []
                if ri < len(rings):
                    mx = max(mx, len(rings[ri].segments))
            self.lambd[v] = self.model.addVar(
                lb=0., ub=float(max(mx, 1)), vtype=GRB.CONTINUOUS,
                name=f"la_r{v.region_id}_v{v.idx}")

        bb = self._get_bounds()
        for v in self.verts:
            self.P_x[v] = self.model.addVar(lb=bb[0], ub=bb[2], vtype=GRB.CONTINUOUS,
                                            name=f"Px_{v}")
            self.P_y[v] = self.model.addVar(lb=bb[1], ub=bb[3], vtype=GRB.CONTINUOUS,
                                            name=f"Py_{v}")
            self.P_z[v] = self.model.addVar(lb=0., ub=self._max_h(), vtype=GRB.CONTINUOUS,
                                            name=f"Pz_{v}")

        for r in self.inst.regions:
            self.rho_sel[r.id] = self.model.addVar(
                lb=0., ub=AtmosphereParams.air_density(0.),
                vtype=GRB.CONTINUOUS, name=f"ro_r{r.id}")

        valid_edges = set()
        for v in self.verts:
            valid_edges.add((self.depot_v, v))
            valid_edges.add((v, self.depot_v))
        for (u, v) in self._intra_rl:
            valid_edges.add((u, v))
        for (a, b) in self._connectors:
            valid_edges.add((a, b))
        for (u, v, _, _) in self._inter_edges:
            valid_edges.add((u, v))
        for (u, v) in valid_edges:
            for o in range(self.O):
                self.x[(u, v, o)] = self.model.addVar(
                    vtype=GRB.BINARY, name=f"x_{u}_{v}_o{o}")

        for v in self.all_nodes:
            for o in range(self.O):
                self.y[(v, o)] = self.model.addVar(
                    vtype=GRB.BINARY, name=f"y_{v}_o{o}")

        for o in range(self.O):
            self.zeta[o] = self.model.addVar(vtype=GRB.BINARY, name=f"zt_o{o}")
            self.k[o] = self.model.addVar(lb=0, ub=len(self.verts), vtype=GRB.INTEGER,
                                          name=f"k_o{o}")

        md = self._max_d()
        mh = self._max_h()
        mc = self._max_chain_len()
        max_edge = md + mh + mc
        for u in self.all_nodes:
            for v in self.all_nodes:
                if u != v:
                    self.edge_dist[(u, v)] = self.model.addVar(
                        lb=0., ub=max_edge, vtype=GRB.CONTINUOUS, name=f"dst_{u}_{v}")
                    self.edge_energy[(u, v)] = self.model.addVar(
                        lb=0., ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"enr_{u}_{v}")

        for v in self.verts:
            r = self.inst.regions[v.region_id]
            ri = self._ring_index[v]
            for ti, ch in enumerate(r.chains):
                rings = ch.rings if ch.rings else []
                if ri < len(rings):
                    ns = len(rings[ri].segments)
                    for si in range(ns):
                        mk = (v.region_id, v.idx, ti, si, ri)
                        if mk in self.mu:
                            self._eta[(v, ti, si, ri)] = self._lin_bcs(
                                self.mu[mk], self.gamma[v],
                                f"eta_r{v.region_id}_v{v.idx}_t{ti}_s{si}_ri{ri}")

        # Segment-traversal indicators for exact wind-aware arc energy
        # (wind_aware + K>1 only): I[(u,v),ti,s] tells whether arc (u,v)
        # covers segment s of the ring on chain ti.
        self._I = {}
        if self.wind_aware and self.K > 1:
            for (u, v) in self._intra_rl:
                r = self.inst.regions[u.region_id]
                ri = self._ring_index[u]
                for ti, ch in enumerate(r.chains):
                    info = self._ring_info.get((r.id, ti, ri))
                    if info is None:
                        continue
                    for si in range(info["num_seg"]):
                        self._I[(u, v, ti, si)] = self.model.addVar(
                            lb=0., ub=1., vtype=GRB.CONTINUOUS,
                            name=f"I_{u}_{v}_t{ti}_s{si}")

    def _get_bounds(self):
        xs, ys = [], []
        for r in self.inst.regions:
            for p in r.boundary:
                xs.append(p.x); ys.append(p.y)
            for c in r.chains:
                for s in c.segments:
                    xs.append(s.start.x); xs.append(s.end.x)
                    ys.append(s.start.y); ys.append(s.end.y)
        return (min(xs) - 10, min(ys) - 10, max(xs) + 10, max(ys) + 10)

    def _max_h(self):
        return max((c.height for r in self.inst.regions for c in r.chains), default=0) + 10.

    def _max_d(self):
        b = self._get_bounds()
        return float(np.sqrt((b[2] - b[0])**2 + (b[3] - b[1])**2))

    def _max_chain_len(self):
        return max((c.total_length for r in self.inst.regions for c in r.chains), default=self._max_d())

    def _valid_ring_info(self, v, ti):
        r = self.inst.regions[v.region_id]
        ri = self._ring_index[v]
        ch = r.chains[ti]
        rings = ch.rings if ch.rings else []
        if ri < len(rings):
            return rings[ri], ti, ri
        return None

    def _add_location_constraints(self):
        for v in self.verts:
            r = self.inst.regions[v.region_id]
            ri = self._ring_index[v]
            ex, ey, ez = gp.LinExpr(), gp.LinExpr(), gp.LinExpr()
            for ti, ch in enumerate(r.chains):
                rings = ch.rings if ch.rings else []
                if ri >= len(rings): continue
                ring = rings[ri]
                for si, seg in enumerate(ring.segments):
                    mk = (v.region_id, v.idx, ti, si, ri)
                    if mk not in self.mu: continue
                    mu = self.mu[mk]
                    et = self._eta.get((v, ti, si, ri))
                    if et is None: continue
                    ex += mu * seg.start.x + et * (seg.end.x - seg.start.x)
                    ey += mu * seg.start.y + et * (seg.end.y - seg.start.y)
                    ez += mu * ch.height
            self.model.addConstr(self.P_x[v] == ex, name=f"LC1x_{v}")
            self.model.addConstr(self.P_y[v] == ey, name=f"LC1y_{v}")
            self.model.addConstr(self.P_z[v] == ez, name=f"LC1z_{v}")

        for v in self.verts:
            r = self.inst.regions[v.region_id]
            ri = self._ring_index[v]
            for ti in range(len(r.chains)):
                rings = r.chains[ti].rings if r.chains[ti].rings else []
                sm = gp.LinExpr()
                if ri < len(rings):
                    ns = len(rings[ri].segments)
                    for si in range(ns):
                        mk = (v.region_id, v.idx, ti, si, ri)
                        if mk in self.mu: sm += self.mu[mk]
                self.model.addConstr(sm == self.alpha[(r.id, ti)], name=f"LC2_{v}_t{ti}")

        for r in self.inst.regions:
            self.model.addConstr(
                gp.quicksum(self.alpha[(r.id, ti)] for ti in range(len(r.chains))) == 1,
                name=f"LC3_r{r.id}")

        for v in self.verts:
            r = self.inst.regions[v.region_id]
            ri = self._ring_index[v]
            e = gp.LinExpr()
            for ti, ch in enumerate(r.chains):
                rings = ch.rings if ch.rings else []
                if ri < len(rings):
                    for si in range(len(rings[ri].segments)):
                        mk = (v.region_id, v.idx, ti, si, ri)
                        if mk in self.mu: e += si * self.mu[mk]
            e += self.gamma[v]
            self.model.addConstr(self.lambd[v] == e, name=f"LC4_{v}")

        if self.K == 1:
            # Single coincident launch/retrieve pair per ring (original).
            for (u, v) in self._intra_rl:
                self.model.addConstr(self.lambd[u] == self.lambd[v], name=f"LC5_{u}_{v}")
        else:
            # K arcs in series partitioning the ring: launch_1 is anchored at
            # the ring start (0), consecutive arcs join at the split points,
            # and retrieve_K is anchored at the ring end of the selected chain.
            # Zero-length arcs are allowed, so K is an upper bound.
            for (r_id, ri), arcs in self._ring_arcs.items():
                r = self.inst.regions[r_id]
                self.model.addConstr(self.lambd[arcs[0][0]] == 0.,
                                     name=f"LC5a_r{r_id}_ri{ri}")
                for j in range(len(arcs) - 1):
                    self.model.addConstr(
                        self.lambd[arcs[j][1]] == self.lambd[arcs[j + 1][0]],
                        name=f"LC5b_r{r_id}_ri{ri}_j{j}")
                e = gp.LinExpr()
                for ti, ch in enumerate(r.chains):
                    info = self._ring_info.get((r_id, ti, ri))
                    if info is not None:
                        e += info["num_seg"] * self.alpha[(r_id, ti)]
                self.model.addConstr(self.lambd[arcs[-1][1]] == e,
                                     name=f"LC5c_r{r_id}_ri{ri}")
            for (u, v) in self._intra_rl:
                self.model.addConstr(self.lambd[u] <= self.lambd[v],
                                     name=f"LC6_{u}_{v}")

        for r in self.inst.regions:
            e = gp.LinExpr()
            for ti, ch in enumerate(r.chains):
                e += AtmosphereParams.air_density(ch.height) * self.alpha[(r.id, ti)]
            self.model.addConstr(self.rho_sel[r.id] == e, name=f"rho_r{r.id}")

    def _add_drone_path_constraints(self):
        V, Vp, Or = self.all_nodes, self.verts, range(self.O)
        for o in Or:
            self.model.addConstr(
                gp.quicksum(self.x[(self.depot_v, v, o)] for v in V if v != self.depot_v
                            if (self.depot_v, v, o) in self.x) == 1 - self.zeta[o],
                name=f"DP1_o{o}")
        for v in Vp:
            for o in Or:
                inn = gp.quicksum(self.x[(u, v, o)] for u in V if u != v
                                  if (u, v, o) in self.x)
                out = gp.quicksum(self.x[(v, u, o)] for u in V if u != v
                                  if (v, u, o) in self.x)
                self.model.addConstr(inn == out, name=f"DP2_{v}_o{o}")
        for o in Or:
            self.model.addConstr(
                gp.quicksum(self.x[(v, self.depot_v, o)] for v in Vp
                            if (v, self.depot_v, o) in self.x) == 1 - self.zeta[o],
                name=f"DP3_o{o}")
        for v in V:
            for o in Or:
                out = gp.quicksum(self.x[(v, u, o)] for u in V if u != v
                                  if (v, u, o) in self.x)
                self.model.addConstr(out == self.y[(v, o)], name=f"DP4_{v}_o{o}")
                inn = gp.quicksum(self.x[(u, v, o)] for u in V if u != v
                                  if (u, v, o) in self.x)
                self.model.addConstr(inn == self.y[(v, o)], name=f"DP5_{v}_o{o}")
        for v in Vp:
            self.model.addConstr(
                gp.quicksum(self.y[(v, o)] for o in Or) == 1, name=f"DP6_{v}")

        nV = len(Vp)
        self._uvar = {}
        for v in Vp:
            for o in Or:
                self._uvar[(v, o)] = self.model.addVar(
                    lb=1, ub=nV, vtype=GRB.INTEGER, name=f"u_{v}_o{o}")
        for v1 in Vp:
            for v2 in Vp:
                if v1 != v2:
                    for o in Or:
                        k = (v1, v2, o)
                        if k in self.x:
                            self.model.addConstr(
                                self._uvar[(v1, o)] - self._uvar[(v2, o)] + 1
                                <= nV * (1 - self.x[k]), name=f"DP7_{v1}_{v2}_o{o}")

        for (u, v) in self._intra_rl:
            e = gp.LinExpr()
            for o in Or:
                kf, kb = (u, v, o), (v, u, o)
                if kf in self.x: e += self.x[kf]
                if kb in self.x: e += self.x[kb]
            self.model.addConstr(e == 1, name=f"DP8_{u}_{v}")

    def _add_valid_inequalities(self):
        for o in range(self.O - 1):
            self.model.addConstr(self.zeta[o] <= self.zeta[o + 1], name=f"Mon_o{o}")
        for o in range(self.O):
            self.model.addConstr(
                self.k[o] == gp.quicksum(self.y[(v, o)] for v in self.verts),
                name=f"kC_o{o}")
        for o in range(self.O):
            self.model.addConstr(
                gp.quicksum(self.k[o2] for o2 in range(o)) >= len(self.verts) * self.zeta[o],
                name=f"VI1_o{o}")
            self.model.addConstr(self.k[o] >= 1 - self.zeta[o], name=f"VI2_o{o}")

    def _add_dfs_constraints(self):
        """Cumulative distance from the ring start for every vertex (paper's
        dfs): dfs[v] = sum_t sum_s [cum_before(t,s)*mu + L(t,s)*eta].
        Only used for K>1, where arc lengths depend on split positions."""
        for v in self.verts:
            r = self.inst.regions[v.region_id]
            ri = self._ring_index[v]
            e = gp.LinExpr()
            for ti, ch in enumerate(r.chains):
                info = self._ring_info.get((r.id, ti, ri))
                if info is None:
                    continue
                for si in range(info["num_seg"]):
                    mk = (v.region_id, v.idx, ti, si, ri)
                    if mk in self.mu:
                        e += (info["cum_before"][si] * self.mu[mk]
                              + info["seg_lens"][si] * self._eta[(v, ti, si, ri)])
            self.model.addConstr(self.dfs[v] == e, name=f"DFS_{v}")

    def _add_intra_ring_distance_energy(self):
        max_cl = self._max_chain_len()
        if self.K == 1 and not self.wind_aware:
            for (u, v) in self._intra_rl:
                r = self.inst.regions[u.region_id]
                ri = self._ring_index[u]
                e = gp.LinExpr()
                for ti, ch in enumerate(r.chains):
                    rings = ch.rings if ch.rings else []
                    if ri < len(rings):
                        p = rings[ri].perimeter
                        e += p * self.alpha[(r.id, ti)]
                self.model.addConstr(self.edge_dist[(u, v)] == e, name=f"irdst_{u}_{v}")
                self.model.addConstr(self.edge_dist[(v, u)] == e, name=f"irdst_r_{u}_{v}")
                self.model.addConstr(self.edge_energy[(u, v)] == e, name=f"irenr_{u}_{v}")
                self.model.addConstr(self.edge_energy[(v, u)] == e, name=f"irenr_r_{u}_{v}")
            return
        if self.K == 1 and self.wind_aware:
            # Full loop: true energy per direction, linear in chain selection.
            for (u, v) in self._intra_rl:
                r = self.inst.regions[u.region_id]
                ri = self._ring_index[u]
                e = gp.LinExpr()
                ef = gp.LinExpr()
                eb = gp.LinExpr()
                for ti, ch in enumerate(r.chains):
                    rings = ch.rings if ch.rings else []
                    if ri < len(rings):
                        p = rings[ri].perimeter
                        e += p * self.alpha[(r.id, ti)]
                        ef += sum(self._seg_C[(r.id, ti, ri, si)][0]
                                  * rings[ri].segments[si].length
                                  for si in range(len(rings[ri].segments))
                                  ) * self.alpha[(r.id, ti)]
                        eb += sum(self._seg_C[(r.id, ti, ri, si)][1]
                                  * rings[ri].segments[si].length
                                  for si in range(len(rings[ri].segments))
                                  ) * self.alpha[(r.id, ti)]
                self.model.addConstr(self.edge_dist[(u, v)] == e, name=f"irdst_{u}_{v}")
                self.model.addConstr(self.edge_dist[(v, u)] == e, name=f"irdst_r_{u}_{v}")
                self.model.addConstr(self.edge_energy[(u, v)] == ef, name=f"irenr_{u}_{v}")
                self.model.addConstr(self.edge_energy[(v, u)] == eb, name=f"irenr_r_{u}_{v}")
            return
        if not self.wind_aware:
            for (u, v) in self._intra_rl:
                self.model.addConstr(self.edge_dist[(u, v)] == self.dfs[v] - self.dfs[u],
                                     name=f"irdst_{u}_{v}")
                self.model.addConstr(self.edge_dist[(v, u)] == self.dfs[v] - self.dfs[u],
                                     name=f"irdst_r_{u}_{v}")
                self.model.addConstr(self.edge_energy[(u, v)] == self.dfs[v] - self.dfs[u],
                                     name=f"irenr_{u}_{v}")
                self.model.addConstr(self.edge_energy[(v, u)] == self.dfs[v] - self.dfs[u],
                                     name=f"irenr_r_{u}_{v}")
            for (a, b) in self._connectors:
                self.model.addConstr(self.edge_dist[(a, b)] == 0., name=f"cdst_{a}_{b}")
                self.model.addConstr(self.edge_energy[(a, b)] == 0., name=f"cenr_{a}_{b}")
            return
        # K>1 + wind_aware: distances as dfs differences (unchanged), exact
        # wind-aware arc energies per direction. For arc (u,v) on chain t and
        # segment s, traversed length = L(s)*I - start/end partials, where the
        # partials reuse eta (gamma*mu) and (mu - eta):
        #   w_fwd (u->v, upward):   sum C_fwd*L*[I - eta_u - (mu_v - eta_v)]
        #   w_bwd (v->u, downward): sum C_bwd*L*[I - (mu_v - eta_v) - eta_u]
        # (start-untraversed at the departure end, end-untraversed at the
        # arrival end; swapping them makes backward energies negative).
        for (u, v) in self._intra_rl:
            r = self.inst.regions[u.region_id]
            ri = self._ring_index[u]
            self.model.addConstr(self.edge_dist[(u, v)] == self.dfs[v] - self.dfs[u],
                                 name=f"irdst_{u}_{v}")
            self.model.addConstr(self.edge_dist[(v, u)] == self.dfs[v] - self.dfs[u],
                                 name=f"irdst_r_{u}_{v}")
            ef = gp.LinExpr()
            eb = gp.LinExpr()
            for ti, ch in enumerate(r.chains):
                info = self._ring_info.get((r.id, ti, ri))
                if info is None:
                    continue
                ns = info["num_seg"]
                for si in range(ns):
                    iv = self._I[(u, v, ti, si)]
                    cf, cb = self._seg_C[(r.id, ti, ri, si)]
                    L = info["seg_lens"][si]
                    mu_u = self.mu.get((r.id, u.idx, ti, si, ri))
                    mu_v = self.mu.get((r.id, v.idx, ti, si, ri))
                    et_u = self._eta.get((u, ti, si, ri))
                    et_v = self._eta.get((v, ti, si, ri))
                    # segment s covered iff si_u <= s <= si_v:
                    #   I >= [si_u<=s] + [s<=si_v] - 1
                    #   I <= [si_u<=s], I <= [s<=si_v]
                    lo_u = gp.quicksum(
                        self.mu[(r.id, u.idx, ti, sp, ri)]
                        for sp in range(si + 1)
                        if (r.id, u.idx, ti, sp, ri) in self.mu)
                    hi_v = gp.quicksum(
                        self.mu[(r.id, v.idx, ti, sp, ri)]
                        for sp in range(si, ns)
                        if (r.id, v.idx, ti, sp, ri) in self.mu)
                    self.model.addConstr(iv >= lo_u + hi_v - 1,
                                         name=f"Ilo_{u}_{v}_t{ti}_s{si}")
                    self.model.addConstr(iv <= lo_u,
                                         name=f"Ihiu_{u}_{v}_t{ti}_s{si}")
                    self.model.addConstr(iv <= hi_v,
                                         name=f"Ihiv_{u}_{v}_t{ti}_s{si}")
                    ef += cf * L * (iv
                                     - (et_u if et_u is not None else 0.)
                                     - ((mu_v - et_v) if (mu_v is not None
                                                          and et_v is not None) else 0.))
                    eb += cb * L * (iv
                                     - ((mu_v - et_v) if (mu_v is not None
                                                          and et_v is not None) else 0.)
                                     - (et_u if et_u is not None else 0.))
            self.model.addConstr(self.edge_energy[(u, v)] == ef, name=f"irenr_{u}_{v}")
            self.model.addConstr(self.edge_energy[(v, u)] == eb, name=f"irenr_r_{u}_{v}")
        for (a, b) in self._connectors:
            self.model.addConstr(self.edge_dist[(a, b)] == 0., name=f"cdst_{a}_{b}")
            self.model.addConstr(self.edge_energy[(a, b)] == 0., name=f"cenr_{a}_{b}")

    def _add_inter_distance_energy(self):
        md = self._max_d(); mh = self._max_h()
        for (u, v, r1, r2) in self._inter_edges:
            du = self.model.addVar(lb=0., ub=md, vtype=GRB.CONTINUOUS, name=f"dxy_{u}_{v}")
            dv = self.model.addVar(lb=0., ub=mh, vtype=GRB.CONTINUOUS, name=f"dz_{u}_{v}")
            dx = self.model.addVar(lb=-GRB.INFINITY, name=f"idx_{u}_{v}")
            dy = self.model.addVar(lb=-GRB.INFINITY, name=f"idy_{u}_{v}")
            self.model.addConstr(dx == self.P_x[u] - self.P_x[v], name=f"dx_{u}_{v}")
            self.model.addConstr(dy == self.P_y[u] - self.P_y[v], name=f"dy_{u}_{v}")
            self.model.addConstr(du * du >= dx * dx + dy * dy, name=f"dxy_{u}_{v}")
            self.model.addConstr(dv >= self.P_z[u] - self.P_z[v], name=f"dz1_{u}_{v}")
            self.model.addConstr(dv >= self.P_z[v] - self.P_z[u], name=f"dz2_{u}_{v}")
            self.model.addConstr(self.edge_dist[(u, v)] == du + dv, name=f"idst_{u}_{v}")
            if not self.wind_aware:
                self.model.addConstr(self.edge_energy[(u, v)] == du + dv, name=f"ienr_{u}_{v}")
                continue
            # Wind-aware (real joules): horizontal energy with the precomputed
            # per-(pair, departure-chain) airspeed factor, vertical energy with
            # the departure density; both bilinear terms discretised exactly
            # over the one-hot chain selection (binary x continuous).
            # Same-region legs (between rings) reuse the depot-direction
            # factors of the departure region as a proxy heading.
            exy = gp.LinExpr()
            e_z = gp.LinExpr()
            for ti in range(len(self.inst.regions[r1].chains)):
                if r1 == r2:
                    f_xy, _, v_z = self._depot_F[r1][ti]
                else:
                    f_xy, v_z = self._pair_F[(r1, r2)][ti]
                q = self._lin_bc(self.alpha[(r1, ti)], du, 0., md,
                                 f"ienrxy_{u}_{v}_t{ti}")
                exy += f_xy * q
                qz = self._lin_bc(self.alpha[(r1, ti)], dv, 0., mh,
                                  f"ienrz_{u}_{v}_t{ti}")
                e_z += v_z * qz
            self.model.addConstr(self.edge_energy[(u, v)] == exy + e_z,
                                 name=f"ienr_{u}_{v}")

    def _add_depot_distance_energy(self):
        dep = self.inst.depot
        for v in self.verts:
            dx = self.model.addVar(lb=-GRB.INFINITY, name=f"ddx_{v}")
            dy = self.model.addVar(lb=-GRB.INFINITY, name=f"ddy_{v}")
            dxy = self.model.addVar(lb=0., ub=self._max_d(), vtype=GRB.CONTINUOUS,
                                    name=f"ddxy_{v}")
            self.model.addConstr(dx == self.P_x[v] - dep.x, name=f"ddx_{v}")
            self.model.addConstr(dy == self.P_y[v] - dep.y, name=f"ddy_{v}")
            self.model.addConstr(dxy * dxy >= dx * dx + dy * dy, name=f"ddxy_{v}")
            dz = self.model.addVar(lb=0., ub=self._max_h(), vtype=GRB.CONTINUOUS,
                                   name=f"ddz_{v}")
            self.model.addConstr(dz >= self.P_z[v], name=f"ddz1_{v}")
            self.model.addConstr(dz >= -self.P_z[v], name=f"ddz2_{v}")

            self.model.addConstr(self.edge_dist[(self.depot_v, v)] == dxy + dz,
                                 name=f"ddst_{v}")
            self.model.addConstr(self.edge_dist[(v, self.depot_v)] == dxy + dz,
                                 name=f"ddst_s_{v}")

            if not self.wind_aware:
                rho0 = AtmosphereParams.air_density(0.)
                r = self.inst.regions[v.region_id]
                centroid = np.mean([(p.x, p.y) for p in r.boundary], axis=0)
                dep_to_centroid = np.array([centroid[0] - dep.x, centroid[1] - dep.y, 0.])
                nrm = np.linalg.norm(dep_to_centroid)
                dir_unit = dep_to_centroid / nrm if nrm > 0 else np.array([1., 0., 0.])
                ws0 = self.inst.wind.speed_at_height(0.)
                wdir = self.inst.wind.direction
                nu_d_xy = float(np.linalg.norm(
                    self.inst.drone.cruise_speed * dir_unit - ws0 * wdir))
                En_xy = self.inst.drone.E_xy * rho0 * nu_d_xy
                En_z = 0.5 * self.inst.drone.E_z * self.inst.drone.vertical_speed * rho0
                self.model.addConstr(
                    self.edge_energy[(self.depot_v, v)] == En_xy * dxy + En_z * dz,
                    name=f"denr_{v}")
                self.model.addConstr(
                    self.edge_energy[(v, self.depot_v)] == En_xy * dxy,
                    name=f"denr_s_{v}")
                continue
            # Wind-aware (real joules): same fixed depot->centroid direction,
            # but density and wind evaluated at the selected chain altitude
            # (discretised exactly over the one-hot chain selection).
            r = self.inst.regions[v.region_id]
            e_out = gp.LinExpr()
            e_ret = gp.LinExpr()
            for ti in range(len(r.chains)):
                f_out, f_ret, v_z = self._depot_F[r.id][ti]
                qx = self._lin_bc(self.alpha[(r.id, ti)], dxy, 0.,
                                  self._max_d(), f"denrxy_{v}_t{ti}")
                e_out += f_out * qx
                e_ret += f_ret * qx
                qz = self._lin_bc(self.alpha[(r.id, ti)], dz, 0.,
                                  self._max_h(), f"denrz_{v}_t{ti}")
                e_out += v_z * qz
            self.model.addConstr(
                self.edge_energy[(self.depot_v, v)] == e_out, name=f"denr_{v}")
            self.model.addConstr(
                self.edge_energy[(v, self.depot_v)] == e_ret, name=f"denr_s_{v}")

    def _add_endurance_constraints(self):
        me = self._max_chain_len() * 10.
        for o in range(self.O):
            e = gp.LinExpr()
            for u in self.all_nodes:
                for v in self.all_nodes:
                    if u != v:
                        ek = (u, v, o)
                        if ek not in self.x: continue
                        le = self._lin_bc(self.x[ek], self.edge_energy[(u, v)],
                                          0., me, f"end_{u}_{v}_o{o}", M=self._big_M)
                        e += le
            self.model.addConstr(e <= self.inst.drone.max_endurance, name=f"End_o{o}")

    def _set_objective(self):
        me = max(self._max_d() * 10., self._max_chain_len())
        obj = gp.LinExpr()
        for u in self.all_nodes:
            for v in self.all_nodes:
                if u != v:
                    for o in range(self.O):
                        xk = (u, v, o)
                        if xk not in self.x: continue
                        ld = self._lin_bc(self.x[xk], self.edge_dist[(u, v)],
                                          0., me, f"obj_{u}_{v}_o{o}", M=self._big_M)
                        obj += ld
        self.model.setObjective(obj, GRB.MINIMIZE)

    # ------------------------------------------------------------------
    # Warm start from a K=1 solution (exact mapping onto K>=2 arcs)
    # ------------------------------------------------------------------
    def set_warm_start_from_k1(self, sol_k1: Solution):
        """Map a K=1 full-loop solution exactly onto this K>=2 model.

        For every ring, the K=1 loop entered at ``c`` becomes the K arcs
        ``[0, c], [c, c], ..., [c, ns]`` covered consecutively in the same
        operation (using the wrap connector), so the mapped solution has
        exactly the same cost. Requires self.K >= 2.
        """
        assert self.K >= 2, "mapping only defined for K>=2"
        dep_v = self.depot_v
        chain_sel = sol_k1.chain_selection

        # alpha
        for (r_id, ti), var in self.alpha.items():
            var.Start = 1.0 if chain_sel.get(r_id) == ti else 0.0

        def _chain_ring(r_id, ri):
            ti = chain_sel.get(r_id)
            ch = self.inst.regions[r_id].chains[ti]
            return ti, ch.rings[ri]

        def _point_on_ring(ti_ring, lam):
            ns = len(ti_ring.segments)
            si = min(max(int(lam), 0), ns - 1)
            gamma = min(max(lam - si, 0.0), 1.0)
            seg = ti_ring.segments[si]
            px = seg.start.x + gamma * (seg.end.x - seg.start.x)
            py = seg.start.y + gamma * (seg.end.y - seg.start.y)
            pz = seg.start.z + gamma * (seg.end.z - seg.start.z)
            return si, gamma, px, py, pz

        def _set_vertex(v, lam):
            r_id, idx = v.region_id, v.idx
            ti = chain_sel.get(r_id)
            r = self.inst.regions[r_id]
            ri = self._ring_index[v]
            _, ring = _chain_ring(r_id, ri)
            ns = len(ring.segments)
            si, gamma, px, py, pz = _point_on_ring(ring, lam)
            for ti2 in range(len(r.chains)):
                rings2 = r.chains[ti2].rings if r.chains[ti2].rings else []
                if ri < len(rings2):
                    for si2 in range(len(rings2[ri].segments)):
                        mk = (r_id, idx, ti2, si2, ri)
                        if mk in self.mu:
                            self.mu[mk].Start = 1.0 if (ti2 == ti and si2 == si) else 0.0
            if v in self.gamma:
                self.gamma[v].Start = gamma
            self.lambd[v].Start = float(lam)
            self.P_x[v].Start = px
            self.P_y[v].Start = py
            self.P_z[v].Start = pz

        # locate each ring's K=1 loop edge (op index + direction)
        loop_of = {}  # (r_id, ri) -> (op_idx, forward: bool, c)
        for oi, op in enumerate(sol_k1.operations):
            for (u, v) in op.edges:
                if (u.region_id >= 0 and u.region_id == v.region_id
                        and u.vtype == VertexType.LAUNCH and v.vtype == VertexType.RETRIEVE
                        and u.idx % 2 == 0 and v.idx == u.idx + 1):
                    ri = u.idx // 2
                    c = sol_k1.vertex_lambdas.get(u, 0.0)
                    loop_of[(u.region_id, ri)] = (oi, True, c)
                elif (u.region_id >= 0 and u.region_id == v.region_id
                        and u.vtype == VertexType.RETRIEVE and v.vtype == VertexType.LAUNCH
                        and v.idx % 2 == 0 and u.idx == v.idx + 1):
                    ri = v.idx // 2
                    c = sol_k1.vertex_lambdas.get(u, 0.0)
                    loop_of[(u.region_id, ri)] = (oi, False, c)

        # new edge lists per operation: K=1 loops are expanded into their
        # K-arc chains, and the neighbouring inter/depot legs are rewritten
        # onto the chain endpoints (a K=1 vertex cannot appear in the K-model,
        # otherwise its Start values would be silently skipped).
        def _is_loop_edge(a, b):
            if (a.region_id >= 0 and a.region_id == b.region_id
                    and a.vtype == VertexType.LAUNCH and b.vtype == VertexType.RETRIEVE
                    and a.idx % 2 == 0 and b.idx == a.idx + 1):
                return (a.region_id, a.idx // 2), True
            if (a.region_id >= 0 and a.region_id == b.region_id
                    and a.vtype == VertexType.RETRIEVE and b.vtype == VertexType.LAUNCH
                    and b.idx % 2 == 0 and a.idx == b.idx + 1):
                return (a.region_id, b.idx // 2), False
            return None

        def _chain_k1(edge_list):
            by_u = {}
            for e in edge_list:
                by_u.setdefault(e[0], []).append(e)
            start = None
            for e in edge_list:
                if e[0] == dep_v or e[0].region_id == -1:
                    start = e
                    break
            if start is None and edge_list:
                start = edge_list[0]
            if start is None:
                return []
            ordered = [start]
            used = {start}
            cur = start[1]
            while True:
                nxt = None
                for e in by_u.get(cur, []):
                    if e not in used:
                        nxt = e
                        break
                if nxt is None:
                    break
                ordered.append(nxt)
                used.add(nxt)
                cur = nxt[1]
            ordered += [e for e in edge_list if e not in used]
            return ordered

        new_edges = []
        for oi, op in enumerate(sol_k1.operations):
            ordered = _chain_k1(list(op.edges))
            # pass 1: arc chains + endpoint rewrites. In a feasible tour each
            # loop vertex touches exactly one non-loop edge (its entry/exit
            # leg, in the loop's own operation): the predecessor leg is
            # retargeted onto the chain start, the successor onto the chain end.
            # NOTE: rewrites are keyed by edge VALUE (tuples of Vertex are
            # hashable), never by id(): unpacking/repacking tuples creates new
            # objects and id-based lookups silently miss.
            rew_src = {}  # edge value -> new source vertex
            rew_tgt = {}  # edge value -> new target vertex
            seqs = []
            for (a, b) in ordered:
                lk = _is_loop_edge(a, b)
                seqs.append(('loop', lk[0], lk[1]) if lk else ('edge', (a, b)))
            for i, item in enumerate(seqs):
                if item[0] != 'loop':
                    continue
                if i == 0 or i == len(seqs) - 1:
                    continue  # degenerate: loop at tour boundary (no neighbour)
                _, key, fwd = item
                arcs = self._ring_arcs[key]
                n = len(arcs)
                if fwd:
                    start_v = arcs[-1][0]
                    end_v = arcs[-2][1]
                else:
                    start_v = arcs[0][1]
                    end_v = arcs[1][0]
                pa, pb = ordered[i - 1], ordered[i + 1]
                # predecessor ends at the loop entry -> retarget to chain start
                rew_tgt[pa] = start_v
                # successor starts at the loop exit -> retarget to chain end
                rew_src[pb] = end_v
            # pass 2: build the mapped edge list
            mapped = []
            for i, item in enumerate(seqs):
                if item[0] == 'edge':
                    (a, b) = item[1]
                    a = rew_src.get(item[1], a)
                    b = rew_tgt.get(item[1], b)
                    mapped.append((a, b))
                else:
                    _, key, fwd = item
                    arcs = self._ring_arcs[key]
                    n = len(arcs)
                    # Forward K=1 edge (lau->ret) at c becomes, in the same
                    # op: arc_K [c,ns], wrap, arc_1 [0,c], conn, zero
                    # arcs..., ending at ret_{K-2} (point c). Backward is
                    # mirrored, ending at lau_1 (point c). No trailing
                    # connector, so all degrees stay balanced (in == out).
                    if fwd:
                        mseq = [(arcs[-1][0], arcs[-1][1]),
                                (arcs[-1][1], arcs[0][0])]
                        for j in range(n - 1):
                            mseq.append((arcs[j][0], arcs[j][1]))
                            if j < n - 2:
                                mseq.append((arcs[j][1], arcs[j + 1][0]))
                    else:
                        mseq = [(arcs[0][1], arcs[0][0]),
                                (arcs[0][0], arcs[-1][1]),
                                (arcs[-1][1], arcs[-1][0])]
                        for j in range(n - 1, 1, -1):
                            mseq.append((arcs[j][0], arcs[j - 1][1]))
                            mseq.append((arcs[j - 1][1], arcs[j - 1][0]))
                    mapped.extend(mseq)
            new_edges.append(mapped)

        # lambda values for the new vertices
        for (r_id, ri), arcs in self._ring_arcs.items():
            info = loop_of.get((r_id, ri))
            c = info[2] if info else 0.0
            _, ring = _chain_ring(r_id, ri)
            ns = float(len(ring.segments))
            lams = {arcs[0][0]: 0.0}
            for j in range(len(arcs) - 1):
                lams[arcs[j][1]] = c
                lams[arcs[j + 1][0]] = c
            lams[arcs[-1][1]] = ns
            for vv, lam in lams.items():
                _set_vertex(vv, lam)

        # operation-level starts (chain edges topologically so MTZ order is valid)
        def _chain(edge_list):
            by_u = {}
            for e in edge_list:
                by_u.setdefault(e[0], []).append(e)
            start = None
            for e in edge_list:
                if e[0] == dep_v or e[0].region_id == -1:
                    start = e
                    break
            if start is None and edge_list:
                start = edge_list[0]
            if start is None:
                return []
            ordered = [start]
            used = {start}
            cur = start[1]
            while True:
                nxt = None
                for e in by_u.get(cur, []):
                    if e not in used:
                        nxt = e
                        break
                if nxt is None:
                    break
                ordered.append(nxt)
                used.add(nxt)
                cur = nxt[1]
            ordered += [e for e in edge_list if e not in used]
            return ordered

        for o in range(self.O):
            if o < len(new_edges) and new_edges[o]:
                edges = _chain(new_edges[o])
                self.zeta[o].Start = 0.0
                verts_in_op = set()
                for (u, v) in edges:
                    if u != dep_v:
                        verts_in_op.add(u)
                    if v != dep_v:
                        verts_in_op.add(v)
                self.k[o].Start = len(verts_in_op)
                for (u, v) in edges:
                    xk = (u, v, o)
                    if xk in self.x:
                        self.x[xk].Start = 1.0
                # MTZ order along the chained edge list
                seen = set()
                pos = 1
                for (u, v) in edges:
                    for w in (u, v):
                        if w != dep_v and w not in seen:
                            seen.add(w)
                            if (w, o) in self._uvar:
                                self._uvar[(w, o)].Start = pos
                            pos += 1
                for w in verts_in_op:
                    if (w, o) in self.y:
                        self.y[(w, o)].Start = 1.0
            else:
                self.zeta[o].Start = 1.0
                self.k[o].Start = 0

        self._complete_starts()
        self.model.update()

    # ------------------------------------------------------------------
    # Warm start from heuristic solution
    # ------------------------------------------------------------------
    # Complete all auxiliary Starts from the primary ones
    # ------------------------------------------------------------------
    def _complete_starts(self):
        """Fill Start values for every auxiliary variable consistently with
        the primary Starts (mu/gamma/lambd/P/alpha/x/y/zeta/k/uvar), so the
        solver accepts the warm start directly instead of running its own
        (often failing) completion on the large SOC model.

        NOTE: gurobipy only makes written Start values readable after
        model.update(), so it is called first; without it every read
        returns GRB.UNDEFINED (1e101) and all derived values are garbage.
        """
        import numpy as _np
        self.model.update()

        _UNDEF = 1e100

        def _st(v, default=0.0):
            try:
                s = v.Start
            except Exception:
                return default
            if s is None:
                return default
            try:
                if s != s or abs(s) >= _UNDEF:
                    return default
            except Exception:
                return default
            return s

        # --- eta = gamma * mu ---
        for (v, ti, si, ri), var in self._eta.items():
            mk = (v.region_id, v.idx, ti, si, ri)
            mu = _st(self.mu[mk]) if mk in self.mu else 0.0
            var.Start = mu * _st(self.gamma[v])
        self.model.update()  # flush eta before dfs reads it
        # --- dfs (cumulative ring distance) ---
        for v, var in self.dfs.items():
            r = self.inst.regions[v.region_id]
            ri = self._ring_index[v]
            tot = 0.0
            for ti, ch in enumerate(r.chains):
                info = self._ring_info.get((r.id, ti, ri))
                if info is None:
                    continue
                for si in range(info["num_seg"]):
                    mk = (v.region_id, v.idx, ti, si, ri)
                    if mk in self.mu:
                        tot += (info["cum_before"][si] * _st(self.mu[mk])
                                + info["seg_lens"][si]
                                * _st(self._eta[(v, ti, si, ri)]))
            var.Start = tot
        self.model.update()  # flush dfs/I before arc energies read them
        # --- rho_sel ---
        for r in self.inst.regions:
            tot = 0.0
            for ti, ch in enumerate(r.chains):
                tot += (AtmosphereParams.air_density(ch.height)
                        * _st(self.alpha[(r.id, ti)]))
            self.rho_sel[r.id].Start = tot
        # --- traversal indicators I (wind-aware K>1) ---
        for (u, v, ti, si), var in self._I.items():
            r = self.inst.regions[u.region_id]
            ri = self._ring_index[u]
            lo = sum(_st(self.mu[(r.id, u.idx, ti, sp, ri)])
                     for sp in range(si + 1)
                     if (r.id, u.idx, ti, sp, ri) in self.mu)
            hi = sum(_st(self.mu[(r.id, v.idx, ti, sp, ri)])
                     for sp in range(si, self._ring_info[(r.id, ti, ri)]["num_seg"])
                     if (r.id, v.idx, ti, sp, ri) in self.mu)
            var.Start = 1.0 if (lo >= 0.5 and hi >= 0.5) else 0.0
        self.model.update()  # flush I before arc energies read it
        # --- binary/integer completion ---
        # y follows the out-degree (DP4), including the depot: y=1 iff the
        # vertex has an outgoing x=1 edge in that operation.
        outdeg = {}
        for (u, w, o), xv in self.x.items():
            if _st(xv) == 1.0:
                outdeg[(u, o)] = 1
        for var in self.x.values():
            var.Start = 1.0 if _st(var) == 1.0 else 0.0
        for (v, o), var in self.y.items():
            var.Start = 1.0 if outdeg.get((v, o), 0) == 1 else 0.0
        for var in self._uvar.values():
            if not (_st(var) >= 1):
                var.Start = 1
        by_name = {vv.VarName: vv for vv in self.model.getVars()}
        # --- inter legs ---
        for (u, v, r1, r2) in self._inter_edges:
            pxu, pyu, pzu = (_st(self.P_x[u]), _st(self.P_y[u]), _st(self.P_z[u]))
            pxv, pyv, pzv = (_st(self.P_x[v]), _st(self.P_y[v]), _st(self.P_z[v]))
            dx, dy = pxu - pxv, pyu - pyv
            du = float(_np.hypot(dx, dy))
            dv = abs(pzu - pzv)
            by_name[f"dxy_{u}_{v}"].Start = du
            by_name[f"dz_{u}_{v}"].Start = dv
            by_name[f"idx_{u}_{v}"].Start = dx
            by_name[f"idy_{u}_{v}"].Start = dy
            self.edge_dist[(u, v)].Start = du + dv
            if not self.wind_aware:
                self.edge_energy[(u, v)].Start = du + dv
            else:
                tot = 0.0
                for ti in range(len(self.inst.regions[r1].chains)):
                    if r1 == r2:
                        f_xy, _, v_z = self._depot_F[r1][ti]
                    else:
                        f_xy, v_z = self._pair_F[(r1, r2)][ti]
                    a = _st(self.alpha[(r1, ti)])
                    q = by_name.get(f"linbc_ienrxy_{u}_{v}_t{ti}")
                    if q is not None:
                        q.Start = du * a
                        tot += f_xy * du * a
                    qz = by_name.get(f"linbc_ienrz_{u}_{v}_t{ti}")
                    if qz is not None:
                        qz.Start = dv * a
                        tot += v_z * dv * a
                self.edge_energy[(u, v)].Start = tot
        # --- depot legs ---
        for v in self.verts:
            r = self.inst.regions[v.region_id]
            dep = self.inst.depot
            px, py, pz = (_st(self.P_x[v]), _st(self.P_y[v]), _st(self.P_z[v]))
            dx, dy = px - dep.x, py - dep.y
            dxy = float(_np.hypot(dx, dy))
            dz = abs(pz)
            by_name[f"ddxy_{v}"].Start = dxy
            by_name[f"ddz_{v}"].Start = dz
            by_name[f"ddx_{v}"].Start = dx
            by_name[f"ddy_{v}"].Start = dy
            self.edge_dist[(self.depot_v, v)].Start = dxy + dz
            self.edge_dist[(v, self.depot_v)].Start = dxy + dz
            if not self.wind_aware:
                rho0 = AtmosphereParams.air_density(0.)
                centroid = _np.mean([(p.x, p.y) for p in r.boundary], axis=0)
                dvec = _np.array([centroid[0] - dep.x, centroid[1] - dep.y, 0.])
                nrm = float(_np.linalg.norm(dvec))
                du_ = dvec / nrm if nrm > 0 else _np.array([1., 0., 0.])
                ws0 = self.inst.wind.speed_at_height(0.)
                nu = float(_np.linalg.norm(
                    self.inst.drone.cruise_speed * du_ - ws0 * self.inst.wind.direction))
                exy = self.inst.drone.E_xy * rho0 * nu
                ez = 0.5 * self.inst.drone.E_z * self.inst.drone.vertical_speed * rho0
                self.edge_energy[(self.depot_v, v)].Start = exy * dxy + ez * dz
                self.edge_energy[(v, self.depot_v)].Start = exy * dxy
            else:
                tot_out = tot_ret = 0.0
                for ti in range(len(r.chains)):
                    f_out, f_ret, v_z = self._depot_F[r.id][ti]
                    a = _st(self.alpha[(r.id, ti)])
                    qx = by_name.get(f"linbc_denrxy_{v}_t{ti}")
                    if qx is not None:
                        qx.Start = dxy * a
                        tot_out += f_out * dxy * a
                        tot_ret += f_ret * dxy * a
                    qz = by_name.get(f"linbc_denrz_{v}_t{ti}")
                    if qz is not None:
                        qz.Start = dz * a
                        tot_out += v_z * dz * a
                self.edge_energy[(self.depot_v, v)].Start = tot_out
                self.edge_energy[(v, self.depot_v)].Start = tot_ret
        # --- intra ring edges ---
        for (u, v) in self._intra_rl:
            r = self.inst.regions[u.region_id]
            ri = self._ring_index[u]
            if self.K == 1 and not self.wind_aware:
                tot = 0.0
                for ti, ch in enumerate(r.chains):
                    rings = ch.rings if ch.rings else []
                    if ri < len(rings):
                        tot += (rings[ri].perimeter * _st(self.alpha[(r.id, ti)]))
                for kk in [(u, v), (v, u)]:
                    self.edge_dist[kk].Start = tot
                    self.edge_energy[kk].Start = tot
            elif self.K == 1 and self.wind_aware:
                tot = ef = eb = 0.0
                for ti, ch in enumerate(r.chains):
                    rings = ch.rings if ch.rings else []
                    if ri < len(rings):
                        a = _st(self.alpha[(r.id, ti)])
                        tot += rings[ri].perimeter * a
                        ef += sum(self._seg_C[(r.id, ti, ri, si)][0]
                                  * rings[ri].segments[si].length
                                  for si in range(len(rings[ri].segments))) * a
                        eb += sum(self._seg_C[(r.id, ti, ri, si)][1]
                                  * rings[ri].segments[si].length
                                  for si in range(len(rings[ri].segments))) * a
                self.edge_dist[(u, v)].Start = tot
                self.edge_dist[(v, u)].Start = tot
                self.edge_energy[(u, v)].Start = ef
                self.edge_energy[(v, u)].Start = eb
            else:
                du_ = _st(self.dfs[v]) - _st(self.dfs[u])
                self.edge_dist[(u, v)].Start = du_
                self.edge_dist[(v, u)].Start = du_
                if not self.wind_aware:
                    self.edge_energy[(u, v)].Start = du_
                    self.edge_energy[(v, u)].Start = du_
                else:
                    ef = eb = 0.0
                    for ti, ch in enumerate(r.chains):
                        info = self._ring_info.get((r.id, ti, ri))
                        if info is None:
                            continue
                        for si in range(info["num_seg"]):
                            cf, cb = self._seg_C[(r.id, ti, ri, si)]
                            L = info["seg_lens"][si]
                            iv = _st(self._I[(u, v, ti, si)])
                            mu_u = _st(self.mu[(r.id, u.idx, ti, si, ri)])
                            mu_v = _st(self.mu[(r.id, v.idx, ti, si, ri)])
                            et_u = _st(self._eta[(u, ti, si, ri)])
                            et_v = _st(self._eta[(v, ti, si, ri)])
                            ef += cf * L * (iv - et_u - (mu_v - et_v))
                            eb += cb * L * (iv - (mu_v - et_v) - et_u)
                    self.edge_energy[(u, v)].Start = ef
                    self.edge_energy[(v, u)].Start = eb
        for (a, b) in self._connectors:
            self.edge_dist[(a, b)].Start = 0.0
            self.edge_energy[(a, b)].Start = 0.0
        self.model.update()  # flush edge values before end/obj aux read them
        # --- endurance / objective Fortet aux: x * value (ALL of them, so the
        # start is 100% complete and the solver only verifies it) ---
        for (u, v, o), xv in self.x.items():
            xs = _st(xv)
            ve = _st(self.edge_energy[(u, v)])
            vd = _st(self.edge_dist[(u, v)])
            var = by_name.get(f"linbc_end_{u}_{v}_o{o}")
            if var is not None:
                var.Start = xs * ve
            var = by_name.get(f"linbc_obj_{u}_{v}_o{o}")
            if var is not None:
                var.Start = xs * vd
        # --- edge_dist/energy of unconstrained pairs: free vars, fix to 0 ---
        for u in self.all_nodes:
            for v in self.all_nodes:
                if u != v:
                    for d_ in (self.edge_dist[(u, v)], self.edge_energy[(u, v)]):
                        try:
                            s_ = d_.Start
                        except Exception:
                            s_ = None
                        if s_ is None or (isinstance(s_, float)
                                          and (s_ != s_ or abs(s_) >= 1e100)):
                            d_.Start = 0.0
        self.model.update()

    # ------------------------------------------------------------------
    # Warm start from heuristic solution
    # ------------------------------------------------------------------
    def set_warm_start(self, solution: Solution):
        chain_sel = solution.chain_selection
        dep_v = self.depot_v

        # alpha
        for (r_id, ti), var in self.alpha.items():
            var.Start = 1.0 if chain_sel.get(r_id) == ti else 0.0

        # Per-vertex variables
        for v in self.verts:
            ti = chain_sel.get(v.region_id)
            if ti is None:
                continue
            r = self.inst.regions[v.region_id]
            ri = self._ring_index[v]
            ch = r.chains[ti]
            rings = ch.rings if ch.rings else []
            if ri >= len(rings):
                continue
            ring = rings[ri]
            lam_val = solution.vertex_lambdas.get(v, 0.0)
            cum = [0.0]
            for seg in ring.segments:
                cum.append(cum[-1] + seg.length)
            total = cum[-1]
            if total <= 0:
                si, gamma_val = 0, 0.0
            else:
                lam_val = max(0.0, min(total, lam_val))
                si = next((s for s in range(len(ring.segments))
                           if cum[s] <= lam_val < cum[s + 1] or s == len(ring.segments) - 1), 0)
                seg_len = ring.segments[si].length
                gamma_val = 0.0 if seg_len <= 0 else (lam_val - cum[si]) / seg_len

            # mu
            ridx, idx = v.region_id, v.idx
            for ti2 in range(len(r.chains)):
                rings2 = r.chains[ti2].rings if r.chains[ti2].rings else []
                if ri < len(rings2):
                    ns2 = len(rings2[ri].segments)
                    for si2 in range(ns2):
                        mk = (ridx, idx, ti2, si2, ri)
                        if mk in self.mu:
                            self.mu[mk].Start = 1.0 if (ti2 == ti and si2 == si) else 0.0

            # gamma
            if v in self.gamma:
                self.gamma[v].Start = gamma_val

            # lambd
            self.lambd[v].Start = float(si) + gamma_val

            # P_x, P_y, P_z
            pos = solution.vertex_positions.get(v)
            if pos is not None:
                self.P_x[v].Start = pos.x
                self.P_y[v].Start = pos.y
                self.P_z[v].Start = pos.z

        # Operation-level variables: x, y, zeta, k, u
        ops = solution.operations

        for o in range(self.O):
            used = o < len(ops)
            self.zeta[o].Start = 0.0 if used else 1.0

            if used:
                edges = ops[o].edges
                verts_in_op = set()
                vert_order = []
                for u, v in edges:
                    if u != dep_v and u not in verts_in_op:
                        verts_in_op.add(u); vert_order.append(u)
                    if v != dep_v and v not in verts_in_op:
                        verts_in_op.add(v); vert_order.append(v)

                self.k[o].Start = len(verts_in_op)

                for u, v in edges:
                    xk = (u, v, o)
                    if xk in self.x:
                        self.x[xk].Start = 1.0

                for pos, vert in enumerate(vert_order, 1):
                    yk = (vert, o)
                    if yk in self.y:
                        self.y[yk].Start = 1.0
                    uk = (vert, o)
                    if uk in self._uvar:
                        self._uvar[uk].Start = pos
            else:
                self.k[o].Start = 0
                for v in self.verts:
                    xk_fwd = (dep_v, v, o)
                    if xk_fwd in self.x:
                        self.x[xk_fwd].Start = 0.0
                    xk_rev = (v, dep_v, o)
                    if xk_rev in self.x:
                        self.x[xk_rev].Start = 0.0
                    yk = (v, o)
                    if yk in self.y:
                        self.y[yk].Start = 0.0

        self.model.update()

    # ------------------------------------------------------------------
    # Optimize with first-incumbent callback
    # ------------------------------------------------------------------
    def optimize(self, tl=3600.):
        self.model.setParam("TimeLimit", tl)
        self.model.setParam("MIPFocus", DEFAULT_MIP_FOCUS)
        self.model.setParam("Heuristics", DEFAULT_HEURISTICS)

        first_info = [None, None]

        def _cb(model, where):
            if where == GRB.Callback.MIPSOL and first_info[0] is None:
                first_info[0] = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                first_info[1] = model.cbGet(GRB.Callback.RUNTIME)

        self.model.optimize(callback=_cb)
        st = self.model.Status
        if self.model.SolCount == 0:
            return None
        if st in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
            ops = []
            for o in range(self.O):
                ed = [(u, v) for u in self.all_nodes for v in self.all_nodes
                      if u != v and (u, v, o) in self.x and self.x[(u, v, o)].X > .5]
                if ed: ops.append(Operation(ed))
            vp = {v: Point3D(self.P_x[v].X, self.P_y[v].X, self.P_z[v].X)
                  for v in self.verts}
            vl = {v: self.lambd[v].X for v in self.verts}
            vr = {}
            for v in self.verts:
                ri = self._ring_index[v]
                for r in self.inst.regions:
                    if r.id == v.region_id:
                        for ti, ch in enumerate(r.chains):
                            if (r.id, ti) in self.alpha and self.alpha[(r.id, ti)].X > .5:
                                rings = ch.rings if ch.rings else []
                                if ri < len(rings):
                                    vr[v] = ri
                                break
            cs = {}
            for r in self.inst.regions:
                for ti in range(len(r.chains)):
                    if (r.id, ti) in self.alpha and self.alpha[(r.id, ti)].X > .5:
                        cs[r.id] = ti
            status_name = {2: "OPTIMAL", 3: "INFEASIBLE", 8: "TIME_LIMIT",
                           9: "SUBOPTIMAL", 11: "INTERRUPTED"}.get(st, str(st))
            return Solution(
                ops, self.model.ObjVal, vp, cs, vertex_lambdas=vl, vertex_rings=vr,
                solve_time=getattr(self.model, "Runtime", None),
                mip_gap=getattr(self.model, "MIPGap", None),
                status=status_name,
                first_incumbent_obj=first_info[0],
                first_incumbent_time=first_info[1],
            )
        return None
