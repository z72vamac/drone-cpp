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
    """Each ring of the selected chain is an independent traversal."""

    def __init__(self, instance: Instance, verbose: bool = True):
        super().__init__(instance)
        self.inst = instance
        self.R = instance.num_regions
        self.O = instance.num_operations
        self.depot_v = instance.depot_vertex

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

        self.verts: List[Vertex] = []
        self._ring_rl: List[Tuple[Vertex, Vertex]] = []
        self._ring_index: Dict[Vertex, int] = {}
        for r in instance.regions:
            for ri in range(self._max_rings_by_region[r.id]):
                v_e = Vertex(r.id, ri * 2, VertexType.LAUNCH)
                v_x = Vertex(r.id, ri * 2 + 1, VertexType.RETRIEVE)
                self.verts.extend([v_e, v_x])
                self._ring_rl.append((v_e, v_x))
                self._ring_index[v_e] = ri
                self._ring_index[v_x] = ri

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

        self._eta = {}
        self._lin_vars = []

        self._create_variables()
        self._add_location_constraints()
        self._add_drone_path_constraints()
        self._add_valid_inequalities()
        self._add_intra_ring_distance_energy()
        self._add_inter_distance_energy()
        self._add_depot_distance_energy()
        self._add_endurance_constraints()
        self._set_objective()
        self.model.update()

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

        for (u, v) in self._intra_rl:
            self.model.addConstr(self.lambd[u] == self.lambd[v], name=f"LC5_{u}_{v}")

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

    def _add_intra_ring_distance_energy(self):
        max_cl = self._max_chain_len()
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
            self.model.addConstr(self.edge_energy[(u, v)] == du + dv, name=f"ienr_{u}_{v}")

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
