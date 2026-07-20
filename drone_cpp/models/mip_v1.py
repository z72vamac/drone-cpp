from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np
import gurobipy as gp
from gurobipy import GRB

from ..data_structures import (
    Point3D, Segment, PolygonalChain, Region, Vertex,
    VertexType, Edge, EdgeType, AtmosphereParams, Instance, Operation, Solution
)
from ..config import DEFAULT_MIP_FOCUS, DEFAULT_HEURISTICS
from .base import BaseModel


class V1Model(BaseModel):
    def __init__(self, instance: Instance, verbose: bool = True):
        super().__init__(instance)
        self.inst = instance
        self.R = instance.num_regions
        self.O = instance.num_operations
        self.verts = instance.all_vertices
        self.depot_v = instance.depot_vertex
        self.all_nodes = [self.depot_v] + self.verts

        self._precompute()

        self._intra_rl: List[Tuple[Vertex, Vertex]] = []
        self._intra_lr: List[Tuple[Vertex, Vertex]] = []
        self._inter_edges: List[Tuple[Vertex, Vertex, int, int]] = []
        self._classify_edges()

        self.model = gp.Model("DroneCPP")
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

        self.d_xy = {}
        self.d_z = {}
        self.edge_dist = {}
        self.edge_energy = {}

        self.phi = {}
        self.psi = {}

        self._eta = {}
        self._lin_vars = []

        self._create_variables()
        self._add_location_constraints()
        self._add_drone_path_constraints()
        self._add_valid_inequalities()
        self._add_intra_distance_energy()
        self._add_inter_distance_energy()
        self._add_depot_distance_energy()
        self._add_endurance_constraints()
        self._set_objective()
        self.model.update()

    @property
    def name(self) -> str:
        return "V1"

    def variable_summary(self) -> Dict[str, int]:
        n_bin = sum(1 for v in self.model.getVars() if v.VType == GRB.BINARY)
        n_cont = sum(1 for v in self.model.getVars() if v.VType == GRB.CONTINUOUS)
        n_int = sum(1 for v in self.model.getVars() if v.VType == GRB.INTEGER)
        return {"binary": n_bin, "continuous": n_cont, "integer": n_int,
                "constraints": self.model.NumConstrs}

    def _precompute(self):
        self._cinfo = {}
        for r in self.inst.regions:
            for t_idx, chain in enumerate(r.chains):
                key = (r.id, t_idx)
                seg_lens = chain.segment_lengths()
                cum_before = chain.cumulative_lengths_before_segment()
                dens = AtmosphereParams.air_density(chain.height)
                wind_sp = self.inst.wind.speed_at_height(chain.height)
                wind_dir = self.inst.wind.direction
                nu_fwd = []
                for s_idx in range(len(seg_lens)):
                    nu_fwd.append(chain.compute_drone_speed_on_segment(
                        s_idx, True, wind_sp, wind_dir, self.inst.drone.cruise_speed))
                self._cinfo[key] = {
                    "seg_lens": seg_lens, "cum_before": cum_before,
                    "density": dens, "nu_d_fwd": nu_fwd,
                    "height": chain.height, "num_seg": len(seg_lens),
                }

    def _classify_edges(self):
        for u in self.verts:
            for v in self.verts:
                if u.region_id == v.region_id and abs(u.idx - v.idx) == 1:
                    if u.idx % 2 == 1 and v.idx == u.idx + 1:
                        self._intra_lr.append((u, v))
                    elif u.idx % 2 == 0 and v.idx == u.idx + 1:
                        self._intra_rl.append((u, v))
                elif u.region_id != v.region_id:
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
        E = self.inst.drone
        for v in self.verts:
            r = self.inst.regions[v.region_id]
            for ti, ch in enumerate(r.chains):
                for si in range(len(ch.segments)):
                    k = (v.region_id, v.idx, ti, si)
                    self.mu[k] = self.model.addVar(
                        vtype=GRB.BINARY, name=f"mu_r{v.region_id}_v{v.idx}_t{ti}_s{si}")

        for v in self.verts:
            self.gamma[v] = self.model.addVar(
                lb=0., ub=1., vtype=GRB.CONTINUOUS, name=f"gm_r{v.region_id}_v{v.idx}")

        for r in self.inst.regions:
            for ti in range(len(r.chains)):
                self.alpha[(r.id, ti)] = self.model.addVar(
                    vtype=GRB.BINARY, name=f"al_r{r.id}_t{ti}")

        for v in self.verts:
            mx = max((len(c.segments) for c in self.inst.regions[v.region_id].chains), default=0)
            self.lambd[v] = self.model.addVar(
                lb=0., ub=float(mx), vtype=GRB.CONTINUOUS, name=f"la_r{v.region_id}_v{v.idx}")

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
        for (u, v) in self._intra_lr:
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
        for u in self.verts:
            for v in self.verts:
                if u.region_id != v.region_id:
                    self.d_xy[(u, v)] = self.model.addVar(
                        lb=0., ub=md, vtype=GRB.CONTINUOUS, name=f"dxy_{u}_{v}")
                    self.d_z[(u, v)] = self.model.addVar(
                        lb=0., ub=mh, vtype=GRB.CONTINUOUS, name=f"dz_{u}_{v}")

        max_edge = md + mh + mc
        for u in self.all_nodes:
            for v in self.all_nodes:
                if u != v:
                    self.edge_dist[(u, v)] = self.model.addVar(
                        lb=0., ub=max_edge, vtype=GRB.CONTINUOUS, name=f"dst_{u}_{v}")
                    self.edge_energy[(u, v)] = self.model.addVar(
                        lb=0., ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"enr_{u}_{v}")

        for r1 in self.inst.regions:
            for r2 in self.inst.regions:
                if r1.id != r2.id:
                    self.phi[(r1.id, r2.id)] = self.model.addVar(
                        vtype=GRB.BINARY, name=f"ph_r{r1.id}_r{r2.id}")
                    self.psi[(r1.id, r2.id)] = self.model.addVar(
                        vtype=GRB.BINARY, name=f"ps_r{r1.id}_r{r2.id}")

        for v in self.verts:
            r = self.inst.regions[v.region_id]
            for ti in range(len(r.chains)):
                for si in range(len(r.chains[ti].segments)):
                    mk = (v.region_id, v.idx, ti, si)
                    if mk in self.mu:
                        self._eta[(v, ti, si)] = self._lin_bcs(
                            self.mu[mk], self.gamma[v],
                            f"eta_r{v.region_id}_v{v.idx}_t{ti}_s{si}")

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

    def _max_h(self): return max((c.height for r in self.inst.regions for c in r.chains), default=0) + 10.

    def _max_d(self):
        b = self._get_bounds()
        return float(np.sqrt((b[2] - b[0])**2 + (b[3] - b[1])**2))

    def _max_chain_len(self):
        return max((c.total_length for r in self.inst.regions for c in r.chains), default=self._max_d())

    def _add_location_constraints(self):
        for v in self.verts:
            r = self.inst.regions[v.region_id]
            ex, ey, ez = gp.LinExpr(), gp.LinExpr(), gp.LinExpr()
            for ti, ch in enumerate(r.chains):
                for si, seg in enumerate(ch.segments):
                    mk = (v.region_id, v.idx, ti, si)
                    if mk not in self.mu: continue
                    mu = self.mu[mk]; et = self._eta[(v, ti, si)]
                    ex += mu * seg.start.x + et * (seg.end.x - seg.start.x)
                    ey += mu * seg.start.y + et * (seg.end.y - seg.start.y)
                    ez += mu * ch.height
            self.model.addConstr(self.P_x[v] == ex, name=f"LC1x_{v}")
            self.model.addConstr(self.P_y[v] == ey, name=f"LC1y_{v}")
            self.model.addConstr(self.P_z[v] == ez, name=f"LC1z_{v}")

        for v in self.verts:
            r = self.inst.regions[v.region_id]
            for ti in range(len(r.chains)):
                sm = gp.LinExpr()
                for si in range(len(r.chains[ti].segments)):
                    mk = (v.region_id, v.idx, ti, si)
                    if mk in self.mu: sm += self.mu[mk]
                self.model.addConstr(sm == self.alpha[(r.id, ti)], name=f"LC2_{v}_t{ti}")

        for r in self.inst.regions:
            self.model.addConstr(
                gp.quicksum(self.alpha[(r.id, ti)] for ti in range(len(r.chains))) == 1,
                name=f"LC3_r{r.id}")

        for v in self.verts:
            r = self.inst.regions[v.region_id]
            e = gp.LinExpr()
            for ti, ch in enumerate(r.chains):
                for si in range(len(ch.segments)):
                    mk = (v.region_id, v.idx, ti, si)
                    if mk in self.mu: e += si * self.mu[mk]
            e += self.gamma[v]
            self.model.addConstr(self.lambd[v] == e, name=f"LC4_{v}")

        for v in self.verts:
            if v.vtype == VertexType.START:
                self.model.addConstr(self.lambd[v] == 0, name=f"LC5_{v}")
                r = self.inst.regions[v.region_id]
                for ti in range(len(r.chains)):
                    mk = (v.region_id, v.idx, ti, 0)
                    self.model.addConstr(
                        self.mu[mk] == self.alpha[(r.id, ti)],
                        name=f"LC5b_{v}_t{ti}")

        for (u, v) in self._intra_rl:
            self.model.addConstr(self.lambd[u] <= self.lambd[v], name=f"LC6_{u}_{v}")

        for (u, v) in self._intra_lr:
            self.model.addConstr(self.lambd[u] == self.lambd[v], name=f"LC7_{u}_{v}")

        for v in self.verts:
            if v.vtype == VertexType.END:
                r = self.inst.regions[v.region_id]
                e = gp.LinExpr()
                for ti in range(len(r.chains)):
                    e += len(r.chains[ti].segments) * self.alpha[(r.id, ti)]
                self.model.addConstr(self.lambd[v] == e, name=f"LC8_{v}")
                r = self.inst.regions[v.region_id]
                for ti, ch in enumerate(r.chains):
                    mk = (v.region_id, v.idx, ti, len(ch.segments) - 1)
                    self.model.addConstr(
                        self.mu[mk] == self.alpha[(r.id, ti)],
                        name=f"LC8b_{v}_t{ti}")

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
                                <= (nV - 1) * (1 - self.x[k]), name=f"DP7_{v1}_{v2}_o{o}")

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

    def _add_intra_distance_energy(self):
        E = self.inst.drone
        maxcl = self._max_chain_len()
        dfs = {}; efs = {}
        for v in self.verts:
            dfs[v] = self.model.addVar(lb=0., ub=maxcl, vtype=GRB.CONTINUOUS,
                                       name=f"dfs_{v}")
            efs[v] = self.model.addVar(lb=0., ub=maxcl * 10, vtype=GRB.CONTINUOUS,
                                       name=f"efs_{v}")
            r = self.inst.regions[v.region_id]
            de, ee = gp.LinExpr(), gp.LinExpr()
            for ti, ch in enumerate(r.chains):
                ci = self._cinfo[(r.id, ti)]
                for si in range(len(ch.segments)):
                    mk = (v.region_id, v.idx, ti, si)
                    if mk not in self.mu: continue
                    de += ci["cum_before"][si] * self.mu[mk]
                    de += ci["seg_lens"][si] * self._eta[(v, ti, si)]
                    e_per_meter = E.E_xy * ci["density"] * ci["nu_d_fwd"][si]
                    ee += e_per_meter * ci["cum_before"][si] * self.mu[mk]
                    ee += e_per_meter * ci["seg_lens"][si] * self._eta[(v, ti, si)]
            self.model.addConstr(dfs[v] == de, name=f"dfs_{v}")
            self.model.addConstr(efs[v] == ee, name=f"efs_{v}")

        for (u, v) in self._intra_rl:
            self.model.addConstr(self.edge_dist[(u, v)] == dfs[v] - dfs[u],
                                 name=f"idst_{u}_{v}")
            self.model.addConstr(self.edge_dist[(v, u)] == self.edge_dist[(u, v)],
                                 name=f"idst_s_{u}_{v}")
            self.model.addConstr(self.edge_energy[(u, v)] == efs[v] - efs[u],
                                 name=f"ienr_{u}_{v}")
            self.model.addConstr(self.edge_energy[(v, u)] == self.edge_energy[(u, v)],
                                 name=f"ienr_s_{u}_{v}")

        for (u, v) in self._intra_lr:
            self.model.addConstr(self.edge_dist[(u, v)] == 0, name=f"idst0_{u}_{v}")
            self.model.addConstr(self.edge_dist[(v, u)] == 0, name=f"idst0s_{u}_{v}")
            self.model.addConstr(self.edge_energy[(u, v)] == 0, name=f"ienr0_{u}_{v}")
            self.model.addConstr(self.edge_energy[(v, u)] == 0, name=f"ienr0s_{u}_{v}")

    def _add_inter_distance_energy(self):
        E = self.inst.drone
        md = self._max_d(); mh = self._max_h()
        for (u, v, r1, r2) in self._inter_edges:
            du = self.d_xy.get((u, v)); dv = self.d_z.get((u, v))
            if du is None: continue
            dx = self.model.addVar(lb=-GRB.INFINITY, name=f"idx_{u}_{v}")
            dy = self.model.addVar(lb=-GRB.INFINITY, name=f"idy_{u}_{v}")
            self.model.addConstr(dx == self.P_x[u] - self.P_x[v], name=f"dx_{u}_{v}")
            self.model.addConstr(dy == self.P_y[u] - self.P_y[v], name=f"dy_{u}_{v}")
            self.model.addConstr(du * du >= dx * dx + dy * dy, name=f"dxy_{u}_{v}")
            self.model.addConstr(dv >= self.P_z[u] - self.P_z[v], name=f"dz1_{u}_{v}")
            self.model.addConstr(dv >= self.P_z[v] - self.P_z[u], name=f"dz2_{u}_{v}")
            self.model.addConstr(self.edge_dist[(u, v)] == du + dv, name=f"idst_{u}_{v}")
            self._add_inter_energy(u, v, r1, r2, du, dv, E.E_xy, E.E_z)

    def _add_inter_energy(self, u, v, r1, r2, dxy, dz, Exy, Ez):
        ph = self.phi[(r1, r2)]; ps = self.psi[(r1, r2)]
        nd = self._get_inter_speed_dir(r1, r2)
        n1 = self._inter_speed(r1, nd); n2 = self._inter_speed(r2, nd)
        Bx1 = Exy * n1; Bx2 = Exy * n2; Bz = 0.5 * Ez * self.inst.drone.vertical_speed

        lpd = self._lin_bc(ph, dxy, 0., md := self._max_d(), f"phd_{u}_{v}")
        th1 = gp.LinExpr()
        for ti, ch in enumerate(self.inst.regions[r1].chains):
            d = AtmosphereParams.air_density(ch.height)
            a = self.alpha[(r1, ti)]
            la = self._lin_bc(a, lpd, 0., md, f"a{ti}phd_{u}_{v}", M=md)
            th1 += d * la
        t1h = Bx1 * th1

        lpp = self._lin_bb(ph, ps, f"pps_{u}_{v}")
        mh = self._max_h()
        lpdz = self._lin_bc(lpp, dz, 0., mh, f"ppdz_{u}_{v}", M=mh)
        tv1 = gp.LinExpr()
        for r_id in (r1, r2):
            for ti, ch in enumerate(self.inst.regions[r_id].chains):
                d = AtmosphereParams.air_density(ch.height)
                a = self.alpha[(r_id, ti)]
                la = self._lin_bc(a, lpdz, 0., mh, f"a{ti}ppdz_{r_id}_{u}_{v}", M=mh)
                tv1 += d * la
        t1v = Bz * tv1

        omph = 1. - ph
        lmh = self._lin_bc(omph, dxy, 0., md, f"1phd_{u}_{v}")
        th2 = gp.LinExpr()
        for ti, ch in enumerate(self.inst.regions[r2].chains):
            d = AtmosphereParams.air_density(ch.height)
            a = self.alpha[(r2, ti)]
            la = self._lin_bc(a, lmh, 0., md, f"a{ti}1phd_{u}_{v}", M=md)
            th2 += d * la
        t2h = Bx2 * th2

        omps = 1. - ps
        l1m = self._lin_bb(omph, omps, f"1p1s_{u}_{v}")
        lmz = self._lin_bc(l1m, dz, 0., mh, f"1p1sdz_{u}_{v}", M=mh)
        tv2 = gp.LinExpr()
        for r_id in (r1, r2):
            for ti, ch in enumerate(self.inst.regions[r_id].chains):
                d = AtmosphereParams.air_density(ch.height)
                a = self.alpha[(r_id, ti)]
                la = self._lin_bc(a, lmz, 0., mh, f"a{ti}1p1s_{r_id}_{u}_{v}", M=mh)
                tv2 += d * la
        t2v = Bz * tv2

        self.model.addConstr(
            self.edge_energy[(u, v)] == t1h + t1v + t2h + t2v, name=f"we_{u}_{v}")

    def _get_inter_speed_dir(self, r1, r2):
        c1 = np.mean([(p.x, p.y) for p in self.inst.regions[r1].boundary], axis=0)
        c2 = np.mean([(p.x, p.y) for p in self.inst.regions[r2].boundary], axis=0)
        v = np.array([c2[0] - c1[0], c2[1] - c1[1], 0.])
        n = np.linalg.norm(v)
        return v / n if n else np.array([1., 0., 0.])

    def _inter_speed(self, rid, direc):
        r = self.inst.regions[rid]
        ch = r.chains[0]
        ws = self.inst.wind.speed_at_height(ch.height)
        wd = self.inst.wind.direction
        return float(np.linalg.norm(self.inst.drone.cruise_speed * direc - ws * wd))

    def _add_depot_distance_energy(self):
        E = self.inst.drone
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
            En_xy = E.E_xy * rho0 * nu_d_xy
            En_z = 0.5 * E.E_z * self.inst.drone.vertical_speed * rho0
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

    def optimize(self, tl=3600.):
        self.model.setParam("TimeLimit", tl)
        self.model.setParam("MIPFocus", DEFAULT_MIP_FOCUS)
        self.model.setParam("Heuristics", DEFAULT_HEURISTICS)
        self.model.optimize()
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
            cs = {}
            for r in self.inst.regions:
                for ti in range(len(r.chains)):
                    if (r.id, ti) in self.alpha and self.alpha[(r.id, ti)].X > .5:
                        cs[r.id] = ti
            status_name = {2: "OPTIMAL", 3: "INFEASIBLE", 8: "TIME_LIMIT",
                           9: "SUBOPTIMAL", 11: "INTERRUPTED"}.get(st, str(st))
            return Solution(
                ops, self.model.ObjVal, vp, cs, vertex_lambdas=vl,
                solve_time=getattr(self.model, "Runtime", None),
                mip_gap=getattr(self.model, "MIPGap", None),
                status=status_name,
            )
        return None
