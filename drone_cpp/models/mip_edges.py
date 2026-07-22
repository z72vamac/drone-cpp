"""Edge-based MIQP formulation with DFJ lazy subtour elimination.

Replaces per-vertex visitation variables y_v^o with per-edge coverage
variables y_{uv}^o defined on intra-ring edges only.  Degree constraints
are relaxed to at-most-one (<= 1).  Instead of MTZ vertex-potential
constraints, subtour elimination is enforced via Dantzig–Fulkerson–Johnson
(DFJ) cut constraints separated lazily during branch-and-bound through a
MIPSOL callback that detects disconnected components per operation.
"""

from __future__ import annotations
from collections import defaultdict
from typing import Dict, Tuple

import gurobipy as gp
from gurobipy import GRB

from ..data_structures import (
    Solution, Operation, Point3D, Vertex, VertexType, AtmosphereParams,
)
from ..config import DEFAULT_MIP_FOCUS, DEFAULT_HEURISTICS
from .mip_rings import RingsModel


class EdgesModel(RingsModel):
    """Edge-based alternative: path variables on edges instead of vertices."""

    @property
    def name(self) -> str:
        return "Edges"

    def variable_summary(self) -> Dict[str, int]:
        n_bin = sum(1 for v in self.model.getVars() if v.VType == GRB.BINARY)
        n_cont = sum(1 for v in self.model.getVars() if v.VType == GRB.CONTINUOUS)
        n_int = sum(1 for v in self.model.getVars() if v.VType == GRB.INTEGER)
        return {"binary": n_bin, "continuous": n_cont, "integer": n_int,
                "constraints": self.model.NumConstrs}

    # ------------------------------------------------------------------
    # Variables: remove y_v^o, add y_{uv}^o for intra edges
    # ------------------------------------------------------------------
    def _create_variables(self):
        # --- mu, gamma, alpha, lambd (identical to RingsModel) ----------
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
                            vtype=GRB.BINARY,
                            name=f"mu_r{v.region_id}_v{v.idx}_t{ti}_s{si}_ri{ri}")

        for v in self.verts:
            self.gamma[v] = self.model.addVar(
                lb=0., ub=1., vtype=GRB.CONTINUOUS,
                name=f"gm_r{v.region_id}_v{v.idx}")

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
            self.P_x[v] = self.model.addVar(
                lb=bb[0], ub=bb[2], vtype=GRB.CONTINUOUS, name=f"Px_{v}")
            self.P_y[v] = self.model.addVar(
                lb=bb[1], ub=bb[3], vtype=GRB.CONTINUOUS, name=f"Py_{v}")
            self.P_z[v] = self.model.addVar(
                lb=0., ub=self._max_h(), vtype=GRB.CONTINUOUS, name=f"Pz_{v}")

        for r in self.inst.regions:
            self.rho_sel[r.id] = self.model.addVar(
                lb=0., ub=AtmosphereParams.air_density(0.),
                vtype=GRB.CONTINUOUS, name=f"ro_r{r.id}")

        # --- x variables (same edge set) --------------------------------
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

        # --- y_edge: coverage per intra edge (replaces y_v^o) ----------
        self.y_edge: Dict[Tuple[Vertex, Vertex, int], gp.Var] = {}
        for (u, v) in self._intra_rl:
            for o in range(self.O):
                self.y_edge[(u, v, o)] = self.model.addVar(
                    vtype=GRB.BINARY, name=f"ye_{u}_{v}_o{o}")

        # --- zeta, k ----------------------------------------------------
        for o in range(self.O):
            self.zeta[o] = self.model.addVar(vtype=GRB.BINARY, name=f"zt_o{o}")
            self.k[o] = self.model.addVar(
                lb=0, ub=len(self._intra_rl), vtype=GRB.INTEGER,
                name=f"k_o{o}")

        # --- edge_dist, edge_energy -------------------------------------
        md = self._max_d()
        mh = self._max_h()
        mc = self._max_chain_len()
        max_edge = md + mh + mc
        for u in self.all_nodes:
            for v in self.all_nodes:
                if u != v:
                    self.edge_dist[(u, v)] = self.model.addVar(
                        lb=0., ub=max_edge, vtype=GRB.CONTINUOUS,
                        name=f"dst_{u}_{v}")
                    self.edge_energy[(u, v)] = self.model.addVar(
                        lb=0., ub=GRB.INFINITY, vtype=GRB.CONTINUOUS,
                        name=f"enr_{u}_{v}")

        # --- eta (aux for mu * gamma) -----------------------------------
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

    # ------------------------------------------------------------------
    # Drone path constraints: degree ≤ 1, edge-coverage linking
    # ------------------------------------------------------------------
    def _add_drone_path_constraints(self):
        V, Vp, Or = self.all_nodes, self.verts, range(self.O)

        # DP1, DP3: unchanged
        for o in Or:
            self.model.addConstr(
                gp.quicksum(self.x[(self.depot_v, v, o)] for v in V if v != self.depot_v
                            if (self.depot_v, v, o) in self.x) == 1 - self.zeta[o],
                name=f"DP1_o{o}")
        for o in Or:
            self.model.addConstr(
                gp.quicksum(self.x[(v, self.depot_v, o)] for v in Vp
                            if (v, self.depot_v, o) in self.x) == 1 - self.zeta[o],
                name=f"DP3_o{o}")

        # DP2: flow conservation (unchanged)
        for v in Vp:
            for o in Or:
                inn = gp.quicksum(self.x[(u, v, o)] for u in V if u != v
                                  if (u, v, o) in self.x)
                out = gp.quicksum(self.x[(v, u, o)] for u in V if u != v
                                  if (v, u, o) in self.x)
                self.model.addConstr(inn == out, name=f"DP2_{v}_o{o}")

        # DP4', DP5': at-most-one traversal per vertex per operation
        # (DP4' out ≤ 1 is implied by DP2 + DP5' for non-depot vertices,
        #  but kept explicitly to provide the solver with tighter LP bounds.)
        for v in V:
            for o in Or:
                out = gp.quicksum(self.x[(v, u, o)] for u in V if u != v
                                  if (v, u, o) in self.x)
                self.model.addConstr(out <= 1, name=f"DP4p_{v}_o{o}")
                inn = gp.quicksum(self.x[(u, v, o)] for u in V if u != v
                                  if (u, v, o) in self.x)
                self.model.addConstr(inn <= 1, name=f"DP5p_{v}_o{o}")

        # DP7 removed (MTZ replaced by DFJ lazy cuts in optimize()) ————
        # DFJ cut separation for subtour elimination is performed via
        # a lazy-constraint callback in self.optimize().
        # ———————————————————————————————————————————————————

        # DP9: idle operations → no edges (tight big-M)
        # M = |V'| + 1 (at most one per vertex + depot outgoing)
        M_tight = len(self.verts) + 1
        for o in Or:
            self.model.addConstr(
                gp.quicksum(self.x[k] for k in self.x if k[2] == o)
                <= M_tight * (1 - self.zeta[o]),
                name=f"DP9_o{o}")

        # DP8 is omitted — implied by EC1 + EC2a + EC3:
        #   EC1: Σ_o y = 1,  EC2a: y ≥ x,  EC3: y ≤ x  ⇒  y = x  ⇒  Σ_o x = 1

        # EC1: every intra edge is covered by exactly one operation
        for (u, v) in self._intra_rl:
            self.model.addConstr(
                gp.quicksum(self.y_edge[(u, v, o)] for o in Or) == 1,
                name=f"EC1_{u}_{v}")

        # EC2a: coverage implies forward traversal
        # EC3:  coverage requires forward traversal
        # (Intra edges are directed launch→retrieve, so the reverse direction
        #  never exists in self.x.  EC2a + EC3 give y = x directly.)
        for (u, v) in self._intra_rl:
            for o in Or:
                kf = (u, v, o)
                self.model.addConstr(
                    self.y_edge[(u, v, o)] >= self.x[kf],
                    name=f"EC2a_{u}_{v}_o{o}")
                self.model.addConstr(
                    self.y_edge[(u, v, o)] <= self.x[kf],
                    name=f"EC3_{u}_{v}_o{o}")

    # ------------------------------------------------------------------
    # Valid inequalities: k counts edges, VI-1 uses |E_int|
    # ------------------------------------------------------------------
    def _add_valid_inequalities(self):
        Or = range(self.O)
        n_edges = len(self._intra_rl)

        # Monotonicity (unchanged)
        for o in range(self.O - 1):
            self.model.addConstr(self.zeta[o] <= self.zeta[o + 1],
                                 name=f"Mon_o{o}")

        # kC: k^o = number of intra edges covered in operation o
        for o in Or:
            self.model.addConstr(
                self.k[o] == gp.quicksum(
                    self.y_edge[(u, v, o)] for (u, v) in self._intra_rl),
                name=f"kC_o{o}")

        # VI-1: cumulative coverage must equal |E_int| before zeta activates
        for o in Or:
            self.model.addConstr(
                gp.quicksum(self.k[o2] for o2 in range(o))
                >= n_edges * self.zeta[o],
                name=f"VI1_o{o}")

        # VI-2 (unchanged)
        for o in Or:
            self.model.addConstr(
                self.k[o] >= 1 - self.zeta[o],
                name=f"VI2_o{o}")

    # ------------------------------------------------------------------
    # Warm start
    # ------------------------------------------------------------------
    def set_warm_start(self, solution: Solution):
        chain_sel = solution.chain_selection
        dep_v = self.depot_v

        # alpha (identical to RingsModel)
        for (r_id, ti), var in self.alpha.items():
            var.Start = 1.0 if chain_sel.get(r_id) == ti else 0.0

        # Per-vertex variables (identical to RingsModel)
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
                           if cum[s] <= lam_val < cum[s + 1]
                           or s == len(ring.segments) - 1), 0)
                seg_len = ring.segments[si].length
                gamma_val = 0.0 if seg_len <= 0 else (lam_val - cum[si]) / seg_len

            ridx, idx = v.region_id, v.idx
            for ti2 in range(len(r.chains)):
                rings2 = r.chains[ti2].rings if r.chains[ti2].rings else []
                if ri < len(rings2):
                    ns2 = len(rings2[ri].segments)
                    for si2 in range(ns2):
                        mk = (ridx, idx, ti2, si2, ri)
                        if mk in self.mu:
                            self.mu[mk].Start = 1.0 if (ti2 == ti and si2 == si) else 0.0

            if v in self.gamma:
                self.gamma[v].Start = gamma_val
            self.lambd[v].Start = float(si) + gamma_val

            pos = solution.vertex_positions.get(v)
            if pos is not None:
                self.P_x[v].Start = pos.x
                self.P_y[v].Start = pos.y
                self.P_z[v].Start = pos.z

        # Operation-level variables
        ops = solution.operations

        for o in range(self.O):
            used = o < len(ops)
            self.zeta[o].Start = 0.0 if used else 1.0

            if used:
                edges = ops[o].edges

                # Count covered intra edges and set y_edge starts
                edge_count = 0
                for (u, v) in self._intra_rl:
                    covered = any(
                        (eu == u and ev == v) or (eu == v and ev == u)
                        for (eu, ev) in edges
                    )
                    for oo in range(self.O):
                        yk = (u, v, oo)
                        if yk in self.y_edge:
                            self.y_edge[yk].Start = 1.0 if (covered and oo == o) else 0.0
                    if covered:
                        edge_count += 1

                self.k[o].Start = edge_count

                for u, v in edges:
                    xk = (u, v, o)
                    if xk in self.x:
                        self.x[xk].Start = 1.0
            else:
                self.k[o].Start = 0
                for (u, v) in self._intra_rl:
                    yk = (u, v, o)
                    if yk in self.y_edge:
                        self.y_edge[yk].Start = 0.0
                for v in self.verts:
                    xk_fwd = (dep_v, v, o)
                    if xk_fwd in self.x:
                        self.x[xk_fwd].Start = 0.0
                    xk_rev = (v, dep_v, o)
                    if xk_rev in self.x:
                        self.x[xk_rev].Start = 0.0

        self.model.update()

    # ------------------------------------------------------------------
    # Optimize — DFJ lazy subtour elimination via MIPSOL callback
    # ------------------------------------------------------------------
    def optimize(self, tl=3600.):
        self.model.setParam("TimeLimit", tl)
        self.model.setParam("MIPFocus", DEFAULT_MIP_FOCUS)
        self.model.setParam("Heuristics", DEFAULT_HEURISTICS)
        self.model.setParam("LazyConstraints", 1)

        first_info = [None, None]
        all_nodes = self.all_nodes
        verts = self.verts
        depot = self.depot_v
        x_dict = self.x
        zeta_dict = self.zeta

        def _separate_subtours(model, where):
            if where == GRB.Callback.MIPSOL:
                # Track first incumbent
                if first_info[0] is None:
                    first_info[0] = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    first_info[1] = model.cbGet(GRB.Callback.RUNTIME)

                # DFJ cut separation per active operation
                for o in range(self.O):
                    zeta_val = model.cbGetSolution(zeta_dict[o])
                    if zeta_val > 0.5:
                        continue  # idle (DP9 forces all x=0)

                    # Build directed adjacency for this operation
                    adj = defaultdict(list)
                    for (u, v, o2), var in x_dict.items():
                        if o2 == o and model.cbGetSolution(var) > 0.5:
                            adj[u].append(v)

                    # BFS from depot to find reachable vertices
                    reached = {depot}
                    stack = [depot]
                    while stack:
                        n = stack.pop()
                        for nb in adj.get(n, []):
                            if nb not in reached:
                                reached.add(nb)
                                stack.append(nb)

                    # Find subtour components (verts not reached from depot)
                    unvisited = [v for v in verts if v not in reached]
                    while unvisited:
                        start = unvisited[0]
                        component = {start}
                        q = [start]
                        while q:
                            n = q.pop()
                            for nb in adj.get(n, []):
                                if nb in all_nodes and nb not in component and nb not in reached:
                                    component.add(nb)
                                    q.append(nb)
                        unvisited = [v for v in unvisited if v not in component]

                        # Build DFJ cut: at least one edge must leave S
                        cut = gp.LinExpr()
                        for u in component:
                            for v in all_nodes:
                                if v not in component and (u, v, o) in x_dict:
                                    cut += x_dict[(u, v, o)]
                        model.cbLazy(cut >= 1)

        self.model.optimize(callback=_separate_subtours)
        st = self.model.Status
        if self.model.SolCount == 0:
            return None
        if st in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
            ops = []
            for o in range(self.O):
                ed = [(u, v) for u in self.all_nodes for v in self.all_nodes
                      if u != v and (u, v, o) in self.x and self.x[(u, v, o)].X > .5]
                if ed:
                    ops.append(Operation(ed))
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
