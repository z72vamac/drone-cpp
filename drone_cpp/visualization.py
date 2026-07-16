from __future__ import annotations
from typing import List, Optional, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D

from .data_structures import (
    Point3D, Region, Instance, Solution, VertexType, EdgeType, Operation
)


class CPPVis:

    @staticmethod
    def lambd_to_cum(chain, lam: float) -> float:
        n = len(chain.segments)
        if lam <= 0: return 0.0
        if lam >= n: return chain.total_length
        si = int(lam)
        if si >= n: si = n - 1
        gamma = lam - si
        cum = sum(s.length for s in chain.segments[:si])
        cum += gamma * chain.segments[si].length
        return cum

    @staticmethod
    def compute_region_cum(chain, vertices, lambdas) -> dict:
        cums = {}
        for v in vertices:
            if v in lambdas:
                cums[v] = CPPVis.lambd_to_cum(chain, lambdas[v])
        return cums

    @staticmethod
    def get_chain_path(chain, cum1: float, cum2: float) -> list:
        if cum2 < cum1: cum1, cum2 = cum2, cum1
        result = []; current = 0.0
        for seg in chain.segments:
            seg_len = seg.length; seg_end = current + seg_len
            if seg_end > cum1 and current < cum2:
                t_start = max(0.0, (cum1 - current) / seg_len) if cum1 > current else 0.0
                t_end = min(1.0, (cum2 - current) / seg_len) if cum2 < seg_end else 1.0
                result.append((
                    seg.start.x + t_start * (seg.end.x - seg.start.x),
                    seg.start.y + t_start * (seg.end.y - seg.start.y),
                    seg.start.z + t_start * (seg.end.z - seg.start.z),
                    seg.start.x + t_end * (seg.end.x - seg.start.x),
                    seg.start.y + t_end * (seg.end.y - seg.start.y),
                    seg.start.z + t_end * (seg.end.z - seg.start.z),
                ))
            current = seg_end
            if current >= cum2: break
        return result

    @staticmethod
    def _draw_arrow_2d(ax, x1, y1, x2, y2, color, lw=1.5, alpha=0.85, label=None):
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.plot([x1, x2], [y1, y2], '-', color=color, lw=lw, alpha=alpha, label=label)
        sx = x1 * 0.65 + x2 * 0.35
        sy = y1 * 0.65 + y2 * 0.35
        ax.annotate('', xy=(mx, my), xytext=(sx, sy),
                    arrowprops=dict(arrowstyle='->', color=color, lw=0.6,
                                    alpha=alpha, shrinkA=0, shrinkB=0,
                                    mutation_scale=9))

    @staticmethod
    def is_rl_edge(u, v) -> bool:
        return (u.region_id >= 0 and u.region_id == v.region_id and
                abs(u.idx - v.idx) == 1 and
                ((u.idx % 2 == 0 and v.idx == u.idx + 1) or
                 (v.idx % 2 == 0 and u.idx == v.idx + 1)))

    @staticmethod
    def is_inter_edge(u, v) -> bool:
        return u.region_id != v.region_id and u.region_id >= 0 and v.region_id >= 0

    @staticmethod
    def _solution_setup(instance: Instance, solution: Solution):
        colors = plt.cm.Set1(np.linspace(0, 1, len(instance.regions)))
        op_colors = plt.cm.Dark2(np.linspace(0, 1, max(1, len(solution.operations))))
        sel_chain = {}
        for r in instance.regions:
            if r.id in solution.chain_selection:
                sel_chain[r.id] = r.chains[solution.chain_selection[r.id]]
        cum_map = {}
        for r_id, chain in sel_chain.items():
            verts_in_region = [v for v in solution.vertex_positions if v.region_id == r_id]
            cum_map.update(CPPVis.compute_region_cum(chain, verts_in_region,
                                                      solution.vertex_lambdas))
        def get_cum(v):
            if v in cum_map and cum_map[v] is not None:
                return cum_map[v]
            if v.region_id in sel_chain and v in solution.vertex_lambdas:
                return CPPVis.lambd_to_cum(sel_chain[v.region_id],
                                            solution.vertex_lambdas[v])
            return None
        return colors, op_colors, sel_chain, cum_map, get_cum

    @staticmethod
    def _draw_regions_2d(ax, instance: Instance, solution: Solution,
                         colors, sel_chain):
        for r_idx, r in enumerate(instance.regions):
            bpts = [[p.x, p.y] for p in r.boundary]
            bpts.append([r.boundary[0].x, r.boundary[0].y])
            pts = np.array(bpts)
            poly = MplPolygon(pts, fill=True, alpha=0.1, color=colors[r_idx],
                              label=f'Region {r.id}')
            ax.add_patch(poly)
            ax.plot(pts[:, 0], pts[:, 1], '-', color=colors[r_idx],
                    linewidth=1.0, alpha=0.5)
            centroid = pts.mean(axis=0)
            ax.annotate(f'R{r.id}', centroid, fontsize=10,
                        fontweight='bold', ha='center', va='center')
            for chain in r.chains:
                style = '-' if chain is sel_chain.get(r.id) else ':'
                lw = 1.0 if chain is sel_chain.get(r.id) else 0.5
                alpha = 0.4 if chain is sel_chain.get(r.id) else 0.15
                for seg in chain.segments:
                    ax.plot([seg.start.x, seg.end.x], [seg.start.y, seg.end.y],
                            style, color='gray', alpha=alpha, linewidth=lw)
        ax.plot(instance.depot.x, instance.depot.y, 'r^', markersize=8, label='Depot')
        for v, pos in solution.vertex_positions.items():
            ax.plot(pos.x, pos.y, 'k.', markersize=2, alpha=0.4)

    @staticmethod
    def _draw_regions_3d(ax, instance: Instance, solution: Solution,
                         colors, sel_chain):
        for r_idx, r in enumerate(instance.regions):
            bpts = np.array([[p.x, p.y] for p in r.boundary])
            poly_3d = np.hstack([bpts, np.zeros((len(bpts), 1))])
            ax.add_collection3d(Poly3DCollection(
                [poly_3d], alpha=0.08, color=colors[r_idx],
                edgecolor=colors[r_idx], linewidth=1.0))
            ax.plot(np.append(bpts[:, 0], bpts[0, 0]),
                    np.append(bpts[:, 1], bpts[0, 1]),
                    np.zeros(len(bpts) + 1), '-',
                    color=colors[r_idx], linewidth=1.0, alpha=0.3)
            for chain in r.chains:
                style = '-' if chain is sel_chain.get(r.id) else ':'
                lw = 1.0 if chain is sel_chain.get(r.id) else 0.5
                alpha = 0.4 if chain is sel_chain.get(r.id) else 0.15
                for seg in chain.segments:
                    ax.plot([seg.start.x, seg.end.x], [seg.start.y, seg.end.y],
                            [seg.start.z, seg.end.z],
                            style, color='gray', alpha=alpha, linewidth=lw)
        ax.plot([instance.depot.x], [instance.depot.y], [0], 'r^', markersize=8)
        for v, pos in solution.vertex_positions.items():
            ax.plot([pos.x], [pos.y], [pos.z], 'k.', markersize=2, alpha=0.4)

    @staticmethod
    def _draw_edges_2d(ax, instance: Instance, solution: Solution,
                       op_colors, sel_chain, get_cum):
        legend_added = set()
        for op_idx, op in enumerate(solution.operations):
            for (u, v) in op.edges:
                is_inter = CPPVis.is_inter_edge(u, v)
                is_rl = (not is_inter and CPPVis.is_rl_edge(u, v) and
                         u.region_id in sel_chain)
                label = f'Op {op_idx}' if op_idx not in legend_added else None
                if is_rl:
                    chain = sel_chain[u.region_id]
                    pu = solution.vertex_positions.get(u)
                    pv = solution.vertex_positions.get(v)
                    if pu is None or pv is None: continue
                    cu, cv = get_cum(u), get_cum(v)
                    if cu is not None and cv is not None:
                        forward = cv >= cu
                        path = CPPVis.get_chain_path(chain, cu, cv)
                        if not forward:
                            path = [(x2, y2, z2, x1, y1, z1)
                                    for (x1, y1, z1, x2, y2, z2) in reversed(path)]
                        for pi, (x1, y1, z1, x2, y2, z2) in enumerate(path):
                            seg_label = label if label and pi == 0 else None
                            CPPVis._draw_arrow_2d(ax, x1, y1, x2, y2,
                                                  op_colors[op_idx],
                                                  lw=1.5, alpha=0.85,
                                                  label=seg_label)
                            if seg_label: legend_added.add(op_idx)
                        if op_idx not in legend_added:
                            legend_added.add(op_idx)
                    elif pu is not None and pv is not None:
                        CPPVis._draw_arrow_2d(ax, pu.x, pu.y, pv.x, pv.y,
                                              op_colors[op_idx], lw=1.0, alpha=0.5)
                else:
                    pu = solution.vertex_positions.get(u)
                    pv = solution.vertex_positions.get(v)
                    if u == instance.depot_vertex or u.region_id == -1:
                        pu = instance.depot
                    if v == instance.depot_vertex or v.region_id == -1:
                        pv = instance.depot
                    if pu is None or pv is None: continue
                    CPPVis._draw_arrow_2d(ax, pu.x, pu.y, pv.x, pv.y,
                                          op_colors[op_idx],
                                          lw=1.5, alpha=0.85, label=label)
                    if label: legend_added.add(op_idx)

    @staticmethod
    def _draw_edges_3d(ax, instance: Instance, solution: Solution,
                       op_colors, sel_chain, get_cum):
        legend_added = set()
        for op_idx, op in enumerate(solution.operations):
            for (u, v) in op.edges:
                is_inter = CPPVis.is_inter_edge(u, v)
                is_rl = (not is_inter and CPPVis.is_rl_edge(u, v) and
                         u.region_id in sel_chain)
                label = f'Op {op_idx}' if op_idx not in legend_added else None
                if is_rl:
                    chain = sel_chain[u.region_id]
                    pu = solution.vertex_positions.get(u)
                    pv = solution.vertex_positions.get(v)
                    if pu is None or pv is None: continue
                    cu, cv = get_cum(u), get_cum(v)
                    if cu is not None and cv is not None:
                        forward = cv >= cu
                        path = CPPVis.get_chain_path(chain, cu, cv)
                        if not forward:
                            path = [(x2, y2, z2, x1, y1, z1)
                                    for (x1, y1, z1, x2, y2, z2) in reversed(path)]
                        for pi, (x1, y1, z1, x2, y2, z2) in enumerate(path):
                            seg_label = label if label and pi == 0 else None
                            ax.plot([x1, x2], [y1, y2], [z1, z2], '-',
                                    color=op_colors[op_idx], alpha=0.9,
                                    linewidth=2.5, label=seg_label)
                            if seg_label: legend_added.add(op_idx)
                        if op_idx not in legend_added:
                            legend_added.add(op_idx)
                    elif pu is not None and pv is not None:
                        ax.plot([pu.x, pv.x], [pu.y, pv.y], [pu.z, pv.z], ':',
                                color=op_colors[op_idx], alpha=0.5, linewidth=1.0)
                else:
                    pu = solution.vertex_positions.get(u)
                    pv = solution.vertex_positions.get(v)
                    if u == instance.depot_vertex or u.region_id == -1:
                        pu = instance.depot
                    if v == instance.depot_vertex or v.region_id == -1:
                        pv = instance.depot
                    if pu is None or pv is None: continue
                    ls = '--' if is_inter else '-'
                    ax.plot([pu.x, pv.x], [pu.y, pv.y], [pu.z, pv.z],
                            ls, color=op_colors[op_idx], alpha=0.9,
                            linewidth=2.5, label=label)
                    if label: legend_added.add(op_idx)

    @staticmethod
    def plot_instance(instance: Instance, show_chains: bool = True,
                      figsize: tuple = (10, 8)) -> plt.Figure:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.plot(instance.depot.x, instance.depot.y, 'r^', markersize=12, label='Depot')
        colors = plt.cm.Set1(np.linspace(0, 1, len(instance.regions)))
        for r_idx, r in enumerate(instance.regions):
            bpts = [[p.x, p.y] for p in r.boundary]
            bpts.append([r.boundary[0].x, r.boundary[0].y])
            pts = np.array(bpts)
            poly = MplPolygon(pts, fill=True, alpha=0.15, color=colors[r_idx],
                              label=f'Region {r.id}')
            ax.add_patch(poly)
            ax.plot(pts[:, 0], pts[:, 1], 'o-', color=colors[r_idx],
                    linewidth=2, markersize=4)
            centroid = pts.mean(axis=0)
            ax.annotate(f'R{r.id}', centroid, fontsize=10, fontweight='bold',
                        ha='center', va='center')
            if show_chains:
                for chain in r.chains:
                    for seg in chain.segments:
                        ax.plot([seg.start.x, seg.end.x], [seg.start.y, seg.end.y],
                                ':', color='gray', alpha=0.4, linewidth=0.8)
                    seg_mid = chain.segments[len(chain.segments) // 2].midpoint
                    ax.annotate(f'h={chain.height:.0f}m', (seg_mid.x, seg_mid.y),
                                fontsize=7, alpha=0.6, color='gray')
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
        ax.set_title('Drone CPP Instance')
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_solution_2d(instance: Instance, solution: Solution,
                         figsize: tuple = (10, 8),
                         title: Optional[str] = None) -> plt.Figure:
        fig, ax = plt.subplots(figsize=figsize)
        colors, op_colors, sel_chain, cum_map, get_cum = CPPVis._solution_setup(instance, solution)
        CPPVis._draw_regions_2d(ax, instance, solution, colors, sel_chain)
        CPPVis._draw_edges_2d(ax, instance, solution, op_colors, sel_chain, get_cum)
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
        ax.set_title(title or f'2D Solution (Objective: {solution.objective_value:.1f})')
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_solution_3d(instance: Instance, solution: Solution,
                         figsize: tuple = (10, 8),
                         title: Optional[str] = None,
                         elev: int = 35, azim: int = -50) -> plt.Figure:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        colors, op_colors, sel_chain, cum_map, get_cum = CPPVis._solution_setup(instance, solution)
        CPPVis._draw_regions_3d(ax, instance, solution, colors, sel_chain)
        CPPVis._draw_edges_3d(ax, instance, solution, op_colors, sel_chain, get_cum)
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
        ax.set_title(title or f'3D Solution (Objective: {solution.objective_value:.1f})')
        ax.legend(loc='upper left', fontsize=8)
        ax.view_init(elev=elev, azim=azim)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_solution(instance: Instance, solution: Solution,
                      figsize: tuple = (14, 6)) -> plt.Figure:
        fig = plt.figure(figsize=figsize)
        ax2d = fig.add_subplot(1, 2, 1)
        ax3d = fig.add_subplot(1, 2, 2, projection='3d')
        colors, op_colors, sel_chain, cum_map, get_cum = CPPVis._solution_setup(instance, solution)
        CPPVis._draw_regions_2d(ax2d, instance, solution, colors, sel_chain)
        CPPVis._draw_edges_2d(ax2d, instance, solution, op_colors, sel_chain, get_cum)
        ax2d.set_xlabel('X (m)'); ax2d.set_ylabel('Y (m)')
        ax2d.set_title(f'2D View (Objective: {solution.objective_value:.1f})')
        ax2d.set_aspect('equal'); ax2d.grid(True, alpha=0.3)
        ax2d.legend(loc='upper right', fontsize=8)
        CPPVis._draw_regions_3d(ax3d, instance, solution, colors, sel_chain)
        CPPVis._draw_edges_3d(ax3d, instance, solution, op_colors, sel_chain, get_cum)
        ax3d.set_xlabel('X (m)'); ax3d.set_ylabel('Y (m)'); ax3d.set_zlabel('Z (m)')
        ax3d.set_title('3D View')
        ax3d.view_init(elev=35, azim=-50)
        plt.tight_layout()
        return fig

    @staticmethod
    def show():
        plt.show()
