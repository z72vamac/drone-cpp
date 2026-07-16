from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
from .data_structures import (
    Point3D, Segment, PolygonalChain, Region, Vertex,
    VertexType, Edge, EdgeType, DroneParams, WindParams,
    AtmosphereParams, Instance
)


class InstanceGenerator:

    @staticmethod
    def generate(num_regions: int = 3,
                 area_bounds: Tuple[float, float, float, float] = (0, 0, 100, 100),
                 min_region_size: float = 10.0,
                 max_region_size: float = 30.0,
                 num_heights: int = 1,
                 min_height: float = 20.0,
                 max_height: float = 60.0,
                 spacing: float = 5.0,
                 num_ops: int = 2,
                 seed: Optional[int] = None) -> Instance:
        rng = np.random.RandomState(seed)

        depot = Point3D(
            rng.uniform(area_bounds[0], area_bounds[2]),
            rng.uniform(area_bounds[1], area_bounds[3]),
            0.0
        )

        regions = []
        for r_id in range(num_regions):
            placed = False
            for _ in range(500):
                region = InstanceGenerator._generate_region(
                    r_id, rng, area_bounds, min_region_size, max_region_size,
                    num_heights, min_height, max_height, spacing
                )
                if not InstanceGenerator._overlaps(region, regions, margin=10.0):
                    placed = True
                    break
            if not placed:
                raise RuntimeError(f"No se pudo colocar región {r_id} sin solapamiento")
            regions.append(region)

        drone = DroneParams(
            front_area=0.1,
            drag_coef=0.3,
            max_endurance=3000.0,
            cruise_speed=15.0,
            vertical_speed=5.0
        )

        angle = rng.uniform(0, 2 * np.pi)
        wind = WindParams(
            direction=np.array([np.cos(angle), np.sin(angle), 0.0]),
            speed_at_10m=rng.uniform(2.0, 8.0),
            hellmann_exponent=0.2
        )

        return Instance(
            regions=regions,
            depot=depot,
            drone=drone,
            wind=wind,
            num_operations=num_ops
        )

    @staticmethod
    def _generate_region(r_id: int, rng: np.random.RandomState,
                         bounds: Tuple[float, float, float, float],
                         min_size: float, max_size: float,
                         num_heights: int, min_h: float, max_h: float,
                         spacing: float) -> Region:
        cx = rng.uniform(bounds[0] + max_size, bounds[2] - max_size)
        cy = rng.uniform(bounds[1] + max_size, bounds[3] - max_size)

        n_edges = 5
        radius = rng.uniform(min_size, max_size)
        angles = np.sort(rng.uniform(0, 2 * np.pi, n_edges))
        boundary = [
            Point3D(cx + radius * np.cos(a), cy + radius * np.sin(a))
            for a in angles
        ]

        heights = np.linspace(min_h, max_h, num_heights)

        num_interrupt = max(1, r_id + 1)
        chains = []

        for h_idx, h in enumerate(heights):
            chain = InstanceGenerator._generate_spiral_chain(
                boundary, h, r_id, h_idx, rng, spacing
            )
            if chain is not None:
                chains.append(chain)

        return Region(
            id=r_id,
            boundary=boundary,
            chains=chains,
            num_interruption_points=num_interrupt
        )

    @staticmethod
    def _spiral_segments(boundary: List[Point3D], height: float,
                         scales: List[float]) -> List[Segment]:
        """Build zig-zag spiral segments for the boundary at the given scales.

        Each scale lives on a level scaled towards the centroid; consecutive
        levels are connected through vertex 0 to form a continuous chain. Odd
        levels are traversed in reverse order to obtain a zig-zag pattern.
        """
        pts = np.array([[p.x, p.y] for p in boundary])
        centroid = pts.mean(axis=0)
        n = len(pts)
        segments: List[Segment] = []
        for k, s in enumerate(scales):
            level = centroid + (pts - centroid) * s
            z = height
            order = list(range(n))
            if k % 2 == 1:
                order = [0] + list(reversed(range(1, n)))
            for i in range(len(order)):
                j = order[i]
                j_next = order[(i + 1) % n]
                segments.append(Segment(
                    Point3D(float(level[j, 0]), float(level[j, 1]), z),
                    Point3D(float(level[j_next, 0]), float(level[j_next, 1]), z)
                ))
            if k < len(scales) - 1:
                next_level = centroid + (pts - centroid) * scales[k + 1]
                segments.append(Segment(
                    Point3D(float(level[0, 0]), float(level[0, 1]), z),
                    Point3D(float(next_level[0, 0]),
                            float(next_level[0, 1]), z)
                ))
        return segments

    @staticmethod
    def _generate_spiral_chain(boundary: List[Point3D], height: float,
                                region_id: int, chain_idx: int,
                                rng: np.random.RandomState,
                                spacing: float) -> Optional[PolygonalChain]:
        if len(boundary) < 3:
            return None

        pts = np.array([[p.x, p.y] for p in boundary])
        centroid = pts.mean(axis=0)
        max_radius = np.linalg.norm(pts - centroid, axis=1).max()
        num_levels = max(1, int(max_radius / spacing))

        scales: List[float] = []
        for k in range(num_levels + 1):
            s = 1.0 - k * spacing / max_radius
            if s <= 0.0:
                break
            scales.append(s)

        if len(scales) < 2:
            return None

        segments = InstanceGenerator._spiral_segments(boundary, height, scales)
        if len(segments) < 2:
            return None

        return PolygonalChain(
            segments=segments,
            height=height,
            region_id=region_id,
            idx=chain_idx
        )



    @staticmethod
    def _overlaps(region: Region, others: List[Region], margin: float = 10.0) -> bool:
        pts = np.array([[p.x, p.y] for p in region.boundary])
        for other in others:
            opt = np.array([[p.x, p.y] for p in other.boundary])
            c1, c2 = pts.mean(axis=0), opt.mean(axis=0)
            r1 = max(np.linalg.norm(p - c1) for p in pts)
            r2 = max(np.linalg.norm(p - c2) for p in opt)
            if np.linalg.norm(c1 - c2) > r1 + r2 + margin:
                continue
            for p in pts:
                if InstanceGenerator._point_in_convex_polygon(p, opt):
                    return True
            for p in opt:
                if InstanceGenerator._point_in_convex_polygon(p, pts):
                    return True
            if np.linalg.norm(c1 - c2) < r1 + r2:
                return True
        return False

    @staticmethod
    def _point_in_convex_polygon(pt: np.ndarray, poly: np.ndarray) -> bool:
        n = len(poly)
        sign = None
        for i in range(n):
            a, b = poly[i], poly[(i + 1) % n]
            cross = np.cross(b - a, pt - a)
            if abs(cross) < 1e-12:
                continue
            s = np.sign(cross)
            if sign is None:
                sign = s
            elif s != sign:
                return False
        return True

    @staticmethod
    def rebuild_instance(base_inst: Instance,
                         base_spacing: float = 5.0,
                         ref_height: float = 15.0,
                         h_cone: float = 90.0,
                         region_heights: Optional[dict] = None) -> Instance:
        if region_heights is None:
            region_heights = {r.id: 15.0 + r.id * 25.0 for r in base_inst.regions}
        regions = []
        for r in base_inst.regions:
            bpts = r.boundary
            centroid = np.array([[p.x, p.y] for p in bpts]).mean(axis=0)
            max_radius = max(np.linalg.norm([p.x - centroid[0], p.y - centroid[1]]) for p in bpts)
            target_h = region_heights[r.id]
            new_chains = []
            for c_idx, chain in enumerate(r.chains):
                h = chain.height
                if abs(h - target_h) > 0.1:
                    continue
                spacing = base_spacing * (h / ref_height)
                s_max = max(0.1, 1.0 - h / h_cone)
                num_levels = max(1, int(s_max * max_radius / spacing))
                scales: List[float] = []
                for k in range(num_levels + 1):
                    s = s_max - k * spacing / max_radius
                    if s <= 0.0: break
                    scales.append(s)
                if len(scales) < 2: scales = [s_max, s_max * 0.5]
                segments = InstanceGenerator._spiral_segments(bpts, h, scales)
                new_chains.append(PolygonalChain(segments=segments, height=h, region_id=r.id, idx=c_idx))
            regions.append(Region(id=r.id, boundary=r.boundary, chains=new_chains,
                                  num_interruption_points=r.num_interruption_points))
        return Instance(
            regions=regions, depot=base_inst.depot,
            drone=base_inst.drone, wind=base_inst.wind,
            num_operations=base_inst.num_operations
        )

    @staticmethod
    def _build_intra_edges(instance: Instance) -> List[Edge]:
        edges = []
        R = instance.num_regions
        for r in instance.regions:
            num_v = r.num_vertices
            for i in range(1, num_v):
                u = Vertex(r.id, i, VertexType.START if i <= 2 else (
                    VertexType.LAUNCH if i % 2 == 1 else VertexType.RETRIEVE
                ))
                v = Vertex(r.id, i + 1, VertexType.START if i + 1 <= 2 else (
                    VertexType.LAUNCH if (i + 1) % 2 == 1 else VertexType.RETRIEVE
                ))
                if i % 2 == 0:
                    etype = EdgeType.INTRA_LR
                else:
                    etype = EdgeType.INTRA_RL
                edges.append(Edge(u, v, etype))
        return edges

    @staticmethod
    def _build_inter_edges(instance: Instance) -> List[Edge]:
        edges = []
        all_v = instance.all_vertices
        for u in all_v:
            for v in all_v:
                if u.region_id != v.region_id:
                    u_allow = (
                        u.vtype != VertexType.RETRIEVE
                        and not (u.idx == 2 or u.idx == 2 * instance.num_regions + 2
                                 if u.region_id < len(instance.regions) else True)
                    )
                    v_allow = (
                        v.vtype != VertexType.LAUNCH
                        and not (v.idx == 1 or v.idx == 2 * instance.num_regions + 1
                                 if v.region_id < len(instance.regions) else True)
                    )
                    edges.append(Edge(u, v, EdgeType.INTER))
        return edges

    @staticmethod
    def build_graph(instance: Instance) -> List[Edge]:
        intra = InstanceGenerator._build_intra_edges(instance)
        inter = InstanceGenerator._build_inter_edges(instance)
        return intra + inter
