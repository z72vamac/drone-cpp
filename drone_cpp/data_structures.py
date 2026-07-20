from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Tuple, Optional
import numpy as np


class VertexType(Enum):
    START = 1
    LAUNCH = 2
    RETRIEVE = 3
    END = 4


class EdgeType(Enum):
    INTRA_LR = 1
    INTRA_RL = 2
    INTER = 3
    RING_TRANSITION = 4


@dataclass
class Point3D:
    x: float
    y: float
    z: float = 0.0

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    def dist_to(self, other: Point3D) -> float:
        return float(np.linalg.norm(self.to_array() - other.to_array()))

    def dist_xy(self, other: Point3D) -> float:
        return float(np.linalg.norm(np.array([self.x - other.x, self.y - other.y])))


@dataclass
class Segment:
    start: Point3D
    end: Point3D

    @property
    def length(self) -> float:
        return self.start.dist_to(self.end)

    @property
    def direction(self) -> np.ndarray:
        d = self.end.to_array() - self.start.to_array()
        n = np.linalg.norm(d)
        return d / n if n > 0 else np.zeros(3)

    @property
    def midpoint(self) -> Point3D:
        return Point3D(
            (self.start.x + self.end.x) / 2,
            (self.start.y + self.end.y) / 2,
            (self.start.z + self.end.z) / 2
        )


@dataclass
class Ring:
    segments: List[Segment]
    scale: float
    height: float

    @property
    def perimeter(self) -> float:
        return sum(s.length for s in self.segments)

    @property
    def num_vertices(self) -> int:
        return len(self.segments)


@dataclass
class PolygonalChain:
    segments: List[Segment] = field(default_factory=list)
    height: float = 0.0
    region_id: int = 0
    idx: int = 0
    rings: List[Ring] = None

    def __post_init__(self):
        if self.rings is None:
            self.rings = []

    @property
    def total_length(self) -> float:
        return sum(s.length for s in self.segments)

    def cumulative_lengths(self) -> List[float]:
        cum = [0.0]
        for s in self.segments:
            cum.append(cum[-1] + s.length)
        return cum

    def cumulative_lengths_before_segment(self) -> List[float]:
        return self.cumulative_lengths()[:-1]

    def segment_lengths(self) -> List[float]:
        return [s.length for s in self.segments]

    def compute_drone_speed_on_segment(self, s_idx: int, forward: bool,
                                        wind_speed: float, wind_dir: np.ndarray,
                                        cruise_speed: float) -> float:
        s = self.segments[s_idx]
        if forward:
            seg_dir = s.direction
        else:
            seg_dir = -s.direction
        wind_vec = wind_speed * wind_dir
        nu_d_vec = cruise_speed * seg_dir - wind_vec
        return float(np.linalg.norm(nu_d_vec))

    def ring_segment_offset(self, ring_idx: int) -> int:
        offset = 0
        for i in range(ring_idx):
            offset += len(self.rings[i].segments)
        return offset


@dataclass
class Region:
    id: int
    boundary: List[Point3D]
    chains: List[PolygonalChain]
    num_interruption_points: int

    @property
    def num_vertices(self) -> int:
        return 2 * self.num_interruption_points + 2


@dataclass
class Vertex:
    region_id: int
    idx: int
    vtype: VertexType

    def __hash__(self):
        return hash((self.region_id, self.idx, self.vtype.value))

    def __repr__(self):
        return f"V(r{self.region_id},{self.vtype.name}{self.idx})"


@dataclass
class Edge:
    u: Vertex
    v: Vertex
    etype: EdgeType

    def __hash__(self):
        return hash((self.u, self.v, self.etype.value))

    def __repr__(self):
        return f"E({self.u}->{self.v},{self.etype.name})"


@dataclass
class DroneParams:
    front_area: float
    drag_coef: float
    max_endurance: float
    cruise_speed: float
    vertical_speed: float

    @property
    def E_xy(self) -> float:
        return 0.5 * self.drag_coef * self.front_area

    @property
    def E_z(self) -> float:
        return 0.5 * self.drag_coef * self.front_area


@dataclass
class WindParams:
    direction: np.ndarray
    speed_at_10m: float
    hellmann_exponent: float

    def speed_at_height(self, h: float) -> float:
        if h <= 0:
            return 0.0
        return self.speed_at_10m * (h / 10.0) ** self.hellmann_exponent


class AtmosphereParams:
    p0 = 101325.0
    T0 = 288.15
    g = 9.80665
    L = 0.0065
    C_R = 8.31446
    M = 0.0289652

    @classmethod
    def air_density(cls, h: float) -> float:
        factor = 1.0 - cls.L * h / cls.T0
        exp = cls.g * cls.M / (cls.C_R * cls.L) - 1.0
        return (cls.p0 * cls.M / (cls.C_R * cls.T0)) * (factor ** exp)


@dataclass
class Instance:
    regions: List[Region]
    depot: Point3D
    drone: DroneParams
    wind: WindParams
    num_operations: int

    @property
    def num_regions(self) -> int:
        return len(self.regions)

    @property
    def all_vertices(self) -> List[Vertex]:
        verts = []
        R = self.num_regions
        total_indices = 2 * R + 2
        for r in self.regions:
            for i in range(1, total_indices + 1):
                if i <= 2:
                    vtype = VertexType.START
                elif i <= 2 * R:
                    if i % 2 == 1:
                        vtype = VertexType.LAUNCH
                    else:
                        vtype = VertexType.RETRIEVE
                else:
                    vtype = VertexType.END
                verts.append(Vertex(r.id, i, vtype))
        return verts

    @property
    def depot_vertex(self) -> Vertex:
        return Vertex(-1, 0, VertexType.START)


@dataclass
class Operation:
    edges: List[Tuple[Vertex, Vertex]]


@dataclass
class Solution:
    operations: List[Operation]
    objective_value: float
    vertex_positions: Dict[Vertex, Point3D] = field(default_factory=dict)
    chain_selection: Dict[int, int] = field(default_factory=dict)
    vertex_lambdas: Dict[Vertex, float] = field(default_factory=dict)
    vertex_rings: Dict[Vertex, int] = field(default_factory=dict)
    solve_time: Optional[float] = None
    mip_gap: Optional[float] = None
    status: Optional[str] = None
    first_incumbent_obj: Optional[float] = None
    first_incumbent_time: Optional[float] = None

    def save(self, path: str):
        import json
        vp = {}
        for k, p in self.vertex_positions.items():
            vp[f"{k.region_id},{k.idx},{k.vtype.name}"] = {"x": p.x, "y": p.y, "z": p.z}
        vl = {}
        for k, lam in self.vertex_lambdas.items():
            vl[f"{k.region_id},{k.idx},{k.vtype.name}"] = lam
        vr = {}
        for k, r_idx in self.vertex_rings.items():
            vr[f"{k.region_id},{k.idx},{k.vtype.name}"] = r_idx
        ops = []
        for op in self.operations:
            ops.append([{"r": u.region_id, "i": u.idx, "t": u.vtype.name,
                         "r2": v.region_id, "i2": v.idx, "t2": v.vtype.name}
                        for u, v in op.edges])
        data = {
            "objective_value": self.objective_value,
            "chain_selection": self.chain_selection,
            "vertex_positions": vp,
            "vertex_lambdas": vl,
            "vertex_rings": vr,
            "operations": ops,
            "solve_time": self.solve_time,
            "mip_gap": self.mip_gap,
            "status": self.status,
            "first_incumbent_obj": self.first_incumbent_obj,
            "first_incumbent_time": self.first_incumbent_time,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> Solution:
        import json
        with open(path) as f:
            data = json.load(f)
        vp = {}
        for k, v in data["vertex_positions"].items():
            parts = k.split(",")
            vk = Vertex(int(parts[0]), int(parts[1]), VertexType[parts[2]])
            vp[vk] = Point3D(v["x"], v["y"], v["z"])
        vl = {}
        if "vertex_lambdas" in data:
            for k, lam in data["vertex_lambdas"].items():
                parts = k.split(",")
                vk = Vertex(int(parts[0]), int(parts[1]), VertexType[parts[2]])
                vl[vk] = lam
        ops = []
        for op_data in data["operations"]:
            edges = [(Vertex(e["r"], e["i"], VertexType[e["t"]]),
                      Vertex(e["r2"], e["i2"], VertexType[e["t2"]])) for e in op_data]
            ops.append(Operation(edges))
        cs = {int(k): int(v) for k, v in data["chain_selection"].items()}
        vr = {}
        if "vertex_rings" in data:
            for k, r_idx in data["vertex_rings"].items():
                parts = k.split(",")
                vk = Vertex(int(parts[0]), int(parts[1]), VertexType[parts[2]])
                vr[vk] = int(r_idx)
        status = data.get("status")
        if isinstance(status, int):
            status = {2: "OPTIMAL", 3: "INFEASIBLE", 8: "TIME_LIMIT",
                       9: "SUBOPTIMAL", 11: "INTERRUPTED"}.get(status, str(status))
        return cls(operations=ops, objective_value=data["objective_value"],
                   vertex_positions=vp, chain_selection=cs, vertex_lambdas=vl,
                   vertex_rings=vr,
                   solve_time=data.get("solve_time"),
                   mip_gap=data.get("mip_gap"),
                   status=status,
                   first_incumbent_obj=data.get("first_incumbent_obj"),
                   first_incumbent_time=data.get("first_incumbent_time"))
