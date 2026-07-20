"""Fast smoke tests that do NOT require Gurobi.

They verify:
- Instance generation returns a well-formed object.
- rebuild_instance retains region count and replaces chains.
- Solution roundtrip (save -> load) preserves all fields, including metadata.
- CPPVis plotting helpers do not raise on a mock solution.
"""
from __future__ import annotations
import os
import json
import tempfile

import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from drone_cpp.instance_generator import InstanceGenerator
from drone_cpp.data_structures import (
    Instance, Solution, Operation, Vertex, VertexType, Point3D,
)
from drone_cpp.visualization import CPPVis


# ---------------------------------------------------------------------------
# Instance generation
# ---------------------------------------------------------------------------
def test_generate_instance_shapes():
    inst = InstanceGenerator.generate(
        num_regions=3, area_bounds=(0, 0, 100, 100),
        min_region_size=12.0, max_region_size=25.0, spacing=6.0,
        num_heights=3, min_height=15.0, max_height=65.0,
        num_ops=3, seed=42,
    )
    assert inst.num_regions == 3
    for r in inst.regions:
        assert r.num_interruption_points >= 1
        assert len(r.boundary) >= 3
        assert len(r.chains) >= 1
    assert inst.depot.z == 0.0
    assert inst.wind.speed_at_10m > 0
    assert abs(inst.wind.direction[2]) < 1e-9


def test_generate_instance_reproducible():
    a = InstanceGenerator.generate(num_regions=2, seed=11)
    b = InstanceGenerator.generate(num_regions=2, seed=11)
    assert a.depot.x == b.depot.x and a.depot.y == b.depot.y
    assert len(a.regions) == len(b.regions)
    for ra, rb in zip(a.regions, b.regions):
        assert len(ra.chains) == len(rb.chains)
        for ca, cb in zip(ra.chains, rb.chains):
            assert ca.height == cb.height
            assert len(ca.segments) == len(cb.segments)


def test_rebuild_instance_preserves_metadata():
    base = InstanceGenerator.generate(
        num_regions=2, area_bounds=(0, 0, 100, 100),
        min_region_size=10.0, max_region_size=20.0, spacing=6.0,
        num_heights=3, min_height=15.0, max_height=65.0,
        num_ops=2, seed=5,
    )
    rebuilt = InstanceGenerator.rebuild_instance(base)
    assert rebuilt.num_regions == base.num_regions
    assert rebuilt.num_operations == base.num_operations
    assert rebuilt.depot.x == base.depot.x
    assert rebuilt.wind.speed_at_10m == base.wind.speed_at_10m
    for r in rebuilt.regions:
        assert len(r.chains) >= 1, f"Region {r.id} lost all chains after rebuild"


def test_spiral_segments_helper():
    boundary = [Point3D(0, 0), Point3D(10, 0), Point3D(10, 10), Point3D(0, 10)]
    segs = InstanceGenerator._spiral_segments(boundary, 25.0, [1.0, 0.5])
    assert len(segs) >= 5
    for s in segs:
        assert s.start.z == 25.0 and s.end.z == 25.0


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
def test_wind_speed_at_height():
    import numpy as np
    from drone_cpp.data_structures import WindParams
    w = WindParams(direction=np.array([1.0, 0.0, 0.0]), speed_at_10m=5.0,
                   hellmann_exponent=0.2)
    assert w.speed_at_height(0.0) == 0.0
    assert abs(w.speed_at_height(10.0) - 5.0) < 1e-9
    assert w.speed_at_height(40.0) > 5.0


def test_polygonal_chain_lengths():
    base = InstanceGenerator.generate(num_regions=1, seed=3,
                                      num_heights=2, num_ops=1)
    for r in base.regions:
        for ch in r.chains:
            seglens = ch.segment_lengths()
            cumbefore = ch.cumulative_lengths_before_segment()
            assert len(seglens) == len(cumbefore)
            assert all(s >= 0 for s in seglens)
            assert ch.total_length == sum(seglens)


# ---------------------------------------------------------------------------
# Solution roundtrip
# ---------------------------------------------------------------------------
def test_solution_roundtrip(tmp_path):
    from conftest import make_instance, mock_solution
    inst = make_instance(num_regions=2, seed=5)
    sol = mock_solution(inst)
    p = tmp_path / "sol.json"
    sol.save(str(p))
    loaded = Solution.load(str(p))
    assert loaded.objective_value == sol.objective_value
    assert loaded.chain_selection == sol.chain_selection
    assert loaded.vertex_lambdas == sol.vertex_lambdas
    assert loaded.solve_time == sol.solve_time
    assert loaded.mip_gap == sol.mip_gap
    assert loaded.status == sol.status
    for v, pos in sol.vertex_positions.items():
        assert v in loaded.vertex_positions
        assert loaded.vertex_positions[v].x == pytest.approx(pos.x)
        assert loaded.vertex_positions[v].y == pytest.approx(pos.y)
        assert loaded.vertex_positions[v].z == pytest.approx(pos.z)


def test_solution_load_legacy_without_metadata(tmp_path):
    from conftest import make_instance, mock_solution
    inst = make_instance(num_regions=2, seed=5)
    sol = mock_solution(inst)
    p = tmp_path / "sol.json"
    sol.save(str(p))
    with open(p) as f:
        data = json.load(f)
    for k in ("solve_time", "mip_gap", "status"):
        data.pop(k, None)
    with open(p, "w") as f:
        json.dump(data, f)
    loaded = Solution.load(str(p))
    assert loaded.solve_time is None
    assert loaded.mip_gap is None
    assert loaded.status is None
    assert loaded.objective_value == sol.objective_value


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def test_plot_instance_returns_figure():
    from conftest import make_instance
    inst = make_instance(num_regions=2, seed=7)
    fig = CPPVis.plot_instance(inst, chain_selection={0: 0, 1: 0})
    assert fig is not None
    plt.close(fig)


def test_plot_solution_2d_returns_figure():
    from conftest import make_instance, mock_solution
    inst = make_instance(num_regions=2, seed=7)
    sol = mock_solution(inst)
    fig = CPPVis.plot_solution_2d(inst, sol)
    assert fig is not None
    plt.close(fig)


def test_plot_solution_3d_returns_figure():
    from conftest import make_instance, mock_solution
    inst = make_instance(num_regions=2, seed=7)
    sol = mock_solution(inst)
    fig = CPPVis.plot_solution_3d(inst, sol)
    assert fig is not None
    plt.close(fig)


def test_plot_solution_combined_returns_figure():
    from conftest import make_instance, mock_solution
    inst = make_instance(num_regions=2, seed=7)
    sol = mock_solution(inst)
    fig = CPPVis.plot_solution(inst, sol)
    assert fig is not None
    plt.close(fig)
