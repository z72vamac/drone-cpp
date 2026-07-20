from __future__ import annotations
import numpy as np
import pytest
from drone_cpp.data_structures import (
    Point3D, Vertex, VertexType, Edge, EdgeType, DroneParams, WindParams, Instance,
)
from drone_cpp.instance_generator import InstanceGenerator


class TestSpiralSegments:
    def test_square_boundary_two_scales(self):
        boundary = [Point3D(0, 0), Point3D(10, 0), Point3D(10, 10), Point3D(0, 10)]
        segs = InstanceGenerator._spiral_segments(boundary, 25.0, [1.0, 0.5])
        assert len(segs) >= 5
        for s in segs:
            assert s.start.z == 25.0 and s.end.z == 25.0

    def test_single_scale(self):
        boundary = [Point3D(0, 0), Point3D(10, 0), Point3D(10, 10), Point3D(0, 10)]
        segs = InstanceGenerator._spiral_segments(boundary, 25.0, [1.0])
        assert len(segs) >= 4

    def test_empty_scales(self):
        boundary = [Point3D(0, 0), Point3D(10, 0), Point3D(10, 10), Point3D(0, 10)]
        segs = InstanceGenerator._spiral_segments(boundary, 25.0, [])
        assert len(segs) == 0

    def test_triangle_boundary(self):
        boundary = [Point3D(0, 0), Point3D(10, 0), Point3D(5, 10)]
        segs = InstanceGenerator._spiral_segments(boundary, 15.0, [1.0, 0.6, 0.2])
        assert len(segs) >= 3


class TestPointInConvexPolygon:
    @pytest.fixture
    def square(self):
        return np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)

    def test_inside(self, square):
        assert InstanceGenerator._point_in_convex_polygon(np.array([5, 5]), square)

    def test_outside(self, square):
        assert not InstanceGenerator._point_in_convex_polygon(np.array([15, 5]), square)

    def test_on_vertex(self, square):
        assert InstanceGenerator._point_in_convex_polygon(np.array([0, 0]), square)

    def test_on_edge(self, square):
        assert InstanceGenerator._point_in_convex_polygon(np.array([5, 0]), square)


class TestOverlaps:
    def _region_from_boundary(self, r_id, bpts):
        return InstanceGenerator._generate_region(
            r_id, np.random.RandomState(0), (0, 0, 100, 100),
            10, 20, 1, 15, 15, 5)

    def test_non_overlapping_regions_separate(self):
        r1 = InstanceGenerator._generate_region(
            0, np.random.RandomState(0), (0, 0, 100, 100),
            10, 20, 1, 15, 15, 5)
        r2 = InstanceGenerator._generate_region(
            1, np.random.RandomState(99), (200, 200, 300, 300),
            10, 20, 1, 15, 15, 5)
        assert not InstanceGenerator._overlaps(r1, [r2], margin=10.0)

    def test_same_region_overlaps(self):
        r1 = InstanceGenerator._generate_region(
            0, np.random.RandomState(0), (0, 0, 100, 100),
            10, 20, 1, 15, 15, 5)
        assert InstanceGenerator._overlaps(r1, [r1], margin=0.0)


class TestGenerateRegion:
    def test_region_has_chains(self):
        r = InstanceGenerator._generate_region(
            0, np.random.RandomState(42), (0, 0, 100, 100),
            10, 20, 2, 15, 40, 5)
        assert r.id == 0
        assert len(r.boundary) >= 3
        assert len(r.chains) >= 1
        assert r.num_interruption_points >= 1

    def test_region_deterministic(self):
        rng = np.random.RandomState(42)
        r1 = InstanceGenerator._generate_region(
            0, rng, (0, 0, 100, 100), 10, 20, 2, 15, 40, 5)
        rng = np.random.RandomState(42)
        r2 = InstanceGenerator._generate_region(
            0, rng, (0, 0, 100, 100), 10, 20, 2, 15, 40, 5)
        assert len(r1.boundary) == len(r2.boundary)
        for p1, p2 in zip(r1.boundary, r2.boundary):
            assert p1.x == p2.x and p1.y == p2.y


class TestBuildEdges:
    @pytest.fixture
    def instance(self):
        from drone_cpp.instance_generator import InstanceGenerator

        base = InstanceGenerator.generate(
            num_regions=2, seed=42, num_heights=1, num_ops=2)
        return InstanceGenerator.rebuild_instance(base)

    def test_intra_edges_have_correct_types(self, instance):
        edges = InstanceGenerator._build_intra_edges(instance)
        for e in edges:
            assert e.etype in (EdgeType.INTRA_LR, EdgeType.INTRA_RL)
            assert e.u.region_id == e.v.region_id
            assert abs(e.u.idx - e.v.idx) == 1

    def test_intra_edges_count(self, instance):
        edges = InstanceGenerator._build_intra_edges(instance)
        expected = sum(
            r.num_vertices - 1 for r in instance.regions
        )
        assert len(edges) == expected

    def test_inter_edges_different_regions(self, instance):
        edges = InstanceGenerator._build_inter_edges(instance)
        for e in edges:
            assert e.u.region_id != e.v.region_id

    def test_build_graph_returns_union(self, instance):
        g = InstanceGenerator.build_graph(instance)
        intra = InstanceGenerator._build_intra_edges(instance)
        inter = InstanceGenerator._build_inter_edges(instance)
        assert len(g) == len(intra) + len(inter)


class TestGenerate:
    def test_min_regions(self):
        inst = InstanceGenerator.generate(
            num_regions=1, seed=0, num_heights=1, num_ops=1)
        assert inst.num_regions == 1
        assert len(inst.regions[0].chains) >= 1

    def test_many_regions(self):
        inst = InstanceGenerator.generate(
            num_regions=3, seed=0, num_heights=1, num_ops=3,
            area_bounds=(0, 0, 200, 200), min_region_size=10, max_region_size=15)
        assert inst.num_regions == 3

    def test_area_bounds_respected(self):
        inst = InstanceGenerator.generate(
            num_regions=2, area_bounds=(0, 0, 100, 100),
            min_region_size=10, max_region_size=15,
            seed=42, num_heights=1, num_ops=2)
        for r in inst.regions:
            for p in r.boundary:
                assert 0 <= p.x <= 100
                assert 0 <= p.y <= 100

    def test_seed_reproducibility(self):
        a = InstanceGenerator.generate(num_regions=2, seed=99, num_heights=1, num_ops=2)
        b = InstanceGenerator.generate(num_regions=2, seed=99, num_heights=1, num_ops=2)
        assert a.depot.x == b.depot.x
        assert len(a.regions) == len(b.regions)
