from __future__ import annotations
import numpy as np
import pytest
from drone_cpp.data_structures import (
    Point3D, Segment, PolygonalChain, Region, Vertex,
    VertexType, Edge, EdgeType, DroneParams, WindParams,
    AtmosphereParams, Instance,
)


class TestPoint3D:
    def test_to_array(self):
        p = Point3D(1.0, 2.0, 3.0)
        arr = p.to_array()
        assert arr.shape == (3,)
        assert arr[0] == 1.0 and arr[1] == 2.0 and arr[2] == 3.0

    def test_to_array_default_z(self):
        p = Point3D(1.0, 2.0)
        assert p.to_array()[2] == 0.0

    def test_dist_to_same_point(self):
        p = Point3D(3, 4, 5)
        assert p.dist_to(p) == 0.0

    def test_dist_to(self):
        a = Point3D(0, 0, 0)
        b = Point3D(3, 4, 0)
        assert a.dist_to(b) == 5.0

    def test_dist_to_3d(self):
        a = Point3D(0, 0, 0)
        b = Point3D(2, 3, 6)
        assert a.dist_to(b) == 7.0

    def test_dist_xy_ignores_z(self):
        a = Point3D(0, 0, 100)
        b = Point3D(3, 4, 200)
        assert a.dist_xy(b) == 5.0


class TestSegment:
    def test_length(self):
        s = Segment(Point3D(0, 0), Point3D(3, 4))
        assert s.length == 5.0

    def test_length_zero(self):
        s = Segment(Point3D(1, 2, 3), Point3D(1, 2, 3))
        assert s.length == 0.0

    def test_direction_unit(self):
        s = Segment(Point3D(0, 0), Point3D(3, 0))
        assert np.allclose(s.direction, [1, 0, 0])

    def test_direction_zero_length(self):
        s = Segment(Point3D(1, 2), Point3D(1, 2))
        assert np.allclose(s.direction, [0, 0, 0])

    def test_midpoint(self):
        s = Segment(Point3D(0, 0, 0), Point3D(2, 4, 6))
        m = s.midpoint
        assert m.x == 1.0 and m.y == 2.0 and m.z == 3.0


class TestPolygonalChain:
    @pytest.fixture
    def chain(self):
        s1 = Segment(Point3D(0, 0, 10), Point3D(3, 0, 10))
        s2 = Segment(Point3D(3, 0, 10), Point3D(3, 4, 10))
        return PolygonalChain(segments=[s1, s2], height=10.0, region_id=0, idx=0)

    def test_total_length(self, chain):
        assert chain.total_length == 7.0

    def test_cumulative_lengths(self, chain):
        assert chain.cumulative_lengths() == [0.0, 3.0, 7.0]

    def test_cumulative_lengths_before_segment(self, chain):
        assert chain.cumulative_lengths_before_segment() == [0.0, 3.0]

    def test_segment_lengths(self, chain):
        assert chain.segment_lengths() == [3.0, 4.0]

    def test_compute_drone_speed_on_segment_forward_no_wind(self, chain):
        v = chain.compute_drone_speed_on_segment(
            0, forward=True, wind_speed=0.0,
            wind_dir=np.array([1, 0, 0]), cruise_speed=15.0)
        assert v == 15.0

    def test_compute_drone_speed_on_segment_backward_no_wind(self, chain):
        v = chain.compute_drone_speed_on_segment(
            0, forward=False, wind_speed=0.0,
            wind_dir=np.array([1, 0, 0]), cruise_speed=15.0)
        assert v == 15.0

    def test_compute_drone_speed_with_wind(self, chain):
        v = chain.compute_drone_speed_on_segment(
            0, forward=True, wind_speed=5.0,
            wind_dir=np.array([1, 0, 0]), cruise_speed=15.0)
        expected = np.linalg.norm(np.array([15, 0, 0]) - np.array([5, 0, 0]))
        assert v == expected

    def test_compute_drone_speed_cross_wind(self):
        seg1 = Segment(Point3D(0, 0, 10), Point3D(0, 3, 10))
        c = PolygonalChain(segments=[seg1], height=10.0, region_id=0, idx=0)
        v = c.compute_drone_speed_on_segment(
            0, forward=True, wind_speed=5.0,
            wind_dir=np.array([1, 0, 0]), cruise_speed=15.0)
        expected = np.linalg.norm(np.array([-5, 15, 0]))
        assert v == pytest.approx(expected, rel=1e-6)


class TestDroneParams:
    def test_E_xy(self):
        d = DroneParams(front_area=0.1, drag_coef=0.3, max_endurance=100.0,
                        cruise_speed=15.0, vertical_speed=5.0)
        assert d.E_xy == 0.5 * 0.3 * 0.1

    def test_E_z(self):
        d = DroneParams(front_area=0.1, drag_coef=0.3, max_endurance=100.0,
                        cruise_speed=15.0, vertical_speed=5.0)
        assert d.E_z == 0.5 * 0.3 * 0.1


class TestWindParams:
    def test_speed_at_ground_zero(self):
        w = WindParams(direction=np.array([1.0, 0.0, 0.0]),
                       speed_at_10m=5.0, hellmann_exponent=0.2)
        assert w.speed_at_height(0.0) == 0.0

    def test_speed_at_reference_height(self):
        w = WindParams(direction=np.array([1.0, 0.0, 0.0]),
                       speed_at_10m=5.0, hellmann_exponent=0.2)
        assert w.speed_at_height(10.0) == 5.0

    def test_speed_increases_with_height(self):
        w = WindParams(direction=np.array([1.0, 0.0, 0.0]),
                       speed_at_10m=5.0, hellmann_exponent=0.2)
        assert w.speed_at_height(40.0) > 5.0

    def test_negative_height_returns_zero(self):
        w = WindParams(direction=np.array([1.0, 0.0, 0.0]),
                       speed_at_10m=5.0, hellmann_exponent=0.2)
        assert w.speed_at_height(-10.0) == 0.0


class TestAtmosphereParams:
    def test_air_density_sea_level(self):
        rho0 = AtmosphereParams.air_density(0.0)
        assert rho0 == pytest.approx(1.225, rel=1e-2)

    def test_air_density_decreases_with_height(self):
        rho0 = AtmosphereParams.air_density(0.0)
        rho1000 = AtmosphereParams.air_density(1000.0)
        assert rho1000 < rho0

    def test_air_density_at_typical_drone_heights(self):
        values = [
            AtmosphereParams.air_density(h)
            for h in [15.0, 40.0, 65.0]
        ]
        for i in range(len(values) - 1):
            assert values[i + 1] < values[i]

    def test_air_density_reasonable_bounds(self):
        rho = AtmosphereParams.air_density(100.0)
        assert 1.0 < rho < 1.3


class TestVertex:
    def test_hash(self):
        v1 = Vertex(0, 1, VertexType.START)
        v2 = Vertex(0, 1, VertexType.START)
        assert hash(v1) == hash(v2)

    def test_hash_different(self):
        v1 = Vertex(0, 1, VertexType.START)
        v2 = Vertex(1, 1, VertexType.START)
        assert hash(v1) != hash(v2)

    def test_repr(self):
        v = Vertex(0, 1, VertexType.START)
        assert repr(v) == "V(r0,START1)"


class TestEdge:
    def test_hash(self):
        e1 = Edge(Vertex(0, 1, VertexType.START),
                  Vertex(0, 2, VertexType.START),
                  EdgeType.INTRA_RL)
        e2 = Edge(Vertex(0, 1, VertexType.START),
                  Vertex(0, 2, VertexType.START),
                  EdgeType.INTRA_RL)
        assert hash(e1) == hash(e2)

    def test_hash_different_type(self):
        e1 = Edge(Vertex(0, 1, VertexType.START),
                  Vertex(0, 2, VertexType.LAUNCH),
                  EdgeType.INTRA_RL)
        e2 = Edge(Vertex(0, 1, VertexType.START),
                  Vertex(0, 2, VertexType.LAUNCH),
                  EdgeType.INTRA_LR)
        assert hash(e1) != hash(e2)

    def test_repr(self):
        e = Edge(Vertex(0, 1, VertexType.START),
                 Vertex(0, 2, VertexType.LAUNCH),
                 EdgeType.INTRA_LR)
        assert "E(" in repr(e) and "INTRA_LR" in repr(e)


class TestInstance:
    def _small_instance(self):
        boundary = [Point3D(0, 0), Point3D(10, 0), Point3D(10, 10), Point3D(0, 10)]
        chain = PolygonalChain(
            segments=[Segment(Point3D(0, 0, 10), Point3D(10, 0, 10))],
            height=10.0, region_id=0, idx=0)
        r = Region(id=0, boundary=boundary, chains=[chain], num_interruption_points=1)
        depot = Point3D(50, 50, 0)
        drone = DroneParams(0.1, 0.3, 100.0, 15.0, 5.0)
        wind = WindParams(np.array([1.0, 0.0, 0.0]), 5.0, 0.2)
        return Instance(regions=[r], depot=depot, drone=drone, wind=wind, num_operations=1)

    def test_num_regions(self):
        inst = self._small_instance()
        assert inst.num_regions == 1

    def test_depot_vertex(self):
        inst = self._small_instance()
        dv = inst.depot_vertex
        assert dv.region_id == -1
        assert dv.idx == 0
        assert dv.vtype == VertexType.START
