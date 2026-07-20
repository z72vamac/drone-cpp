from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from drone_cpp.data_structures import (
    Point3D, Segment, PolygonalChain, Vertex, VertexType,
    Solution, Operation,
)
from drone_cpp.visualization import CPPVis


class TestLambdToCum:
    @pytest.fixture
    def chain(self):
        s1 = Segment(Point3D(0, 0, 10), Point3D(3, 0, 10))
        s2 = Segment(Point3D(3, 0, 10), Point3D(3, 4, 10))
        return PolygonalChain(segments=[s1, s2], height=10.0, region_id=0, idx=0)

    def test_lam_zero(self, chain):
        assert CPPVis.lambd_to_cum(chain, 0.0) == 0.0

    def test_lam_negative(self, chain):
        assert CPPVis.lambd_to_cum(chain, -1.0) == 0.0

    def test_lam_equal_to_num_segments(self, chain):
        assert CPPVis.lambd_to_cum(chain, 2.0) == 7.0

    def test_lam_beyond_num_segments(self, chain):
        assert CPPVis.lambd_to_cum(chain, 10.0) == 7.0

    def test_lam_mid_segment(self, chain):
        result = CPPVis.lambd_to_cum(chain, 0.5)
        assert result == pytest.approx(1.5)

    def test_lam_at_segment_boundary(self, chain):
        assert CPPVis.lambd_to_cum(chain, 1.0) == 3.0


class TestGetChainPath:
    @pytest.fixture
    def chain(self):
        s1 = Segment(Point3D(0, 0, 10), Point3D(3, 0, 10))
        s2 = Segment(Point3D(3, 0, 10), Point3D(3, 4, 10))
        return PolygonalChain(segments=[s1, s2], height=10.0, region_id=0, idx=0)

    def test_normal_order(self, chain):
        path = CPPVis.get_chain_path(chain, 0.0, 3.0)
        assert len(path) >= 1
        _, _, _, x2, y2, _ = path[0]
        assert x2 == 3.0 and y2 == 0.0

    def test_reversed_order(self, chain):
        path = CPPVis.get_chain_path(chain, 3.0, 0.0)
        assert len(path) >= 1

    def test_same_cum_returns_zero_length_path(self, chain):
        path = CPPVis.get_chain_path(chain, 1.5, 1.5)
        assert len(path) == 1
        x1, y1, z1, x2, y2, z2 = path[0]
        assert x1 == pytest.approx(x2) and y1 == pytest.approx(y2)

    def test_full_chain(self, chain):
        path = CPPVis.get_chain_path(chain, 0.0, chain.total_length)
        assert len(path) == 2

    def test_partial_mid_chain(self, chain):
        path = CPPVis.get_chain_path(chain, 1.0, 5.0)
        assert len(path) >= 1
        _, _, _, x2, y2, _ = path[-1]
        assert y2 == pytest.approx(2.0, abs=0.01)

    def test_single_segment_chain(self):
        chain = PolygonalChain(
            segments=[Segment(Point3D(0, 0, 10), Point3D(5, 0, 10))],
            height=10.0, region_id=0, idx=0)
        path = CPPVis.get_chain_path(chain, 1.0, 4.0)
        assert len(path) == 1


class TestEdgeClassification:
    @pytest.fixture
    def vertices(self):
        return (
            Vertex(0, 1, VertexType.START),
            Vertex(0, 2, VertexType.START),
            Vertex(0, 3, VertexType.LAUNCH),
            Vertex(1, 1, VertexType.START),
        )

    def test_rl_edge(self, vertices):
        u, v = vertices[1], vertices[2]
        assert CPPVis.is_rl_edge(u, v)

    def test_not_rl_edge_lr(self, vertices):
        u, v = vertices[0], vertices[1]
        assert not CPPVis.is_rl_edge(u, v)

    def test_inter_edge(self, vertices):
        u, v = vertices[0], vertices[3]
        assert CPPVis.is_inter_edge(u, v)

    def test_not_inter_edge_same_region(self, vertices):
        u, v = vertices[0], vertices[1]
        assert not CPPVis.is_inter_edge(u, v)

    def test_depot_not_inter(self):
        u = Vertex(-1, 0, VertexType.START)
        v = Vertex(0, 1, VertexType.START)
        assert not CPPVis.is_inter_edge(u, v)


class TestPlotInstance:
    def test_returns_figure_with_chains(self):
        from conftest import make_instance
        inst = make_instance(num_regions=1, seed=7)
        fig = CPPVis.plot_instance(inst, chain_selection={0: 0})
        assert fig is not None
        plt.close(fig)

    def test_returns_figure_without_chains(self):
        from conftest import make_instance
        inst = make_instance(num_regions=1, seed=7)
        fig = CPPVis.plot_instance(inst)
        assert fig is not None
        plt.close(fig)

    def test_two_regions(self):
        from conftest import make_instance
        inst = make_instance(num_regions=2, seed=7)
        fig = CPPVis.plot_instance(inst, chain_selection={0: 0, 1: 0})
        assert fig is not None
        plt.close(fig)


class TestPlotSolution:
    def test_2d_empty_operations(self):
        from conftest import make_instance
        inst = make_instance(num_regions=1, seed=7)
        sol = Solution(operations=[], objective_value=0.0)
        fig = CPPVis.plot_solution_2d(inst, sol)
        assert fig is not None
        plt.close(fig)

    def test_3d_empty_operations(self):
        from conftest import make_instance
        inst = make_instance(num_regions=1, seed=7)
        sol = Solution(operations=[], objective_value=0.0)
        fig = CPPVis.plot_solution_3d(inst, sol)
        assert fig is not None
        plt.close(fig)

    def test_combined_no_solution_metadata(self):
        from conftest import make_instance, mock_solution
        inst = make_instance(num_regions=1, seed=7)
        sol = mock_solution(inst)
        fig = CPPVis.plot_solution(inst, sol)
        assert fig is not None
        plt.close(fig)
