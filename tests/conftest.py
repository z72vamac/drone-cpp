"""Pytest configuration shared across the test suite."""
import sys
import os
import pytest

# Make the package importable when running from the repo root without install.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Use a non-interactive matplotlib backend so plots don't try to open windows.
import matplotlib
matplotlib.use("Agg")


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False,
        help="Run slow tests that require a Gurobi license.",
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def make_instance(num_regions: int = 2, seed: int = 7):
    """Build an instance whose chains survive rebuild_instance's defaults.

    rebuild_instance assigns region r the target altitude 15 + r*25, which
    must match one of the heights produced by generate(). With num_heights=3
    the altitudes are 15, 40 and 65 m, so any region_id in {0,1,2} matches.
    """
    from drone_cpp.instance_generator import InstanceGenerator
    base = InstanceGenerator.generate(
        num_regions=num_regions, area_bounds=(0, 0, 100, 100),
        min_region_size=10.0, max_region_size=20.0, spacing=6.0,
        num_heights=3, min_height=15.0, max_height=65.0,
        num_ops=num_regions, seed=seed,
    )
    return InstanceGenerator.rebuild_instance(base)


def mock_solution(inst):
    """Build a trivial mock solution picking the first chain of each region."""
    from drone_cpp.data_structures import Solution, Operation, Vertex, VertexType
    chain_selection = {r.id: 0 for r in inst.regions}
    vp, vl, ops = {}, {}, []
    for r in inst.regions:
        sel = r.chains[0]
        depot_vertex = inst.depot_vertex
        v_start = Vertex(r.id, 1, VertexType.START)
        v_end = Vertex(r.id, 2 * r.num_interruption_points + 2, VertexType.END)
        vp[v_start] = sel.segments[0].start
        vp[v_end] = sel.segments[-1].end
        vl[v_start] = 0.0
        vl[v_end] = float(len(sel.segments))
        ops.append(Operation([(depot_vertex, v_start), (v_start, v_end),
                              (v_end, depot_vertex)]))
    return Solution(
        operations=ops, objective_value=1234.5,
        vertex_positions=vp, chain_selection=chain_selection,
        vertex_lambdas=vl,
        solve_time=1.25, mip_gap=0.01, status="OPTIMAL",
    )


def small_instance():
    """Create a 2-region instance suitable for model tests."""
    from drone_cpp.instance_generator import InstanceGenerator
    base = InstanceGenerator.generate(
        num_regions=2, area_bounds=(0, 0, 100, 100),
        min_region_size=10.0, max_region_size=20.0, spacing=6.0,
        num_heights=2, min_height=15.0, max_height=40.0,
        num_ops=2, seed=13,
    )
    return InstanceGenerator.rebuild_instance(base)


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="Needs --runslow and a Gurobi license")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)