"""Slow tests that build and solve a small CPPModel via Gurobi.

Run with:  pytest --runslow
These are skipped by default to keep CI fast and license-free.
"""
from __future__ import annotations
import pytest

from drone_cpp.instance_generator import InstanceGenerator
from drone_cpp.model import CPPModel


def _small_instance():
    base = InstanceGenerator.generate(
        num_regions=2, area_bounds=(0, 0, 100, 100),
        min_region_size=10.0, max_region_size=20.0, spacing=6.0,
        num_heights=2, min_height=15.0, max_height=40.0,
        num_ops=2, seed=13,
    )
    return InstanceGenerator.rebuild_instance(base)


@pytest.mark.slow
def test_model_builds():
    inst = _small_instance()
    model = CPPModel(inst, verbose=False)
    assert model.model.NumVars > 0
    assert model.model.NumConstrs > 0
    # Number of intra-region edges should be > 0.
    assert len(model._intra_rl) + len(model._intra_lr) > 0
    # Inter-region edges should exist (two different regions).
    assert len(model._inter_edges) > 0


@pytest.mark.slow
def test_small_instance_solves():
    inst = _small_instance()
    model = CPPModel(inst, verbose=False)
    model.model.setParam("MIPFocus", 1)
    solution = model.optimize(tl=30.0)
    if solution is None:
        pytest.skip("Gurobi returned no feasible solution within 30s "
                    "(license or model size limit)")
    assert solution.objective_value >= 0
    assert len(solution.operations) >= 1
    assert len(solution.chain_selection) == inst.num_regions
    # Metadata should now be populated by optimize().
    assert solution.status in ("OPTIMAL", "TIME_LIMIT", "SUBOPTIMAL", "INTERRUPTED")
    assert solution.solve_time is not None and solution.solve_time > 0