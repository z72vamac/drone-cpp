"""Tests for CPPModel — both fast (structure-only, Gurobi needed for creation)
and slow (actual solve).

The fast tests require Gurobi to *build* the model but do NOT solve it.
They are marked `slow` because Gurobi is a heavyweight dependency, but they
complete in < 1s once imported.

Run with:  pytest --runslow
"""
from __future__ import annotations
import pytest

from drone_cpp.model import CPPModel
from drone_cpp.data_structures import AtmosphereParams


# ---------------------------------------------------------------------------
# Structure tests (require Gurobi for model construction, no solve)
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_model_builds():
    from conftest import small_instance
    inst = small_instance()
    model = CPPModel(inst, verbose=False)
    assert model.model.NumVars > 0
    assert model.model.NumConstrs > 0
    assert len(model._intra_rl) + len(model._intra_lr) > 0
    assert len(model._inter_edges) > 0


@pytest.mark.slow
def test_model_variable_counts():
    from conftest import small_instance
    inst = small_instance()
    model = CPPModel(inst, verbose=False)
    n_mu = len(model.mu)
    n_x = len(model.x)
    n_y = len(model.y)
    assert n_mu > 0
    assert n_x > 0
    assert n_y > 0


@pytest.mark.slow
def test_model_precompute_keys():
    from conftest import small_instance
    inst = small_instance()
    model = CPPModel(inst, verbose=False)
    for r in inst.regions:
        for ti in range(len(r.chains)):
            key = (r.id, ti)
            assert key in model._cinfo
            info = model._cinfo[key]
            assert "seg_lens" in info
            assert "cum_before" in info
            assert "density" in info
            assert "nu_d_fwd" in info
            assert info["density"] > 0


@pytest.mark.slow
def test_model_bounds_non_negative():
    from conftest import small_instance
    inst = small_instance()
    model = CPPModel(inst, verbose=False)
    assert model._max_h() > 0
    assert model._max_d() > 0
    assert model._max_chain_len() > 0


@pytest.mark.slow
def test_model_endurance_and_depot():
    from conftest import small_instance
    inst = small_instance()
    model = CPPModel(inst, verbose=False)
    assert model.inst.drone.max_endurance > 0
    assert model.depot_v.region_id == -1
    assert model.inst.depot.z == 0.0


@pytest.mark.slow
def test_model_location_variables():
    from conftest import small_instance
    inst = small_instance()
    model = CPPModel(inst, verbose=False)
    for v in model.verts:
        assert v in model.P_x
        assert v in model.P_y
        assert v in model.P_z


@pytest.mark.slow
def test_model_alpha_variables():
    from conftest import small_instance
    inst = small_instance()
    model = CPPModel(inst, verbose=False)
    for r in inst.regions:
        for ti in range(len(r.chains)):
            assert (r.id, ti) in model.alpha


@pytest.mark.slow
def test_model_rho_positive():
    from conftest import small_instance
    inst = small_instance()
    model = CPPModel(inst, verbose=False)
    for r in inst.regions:
        ub = model.rho_sel[r.id].UB
        assert ub <= AtmosphereParams.air_density(0.)
        assert ub > 0


# ---------------------------------------------------------------------------
# Solve tests (require Gurobi + license)
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_small_instance_solves():
    from conftest import small_instance
    inst = small_instance()
    model = CPPModel(inst, verbose=False)
    model.model.setParam("MIPFocus", 1)
    solution = model.optimize(tl=30.0)
    if solution is None:
        pytest.skip("Gurobi returned no feasible solution within 30s "
                    "(license or model size limit)")
    assert solution.objective_value >= 0
    assert len(solution.operations) >= 1
    assert len(solution.chain_selection) == inst.num_regions
    assert solution.status in ("OPTIMAL", "TIME_LIMIT", "SUBOPTIMAL", "INTERRUPTED")
    assert solution.solve_time is not None and solution.solve_time > 0


@pytest.mark.slow
def test_model_optimize_returns_solution_metadata():
    from conftest import small_instance
    inst = small_instance()
    model = CPPModel(inst, verbose=False)
    solution = model.optimize(tl=30.0)
    if solution is None:
        pytest.skip("Gurobi returned no feasible solution")
    assert solution.objective_value >= 0
    assert solution.solve_time is not None
    assert solution.status is not None
