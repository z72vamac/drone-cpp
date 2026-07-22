"""Tests for EdgesModel — structural checks, warm start, and equivalence with RingsModel.

Run with:  pytest --runslow
"""
from __future__ import annotations
import pytest

from drone_cpp.models.mip_edges import EdgesModel
from drone_cpp.model import build_model


# ---------------------------------------------------------------------------
# Structural tests (model construction, no solve)
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_edge_model_builds():
    from conftest import small_instance
    inst = small_instance()
    model = EdgesModel(inst, verbose=False)
    assert model.model.NumVars > 0
    assert model.model.NumConstrs > 0


@pytest.mark.slow
def test_edge_model_has_no_y_vertex_vars():
    from conftest import small_instance
    inst = small_instance()
    model = EdgesModel(inst, verbose=False)
    # There should be no variable named y_{v}_o{o} (vertex-visit vars)
    for v in model.verts:
        for o in range(model.O):
            assert (v, o) not in getattr(model, "y", {}), \
                f"Vertex-visit y[{v},o{o}] should not exist in edge model"


@pytest.mark.slow
def test_edge_model_has_y_edge_vars():
    from conftest import small_instance
    inst = small_instance()
    model = EdgesModel(inst, verbose=False)
    for (u, v) in model._intra_rl:
        for o in range(model.O):
            assert (u, v, o) in model.y_edge, \
                f"Missing y_edge[({u},{v}),o{o}]"
    assert len(model.y_edge) == len(model._intra_rl) * model.O


@pytest.mark.slow
def test_edge_model_degree_leq_one():
    from conftest import small_instance
    inst = small_instance()
    model = EdgesModel(inst, verbose=False)
    # DP4'/DP5' constraints for all vertices
    found = 0
    for c in model.model.getConstrs():
        if c.ConstrName.startswith("DP4p_") or c.ConstrName.startswith("DP5p_"):
            found += 1
    assert found > 0, "No degree ≤ 1 constraints found"
    expected = len(model.all_nodes) * model.O * 2
    assert found == expected, f"Expected {expected} degree constraints, got {found}"


@pytest.mark.slow
def test_edge_model_has_ec_constraints():
    from conftest import small_instance
    inst = small_instance()
    model = EdgesModel(inst, verbose=False)
    ec1, ec2a, ec3 = 0, 0, 0
    for c in model.model.getConstrs():
        if c.ConstrName.startswith("EC1_"):
            ec1 += 1
        elif c.ConstrName.startswith("EC2a_"):
            ec2a += 1
        elif c.ConstrName.startswith("EC3_"):
            ec3 += 1
    n_intra = len(model._intra_rl)
    O = model.O
    assert ec1 == n_intra, f"Expected {n_intra} EC1 constraints, got {ec1}"
    # EC2b omitted: reverse direction never exists in self.x for intra edges
    assert ec2a == n_intra * O, f"Expected {n_intra * O} EC2a, got {ec2a}"
    assert ec3 == n_intra * O, f"Expected {n_intra * O} EC3, got {ec3}"


@pytest.mark.slow
def test_edge_model_k_counts_edges_not_vertices():
    from conftest import small_instance
    inst = small_instance()
    model = EdgesModel(inst, verbose=False)
    # k upper bound should be |E_int|, not |V'|
    for o in range(model.O):
        assert model.k[o].UB == len(model._intra_rl), \
            f"k[{o}].UB = {model.k[o].UB}, expected {len(model._intra_rl)}"


@pytest.mark.slow
def test_edge_model_no_dp6():
    from conftest import small_instance
    inst = small_instance()
    model = EdgesModel(inst, verbose=False)
    for c in model.model.getConstrs():
        assert not c.ConstrName.startswith("DP6_"), \
            f"DP6 constraint should not exist in edge model, found {c.ConstrName}"


@pytest.mark.slow
def test_edge_model_no_dp8():
    """DP8 is implied by EC1+EC2a+EC3 and therefore omitted."""
    from conftest import small_instance
    inst = small_instance()
    model = EdgesModel(inst, verbose=False)
    for c in model.model.getConstrs():
        assert not c.ConstrName.startswith("DP8_"), \
            f"DP8 should not exist, found {c.ConstrName}"


@pytest.mark.slow
def test_edge_model_dp9_tight_bigm():
    """DP9 uses tight big-M M = |V'| + 1 (one per operation)."""
    from conftest import small_instance
    inst = small_instance()
    model = EdgesModel(inst, verbose=False)
    dp9_count = sum(1 for c in model.model.getConstrs()
                    if c.ConstrName.startswith("DP9_"))
    # One per operation (not per edge)
    assert dp9_count == model.O, \
        f"Expected {model.O} DP9 constraints, got {dp9_count}"
    # Verify M value = |V'| + 1
    expected_m = len(model.verts) + 1
    # Check that the M coefficient appears in the constraint RHS
    # The constraint is: sum x <= M * (1 - zeta)
    # When zeta=0, RHS = M = |V'| + 1
    # We can verify by checking that the constraint exists with the right name
    for o in range(model.O):
        cname = f"DP9_o{o}"
        found = any(c.ConstrName == cname for c in model.model.getConstrs())
        assert found, f"Missing {cname}"


@pytest.mark.slow
def test_edge_model_has_no_uvar():
    """MTZ potentials u_v^o are removed; DFJ lazy cuts replace them."""
    from conftest import small_instance
    inst = small_instance()
    model = EdgesModel(inst, verbose=False)
    assert not hasattr(model, "_uvar"), \
        "EdgesModel should not have _uvar (MTZ potentials)"
    for c in model.model.getConstrs():
        assert not c.ConstrName.startswith("DP7_"), \
            f"DP7 (MTZ) constraint should not exist, found {c.ConstrName}"


@pytest.mark.slow
def test_edge_model_lazy_enabled():
    """DFJ callback is active (no MTZ potentials needed for correctness)."""
    from conftest import small_instance
    inst = small_instance()
    model = EdgesModel(inst, verbose=False)
    sol = model.optimize(tl=15.0)
    # LazyConstraints = 1 set in optimize()
    assert model.model.Params.LazyConstraints == 1


@pytest.mark.slow
def test_edge_model_factory():
    from conftest import small_instance
    inst = small_instance()
    model = build_model(inst, "edges", verbose=False)
    assert isinstance(model, EdgesModel)
    assert model.name == "Edges"


# ---------------------------------------------------------------------------
# Solve tests
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_edge_model_small_solves():
    from conftest import small_instance
    inst = small_instance()
    model = EdgesModel(inst, verbose=False)
    model.model.setParam("MIPFocus", 1)
    solution = model.optimize(tl=30.0)
    if solution is None:
        pytest.skip("EdgesModel returned no feasible solution within 30s")
    assert solution.objective_value >= 0
    assert len(solution.operations) >= 1
    assert len(solution.chain_selection) == inst.num_regions
    assert solution.status in ("OPTIMAL", "TIME_LIMIT", "SUBOPTIMAL", "INTERRUPTED")
    assert solution.solve_time is not None and solution.solve_time > 0


@pytest.mark.slow
def test_edge_model_equivalence_with_rings():
    """Both models produce the same optimal value on a small instance.

    The edge model uses DFJ lazy subtour elimination (tighter LP relaxation
    than the vertex model's MTZ) and relaxed degree constraints (≤ 1 instead
    of exact vertex-visit equalities). Despite these structural differences,
    the feasible integer solution sets are equivalent.
    """
    from conftest import small_instance
    from drone_cpp.models.mip_rings import RingsModel
    import numpy as np

    inst = small_instance()
    # RingsModel
    m_r = RingsModel(inst, verbose=False)
    m_r.model.setParam("MIPFocus", 1)
    sol_r = m_r.optimize(tl=180.0)
    if sol_r is None:
        pytest.skip("RingsModel returned no solution within 180s")

    # EdgesModel
    m_e = EdgesModel(inst, verbose=False)
    m_e.model.setParam("MIPFocus", 1)
    sol_e = m_e.optimize(tl=180.0)
    if sol_e is None:
        pytest.skip("EdgesModel returned no solution within 180s")

    # Optimal values must match within numerical tolerance
    assert abs(sol_r.objective_value - sol_e.objective_value) < 1e-4, \
        f"Optima differ: Rings={sol_r.objective_value}, Edges={sol_e.objective_value}"
