"""Tests for the wind-aware RingsModel (structure only, no solve).

Run with:  pytest -m slow  (requires Gurobi)
"""
from __future__ import annotations
import pytest


def _showcase():
    import sys
    sys.path.insert(0, ".")
    from sweep_endurance import build_base
    return build_base(42)


@pytest.mark.slow
def test_wind_aware_builds_k1():
    from drone_cpp.model import build_model
    inst = _showcase()
    m = build_model(inst, "rings", verbose=False, num_split_points=1,
                    wind_aware=True)
    assert m.wind_aware is True
    assert len(m._inter_edges) > 0
    # full-loop wind energies are linear in alpha (no I variables needed)
    assert len(m._I) == 0
    # forward/backward loop energies differ under wind (asymmetric airspeed)
    assert len(m._seg_C) > 0


@pytest.mark.slow
def test_wind_aware_builds_k2():
    from drone_cpp.model import build_model
    inst = _showcase()
    m = build_model(inst, "rings", verbose=False, num_split_points=2,
                    wind_aware=True)
    assert m.K == 2
    assert len(m._I) > 0
    # every arc/chain/segment has an indicator
    for (u, v) in m._intra_rl:
        r = m.inst.regions[u.region_id]
        ri = m._ring_index[u]
        for ti, ch in enumerate(r.chains):
            info = m._ring_info.get((r.id, ti, ri))
            if info is None:
                continue
            for si in range(info["num_seg"]):
                assert (u, v, ti, si) in m._I
    # connectors exist (including wrap) with zero energy
    assert len(m._connectors) == sum(
        2 * len(a) for a in m._ring_arcs.values())


@pytest.mark.slow
def test_wind_aware_backward_energy_nonnegative():
    """Regression test: backward arc energies must be >= 0.

    The start/end partials were once swapped, producing negative energies
    (caught via an out-of-bounds Start of -0.47 on a backward edge).
    """
    from drone_cpp.model import build_model
    from drone_cpp.data_structures import Instance, DroneParams, WindParams
    import numpy as np
    inst = _showcase()
    wind = WindParams(direction=np.array([1.0, 0.0, 0.0]), speed_at_10m=8.0,
                      hellmann_exponent=0.2)
    drone = DroneParams(front_area=inst.drone.front_area,
                        drag_coef=inst.drone.drag_coef, max_endurance=104.0,
                        cruise_speed=inst.drone.cruise_speed,
                        vertical_speed=inst.drone.vertical_speed)
    inst2 = Instance(regions=inst.regions, depot=inst.depot, drone=drone,
                     wind=wind, num_operations=6)
    m = build_model(inst2, "rings", verbose=False, num_split_points=2,
                    wind_aware=True)
    # all precomputed per-meter coefficients must be non-negative
    for key, (cf, cb) in m._seg_C.items():
        assert cf >= 0 and cb >= 0, key
    for key, ent in m._pair_F.items():
        for ti, (f, vz) in ent.items():
            assert f >= 0 and vz >= 0, (key, ti)
    for key, ent in m._depot_F.items():
        for ti, (fo, fr, vz) in ent.items():
            assert fo >= 0 and fr >= 0 and vz >= 0, (key, ti)
