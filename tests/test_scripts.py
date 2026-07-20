"""Integration tests for the CLI entry-point scripts.

These tests verify that argument parsing works and that the main pipeline
functions without errors. The actual Gurobi solve is skipped when no license
is available (the model-building step *is* tested here since it does not
require a license).
"""
from __future__ import annotations
import sys
import os
import json
import tempfile
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _clean_argv():
    """Save and restore sys.argv so each test has clean state."""
    saved = sys.argv[:]
    sys.argv = [sys.argv[0]]
    yield
    sys.argv = saved


# ---------------------------------------------------------------------------
# solve_and_plot.py
# ---------------------------------------------------------------------------
class TestSolveAndPlotArgs:
    def test_defaults(self):
        sys.argv = ["solve_and_plot.py"]
        from solve_and_plot import parse_args
        args = parse_args()
        assert args.endurance == 100.0
        assert args.time_limit == 120.0
        assert args.mip_gap == 0.1
        assert args.seed == 42
        assert args.out == "cone_solution.json"
        assert args.plot_prefix == "cone_solution"
        assert not args.skip_plots

    def test_custom_args(self):
        sys.argv = ["solve_and_plot.py",
                    "--endurance", "200",
                    "--time-limit", "60",
                    "--mip-gap", "0.05",
                    "--seed", "99",
                    "--out", "custom.json",
                    "--plot-prefix", "custom",
                    "--skip-plots"]
        from solve_and_plot import parse_args
        args = parse_args()
        assert args.endurance == 200.0
        assert args.time_limit == 60.0
        assert args.seed == 99
        assert args.out == "custom.json"
        assert args.plot_prefix == "custom"
        assert args.skip_plots

    def test_build_instance_no_errors(self):
        from solve_and_plot import build_instance
        inst = build_instance(endurance=100.0, seed=42)
        assert inst.num_regions == 3
        assert inst.drone.max_endurance == 100.0

    def test_skip_plots_flag(self):
        sys.argv = ["solve_and_plot.py", "--skip-plots"]
        from solve_and_plot import parse_args
        args = parse_args()
        assert args.skip_plots


# ---------------------------------------------------------------------------
# sweep_endurance.py
# ---------------------------------------------------------------------------
class TestSweepEnduranceArgs:
    def test_defaults(self):
        sys.argv = ["sweep_endurance.py"]
        from sweep_endurance import parse_args
        args = parse_args()
        assert args.start == 60
        assert args.stop == 121
        assert args.step == 10
        assert args.time_limit == 600.0
        assert args.seed == 42

    def test_custom_range(self):
        sys.argv = ["sweep_endurance.py",
                    "--start", "50", "--stop", "100", "--step", "5"]
        from sweep_endurance import parse_args
        args = parse_args()
        assert args.start == 50
        assert args.stop == 100
        assert args.step == 5

    def test_build_base(self):
        from sweep_endurance import build_base
        inst = build_base(seed=42)
        assert inst.num_regions == 3


# ---------------------------------------------------------------------------
# compare_endurance.py
# ---------------------------------------------------------------------------
class TestCompareEnduranceArgs:
    def test_defaults(self):
        sys.argv = ["compare_endurance.py"]
        from compare_endurance import parse_args
        args = parse_args()
        assert args.endurance_high == 3000.0
        assert args.endurance_limited == 62.0
        assert args.seed == 42

    def test_custom_values(self):
        sys.argv = ["compare_endurance.py",
                    "--endurance-high", "5000",
                    "--endurance-limited", "100",
                    "--seed", "7"]
        from compare_endurance import parse_args
        args = parse_args()
        assert args.endurance_high == 5000.0
        assert args.endurance_limited == 100.0
        assert args.seed == 7
