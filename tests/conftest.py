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


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="Needs --runslow and a Gurobi license")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)