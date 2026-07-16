"""Default configuration constants for the Drone CPP package.

These values are used as fallback when the entry-point scripts do not receive
explicit overrides via CLI flags. Keeping them in a single module avoids the
proliferation of magic numbers across `solve_and_plot.py`, `sweep_endurance.py`
and `compare_endurance.py`.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Instance generation
# ---------------------------------------------------------------------------
DEFAULT_SEED: int = 42
DEFAULT_AREA_BOUNDS: tuple = (0.0, 0.0, 100.0, 100.0)
DEFAULT_REGION_SIZES: tuple = (12.0, 25.0)        # (min, max) polygon radius
DEFAULT_NUM_REGIONS: int = 3
DEFAULT_SPACING: float = 6.0                       # chain segment spacing at ref height
DEFAULT_NUM_HEIGHTS: int = 3
DEFAULT_HEIGHT_RANGE: tuple = (15.0, 65.0)         # (min, max) chain altitude
DEFAULT_NUM_OPS: int = 3

# Cone-effect (rebuild_instance) parameters
DEFAULT_BASE_SPACING: float = 5.0
DEFAULT_REF_HEIGHT: float = 15.0
DEFAULT_CONE_HEIGHT: float = 90.0

# ---------------------------------------------------------------------------
# Solver (Gurobi)
# ---------------------------------------------------------------------------
DEFAULT_TIME_LIMIT: float = 120.0         # seconds
DEFAULT_MIP_GAP: float = 0.1              # 10%
DEFAULT_MIP_FOCUS: int = 1               # feasibility focus
DEFAULT_HEURISTICS: float = 0.5

# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
SWEEP_ENDURANCE_RANGE: tuple = (60, 121, 10)   # range(start, stop, step) in Joules

# ---------------------------------------------------------------------------
# Drone defaults (frontal area and drag; endurance is set per-run)
# ---------------------------------------------------------------------------
DEFAULT_FRONT_AREA: float = 0.1
DEFAULT_DRAG_COEF: float = 0.3
DEFAULT_CRUISE_SPEED: float = 15.0        # m/s
DEFAULT_VERTICAL_SPEED: float = 5.0       # m/s
DEFAULT_ENDURANCE: float = 100.0           # J, default for solve_and_plot.py

# ---------------------------------------------------------------------------
# Wind defaults
# ---------------------------------------------------------------------------
DEFAULT_WIND_SPEED_RANGE: tuple = (2.0, 8.0)  # m/s at 10m
DEFAULT_HELLMANN_EXPONENT: float = 0.2