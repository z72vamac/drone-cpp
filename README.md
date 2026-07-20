# Drone Coverage Path Planning

Mixed-Integer (Mixed-Quadratic) Programming formulation for the **Drone Coverage
Path Planning** problem: planning the routes of a fleet of drones that must
cover a set of polygonal regions by flying along precomputed spiral chains at
different altitudes, with limited endurance and wind-aware energy consumption.

Companion code for the work *Drone Coverage Path Planning*
(Amorosi, Dell'Olmo, Puerto, Valverde). See
[`articulo/implemented_model.tex`](articulo/implemented_model.tex) for the
documentation of the implemented model and
[`articulo/main.tex`](articulo/main.tex) for the paper manuscript.

## Installation

```bash
pip install -r requirements.txt
```

This installs `numpy`, `matplotlib`, `gurobipy` and `pytest`. **A valid Gurobi
license is required** to run the optimization (the MIP/MIQP is solved with
Gurobi). The visualization and instance-generation parts work without a
license.

## Quick start

Build an instance, solve it with limited endurance and save the plots:

```bash
python solve_and_plot.py --endurance 100 --time-limit 120 --seed 42
```

This writes `cone_instance.png`, `cone_solution.png` and `cone_solution.json`.

Other entry points:

| Script | Purpose |
|---|---|
| `solve_and_plot.py` | Generate one instance, solve it, save solution and plots. |
| `sweep_endurance.py` | Sweep endurance values 60-120 J, save JSON + 2D/3D PNG per level. |
| `compare_endurance.py` | Solve the same instance at two endurance levels and compare. |

## Package layout

```
drone_cpp/
  __init__.py           Public API exports
  config.py             Default constants (seed, area bounds, heights, ...)
  data_structures.py    Point3D, Segment, PolygonalChain, Region, Vertex,
                       Instance, Solution, Operation, drone/wind params
  instance_generator.py InstanceGenerator.generate / rebuild_instance
  model.py              CPPModel — Gurobi MIQP model (location, path, energy)
  visualization.py      CPPVis — 2D/3D plotting with arrows along chains
articulo/               LaTeX sources, figures and bibliography
  main.tex              Paper manuscript (Elsevier format)
  implemented_model.tex Model documentation comparing code vs formulation
  presentation.tex      Beamer deck with animations across endurance values
  mybib.bib             Bibliography
  pictures/             Figures referenced by main.tex
  tables/               LaTeX \input table fragments
sweep_results/          Output of sweep_endurance.py (JSON + PNG)
```

## Energy model

The energy of a flight segment is `E = ½ · c · A · ρ(h) · ν_d · d`, where `c`
is the drag coefficient, `A` the frontal area, `ρ(h)` the air density at height
`h` (barometric formula via `AtmosphereParams`), and `ν_d` the drone's
ground-projected speed (cruise speed projected against the wind vector). The
endurance bound `E_max` (in Joules) limits the total energy per operation.

## Tests

```bash
pytest -m "not slow"     # fast smoke tests (no Gurobi required)
pytest -m slow           # slow tests (Gurobi required)
pytest                   # all tests
```

Test files:

| File | Scope |
|---|---|
| `tests/test_smoke.py` | Instance generation, rebuild, solution roundtrip, basic visualization |
| `tests/test_data_structures.py` | Point3D, Segment, PolygonalChain, AtmosphereParams, Vertex/Edge, Drone/Wind params |
| `tests/test_instance_generator.py` | Spiral segments, convex polygon, overlap detection, edge building |
| `tests/test_visualization.py` | lambd_to_cum, get_chain_path, edge classification, plot edge cases |
| `tests/test_model.py` | CPPModel structure, variable counts, solve (slow) |
| `tests/test_scripts.py` | CLI argument parsing, default/custom values |

## Solution file format

`solve_and_plot.py` writes a JSON file with the following structure:

```json
{
  "objective_value": 1234.5,
  "chain_selection": { "0": 0, "1": 1 },
  "vertex_positions": { "0,1,START": { "x": 10.0, "y": 20.0, "z": 15.0 }, ... },
  "vertex_lambdas": { "0,1,START": 0.0, ... },
  "operations": [ [ { "r": 0, "i": 1, "t": "START", "r2": 0, "i2": 2, "t2": "LAUNCH" }, ... ] ],
  "solve_time": 1.25,
  "mip_gap": 0.01,
  "status": "OPTIMAL"
}
```

- `objective_value`: total distance in meters.
- `chain_selection`: for each region ID, the index of the chosen spiral chain.
- `vertex_positions`: maps `"region_id,idx,type"` → 3D coordinates.
- `vertex_lambdas`: maps `"region_id,idx,type"` → fractional segment index.
- `operations`: list of operations, each with an ordered list of edges.
  Each edge references a source vertex (`r`, `i`, `t`) and a target vertex
  (`r2`, `i2`, `t2`).
- `solve_time`: Gurobi runtime in seconds (may be `null` for legacy files).
- `mip_gap`: relative MIP optimality gap (may be `null`).
- `status`: solver termination status (`OPTIMAL`, `TIME_LIMIT`, `SUBOPTIMAL`, ...).

## Reproducibility

All random sources use a single `numpy.random.RandomState(seed)`. Instance
generation, depot placement and wind direction/speed are derived from it, so a
fixed `seed` produces an identical instance across runs and machines.

## Development status

### Implemented improvements

- **Unit tests**: comprehensive tests for `data_structures.py`,
  `instance_generator.py`, `visualization.py`, `scripts`. Shared helpers
  (`make_instance`, `mock_solution`, `small_instance`) live in `conftest.py`.
- **.gitignore**: `*.pdf` added to ignore list, exempting `articulo/*.pdf`;
  `!pictures/*.png` corrected to `!articulo/pictures/*.png`.
- **instance_params.txt**: moved to `articulo/`.
- **CI/CD**: `.github/workflows/tests.yml` runs `pytest -m "not slow"` on
  Python 3.10–3.12 on every push/PR.
- **Linter config**: `pyproject.toml` with `ruff` and `black` settings.
- **CLI**: `solve_and_plot.py` gains `--skip-plots` to batch without I/O.
- **Solution JSON documentation**: this README section above.

### Remaining / future

- `articulo/implemented_model.pdf` is outdated vs the `.tex`; recompile with
  `pdflatex` inside `articulo/`.
- 3D region fill uses `Poly3DCollection` (already) but `mpl_toolkits.mplot3d`
  does not support `fill_between`; consider `mayavi` or `plotly` for richer
  3D visuals.
- No `isort`/`black` pre-commit hook; CI only lints on push.