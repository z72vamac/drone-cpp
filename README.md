# Drone Coverage Path Planning

Mixed-Integer (Mixed-Quadratic) Programming formulation for the **Drone Coverage
Path Planning** problem: planning the routes of a fleet of drones that must
cover a set of polygonal regions by flying along precomputed spiral chains at
different altitudes, with limited endurance and wind-aware energy consumption.

Companion code for the work *Drone Coverage Path Planning*
(Amorosi, Dell'Olmo, Puerto, Valverde). See
[`implemented_model.tex`](implemented_model.tex) for the documentation of the
implemented model and [`main.tex`](main.tex) for the paper manuscript.

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
main.tex                Paper manuscript (Elsevier format)
implemented_model.tex   Model documentation comparing code vs formulation
sweep_results/          Output of sweep_endurance.py (JSON + PNG)
IFORS/                  IFORS presentation (LaTeX + frames)
```

## Energy model

The energy of a flight segment is `E = ½ · c · A · ρ(h) · ν_d · d`, where `c`
is the drag coefficient, `A` the frontal area, `ρ(h)` the air density at height
`h` (barometric formula via `AtmosphereParams`), and `ν_d` the drone's
ground-projected speed (cruise speed projected against the wind vector). The
endurance bound `E_max` (in Joules) limits the total energy per operation.

## Tests

```bash
pytest -m "not slow"     # fast smoke tests, no Gurobi required
pytest -m slow           # build + solve small instance (~10 s), Gurobi required
pytest                   # everything
```

## Reproducibility

All random sources use a single `numpy.random.RandomState(seed)`. Instance
generation, depot placement and wind direction/speed are derived from it, so a
fixed `seed` produces an identical instance across runs and machines.