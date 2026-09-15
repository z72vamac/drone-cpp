# Project Report: Drone Coverage Path Planning (CPP)

**Date:** September 2026 · **Repo:** `z72vamac/drone-cpp` (`master` branch, up to date)
**Authors:** Lavinia Amorosi, Paolo Dell'Olmo, Justo Puerto, **Carlos Valverde**
**Solver:** Gurobi 11.0.1 (academic license) · **Stack:** Python (NumPy, matplotlib, gurobipy, pytest)

---

## 1. Executive summary

We developed a **MIQP formulation for drone coverage path planning** over disjoint polygonal regions: one endurance-limited drone departs from a depot, covers spiral chains at different altitudes, and returns to swap batteries, in minimum distance/time. On top of the base model we built **three original extensions**:

1. **Continuous split points per ring (K)** — each ring can be divided into K arcs with free (non-coincident) entry/exit points, coverable in different operations.
2. **Wind-aware model** — real energy in joules with per-heading airspeed (`|cruise·t − w(h)|`), density `rho(h)` and wind power law.
3. **Visual demo suite** — central + 4-satellite instance (square and irregular), sweeps over endurance and K, and segment-by-segment animations.

**Headline results:**
- K=2 improves the K=1 optimum by 1.5–3.8% and makes feasible instances that are impossible for K=1 (E≤180).
- Wind (8 m/s vs calm) **does not change the optimal tours** of the showcase: geometry and endurance dominate.
- Practical knee at **K=2–3**; K≥4 only pays off with huge budgets; K=5 is counterproductive at equal budget.
- In-house heuristic (<0.6 s) + exact warm starts that accelerate the first incumbent 10–100×.

---

## 2. Problem and base model

**Problem.** Cover a set of disjoint polygonal regions with a single endurance-limited drone. Each region admits several candidate **chains** (spirals at different altitude/resolution); the drone flies **operations** (depot→coverage→depot tours). Unlike classic two-stage approaches (first cover each region whole, then route), interrupting a region's coverage to visit another (**preemption**) is allowed, and provably strictly better (triangle + satellites example, gap `l/6`).

**Vertex-based formulation (MIQP).** Four constraint families + objective (total distance):
- *(i) Location* (`Loc-C`): each vertex lies on a segment of the chosen chain (binary `μ` × continuous position `γ`, linearised with Fortet via `η`; global parametrisation `λ`; chain selection `α`).
- *(ii) Routing* (`D_path-C`): flow per operation, visit-once per vertex, **MTZ** subtour elimination + valid inequalities (idle-operation monotonicity, `k`-counting).
- *(iii) Distances* (`Dist-C`): compact intra-ring form (accumulated `dfs` variable, avoids bilinear `μβ` products), inter-region as second-order cone (SOC) horizontal + linear vertical, depot legs.
- *(iv) Energy and endurance*: pseudo-energy (metres on coverage/transits; aerodynamic factor on depot legs), big-M linearisation of the `x·w` product, `Wmax` budget per operation.

**Equivalent edge-based reformulation.** Replaces vertex visits by edge coverage with **DFJ lazy-cut** subtour elimination (callback). Same optimum, different relaxation/structure trade-off. Full 540-config vertex-vs-edge comparison published in the paper: mean parity 1.023 cold (n=311) / 1.003 warm (n=528); warm-started formulations agree within 1% even at $n_r=10$; edge-based cold starts are slower on small instances but find better incumbents on large ones (with selection bias: it solves only 2/90 cold runs at $n_r=10$).

**Multi-phase heuristic (<0.6 s).** Proxy-cost chain selection → giant tour (nearest-neighbour + expansion) → greedy endurance-aware split → coordinate-descent entry-point optimisation → intra-operation 2-opt + inter-operation Or-opt. Used as a **warm start** for the exact solver.

---

## 3. Extension 1 — K continuous split points per ring

**Motivation.** The original model only admits **one coincident** entry/exit point per ring (indivisible full loop). The new `num_split_points` (K) parameter allows K continuous points per ring.

**Design** (`RingsModel(..., num_split_points=K)`; K=1 reproduces the original behaviour):
- Each ring is partitioned into **K arcs in series with anchors** (launch_1 at 0, retrieve_j = launch_{j+1}, retrieve_K at the selected chain's ring end); zero-length arcs allowed (K is an upper bound).
- **Zero-cost connectors** in both directions (including the retrieve_K ↔ launch_1 wrap) to chain consecutive arcs inside one operation.
- Arc lengths via `dfs` (pseudo) or exact I-formulation (wind-aware); each arc covered exactly once in either direction (DP8).
- Supporting theory: anchoring is lossless up to one extra arc (any tour with m cuts replicates with K = m+1), and models are **nested** (K ⊂ K+1 via degenerate arcs) → non-increasing optima in K.
- CLI: `solve_and_plot.py --split-points K`.

**Experimental result (number of points needed).** K=1…5 sweep:
- *Showcase* (E=150, no wind): 708.0 (K=1 OPT) → **697.2** (K=2) → 700.1 (K=3) → 708.0 (K=4). **Knee at K\*=2** (−1.5%); 3rd/4th points add nothing within solver reach.
- *Wind-aware demo* E=173: 677.6 → **652.09 OPTIMAL (2%)** → 630.88 → 621.66 → 624.9 (K=5 needs 8 h for 624.9). Practical knee K=2–3.
- *Irregular demo* E=153: 776.13 OPT → 581.95 → **559.71** (K=3 and K=4 tie) → K=5 collapses (1508–1312, 67% gap).
- Verified mechanism: K=2 **decouples entry/exit** (K=1 forces entry=exit at the same point) and interleaves arcs; cuts land on genuinely fractional points (λ=2.05, 3.61, 4.49…), not vertices.

---

## 4. Extension 2 — wind-aware model (real joules)

**Motivation.** The base model is nearly wind-blind (wind enters only the depot legs, and energy mixes units). A full mode was implemented on request, optional (`wind_aware=False` by default; all previous results untouched).

**Design** (enabled with `build_model(..., wind_aware=True)` / `--wind-aware`):
- Constant-direction wind, **power-law** speed `w(h)` (hellmann 0.2).
- **Exact per-heading intra-ring arcs**: `C = E_xy·rho(h)·|cruise·t − w(h)|` precomputed per segment and direction; exact arc energies via I-formulation (overlap indicator + partials via `η`), with different values per traversal direction.
- **Inter-region legs**: precomputed direction between region centroids (paper approximation), per-departure-chain coefficients, bilinearities discretised exactly with Fortet.
- **Depot**: same fixed depot→centroid direction but with per-chain `rho(h_t)`/`w(h_t)`.
- Objective untouched (distance); only the endurance-constraint energy changes → **E recalibration** required (showcase E_min: 153 pseudo-J → 68.7 real J).

**Validation results (showcase, `scripts/validate_wind.py`):**
| Case | Obj | Ops | Tours |
|------|-----|-----|-------|
| WA K=1 tight (104 J), wind 8 m/s E | 594.66 OPTIMAL | 2 | [[(1,0)], rest] |
| WA K=1 tight, calm | 594.66 OPTIMAL | 2 | identical partition |
| WA K=1 loose (207 J), both winds | 500.16 | 1 | identical |
| WA K=2 tight, wind 8 | 594.66 (holds) | 2 | — |
| No-wind reference (E=230 pseudo) | 611.53 | 2 | different partition |

**Conclusion: wind (8 m/s vs calm) does not change the showcase optimal tours** — geometry/endurance dominate; wind is second-order here. Side measurement: ignoring it underestimates true energy ~8–16% at 8 m/s; pseudo-energy overstates joules ~1.3× (depot) / ~2.3× (coverage).

---

## 5. Visual demos and sweeps

**Central + 4-satellite demo** (`scripts/demo_interruptions.py`, `demo_interruptions/`):
- Central 40×40 square (2 rings) + 4 satellites 9×9 at N/S/E/W + corner depot.
- Key finding: at E≤180 J, **K=1 is INFEASIBLE** (the outer ring never fits whole in one operation) while **K=2 is feasible** (1255.5 m, 4 ops) with both central rings **split across operations** (cuts at λ=1.27 and λ=1.0) — interruptions are necessary, not optional.
- Sweep E=170…400 (K=2): objective 1234→802 m, operations 4→2, interruptions present at every level (peak of 4 central visits at E=180 and E≥350).

**Irregular demo** (`scripts/demo_irregular.py`, `demo_irregular/`): irregular central heptagon + 4 irregular pentagons. K=1…5 sweep at E=153 J with up to 8 h/level: 776.13 → 581.95 → **559.71** → 559.71 → 1311.98. K=5, even with 8 h, ends worse than K=1 (65k-variable model out of reach; the exact warm start stops being accepted at that scale).

**Animations**: per-level PNG sequences + GIFs (`anim_frames_*/`, `demo_anim/`, ~40–55 frames, 0 disconnections verified), embeddable with `\animategraphics` in Beamer. Each frame shows operation and step; plots carry a wind arrow.

---

## 6. Paper and presentation

- **Paper** (`articulo/main.tex`, 27 pages, Elsevier): full formulation, heuristic, experiments with **real mean-based tables** (both sweeps complete: 540 configs × 3 methods each) and 4 comparison figures, including the full vertex-vs-edge analysis (parity 1.023/1.003, solved-count caveats for cold edges on large instances). No TODOs. Compiles cleanly.
- **Conference talk** (`articulo/congreso_beamer.tex` → 26-page PDF, 16:9 Madrid/whale, in English): animated 150 J/600 J showcase after the formulation, one equation family per slide, mean-based results, backup removed. Timing guide in `articulo/GUIA_PRESENTACION.md` (16'00'' for a 20-min slot).
- Proposed, not done: document K and wind-aware in `implemented_model.tex`, add a with/without-wind slide to the deck.

## 7. Relevant bugs fixed (all with test or verification)

1. Disconnected animation: edges stored without chronological order → topological chaining (`chain_edges`); loops drawn from ring vertex 0 → `ring_full_loop` from the actual entry point.
2. Double-drawn rings: the wrap connector rendered as a full ring → skipped when endpoints coincide.
3. Rejected warm starts: reading `.Start` without `update()` returns `GRB.UNDEFINED` (1e101); `y=0` at depot violated DP4 (now `y` = out-degree); swapped start/end partials in backward energies (gave negatives); id()-keyed rewrites on rebuilt tuples (never applied); `set_warm_start` read λ in metres (heuristic) vs segment units (model) → `lambda_in_meters` flag.
4. Helper `_complete_starts()`: fills every auxiliary Start consistently (with `update()` between phases).

## 8. How to reproduce / repo state

- `master` branch in sync with GitHub (`ad5ba9e`). Tests: 101 fast + 3 wind-aware OK.
- Key commands:
  - `python scripts/demo_irregular.py --ks 5 --warm-from demo_irregular/irr_E0153_K5.json --time-limit 86400 --threads 32` (K=5, 24 h, multi-core)
  - `python solve_and_plot.py --model rings --split-points 2 --wind-aware --endurance 150`
  - `python scripts/validate_wind.py`, `python scripts/sweep_demo_wind_k.py --ks 1,2,3,4`
- Suggested pending work: document the extensions in the paper, add the wind slide, final commit of K=5 8 h results (already saved on disk).
