import sys
import numpy as np
from .instance_generator import InstanceGenerator
from .model import CPPModel
from .visualization import CPPVis
from .data_structures import Solution


def run_example(num_regions: int = 3, time_limit: float = 120.0, seed: int = 42):
    print("=" * 60)
    print("Drone Coverage Path Planning - MIQP Formulation")
    print("=" * 60)

    print(f"\n[1/4] Generating instance with {num_regions} regions...")
    inst = InstanceGenerator.generate(
        num_regions=num_regions,
        area_bounds=(0, 0, 100, 100),
        min_region_size=12.0,
        max_region_size=25.0,
        spacing=6.0,
        num_ops=num_regions,
        seed=seed
    )
    print(f"  Regions: {inst.num_regions}")
    for r in inst.regions:
        print(f"    R{r.id}: {len(r.boundary)} vertices, "
              f"{len(r.chains)} chain(s)")
    print(f"  Drone endurance: {inst.drone.max_endurance}")
    print(f"  Wind: {inst.wind.speed_at_10m:.1f} m/s at 10m")

    print(f"\n[2/4] Building Gurobi model...")
    try:
        model = CPPModel(inst, verbose=True)
        print(f"  Model built: {model.model.NumVars} variables, "
              f"{model.model.NumConstrs} constraints")
    except Exception as e:
        print(f"  ERROR building model: {e}")
        sys.exit(1)

    print(f"\n[3/4] Solving (time limit: {time_limit}s)...")
    solution = model.optimize(time_limit=time_limit)

    if solution is None:
        print("  No feasible solution found.")
        return

    print(f"\n[4/4] Solution found!")
    print(f"  Objective value: {solution.objective_value:.2f}")
    for op_idx, op in enumerate(solution.operations):
        print(f"  Operation {op_idx}: {len(op.edges)} edges")
    print(f"  Chain selection:")
    for r_id, t_idx in solution.chain_selection.items():
        h = inst.regions[r_id].chains[t_idx].height
        print(f"    Region {r_id} -> chain {t_idx} (h={h:.1f}m)")

    sol_path = f"solution_{num_regions}r.sol"
    solution.save(sol_path)
    print(f"\n  Solution saved to {sol_path}")

    print(f"\nGenerating plots...")
    fig1 = CPPVis.plot_instance(inst)
    fig1.savefig("instance.png", dpi=150)
    print("  Saved instance.png")

    fig2 = CPPVis.plot_solution(inst, solution)
    fig2.savefig("solution.png", dpi=150)
    print("  Saved solution.png")

    CPPVis.show()


if __name__ == "__main__":
    num_regions = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    time_limit = float(sys.argv[2]) if len(sys.argv) > 2 else 120.0
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
    run_example(num_regions, time_limit, seed)
