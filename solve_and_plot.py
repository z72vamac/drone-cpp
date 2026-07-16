import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
sys.path.insert(0, '.')
from drone_cpp.instance_generator import InstanceGenerator
from drone_cpp.data_structures import (
    Point3D, Segment, PolygonalChain, Region, Instance, DroneParams, WindParams,
    Vertex, VertexType
)
from drone_cpp.model import CPPModel
from drone_cpp.visualization import CPPVis

base_inst = InstanceGenerator.generate(
    num_regions=3,
    area_bounds=(0, 0, 100, 100),
    min_region_size=12.0,
    max_region_size=25.0,
    spacing=6.0,
    num_heights=3,
    min_height=15.0,
    max_height=65.0,
    num_ops=3,
    seed=42
)

inst = InstanceGenerator.rebuild_instance(base_inst)

# Reduce endurance so drone cannot cover all 3 chains in one operation
limited_drone = DroneParams(
    front_area=base_inst.drone.front_area,
    drag_coef=base_inst.drone.drag_coef,
    max_endurance=100.0,
    cruise_speed=base_inst.drone.cruise_speed,
    vertical_speed=base_inst.drone.vertical_speed
)

inst = Instance(
    regions=inst.regions,
    depot=base_inst.depot,
    drone=limited_drone,
    wind=base_inst.wind,
    num_operations=base_inst.num_operations
)

print(f"Instance with cone-effect chains: {inst.num_regions} regions")
for r in inst.regions:
    for c in r.chains:
        print(f"  R{r.id} h={c.height:.0f}m: {len(c.segments)} segments, "
              f"length={c.total_length:.1f}m")
print(f"  Depot: ({inst.depot.x:.1f}, {inst.depot.y:.1f})")
print(f"  Wind: {inst.wind.speed_at_10m:.1f} m/s")

print("\nBuilding Gurobi MIQP model...")
model = CPPModel(inst, verbose=True)
print(f"  Variables: {model.model.NumVars}, Constraints: {model.model.NumConstrs}")

print("\nSolving (time limit=120s)...")
solution = model.optimize(tl=120.0)

if solution is None:
    print("  No feasible solution found.")
    sys.exit(1)

print(f"\nSolution found!")
print(f"  Objective (total distance): {solution.objective_value:.2f} m")
print(f"  Operations: {len(solution.operations)}")
for oi, op in enumerate(solution.operations):
    print(f"    Op {oi}: {len(op.edges)} edges")
print(f"  Chain selection:")
for r_id, t_idx in solution.chain_selection.items():
    ch = inst.regions[r_id].chains[t_idx]
    print(f"    R{r_id} -> chain {t_idx} (h={ch.height:.0f}m, {len(ch.segments)} segments)")

solution.save("cone_solution.json")
print("\nSolution saved to cone_solution.json")

print("\nGenerating plots...")

fig1 = CPPVis.plot_instance(inst, show_chains=True)
fig1.savefig("cone_instance.png", dpi=150)
plt.close(fig1)
print("  Saved cone_instance.png")

fig2 = CPPVis.plot_solution(inst, solution)
fig2.savefig("cone_solution.png", dpi=150)
plt.close(fig2)
print("  Saved cone_solution.png")

print("\nDone!")
