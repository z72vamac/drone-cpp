import sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
sys.path.insert(0, '.')
from drone_cpp.instance_generator import InstanceGenerator
from drone_cpp.data_structures import (
    Point3D, Segment, PolygonalChain, Region, Instance, DroneParams, WindParams,
    Vertex, VertexType, Solution, Operation
)
from drone_cpp.model import CPPModel
from drone_cpp.visualization import CPPVis

# ============================================================
# Build the instance (same for both runs)
# ============================================================
base_inst = InstanceGenerator.generate(
    num_regions=3, area_bounds=(0, 0, 100, 100),
    min_region_size=12.0, max_region_size=25.0, spacing=6.0,
    num_heights=3, min_height=15.0, max_height=65.0, num_ops=3, seed=42
)

def build_instance():
    return InstanceGenerator.rebuild_instance(base_inst)

inst = build_instance()

# ============================================================
# Run 1: High endurance (1 operation)
# ============================================================
print("=" * 60)
print("RUN 1: HIGH ENDURANCE (3000)")
print("=" * 60)
high_drone = DroneParams(
    front_area=base_inst.drone.front_area,
    drag_coef=base_inst.drone.drag_coef,
    max_endurance=3000.0,
    cruise_speed=base_inst.drone.cruise_speed,
    vertical_speed=base_inst.drone.vertical_speed
)
inst_high = Instance(regions=inst.regions, depot=inst.depot, drone=high_drone,
                     wind=inst.wind, num_operations=inst.num_operations)

model_high = CPPModel(inst_high, verbose=False)
sol_high = model_high.optimize(tl=60.0)

if sol_high is None:
    print("  No solution found!")
    sys.exit(1)

print(f"  Objective: {sol_high.objective_value:.2f} m")
print(f"  Operations: {len(sol_high.operations)}")
sol_high.save("solution_high.json")
fig_high = CPPVis.plot_solution(inst_high, sol_high)
fig_high.suptitle(f"High endurance (3000): {sol_high.objective_value:.1f}m, {len(sol_high.operations)} op(s)", fontsize=12)
fig_high.savefig("solution_high.png", dpi=150)
plt.close(fig_high)
print("  Saved solution_high.png")

# ============================================================
# Run 2: Limited endurance (forces 2 operations)
# ============================================================
print("=" * 60)
print("RUN 2: LIMITED ENDURANCE")
print("=" * 60)

# Find endurance that forces exactly 2 operations
# Try 58 (below ~60 which gave 1 op with corrected energy)
target_endurance = 62.0
limited_drone = DroneParams(
    front_area=base_inst.drone.front_area,
    drag_coef=base_inst.drone.drag_coef,
    max_endurance=target_endurance,
    cruise_speed=base_inst.drone.cruise_speed,
    vertical_speed=base_inst.drone.vertical_speed
)
inst_lim = Instance(regions=inst.regions, depot=inst.depot, drone=limited_drone,
                    wind=inst.wind, num_operations=inst.num_operations)

model_lim = CPPModel(inst_lim, verbose=False)
model_lim.model.setParam("MIPFocus", 1)
model_lim.model.setParam("Heuristics", 0.5)
sol_lim = model_lim.optimize(tl=120.0)

if sol_lim is None:
    print(f"  No solution found at endurance={target_endurance}")
    sys.exit(1)

print(f"  Objective: {sol_lim.objective_value:.2f} m")
print(f"  Operations: {len(sol_lim.operations)}")
sol_lim.save("solution_limited.json")
fig_lim = CPPVis.plot_solution(inst_lim, sol_lim)
fig_lim.suptitle(f"Limited endurance ({target_endurance:.0f}): {sol_lim.objective_value:.1f}m, {len(sol_lim.operations)} op(s)", fontsize=12)
fig_lim.savefig("solution_limited.png", dpi=150)
plt.close(fig_lim)
print(f"  Saved solution_limited.png")

print("\nDone! Both solutions saved.")
