import sys, os, csv, json, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
sys.path.insert(0, '.')
from drone_cpp.instance_generator import InstanceGenerator
from drone_cpp.data_structures import (
    Point3D, Segment, PolygonalChain, Region, Instance, DroneParams
)
from drone_cpp.model import CPPModel
from drone_cpp.visualization import CPPVis

base_inst = InstanceGenerator.generate(
    num_regions=3, area_bounds=(0, 0, 100, 100),
    min_region_size=12.0, max_region_size=25.0, spacing=6.0,
    num_heights=3, min_height=15.0, max_height=65.0, num_ops=3, seed=42
)

inst = InstanceGenerator.rebuild_instance(base_inst)

os.makedirs("sweep_results", exist_ok=True)
summary_rows = []

endurance_values = list(range(60, 121, 10))

for end in endurance_values:
    print(f"\n{'='*60}")
    print(f"ENDURANCE = {end}")
    print(f"{'='*60}")

    limited_drone = DroneParams(
        front_area=base_inst.drone.front_area,
        drag_coef=base_inst.drone.drag_coef,
        max_endurance=float(end),
        cruise_speed=base_inst.drone.cruise_speed,
        vertical_speed=base_inst.drone.vertical_speed
    )

    inst = Instance(regions=inst.regions, depot=base_inst.depot, drone=limited_drone,
                    wind=base_inst.wind, num_operations=base_inst.num_operations)

    model = CPPModel(inst, verbose=True)
    model.model.setParam("MIPGap", 0.1)
    t0 = time.time()
    solution = model.optimize(tl=600.0)
    elapsed = time.time() - t0
    gap = model.model.MIPGap
    node_count = model.model.NodeCount
    status = model.model.Status

    row = {"endurance": end, "elapsed_s": f"{elapsed:.1f}", "gap_pct": f"{gap*100:.2f}",
           "nodes": node_count, "status": status}

    if solution is None:
        print(f"  No feasible solution for endurance={end}")
        row.update({"obj": "-", "n_ops": "-", "ops_detail": "-", "chain_sel": "-",
                     "total_chain_len": "-"})
        summary_rows.append(row)
        json.dump(row, open(f"sweep_results/end_{end:03d}_data.json", "w"), indent=2)
        continue

    obj = solution.objective_value
    n_ops = len(solution.operations)
    ops_detail = [(oi, len(op.edges)) for oi, op in enumerate(solution.operations)]
    chain_sel = {f"R{r_id}": f"ch{ti}" for r_id, ti in solution.chain_selection.items()}

    total_chain_len = 0.0
    for r_id, ti in solution.chain_selection.items():
        ch = inst.regions[r_id].chains[ti]
        total_chain_len += ch.total_length

    print(f"  Objective: {obj:.2f} m, Operations: {n_ops}, Gap: {gap*100:.2f}%, Time: {elapsed:.1f}s")

    solution.save(f"sweep_results/end_{end:03d}_sol.json")
    print(f"  Saved sweep_results/end_{end:03d}_sol.json")

    fig2d = CPPVis.plot_solution_2d(
        inst, solution, figsize=(9, 7),
        title=f'Endurance={end}J  |  {n_ops} op(s), {obj:.1f}m')
    fig2d.savefig(f"sweep_results/endurance_{end:03d}.png", dpi=300, bbox_inches='tight')
    plt.close(fig2d)
    print(f"  Saved sweep_results/endurance_{end:03d}.png")

    fig3d = CPPVis.plot_solution_3d(
        inst, solution, figsize=(10, 8),
        title=f'3D - Endurance={end}J  |  {n_ops} op(s), {obj:.1f}m')
    fig3d.savefig(f"sweep_results/endurance_{end:03d}_3d.png", dpi=300, bbox_inches='tight')
    plt.close(fig3d)
    print(f"  Saved sweep_results/endurance_{end:03d}_3d.png")

    row.update({"obj": f"{obj:.1f}", "n_ops": n_ops,
                "ops_detail": json.dumps(ops_detail), "chain_sel": json.dumps(chain_sel),
                "total_chain_len": f"{total_chain_len:.1f}"})
    summary_rows.append(row)
    data = {**row,
            "edges_per_op": [[str(e) for e in op.edges] for op in solution.operations],
            "total_chain_len_m": total_chain_len}
    json.dump(data, open(f"sweep_results/end_{end:03d}_data.json", "w"), indent=2)

with open("sweep_results/summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["endurance", "n_ops", "obj", "gap_pct", "elapsed_s",
                                        "nodes", "status", "ops_detail", "chain_sel",
                                        "total_chain_len"])
    w.writeheader(); w.writerows(summary_rows)
print(f"  Saved sweep_results/summary.csv")
print(f"\n{'='*60}")
print("Done! Results in sweep_results/")
print(f"{'='*60}")
