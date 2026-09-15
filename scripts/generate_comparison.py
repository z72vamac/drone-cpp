#!/usr/bin/env python
"""Generate comparative figures and LaTeX tables for the drone CPP paper.

Reads compare_results/results.csv (Vertex-Approach, complete: 540 configs)
and compare_results_edges/results.csv (Edges-Approach, complete: 540 configs).

Modes:
  default        -> articulo/pictures/compare_*.pdf + 3 tables (paper)
  --vertex-only  -> articulo/pictures/compare_*_vertex.pdf + summary and
                    by-endurance tables only (beamer; no Edges-Approach)

Tables are written as UTF-8 so LaTeX (inputenc utf8) accepts them.
"""
import os, sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VERTEX_ONLY = "--vertex-only" in sys.argv
SUF = "_vertex" if VERTEX_ONLY else ""

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
})

RINGS_CSV = "compare_results/results.csv"
EDGES_CSV = "compare_results_edges/results.csv"
OUT_PICS = "articulo/pictures"
OUT_TABLES = "articulo/tables"

os.makedirs(OUT_PICS, exist_ok=True)
os.makedirs(OUT_TABLES, exist_ok=True)

rings = pd.read_csv(RINGS_CSV)
edges = pd.read_csv(EDGES_CSV)

# Deduplicate re-runs (keep last) so each (config, method) counts once
_key = ["num_regions", "num_heights", "seed", "endurance", "method"]
rings = rings.drop_duplicates(subset=_key, keep="last").reset_index(drop=True)
edges = edges.drop_duplicates(subset=_key, keep="last").reset_index(drop=True)

for col in ["objective", "solve_time", "mip_gap", "first_incumbent_time",
            "heuristic_time", "n_vars", "n_constrs"]:
    if col in rings.columns:
        rings[col] = pd.to_numeric(rings[col], errors="coerce")
    if col in edges.columns:
        edges[col] = pd.to_numeric(edges[col], errors="coerce")


def subset(df, method):
    return df[df["method"] == method].copy()


C_V = "#003366"
C_VWS = "#6699CC"
C_E = "#BA0C2F"
C_EWS = "#E57373"
C_HEU = "#888888"

REGIONS = [1, 2, 3, 5, 8, 10]

# ---------------------------------------------------------------------------
# 1) Mean solve time by n_r
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.2, 3.4))
series = [
    ("heuristic", "Heuristic", C_HEU),
    ("rings", "Vertex-Approach", C_V),
    ("rings_ws", "Vertex-Approach+WS", C_VWS),
]
if not VERTEX_ONLY:
    series += [("edges", "Edges-Approach", C_E),
               ("edges_ws", "Edges-Approach+WS", C_EWS)]

means = {}
for nr in REGIONS:
    for meth, label, color in series:
        if meth == "heuristic":
            sub = subset(rings, "heuristic")
        elif "rings" in meth:
            sub = subset(rings, meth)
        else:
            sub = subset(edges, meth)
        vals = sub[sub["num_regions"] == nr]["solve_time"].dropna()
        means.setdefault(meth, []).append(vals.mean() if len(vals) else np.nan)

x = np.arange(len(REGIONS))
w = 0.15
n = len(series)
for i, (meth, label, color) in enumerate(series):
    ax.bar(x + (i - (n - 1) / 2) * w, means[meth], width=w, label=label,
           color=color, edgecolor="white", linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(REGIONS)
ax.set_xlabel("Number of regions $n_r$")
ax.set_ylabel("Mean solve time (s, log)")
ax.set_yscale("log")
ax.set_ylim(bottom=0.01)
ax.legend(ncols=3, fontsize=6, loc="upper left")
ax.set_title("Mean solve time by $n_r$ (averaged over $n_h$, seeds, $E$)")
ax.grid(axis="y", linestyle="--", alpha=0.4)
fig.tight_layout()
fig.savefig(os.path.join(OUT_PICS, f"compare_time_by_regions{SUF}.pdf"))
fig.savefig(os.path.join(OUT_PICS, f"compare_time_by_regions{SUF}.png"), dpi=200)
print(f"Saved compare_time_by_regions{SUF}")

# ---------------------------------------------------------------------------
# 2) Mean MIP gap by n_r
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.2, 3.4))
gap_series = [
    ("rings", "Vertex-Approach", C_V),
    ("rings_ws", "Vertex-Approach+WS", C_VWS),
]
if not VERTEX_ONLY:
    gap_series += [("edges", "Edges-Approach", C_E),
                   ("edges_ws", "Edges-Approach+WS", C_EWS)]

gap_means = {}
for nr in REGIONS:
    for meth, label, color in gap_series:
        df = rings if "rings" in meth else edges
        sub = subset(df, meth)
        vals = sub[sub["num_regions"] == nr]["mip_gap"].dropna()
        gap_means.setdefault(meth, []).append(vals.mean() * 100 if len(vals) else np.nan)

x = np.arange(len(REGIONS))
w = 0.18
n = len(gap_series)
for i, (meth, label, color) in enumerate(gap_series):
    ax.bar(x + (i - (n - 1) / 2) * w, gap_means[meth], width=w, label=label,
           color=color, edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(REGIONS)
ax.set_xlabel("Number of regions $n_r$")
ax.set_ylabel("Mean MIP gap (%)")
ax.legend(ncols=3, fontsize=6, loc="upper left")
ax.set_title("Mean MIP gap at time limit (1800s, averaged)")
ax.grid(axis="y", linestyle="--", alpha=0.4)
fig.tight_layout()
fig.savefig(os.path.join(OUT_PICS, f"compare_gap_by_regions{SUF}.pdf"))
fig.savefig(os.path.join(OUT_PICS, f"compare_gap_by_regions{SUF}.png"), dpi=200)
print(f"Saved compare_gap_by_regions{SUF}")

# ---------------------------------------------------------------------------
# 3) Objective parity (skip in vertex-only mode; needs Edges-Approach)
# ---------------------------------------------------------------------------
if not VERTEX_ONLY:
    rings_r = subset(rings, "rings")[["num_regions", "num_heights", "seed", "endurance", "objective"]] \
        .rename(columns={"objective": "obj_rings"})
    rings_ws = subset(rings, "rings_ws")[["num_regions", "num_heights", "seed", "endurance", "objective"]] \
        .rename(columns={"objective": "obj_rings_ws"})
    edges_r = subset(edges, "edges")[["num_regions", "num_heights", "seed", "endurance", "objective"]] \
        .rename(columns={"objective": "obj_edges"})
    edges_ws = subset(edges, "edges_ws")[["num_regions", "num_heights", "seed", "endurance", "objective"]] \
        .rename(columns={"objective": "obj_edges_ws"})

    merged = pd.merge(rings_r, edges_r, on=["num_regions", "num_heights", "seed", "endurance"], how="inner").dropna()
    merged_ws = pd.merge(rings_ws, edges_ws, on=["num_regions", "num_heights", "seed", "endurance"], how="inner").dropna()

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.2), sharex=True, sharey=True)
    for ax, df, title in zip(axes, [merged, merged_ws],
                             ["Vertex-Approach vs Edges-Approach (cold)",
                              "Vertex-Approach+WS vs Edges-Approach+WS (warm)"]):
        ax.scatter(df.iloc[:, 4], df.iloc[:, 5], s=12, alpha=0.5, color=C_V,
                   edgecolors="white", linewidth=0.3)
        mn = min(df.iloc[:, 4].min(), df.iloc[:, 5].min())
        mx = max(df.iloc[:, 4].max(), df.iloc[:, 5].max())
        ax.plot([mn, mx], [mn, mx], "--", color=C_E, linewidth=1)
        ax.set_xlabel("Vertex-Approach objective")
        ax.set_ylabel("Edges-Approach objective")
        ax.set_title(title)
        ratio = (df.iloc[:, 5] / df.iloc[:, 4]).mean()
        ax.text(0.05, 0.95, f"mean ratio {ratio:.4f}\nn={len(df)}",
                transform=ax.transAxes, va="top", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        ax.grid(True, linestyle="--", alpha=0.3)
    fig.suptitle("Objective parity (equivalence check, full testbed)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_PICS, "compare_objective_parity.pdf"))
    fig.savefig(os.path.join(OUT_PICS, "compare_objective_parity.png"), dpi=200)
    print(f"Saved parity, cold {len(merged)} warm {len(merged_ws)}")

# ---------------------------------------------------------------------------
# 4) First incumbent time (warm-start benefit)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.2, 3.4))
vp = pd.merge(
    subset(rings, "rings")[["num_regions", "num_heights", "seed", "endurance", "first_incumbent_time"]]
        .rename(columns={"first_incumbent_time": "t_cold"}),
    subset(rings, "rings_ws")[["num_regions", "num_heights", "seed", "endurance", "first_incumbent_time"]]
        .rename(columns={"first_incumbent_time": "t_ws"}),
    on=["num_regions", "num_heights", "seed", "endurance"], how="inner").dropna()

inc_series = [
    ("Vertex-Approach", vp, C_V, "o", [1, 2, 3, 5, 8, 10]),
]
if not VERTEX_ONLY:
    ep = pd.merge(
        subset(edges, "edges")[["num_regions", "num_heights", "seed", "endurance", "first_incumbent_time"]]
            .rename(columns={"first_incumbent_time": "t_cold"}),
        subset(edges, "edges_ws")[["num_regions", "num_heights", "seed", "endurance", "first_incumbent_time"]]
            .rename(columns={"first_incumbent_time": "t_ws"}),
        on=["num_regions", "num_heights", "seed", "endurance"], how="inner").dropna()
    inc_series.append(("Edges-Approach", ep, C_E, "s", [1, 2, 3, 5, 8, 10]))

for label, df, color, marker, xs in inc_series:
    cold = [df[df["num_regions"] == nr]["t_cold"].mean() for nr in xs]
    warm = [df[df["num_regions"] == nr]["t_ws"].mean() for nr in xs]
    ax.plot(xs, cold, marker=marker, color=color, label=f"{label} cold", linewidth=1.5)
    ax.plot(xs, warm, marker=marker, color=color, linestyle="--",
            label=f"{label}+WS", linewidth=1.5)

ax.set_xticks([1, 2, 3, 5, 8, 10])
ax.set_xticklabels(["1", "2", "3", "5", "8", "10"])
ax.set_xlabel("Number of regions $n_r$")
ax.set_ylabel("Mean first incumbent time (s, log)")
ax.set_yscale("log")
ax.legend(fontsize=6, ncols=2)
ax.set_title("Warm-start benefit: time to first feasible incumbent")
ax.grid(True, linestyle="--", alpha=0.4)
fig.tight_layout()
fig.savefig(os.path.join(OUT_PICS, f"compare_first_incumbent{SUF}.pdf"))
fig.savefig(os.path.join(OUT_PICS, f"compare_first_incumbent{SUF}.png"), dpi=200)
print(f"Saved compare_first_incumbent{SUF}")

# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def fmt(x, nd=1):
    if pd.isna(x):
        return "---"
    return f"{x:.{nd}f}"


def stats(sub, nr):
    vals = sub[sub["num_regions"] == nr]
    obj = vals["objective"].mean()
    tim = vals["solve_time"].mean()
    gap = vals["mip_gap"].mean() * 100 if "mip_gap" in vals else np.nan
    return obj, tim, gap


# Table A: summary by n_r (vertex)
rows = []
for nr in REGIONS:
    oh, th, _ = stats(subset(rings, "heuristic"), nr)
    or_, tr, gr = stats(subset(rings, "rings"), nr)
    ow, tw, gw = stats(subset(rings, "rings_ws"), nr)
    rows.append((nr, oh, th, or_, tr, gr, ow, tw, gw))

with open(os.path.join(OUT_TABLES, "table_results_summary.tex"), "w", encoding="utf-8") as f:
    f.write(r"\begin{table}[t]" + "\n")
    f.write(r"\centering" + "\n")
    f.write(r"\caption{Aggregated results for the Vertex-Approach (MTZ) formulation. "
            r"Mean objective and solve time averaged over height levels ($n_h=1,2,3$), "
            r"endurance levels ($E=300$--2400\,J) and 5 seeds. MIP gap at the 1800\,s limit. "
            r"$H$ = heuristic standalone.}" + "\n")
    f.write(r"\label{tab:results-summary-rings}" + "\n")
    f.write(r"\resizebox{\textwidth}{!}{%" + "\n")
    f.write(r"\begin{tabular}{crrrrrrrr}" + "\n")
    f.write(r"\toprule" + "\n")
    f.write(r"$n_r$ & \multicolumn{2}{c}{$H$} & \multicolumn{3}{c}{$R$ (Vertex-Approach)} "
            r"& \multicolumn{3}{c}{$R$+WS (Vertex-Approach+WS)} \\" + "\n")
    f.write(r"\cmidrule(lr){2-3} \cmidrule(lr){4-6} \cmidrule(lr){7-9}" + "\n")
    f.write(r" & Obj. & Time (s) & Obj. & Time (s) & Gap (\%) & Obj. & Time (s) & Gap (\%) \\" + "\n")
    f.write(r"\midrule" + "\n")
    for nr, oh, th, or_, tr, gr, ow, tw, gw in rows:
        f.write(f"{nr} & {fmt(oh,1)} & {fmt(th,2)} & {fmt(or_,1)} & {fmt(tr,1)} "
                f"& {fmt(gr,1)} & {fmt(ow,1)} & {fmt(tw,1)} & {fmt(gw,1)} \\\\\n")
    f.write(r"\bottomrule" + "\n")
    f.write(r"\end{tabular}%" + "\n")
    f.write(r"}" + "\n")
    f.write(r"\end{table}" + "\n")
print("Wrote table_results_summary.tex")

# Table B: vertex vs edges (only in full mode, for the paper)
if not VERTEX_ONLY:
    rows2 = []
    for nr in REGIONS:
        or_, tr, gr = stats(subset(rings, "rings"), nr)
        ow, tw, gw = stats(subset(rings, "rings_ws"), nr)
        oe, te, ge = stats(subset(edges, "edges"), nr)
        oew, tew, gew = stats(subset(edges, "edges_ws"), nr)
        sub_e = subset(edges, "edges")
        ne = len(sub_e[sub_e["num_regions"] == nr])
        se = sub_e[sub_e["num_regions"] == nr]["objective"].notna().sum()
        rows2.append((nr, or_, tr, oe, te, ow, tw, oew, tew, ne, se))

    with open(os.path.join(OUT_TABLES, "table_results_vertex_edge.tex"), "w", encoding="utf-8") as f:
        f.write(r"\begin{table}[t]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\caption{Vertex-Approach (MTZ) vs Edges-Approach (DFJ lazy) on the full "
                r"testbed: 90 configs per $n_r$ ($n_h \times E \times$ seeds), 1620 runs per "
                r"formulation. Objective/time are means over runs that produced a solution; "
                r"$n$ = solved/total runs for cold Edges-Approach (it finds no solution "
                r"within 1800\,s on most $n_r\ge5$ configs).}" + "\n")
        f.write(r"\label{tab:results-vertex-edge}" + "\n")
        f.write(r"\resizebox{\textwidth}{!}{%" + "\n")
        f.write(r"\begin{tabular}{crrrrrrrrrr}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"$n_r$ & \multicolumn{2}{c}{Vertex-Approach} & \multicolumn{2}{c}{Edges-Approach} "
                r"& \multicolumn{2}{c}{Vertex-Approach+WS} & \multicolumn{2}{c}{Edges-Approach+WS} "
                r"& $n$ (Edges-Approach)\\" + "\n")
        f.write(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}" + "\n")
        f.write(r" & Obj. & Time & Obj. & Time & Obj. & Time & Obj. & Time & solved \\" + "\n")
        f.write(r"\midrule" + "\n")
        for nr, or_, tr, oe, te, ow, tw, oew, tew, ne, se in rows2:
            f.write(f"{nr} & {fmt(or_,1)} & {fmt(tr,1)} & {fmt(oe,1)} & {fmt(te,1)} "
                    f"& {fmt(ow,1)} & {fmt(tw,1)} & {fmt(oew,1)} & {fmt(tew,1)} & {se}/{ne} \\\\\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}%" + "\n")
        f.write(r"}" + "\n")
        f.write(r"\end{table}" + "\n")
    print("Wrote table_results_vertex_edge.tex")

# Table C: endurance effect (n_r=3)
rows3 = []
for E in [300, 450, 600, 900, 1500, 2400]:
    def statsE(sub):
        vals = sub[(sub["num_regions"] == 3) & (sub["endurance"] == E)]
        return vals["objective"].mean(), vals["solve_time"].mean(), vals["mip_gap"].mean() * 100
    oh, th, _ = statsE(subset(rings, "heuristic"))
    or_, tr, gr = statsE(subset(rings, "rings"))
    ow, tw, gw = statsE(subset(rings, "rings_ws"))
    rows3.append((E, oh, th, or_, tr, gr, ow, tw, gw))

with open(os.path.join(OUT_TABLES, "table_results_by_endurance.tex"), "w", encoding="utf-8") as f:
    f.write(r"\begin{table}[t]" + "\n")
    f.write(r"\centering" + "\n")
    f.write(r"\caption{Effect of endurance $E$ for $n_r=3$ (averaged over $n_h=1,2,3$ and 5 seeds). "
            r"Tight $E$ implies more operations and higher total distance.}" + "\n")
    f.write(r"\label{tab:results-by-endurance}" + "\n")
    f.write(r"\resizebox{\textwidth}{!}{%" + "\n")
    f.write(r"\begin{tabular}{crrrrrrrr}" + "\n")
    f.write(r"\toprule" + "\n")
    f.write(r"$E$ (J) & \multicolumn{2}{c}{$H$} & \multicolumn{3}{c}{$R$} "
            r"& \multicolumn{3}{c}{$R$+WS} \\" + "\n")
    f.write(r"\cmidrule(lr){2-3} \cmidrule(lr){4-6} \cmidrule(lr){7-9}" + "\n")
    f.write(r" & Obj. & Time & Obj. & Gap & Time & Obj. & Gap & Time \\" + "\n")
    f.write(r"\midrule" + "\n")
    for E, oh, th, or_, tr, gr, ow, tw, gw in rows3:
        f.write(f"{E} & {fmt(oh,1)} & {fmt(th,2)} & {fmt(or_,1)} & {fmt(gr,1)} & {fmt(tr,1)} "
                f"& {fmt(ow,1)} & {fmt(gw,1)} & {fmt(tw,1)} \\\\\n")
    f.write(r"\bottomrule" + "\n")
    f.write(r"\end{tabular}%" + "\n")
    f.write(r"}" + "\n")
    f.write(r"\end{table}" + "\n")
print("Wrote table_results_by_endurance.tex")

print("Done")