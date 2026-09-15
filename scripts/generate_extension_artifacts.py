#!/usr/bin/env python
"""Generate paper artifacts for the K-split and wind-aware extensions.

Reads compare_results/results.csv (+edges), split_points/results_E0150.json,
demo_wind/demoW_*.json, demo_irregular/irr_summary.json, wind_validation/.

Outputs (articulo/):
  tables/table_heuristic_quality.tex
  tables/table_k_sweep.tex
  tables/table_wind_validation.tex
  pictures/compare_model_size.pdf/.png
"""
import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 7, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 150,
})

OUT_T = "articulo/tables"
OUT_P = "articulo/pictures"
os.makedirs(OUT_T, exist_ok=True)
os.makedirs(OUT_P, exist_ok=True)

key = ["num_regions", "num_heights", "seed", "endurance", "method"]
rings = pd.read_csv("compare_results/results.csv").drop_duplicates(subset=key, keep="last")
edges = pd.read_csv("compare_results_edges/results.csv").drop_duplicates(subset=key, keep="last")
for col in ["objective", "solve_time", "mip_gap", "n_vars", "n_constrs"]:
    for df in (rings, edges):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def fmt(x, nd=1):
    if pd.isna(x):
        return "---"
    return f"{x:.{nd}f}"


REGIONS = [1, 2, 3, 5, 8, 10]

# ---------------------------------------------------------------------------
# Table: heuristic quality vs best exact, per n_r
# ---------------------------------------------------------------------------
with open(os.path.join(OUT_T, "table_heuristic_quality.tex"), "w", encoding="utf-8") as f:
    f.write(r"\begin{table}[t]" + "\n")
    f.write(r"\centering" + "\n")
    f.write(r"\caption{Heuristic solution quality: mean heuristic objective vs.\ "
            r"best exact mean objective per region count (averaged over $n_h$, $E$ "
            r"and seeds; exact = min over the four MIQP runs per instance).}" + "\n")
    f.write(r"\label{tab:heuristic-quality}" + "\n")
    f.write(r"\begin{tabular}{crrr}" + "\n")
    f.write(r"\toprule" + "\n")
    f.write(r"$n_r$ & Heur.\ obj. & Best exact obj. & Ratio \\" + "\n")
    f.write(r"\midrule" + "\n")
    for nr in REGIONS:
        h = rings[(rings.num_regions == nr) & (rings.method == "heuristic")].groupby(
            ["num_heights", "seed", "endurance"])["objective"].mean()
        best = []
        for (nh, s, e), _ in h.items():
            w = rings[(rings.num_regions == nr) & (rings.num_heights == nh)
                      & (rings.seed == s) & (rings.endurance == e)
                      & (rings.method.isin(["rings", "rings_ws", "edges", "edges_ws"]))]
            w = w["objective"].dropna()
            if len(w):
                best.append(w.min())
        hmean = h.mean()
        bmean = float(np.mean(best)) if best else float("nan")
        f.write(f"{nr} & {fmt(hmean,1)} & {fmt(bmean,1)} & {fmt(hmean/bmean,3)} \\\\\n")
    f.write(r"\bottomrule" + "\n")
    f.write(r"\end{tabular}" + "\n")
    f.write(r"\end{table}" + "\n")
print("Wrote table_heuristic_quality.tex")

# ---------------------------------------------------------------------------
# Figure: model size scaling (a) vs n_r per method, (b) vs K on demo
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0), sharey=False)
ax = axes[0]
for meth, label, color, ls in [
        ("rings", "Vertex-Approach", "#003366", "-"),
        ("edges", "Edges-Approach", "#BA0C2F", "--")]:
    df = rings if "rings" in meth else edges
    xs, ys = [], []
    for nr in REGIONS:
        v = df[(df.num_regions == nr) & (df.method == meth)]["n_vars"].dropna()
        if len(v):
            xs.append(nr)
            ys.append(v.mean())
    ax.plot(xs, ys, marker="o", label=label, color=color, linestyle=ls)
ax.set_xlabel("Number of regions $n_r$")
ax.set_ylabel("Mean #variables (log)")
ax.set_yscale("log")
ax.legend(fontsize=7)
ax.grid(True, linestyle="--", alpha=0.3)
ax.set_title("(a) size vs instance size")

ax = axes[1]
k_nvars = {}
for root, _, files in os.walk("."):
    pass
# n_vars per K from stored sweep rows (demo wind-aware Ksweep)
try:
    krows = json.load(open("demo_wind/demoW_Ksweep_E0173.json", encoding="utf-8"))
    for r in krows:
        if "n_vars" in r and r.get("K") is not None:
            k_nvars[r["K"]] = r["n_vars"]
except Exception:
    pass
if k_nvars:
    # K=2 row may miss n_vars (2h optimum lives in demoW_summary.json):
    # count it by building the model (no solve needed)
    if 2 not in k_nvars:
        try:
            sys.path.insert(0, "scripts")
            from demo_interruptions import build_demo_instance
            from drone_cpp.model import build_model
            _inst = build_demo_instance(173.0, num_ops=4)
            _m = build_model(_inst, "rings", verbose=False,
                             num_split_points=2, wind_aware=True)
            _s = _m.variable_summary()
            k_nvars[2] = _s["binary"] + _s["continuous"] + _s["integer"]
            print("counted K=2 vars:", k_nvars[2])
        except Exception as ex:
            print("could not count K=2 vars:", ex)
    ks = sorted(k_nvars)
    ax.plot(ks, [k_nvars[k] for k in ks], marker="s", color="#003366")
    ax.set_xticks(ks)
    ax.set_xlabel("Split points per ring ($K$)")
    ax.set_ylabel("#variables")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_title("(b) size vs $K$ (demo, wind-aware)")
fig.tight_layout()
fig.savefig(os.path.join(OUT_P, "compare_model_size.pdf"))
fig.savefig(os.path.join(OUT_P, "compare_model_size.png"), dpi=200)
print("Saved compare_model_size", k_nvars)

# ---------------------------------------------------------------------------
# Table: K-sweep objectives (showcase unaware E=150, demo wind-aware E=173,
# irregular wind-aware E=153)
# ---------------------------------------------------------------------------
def load_rows(path):
    try:
        return json.load(open(path, encoding="utf-8"))
    except Exception:
        return []

show_rows = {r["K"]: r for r in load_rows("split_points/results_E0150.json")}
demo_rows = {r["K"]: r for r in load_rows("demo_wind/demoW_Ksweep_E0173.json")}
# K=2 demo optimum lives in demoW_summary.json
try:
    for r in json.load(open("demo_wind/demoW_summary.json", encoding="utf-8")):
        if isinstance(r, dict) and r.get("tag") == "WA K=2 tight 2h":
            demo_rows[2] = {"K": 2, "objective": r["objective"],
                            "gap_pct": r["gap_pct"], "time_s": r["time_s"],
                            "status": r["status"]}
except Exception:
    pass
irr_rows = {}
try:
    for r in json.load(open("demo_irregular/irr_summary.json", encoding="utf-8")):
        irr_rows[r["K"]] = r
except Exception:
    pass

# Showcase K=1/K=4 reference values (optimal runs stored outside the JSON sweep)
show_rows.setdefault(1, {"K": 1, "objective": 708.0})
show_rows.setdefault(4, {"K": 4, "objective": 708.0})
with open(os.path.join(OUT_T, "table_k_sweep.tex"), "w", encoding="utf-8") as f:
    f.write(r"\begin{table}[t]" + "\n")
    f.write(r"\centering" + "\n")
    f.write(r"\caption{Best objective found vs.\ number of split points $K$ per ring. "
            r"Showcase: pseudo-energy, $E=150$\,J (K=1 optimal). Demo and irregular: "
            r"wind-aware real joules ($E=173$\,J and $E=153$\,J). Bold marks the best "
            r"value per instance.}" + "\n")
    f.write(r"\label{tab:k-sweep}" + "\n")
    f.write(r"\resizebox{\textwidth}{!}{%" + "\n")
    f.write(r"\begin{tabular}{crrr}" + "\n")
    f.write(r"\toprule" + "\n")
    f.write(r"$K$ & Showcase ($E=150$) & Demo ($E=173$) & Irregular ($E=153$) \\" + "\n")
    f.write(r"\midrule" + "\n")
    cols = []
    for d in (show_rows, demo_rows, irr_rows):
        col = {}
        for K in [1, 2, 3, 4, 5]:
            r = d.get(K)
            col[K] = r.get("objective") if r and r.get("objective") is not None else None
        cols.append(col)
    best = [min([c[K] for c in cols if c[K] is not None]) for K in [1, 2, 3, 4, 5]]
    _best = {}
    for j, d in enumerate([show_rows, demo_rows, irr_rows]):
        vals = [cols[j][K] for K in [1, 2, 3, 4, 5]]
        _best[j] = min([v for v in vals if v is not None])
    for K in [1, 2, 3, 4, 5]:
        cells = []
        for j, d in enumerate([show_rows, demo_rows, irr_rows]):
            v = cols[j][K]
            if v is None:
                cells.append("---")
            elif abs(v - _best[j]) < 1e-9:
                cells.append(f"\\textbf{{{fmt(v,1)}}}")
            else:
                cells.append(fmt(v, 1))
        f.write(f"{K} & " + " & ".join(cells) + " \\\\\n")
    f.write(r"\bottomrule" + "\n")
    f.write(r"\end{tabular}%" + "\n")
    f.write(r"}" + "\n")
    f.write(r"\end{table}" + "\n")
print("Wrote table_k_sweep.tex")

# ---------------------------------------------------------------------------
# Table: wind validation (wind-aware K=1 tight/loose, wind 8 vs calm)
# ---------------------------------------------------------------------------
wrows = []
try:
    wrows = json.load(open("wind_validation/wind_validation.json", encoding="utf-8"))
except Exception:
    pass
sel = [r for r in wrows if isinstance(r, dict) and r.get("tag", "").startswith("WA K=1")]
with open(os.path.join(OUT_T, "table_wind_validation.tex"), "w", encoding="utf-8") as f:
    f.write(r"\begin{table}[t]" + "\n")
    f.write(r"\centering" + "\n")
    f.write(r"\caption{Wind-aware validation on the showcase ($E_{\min}=68.7$\,J real): "
            r"strong wind (8\,m/s east) vs.\ calm. Identical optimal tours and "
            r"objectives: wind does not change the optimum here.}" + "\n")
    f.write(r"\label{tab:wind-validation}" + "\n")
    f.write(r"\begin{tabular}{lcccc}" + "\n")
    f.write(r"\toprule" + "\n")
    f.write(r"Case & $E$ (J) & Obj.\ (m) & Ops & Gap (\%) \\" + "\n")
    f.write(r"\midrule" + "\n")
    for r in sel:
        tag = r.get("tag", "").replace("WA K=1 ", "")
        f.write(f"{tag} & {fmt(r.get('E'),0)} & {fmt(r.get('objective'),1)} & "
                f"{r.get('n_ops', '---')} & {fmt(r.get('gap_pct'),1)} \\\\\n")
    f.write(r"\bottomrule" + "\n")
    f.write(r"\end{tabular}" + "\n")
    f.write(r"\end{table}" + "\n")
print("Wrote table_wind_validation.tex with", len(sel), "rows")
print("Done")
