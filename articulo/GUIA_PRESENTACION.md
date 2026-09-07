# Presentation Guide — Coverage Path Planning (v8, 1 Sep 2026)

**Source:** `articulo/congreso_beamer.tex` → `articulo/congreso_beamer.pdf` (26 pages, 16:9, Madrid/whale)
**Talk design:** 15–17 min + Q&A (20 min slot). OR audience. **Slides in English, spoken talk in Spanish.**
**Paper:** `articulo/main.tex` → `main.pdf` (27 pages).

---

## v8 changes (this iteration)

- **Backup slides removed** (`\appendix` + 4 backup frames deleted; `appendixnumberbeamer` dropped).
- **Comments/text about incompleteness removed**: "edge partial" header comment, "(pending)" figure caption, the "Edges-Approach excluded / incomplete" bullet in the analysis slide, "backup slides follow" note.
- **First-incumbent analysis extended to $n_r = 10$**: `scripts/generate_comparison.py` now plots $n_r \in \{1,2,3,5,8,10\}$ for the warm-start benefit figure (both `_vertex` and paper versions regenerated); slide caption updated.
- **Variables as itemize slide**: new "Decision variables" slide right before the formulations — narrative intro ("four families of constraints — location, drone path, distance, consumption — plus the objective") + itemized binary/integer and continuous variables, mirroring the article's description.

## Structure & timing (16'00'')

| # | Slide | Time | Cum |
|---|-------|------|-----|
| 1 | Title | 0:30 | 0:30 |
| 2 | Introduction: the problem (narrative + Exagon) | 1:00 | 1:30 |
| 3 | The general problem ($\mathcal R$, chains, preemption, $\mathcal G$, `set_I.pdf`) | 1:00 | 2:30 |
| 4 | Decision variables (itemize, before formulations) | 0:45 | 3:15 |
| 5 | Formulation overview (4 families + objective) | 0:40 | 3:55 |
| 6 | (i) Location | 0:55 | 4:50 |
| 7 | (ii) Routing + MTZ + VI | 0:55 | 5:45 |
| 8 | (iii) Distances (DFS/SOC/Dep) | 0:50 | 6:35 |
| 9 | (iv) Energy intra (EFS) + depot | 0:40 | 7:15 |
| 10 | (iv) Energy inter ($w$-C3) | 0:45 | 8:00 |
| 11 | Endurance & objective (big-M) | 0:40 | 8:40 |
| 12 | Problem setup (showcase: 3 regions, 15/40/65 m, drone params, free entry/exit) | 0:50 | 9:30 |
| 13 | Visual guide (solid/dotted/colors/depot/markers) | 0:30 | 10:00 |
| 14 | Endurance = 150 J — 3 operations (animation) | 0:30 | 10:30 |
| 15 | Endurance = 600 J — 1 operation (animation) | 0:30 | 11:00 |
| 16 | Showcase summary (150 vs 600 J) | 0:25 | 11:25 |
| 17 | Multi-phase heuristic: overview (`heuristic_flow.pdf`) | 0:35 | 12:00 |
| 18 | Heuristic phases 1–3 + warm-start | 0:50 | 12:50 |
| 19 | Experimental design (540 configs, calibration, Vertex-Approach complete) | 0:45 | 13:35 |
| 20 | Results — aggregated means (`table_results_summary`) | 0:35 | 14:10 |
| 21 | Results — endurance effect $n_r=3$ (`table_results_by_endurance`) | 0:25 | 14:35 |
| 22 | Comparative figures — time & gap (vertex-only) | 0:40 | 15:15 |
| 23 | Warm-start benefit (first incumbent, $n_r=1..10$) & visual solutions | 0:40 | 15:55 |
| 24 | Vertex-Approach analysis — the complete case (takeaways) | 0:35 | 16:30 |
| 25 | Conclusions & future | 0:35 | 17:05 |
| 26 | Thank you / Q&A | — | — |

> **15 min fast path:** cut 15 s from slides 3, 12 and 24. **17 min stretch:** add 10 s on slides 14–15 (let the animations loop) and 15 s on slide 18.

---

## Compile

```bash
cd articulo
pdflatex congreso_beamer.tex   # ×2 (needs animate package; frames already generated)
pdflatex main.tex; bibtex main; pdflatex main.tex; pdflatex main.tex
```

Notes:
- Only `anim_frames_150/` and `anim_frames_600/` are referenced (frames 001–050 and 001–048). Regenerate with `python scripts/make_animation_frames.py` (reads `sweep_results_new/end_*_sol.json`, no Gurobi needed).
- If `congreso_beamer.pdf` is open in a viewer, pdflatex fails with "I can't write on file"; close it and recompile (or rename `congreso_beamer_anim.pdf`).
- Beamer figures use the `*_vertex` variants (`scripts/generate_comparison.py --vertex-only`); the paper keeps the full versions.
- `scripts/generate_comparison.py` (default mode) regenerates the paper tables/figures, including the vertex-vs-edges table still referenced by the article.

## Data status (31 Aug 2026)

- **Vertex-Approach**: complete — 540 configs × 3 methods = 1703 rows in `compare_results/results.csv`. All analysis slides use means over the complete sweep.

---

## Pre-talk checklist

- [ ] 16:9 projector (fallback `aspectratio=43`)
- [ ] Test animations on the venue machine (Adobe Reader / PDF.js needed for `animate`)
- [ ] Rehearse to 15:30–16:00
- [ ] Demo: `python compare_methods.py --resume`