# K-split battery — server runbook

Runs the **same testbed as the vertex battery** (`compare_methods.py`:
$n_r \in \{1,2,3,5,8,10\} \times n_h \in \{1,2,3\} \times$ 5 seeds
$\times$ $E \in \{300,450,600,900,1500,2400\}$ = 540 configs) with
`num_split_points = K >= 2`.

Per (config, K) two Gurobi runs (heuristic is K-agnostic and already in
`compare_results/results.csv`, so it is **not** re-run):

| method | description |
|---|---|
| `ringsK{K}_cold` | K-arc model, cold start |
| `ringsK{K}_wsk1` | K-arc model warm-started from the K=1 vertex solution (`compare_results/solutions/…_rings.json`, loaded from disk — shipped with this commit) |

## 0. Prerrequisitos en el servidor

```bash
git clone <repo> && cd <repo>
python -m venv .venv && source .venv/bin/activate
pip install -r server/requirements.txt
```

Gurobi necesita licencia: o bien `gurobi.lic` en el directorio de trabajo
(está en `.gitignore`, **no** se sube), o bien variable de entorno:

```bash
export GRB_LICENSE_FILE=/ruta/a/gurobi.lic
```

Este commit ya incluye `compare_results/endurance_levels.json` (niveles E
idénticos al vertex) y las 1578 soluciones K=1 para el warm-start — no hay
que copiar datos.

## 1. Planificar capacidad (sin resolver nada)

```bash
# Tier 1 (core): K=2,3 en regiones pequeñas, grid completo
python compare_methods_k.py --dry-run --ks 2,3 --nr 1,2,3
# -> 270 configs x 2 metodos x 2 K = 1080 runs

# Tier 2 (grande, disperso): K=2 en regiones grandes, solo E tight+loose
python compare_methods_k.py --dry-run --ks 2 --nr 5,8,10 --E 300,2400
# -> 90 configs x 2 metodos = 180 runs
```

Orientación de coste: el tamaño del modelo crece de forma pronunciada con K
(≈3k variables en K=1 frente a ≈66k en K=5 en la instancia demo del paper). Con `--time-limit 900` y
`--mip-gap 0.02`, Tier 1 ≈ 1–2 días en 8 workers de 4 cores.

## 2. Lanzar en paralelo

Cada shard escribe su propio `results_shard{i}.csv` (sin escrituras
concurrentes al mismo fichero) y admite `--resume`.

**SLURM array** (editar `#SBATCH` y el bloque TIER en
`server/slurm_k_battery.sh`):

```bash
sbatch server/slurm_k_battery.sh        # array 0-7, Tier 1 por defecto
```

**Manual / varias máquinas** (N shards totales, uno por proceso):

```bash
export VENV=$PWD/.venv   # opcional
./server/run_shard.sh 0 8 --ks 2,3 --nr 1,2,3 --threads 4 --time-limit 900 &
./server/run_shard.sh 1 8 --ks 2,3 --nr 1,2,3 --threads 4 --time-limit 900 &
# ...
```

`--threads` = cores por tarea Gurobi. Reanudar tras una caída: añadir
`--resume` (salta lo registrado en `completed_shard{i}.txt`).

## 3. Agregar resultados

```bash
python server/aggregate_k.py                  # -> compare_results_k/results_k.csv
python server/aggregate_k.py --out-dir <dir>  # si se cambió --out-dir
```

Dedup por clave `(nr, nh, seed, E, K, method)` quedándose con la **última**
fila (misma convención que el vertex). Imprime cobertura por (K, método,
$n_r$) para detectar huecos; relanzar shards con `--resume` para taparlos.

## 4. Analizar

`results_k.csv` tiene las mismas columnas que `results.csv` más `k` al
final. Los scripts de análisis (`scripts/generate_comparison.py`,
`scripts/generate_extension_artifacts.py`) se extenderán para leerlo
(pte. tras completar Tier 1).

## Notas

- `SKIPPED_NO_K1`: la solución vertex de esa config no existe en disco
  (el vertex resolvió 509/540 en frío); esa fila queda marcada y el cold
  sigue ejecutándose.
- `ERROR`/`TIME_LIMIT_NO_SOL`: ver `run_shard*.log` en `--out-dir`.
- Memoria: reservar ≥4 GB/core; K=3 en $n_r \ge 8$ puede superar 30 GB —
  de ahí que el Tier 2 use solo K=2 y E extremos.
- Reproducibilidad: instancias idénticas al vertex (mismo
  `compare_methods.build_instance`, mismas seeds y niveles E).
