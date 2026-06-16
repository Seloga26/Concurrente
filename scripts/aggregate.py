#!/usr/bin/env python3
"""Agrega las corridas JSONL del benchmark en results/benchmark.csv (resumen) y benchmark_runs.csv.

Lee uno o varios JSONL etiquetados por plataforma (cada linea = la salida JSON de una impl) y produce:
  - results/benchmark.csv      : RESUMEN agregado (una fila por platform/impl/mode/K/P) con la mediana
                                  de t_core/t_search sobre las repeticiones + speedup y efficiency.
                                  Es el entregable canonico (versionado).
  - results/benchmark_runs.csv : todas las corridas crudas (para anexos; gitignored).

Speedup por (platform, mode, K): familia Python -> baseline python_sequential (P=1); familia C
(c_serial/c_openmp/c_mpi) -> c_serial (P=1); CUDA -> c_serial del MISMO platform (speedup vs CPU
serial, sin efficiency). El speedup nunca mezcla plataformas (HW distinto).

Uso:  python scripts/aggregate.py --out results/ wsl=results/raw_wsl.jsonl [colab=results/raw_colab.jsonl]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics

# Familia -> implementacion baseline (serial, P=1) para el speedup.
_FAMILY = {
    "python_sequential": "python", "python_multicore": "python",
    "c_serial": "c", "c_openmp": "c", "c_mpi": "c",
    "cuda": "cuda", "cuda_pycuda": "cuda",
}
_BASELINE = {"python": "python_sequential", "c": "c_serial", "cuda": "c_serial"}
# Impls a las que aplica efficiency = speedup / P (paralelismo sobre P unidades).
_HAS_EFFICIENCY = {"python_multicore", "c_openmp", "c_mpi"}


def family(impl):
    return _FAMILY.get(impl, "other")


def parallelism(rec):
    """Grado de paralelismo P de una corrida: n_workers|n_threads|n_procs, o 1 (serial/seq/cuda)."""
    for key in ("n_workers", "n_threads", "n_procs"):
        if key in rec and rec[key]:
            return int(rec[key])
    return 1


def load_runs(label_files):
    """label_files: lista de (platform, ruta_jsonl). Devuelve lista de records con 'platform' y 'P'."""
    runs = []
    for platform, path in label_files:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rec["platform"] = platform
                rec["P"] = parallelism(rec)
                runs.append(rec)
    return runs


def _key(rec):
    return (rec["platform"], rec["implementation"], rec["mode"], rec["n_candidates"], rec["P"])


def summarize(runs):
    """Agrupa por (platform, impl, mode, K, P) y devuelve filas con medianas (sin speedup aun)."""
    groups = {}
    for r in runs:
        groups.setdefault(_key(r), []).append(r)

    rows = []
    for (platform, impl, mode, K, P), recs in groups.items():
        rep = recs[0]
        rows.append({
            "platform": platform,
            "implementation": impl,
            "mode": mode,
            "N": rep["n_items"],
            "K": K,
            "P": P,
            "reps": len(recs),
            "t_core_median_s": statistics.median(r["t_core_seconds"] for r in recs),
            "t_search_median_s": statistics.median(r["t_search_seconds"] for r in recs),
            "best_k": rep["best_k"],
            "auc_units": rep["auc_units"],
            "auc": rep["auc"],
            "consistency": rep["consistency"],
            "consistency_pass": rep["consistency_pass"],
            "device": rep.get("device", ""),
        })
    return rows


def attach_speedup(rows):
    """Rellena speedup y efficiency in-place. Baseline = serial de la familia, mismo platform/mode/K."""
    # Tiempo base por (platform, mode, K) para cada baseline serial.
    base = {}
    for row in rows:
        if row["P"] == 1 and row["implementation"] in ("python_sequential", "c_serial"):
            base[(row["platform"], row["mode"], row["K"], row["implementation"])] = row["t_core_median_s"]

    for row in rows:
        fam = family(row["implementation"])
        baseline_impl = _BASELINE.get(fam)
        t_base = base.get((row["platform"], row["mode"], row["K"], baseline_impl))
        if t_base and row["t_core_median_s"] > 0:
            row["speedup"] = round(t_base / row["t_core_median_s"], 6)
        else:
            row["speedup"] = ""
        if row["implementation"] in _HAS_EFFICIENCY and isinstance(row["speedup"], float) and row["P"] > 0:
            row["efficiency"] = round(row["speedup"] / row["P"], 6)
        elif row["implementation"] in ("python_sequential", "c_serial"):
            row["efficiency"] = 1.0
        else:
            row["efficiency"] = ""   # CUDA u otros: sin efficiency
    return rows


_SUMMARY_COLS = ["platform", "implementation", "mode", "N", "K", "P", "reps",
                 "t_core_median_s", "t_search_median_s", "speedup", "efficiency",
                 "best_k", "auc_units", "auc", "consistency", "consistency_pass", "device"]
_RUN_COLS = ["platform", "implementation", "mode", "N", "K", "P",
             "t_core_seconds", "t_search_seconds", "best_k", "auc_units", "auc",
             "consistency", "consistency_pass", "device"]


def _write_csv(path, cols, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_outputs(runs, out_dir):
    """Escribe benchmark.csv (resumen ordenado) y benchmark_runs.csv (crudo)."""
    summary = attach_speedup(summarize(runs))
    summary.sort(key=lambda r: (r["platform"], r["mode"], r["K"], r["implementation"], r["P"]))
    _write_csv(os.path.join(out_dir, "benchmark.csv"), _SUMMARY_COLS, summary)

    run_rows = [{
        "platform": r["platform"], "implementation": r["implementation"], "mode": r["mode"],
        "N": r["n_items"], "K": r["n_candidates"], "P": r["P"],
        "t_core_seconds": r["t_core_seconds"], "t_search_seconds": r["t_search_seconds"],
        "best_k": r["best_k"], "auc_units": r["auc_units"], "auc": r["auc"],
        "consistency": r["consistency"], "consistency_pass": r["consistency_pass"],
        "device": r.get("device", ""),
    } for r in runs]
    run_rows.sort(key=lambda r: (r["platform"], r["mode"], r["K"], r["implementation"], r["P"]))
    _write_csv(os.path.join(out_dir, "benchmark_runs.csv"), _RUN_COLS, run_rows)
    return summary


def _parse_label_file(tok):
    if "=" not in tok:
        raise ValueError(f"esperado LABEL=FILE, recibido {tok!r}")
    label, path = tok.split("=", 1)
    return label, path


def main(argv=None):
    p = argparse.ArgumentParser(description="Agrega corridas JSONL en results/benchmark.csv.")
    p.add_argument("--out", default="results", help="Directorio de salida (default: results).")
    p.add_argument("inputs", nargs="+", help="Entradas etiquetadas: LABEL=ruta.jsonl")
    args = p.parse_args(argv)
    label_files = [_parse_label_file(t) for t in args.inputs]
    runs = load_runs(label_files)
    if not runs:
        print("aggregate: no hay corridas en las entradas", flush=True)
        return 1
    summary = write_outputs(runs, args.out)
    print(f"aggregate: {len(runs)} corridas -> {len(summary)} filas de resumen en {args.out}/benchmark.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
