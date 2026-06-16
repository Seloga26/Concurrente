#!/usr/bin/env python3
"""Scoring multicore (Nivel 1): reparte los K candidatos entre varios procesos.

Invocación canónica: `python -m python.multicore <args>` (desde la raíz del repo).
Mismo algoritmo y mismo contrato de E/S que `python.sequential` (una línea JSON a stdout;
avisos/errores a stderr; exit≠0 ante error), con `implementation="python_multicore"` y la
clave adicional `n_workers`.

Paralelismo por **procesos** (no hilos): cada worker fija BLAS a monohilo, por lo que el
speedup proviene de repartir el rango de candidatos. Se usa el contexto `spawn` para que el
comportamiento (y la herencia de las variables BLAS) sea idéntico en Windows y Linux: cada
hijo reimporta este módulo y vuelve a ejecutar la asignación de hilos del tope ANTES de numpy.
"""
# --- BLAS monohilo: asignación explícita ANTES de importar numpy/common (igual que sequential) ---
import os  # noqa: E402

_BLAS_VARS = ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS")
for _v in _BLAS_VARS:
    os.environ[_v] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from common.auc import pairwise_auc  # noqa: E402
from common.keys import better  # noqa: E402
from common.loader import load_and_validate_inputs  # noqa: E402
from common.metrics import compute_theta, consistency  # noqa: E402
from common.scoring import prepare_work_arrays, score_candidate, search_best  # noqa: E402

IMPLEMENTATION = "python_multicore"
KERNEL_VARIANT = "materialized_numpy"
VALID_ALGORITHM = "literal"
VALID_THETA_POLICY = "class_mean_midpoint"
MODE_ACCUM = {"reference": "float64", "benchmark": "float32"}

# Estado por proceso worker, poblado por _init_worker (variables de módulo: con `spawn` cada
# hijo tiene su propia copia, sin compartir memoria ni picklear arrays grandes por tarea).
_W = {}


def _build_parser():
    p = argparse.ArgumentParser(description="Scoring multicore (Nivel 1): emite una línea JSON.")
    p.add_argument("--N", type=int, required=True)
    p.add_argument("--K", type=int, required=True)
    p.add_argument("--mode", required=True)
    p.add_argument("--accum", required=True)
    p.add_argument("--algorithm", required=True)
    p.add_argument("--tie-atol", type=float, required=True)
    p.add_argument("--tie-rtol", type=float, required=True)
    p.add_argument("--matrix-a", required=True)
    p.add_argument("--profile-t", required=True)
    p.add_argument("--profile-s", required=True)
    p.add_argument("--profile-f", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--candidates", required=True)
    p.add_argument("--theta-policy", required=True)
    p.add_argument("--consistency-threshold", type=float, required=True)
    p.add_argument("--workers", type=int, default=None,
                   help="Nº de procesos (default: max(1, cpu_count()-1), deja un núcleo libre).")
    p.add_argument("--output-json", default=None, help="Ruta opcional para escribir el mismo JSON.")
    return p


def _validate_cli(args):
    if args.mode not in MODE_ACCUM:
        raise ValueError(f"mode debe ser reference|benchmark, no {args.mode!r}")
    if args.accum != MODE_ACCUM[args.mode]:
        raise ValueError(
            f"accum={args.accum!r} incompatible con mode={args.mode!r} "
            f"(esperado {MODE_ACCUM[args.mode]})")
    if args.algorithm != VALID_ALGORITHM:
        raise ValueError(f"algorithm debe ser {VALID_ALGORITHM!r}, no {args.algorithm!r}")
    if args.theta_policy != VALID_THETA_POLICY:
        raise ValueError(f"theta_policy debe ser {VALID_THETA_POLICY!r}, no {args.theta_policy!r}")
    if args.workers is not None and args.workers < 1:
        raise ValueError(f"--workers debe ser entero >= 1: {args.workers}")


def _default_workers():
    return max(1, (os.cpu_count() or 1) - 1)


def _make_chunks(K, n_workers):
    """Reparte [0,K) en `n_workers` rangos contiguos casi iguales; descarta los vacíos."""
    base, extra = divmod(K, n_workers)
    chunks = []
    start = 0
    for i in range(n_workers):
        size = base + (1 if i < extra else 0)
        if size > 0:
            chunks.append((start, start + size))
            start += size
    return chunks


def _init_worker(paths, accum, N, K, atol, rtol):
    """Initializer del Pool: carga/valida inputs una vez por worker y hace warmup.

    Carga desde disco (arrays pequeños) en vez de picklear arrays grandes por tarea. El warmup
    saca el coste de import/BLAS/primera multiplicación de la región cronometrada del padre.
    """
    inputs = load_and_validate_inputs(
        paths["matrix_a"], paths["profile_t"], paths["profile_s"], paths["profile_f"],
        paths["labels"], paths["candidates"], N, K)
    wd = np.dtype(accum)
    A_w, T_w, S_w, F_w = prepare_work_arrays(inputs.A, inputs.T, inputs.S, inputs.F, accum)
    _W.update(A=A_w, T=T_w, S=S_w, F=F_w, candidates=inputs.candidates_W,
              pos_idx=inputs.pos_idx, neg_idx=inputs.neg_idx,
              atol=atol, rtol=rtol, work_dtype=wd)
    st = wd.type
    c0 = inputs.candidates_W[0]
    _ = score_candidate(st(c0[0]), st(c0[1]), st(c0[2]), A_w, T_w, S_w, F_w)  # warmup


def _search_chunk(chunk):
    """Explora [k_start, k_stop) sobre el estado del worker. Devuelve (units, k_global, t_core)."""
    k_start, k_stop = chunk
    t0 = time.perf_counter()
    units, k = search_best(
        _W["A"], _W["T"], _W["S"], _W["F"], _W["candidates"], _W["pos_idx"], _W["neg_idx"],
        _W["atol"], _W["rtol"], _W["work_dtype"], k_start=k_start, k_stop=k_stop)
    return units, k, time.perf_counter() - t0


def run(args):
    _validate_cli(args)
    accum = MODE_ACCUM[args.mode]
    wd = np.dtype(accum)

    requested = args.workers if args.workers is not None else _default_workers()
    n_workers = max(1, min(requested, args.K))
    chunks = _make_chunks(args.K, n_workers)
    n_workers = len(chunks)  # K pequeño puede dejar menos chunks que workers solicitados

    paths = {
        "matrix_a": args.matrix_a, "profile_t": args.profile_t, "profile_s": args.profile_s,
        "profile_f": args.profile_f, "labels": args.labels, "candidates": args.candidates,
    }

    # --- Carga + validación en el padre (FUERA del cronómetro), para el recompute del ganador ---
    inputs = load_and_validate_inputs(
        args.matrix_a, args.profile_t, args.profile_s, args.profile_f,
        args.labels, args.candidates, args.N, args.K)
    A_w, T_w, S_w, F_w = prepare_work_arrays(inputs.A, inputs.T, inputs.S, inputs.F, accum)

    # --- Pool con contexto spawn (FUERA del cronómetro: startup fijo, como la carga) ---
    ctx = mp.get_context("spawn")
    initargs = (paths, accum, args.N, args.K, args.tie_atol, args.tie_rtol)
    with ctx.Pool(processes=n_workers, initializer=_init_worker, initargs=initargs) as pool:
        # --- Región cronometrada: búsqueda paralela + reducción determinista ---
        t0 = time.perf_counter()
        results = pool.map(_search_chunk, chunks)
        best_units, best_k = -1, -1
        for units, k, _t in results:
            if better(units, k, best_units, best_k):
                best_units, best_k = units, k
        t1 = time.perf_counter()

    t_search = t1 - t0                              # incluye dispatch/IPC/reducción
    t_core = max(t for _u, _k, t in results)        # span de cómputo paralelo útil (para Amdahl)

    # --- Recompute del ganador en el padre (FUERA del cronómetro) ---
    bw = inputs.candidates_W[best_k]
    st = wd.type
    best_score = score_candidate(st(bw[0]), st(bw[1]), st(bw[2]), A_w, T_w, S_w, F_w)
    auc = pairwise_auc(best_score, inputs.pos_idx, inputs.neg_idx, args.tie_atol, args.tie_rtol)
    theta = compute_theta(best_score, inputs.pos_idx, inputs.neg_idx)
    cons = consistency(best_score, theta, inputs.pos_idx, inputs.neg_idx)

    return {
        "implementation": IMPLEMENTATION,
        "mode": args.mode,
        "algorithm": args.algorithm,
        "kernel_variant": KERNEL_VARIANT,
        "accum_dtype": accum,
        "n_items": int(inputs.N),
        "n_candidates": int(inputs.K),
        "n_workers": int(n_workers),
        "best_k": int(best_k),
        "best_w": [float(x) for x in bw],
        "auc_units": int(auc.auc_units),
        "auc_denominator": int(auc.denominator),
        "auc": float(auc.auc),
        "scores": [float(x) for x in best_score],
        "theta": float(theta),
        "consistency": float(cons),
        "consistency_threshold": float(args.consistency_threshold),
        "consistency_pass": bool(cons >= args.consistency_threshold),
        "tie_atol": float(args.tie_atol),
        "tie_rtol": float(args.tie_rtol),
        "theta_policy": args.theta_policy,
        "t_core_seconds": float(t_core),
        "t_search_seconds": float(t_search),
        "blas_threads": {v: os.environ.get(v) for v in _BLAS_VARS},
    }


def main(argv=None):
    args = _build_parser().parse_args(argv)
    try:
        result = run(args)
    except (ValueError, FileNotFoundError, AssertionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    line = json.dumps(result, ensure_ascii=False)
    print(line)                       # exactamente una línea JSON a stdout
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as fh:
            fh.write(line + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
