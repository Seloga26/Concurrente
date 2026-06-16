#!/usr/bin/env python3
"""Scoring secuencial de referencia (Nivel 1).

Invocación canónica: `python -m python.sequential <args>` (desde la raíz del repo).
Fija las variables de hilos BLAS a "1" ANTES de importar numpy para garantizar un
baseline realmente monohilo (solo afecta a este proceso). Emite UNA línea JSON a stdout;
avisos/errores van a stderr.
"""
# --- BLAS monohilo: asignación explícita ANTES de importar numpy/common (corrección 1) ---
import os  # noqa: E402

_BLAS_VARS = ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS")
for _v in _BLAS_VARS:
    os.environ[_v] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from common.auc import pairwise_auc  # noqa: E402
from common.loader import load_and_validate_inputs  # noqa: E402
from common.metrics import compute_theta, consistency  # noqa: E402
from common.scoring import prepare_work_arrays, score_candidate, search_best  # noqa: E402

IMPLEMENTATION = "python_sequential"
KERNEL_VARIANT = "materialized_numpy"
VALID_ALGORITHM = "literal"
VALID_THETA_POLICY = "class_mean_midpoint"
MODE_ACCUM = {"reference": "float64", "benchmark": "float32"}


def _build_parser():
    p = argparse.ArgumentParser(description="Scoring secuencial (Nivel 1): emite una línea JSON.")
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


def run(args):
    _validate_cli(args)
    accum = MODE_ACCUM[args.mode]
    wd = np.dtype(accum)

    # --- Carga + validación (FUERA del cronómetro) ---
    inputs = load_and_validate_inputs(
        args.matrix_a, args.profile_t, args.profile_s, args.profile_f,
        args.labels, args.candidates, args.N, args.K)

    A_w, T_w, S_w, F_w = prepare_work_arrays(inputs.A, inputs.T, inputs.S, inputs.F, accum)

    # --- Warmup (FUERA del cronómetro), no muta entradas; verifica dtype ---
    st = wd.type
    w0 = st(inputs.candidates_W[0, 0]); w1 = st(inputs.candidates_W[0, 1]); w2 = st(inputs.candidates_W[0, 2])
    warm = score_candidate(w0, w1, w2, A_w, T_w, S_w, F_w)
    assert warm.dtype == wd, f"dtype de Score {warm.dtype} != {wd}"

    # --- Búsqueda (DENTRO del cronómetro, una sola ejecución) ---
    t0 = time.perf_counter()
    best_units, best_k = search_best(
        A_w, T_w, S_w, F_w, inputs.candidates_W, inputs.pos_idx, inputs.neg_idx,
        args.tie_atol, args.tie_rtol, wd)
    t1 = time.perf_counter()
    t_core = t1 - t0
    t_search = t_core   # secuencial: misma región (divergen con overhead de comunicación)

    # --- Recompute del ganador (FUERA del cronómetro) ---
    bw = inputs.candidates_W[best_k]
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
