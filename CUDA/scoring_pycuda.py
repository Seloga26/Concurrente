#!/usr/bin/env python3
"""Variante CUDA secundaria via PyCUDA (Nivel 3), implementation = "cuda_pycuda".

Mismo contrato/CLI que python/sequential, pero la BUSQUEDA del mejor candidato corre en la GPU
usando EXACTAMENTE el mismo kernel device que el driver nvcc (CUDA/scoring_device.cuh, leido como
string para SourceModule). La carga/validacion y el recompute del ganador reutilizan la capa Python
`common/` (identico a sequential), por lo que scores/theta/consistency coinciden con el oraculo.

Sin GPU NVIDIA local: ejecutar en Google Colab (runtime GPU, `pip install pycuda`). Ver README_colab.md.

Invocacion: `python CUDA/scoring_pycuda.py <mismos flags que python -m python.sequential>`.
"""
# --- BLAS monohilo ANTES de numpy (el recompute usa numpy en CPU) ---
import os  # noqa: E402
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import sys  # noqa: E402
# La raiz del repo en sys.path para importar `common` (CUDA/ no es paquete).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402

from common.auc import pairwise_auc  # noqa: E402
from common.keys import unpack_key  # noqa: E402
from common.loader import load_and_validate_inputs  # noqa: E402
from common.metrics import compute_theta, consistency  # noqa: E402
from common.scoring import prepare_work_arrays, score_candidate  # noqa: E402

IMPLEMENTATION = "cuda_pycuda"
KERNEL_VARIANT = "fused"
VALID_ALGORITHM = "literal"
VALID_THETA_POLICY = "class_mean_midpoint"
MODE_ACCUM = {"reference": "float64", "benchmark": "float32"}
_DEVICE_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scoring_device.cuh")


def _build_parser():
    p = argparse.ArgumentParser(description="Scoring CUDA via PyCUDA (Nivel 3): emite una linea JSON.")
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
    p.add_argument("--block-size", type=int, default=256, help="hilos por bloque (potencia de 2).")
    p.add_argument("--output-json", default=None)
    return p


def _validate_cli(args):
    if args.mode not in MODE_ACCUM:
        raise ValueError(f"mode debe ser reference|benchmark, no {args.mode!r}")
    if args.accum != MODE_ACCUM[args.mode]:
        raise ValueError(f"accum={args.accum!r} incompatible con mode={args.mode!r} "
                         f"(esperado {MODE_ACCUM[args.mode]})")
    if args.algorithm != VALID_ALGORITHM:
        raise ValueError(f"algorithm debe ser {VALID_ALGORITHM!r}, no {args.algorithm!r}")
    if args.theta_policy != VALID_THETA_POLICY:
        raise ValueError(f"theta_policy debe ser {VALID_THETA_POLICY!r}, no {args.theta_policy!r}")
    if args.block_size < 1 or (args.block_size & (args.block_size - 1)) != 0:
        raise ValueError(f"block-size debe ser potencia de 2, no {args.block_size}")


def _gpu_search(inputs, A_w, T_w, S_w, F_w, accum, atol, rtol, block_size):
    """Lanza el kernel device y devuelve (best_units, best_k, grid, t_core, t_search, device)."""
    import pycuda.autoinit  # noqa: F401  (inicializa contexto)
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule

    with open(_DEVICE_SRC, "r", encoding="utf-8") as fh:
        src = fh.read()
    # no_extern_c=True: el .cuh tiene templates (no pueden ir dentro de extern "C") y marca sus
    # kernels como extern "C" explicitamente.
    mod = SourceModule(src, no_extern_c=True)
    fname = "search_kernel_f64" if accum == "float64" else "search_kernel_f32"
    func = mod.get_function(fname)

    M = inputs.A.shape[0]
    N = int(inputs.N)
    K = int(inputs.K)
    pos = inputs.pos_idx.astype(np.int32)
    neg = inputs.neg_idx.astype(np.int32)
    cand = np.ascontiguousarray(inputs.candidates_W, dtype=np.float32)

    d_A = cuda.mem_alloc(A_w.nbytes); d_T = cuda.mem_alloc(T_w.nbytes)
    d_S = cuda.mem_alloc(S_w.nbytes); d_F = cuda.mem_alloc(F_w.nbytes)
    d_pos = cuda.mem_alloc(pos.nbytes); d_neg = cuda.mem_alloc(neg.nbytes)
    d_cand = cuda.mem_alloc(cand.nbytes); d_best = cuda.mem_alloc(8)
    for dst, srcarr in ((d_A, A_w), (d_T, T_w), (d_S, S_w), (d_F, F_w), (d_pos, pos), (d_neg, neg)):
        cuda.memcpy_htod(dst, srcarr)

    grid = min((K + block_size - 1) // block_size, 32768)
    shmem = block_size * 8
    ev_s, ev_k0, ev_k1, ev_e = (cuda.Event() for _ in range(4))

    ev_s.record()
    cuda.memcpy_htod(d_cand, cand)
    cuda.memcpy_htod(d_best, np.zeros(1, dtype=np.uint64))
    ev_k0.record()
    func(np.int32(M), np.int32(N), d_A, d_T, d_S, d_F, d_cand,
         d_pos, np.int32(len(pos)), d_neg, np.int32(len(neg)),
         np.float64(atol), np.float64(rtol), np.int64(K), d_best,
         block=(block_size, 1, 1), grid=(grid, 1), shared=shmem)
    ev_k1.record()
    best_key = np.zeros(1, dtype=np.uint64)
    cuda.memcpy_dtoh(best_key, d_best)
    ev_e.record(); ev_e.synchronize()

    t_core = ev_k0.time_till(ev_k1) / 1000.0
    t_search = ev_s.time_till(ev_e) / 1000.0
    au, k = unpack_key(int(best_key[0]), K)
    device = cuda.Context.get_device().name()
    return au, k, grid, t_core, t_search, device


def run(args):
    _validate_cli(args)
    accum = MODE_ACCUM[args.mode]
    wd = np.dtype(accum)
    st = wd.type

    inputs = load_and_validate_inputs(
        args.matrix_a, args.profile_t, args.profile_s, args.profile_f,
        args.labels, args.candidates, args.N, args.K)
    A_w, T_w, S_w, F_w = prepare_work_arrays(inputs.A, inputs.T, inputs.S, inputs.F, accum)

    best_units, best_k, grid, t_core, t_search, device = _gpu_search(
        inputs, A_w, T_w, S_w, F_w, accum, args.tie_atol, args.tie_rtol, args.block_size)

    # Recompute del ganador en CPU (identico a sequential => mismos scores/theta/consistency).
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
        "cuda_block_size": int(args.block_size),
        "cuda_grid_size": int(grid),
        "device": device,
    }


def main(argv=None):
    args = _build_parser().parse_args(argv)
    try:
        result = run(args)
    except (ValueError, FileNotFoundError, AssertionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    line = json.dumps(result, ensure_ascii=False)
    print(line)
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as fh:
            fh.write(line + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
