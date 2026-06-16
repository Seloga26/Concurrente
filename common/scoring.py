"""Núcleo de scoring literal (P materializada) y búsqueda secuencial.

`algorithm = "literal"` (kernel_variant Python = "materialized_numpy"): `P = w0*T+w1*S+w2*F`,
`Score = A @ P`, por candidato. La precisión la fija el dtype de trabajo (f64 reference /
f32 benchmark). Para evitar promociones y allocaciones, `search_best` extrae los tres pesos
del candidato como **escalares** del dtype de trabajo (sin crear un arreglo por iteración).
"""
from __future__ import annotations

import numpy as np

from common.auc import pairwise_auc_units


def prepare_work_arrays(A, T, S, F, accum_dtype):
    """Castea A,T,S,F al dtype de acumulación una sola vez (fuera del cronómetro)."""
    dt = np.dtype(accum_dtype)
    return (np.ascontiguousarray(A, dtype=dt),
            np.ascontiguousarray(T, dtype=dt),
            np.ascontiguousarray(S, dtype=dt),
            np.ascontiguousarray(F, dtype=dt))


def materialize_p(w0, w1, w2, T, S, F):
    """P = w0*T + w1*S + w2*F (w* escalares NumPy del dtype de T/S/F)."""
    return w0 * T + w1 * S + w2 * F


def score_candidate(w0, w1, w2, A, T, S, F):
    """Score = A @ P (para tests y recompute del ganador)."""
    return A @ materialize_p(w0, w1, w2, T, S, F)


def search_best(A_w, T_w, S_w, F_w, candidates, pos_idx, neg_idx, atol, rtol, work_dtype):
    """NÚCLEO CRONOMETRADO. Devuelve (best_auc_units, best_k); solo enteros, sin guardar scores."""
    st = np.dtype(work_dtype).type
    K = candidates.shape[0]
    best_units = -1
    best_k = -1
    for k in range(K):
        w0 = st(candidates[k, 0])
        w1 = st(candidates[k, 1])
        w2 = st(candidates[k, 2])
        score = A_w @ (w0 * T_w + w1 * S_w + w2 * F_w)
        au = pairwise_auc_units(score, pos_idx, neg_idx, atol, rtol)
        if au > best_units:        # '>' estricto => ante empate conserva el menor k
            best_units = au
            best_k = k
    return best_units, best_k
