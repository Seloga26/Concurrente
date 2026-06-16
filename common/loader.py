"""Carga y validación de inputs del scoring (fuera de la región cronometrada).

NO lee `generation_metadata.json`, `generation_truth.npz` ni `W_true`: el scoring solo
consume A, T, S, F, y, candidates_W.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass
class ScoringInputs:
    A: np.ndarray
    T: np.ndarray
    S: np.ndarray
    F: np.ndarray
    y: np.ndarray
    candidates_W: np.ndarray
    pos_idx: np.ndarray
    neg_idx: np.ndarray
    N: int
    K: int


def _require(cond, msg):
    if not cond:
        raise ValueError(msg)


def load_and_validate_inputs(matrix_a, profile_t, profile_s, profile_f, labels, candidates,
                             N, K, n_healthy=5, n_sick=5, simplex_atol=1e-4, profile_atol=1e-4):
    n_samples = n_healthy + n_sick
    _require(isinstance(N, int) and not isinstance(N, bool) and N >= 1, f"N debe ser entero >= 1: {N}")
    _require(isinstance(K, int) and not isinstance(K, bool) and K >= 1, f"K debe ser entero >= 1: {K}")

    paths = {
        "matrix_a": matrix_a, "profile_t": profile_t, "profile_s": profile_s,
        "profile_f": profile_f, "labels": labels, "candidates": candidates,
    }
    for name, p in paths.items():
        _require(os.path.exists(p), f"falta el archivo de entrada '{name}': {p}")

    A = np.load(matrix_a)
    T = np.load(profile_t)
    S = np.load(profile_s)
    F = np.load(profile_f)
    y = np.load(labels)
    cand = np.load(candidates)

    _require(A.shape == (n_samples, N) and A.dtype == np.float32,
             f"A debe ser ({n_samples},{N}) float32; es {A.shape} {A.dtype}")
    for nm, arr in (("T", T), ("S", S), ("F", F)):
        _require(arr.shape == (N,) and arr.dtype == np.float32,
                 f"{nm} debe ser ({N},) float32; es {arr.shape} {arr.dtype}")
    _require(y.shape == (n_samples,) and y.dtype == np.int32,
             f"y debe ser ({n_samples},) int32; es {y.shape} {y.dtype}")
    _require(cand.shape == (K, 3) and cand.dtype == np.float32,
             f"candidates_W debe ser ({K},3) float32; es {cand.shape} {cand.dtype}")

    expected_y = np.array([0] * n_healthy + [1] * n_sick, dtype=np.int32)
    _require(np.array_equal(y, expected_y),
             f"y debe ser {n_healthy} ceros seguidos de {n_sick} unos")

    for nm, arr in (("A", A), ("T", T), ("S", S), ("F", F), ("candidates_W", cand)):
        _require(np.all(np.isfinite(arr)), f"{nm} contiene valores no finitos")

    _require(np.all(A >= 0), "A tiene entradas negativas")
    _require(np.allclose(A.sum(axis=1), 1.0, atol=1e-5), "las filas de A no suman 1")
    _require(np.all(cand >= 0), "candidates_W tiene entradas negativas")
    _require(np.allclose(cand.sum(axis=1), 1.0, atol=simplex_atol), "los candidatos no suman 1")

    # T, S, F en [0,1] (con tolerancia); F NO se exige estrictamente binario.
    for nm, arr in (("T", T), ("S", S), ("F", F)):
        _require(np.all(arr >= -profile_atol) and np.all(arr <= 1.0 + profile_atol),
                 f"{nm} fuera de [0,1]")

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    return ScoringInputs(A, T, S, F, y, cand, pos_idx, neg_idx, N, K)
