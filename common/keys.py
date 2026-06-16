"""Clave int64 para el argmax determinista (reutilizable por OpenMP/MPI/CUDA).

**Solo biblioteca estándar**: la capa matemática común NO depende de la capa CLI.
`config.launcher` importa `INT64_MAX`/`KEY_FACTOR` desde aquí (nunca al revés), por lo
que no hay ciclo de imports (`config.launcher -> common.keys -> stdlib`).

Clave: `key = auc_units*K + (K-1-k)`. Maximizar la clave maximiza `auc_units` y, ante
empate, **minimiza k** (menor índice global). El caso principal 5/5 tiene
`auc_units ∈ [0, MAIN_MAX_AUC_UNITS]`; `pack_key` se generaliza con `max_auc_units`.
"""
from __future__ import annotations

INT64_MAX = 9223372036854775807
MAIN_MAX_AUC_UNITS = 50           # solo válido para el dataset principal 5/5
KEY_FACTOR = MAIN_MAX_AUC_UNITS + 1   # = 51
KEY_SENTINEL = -1                 # "sin candidato": pierde ante cualquier candidato válido


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def validate_K(K, max_auc_units=MAIN_MAX_AUC_UNITS):
    if not _is_int(max_auc_units) or max_auc_units < 0:
        raise ValueError(f"max_auc_units debe ser entero >= 0: {max_auc_units}")
    if not _is_int(K) or K < 1:
        raise ValueError(f"K debe ser entero >= 1: {K}")
    limit = INT64_MAX // (max_auc_units + 1)
    if K > limit:
        raise ValueError(f"K={K} excede el límite de overflow de la clave int64 (K <= {limit})")
    return K


def pack_key(auc_units, k, K, max_auc_units=MAIN_MAX_AUC_UNITS):
    validate_K(K, max_auc_units)
    if not _is_int(auc_units) or not (0 <= auc_units <= max_auc_units):
        raise ValueError(f"auc_units fuera de [0,{max_auc_units}]: {auc_units}")
    if not _is_int(k) or not (0 <= k < K):
        raise ValueError(f"k fuera de [0,{K}): {k}")
    return auc_units * K + (K - 1 - k)


def unpack_key(key, K):
    if not _is_int(K) or K < 1:
        raise ValueError(f"K debe ser entero >= 1: {K}")
    if not _is_int(key) or key < 0:
        raise ValueError(f"key inválida (negativa o centinela): {key}")
    auc_units = key // K
    k = K - 1 - (key % K)
    return auc_units, k


def is_sentinel(key):
    return key == KEY_SENTINEL


def better(units_a, k_a, units_b, k_b):
    """¿(units_a,k_a) es mejor que (units_b,k_b)? Mayor units; empate -> menor k.

    El centinela se modela con `units = -1`, por lo que pierde ante cualquier
    candidato válido (`units >= 0`).
    """
    if units_a != units_b:
        return units_a > units_b
    return k_a < k_b
