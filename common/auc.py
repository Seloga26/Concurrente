"""AUC por comparaciones de pares (portable, sin scikit-learn).

Dos APIs:
- `pairwise_auc_units(...)` -> int : NÚCLEO para `search_best`. Doble bucle escalar 5×5,
  sin matrices temporales ni objetos por candidato. Devuelve solo `auc_units`.
- `pairwise_auc(...)` -> AucResult : DETALLADO (tests, diagnóstico, recompute del ganador).

Convención: `wins` = pos > neg fuera de la banda; `ties` = |pos-neg| <= banda;
`auc_units = 2*wins + ties`; `denominator = 2*n_pos*n_neg`; `auc = auc_units/denominator`.
Banda de empate: `atol + rtol * max(|sp|, |sn|)`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AucResult:
    auc_units: int
    denominator: int
    auc: float
    wins: int
    ties: int


def _validate_classes(n_pos, n_neg):
    if n_pos < 1 or n_neg < 1:
        raise ValueError(f"AUC requiere ambas clases (n_pos={n_pos}, n_neg={n_neg})")


def pairwise_auc_units(scores, pos_idx, neg_idx, atol, rtol):
    """Núcleo escalar: devuelve `auc_units` (int) sin asignaciones por candidato."""
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)
    _validate_classes(n_pos, n_neg)
    pos = [float(scores[i]) for i in pos_idx]
    neg = [float(scores[j]) for j in neg_idx]
    wins = 0
    ties = 0
    for a in pos:
        aa = a if a >= 0.0 else -a
        for b in neg:
            ab = b if b >= 0.0 else -b
            band = atol + rtol * (aa if aa > ab else ab)
            d = a - b
            if d > band:
                wins += 1
            elif d >= -band:        # |d| <= band
                ties += 1
            # else: loss (no suma)
    return 2 * wins + ties


def pairwise_auc(scores, pos_idx, neg_idx, atol, rtol):
    """Detallado: AucResult con wins/ties/auc_units/denominator/auc (vectorizado)."""
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)
    _validate_classes(n_pos, n_neg)
    s = np.asarray(scores)
    sp = s[np.asarray(pos_idx)]
    sn = s[np.asarray(neg_idx)]
    d = sp[:, None] - sn[None, :]
    band = atol + rtol * np.maximum(np.abs(sp)[:, None], np.abs(sn)[None, :])
    ties = int(np.count_nonzero(np.abs(d) <= band))
    wins = int(np.count_nonzero(d > band))
    auc_units = 2 * wins + ties
    denominator = 2 * n_pos * n_neg
    return AucResult(auc_units=auc_units, denominator=denominator,
                     auc=auc_units / denominator, wins=wins, ties=ties)
