"""Métricas posteriores al cronómetro: umbral theta y consistencia (balanced accuracy)."""
from __future__ import annotations

import numpy as np


def compute_theta(scores, pos_idx, neg_idx):
    """Umbral de ajuste: punto medio de las medias por clase."""
    s = np.asarray(scores)
    sp = s[np.asarray(pos_idx, dtype=np.intp)]
    sn = s[np.asarray(neg_idx, dtype=np.intp)]
    if sp.size == 0 or sn.size == 0:
        raise ValueError("compute_theta requiere ambas clases")
    return 0.5 * (float(sp.mean()) + float(sn.mean()))


def consistency(scores, theta, pos_idx, neg_idx):
    """Balanced accuracy: 0.5*(TP/n_pos + TN/n_neg). Enfermo: score>theta; sano: score<=theta."""
    s = np.asarray(scores)
    sp = s[np.asarray(pos_idx, dtype=np.intp)]
    sn = s[np.asarray(neg_idx, dtype=np.intp)]
    if sp.size == 0 or sn.size == 0:
        raise ValueError("consistency requiere ambas clases")
    tp = int(np.count_nonzero(sp > theta))
    tn = int(np.count_nonzero(sn <= theta))
    return 0.5 * (tp / sp.size + tn / sn.size)
