"""Fixtures deterministas para las pruebas del generador.

Configuraciones deterministas + re-exporta el validador puro desde `data.genconfig`
(fuente única). La lógica de generación NO vive aquí.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.genconfig import GenConfigError, validate_generation_config  # noqa: E402,F401

_GEN_CONFIG_PATH = _ROOT / "generation_config.json"


# --------------------------------------------------------------------------- #
# Configuraciones deterministas
# --------------------------------------------------------------------------- #
def base_generation_config():
    """Devuelve la configuración base real (generation_config.json)."""
    with _GEN_CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def with_overrides(**overrides):
    """Configuración base con campos sobreescritos (copia)."""
    cfg = base_generation_config()
    cfg.update(overrides)
    return cfg


def neutral_config(N=None):
    """Modo neutral: signal_strength=0 y kappa=N (debe reproducir Dirichlet(1,...,1))."""
    base = base_generation_config()
    n = N if N is not None else base["N"]
    cfg = copy.deepcopy(base)
    cfg.update(N=n, signal_strength=0.0, kappa=n)
    return cfg


def signal_config(signal_strength=1.0, N=None):
    """Modo con señal (signal_strength > 0)."""
    base = base_generation_config()
    n = N if N is not None else base["N"]
    cfg = copy.deepcopy(base)
    cfg.update(N=n, signal_strength=signal_strength)
    return cfg


def tiny_config(N=4, signal_strength=1.0):
    """Configuración pequeña para pruebas rápidas. NO incluye K (se pasa explícito)."""
    cfg = copy.deepcopy(base_generation_config())
    cfg.update(N=N, signal_strength=signal_strength, kappa=N)
    return cfg


# --------------------------------------------------------------------------- #
# Formas y dtypes esperados de los arreglos producidos
# --------------------------------------------------------------------------- #
def expected_shapes(N, K):
    return {
        "A": (10, N),
        "T": (N,),
        "S": (N,),
        "F": (N,),
        "y": (10,),
        "candidates_W": (K, 3),
        "W_true": (3,),
    }


def expected_dtypes():
    return {
        "A": "float32",
        "T": "float32",
        "S": "float32",
        "F": "float32",
        "y": "int32",
        "candidates_W": "float32",
        "W_true": "float64",
    }
