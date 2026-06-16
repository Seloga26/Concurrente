"""Validador puro de la configuración del generador (sin numpy).

Fuente única de `validate_generation_config` y `GenConfigError`. Lo usan el
generador (`data/generate_data.py`) y las pruebas (`tests/fixtures.py`). Mantenerlo
aquí (no en el código de test) evita que producción importe código de prueba y deja
las pruebas de configuración independientes de bugs del generador.
"""
from __future__ import annotations


class GenConfigError(ValueError):
    """Error de validación de la configuración del generador."""


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def validate_generation_config(cfg):
    """Valida la configuración del generador. Devuelve True o lanza GenConfigError.

    No exige n_healthy/n_sick == 5 (ese contrato del PDF lo impone `generate_dataset`
    mediante `enforce_main_counts`); aquí solo se exige positividad.
    """
    if not isinstance(cfg, dict):
        raise GenConfigError("config debe ser un objeto JSON")

    N = cfg.get("N")
    if not _is_int(N) or N < 2:
        raise GenConfigError("N debe ser un entero >= 2")

    for key in ("n_healthy", "n_sick"):
        v = cfg.get(key)
        if not _is_int(v) or v < 1:
            raise GenConfigError(f"{key} debe ser un entero >= 1")

    kappa = cfg.get("kappa")
    if not _is_number(kappa) or kappa <= 0:
        raise GenConfigError("kappa debe ser > 0")

    s = cfg.get("signal_strength")
    if not _is_number(s) or s < 0:
        raise GenConfigError("signal_strength debe ser >= 0")

    p_F = cfg.get("p_F")
    if not _is_number(p_F) or not (0.0 <= p_F <= 1.0):
        raise GenConfigError("p_F debe estar en [0, 1]")

    eps = cfg.get("eps_noise")
    if not _is_number(eps) or not (0.0 <= eps < 1.0):
        raise GenConfigError("eps_noise debe estar en [0, 1)")

    for key in ("data_seed", "candidate_seed"):
        if not _is_int(cfg.get(key)):
            raise GenConfigError(f"{key} debe ser un entero")
    if "holdout_seed" in cfg and not _is_int(cfg["holdout_seed"]):
        raise GenConfigError("holdout_seed debe ser un entero")

    for key in ("holdout_n_healthy", "holdout_n_sick"):
        if key in cfg:
            v = cfg[key]
            if not _is_int(v) or v < 1:
                raise GenConfigError(f"{key} debe ser un entero >= 1")

    if not isinstance(cfg.get("F_binary"), bool):
        raise GenConfigError("F_binary debe ser booleano")

    return True
