#!/usr/bin/env python3
"""Launcher: única fuente de traducción de experiment_config.json a comandos CLI.

Lee `experiment_config.json` (fuente de verdad), valida sus campos y construye
—sin ejecutar por defecto— la línea de comandos de una implementación de scoring.
No requiere dependencias externas (solo biblioteca estándar).

En esta fase (estructura y configuración) el launcher NUNCA ejecuta una
implementación: las implementaciones todavía no existen. Solo valida y emite el
comando que se ejecutaría.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Límite de overflow de la clave del argmax: key = auc_units*K + (K-1-k);
# key_max = 51*K - 1 debe caber en int64 con signo.
INT64_MAX = 9223372036854775807
KEY_FACTOR = 51
VALID_ALGORITHM = "literal_fused"
DEFAULT_CONFIG = "experiment_config.json"
DEFAULT_GENERATION_CONFIG = "generation_config.json"

IMPL_PYTHON = {"python_sequential", "python_multicore"}
IMPL_NATIVE = {"c_serial", "openmp", "mpi", "cuda"}
ALL_IMPLS = IMPL_PYTHON | IMPL_NATIVE


class ConfigError(ValueError):
    """Error de validación de la configuración."""


def load_config(path=DEFAULT_CONFIG):
    """Carga y devuelve el JSON de configuración como dict."""
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _check_tol(d, label):
    if not isinstance(d, dict):
        raise ConfigError(f"{label} debe ser objeto con atol/rtol")
    for key in ("atol", "rtol"):
        v = d.get(key)
        if not _is_number(v) or v < 0:
            raise ConfigError(f"{label}.{key} debe ser un número >= 0")


def validate_config(cfg, mode=None):
    """Valida `cfg` y devuelve el modo resuelto. Lanza ConfigError si algo falla."""
    if not isinstance(cfg, dict):
        raise ConfigError("config debe ser un objeto JSON")

    N = cfg.get("N")
    K = cfg.get("K")
    if not isinstance(N, int) or isinstance(N, bool) or N < 1:
        raise ConfigError("N debe ser un entero >= 1")
    if not isinstance(K, int) or isinstance(K, bool) or K < 1:
        raise ConfigError("K debe ser un entero >= 1")
    if K > INT64_MAX // KEY_FACTOR:
        raise ConfigError(
            f"K={K} excede el límite de overflow de la clave int64 "
            f"(K <= {INT64_MAX // KEY_FACTOR})"
        )

    if cfg.get("algorithm") != VALID_ALGORITHM:
        raise ConfigError(f"algorithm debe ser '{VALID_ALGORITHM}'")

    allowed = cfg.get("allowed_modes")
    if not isinstance(allowed, list) or not allowed:
        raise ConfigError("allowed_modes debe ser una lista no vacía")

    resolved = mode if mode is not None else cfg.get("default_mode")
    if resolved not in allowed:
        raise ConfigError(f"modo desconocido: {resolved!r} (permitidos: {allowed})")

    accum = cfg.get("accumulation")
    if not isinstance(accum, dict):
        raise ConfigError("accumulation debe ser objeto")
    tie = cfg.get("tie_tolerance")
    if not isinstance(tie, dict):
        raise ConfigError("tie_tolerance debe ser objeto")
    for mk in allowed:
        if mk not in accum:
            raise ConfigError(f"falta accumulation para el modo {mk}")
        if mk not in tie:
            raise ConfigError(f"falta tie_tolerance para el modo {mk}")
        _check_tol(tie[mk], f"tie_tolerance.{mk}")

    cons = cfg.get("consistency_satisfactory")
    if not _is_number(cons) or not (0.0 <= cons <= 1.0):
        raise ConfigError("consistency_satisfactory debe estar en [0, 1]")

    if not isinstance(cfg.get("files"), dict):
        raise ConfigError("files debe ser objeto")
    if not isinstance(cfg.get("output"), dict):
        raise ConfigError("output debe ser objeto")

    return resolved


def validate_base_consistency(experiment_cfg, generation_cfg):
    """Verifica la coherencia entre las configuraciones base.

    Exige que `experiment_config.N` coincida con `generation_config.N`. Esta
    comprobación valida la configuración del proyecto antes de generar o ejecutar;
    no implica que los ejecutables de scoring deban abrir generation_config.json.

    Devuelve el N común. Lanza ConfigError con un mensaje claro si difieren.
    """
    if not isinstance(experiment_cfg, dict) or not isinstance(generation_cfg, dict):
        raise ConfigError("ambas configuraciones deben ser objetos JSON")
    n_exp = experiment_cfg.get("N")
    n_gen = generation_cfg.get("N")
    if n_exp != n_gen:
        raise ConfigError(
            "N incoherente entre configuraciones base: "
            f"experiment_config.N={n_exp} != generation_config.N={n_gen}"
        )
    return n_exp


def _fmt(v):
    """Formato determinista de un valor para la línea de comandos."""
    return repr(v) if isinstance(v, float) else str(v)


def build_command(cfg, impl, mode=None, processes=1):
    """Construye (sin ejecutar) la lista de tokens del comando para `impl`."""
    resolved = validate_config(cfg, mode)
    if impl not in ALL_IMPLS:
        raise ConfigError(f"implementación desconocida: {impl!r}")
    if not isinstance(processes, int) or isinstance(processes, bool) or processes < 1:
        raise ConfigError("processes debe ser un entero >= 1")

    files = cfg["files"]
    if impl in IMPL_PYTHON:
        a = files["matrix_A_npy"]
        t = files["profile_T_npy"]
        s = files["profile_S_npy"]
        f = files["profile_F_npy"]
        labels = files["labels_npy"]
        cand = files["candidates_npy"]
    else:
        a = files["matrix_A_bin"]
        t = files["profile_T_bin"]
        s = files["profile_S_bin"]
        f = files["profile_F_bin"]
        labels = files["labels_bin"]
        cand = files["candidates_bin"]

    prefix = {
        "python_sequential": ["python", "python/sequential.py"],
        "python_multicore": ["python", "python/multicore.py"],
        "c_serial": ["./C_OpenMP_MPI/scoring_serial"],
        "openmp": ["./C_OpenMP_MPI/scoring_openmp"],
        "mpi": ["mpirun", "-np", str(processes), "./C_OpenMP_MPI/scoring_mpi"],
        "cuda": ["./CUDA/scoring_kernel"],
    }[impl]

    tie = cfg["tie_tolerance"][resolved]
    accum = cfg["accumulation"][resolved]

    return prefix + [
        "--N", str(cfg["N"]),
        "--K", str(cfg["K"]),
        "--mode", resolved,
        "--accum", str(accum),
        "--algorithm", cfg["algorithm"],
        "--tie-atol", _fmt(tie["atol"]),
        "--tie-rtol", _fmt(tie["rtol"]),
        "--matrix-a", a,
        "--profile-t", t,
        "--profile-s", s,
        "--profile-f", f,
        "--labels", labels,
        "--candidates", cand,
        "--theta-policy", str(cfg["theta_policy"]),
        "--consistency-threshold", _fmt(cfg["consistency_satisfactory"]),
        "--output", cfg["output"]["benchmark_csv"],
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Launcher de scoring metagenómico (fase 1: solo construye comandos)."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Ruta de experiment_config.json")
    parser.add_argument("--impl", default="python_sequential", choices=sorted(ALL_IMPLS))
    parser.add_argument("--mode", default=None, help="reference | benchmark (por defecto, default_mode)")
    parser.add_argument("--processes", type=int, default=1, help="Nº de procesos MPI (solo impl=mpi)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Construye e imprime el comando sin ejecutarlo (comportamiento por defecto en esta fase).",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    cmd = build_command(cfg, args.impl, args.mode, args.processes)
    print(" ".join(cmd))
    # Fase 1: nunca se ejecuta ninguna implementación (los binarios aún no existen).
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
