#!/usr/bin/env python3
"""Emite los tokens CLI COMPARTIDOS de una corrida de scoring, leidos de experiment_config.json.

Mantiene experiment_config.json como unica fuente de N, tolerancias, theta_policy, umbral de
consistencia y rutas de los .npy. run_all.sh captura esta salida y le anade `--K $K` y el flag de
paralelismo por implementacion. NO incluye --K (varia en el barrido) ni el prefijo del binario.

Uso:  python scripts/bench_args.py --mode reference   # imprime una linea con los tokens
"""
from __future__ import annotations

import argparse
import os
import sys

# Raiz del repo en sys.path para importar config.launcher (scripts/ no es paquete).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.launcher import load_config, validate_config  # noqa: E402


def _fmt(v):
    """repr() para floats (precision completa, sin redondeo); str() para el resto."""
    return repr(v) if isinstance(v, float) else str(v)


def shared_args(cfg, mode):
    """Lista de tokens CLI compartidos (sin --K ni prefijo del binario)."""
    resolved = validate_config(cfg, mode)
    accum = cfg["accumulation"][resolved]
    tie = cfg["tie_tolerance"][resolved]
    files = cfg["files"]
    return [
        "--N", str(cfg["N"]),
        "--mode", resolved,
        "--accum", str(accum),
        "--algorithm", str(cfg["algorithm"]),
        "--tie-atol", _fmt(tie["atol"]),
        "--tie-rtol", _fmt(tie["rtol"]),
        "--matrix-a", files["matrix_A_npy"],
        "--profile-t", files["profile_T_npy"],
        "--profile-s", files["profile_S_npy"],
        "--profile-f", files["profile_F_npy"],
        "--labels", files["labels_npy"],
        "--candidates", files["candidates_npy"],
        "--theta-policy", str(cfg["theta_policy"]),
        "--consistency-threshold", _fmt(cfg["consistency_satisfactory"]),
    ]


def main(argv=None):
    p = argparse.ArgumentParser(description="Emite los args CLI compartidos del scoring.")
    p.add_argument("--mode", required=True, choices=["reference", "benchmark"])
    p.add_argument("--config", default=None, help="Ruta de experiment_config.json (default: la del repo).")
    args = p.parse_args(argv)
    cfg = load_config(args.config) if args.config else load_config()
    print(" ".join(shared_args(cfg, args.mode)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
