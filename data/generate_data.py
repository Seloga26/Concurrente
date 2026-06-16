#!/usr/bin/env python3
"""Generador sintético de datos del scoring metagenómico.

Produce A, T, S, F, y, candidates_W + verdad oculta (W_true, r, m_san, m_enf) y
metadatos. Reproducible (streams RNG separados vía SeedSequence), escritura
atómica por archivo y metadatos sin vectores de longitud N.

Modelo de señal: r_i = W_true·(T_i,S_i,F_i); tilt softmax(±s·r̃) por clase, mezcla
opcional con uniforme (eps_noise), Dirichlet(kappa·m) por fila de A.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.launcher import (  # noqa: E402
    INT64_MAX,
    KEY_FACTOR,
    load_config,
    validate_base_consistency,
)
from data.genconfig import GenConfigError, validate_generation_config  # noqa: E402

# Contrato del PDF para el dataset principal.
N_MAIN_HEALTHY = 5
N_MAIN_SICK = 5
# Piso de probabilidad (ver plan §C): ~epsilon de máquina, no-op en modo normal.
M_FLOOR = 1e-15
# Umbral degenerado relativo a la magnitud de r.
STD_REL_EPS = 1e-12
SCHEMA_VERSION = "1.0"
ALGORITHM = "dirichlet_tilt_softmax_v1"


@dataclass
class GeneratedDataset:
    A: np.ndarray              # (n_samples, N) float32
    T: np.ndarray              # (N,) float32
    S: np.ndarray              # (N,) float32
    F: np.ndarray              # (N,) float32
    y: np.ndarray              # (n_samples,) int32
    W_true: np.ndarray         # (3,) float64
    r: np.ndarray              # (N,) float64
    m_san: np.ndarray          # (N,) float64
    m_enf: np.ndarray          # (N,) float64
    candidates_W: Optional[np.ndarray] = None   # (K,3) float32 ; None en hold-out / sin K
    config: Optional[dict] = None               # config efectiva usada


# --------------------------------------------------------------------------- #
# Helpers puros
# --------------------------------------------------------------------------- #
def softmax_stable(z):
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 1 or z.size == 0:
        raise ValueError("softmax_stable: z debe ser 1-D no vacío")
    if not np.all(np.isfinite(z)):
        raise ValueError("softmax_stable: z contiene valores no finitos")
    e = np.exp(z - z.max())
    return e / e.sum()


def latent_risk(T, S, F, W_true):
    T = np.asarray(T, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    F = np.asarray(F, dtype=np.float64)
    W_true = np.asarray(W_true, dtype=np.float64)
    if T.ndim != 1 or T.size == 0:
        raise ValueError("latent_risk: T,S,F deben ser 1-D no vacíos")
    if not (T.shape == S.shape == F.shape):
        raise ValueError("latent_risk: T,S,F deben tener la misma forma")
    if W_true.shape != (3,):
        raise ValueError("latent_risk: W_true debe tener forma (3,)")
    for name, arr in (("T", T), ("S", S), ("F", F), ("W_true", W_true)):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"latent_risk: {name} contiene valores no finitos")
    return W_true[0] * T + W_true[1] * S + W_true[2] * F


def dirichlet_alpha(m, kappa):
    m = np.asarray(m, dtype=np.float64)
    if m.ndim != 1 or m.size == 0:
        raise ValueError("dirichlet_alpha: m debe ser 1-D no vacío")
    if not (isinstance(kappa, (int, float)) and not isinstance(kappa, bool)) or kappa <= 0:
        raise ValueError("dirichlet_alpha: kappa debe ser > 0")
    if np.any(m < 0):
        raise ValueError("dirichlet_alpha: m no puede tener entradas negativas")
    return kappa * m


def class_means(r, signal_strength, eps_noise=0.0):
    """Devuelve (m_san, m_enf), medias teóricas por clase. Protección numérica incluida."""
    r = np.asarray(r, dtype=np.float64)
    if r.ndim != 1 or r.size == 0:
        raise ValueError("class_means: r debe ser 1-D no vacío")
    if not np.all(np.isfinite(r)):
        raise ValueError("class_means: r contiene valores no finitos")
    N = r.size
    r_scale = max(1.0, float(np.max(np.abs(r))))
    if float(np.std(r)) <= STD_REL_EPS * r_scale:        # degenerado
        r_tilde = np.zeros(N)
    else:
        r_tilde = (r - r.mean()) / r.std()

    def protected(sign):
        m = softmax_stable(sign * signal_strength * r_tilde)
        if eps_noise > 0:
            m = (1.0 - eps_noise) * m + eps_noise * (np.ones(N) / N)
        m = np.maximum(m, M_FLOOR)                        # piso positivo
        m = m / m.sum()                                  # renormaliza a suma 1
        if not np.all(np.isfinite(m)):
            raise ValueError("class_means: m no finito")
        return m

    return protected(-1.0), protected(+1.0)


def make_labels(n_healthy, n_sick):
    if n_healthy < 1 or n_sick < 1:
        raise ValueError("make_labels: n_healthy y n_sick deben ser >= 1")
    return np.array([0] * n_healthy + [1] * n_sick, dtype=np.int32)


# --------------------------------------------------------------------------- #
# Helpers con RNG
# --------------------------------------------------------------------------- #
def sample_profiles(N, p_F, F_binary, rng):
    if not (isinstance(N, int) and not isinstance(N, bool)) or N < 1:
        raise ValueError("sample_profiles: N debe ser un entero >= 1")
    T = rng.uniform(0.0, 1.0, size=N).astype(np.float32)
    S = rng.uniform(0.0, 1.0, size=N).astype(np.float32)
    if F_binary:
        F = (rng.random(size=N) < p_F).astype(np.float32)
    else:
        F = rng.uniform(0.0, 1.0, size=N).astype(np.float32)
    return T, S, F


def sample_w_true(rng):
    return rng.dirichlet(np.ones(3)).astype(np.float64)


def sample_A(m_san, m_enf, kappa, n_healthy, n_sick, rng):
    if n_healthy < 1 or n_sick < 1:
        raise ValueError("sample_A: n_healthy y n_sick deben ser >= 1")
    alpha_san = dirichlet_alpha(m_san, kappa)
    alpha_enf = dirichlet_alpha(m_enf, kappa)
    for name, a in (("alpha_san", alpha_san), ("alpha_enf", alpha_enf)):
        if not (np.all(np.isfinite(a)) and np.all(a > 0)):
            raise ValueError(f"sample_A: {name} no finito o no positivo")
    N = alpha_san.size
    A = np.empty((n_healthy + n_sick, N), dtype=np.float64)
    A[:n_healthy] = rng.dirichlet(alpha_san, size=n_healthy)
    A[n_healthy:] = rng.dirichlet(alpha_enf, size=n_sick)
    return A.astype(np.float32)


def generate_candidates(K, seed):
    if not (isinstance(K, int) and not isinstance(K, bool)) or K < 1:
        raise GenConfigError("K debe ser un entero >= 1")
    if K > INT64_MAX // KEY_FACTOR:
        raise GenConfigError(
            f"K={K} excede el límite de overflow de la clave int64 (K <= {INT64_MAX // KEY_FACTOR})"
        )
    rng = np.random.default_rng(seed)
    return rng.dirichlet(np.ones(3), size=K).astype(np.float32)


# --------------------------------------------------------------------------- #
# Orquestación
# --------------------------------------------------------------------------- #
def generate_dataset(config, K=None, enforce_main_counts=True):
    """Genera el dataset principal. Nunca lee config['K']; K se pasa explícito."""
    validate_generation_config(config)
    n_healthy = config["n_healthy"]
    n_sick = config["n_sick"]
    if enforce_main_counts and (n_healthy != N_MAIN_HEALTHY or n_sick != N_MAIN_SICK):
        raise GenConfigError(
            f"El dataset principal exige {N_MAIN_HEALTHY}/{N_MAIN_SICK} (PDF); "
            f"recibido {n_healthy}/{n_sick}. Usa enforce_main_counts=False para otros tamaños."
        )

    N = config["N"]
    ss = np.random.SeedSequence(config["data_seed"])
    ss_prof, ss_wtrue, ss_A = ss.spawn(3)
    rng_prof = np.random.default_rng(ss_prof)
    rng_wtrue = np.random.default_rng(ss_wtrue)
    rng_A = np.random.default_rng(ss_A)

    T, S, F = sample_profiles(N, config["p_F"], config["F_binary"], rng_prof)
    W_true = sample_w_true(rng_wtrue)
    r = latent_risk(T, S, F, W_true)
    m_san, m_enf = class_means(r, config["signal_strength"], config.get("eps_noise", 0.0))
    A = sample_A(m_san, m_enf, config["kappa"], n_healthy, n_sick, rng_A)
    y = make_labels(n_healthy, n_sick)

    candidates_W = None
    if K is not None:
        candidates_W = generate_candidates(K, config["candidate_seed"])

    return GeneratedDataset(
        A=A, T=T, S=S, F=F, y=y, W_true=W_true, r=r,
        m_san=m_san, m_enf=m_enf, candidates_W=candidates_W, config=dict(config),
    )


def generate_holdout(dataset, holdout_seed, holdout_n_healthy=5, holdout_n_sick=5):
    """Hold-out: reusa T,S,F,W_true,r,m_*; A,y nuevos; nunca genera candidatos."""
    rng_hold = np.random.default_rng(holdout_seed)
    A = sample_A(dataset.m_san, dataset.m_enf, dataset.config["kappa"],
                 holdout_n_healthy, holdout_n_sick, rng_hold)
    y = make_labels(holdout_n_healthy, holdout_n_sick)
    return GeneratedDataset(
        A=A, T=dataset.T, S=dataset.S, F=dataset.F, y=y, W_true=dataset.W_true,
        r=dataset.r, m_san=dataset.m_san, m_enf=dataset.m_enf,
        candidates_W=None, config=dataset.config,
    )


# --------------------------------------------------------------------------- #
# Hashes y metadatos
# --------------------------------------------------------------------------- #
def _file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _array_content_sha256(arr):
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(str(arr.shape).encode("utf-8"))
    h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def build_metadata(dataset, file_hashes, truth_content_hashes=None):
    cfg = dataset.config or {}
    N = int(dataset.T.shape[0])
    K = int(dataset.candidates_W.shape[0]) if dataset.candidates_W is not None else None
    row_sums = dataset.A.sum(axis=1)
    shapes = {k: list(getattr(dataset, k).shape) for k in ("A", "T", "S", "F", "y", "W_true")}
    dtypes = {k: str(getattr(dataset, k).dtype) for k in ("A", "T", "S", "F", "y", "W_true")}
    if dataset.candidates_W is not None:
        shapes["candidates_W"] = list(dataset.candidates_W.shape)
        dtypes["candidates_W"] = str(dataset.candidates_W.dtype)

    meta = {
        "schema_version": SCHEMA_VERSION,
        "algorithm": ALGORITHM,
        "params": {
            "N": N,
            "signal_strength": cfg.get("signal_strength"),
            "kappa": cfg.get("kappa"),
            "p_F": cfg.get("p_F"),
            "F_binary": cfg.get("F_binary"),
            "eps_noise": cfg.get("eps_noise"),
            "n_healthy": cfg.get("n_healthy"),
            "n_sick": cfg.get("n_sick"),
            "K": K,
            "n_samples": int(dataset.A.shape[0]),
        },
        "seeds": {
            "data_seed": cfg.get("data_seed"),
            "candidate_seed": cfg.get("candidate_seed"),
            "holdout_seed": cfg.get("holdout_seed"),
        },
        "W_true": [float(x) for x in dataset.W_true],
        "shapes": shapes,
        "dtypes": dtypes,
        "files": sorted(file_hashes.keys()),
        "hashes": dict(file_hashes),
        "summary": {
            "A_row_sum_min": float(row_sums.min()),
            "A_row_sum_max": float(row_sums.max()),
            "F_fraction_ones": float(np.mean(dataset.F)) if dataset.F.size else 0.0,
            "r_mean": float(np.mean(dataset.r)),
            "r_std": float(np.std(dataset.r)),
            "score_mean_healthy_theoretical": float(np.dot(dataset.m_san, dataset.r)),
            "score_mean_sick_theoretical": float(np.dot(dataset.m_enf, dataset.r)),
        },
    }
    if truth_content_hashes is not None:
        meta["truth_content_hashes"] = dict(truth_content_hashes)
    return meta


# --------------------------------------------------------------------------- #
# Escritura atómica (por archivo, NO transacción multiarchivo)
# --------------------------------------------------------------------------- #
def write_outputs(dataset, output_dir, overwrite=False, write_truth=False, holdout=None):
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    pid = os.getpid()

    arrays = {
        "matrix_A.npy": dataset.A,
        "profile_T.npy": dataset.T,
        "profile_S.npy": dataset.S,
        "profile_F.npy": dataset.F,
        "labels.npy": dataset.y,
    }
    if dataset.candidates_W is not None:
        arrays["candidates_W.npy"] = dataset.candidates_W
    if holdout is not None:
        arrays["matrix_A_holdout.npy"] = holdout.A
        arrays["labels_holdout.npy"] = holdout.y

    truth_arrays = None
    if write_truth:
        truth_arrays = {
            "r": dataset.r, "m_san": dataset.m_san,
            "m_enf": dataset.m_enf, "W_true": dataset.W_true,
        }

    meta_name = "generation_metadata.json"
    truth_name = "generation_truth.npz"

    # Preflight de TODOS los destinos.
    final_names = list(arrays.keys()) + [meta_name]
    if truth_arrays is not None:
        final_names.append(truth_name)
    if not overwrite:
        existing = sorted(n for n in final_names if (output_dir / n).exists())
        if existing:
            raise FileExistsError("Ya existen (usa overwrite=True): " + ", ".join(existing))

    temps = []
    written = {}
    try:
        # Escribir TODOS los temporales primero (datos).
        npy_temps = []
        for i, (name, arr) in enumerate(arrays.items()):
            tmp = output_dir / f"{name}.tmp-{pid}-{i}"
            temps.append(tmp)
            with open(tmp, "wb") as f:
                np.save(f, arr)
            npy_temps.append((name, tmp))
        truth_tmp = None
        if truth_arrays is not None:
            truth_tmp = output_dir / f"{truth_name}.tmp-{pid}-truth"
            temps.append(truth_tmp)
            with open(truth_tmp, "wb") as f:
                np.savez(f, **truth_arrays)

        # Reemplazar todos los de datos.
        for name, tmp in npy_temps:
            os.replace(tmp, output_dir / name)
            temps.remove(tmp)
            written[name] = output_dir / name
        if truth_tmp is not None:
            os.replace(truth_tmp, output_dir / truth_name)
            temps.remove(truth_tmp)
            written[truth_name] = output_dir / truth_name

        # Hashes: archivo para .npy, contenido para arreglos del truth.
        file_hashes = {name: _file_sha256(output_dir / name) for name in arrays.keys()}
        truth_content_hashes = None
        if truth_arrays is not None:
            truth_content_hashes = {k: _array_content_sha256(v) for k, v in truth_arrays.items()}

        # Metadatos el último (atómico, determinista, sin timestamps).
        meta = build_metadata(dataset, file_hashes, truth_content_hashes)
        meta_tmp = output_dir / f"{meta_name}.tmp-{pid}-meta"
        temps.append(meta_tmp)
        text = json.dumps(meta, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
        with open(meta_tmp, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(meta_tmp, output_dir / meta_name)
        temps.remove(meta_tmp)
        written[meta_name] = output_dir / meta_name

        return {name: str(p) for name, p in written.items()}
    finally:
        for tmp in temps:
            try:
                if tmp.exists():
                    os.remove(tmp)
            except OSError:
                pass


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _build_arg_parser():
    p = argparse.ArgumentParser(description="Generador sintético de datos del scoring metagenómico.")
    p.add_argument("--generation-config", default=str(_ROOT / "generation_config.json"))
    p.add_argument("--experiment-config", default=str(_ROOT / "experiment_config.json"))
    p.add_argument("--output-dir", default=str(_ROOT / "data"))
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--write-truth", action="store_true")
    p.add_argument("--generate-holdout", dest="generate_holdout", action="store_true", default=None)
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--data-seed", type=int, default=None)
    p.add_argument("--candidate-seed", type=int, default=None)
    p.add_argument("--signal-strength", type=float, default=None)
    p.add_argument("--kappa", type=float, default=None)
    p.add_argument("--p-f", dest="p_f", type=float, default=None)
    fbin = p.add_mutually_exclusive_group()
    fbin.add_argument("--f-binary", dest="f_binary", action="store_true", default=None)
    fbin.add_argument("--no-f-binary", dest="f_binary", action="store_false", default=None)
    p.add_argument("--eps-noise", type=float, default=None)
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--holdout-seed", type=int, default=None)
    p.add_argument("--holdout-n-healthy", type=int, default=None)
    p.add_argument("--holdout-n-sick", type=int, default=None)
    return p


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    gen_cfg = load_config(args.generation_config)
    exp_cfg = load_config(args.experiment_config)

    # Override de N (opción B): aplica a AMBAS configs en memoria.
    if args.n is not None:
        gen_cfg["N"] = args.n
        exp_cfg["N"] = args.n

    for key, val in (
        ("data_seed", args.data_seed),
        ("candidate_seed", args.candidate_seed),
        ("signal_strength", args.signal_strength),
        ("kappa", args.kappa),
        ("p_F", args.p_f),
        ("eps_noise", args.eps_noise),
        ("holdout_seed", args.holdout_seed),
        ("holdout_n_healthy", args.holdout_n_healthy),
        ("holdout_n_sick", args.holdout_n_sick),
    ):
        if val is not None:
            gen_cfg[key] = val
    if args.f_binary is not None:
        gen_cfg["F_binary"] = args.f_binary
    if args.generate_holdout is not None:
        gen_cfg["generate_holdout"] = args.generate_holdout

    validate_generation_config(gen_cfg)
    validate_base_consistency(exp_cfg, gen_cfg)

    K = args.k if args.k is not None else exp_cfg.get("K")
    if not (isinstance(K, int) and not isinstance(K, bool)) or K < 1:
        raise GenConfigError("K debe ser un entero >= 1 (de --k o experiment_config['K'])")
    if K > INT64_MAX // KEY_FACTOR:
        raise GenConfigError(f"K={K} excede el límite de overflow de la clave int64")

    dataset = generate_dataset(gen_cfg, K=K, enforce_main_counts=True)

    holdout = None
    if gen_cfg.get("generate_holdout"):
        holdout = generate_holdout(
            dataset,
            gen_cfg.get("holdout_seed"),
            gen_cfg.get("holdout_n_healthy", 5),
            gen_cfg.get("holdout_n_sick", 5),
        )

    paths = write_outputs(
        dataset, args.output_dir,
        overwrite=args.overwrite, write_truth=args.write_truth, holdout=holdout,
    )
    print(f"Generado en {args.output_dir}:")
    for name in sorted(paths):
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
