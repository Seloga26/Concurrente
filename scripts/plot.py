#!/usr/bin/env python3
"""Genera las graficas del informe (Fase 5) a partir de results/benchmark.csv.

Lee el RESUMEN agregado (`results/benchmark.csv`, ya con speedup/efficiency calculados por
`scripts/aggregate.py`) y produce PNGs en `results/plots/`. NO recomputa el speedup ni el
baseline: usa las columnas del CSV (unica fuente, ya validada por familia/platform/mode/K).

Por cada modo presente (`benchmark` f32, `reference` f64) genera, sobre el escalado CPU (WSL):
  - speedup_<mode>.png     : speedup vs P (subplots por K) para openmp/mpi/multicore + ideal.
  - efficiency_<mode>.png  : eficiencia vs P (mismos grupos) + linea y=1.
  - amdahl_<mode>.png      : speedup empirico vs curva de Amdahl con f ajustado por familia (K mayor).
  - scaling_K_<mode>.png   : t_core vs K (log-log) a P=1 para todas las impls (O(K) lineal).
Y, sobre Colab (CUDA), independiente del modo en la misma figura:
  - cuda_comparison.png    : t_core (log) de c_serial/cuda/cuda_pycuda a los 3 K, ambos modos.

Uso:  python scripts/plot.py --csv results/benchmark.csv --out results/plots
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")  # backend sin ventana (reproducible en CI/headless)
import matplotlib.pyplot as plt
import pandas as pd

# Familias con escalado P (CPU, WSL) -> etiqueta legible para la leyenda.
_PARALLEL_IMPLS = {
    "c_openmp": "C + OpenMP",
    "c_mpi": "C + MPI",
    "python_multicore": "Python multicore",
}
# Orden e impls de la comparativa CUDA (Colab).
_CUDA_IMPLS = ["c_serial", "cuda", "cuda_pycuda"]
_CUDA_LABEL = {"c_serial": "C serial (CPU)", "cuda": "CUDA (nvcc)", "cuda_pycuda": "CUDA (PyCUDA)"}

_K_FMT = {100000: "K=10^5", 1000000: "K=10^6", 10000000: "K=10^7"}


# --------------------------------------------------------------------------- #
# Helpers puros (sin I/O ni matplotlib): testeables.
# --------------------------------------------------------------------------- #
def amdahl_speedup(f, p):
    """Speedup teorico de Amdahl para fraccion paralela f en P unidades: 1/((1-f)+f/P)."""
    return 1.0 / ((1.0 - f) + f / p)


def amdahl_fraction(p_values, speedups):
    """Estima la fraccion paralela f del modelo de Amdahl por minimos cuadrados.

    Se ajusta S(P)=1/((1-f)+f/P) en su forma lineal `1/S - 1 = f*(1/P - 1)`, que fuerza
    S(1)=1 (un solo parametro f). f = sum((x)(y)) / sum(x^2) con x=1/P-1, y=1/S-1.
    Devuelve f recortado a [0, 1].
    """
    num = 0.0
    den = 0.0
    for p, s in zip(p_values, speedups):
        if p <= 0 or s <= 0:
            continue
        x = 1.0 / p - 1.0
        y = 1.0 / s - 1.0
        num += x * y
        den += x * x
    if den == 0.0:
        return 0.0
    f = num / den
    return max(0.0, min(1.0, f))


# --------------------------------------------------------------------------- #
# Carga y filtrado.
# --------------------------------------------------------------------------- #
def load_summary(csv_path):
    """Carga el CSV de resumen como DataFrame, con columnas numericas tipadas."""
    df = pd.read_csv(csv_path)
    for col in ("K", "P", "t_core_median_s", "t_search_median_s", "speedup", "efficiency"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _ks(df):
    """K presentes, ordenados ascendentemente."""
    return sorted(int(k) for k in df["K"].dropna().unique())


# --------------------------------------------------------------------------- #
# Figuras de escalado CPU (platform=wsl).
# --------------------------------------------------------------------------- #
def plot_speedup(df, mode, out_dir):
    """Speedup vs P, un subplot por K, lineas por impl paralela + recta ideal."""
    cpu = df[(df["platform"] == "wsl") & (df["mode"] == mode)]
    ks = _ks(cpu)
    if not ks:
        return None
    fig, axes = plt.subplots(1, len(ks), figsize=(5 * len(ks), 4.2), squeeze=False)
    for ax, k in zip(axes[0], ks):
        sub = cpu[cpu["K"] == k]
        for impl, label in _PARALLEL_IMPLS.items():
            d = sub[sub["implementation"] == impl].sort_values("P")
            if not d.empty:
                ax.plot(d["P"], d["speedup"], marker="o", label=label)
        p_all = sorted(sub["P"].dropna().unique())
        if p_all:
            ax.plot(p_all, p_all, "k--", alpha=0.5, label="Ideal (lineal)")
        ax.set_title(_K_FMT.get(k, f"K={k}"))
        ax.set_xlabel("P (procesos / hilos)")
        ax.set_ylabel("Speedup")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"Speedup vs paralelismo — modo {mode} (WSL, 12 cores)")
    return _save(fig, out_dir, f"speedup_{mode}.png")


def plot_efficiency(df, mode, out_dir):
    """Eficiencia (speedup/P) vs P, un subplot por K, con linea de referencia y=1."""
    cpu = df[(df["platform"] == "wsl") & (df["mode"] == mode)]
    ks = _ks(cpu)
    if not ks:
        return None
    fig, axes = plt.subplots(1, len(ks), figsize=(5 * len(ks), 4.2), squeeze=False)
    for ax, k in zip(axes[0], ks):
        sub = cpu[cpu["K"] == k]
        for impl, label in _PARALLEL_IMPLS.items():
            d = sub[sub["implementation"] == impl].sort_values("P")
            if not d.empty:
                ax.plot(d["P"], d["efficiency"], marker="o", label=label)
        ax.axhline(1.0, color="k", ls="--", alpha=0.5, label="Ideal (=1)")
        ax.set_ylim(0, 1.15)
        ax.set_title(_K_FMT.get(k, f"K={k}"))
        ax.set_xlabel("P (procesos / hilos)")
        ax.set_ylabel("Eficiencia")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"Eficiencia vs paralelismo — modo {mode} (WSL, 12 cores)")
    return _save(fig, out_dir, f"efficiency_{mode}.png")


def plot_amdahl(df, mode, out_dir):
    """Speedup empirico (K mayor) vs curva de Amdahl con f ajustado por familia."""
    cpu = df[(df["platform"] == "wsl") & (df["mode"] == mode)]
    ks = _ks(cpu)
    if not ks:
        return None
    k = ks[-1]  # K mayor: mejor senal de escalado (menos peso del overhead)
    sub = cpu[cpu["K"] == k]
    fig, ax = plt.subplots(figsize=(7, 5))
    for impl, label in _PARALLEL_IMPLS.items():
        d = sub[sub["implementation"] == impl].sort_values("P")
        if d.empty:
            continue
        p_vals = d["P"].tolist()
        s_vals = d["speedup"].tolist()
        f = amdahl_fraction(p_vals, s_vals)
        line = ax.plot(p_vals, s_vals, marker="o", ls="none",
                       label=f"{label} (empirico)")[0]
        p_dense = [p_vals[0] + i * (p_vals[-1] - p_vals[0]) / 60.0 for i in range(61)]
        ax.plot(p_dense, [amdahl_speedup(f, p) for p in p_dense],
                color=line.get_color(), alpha=0.8,
                label=f"  Amdahl f={f:.3f} (max {1.0/(1.0-f):.1f}x)" if f < 1 else f"  Amdahl f={f:.3f}")
    ax.set_title(f"Ley de Amdahl — modo {mode}, {_K_FMT.get(k, f'K={k}')} (WSL)")
    ax.set_xlabel("P (procesos / hilos)")
    ax.set_ylabel("Speedup")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    return _save(fig, out_dir, f"amdahl_{mode}.png")


def plot_scaling_k(df, mode, out_dir):
    """t_core vs K (log-log) a P=1 para todas las impls de WSL (escalado del kernel O(K))."""
    cpu = df[(df["platform"] == "wsl") & (df["mode"] == mode) & (df["P"] == 1)]
    if cpu.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 5))
    for impl in sorted(cpu["implementation"].unique()):
        d = cpu[cpu["implementation"] == impl].sort_values("K")
        if not d.empty:
            ax.plot(d["K"], d["t_core_median_s"], marker="o", label=impl)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"Escalado con K — modo {mode}, P=1 (WSL)")
    ax.set_xlabel("K (numero de candidatos)")
    ax.set_ylabel("t_core mediana (s)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    return _save(fig, out_dir, f"scaling_K_{mode}.png")


# --------------------------------------------------------------------------- #
# Figura CUDA (platform=colab), ambos modos en una sola imagen.
# --------------------------------------------------------------------------- #
def plot_cuda_comparison(df, out_dir):
    """t_core (log) de c_serial/cuda/cuda_pycuda en Colab, un subplot por modo."""
    gpu = df[df["platform"] == "colab"]
    if gpu.empty:
        return None
    modes = sorted(gpu["mode"].unique())
    ks = _ks(gpu)
    fig, axes = plt.subplots(1, len(modes), figsize=(6 * len(modes), 4.5), squeeze=False)
    x = range(len(ks))
    width = 0.25
    for ax, mode in zip(axes[0], modes):
        sub = gpu[gpu["mode"] == mode]
        for j, impl in enumerate(_CUDA_IMPLS):
            d = sub[sub["implementation"] == impl].set_index("K")
            ys = [d.loc[k, "t_core_median_s"] if k in d.index else 0.0 for k in ks]
            ax.bar([xi + (j - 1) * width for xi in x], ys, width, label=_CUDA_LABEL[impl])
        # Anota el speedup de cuda (nvcc) vs c_serial encima de cada grupo.
        for i, k in enumerate(ks):
            row = sub[(sub["implementation"] == "cuda") & (sub["K"] == k)]
            if not row.empty and pd.notna(row["speedup"].iloc[0]):
                ax.text(i, row["t_core_median_s"].iloc[0], f"{row['speedup'].iloc[0]:.0f}x",
                        ha="center", va="bottom", fontsize=8)
        ax.set_yscale("log")
        ax.set_xticks(list(x))
        ax.set_xticklabels([_K_FMT.get(k, str(k)) for k in ks])
        ax.set_title(f"modo {mode}")
        ax.set_ylabel("t_core mediana (s, log)")
        ax.grid(True, axis="y", which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("CUDA vs CPU serial — Colab (Tesla T4); etiqueta = speedup de CUDA nvcc")
    return _save(fig, out_dir, "cuda_comparison.png")


# --------------------------------------------------------------------------- #
# Orquestacion.
# --------------------------------------------------------------------------- #
def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def generate_all(df, out_dir):
    """Genera todas las figuras y devuelve la lista de rutas escritas."""
    written = []
    for mode in sorted(df["mode"].unique()):
        for fn in (plot_speedup, plot_efficiency, plot_amdahl, plot_scaling_k):
            path = fn(df, mode, out_dir)
            if path:
                written.append(path)
    cuda = plot_cuda_comparison(df, out_dir)
    if cuda:
        written.append(cuda)
    return written


def main(argv=None):
    p = argparse.ArgumentParser(description="Genera las graficas del informe desde benchmark.csv.")
    p.add_argument("--csv", default="results/benchmark.csv", help="Ruta al CSV de resumen.")
    p.add_argument("--out", default="results/plots", help="Directorio de salida (default: results/plots).")
    args = p.parse_args(argv)
    df = load_summary(args.csv)
    written = generate_all(df, args.out)
    print(f"plot: {len(written)} figuras -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
