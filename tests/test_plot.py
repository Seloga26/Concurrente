"""Pruebas del generador de graficas (scripts/plot.py): helpers puros + smoke de main()."""
import csv
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import plot  # noqa: E402


class TestAmdahlHelpers(unittest.TestCase):
    """amdahl_speedup / amdahl_fraction (modelo de un parametro que fuerza S(1)=1)."""

    def test_speedup_basic_properties(self):
        # S(1) = 1 para cualquier f.
        for f in (0.0, 0.3, 0.5, 0.9, 1.0):
            self.assertAlmostEqual(plot.amdahl_speedup(f, 1), 1.0, places=12)
        # f=1 (perfectamente paralelo) -> S(P)=P.
        self.assertAlmostEqual(plot.amdahl_speedup(1.0, 8), 8.0, places=12)
        # f=0 (todo serial) -> S(P)=1.
        self.assertAlmostEqual(plot.amdahl_speedup(0.0, 8), 1.0, places=12)

    def test_speedup_monotone_and_asymptote(self):
        f = 0.75
        prev = 0.0
        for p in (1, 2, 4, 8, 16, 1000):
            s = plot.amdahl_speedup(f, p)
            self.assertGreaterEqual(s, prev)  # creciente en P
            prev = s
        # Limite P->inf tiende a 1/(1-f).
        self.assertAlmostEqual(plot.amdahl_speedup(f, 10_000_000), 1.0 / (1.0 - f), places=4)

    def test_fraction_recovers_known_f(self):
        # Genera speedups sinteticos desde un f conocido y comprueba que se recupera.
        p_values = [1, 2, 4, 6, 8, 12]
        for f_true in (0.2, 0.5, 0.642, 0.9):
            speedups = [plot.amdahl_speedup(f_true, p) for p in p_values]
            f_hat = plot.amdahl_fraction(p_values, speedups)
            self.assertAlmostEqual(f_hat, f_true, places=6)

    def test_fraction_clamped_and_degenerate(self):
        # Solo P=1: no hay informacion de escalado -> 0.0 (den=0).
        self.assertEqual(plot.amdahl_fraction([1], [1.0]), 0.0)
        # Datos super-lineales: f se recorta a [0,1].
        f_hat = plot.amdahl_fraction([1, 2, 4], [1.0, 3.0, 9.0])
        self.assertGreaterEqual(f_hat, 0.0)
        self.assertLessEqual(f_hat, 1.0)


_COLS = ["platform", "implementation", "mode", "N", "K", "P", "reps",
         "t_core_median_s", "t_search_median_s", "speedup", "efficiency",
         "best_k", "auc_units", "auc", "consistency", "consistency_pass", "device"]


def _row(platform, impl, mode, K, P, tcore, speedup, eff, device=""):
    return {
        "platform": platform, "implementation": impl, "mode": mode, "N": 50,
        "K": K, "P": P, "reps": 3, "t_core_median_s": tcore,
        "t_search_median_s": tcore * 1.1, "speedup": speedup, "efficiency": eff,
        "best_k": 0, "auc_units": 50, "auc": 1.0, "consistency": 1.0,
        "consistency_pass": True, "device": device,
    }


def _write_min_csv(path):
    """CSV minimo con WSL (escalado P) + Colab (CUDA) en un solo modo (benchmark)."""
    rows = []
    # WSL serial + dos impls paralelas a P=1,2,4 para K=1e5 y 1e6 (dos K -> scaling_K).
    for K in (100000, 1000000):
        rows.append(_row("wsl", "c_serial", "benchmark", K, 1, 1.0, 1.0, 1.0))
        rows.append(_row("wsl", "python_sequential", "benchmark", K, 1, 5.0, 1.0, 1.0))
        for P, s in ((1, 1.0), (2, 1.8), (4, 3.2)):
            rows.append(_row("wsl", "c_openmp", "benchmark", K, P, 1.0 / s, s, s / P))
            rows.append(_row("wsl", "c_mpi", "benchmark", K, P, 1.0 / s, s, s / P))
            rows.append(_row("wsl", "python_multicore", "benchmark", K, P, 5.0 / s, s, s / P))
    # Colab CUDA.
    for K in (100000, 1000000):
        rows.append(_row("colab", "c_serial", "benchmark", K, 1, 1.0, 1.0, 1.0))
        rows.append(_row("colab", "cuda", "benchmark", K, 1, 0.01, 100.0, "", "Tesla T4"))
        rows.append(_row("colab", "cuda_pycuda", "benchmark", K, 1, 0.012, 83.0, "", "Tesla T4"))
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_COLS)
        w.writeheader()
        w.writerows(rows)


class TestMainSmoke(unittest.TestCase):
    """Corre main() sobre un CSV minimo y verifica que se generan los PNG esperados."""

    def test_generates_expected_pngs(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "benchmark.csv")
            out_dir = os.path.join(tmp, "plots")
            _write_min_csv(csv_path)
            rc = plot.main(["--csv", csv_path, "--out", out_dir])
            self.assertEqual(rc, 0)
            for name in ("speedup_benchmark.png", "efficiency_benchmark.png",
                         "amdahl_benchmark.png", "scaling_K_benchmark.png",
                         "cuda_comparison.png"):
                self.assertTrue(os.path.isfile(os.path.join(out_dir, name)), name)


if __name__ == "__main__":
    unittest.main()
