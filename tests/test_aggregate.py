"""Pruebas deterministas del agregador del benchmark (scripts/aggregate.py)."""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import aggregate  # noqa: E402


def _rec(impl, mode, K, tcore, platform="wsl", extra=None, device=""):
    d = {
        "platform": platform, "implementation": impl, "mode": mode,
        "n_candidates": K, "n_items": 50,
        "t_core_seconds": tcore, "t_search_seconds": tcore * 1.1,
        "best_k": 0, "auc_units": 50, "auc": 1.0,
        "consistency": 1.0, "consistency_pass": True, "device": device,
    }
    if extra:
        d.update(extra)
    d["P"] = aggregate.parallelism(d)
    return d


def _runs():
    runs = []
    K = 1000
    # baseline familia Python: python_sequential P=1, t_core=2.0
    runs += [_rec("python_sequential", "reference", K, 2.0) for _ in range(3)]
    # python_multicore 4 workers, t_core=0.5 -> speedup 4.0, eff 1.0
    runs += [_rec("python_multicore", "reference", K, 0.5, extra={"n_workers": 4}) for _ in range(3)]
    # baseline familia C: c_serial P=1, t_core=1.0
    runs += [_rec("c_serial", "reference", K, 1.0) for _ in range(3)]
    # c_openmp 2 hilos, t_core=0.4 -> speedup 2.5, eff 1.25
    runs += [_rec("c_openmp", "reference", K, 0.4, extra={"n_threads": 2}) for _ in range(3)]
    # cuda P=1, t_core=0.01 -> speedup vs c_serial 100, eff vacia
    runs += [_rec("cuda", "reference", K, 0.01, device="Tesla T4") for _ in range(3)]
    return runs


class TestAggregate(unittest.TestCase):
    def setUp(self):
        rows = aggregate.attach_speedup(aggregate.summarize(_runs()))
        self.by_impl = {r["implementation"]: r for r in rows}

    def test_parallelism_detection(self):
        self.assertEqual(aggregate.parallelism({"n_workers": 4}), 4)
        self.assertEqual(aggregate.parallelism({"n_threads": 8}), 8)
        self.assertEqual(aggregate.parallelism({"n_procs": 2}), 2)
        self.assertEqual(aggregate.parallelism({}), 1)

    def test_one_row_per_group_with_reps(self):
        self.assertEqual(len(self.by_impl), 5)
        for r in self.by_impl.values():
            self.assertEqual(r["reps"], 3)

    def test_median_t_core(self):
        self.assertAlmostEqual(self.by_impl["python_multicore"]["t_core_median_s"], 0.5)
        self.assertAlmostEqual(self.by_impl["c_openmp"]["t_core_median_s"], 0.4)

    def test_speedup_and_efficiency_multicore(self):
        r = self.by_impl["python_multicore"]
        self.assertEqual(r["P"], 4)
        self.assertAlmostEqual(r["speedup"], 4.0)        # 2.0 / 0.5
        self.assertAlmostEqual(r["efficiency"], 1.0)     # 4.0 / 4

    def test_speedup_and_efficiency_openmp(self):
        r = self.by_impl["c_openmp"]
        self.assertEqual(r["P"], 2)
        self.assertAlmostEqual(r["speedup"], 2.5)        # 1.0 / 0.4
        self.assertAlmostEqual(r["efficiency"], 1.25)    # 2.5 / 2

    def test_serial_baselines(self):
        for impl in ("python_sequential", "c_serial"):
            self.assertAlmostEqual(self.by_impl[impl]["speedup"], 1.0)
            self.assertAlmostEqual(self.by_impl[impl]["efficiency"], 1.0)

    def test_cuda_speedup_vs_c_serial_no_efficiency(self):
        r = self.by_impl["cuda"]
        self.assertAlmostEqual(r["speedup"], 100.0)      # c_serial 1.0 / 0.01
        self.assertEqual(r["efficiency"], "")            # CUDA sin efficiency
        self.assertEqual(r["device"], "Tesla T4")

    def test_speedup_not_crossing_platforms(self):
        # Mismo impl en otra plataforma sin baseline -> speedup vacio (no mezcla HW).
        runs = [_rec("c_openmp", "reference", 1000, 0.4, platform="colab", extra={"n_threads": 2})
                for _ in range(3)]
        rows = aggregate.attach_speedup(aggregate.summarize(runs))
        self.assertEqual(rows[0]["speedup"], "")  # no hay c_serial en 'colab'


if __name__ == "__main__":
    unittest.main()
