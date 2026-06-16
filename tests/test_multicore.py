"""Pruebas de python/multicore.py: chunking, equivalencia con sequential y determinismo.

Las pruebas que crean el Pool (con contexto `spawn`) se ejecutan por **subprocess** para ser
robustas en Windows; la lógica pura y la validación de CLI se prueban in-process (no crean Pool).
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fixtures  # noqa: E402
from fixtures import manual_scoring_fixture, tie_fixture  # noqa: E402

from python import multicore as mc  # noqa: E402

_NAMES = ["matrix_A.npy", "profile_T.npy", "profile_S.npy", "profile_F.npy", "labels.npy", "candidates_W.npy"]


def _dump(d, fx):
    arrs = {
        "matrix_A.npy": fx["A"], "profile_T.npy": fx["T"], "profile_S.npy": fx["S"],
        "profile_F.npy": fx["F"], "labels.npy": fx["y"], "candidates_W.npy": fx["candidates_W"],
    }
    paths = {}
    for name in _NAMES:
        p = os.path.join(d, name)
        np.save(p, arrs[name])
        paths[name] = p
    return paths


def _argv(paths, N, K, **over):
    d = dict(mode="reference", accum="float64", algorithm="literal",
             theta_policy="class_mean_midpoint", tie_atol="1e-9", tie_rtol="1e-9", cons="0.8")
    d.update(over)
    argv = [
        "--N", str(N), "--K", str(K), "--mode", d["mode"], "--accum", d["accum"],
        "--algorithm", d["algorithm"], "--tie-atol", d["tie_atol"], "--tie-rtol", d["tie_rtol"],
        "--matrix-a", paths["matrix_A.npy"], "--profile-t", paths["profile_T.npy"],
        "--profile-s", paths["profile_S.npy"], "--profile-f", paths["profile_F.npy"],
        "--labels", paths["labels.npy"], "--candidates", paths["candidates_W.npy"],
        "--theta-policy", d["theta_policy"], "--consistency-threshold", d["cons"],
    ]
    if "workers" in d:
        argv += ["--workers", str(d["workers"])]
    return argv


def _run_subprocess(module, paths, N, K, **over):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    proc = subprocess.run(
        [sys.executable, "-m", module, *_argv(paths, N, K, **over)],
        cwd=str(_ROOT), env=env, capture_output=True, text=True)
    return proc


def _json_subprocess(module, paths, N, K, **over):
    proc = _run_subprocess(module, paths, N, K, **over)
    assert proc.returncode == 0, proc.stderr
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    assert len(lines) == 1, f"se esperaba 1 línea JSON, hay {len(lines)}: {proc.stdout!r}"
    return json.loads(lines[0])


class ChunkingTests(unittest.TestCase):
    def test_even_split(self):
        self.assertEqual(mc._make_chunks(6, 3), [(0, 2), (2, 4), (4, 6)])

    def test_uneven_split_front_loaded(self):
        # base=2, extra=1 -> tamaños [3,2,2]; los chunks iniciales reciben el resto.
        self.assertEqual(mc._make_chunks(7, 3), [(0, 3), (3, 5), (5, 7)])

    def test_more_workers_than_K_drops_empty(self):
        self.assertEqual(mc._make_chunks(2, 5), [(0, 1), (1, 2)])

    def test_chunks_cover_range_disjoint(self):
        chunks = mc._make_chunks(100, 7)
        covered = []
        for a, b in chunks:
            covered.extend(range(a, b))
        self.assertEqual(covered, list(range(100)))

    def test_default_workers_leaves_one_core(self):
        self.assertEqual(mc._default_workers(), max(1, (os.cpu_count() or 1) - 1))


class CliValidationTests(unittest.TestCase):
    """La validación ocurre antes de crear el Pool: seguro in-process."""

    def _run_main(self, fx, N, K, **over):
        with tempfile.TemporaryDirectory() as d:
            paths = _dump(d, fx)
            return mc.main(_argv(paths, N, K, **over))

    def test_reference_requires_float64(self):
        self.assertEqual(self._run_main(manual_scoring_fixture(), 2, 3, mode="reference", accum="float32"), 1)

    def test_benchmark_requires_float32(self):
        self.assertEqual(self._run_main(manual_scoring_fixture(), 2, 3, mode="benchmark", accum="float64"), 1)

    def test_bad_algorithm(self):
        self.assertEqual(self._run_main(manual_scoring_fixture(), 2, 3, algorithm="foo"), 1)

    def test_bad_theta_policy(self):
        self.assertEqual(self._run_main(manual_scoring_fixture(), 2, 3, theta_policy="otra"), 1)

    def test_bad_workers(self):
        self.assertEqual(self._run_main(manual_scoring_fixture(), 2, 3, workers="0"), 1)

    def test_help_exits_zero(self):
        with self.assertRaises(SystemExit) as cm:
            mc.main(["--help"])
        self.assertEqual(cm.exception.code, 0)


class EquivalenceTests(unittest.TestCase):
    def _both(self, fx, N, K, **over):
        with tempfile.TemporaryDirectory() as d:
            paths = _dump(d, fx)
            seq = _json_subprocess("python.sequential", paths, N, K, **over)
            mlt = _json_subprocess("python.multicore", paths, N, K, **over)
            return seq, mlt

    def test_matches_sequential_reference(self):
        seq, mlt = self._both(manual_scoring_fixture(), 2, 3, mode="reference", accum="float64")
        for key in ("best_k", "auc_units", "auc_denominator", "scores", "theta", "consistency"):
            self.assertEqual(mlt[key], seq[key], key)
        self.assertEqual(mlt["implementation"], "python_multicore")
        self.assertIn("n_workers", mlt)

    def test_matches_sequential_benchmark(self):
        seq, mlt = self._both(manual_scoring_fixture(), 2, 3, mode="benchmark", accum="float32")
        self.assertEqual((mlt["best_k"], mlt["auc_units"]), (seq["best_k"], seq["auc_units"]))
        self.assertEqual(mlt["accum_dtype"], "float32")

    def test_cross_chunk_tie_smallest_global_index(self):
        # auc_units empata en k=2 y k=5 (chunks distintos con >=2 workers): gana el menor k=2.
        with tempfile.TemporaryDirectory() as d:
            paths = _dump(d, tie_fixture())
            for w in (1, 2, 3, 6):
                r = _json_subprocess("python.multicore", paths, 2, 6, workers=w)
                self.assertEqual((r["best_k"], r["auc_units"]), (2, 50), f"workers={w}")

    def test_winner_invariant_to_worker_count(self):
        with tempfile.TemporaryDirectory() as d:
            paths = _dump(d, manual_scoring_fixture())
            winners = {
                w: tuple(_json_subprocess("python.multicore", paths, 2, 3, workers=w)[k]
                         for k in ("best_k", "auc_units"))
                for w in (1, 2, 3)
            }
            self.assertEqual(len(set(winners.values())), 1, winners)


class SubprocessContractTests(unittest.TestCase):
    def test_single_json_line_and_blas_single_thread(self):
        with tempfile.TemporaryDirectory() as d:
            paths = _dump(d, manual_scoring_fixture())
            r = _json_subprocess("python.multicore", paths, 2, 3)
            self.assertEqual(r["implementation"], "python_multicore")
            self.assertEqual(r["kernel_variant"], "materialized_numpy")
            self.assertEqual(set(r["blas_threads"].values()), {"1"})
            self.assertGreaterEqual(r["n_workers"], 1)
            for key in ("t_core_seconds", "t_search_seconds", "n_workers"):
                self.assertIn(key, r)


if __name__ == "__main__":
    unittest.main()
