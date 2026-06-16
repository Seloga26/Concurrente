"""Pruebas de loader, validación CLI e integración de python/sequential.py."""
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fixtures  # noqa: E402
from fixtures import manual_scoring_fixture  # noqa: E402

from common.loader import load_and_validate_inputs  # noqa: E402
from python import sequential as seq  # noqa: E402

_NAMES = ["matrix_A.npy", "profile_T.npy", "profile_S.npy", "profile_F.npy", "labels.npy", "candidates_W.npy"]


def _dump(d, **overrides):
    fx = manual_scoring_fixture()
    arrs = {
        "matrix_A.npy": fx["A"], "profile_T.npy": fx["T"], "profile_S.npy": fx["S"],
        "profile_F.npy": fx["F"], "labels.npy": fx["y"], "candidates_W.npy": fx["candidates_W"],
    }
    arrs.update(overrides)
    paths = {}
    for name in _NAMES:
        p = os.path.join(d, name)
        np.save(p, arrs[name])
        paths[name] = p
    return paths


def _load(paths, N=2, K=3):
    return load_and_validate_inputs(
        paths["matrix_A.npy"], paths["profile_T.npy"], paths["profile_S.npy"],
        paths["profile_F.npy"], paths["labels.npy"], paths["candidates_W.npy"], N, K)


def _argv(paths, **over):
    d = dict(N="2", K="3", mode="reference", accum="float64", algorithm="literal",
             theta_policy="class_mean_midpoint", tie_atol="1e-9", tie_rtol="1e-9", cons="0.8")
    d.update(over)
    return [
        "--N", d["N"], "--K", d["K"], "--mode", d["mode"], "--accum", d["accum"],
        "--algorithm", d["algorithm"], "--tie-atol", d["tie_atol"], "--tie-rtol", d["tie_rtol"],
        "--matrix-a", paths["matrix_A.npy"], "--profile-t", paths["profile_T.npy"],
        "--profile-s", paths["profile_S.npy"], "--profile-f", paths["profile_F.npy"],
        "--labels", paths["labels.npy"], "--candidates", paths["candidates_W.npy"],
        "--theta-policy", d["theta_policy"], "--consistency-threshold", d["cons"],
    ]


class LoaderTests(unittest.TestCase):
    def test_valid_load_without_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            paths = _dump(d)
            self.assertNotIn("generation_metadata.json", os.listdir(d))  # el scoring no lo necesita
            inp = _load(paths)
            self.assertEqual(inp.A.shape, (10, 2))
            np.testing.assert_array_equal(inp.pos_idx, [5, 6, 7, 8, 9])

    def test_missing_file(self):
        with tempfile.TemporaryDirectory() as d:
            paths = _dump(d)
            os.remove(paths["labels.npy"])
            with self.assertRaises(ValueError):
                _load(paths)

    def test_wrong_dtype(self):
        with tempfile.TemporaryDirectory() as d:
            paths = _dump(d, **{"matrix_A.npy": manual_scoring_fixture()["A"].astype(np.float64)})
            with self.assertRaises(ValueError):
                _load(paths)

    def test_wrong_shape(self):
        with tempfile.TemporaryDirectory() as d:
            paths = _dump(d)
            with self.assertRaises(ValueError):
                _load(paths, N=3)        # arrays son N=2

    def test_non_finite(self):
        bad = manual_scoring_fixture()["T"].copy()
        bad[0] = np.nan
        with tempfile.TemporaryDirectory() as d:
            paths = _dump(d, **{"profile_T.npy": bad})
            with self.assertRaises(ValueError):
                _load(paths)

    def test_A_not_normalized(self):
        bad = manual_scoring_fixture()["A"].copy()
        bad[0, 0] = 0.5            # fila ya no suma 1
        with tempfile.TemporaryDirectory() as d:
            paths = _dump(d, **{"matrix_A.npy": bad})
            with self.assertRaises(ValueError):
                _load(paths)

    def test_candidate_off_simplex(self):
        bad = manual_scoring_fixture()["candidates_W"].copy()
        bad[0, 0] = 0.5            # fila suma 1.5
        with tempfile.TemporaryDirectory() as d:
            paths = _dump(d, **{"candidates_W.npy": bad})
            with self.assertRaises(ValueError):
                _load(paths)

    def test_profile_out_of_range(self):
        bad = manual_scoring_fixture()["T"].copy()
        bad[0] = 1.5              # fuera de [0,1]
        with tempfile.TemporaryDirectory() as d:
            paths = _dump(d, **{"profile_T.npy": bad})
            with self.assertRaises(ValueError):
                _load(paths)

    def test_NK_incoherent(self):
        with tempfile.TemporaryDirectory() as d:
            paths = _dump(d)
            with self.assertRaises(ValueError):
                _load(paths, K=5)        # candidatos son K=3


class CliValidationTests(unittest.TestCase):
    def _run_main(self, **over):
        with tempfile.TemporaryDirectory() as d:
            paths = _dump(d)
            return seq.main(_argv(paths, **over))

    def test_reference_requires_float64(self):
        self.assertEqual(self._run_main(mode="reference", accum="float32"), 1)

    def test_benchmark_requires_float32(self):
        self.assertEqual(self._run_main(mode="benchmark", accum="float64"), 1)

    def test_bad_algorithm(self):
        self.assertEqual(self._run_main(algorithm="foo"), 1)

    def test_bad_theta_policy(self):
        self.assertEqual(self._run_main(theta_policy="otra"), 1)

    def test_bad_mode(self):
        self.assertEqual(self._run_main(mode="turbo", accum="float64"), 1)

    def test_help_exits_zero(self):
        with self.assertRaises(SystemExit) as cm:
            seq.main(["--help"])
        self.assertEqual(cm.exception.code, 0)


class IntegrationInProcessTests(unittest.TestCase):
    def _json_from_main(self, **over):
        with tempfile.TemporaryDirectory() as d:
            paths = _dump(d)
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = seq.main(_argv(paths, **over))
            self.assertEqual(rc, 0)
            lines = [ln for ln in buf.getvalue().splitlines() if ln.strip()]
            self.assertEqual(len(lines), 1)              # exactamente una línea JSON
            return json.loads(lines[0])

    def test_reference_result(self):
        r = self._json_from_main(mode="reference", accum="float64")
        self.assertEqual(r["best_k"], 0)
        self.assertEqual(r["auc_units"], 50)
        self.assertEqual(r["auc_denominator"], 50)
        self.assertEqual(r["accum_dtype"], "float64")
        self.assertEqual(r["algorithm"], "literal")
        self.assertEqual(r["kernel_variant"], "materialized_numpy")
        for key in ("tie_atol", "tie_rtol", "theta_policy", "consistency_threshold",
                    "consistency_pass", "blas_threads", "t_core_seconds", "t_search_seconds"):
            self.assertIn(key, r)

    def test_benchmark_same_winner(self):
        rr = self._json_from_main(mode="reference", accum="float64")
        rb = self._json_from_main(mode="benchmark", accum="float32")
        self.assertEqual((rr["best_k"], rr["auc_units"]), (rb["best_k"], rb["auc_units"]))
        self.assertEqual(rb["accum_dtype"], "float32")


class SubprocessIntegrationTests(unittest.TestCase):
    def test_subprocess_run_single_thread(self):
        with tempfile.TemporaryDirectory() as d:
            paths = _dump(d)
            env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
            proc = subprocess.run(
                [sys.executable, "-m", "python.sequential", *_argv(paths)],
                cwd=str(_ROOT), env=env, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
            self.assertEqual(len(lines), 1)
            r = json.loads(lines[0])
            self.assertEqual(r["best_k"], 0)
            self.assertEqual(r["auc_units"], 50)
            self.assertEqual(set(r["blas_threads"].values()), {"1"})


if __name__ == "__main__":
    unittest.main()
