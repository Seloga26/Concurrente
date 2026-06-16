"""Pruebas del generador `data/generate_data.py` (contrato + invariantes).

- Validación de configuración (punto 1): usa el validador puro de `data.genconfig`.
- Generación: reproducibilidad, formas/dtypes, invariantes, neutral/señal por medias
  teóricas (sin AUC aleatorio), metadatos, hold-out, candidatos, escritura atómica.

Ejecutar desde la raíz del repositorio:
    python -m unittest discover -s tests -v
"""
import glob
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fixtures  # noqa: E402
from fixtures import (  # noqa: E402
    GenConfigError,
    base_generation_config,
    expected_dtypes,
    expected_shapes,
    neutral_config,
    signal_config,
    tiny_config,
    validate_generation_config,
)

from data import generate_data as gd  # noqa: E402


# =========================================================================== #
# Punto 1 — validación de configuración (validador puro)
# =========================================================================== #
class GenerationConfigContractTests(unittest.TestCase):
    def test_base_config_valid(self):
        self.assertTrue(validate_generation_config(base_generation_config()))

    def test_base_has_5_healthy_5_sick(self):
        cfg = base_generation_config()
        self.assertEqual(cfg["n_healthy"], 5)
        self.assertEqual(cfg["n_sick"], 5)

    def test_base_signal_strength_is_one(self):
        self.assertEqual(base_generation_config()["signal_strength"], 1.0)

    def test_reject_N_too_small(self):
        with self.assertRaises(GenConfigError):
            validate_generation_config({**base_generation_config(), "N": 1})

    def test_reject_kappa_zero(self):
        with self.assertRaises(GenConfigError):
            validate_generation_config({**base_generation_config(), "kappa": 0})

    def test_reject_negative_signal_strength(self):
        with self.assertRaises(GenConfigError):
            validate_generation_config({**base_generation_config(), "signal_strength": -0.1})

    def test_reject_pF_out_of_range(self):
        with self.assertRaises(GenConfigError):
            validate_generation_config({**base_generation_config(), "p_F": 1.5})
        with self.assertRaises(GenConfigError):
            validate_generation_config({**base_generation_config(), "p_F": -0.01})

    def test_reject_eps_noise_one(self):
        with self.assertRaises(GenConfigError):
            validate_generation_config({**base_generation_config(), "eps_noise": 1.0})

    def test_reject_non_integer_seed(self):
        with self.assertRaises(GenConfigError):
            validate_generation_config({**base_generation_config(), "data_seed": 1.5})

    def test_reject_non_bool_F_binary(self):
        with self.assertRaises(GenConfigError):
            validate_generation_config({**base_generation_config(), "F_binary": "yes"})

    def test_holdout_fields_validated(self):
        with self.assertRaises(GenConfigError):
            validate_generation_config({**base_generation_config(), "holdout_n_healthy": 0})

    def test_neutral_config_kappa_equals_N(self):
        cfg = neutral_config()
        self.assertEqual(cfg["signal_strength"], 0.0)
        self.assertEqual(cfg["kappa"], cfg["N"])
        self.assertTrue(validate_generation_config(cfg))

    def test_expected_shapes_dtypes_helpers(self):
        sh = expected_shapes(N=50, K=100)
        self.assertEqual(sh["A"], (10, 50))
        self.assertEqual(sh["candidates_W"], (100, 3))
        dt = expected_dtypes()
        self.assertEqual(dt["A"], "float32")
        self.assertEqual(dt["W_true"], "float64")


# =========================================================================== #
# Pure helpers
# =========================================================================== #
class PureHelperTests(unittest.TestCase):
    def test_softmax_stable_uniform(self):
        np.testing.assert_allclose(gd.softmax_stable(np.zeros(4)), np.full(4, 0.25))

    def test_dirichlet_alpha_value(self):
        np.testing.assert_allclose(gd.dirichlet_alpha(np.array([0.5, 0.5]), 10.0), [5.0, 5.0])

    def test_latent_risk_value(self):
        r = gd.latent_risk(np.array([1.0, 0.0]), np.array([0.0, 1.0]),
                           np.array([0.0, 0.0]), np.array([0.5, 0.5, 0.0]))
        np.testing.assert_allclose(r, [0.5, 0.5])

    def test_class_means_degenerate_constant_r(self):
        m_san, m_enf = gd.class_means(np.full(5, 0.3), signal_strength=1.0, eps_noise=0.0)
        np.testing.assert_allclose(m_san, np.full(5, 0.2))
        np.testing.assert_allclose(m_enf, np.full(5, 0.2))

    def test_class_means_extreme_signal_finite_positive(self):
        r = np.random.default_rng(0).random(20)
        m_san, m_enf = gd.class_means(r, signal_strength=1e6, eps_noise=0.0)
        for m in (m_san, m_enf):
            self.assertTrue(np.all(np.isfinite(m)))
            self.assertTrue(np.all(m > 0))
            np.testing.assert_allclose(m.sum(), 1.0, atol=1e-9)
        alpha = gd.dirichlet_alpha(m_enf, 50.0)
        self.assertTrue(np.all(np.isfinite(alpha)) and np.all(alpha > 0))

    def test_class_means_non_finite_raises(self):
        with self.assertRaises(ValueError):
            gd.class_means(np.array([0.1, np.inf, 0.3]), 1.0, 0.0)

    def test_sample_profiles_pF_edges(self):
        rng = np.random.default_rng(0)
        _, _, f0 = gd.sample_profiles(50, 0.0, True, rng)
        self.assertTrue(np.all(f0 == 0))
        _, _, f1 = gd.sample_profiles(50, 1.0, True, rng)
        self.assertTrue(np.all(f1 == 1))

    def test_helper_validations(self):
        with self.assertRaises(ValueError):  # shapes incompatibles
            gd.latent_risk(np.zeros(3), np.zeros(4), np.zeros(3), np.ones(3) / 3)
        with self.assertRaises(ValueError):  # W_true mal
            gd.latent_risk(np.zeros(3), np.zeros(3), np.zeros(3), np.array([0.5, 0.5]))
        with self.assertRaises(ValueError):  # vacío
            gd.latent_risk(np.zeros(0), np.zeros(0), np.zeros(0), np.ones(3) / 3)
        with self.assertRaises(ValueError):  # tamaños
            gd.make_labels(0, 5)
        with self.assertRaises(GenConfigError):  # K inválido
            gd.generate_candidates(0, 1)


# =========================================================================== #
# Reproducibilidad y streams RNG
# =========================================================================== #
class ReproducibilityTests(unittest.TestCase):
    def test_same_seed_identical(self):
        cfg = tiny_config()
        a = gd.generate_dataset(cfg, K=10)
        b = gd.generate_dataset(cfg, K=10)
        for attr in ("A", "T", "S", "F", "y", "candidates_W", "W_true"):
            np.testing.assert_array_equal(getattr(a, attr), getattr(b, attr))

    def test_change_data_seed_changes_profiles_or_A(self):
        a = gd.generate_dataset(tiny_config(), K=10)
        b = gd.generate_dataset({**tiny_config(), "data_seed": 999}, K=10)
        self.assertTrue(any(not np.array_equal(getattr(a, k), getattr(b, k))
                            for k in ("A", "T", "S", "F")))

    def test_change_candidate_seed_only_changes_candidates(self):
        a = gd.generate_dataset(tiny_config(), K=10)
        b = gd.generate_dataset({**tiny_config(), "candidate_seed": 777}, K=10)
        for k in ("A", "T", "S", "F", "y", "W_true"):
            np.testing.assert_array_equal(getattr(a, k), getattr(b, k))
        self.assertFalse(np.array_equal(a.candidates_W, b.candidates_W))

    def test_stream_independence_K_none_vs_K(self):
        a = gd.generate_dataset(tiny_config(), K=None)
        b = gd.generate_dataset(tiny_config(), K=10)
        for k in ("A", "T", "S", "F", "W_true"):
            np.testing.assert_array_equal(getattr(a, k), getattr(b, k))
        self.assertIsNone(a.candidates_W)
        self.assertIsNotNone(b.candidates_W)


# =========================================================================== #
# Formas, dtypes e invariantes
# =========================================================================== #
class ShapesAndInvariantsTests(unittest.TestCase):
    def test_shapes_and_dtypes(self):
        cfg = tiny_config(N=6)
        ds = gd.generate_dataset(cfg, K=12)
        shapes = expected_shapes(cfg["N"], 12)
        dtypes = expected_dtypes()
        for k, shp in shapes.items():
            self.assertEqual(tuple(getattr(ds, k).shape), shp, k)
            self.assertEqual(str(getattr(ds, k).dtype), dtypes[k], k)

    def test_A_rows_nonneg_sum_one(self):
        ds = gd.generate_dataset(tiny_config(), K=10)
        self.assertTrue(np.all(ds.A >= 0))
        np.testing.assert_allclose(ds.A.sum(axis=1), 1.0, atol=1e-5)

    def test_candidates_nonneg_sum_one(self):
        ds = gd.generate_dataset(tiny_config(), K=50)
        self.assertTrue(np.all(ds.candidates_W >= 0))
        np.testing.assert_allclose(ds.candidates_W.sum(axis=1), 1.0, atol=1e-5)

    def test_labels_five_zeros_then_five_ones(self):
        ds = gd.generate_dataset(tiny_config(), K=None)
        np.testing.assert_array_equal(ds.y, np.array([0] * 5 + [1] * 5, dtype=np.int32))

    def test_T_S_in_unit_interval(self):
        ds = gd.generate_dataset(tiny_config(), K=None)
        for k in ("T", "S"):
            arr = getattr(ds, k)
            self.assertTrue(np.all(arr >= 0.0) and np.all(arr <= 1.0))

    def test_F_binary_contract(self):
        ds = gd.generate_dataset(tiny_config(), K=None)
        self.assertTrue(np.all(np.isin(ds.F, (0.0, 1.0))))

    def test_dirichlet_alpha_strictly_positive(self):
        ds = gd.generate_dataset(tiny_config(), K=None)
        for m in (ds.m_san, ds.m_enf):
            self.assertTrue(np.all(gd.dirichlet_alpha(m, tiny_config()["kappa"]) > 0))


# =========================================================================== #
# Modo neutral y modo con señal (medias teóricas, sin AUC aleatorio)
# =========================================================================== #
class NeutralAndSignalTests(unittest.TestCase):
    def test_neutral_class_means_equal(self):
        ds = gd.generate_dataset(neutral_config(), K=None)
        np.testing.assert_allclose(ds.m_san, ds.m_enf, atol=1e-12)

    def test_neutral_kappa_N_alpha_is_ones(self):
        cfg = neutral_config()
        ds = gd.generate_dataset(cfg, K=None)
        alpha = gd.dirichlet_alpha(ds.m_san, cfg["kappa"])
        np.testing.assert_allclose(alpha, np.ones(cfg["N"]), atol=1e-9)

    def test_signal_class_means_differ(self):
        ds = gd.generate_dataset(signal_config(signal_strength=1.0), K=None)
        self.assertFalse(np.allclose(ds.m_san, ds.m_enf))

    def test_theoretical_sick_score_greater_than_healthy(self):
        ds = gd.generate_dataset(signal_config(signal_strength=1.0), K=None)
        e_sick = float(np.dot(ds.m_enf, ds.r))
        e_healthy = float(np.dot(ds.m_san, ds.r))
        self.assertGreater(e_sick, e_healthy)

    def test_enforce_main_counts(self):
        cfg = tiny_config()
        cfg["n_healthy"], cfg["n_sick"] = 3, 4
        with self.assertRaises(GenConfigError):
            gd.generate_dataset(cfg, K=None)
        ds = gd.generate_dataset(cfg, K=None, enforce_main_counts=False)
        self.assertEqual(ds.A.shape[0], 7)

    def test_generate_dataset_invalid_config_raises(self):
        bad = signal_config()
        bad["N"] = 1
        with self.assertRaises(GenConfigError):
            gd.generate_dataset(bad, K=10)


# =========================================================================== #
# Metadatos
# =========================================================================== #
class MetadataTests(unittest.TestCase):
    def _assert_no_long_numeric_list(self, obj):
        if isinstance(obj, list):
            if obj and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in obj):
                self.assertLessEqual(len(obj), 3, f"lista numérica larga: {obj[:5]}")
            for x in obj:
                self._assert_no_long_numeric_list(x)
        elif isinstance(obj, dict):
            for v in obj.values():
                self._assert_no_long_numeric_list(v)

    def test_metadata_excludes_large_vectors(self):
        ds = gd.generate_dataset(tiny_config(), K=10)
        meta = gd.build_metadata(ds, file_hashes={})
        for forbidden in ("r", "m_san", "m_enf"):
            self.assertNotIn(forbidden, meta)

    def test_metadata_includes_required_keys(self):
        ds = gd.generate_dataset(tiny_config(), K=10)
        meta = gd.build_metadata(ds, file_hashes={"matrix_A.npy": "dead"})
        for key in ("schema_version", "algorithm", "params", "seeds",
                    "W_true", "shapes", "dtypes", "files", "hashes", "summary"):
            self.assertIn(key, meta)

    def test_metadata_no_long_numeric_lists(self):
        ds = gd.generate_dataset(tiny_config(N=30), K=10)
        meta = gd.build_metadata(ds, file_hashes={"matrix_A.npy": "x"})
        self._assert_no_long_numeric_list(meta)


# =========================================================================== #
# Hold-out
# =========================================================================== #
class HoldoutTests(unittest.TestCase):
    def test_holdout_reuses_profiles_changes_A_no_candidates(self):
        cfg = tiny_config()
        ds = gd.generate_dataset(cfg, K=10)
        hold = gd.generate_holdout(ds, holdout_seed=cfg.get("holdout_seed", 2024))
        for k in ("T", "S", "F", "W_true"):
            np.testing.assert_array_equal(getattr(ds, k), getattr(hold, k))
        self.assertFalse(np.array_equal(ds.A, hold.A))
        np.testing.assert_array_equal(hold.y, np.array([0] * 5 + [1] * 5, dtype=np.int32))
        self.assertIsNone(hold.candidates_W)


# =========================================================================== #
# Candidatos
# =========================================================================== #
class CandidatesTests(unittest.TestCase):
    def test_candidates_uniform_simplex_reproducible(self):
        w1 = gd.generate_candidates(2000, 123)
        w2 = gd.generate_candidates(2000, 123)
        np.testing.assert_array_equal(w1, w2)
        self.assertEqual(w1.shape, (2000, 3))
        self.assertTrue(np.all(w1 >= 0))
        np.testing.assert_allclose(w1.sum(axis=1), 1.0, atol=1e-5)

    def test_candidates_generation_order_no_sort(self):
        w = gd.generate_candidates(500, 123)
        expected = np.random.default_rng(123).dirichlet(np.ones(3), size=500).astype(np.float32)
        np.testing.assert_array_equal(w, expected)


# =========================================================================== #
# Escritura segura
# =========================================================================== #
class SafeWriteTests(unittest.TestCase):
    def test_write_creates_missing_dir(self):
        ds = gd.generate_dataset(tiny_config(), K=10)
        with tempfile.TemporaryDirectory() as out:
            target = os.path.join(out, "sub", "nested")
            gd.write_outputs(ds, output_dir=target, overwrite=False)
            self.assertTrue(os.path.exists(os.path.join(target, "matrix_A.npy")))

    def test_no_spurious_extension(self):
        ds = gd.generate_dataset(tiny_config(), K=10)
        with tempfile.TemporaryDirectory() as out:
            gd.write_outputs(ds, output_dir=out)
            self.assertEqual(glob.glob(os.path.join(out, "*.tmp-*")), [])
            self.assertEqual(glob.glob(os.path.join(out, "*.npy.npy")), [])
            self.assertTrue(os.path.exists(os.path.join(out, "matrix_A.npy")))

    def test_no_silent_overwrite(self):
        ds = gd.generate_dataset(tiny_config(), K=10)
        with tempfile.TemporaryDirectory() as out:
            gd.write_outputs(ds, output_dir=out, overwrite=False)
            with self.assertRaises(FileExistsError):
                gd.write_outputs(ds, output_dir=out, overwrite=False)

    def test_overwrite_flag_allows_replacement(self):
        ds = gd.generate_dataset(tiny_config(), K=10)
        with tempfile.TemporaryDirectory() as out:
            gd.write_outputs(ds, output_dir=out, overwrite=False)
            gd.write_outputs(ds, output_dir=out, overwrite=True)

    def test_failure_before_replace_leaves_no_changes(self):
        ds = gd.generate_dataset(tiny_config(), K=10)
        with tempfile.TemporaryDirectory() as out:
            with mock.patch("data.generate_data.np.save", side_effect=RuntimeError("boom")):
                with self.assertRaises(RuntimeError):
                    gd.write_outputs(ds, output_dir=out)
            self.assertEqual(glob.glob(os.path.join(out, "*.npy")), [])
            self.assertEqual(glob.glob(os.path.join(out, "*.tmp-*")), [])
            self.assertFalse(os.path.exists(os.path.join(out, "generation_metadata.json")))

    def test_hashes_reproducible(self):
        cfg = tiny_config()
        with tempfile.TemporaryDirectory() as o1, tempfile.TemporaryDirectory() as o2:
            gd.write_outputs(gd.generate_dataset(cfg, K=10), output_dir=o1, write_truth=True)
            gd.write_outputs(gd.generate_dataset(cfg, K=10), output_dir=o2, write_truth=True)
            with open(os.path.join(o1, "generation_metadata.json")) as f:
                m1 = json.load(f)
            with open(os.path.join(o2, "generation_metadata.json")) as f:
                m2 = json.load(f)
            self.assertEqual(m1["hashes"], m2["hashes"])
            self.assertEqual(m1["truth_content_hashes"], m2["truth_content_hashes"])


# =========================================================================== #
# CLI y coherencia
# =========================================================================== #
class CliTests(unittest.TestCase):
    def test_cli_help(self):
        with self.assertRaises(SystemExit) as cm:
            gd.main(["--help"])
        self.assertEqual(cm.exception.code, 0)

    def test_main_generates_with_n_override(self):
        with tempfile.TemporaryDirectory() as out:
            rc = gd.main(["--n", "8", "--k", "20", "--output-dir", out, "--overwrite"])
            self.assertEqual(rc, 0)
            self.assertTrue(os.path.exists(os.path.join(out, "matrix_A.npy")))
            self.assertTrue(os.path.exists(os.path.join(out, "generation_metadata.json")))

    def test_n_override_coherence(self):
        from config.launcher import ConfigError, validate_base_consistency
        self.assertEqual(validate_base_consistency({"N": 8}, {"N": 8}), 8)
        with self.assertRaises(ConfigError):
            validate_base_consistency({"N": 50}, {"N": 8})


if __name__ == "__main__":
    unittest.main()
