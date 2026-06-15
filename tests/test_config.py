"""Pruebas de configuración y launcher (unittest, sin dependencias externas).

Ejecutar desde la raíz del repositorio:
    python -m unittest discover -s tests -v
"""
import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.launcher import (  # noqa: E402
    INT64_MAX,
    KEY_FACTOR,
    ConfigError,
    build_command,
    load_config,
    validate_base_consistency,
    validate_config,
)

CONFIG_PATH = ROOT / "experiment_config.json"
GEN_CONFIG_PATH = ROOT / "generation_config.json"


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(CONFIG_PATH)
        self.gen = load_config(GEN_CONFIG_PATH)

    # --- carga / validación base ---
    def test_load_base(self):
        self.assertEqual(self.cfg["N"], 50)
        self.assertEqual(self.cfg["K"], 100000)
        self.assertEqual(self.cfg["algorithm"], "literal_fused")
        self.assertEqual(self.cfg["default_mode"], "reference")
        self.assertEqual(self.cfg["allowed_modes"], ["reference", "benchmark"])
        self.assertEqual(validate_config(self.cfg), "reference")

    def test_reject_N_zero(self):
        bad = copy.deepcopy(self.cfg)
        bad["N"] = 0
        with self.assertRaises(ConfigError):
            validate_config(bad)

    def test_reject_K_zero(self):
        bad = copy.deepcopy(self.cfg)
        bad["K"] = 0
        with self.assertRaises(ConfigError):
            validate_config(bad)

    def test_reject_K_overflow(self):
        bad = copy.deepcopy(self.cfg)
        bad["K"] = INT64_MAX // KEY_FACTOR + 1
        with self.assertRaises(ConfigError):
            validate_config(bad)

    def test_accept_K_overflow_boundary(self):
        ok = copy.deepcopy(self.cfg)
        ok["K"] = INT64_MAX // KEY_FACTOR
        self.assertEqual(validate_config(ok), "reference")

    def test_reject_unknown_mode(self):
        with self.assertRaises(ConfigError):
            validate_config(self.cfg, mode="nope")

    def test_reject_negative_tolerance(self):
        bad = copy.deepcopy(self.cfg)
        bad["tie_tolerance"]["reference"]["atol"] = -1.0
        with self.assertRaises(ConfigError):
            validate_config(bad)

    def test_reject_bad_algorithm(self):
        bad = copy.deepcopy(self.cfg)
        bad["algorithm"] = "precomputed"
        with self.assertRaises(ConfigError):
            validate_config(bad)

    # --- coherencia entre configuraciones base ---
    def test_base_consistency_same_N(self):
        self.assertEqual(validate_base_consistency(self.cfg, self.gen), 50)

    def test_base_consistency_diff_N(self):
        bad_gen = copy.deepcopy(self.gen)
        bad_gen["N"] = 64
        with self.assertRaisesRegex(ConfigError, r"N incoherente"):
            validate_base_consistency(self.cfg, bad_gen)

    # --- construcción de comandos (dry-run) ---
    def test_dry_run_command_deterministic(self):
        c1 = build_command(self.cfg, "python_sequential", "reference")
        c2 = build_command(self.cfg, "python_sequential", "reference")
        self.assertEqual(c1, c2)
        self.assertEqual(c1[:2], ["python", "python/sequential.py"])
        self.assertEqual(c1[c1.index("--N") + 1], "50")
        self.assertEqual(c1[c1.index("--K") + 1], "100000")
        self.assertEqual(c1[c1.index("--mode") + 1], "reference")
        self.assertEqual(c1[c1.index("--accum") + 1], "float64")
        self.assertEqual(c1[c1.index("--algorithm") + 1], "literal_fused")
        self.assertEqual(c1[c1.index("--candidates") + 1], "data/candidates_W.npy")

    def test_build_command_returns_list_of_str(self):
        cmd = build_command(self.cfg, "python_sequential", "reference")
        self.assertIsInstance(cmd, list)
        self.assertTrue(all(isinstance(tok, str) for tok in cmd))

    def test_floats_not_rounded(self):
        # Las tolerancias se transmiten con repr() (precisión completa), sin redondeo.
        cmd = build_command(self.cfg, "python_sequential", "reference")
        atol = self.cfg["tie_tolerance"]["reference"]["atol"]
        rtol = self.cfg["tie_tolerance"]["reference"]["rtol"]
        self.assertEqual(cmd[cmd.index("--tie-atol") + 1], repr(atol))
        self.assertEqual(cmd[cmd.index("--tie-rtol") + 1], repr(rtol))

    def test_benchmark_mode_uses_float32(self):
        cmd = build_command(self.cfg, "python_sequential", "benchmark")
        self.assertEqual(cmd[cmd.index("--accum") + 1], "float32")
        self.assertEqual(cmd[cmd.index("--mode") + 1], "benchmark")

    def test_mpi_command_has_processes_and_binaries(self):
        cmd = build_command(self.cfg, "mpi", "benchmark", processes=4)
        self.assertEqual(cmd[:4], ["mpirun", "-np", "4", "./C_OpenMP_MPI/scoring_mpi"])
        self.assertEqual(cmd[cmd.index("--candidates") + 1], "data/candidates_W.f32")

    def test_reject_bad_impl(self):
        with self.assertRaises(ConfigError):
            build_command(self.cfg, "does_not_exist")

    def test_reject_bad_processes(self):
        with self.assertRaises(ConfigError):
            build_command(self.cfg, "mpi", "benchmark", processes=0)
        with self.assertRaises(ConfigError):
            build_command(self.cfg, "mpi", "benchmark", processes=-3)


if __name__ == "__main__":
    unittest.main()
