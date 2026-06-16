"""Pruebas de la clave int64 del argmax (common/keys.py)."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.keys import (  # noqa: E402
    INT64_MAX,
    KEY_FACTOR,
    KEY_SENTINEL,
    MAIN_MAX_AUC_UNITS,
    better,
    is_sentinel,
    pack_key,
    unpack_key,
    validate_K,
)


class KeyTests(unittest.TestCase):
    def test_constants(self):
        self.assertEqual(MAIN_MAX_AUC_UNITS, 50)
        self.assertEqual(KEY_FACTOR, 51)
        self.assertEqual(KEY_SENTINEL, -1)

    def test_pack_unpack_roundtrip(self):
        for au, k, K in [(0, 0, 1), (50, 0, 100), (37, 5, 100), (50, 99, 100), (0, 99, 100)]:
            self.assertEqual(unpack_key(pack_key(au, k, K), K), (au, k))

    def test_min_index_wins_on_tie(self):
        K = 100
        self.assertGreater(pack_key(50, 3, K), pack_key(50, 7, K))

    def test_better(self):
        self.assertTrue(better(50, 3, 50, 7))    # mismo units, menor k
        self.assertTrue(better(50, 9, 49, 0))    # mayor units
        self.assertFalse(better(49, 0, 50, 9))

    def test_sentinel_loses(self):
        self.assertTrue(better(0, 5, -1, 0))     # candidato válido vence al centinela (units=-1)
        self.assertTrue(is_sentinel(KEY_SENTINEL))
        self.assertFalse(is_sentinel(0))

    def test_unpack_rejects_negative(self):
        with self.assertRaises(ValueError):
            unpack_key(-1, 100)

    def test_pack_validations(self):
        with self.assertRaises(ValueError):
            pack_key(51, 0, 100)        # auc_units > max
        with self.assertRaises(ValueError):
            pack_key(-1, 0, 100)
        with self.assertRaises(ValueError):
            pack_key(50, 100, 100)      # k >= K
        with self.assertRaises(ValueError):
            pack_key(50, -1, 100)

    def test_validate_K_overflow(self):
        self.assertEqual(validate_K(INT64_MAX // 51), INT64_MAX // 51)
        with self.assertRaises(ValueError):
            validate_K(INT64_MAX // 51 + 1)
        with self.assertRaises(ValueError):
            validate_K(0)

    def test_generalized_max_auc_units(self):
        # n_pos=n_neg=10 -> denom 200 -> max auc_units 200
        key = pack_key(200, 5, 1000, max_auc_units=200)
        self.assertEqual(unpack_key(key, 1000), (200, 5))
        with self.assertRaises(ValueError):
            pack_key(201, 5, 1000, max_auc_units=200)


if __name__ == "__main__":
    unittest.main()
