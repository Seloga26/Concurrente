"""Pruebas del AUC por pares (common/auc.py): núcleo escalar y detallado."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.auc import pairwise_auc, pairwise_auc_units  # noqa: E402

ATOL = 1e-9
RTOL = 1e-9
NEG = [0, 1, 2, 3, 4]
POS = [5, 6, 7, 8, 9]


def scores(neg_vals, pos_vals):
    return np.array(list(neg_vals) + list(pos_vals), dtype=np.float64)


class AucTests(unittest.TestCase):
    def units(self, s, atol=ATOL, rtol=RTOL):
        return pairwise_auc_units(s, POS, NEG, atol, rtol)

    def res(self, s):
        return pairwise_auc(s, POS, NEG, ATOL, RTOL)

    def test_perfect(self):
        s = scores([0.1] * 5, [0.9] * 5)
        self.assertEqual(self.units(s), 50)
        r = self.res(s)
        self.assertEqual((r.auc_units, r.denominator, r.wins, r.ties), (50, 50, 25, 0))
        self.assertEqual(r.auc, 1.0)

    def test_inverted(self):
        s = scores([0.9] * 5, [0.1] * 5)
        self.assertEqual(self.units(s), 0)
        self.assertEqual(self.res(s).auc, 0.0)

    def test_all_tied(self):
        s = scores([0.5] * 5, [0.5] * 5)
        self.assertEqual(self.units(s), 25)
        self.assertEqual(self.res(s).ties, 25)

    def test_some_ties(self):
        # pos todos 1; neg=[1,0,0,0,0] -> 5 empates + 20 wins -> 45
        s = scores([1, 0, 0, 0, 0], [1, 1, 1, 1, 1])
        self.assertEqual(self.units(s), 45)
        r = self.res(s)
        self.assertEqual((r.wins, r.ties, r.auc_units), (20, 5, 45))
        self.assertAlmostEqual(r.auc, 0.9)

    def test_near_tie_within_tol(self):
        s = scores([1.0] * 5, [1.0 + 5e-7] * 5)   # diff 5e-7 < banda 1e-6 -> empates
        self.assertEqual(self.units(s, atol=1e-6, rtol=1e-9), 25)

    def test_near_tie_outside_tol(self):
        s = scores([1.0] * 5, [1.0 + 1e-3] * 5)   # diff 1e-3 > banda -> wins
        self.assertEqual(self.units(s, atol=1e-9, rtol=1e-9), 50)

    def test_missing_class(self):
        s = np.zeros(10)
        with self.assertRaises(ValueError):
            pairwise_auc_units(s, [], NEG, ATOL, RTOL)
        with self.assertRaises(ValueError):
            pairwise_auc(s, POS, [], ATOL, RTOL)

    def test_core_matches_detailed(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            s = rng.random(10)
            self.assertEqual(self.units(s), self.res(s).auc_units)


if __name__ == "__main__":
    unittest.main()
