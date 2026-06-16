"""Pruebas de theta y consistencia (common/metrics.py)."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.metrics import compute_theta, consistency  # noqa: E402

NEG = [0, 1, 2, 3, 4]
POS = [5, 6, 7, 8, 9]


class MetricsTests(unittest.TestCase):
    def test_theta_midpoint(self):
        s = np.array([0.2] * 5 + [0.8] * 5)
        self.assertAlmostEqual(compute_theta(s, POS, NEG), 0.5)

    def test_consistency_perfect(self):
        s = np.array([0.1] * 5 + [0.9] * 5)
        theta = compute_theta(s, POS, NEG)
        self.assertAlmostEqual(consistency(s, theta, POS, NEG), 1.0)

    def test_consistency_rules(self):
        # theta=0.5 ; enfermo: score>theta ; sano: score<=theta
        s = np.array([0.5, 0.4, 0.6, 0.5, 0.4, 0.6, 0.6, 0.4, 0.6, 0.6])
        theta = 0.5
        tn = sum(s[n] <= theta for n in NEG)   # 0.5,0.4,0.6,0.5,0.4 -> 4
        tp = sum(s[p] > theta for p in POS)    # 0.6,0.6,0.4,0.6,0.6 -> 4
        self.assertAlmostEqual(consistency(s, theta, POS, NEG), 0.5 * (tp / 5 + tn / 5))

    def test_missing_class(self):
        with self.assertRaises(ValueError):
            compute_theta(np.zeros(10), [], NEG)
        with self.assertRaises(ValueError):
            consistency(np.zeros(10), 0.5, POS, [])


if __name__ == "__main__":
    unittest.main()
