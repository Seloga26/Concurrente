"""Pruebas del núcleo de scoring y búsqueda (common/scoring.py) con fixtures manuales."""
import sys
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

from common.auc import pairwise_auc  # noqa: E402
from common.scoring import (  # noqa: E402
    materialize_p,
    prepare_work_arrays,
    score_candidate,
    search_best,
)

ATOL = 1e-9
RTOL = 1e-9


class ScoringTests(unittest.TestCase):
    def _search(self, fx, accum):
        a, t, s, f = prepare_work_arrays(fx["A"], fx["T"], fx["S"], fx["F"], accum)
        return search_best(a, t, s, f, fx["candidates_W"], fx["pos_idx"], fx["neg_idx"],
                           ATOL, RTOL, np.dtype(accum))

    def test_manual_reference(self):
        self.assertEqual(self._search(manual_scoring_fixture(), "float64"), (50, 0))

    def test_manual_benchmark(self):
        self.assertEqual(self._search(manual_scoring_fixture(), "float32"), (50, 0))

    def test_per_candidate_auc_units(self):
        fx = manual_scoring_fixture()
        a, t, s, f = prepare_work_arrays(fx["A"], fx["T"], fx["S"], fx["F"], "float64")
        for k, expected in enumerate(fx["expected_auc_units"]):
            w = fx["candidates_W"][k]
            score = score_candidate(np.float64(w[0]), np.float64(w[1]), np.float64(w[2]), a, t, s, f)
            self.assertEqual(
                pairwise_auc(score, fx["pos_idx"], fx["neg_idx"], ATOL, RTOL).auc_units, expected)

    def test_tie_smallest_index(self):
        self.assertEqual(self._search(tie_fixture(), "float64"), (50, 2))

    def test_dtype_both_modes(self):
        fx = manual_scoring_fixture()
        for accum in ("float64", "float32"):
            a, t, s, f = prepare_work_arrays(fx["A"], fx["T"], fx["S"], fx["F"], accum)
            st = np.dtype(accum).type
            w = fx["candidates_W"][0]
            p = materialize_p(st(w[0]), st(w[1]), st(w[2]), t, s, f)
            score = a @ p
            self.assertEqual(p.dtype, np.dtype(accum))
            self.assertEqual(score.dtype, np.dtype(accum))

    def test_nonboundary_same_winner_both_modes(self):
        fx = manual_scoring_fixture()
        self.assertEqual(self._search(fx, "float64"), self._search(fx, "float32"))


if __name__ == "__main__":
    unittest.main()
