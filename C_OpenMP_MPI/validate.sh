#!/usr/bin/env bash
# Valida las variantes nativas (scoring_serial / scoring_openmp) contra el oraculo
# python.sequential: compara best_k, auc_units, auc, best_w, scores, theta y consistency
# en ambos modos (reference/benchmark). Construye los binarios si faltan.
#
# Uso:  bash C_OpenMP_MPI/validate.sh        (desde cualquier directorio)
# Requiere: gcc/make, python3 + numpy, y el dataset generado en data/.
set -e

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
D="$ROOT/data"
cd "$ROOT"

make -C "$HERE" >/dev/null

ARGS="--algorithm literal --tie-atol 1e-9 --tie-rtol 1e-9 \
  --matrix-a $D/matrix_A.npy --profile-t $D/profile_T.npy --profile-s $D/profile_S.npy \
  --profile-f $D/profile_F.npy --labels $D/labels.npy --candidates $D/candidates_W.npy \
  --theta-policy class_mean_midpoint --consistency-threshold 0.8"

cmp_mode() {
  local mode=$1 accum=$2
  python3 -m python.sequential --N 50 --K 100000 --mode "$mode" --accum "$accum" $ARGS > /tmp/py.json
  "$HERE/scoring_serial" --N 50 --K 100000 --mode "$mode" --accum "$accum" $ARGS > /tmp/cs.json
  "$HERE/scoring_openmp" --N 50 --K 100000 --mode "$mode" --accum "$accum" $ARGS > /tmp/co.json
  python3 - "$mode" <<'PY'
import json, sys
mode = sys.argv[1]
py = json.load(open("/tmp/py.json")); cs = json.load(open("/tmp/cs.json")); co = json.load(open("/tmp/co.json"))
def close(a, b):
    if isinstance(a, list): return all(close(x, y) for x, y in zip(a, b))
    if isinstance(a, float): return abs(a - b) <= 1e-12 + 1e-9 * max(abs(a), abs(b))
    return a == b
keys = ["best_k","auc_units","auc_denominator","auc","best_w","scores","theta",
        "consistency","consistency_pass"]
print(f"== mode={mode} ==")
ok = True
for impl, d in (("c_serial", cs), ("c_openmp", co)):
    bad = [k for k in keys if not close(py[k], d[k])]
    status = "OK  (exacto vs python)" if not bad else f"DIFIERE en {bad}"
    print(f"  {impl:9s}: best_k={d['best_k']} auc_units={d['auc_units']} "
          f"theta={d['theta']:.12g}  -> {status}")
    ok = ok and not bad
print(f"  t_search: python={py['t_search_seconds']:.4f}s  c_serial={cs['t_search_seconds']:.4f}s  "
      f"c_openmp={co['t_search_seconds']:.4f}s (x{co.get('n_threads','?')} hilos)")
sys.exit(0 if ok else 1)
PY
}

cmp_mode reference float64
cmp_mode benchmark float32
echo "VALIDACION OK"
