#!/usr/bin/env bash
# Valida las variantes CUDA (scoring_cuda por nvcc + scoring_pycuda) contra el oraculo
# python.sequential, en Google Colab (runtime GPU). Compara best_k, auc_units, auc, best_w,
# scores, theta y consistency en ambos modos (reference/benchmark).
#
# Uso en Colab (desde la raiz del repo):  bash CUDA/colab_validate.sh
# Requiere: nvcc + GPU NVIDIA, python3 + numpy, pycuda (pip install pycuda), y el dataset.
set -e

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
D="$ROOT/data"
cd "$ROOT"

# Generar el dataset si falta.
if [ ! -f "$D/candidates_W.npy" ]; then
  echo "[colab] generando dataset..."
  python3 data/generate_data.py --output-dir data --overwrite
fi

echo "[colab] compilando scoring_cuda (nvcc)..."
make -C "$HERE" >/dev/null

ARGS="--algorithm literal --tie-atol 1e-9 --tie-rtol 1e-9 \
  --matrix-a $D/matrix_A.npy --profile-t $D/profile_T.npy --profile-s $D/profile_S.npy \
  --profile-f $D/profile_F.npy --labels $D/labels.npy --candidates $D/candidates_W.npy \
  --theta-policy class_mean_midpoint --consistency-threshold 0.8"

cmp_mode() {
  local mode=$1 accum=$2
  python3 -m python.sequential --N 50 --K 100000 --mode "$mode" --accum "$accum" $ARGS > /tmp/py.json
  "$HERE/scoring_cuda"      --N 50 --K 100000 --mode "$mode" --accum "$accum" $ARGS > /tmp/cu.json
  python3 "$HERE/scoring_pycuda.py" --N 50 --K 100000 --mode "$mode" --accum "$accum" $ARGS > /tmp/cp.json
  python3 - "$mode" <<'PY'
import json, sys
mode = sys.argv[1]
py = json.load(open("/tmp/py.json")); cu = json.load(open("/tmp/cu.json")); cp = json.load(open("/tmp/cp.json"))
def close(a, b):
    if isinstance(a, list): return all(close(x, y) for x, y in zip(a, b))
    if isinstance(a, float): return abs(a - b) <= 1e-12 + 1e-9 * max(abs(a), abs(b))
    return a == b
keys = ["best_k","auc_units","auc_denominator","auc","best_w","scores","theta",
        "consistency","consistency_pass"]
print(f"== mode={mode} ==")
ok = True
for impl, d in (("cuda", cu), ("cuda_pycuda", cp)):
    bad = [k for k in keys if not close(py[k], d[k])]
    status = "OK  (exacto vs python)" if not bad else f"DIFIERE en {bad}"
    print(f"  {impl:11s}: best_k={d['best_k']} auc_units={d['auc_units']} "
          f"theta={d['theta']:.12g}  dev={d.get('device','?')}  -> {status}")
    ok = ok and not bad
print(f"  t_core(kernel): python={py['t_search_seconds']:.4f}s  cuda={cu['t_core_seconds']:.5f}s  "
      f"cuda_pycuda={cp['t_core_seconds']:.5f}s")
sys.exit(0 if ok else 1)
PY
}

cmp_mode reference float64
cmp_mode benchmark float32
echo "VALIDACION CUDA OK"
