#!/usr/bin/env bash
# Sanity rapido (~segundos): K pequeno, corre cada impl disponible una vez y verifica que emite
# una linea JSON con best_k/auc_units. No escribe CSV. Util antes de un run_all largo.
set -eo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"
# shellcheck source=/dev/null
source "$HERE/detect_env.sh"

export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
K="${K:-2000}"
MODE="${MODE:-reference}"

echo "[smoke] dataset K=$K..."
python3 data/generate_data.py --output-dir data --overwrite --k "$K" >/dev/null
[ "$HAVE_GCC" = 1 ] && [ "$HAVE_MAKE" = 1 ] && make -C C_OpenMP_MPI >/dev/null
echo "$(available_impls)" | grep -qw cuda && make -C CUDA >/dev/null || true

ARGS="$(python3 scripts/bench_args.py --mode "$MODE") --K $K"
P="$NPROC"
fail=0

check() {  # $1=etiqueta ; $2..=comando
  local label=$1; shift
  local out
  if ! out="$("$@" 2>/dev/null)"; then echo "  $label: ERROR al ejecutar"; fail=1; return; fi
  if echo "$out" | python3 -c "import json,sys; d=json.load(sys.stdin); assert 'best_k' in d and 'auc_units' in d; print('best_k=%d auc_units=%d' % (d['best_k'], d['auc_units']))" 2>/dev/null; then
    :
  else
    echo "  $label: JSON invalido"; fail=1
  fi
}

echo "[smoke] corriendo impls (P=$P)..."
for impl in $(available_impls); do
  printf "  %-14s " "$impl"
  case "$impl" in
    python_sequential) check "$impl" python3 -m python.sequential $ARGS ;;
    python_multicore)  check "$impl" python3 -m python.multicore $ARGS --workers "$P" ;;
    c_serial)          check "$impl" ./C_OpenMP_MPI/scoring_serial $ARGS ;;
    c_openmp)          check "$impl" ./C_OpenMP_MPI/scoring_openmp $ARGS --threads "$P" ;;
    c_mpi)             check "$impl" mpirun --oversubscribe -np "$P" ./C_OpenMP_MPI/scoring_mpi $ARGS ;;
    cuda)              check "$impl" ./CUDA/scoring_cuda $ARGS ;;
    cuda_pycuda)       check "$impl" python3 CUDA/scoring_pycuda.py $ARGS --block-size 256 ;;
  esac
done

[ "$fail" = 0 ] && echo "[smoke] OK" || { echo "[smoke] FALLO"; exit 1; }
