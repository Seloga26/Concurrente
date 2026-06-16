#!/usr/bin/env bash
# Orquestador del benchmark: ejecuta las implementaciones disponibles sobre la rejilla K x P,
# repite REPS veces, captura el JSON de cada corrida en results/raw_<LABEL>.jsonl y agrega a
# results/benchmark.csv. Portable: en WSL corre Python+C; en Colab ademas CUDA.
#
# Parametros (override por entorno):
#   K_GRID="100000 1000000 10000000"   P_GRID="1 2 4 6 8 10 12"   REPS=3
#   MODES="reference benchmark"        LABEL=wsl   ONLY_IMPLS="..." (restringe impls)
# Ejemplo grid reducido:  K_GRID="100000" P_GRID="1 2 12" REPS=1 bash scripts/run_all.sh
# En Colab (solo GPU + baseline):  ONLY_IMPLS="c_serial cuda cuda_pycuda" P_GRID="1" LABEL=colab bash scripts/run_all.sh
set -eo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"
# shellcheck source=/dev/null
source "$HERE/detect_env.sh"

K_GRID="${K_GRID:-100000 1000000 10000000}"
P_GRID="${P_GRID:-1 2 4 6 8 10 12}"
REPS="${REPS:-3}"
MODES="${MODES:-reference benchmark}"
LABEL="${LABEL:-wsl}"
RAW="results/raw_${LABEL}.jsonl"

# BLAS del recompute monohilo. OJO: NO exportar OMP_NUM_THREADS (romperia el escalado de c_openmp;
# las impls Python ya fijan sus vars BLAS en-proceso).
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

mkdir -p results
: > "$RAW"

IMPLS="$(available_impls)"
# Restringir impls via env ONLY_IMPLS (interseccion con las disponibles). Util en Colab para
# correr solo CUDA + un baseline c_serial:  ONLY_IMPLS="c_serial cuda cuda_pycuda" ...
if [ -n "${ONLY_IMPLS:-}" ]; then
  filtered=""
  for i in $IMPLS; do for w in $ONLY_IMPLS; do [ "$i" = "$w" ] && filtered="$filtered $i"; done; done
  IMPLS="$filtered"
fi
# Recortar P_GRID a nproc.
PGRID=""
for p in $P_GRID; do [ "$p" -le "$NPROC" ] && PGRID="$PGRID $p"; done

echo "[run_all] LABEL=$LABEL impls={$IMPLS} K_GRID={$K_GRID} P={$PGRID} REPS=$REPS MODES={$MODES}"

# Compilar lo disponible.
[ "$HAVE_GCC" = 1 ] && [ "$HAVE_MAKE" = 1 ] && make -C C_OpenMP_MPI >/dev/null
echo "$IMPLS" | grep -qw cuda && make -C CUDA >/dev/null || true

# Construye el comando completo para (impl, mode, K, P) en la variable global CMD (usa $ARGS).
make_cmd() {
  local impl=$1 K=$3 P=$4
  case "$impl" in
    python_sequential) CMD="python3 -m python.sequential $ARGS --K $K" ;;
    python_multicore)  CMD="python3 -m python.multicore $ARGS --K $K --workers $P" ;;
    c_serial)          CMD="./C_OpenMP_MPI/scoring_serial $ARGS --K $K" ;;
    c_openmp)          CMD="./C_OpenMP_MPI/scoring_openmp $ARGS --K $K --threads $P" ;;
    c_mpi)             CMD="mpirun --oversubscribe -np $P ./C_OpenMP_MPI/scoring_mpi $ARGS --K $K" ;;
    cuda)              CMD="./CUDA/scoring_cuda $ARGS --K $K" ;;
    cuda_pycuda)       CMD="python3 CUDA/scoring_pycuda.py $ARGS --K $K --block-size $P" ;;
    *) echo "impl desconocida: $impl" >&2; return 1 ;;
  esac
}

# P aplicable por impl (los seriales/seq/cuda solo P=1; cuda_pycuda usa block-size=256).
p_list_for() {
  case "$1" in
    python_sequential|c_serial|cuda) echo "1" ;;
    cuda_pycuda) echo "256" ;;
    *) echo "$PGRID" ;;
  esac
}

for K in $K_GRID; do
  echo "[run_all] K=$K : regenerando dataset..."
  python3 data/generate_data.py --output-dir data --overwrite --k "$K" >/dev/null
  for mode in $MODES; do
    ARGS="$(python3 scripts/bench_args.py --mode "$mode")"
    for impl in $IMPLS; do
      # Warmup GPU (descarta init de contexto/JIT; no se registra).
      case "$impl" in
        cuda|cuda_pycuda) make_cmd "$impl" "$mode" "$K" 256; eval "$CMD" >/dev/null 2>&1 || true ;;
      esac
      for P in $(p_list_for "$impl"); do
        for r in $(seq 1 "$REPS"); do
          make_cmd "$impl" "$mode" "$K" "$P"
          echo "  $impl mode=$mode K=$K P=$P rep=$r"
          eval "$CMD" >> "$RAW"
        done
      done
    done
  done
done

python3 scripts/aggregate.py --out results "${LABEL}=${RAW}"
echo "[run_all] listo -> results/benchmark.csv (+ benchmark_runs.csv, $RAW)"
