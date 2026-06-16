#!/usr/bin/env bash
# Detecta el toolchain disponible y deriva la lista de implementaciones ejecutables aqui.
# Se puede:
#   - ejecutar directamente:  bash scripts/detect_env.sh   (imprime un reporte)
#   - sourcear desde run_all.sh:  source scripts/detect_env.sh; available_impls

_have() { command -v "$1" >/dev/null 2>&1; }
_py_mod() { python3 -c "import $1" >/dev/null 2>&1; }

HAVE_PY=0;    _have python3 && HAVE_PY=1
HAVE_NUMPY=0; [ "$HAVE_PY" = 1 ] && _py_mod numpy && HAVE_NUMPY=1
HAVE_GCC=0;   _have gcc && HAVE_GCC=1
HAVE_MAKE=0;  _have make && HAVE_MAKE=1
HAVE_MPICC=0; _have mpicc && HAVE_MPICC=1
HAVE_MPIRUN=0; _have mpirun && HAVE_MPIRUN=1
HAVE_NVCC=0;  _have nvcc && HAVE_NVCC=1
HAVE_GPU=0;   nvidia-smi -L >/dev/null 2>&1 && HAVE_GPU=1
HAVE_PYCUDA=0; [ "$HAVE_PY" = 1 ] && _py_mod pycuda && HAVE_PYCUDA=1
NPROC=$(nproc 2>/dev/null || echo 1)

# Lista de impls ejecutables segun lo detectado.
available_impls() {
  local impls=""
  if [ "$HAVE_PY" = 1 ] && [ "$HAVE_NUMPY" = 1 ]; then
    impls="$impls python_sequential python_multicore"
  fi
  if [ "$HAVE_GCC" = 1 ] && [ "$HAVE_MAKE" = 1 ]; then
    impls="$impls c_serial c_openmp"
  fi
  if [ "$HAVE_MPICC" = 1 ] && [ "$HAVE_MPIRUN" = 1 ]; then
    impls="$impls c_mpi"
  fi
  if [ "$HAVE_NVCC" = 1 ] && [ "$HAVE_GPU" = 1 ]; then
    impls="$impls cuda"
    [ "$HAVE_PYCUDA" = 1 ] && impls="$impls cuda_pycuda"
  fi
  echo $impls
}

_report() {
  echo "=== Entorno detectado ==="
  printf "  %-14s %s\n" "python3"  "$([ $HAVE_PY = 1 ] && python3 --version 2>&1 || echo NO)"
  printf "  %-14s %s\n" "numpy"    "$([ $HAVE_NUMPY = 1 ] && echo si || echo NO)"
  printf "  %-14s %s\n" "gcc"      "$([ $HAVE_GCC = 1 ] && gcc -dumpversion || echo NO)"
  printf "  %-14s %s\n" "make"     "$([ $HAVE_MAKE = 1 ] && echo si || echo NO)"
  printf "  %-14s %s\n" "mpicc"    "$([ $HAVE_MPICC = 1 ] && echo si || echo NO)"
  printf "  %-14s %s\n" "mpirun"   "$([ $HAVE_MPIRUN = 1 ] && echo si || echo NO)"
  printf "  %-14s %s\n" "nvcc"     "$([ $HAVE_NVCC = 1 ] && echo si || echo NO)"
  printf "  %-14s %s\n" "GPU"      "$([ $HAVE_GPU = 1 ] && nvidia-smi -L 2>/dev/null | head -1 || echo NO)"
  printf "  %-14s %s\n" "pycuda"   "$([ $HAVE_PYCUDA = 1 ] && echo si || echo NO)"
  printf "  %-14s %s\n" "nproc"    "$NPROC"
  echo "  impls disponibles: $(available_impls)"
}

# Si se ejecuta directamente (no se sourcea), imprimir el reporte.
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  _report
fi
