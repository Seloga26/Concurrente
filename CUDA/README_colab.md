# CUDA (Nivel 3) — compilar y validar en Google Colab

El host de desarrollo **no tiene GPU NVIDIA** (Intel Iris Xe), así que la variante CUDA se compila
y valida en **Google Colab** con runtime GPU. Dos implementaciones, mismo kernel y mismo contrato JSON:

- **`scoring_cuda`** (primaria): driver nvcc que reutiliza la capa C de `../C_OpenMP_MPI`
  (`runner.c`/`npyio.c`/`scoring.c`). `implementation = "cuda"`.
- **`scoring_pycuda.py`** (secundaria): mismo kernel (`scoring_device.cuh`) vía PyCUDA, reutilizando
  la capa Python `common/`. `implementation = "cuda_pycuda"`.

Ambas: kernel **fused**, modos `reference` (f64) / `benchmark` (f32), argmax por **clave int64**
reducida con `atomicMax`. Emiten una línea JSON igual al resto + `cuda_block_size`, `cuda_grid_size`,
`device` (omiten `blas_threads`: la búsqueda no usa BLAS).

## Pasos en Colab

1. **Runtime con GPU**: menú *Entorno de ejecución → Cambiar tipo de entorno → GPU* (T4).
2. **Obtener el código** (el repo es local, no está pusheado). Una opción:
   ```python
   # a) si lo subes a un remoto:
   !git clone <URL_DE_TU_REPO> repo && cd repo
   # b) o sube un .zip del proyecto y descomprímelo:
   #    from google.colab import files; files.upload()  # sube proyecto.zip
   #    !unzip -q proyecto.zip -d repo && cd repo
   ```
3. **Dependencias**:
   ```bash
   !pip -q install pycuda
   ```
4. **Validar todo de una** (genera datos, compila, compara contra el oráculo Python):
   ```bash
   !bash CUDA/colab_validate.sh
   ```
   Salida esperada: `cuda` y `cuda_pycuda` → `OK (exacto vs python)` en ambos modos, y
   `VALIDACION CUDA OK`.

## Ejecutar manualmente una variante

```bash
# nvcc:
!make -C CUDA                     # (opcional) NVFLAGS="-O3 -arch=sm_75" para fijar la arch T4
!./CUDA/scoring_cuda --N 50 --K 100000 --mode reference --accum float64 \
   --algorithm literal --tie-atol 1e-9 --tie-rtol 1e-9 \
   --matrix-a data/matrix_A.npy --profile-t data/profile_T.npy --profile-s data/profile_S.npy \
   --profile-f data/profile_F.npy --labels data/labels.npy --candidates data/candidates_W.npy \
   --theta-policy class_mean_midpoint --consistency-threshold 0.8

# PyCUDA (mismos flags + --block-size opcional, default 256):
!python CUDA/scoring_pycuda.py --N 50 --K 100000 --mode reference --accum float64 ... 
```

## Notas

- **`-arch`**: por defecto nvcc usa la arch del toolkit; para la T4 de Colab conviene
  `make NVFLAGS="-O3 -arch=sm_75"`. `atomicMax(unsigned long long)` requiere ≥ sm_35.
- **Benchmark (f32)**: el orden de suma/FMA de la GPU puede diferir del CPU y cambiar `auc_units`
  en ±1 para candidatos límite (variabilidad f32 documentada). En `reference` (f64, argmax entero
  exacto) el resultado coincide con las demás variantes. En el dataset principal (ganador `au=50`,
  bien separado) `best_k=0` coincide en ambos modos.
- Para K grandes (escalado, p.ej. 10⁷) sube el `--K` y regenera el dataset; la GPU es donde más se
  nota el speedup.
