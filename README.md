# Optimización Paralela del Sistema de Scoring Metagenómico

Proyecto de Computación de Alto Rendimiento (HPC). Busca un vector de pesos
`W = (W1, W2, W3)` sobre el simplex que **maximice el AUC** en la clasificación
binaria (sano `y=0` / enfermo `y=1`) de 10 muestras metagenómicas, y compara el
rendimiento de cinco implementaciones: Python secuencial, Python multicore,
C + OpenMP, C + MPI y CUDA.

> **Estado actual:** Fase 1 (estructura y configuración). El generador, el
> scoring y las implementaciones paralelas aún no están implementados.

## Modelo (resumen)

- `P_i = W1*T_i + W2*S_i + W3*F_i`; `Score = A · P` (forma **literal** del PDF, O(NK), sin precomputar G).
- `algorithm = "literal"` identifica la misma ecuación en todas las implementaciones; el `kernel_variant` indica cómo la realiza cada una: Python = `materialized_numpy` (materializa P y usa `A @ P`); el código nativo (C/OpenMP/MPI/CUDA) usará `fused` (`Score_j += A[j,i]*(W1*T_i+W2*S_i+W3*F_i)`, sin materializar P).
- AUC por comparaciones de pares, representado como entero `auc_units = 2*wins + ties ∈ [0,50]`.
- Argmax con clave int64 `key = auc_units*K + (K-1-k)` (desempate por **menor índice global**).

## Estructura del repositorio

```
.
├── experiment_config.json   # FUENTE DE VERDAD de ejecución/benchmark (contrato de scoring)
├── generation_config.json   # parámetros públicos del generador (semillas, signal_strength, ...)
├── config/launcher.py       # lee el JSON y construye comandos CLI (no ejecuta en fase 1)
├── common/                  # utilidades de scoring (AUC, clave, métricas) — fases posteriores
├── data/                    # generador + datos generados (.npy/.f32, ignorados por git)
├── python/                  # Nivel 1: sequential.py, multicore.py
├── C_OpenMP_MPI/            # Nivel 2: C serial/OpenMP/MPI + Makefile
├── CUDA/                    # Nivel 3: scoring_kernel.cu (+ scoring_pycuda.py secundario)
├── tests/                   # pruebas unittest (sin necesidad de pytest)
├── scripts/                 # run_all.sh, smoke.sh, detect_env.sh (fases posteriores)
├── results/                 # benchmark.csv y plots/ (salida, ignorada por git)
└── report/                  # informe técnico (fuente .md versionada, .pdf ignorado)
```

## Dos modos de precisión

- `reference`: acumulación **float64**, recorrido de ítems en orden ascendente (oráculo de corrección).
- `benchmark`: acumulación **float32** (todas las implementaciones). El rendimiento se compara **solo dentro del mismo modo**.
- Suma directa en ambos modos; la suma compensada/pairwise queda como experimento numérico adicional.

## Requisitos y entorno objetivo

- **Python 3.10+** (probado en 3.13). Dependencias en `requirements.txt`.
- Entorno objetivo de Niveles 2–3: **Linux / WSL Ubuntu** con `build-essential`
  (gcc + OpenMP), una implementación MPI (OpenMPI/MPICH) y CUDA Toolkit (`nvcc`).
- El launcher y las pruebas de configuración **no requieren dependencias externas**.

## Uso (fase 1)

```bash
# Ejecutar las pruebas de configuración (sin instalar nada):
python -m unittest discover -s tests -v

# Construir (sin ejecutar) el comando de una implementación:
python config/launcher.py --impl python_sequential --mode reference --dry-run
python config/launcher.py --impl mpi --processes 4 --mode benchmark --dry-run
```

## Degradación sin MPI/GPU

El proyecto está diseñado para degradarse con gracia: si faltan `nvcc`,
`mpicc`/`mpirun` o `gcc`, las implementaciones correspondientes se omiten y el
benchmark marca esas filas como no disponibles. El Nivel 1 (Python) funciona
siempre.

## Notas

- `Proyecto.pdf` (enunciado) se conserva intacto en la raíz.
- No se hace `git push` automáticamente; los commits los revisa y publica el autor.
