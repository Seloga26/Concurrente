/* Orquestacion compartida por las variantes nativas (serial, OpenMP).
 *
 * Hace todo lo comun: parseo de CLI (los mismos flags que python/sequential), carga y
 * validacion de los .npy, preparacion de los arreglos de trabajo, recompute del ganador
 * (AUC detallado, theta, consistencia) y emision de UNA linea JSON a stdout. La estrategia
 * de busqueda (serial vs OpenMP) se inyecta como funcion.
 */
#ifndef RUNNER_H
#define RUNNER_H

#include "scoring.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Argumentos parseados de la CLI. */
typedef struct {
    long   N, K;
    const char *mode;          /* "reference" | "benchmark" */
    const char *accum;         /* "float64"   | "float32"   */
    const char *algorithm;     /* "literal" */
    double tie_atol, tie_rtol;
    const char *matrix_a, *profile_t, *profile_s, *profile_f, *labels, *candidates;
    const char *theta_policy;  /* "class_mean_midpoint" */
    double consistency_threshold;
    int    threads;            /* OpenMP: nº de hilos (0 = automatico) */
    const char *output_json;   /* opcional */
} RunnerArgs;

/* Resultado de la busqueda cronometrada que la estrategia debe rellenar. */
typedef struct {
    long   best_units;
    long   best_k;
    int    n_threads;          /* 0 => no emitir "n_threads"; >0 => emitirlo (OpenMP) */
    int    n_procs;            /* 0 => no emitir "n_procs";   >0 => emitirlo (MPI)    */
    int    is_root;            /* 1 => este proceso emite el JSON (MPI: solo rank 0)  */
    int    cuda_block_size;    /* 0 => no emitir; >0 => emitir "cuda_block_size" (CUDA) */
    long   cuda_grid_size;     /* 0 => no emitir; >0 => emitir "cuda_grid_size"  (CUDA) */
    const char *device_name;   /* NULL => no emitir; si no => emitir "device" (CUDA)    */
    double t_core_seconds;
    double t_search_seconds;
} SearchOutcome;

/* Estrategia de busqueda: recibe los datos preparados y los argumentos, ejecuta la region
 * cronometrada y rellena 'out'. Devuelve 0 en exito. */
typedef int (*search_fn)(const ScoringData *sd, const RunnerArgs *args, SearchOutcome *out);

/* Punto de entrada: parsea, carga, busca (via 'search'), recompone y emite JSON.
 * Devuelve el codigo de salida del proceso (0 = ok). */
int runner_main(int argc, char **argv, const char *impl_name, search_fn search);

/* Reloj monotono en segundos (para cronometrar la busqueda). */
double runner_now_seconds(void);

#ifdef __cplusplus
}
#endif

#endif /* RUNNER_H */
