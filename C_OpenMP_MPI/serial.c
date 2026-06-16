/* Variante C serial de control (Nivel 2): un solo hilo, recorre todo el rango [0,K).
 *
 * Sirve de baseline de rendimiento nativo (comparable con OpenMP/MPI/CUDA bajo el mismo
 * kernel "fused"). Misma CLI y mismo contrato JSON que python/sequential (sin blas_threads:
 * no hay capa BLAS). Invocacion: ./serial <flags>.
 */
#include "runner.h"

static int search_serial(const ScoringData *sd, const RunnerArgs *args, SearchOutcome *out) {
    (void)args;
    double t0 = runner_now_seconds();
    sd_search_range(sd, 0, sd->n_cand, &out->best_units, &out->best_k);
    double t1 = runner_now_seconds();
    out->t_core_seconds = t1 - t0;
    out->t_search_seconds = t1 - t0;   /* serial: misma region */
    out->n_threads = 0;                /* => no se emite "n_threads" */
    return 0;
}

int main(int argc, char **argv) {
    return runner_main(argc, argv, "c_serial", search_serial);
}
