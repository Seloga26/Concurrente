/* Variante C + OpenMP (Nivel 2): reparte el rango [0,K) de candidatos entre hilos.
 *
 * Cada hilo mantiene su mejor clave int64 local (au*K + (K-1-k)); la reduccion final toma
 * el maximo (mayor auc_units; empate => menor k global), igual que common.keys.better en
 * Python. Misma CLI/contrato JSON que la variante serial + el campo "n_threads".
 *   t_core   = maximo tiempo de computo por hilo (span paralelo util, para Amdahl).
 *   t_search = wall-time de la region paralela (incl. dispatch y reduccion).
 */
#include "runner.h"

#include <omp.h>

static int search_openmp(const ScoringData *sd, const RunnerArgs *args, SearchOutcome *out) {
    const long K = sd->n_cand;
    int nthreads = (args->threads > 0) ? args->threads : omp_get_max_threads();
    omp_set_num_threads(nthreads);

    long long best_key = -1;     /* centinela: pierde ante cualquier candidato (key >= 0) */
    double t_core_max = 0.0;

    double t0 = omp_get_wtime();
    #pragma omp parallel
    {
        long long local_best = -1;
        double tw0 = omp_get_wtime();

        #pragma omp for schedule(static) nowait
        for (long k = 0; k < K; ++k) {
            long au = sd_eval(sd, k);
            long long key = sd_pack_key(au, k, K);
            if (key > local_best) local_best = key;
        }
        double tw1 = omp_get_wtime();

        #pragma omp critical
        {
            if (local_best > best_key) best_key = local_best;
            double mine = tw1 - tw0;
            if (mine > t_core_max) t_core_max = mine;
        }
    }
    double t1 = omp_get_wtime();

    long bu, bk;
    sd_unpack_key(best_key, K, &bu, &bk);
    out->best_units = bu;
    out->best_k = bk;
    out->t_core_seconds = t_core_max;
    out->t_search_seconds = t1 - t0;
    out->n_threads = nthreads;
    return 0;
}

int main(int argc, char **argv) {
    return runner_main(argc, argv, "c_openmp", search_openmp);
}
