/* Variante C + MPI (Nivel 2): reparte el rango [0,K) de candidatos entre procesos (ranks).
 *
 * Cada rank carga los mismos .npy (carga replicada, fuera del cronometro), busca su sub-rango
 * contiguo y reduce su mejor clave int64 local (au*K + (K-1-k)). Como la clave codifica el
 * desempate en un solo entero a maximizar, la reduccion global es un MPI_MAX directo (sin
 * MAXLOC): mayor auc_units y, ante empate, menor k global -> invariante al nº de procesos.
 * Solo el rank 0 emite la unica linea JSON (gateado en runner_main via out->is_root).
 *   t_core   = maximo tiempo de computo por rank (span paralelo util, para Amdahl).
 *   t_search = maximo wall por rank incl. la reduccion.
 * Misma CLI/contrato JSON que serial/openmp + el campo "n_procs".
 */
#include "runner.h"

#include <mpi.h>

/* Particion contigua de [0,K) por rank (identica a _make_chunks de python/multicore):
 * los primeros (K % size) ranks reciben un candidato extra. k es indice GLOBAL. */
static void chunk_range(long K, int rank, int size, long *start, long *stop) {
    long base = K / size;
    long rem  = K % size;
    long extra_before = (rank < rem) ? rank : rem;
    *start = rank * base + extra_before;
    long len = base + (rank < rem ? 1 : 0);
    *stop = *start + len;
}

static int search_mpi(const ScoringData *sd, const RunnerArgs *args, SearchOutcome *out) {
    (void)args;
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const long K = sd->n_cand;
    long start, stop;
    chunk_range(K, rank, size, &start, &stop);

    double t0 = MPI_Wtime();
    long lu = -1, lk = -1;
    sd_search_range(sd, start, stop, &lu, &lk);          /* rango vacio => lu = -1 */
    long long local_key = (lu < 0) ? -1 : sd_pack_key(lu, lk, K);
    double t_compute = MPI_Wtime() - t0;

    /* Reduccion del ganador: max de la clave int64 (centinela -1 pierde). */
    long long global_key = -1;
    MPI_Allreduce(&local_key, &global_key, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);

    /* Tiempos: max del computo por rank (Amdahl) y max del wall incl. reduccion. */
    double local_wall = MPI_Wtime() - t0;
    double t_core_max = 0.0, t_search_max = 0.0;
    MPI_Allreduce(&t_compute, &t_core_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&local_wall, &t_search_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    long bu, bk;
    sd_unpack_key(global_key, K, &bu, &bk);
    out->best_units = bu;
    out->best_k = bk;
    out->t_core_seconds = t_core_max;
    out->t_search_seconds = t_search_max;
    out->n_procs = size;
    out->is_root = (rank == 0);
    return 0;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rc = runner_main(argc, argv, "c_mpi", search_mpi);
    MPI_Finalize();
    return rc;
}
