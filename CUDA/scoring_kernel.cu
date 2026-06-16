/* Variante CUDA (Nivel 3): driver host, implementation = "cuda".
 *
 * Reutiliza la capa C de C_OpenMP_MPI/ (runner.c, npyio.c, scoring.c): este .cu solo aporta la
 * search_fn (sube datos a la GPU, lanza el kernel y reduce la clave int64) y el main, igual que
 * serial.c/openmp.c/mpi.c. El recompute del ganador y la emision del JSON los hace runner_main en
 * la CPU, asi los campos scores/theta/consistency coinciden con python.sequential.
 *
 * Compila con nvcc; ver CUDA/Makefile. Sin GPU NVIDIA local: se valida en Google Colab.
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

#include "runner.h"            /* capa C, con guardas extern "C" */
#include "scoring_device.cuh" /* kernels search_kernel_f32 / _f64 */

#define CUDA_CHECK(call)                                                            \
    do {                                                                            \
        cudaError_t _e = (call);                                                    \
        if (_e != cudaSuccess) {                                                    \
            fprintf(stderr, "error CUDA en %s:%d: %s\n", __FILE__, __LINE__,        \
                    cudaGetErrorString(_e));                                        \
            return 1;                                                               \
        }                                                                           \
    } while (0)

/* Sube A,T,S,F al dtype de trabajo activo (double=reference / float=benchmark). Devuelve los
 * punteros device en dA..dF (tamanos: A = M*N, T/S/F = N). 0 en exito. */
static int upload_work_arrays(const ScoringData *sd, int M, int N,
                              void **dA, void **dT, void **dS, void **dF) {
    const size_t na = (size_t)M * N, nv = (size_t)N;
    if (sd->use_double) {
        const size_t e = sizeof(double);
        CUDA_CHECK(cudaMalloc(dA, na * e)); CUDA_CHECK(cudaMalloc(dT, nv * e));
        CUDA_CHECK(cudaMalloc(dS, nv * e)); CUDA_CHECK(cudaMalloc(dF, nv * e));
        CUDA_CHECK(cudaMemcpy(*dA, sd->Ad, na * e, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(*dT, sd->Td, nv * e, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(*dS, sd->Sd, nv * e, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(*dF, sd->Fd, nv * e, cudaMemcpyHostToDevice));
    } else {
        const size_t e = sizeof(float);
        CUDA_CHECK(cudaMalloc(dA, na * e)); CUDA_CHECK(cudaMalloc(dT, nv * e));
        CUDA_CHECK(cudaMalloc(dS, nv * e)); CUDA_CHECK(cudaMalloc(dF, nv * e));
        CUDA_CHECK(cudaMemcpy(*dA, sd->Af, na * e, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(*dT, sd->Tf, nv * e, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(*dS, sd->Sf, nv * e, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(*dF, sd->Ff, nv * e, cudaMemcpyHostToDevice));
    }
    return 0;
}

extern "C" int search_cuda(const ScoringData *sd, const RunnerArgs *args, SearchOutcome *out) {
    (void)args;
    const int  M = sd->n_samples;
    const int  N = sd->n_items;
    const long K = sd->n_cand;

    /* Nombre del dispositivo (para el JSON). */
    static char device_name[256];
    int dev = 0;
    cudaDeviceProp prop;
    if (cudaGetDevice(&dev) == cudaSuccess && cudaGetDeviceProperties(&prop, dev) == cudaSuccess) {
        strncpy(device_name, prop.name, sizeof(device_name) - 1);
        device_name[sizeof(device_name) - 1] = '\0';
        out->device_name = device_name;
    }

    /* --- Setup fuera del cronometro: A/T/S/F (diminutos), pos/neg, candidatos, g_best --- */
    void *dA = NULL, *dT = NULL, *dS = NULL, *dF = NULL;
    float *dCand = NULL;
    int *dPos = NULL, *dNeg = NULL;
    unsigned long long *dBest = NULL;
    int rc = 1;

    if (upload_work_arrays(sd, M, N, &dA, &dT, &dS, &dF) != 0) goto cleanup;

    CUDA_CHECK(cudaMalloc(&dPos, (size_t)sd->n_pos * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dNeg, (size_t)sd->n_neg * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dPos, sd->pos, (size_t)sd->n_pos * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dNeg, sd->neg, (size_t)sd->n_neg * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&dCand, (size_t)K * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dBest, sizeof(unsigned long long)));

    {
        const int  block = 256;
        long grid_l = (K + block - 1) / block;
        if (grid_l > 32768) grid_l = 32768;     /* grid-stride cubre el resto */
        const int  grid = (int)grid_l;
        const size_t shmem = (size_t)block * sizeof(unsigned long long);

        cudaEvent_t ev_s, ev_k0, ev_k1, ev_e;
        cudaEventCreate(&ev_s); cudaEventCreate(&ev_k0);
        cudaEventCreate(&ev_k1); cudaEventCreate(&ev_e);

        /* --- Region cronometrada: H2D candidatos + kernel + D2H resultado --- */
        cudaEventRecord(ev_s);
        CUDA_CHECK(cudaMemcpy(dCand, sd->cand, (size_t)K * 3 * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(dBest, 0, sizeof(unsigned long long)));

        cudaEventRecord(ev_k0);
        if (sd->use_double)
            search_kernel_f64<<<grid, block, shmem>>>(
                M, N, (const double *)dA, (const double *)dT, (const double *)dS, (const double *)dF,
                dCand, dPos, sd->n_pos, dNeg, sd->n_neg, sd->atol, sd->rtol, K, dBest);
        else
            search_kernel_f32<<<grid, block, shmem>>>(
                M, N, (const float *)dA, (const float *)dT, (const float *)dS, (const float *)dF,
                dCand, dPos, sd->n_pos, dNeg, sd->n_neg, sd->atol, sd->rtol, K, dBest);
        cudaEventRecord(ev_k1);
        CUDA_CHECK(cudaGetLastError());

        unsigned long long best_key = 0;
        CUDA_CHECK(cudaMemcpy(&best_key, dBest, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        cudaEventRecord(ev_e);
        CUDA_CHECK(cudaEventSynchronize(ev_e));

        float ms_core = 0.0f, ms_search = 0.0f;
        cudaEventElapsedTime(&ms_core, ev_k0, ev_k1);
        cudaEventElapsedTime(&ms_search, ev_s, ev_e);
        cudaEventDestroy(ev_s); cudaEventDestroy(ev_k0);
        cudaEventDestroy(ev_k1); cudaEventDestroy(ev_e);

        long bu, bk;
        sd_unpack_key((long long)best_key, K, &bu, &bk);
        out->best_units = bu;
        out->best_k = bk;
        out->t_core_seconds = (double)ms_core / 1000.0;
        out->t_search_seconds = (double)ms_search / 1000.0;
        out->cuda_block_size = block;
        out->cuda_grid_size = grid;
        out->is_root = 1;
        rc = 0;
    }

cleanup:
    cudaFree(dA); cudaFree(dT); cudaFree(dS); cudaFree(dF);
    cudaFree(dCand); cudaFree(dPos); cudaFree(dNeg); cudaFree(dBest);
    return rc;
}

int main(int argc, char **argv) {
    return runner_main(argc, argv, "cuda", search_cuda);
}
