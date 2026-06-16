/* Kernel device CUDA compartido por el driver nvcc (scoring_kernel.cu) y por PyCUDA
 * (scoring_pycuda.py, que lee este archivo como string para SourceModule).
 *
 * Contrato identico al de la capa C (C_OpenMP_MPI/scoring.c):
 *   - kernel "fused": Score_j = sum_i A[j,i]*(w0*T_i + w1*S_i + w2*F_i), sin materializar P.
 *   - dos precisiones: scores acumulados en 'real' (float=benchmark / double=reference); el AUC
 *     SIEMPRE se compara en double (igual que float(scores[i]) en Python).
 *   - auc_units = 2*wins + ties; banda = atol + rtol*max(|sa|,|sb|).
 *   - argmax por clave int64 (sin signo aqui): key = au*K + (K-1-k). Maximizar la clave maximiza
 *     au y, ante empate, minimiza k. Las claves son >= 0, por eso se reducen con atomicMax sobre
 *     unsigned long long (el centinela -1 no hace falta en device).
 *
 * Dos entry points NO-template (extern "C") para que PyCUDA los localice por nombre; ambos
 * delegan en el mismo nucleo template<real>.
 */
#ifndef SCORING_DEVICE_CUH
#define SCORING_DEVICE_CUH

#define SD_MAX_SAMPLES 32   /* n_samples (10) << 32 */

/* Evalua el candidato k y devuelve su clave int64 (sin signo). M = n_samples. */
template <typename real>
__device__ unsigned long long sd_eval_key(
        long k, int M, int N,
        const real *A, const real *T, const real *S, const real *F,
        const float *cand,
        const int *pos, int npos, const int *neg, int nneg,
        double atol, double rtol, long K) {

    const float *cw = cand + (size_t)k * 3;
    const real w0 = (real)cw[0], w1 = (real)cw[1], w2 = (real)cw[2];

    double scores[SD_MAX_SAMPLES];
    for (int j = 0; j < M; ++j) {
        const real *Arow = A + (size_t)j * N;
        real acc = (real)0;
        for (int i = 0; i < N; ++i)
            acc += Arow[i] * (w0 * T[i] + w1 * S[i] + w2 * F[i]);
        scores[j] = (double)acc;          /* a double para la comparacion AUC */
    }

    long wins = 0, ties = 0;
    for (int a = 0; a < npos; ++a) {
        const double sa = scores[pos[a]];
        const double aa = fabs(sa);
        for (int b = 0; b < nneg; ++b) {
            const double sb = scores[neg[b]];
            const double ab = fabs(sb);
            const double band = atol + rtol * (aa > ab ? aa : ab);
            const double d = sa - sb;
            if (d > band)        wins++;
            else if (d >= -band) ties++;
        }
    }
    const unsigned long long au = (unsigned long long)(2 * wins + ties);
    return au * (unsigned long long)K + (unsigned long long)(K - 1 - k);
}

/* Nucleo de busqueda: grid-stride sobre [0,K), maximo local, reduccion por bloque en shared
 * memory y atomicMax al acumulador global. blockDim.x debe ser potencia de 2. */
template <typename real>
__device__ void sd_search_device(
        int M, int N,
        const real *A, const real *T, const real *S, const real *F,
        const float *cand,
        const int *pos, int npos, const int *neg, int nneg,
        double atol, double rtol, long K,
        unsigned long long *g_best) {

    extern __shared__ unsigned long long sdata[];
    unsigned long long local = 0ULL;

    for (long k = (long)blockIdx.x * blockDim.x + threadIdx.x;
         k < K;
         k += (long)gridDim.x * blockDim.x) {
        unsigned long long key = sd_eval_key<real>(k, M, N, A, T, S, F, cand,
                                                   pos, npos, neg, nneg, atol, rtol, K);
        if (key > local) local = key;
    }

    sdata[threadIdx.x] = local;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s && sdata[threadIdx.x + s] > sdata[threadIdx.x])
            sdata[threadIdx.x] = sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        atomicMax(g_best, sdata[0]);
}

extern "C" __global__ void search_kernel_f32(
        int M, int N,
        const float *A, const float *T, const float *S, const float *F,
        const float *cand,
        const int *pos, int npos, const int *neg, int nneg,
        double atol, double rtol, long K,
        unsigned long long *g_best) {
    sd_search_device<float>(M, N, A, T, S, F, cand, pos, npos, neg, nneg, atol, rtol, K, g_best);
}

extern "C" __global__ void search_kernel_f64(
        int M, int N,
        const double *A, const double *T, const double *S, const double *F,
        const float *cand,
        const int *pos, int npos, const int *neg, int nneg,
        double atol, double rtol, long K,
        unsigned long long *g_best) {
    sd_search_device<double>(M, N, A, T, S, F, cand, pos, npos, neg, nneg, atol, rtol, K, g_best);
}

#endif /* SCORING_DEVICE_CUH */
