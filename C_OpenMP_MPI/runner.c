/* Implementacion de la orquestacion compartida. Ver runner.h. */
#define _POSIX_C_SOURCE 199309L

#include "runner.h"
#include "npyio.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N_HEALTHY 5
#define N_SICK    5
#define N_SAMPLES (N_HEALTHY + N_SICK)
#define KERNEL_VARIANT "fused"
#define VALID_ALGORITHM "literal"
#define VALID_THETA_POLICY "class_mean_midpoint"
#define MAIN_MAX_AUC_UNITS 50
#define INT64_MAX_LL 9223372036854775807LL

double runner_now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ------------------------------- Parseo CLI ------------------------------- */

static int need_value(int i, int argc, const char *flag) {
    if (i + 1 >= argc) {
        fprintf(stderr, "error: el flag %s requiere un valor\n", flag);
        return 0;
    }
    return 1;
}

static int parse_args(int argc, char **argv, RunnerArgs *a) {
    memset(a, 0, sizeof(*a));
    a->threads = 0;
    a->N = a->K = -1;
    a->tie_atol = a->tie_rtol = -1.0;
    a->consistency_threshold = -1.0;

    for (int i = 1; i < argc; ++i) {
        const char *f = argv[i];
        #define STRARG(name, field) \
            else if (strcmp(f, name) == 0) { if (!need_value(i, argc, name)) return 0; a->field = argv[++i]; }
        #define LONGARG(name, field) \
            else if (strcmp(f, name) == 0) { if (!need_value(i, argc, name)) return 0; a->field = strtol(argv[++i], NULL, 10); }
        #define DBLARG(name, field) \
            else if (strcmp(f, name) == 0) { if (!need_value(i, argc, name)) return 0; a->field = strtod(argv[++i], NULL); }

        if (0) { }
        LONGARG("--N", N)
        LONGARG("--K", K)
        STRARG("--mode", mode)
        STRARG("--accum", accum)
        STRARG("--algorithm", algorithm)
        DBLARG("--tie-atol", tie_atol)
        DBLARG("--tie-rtol", tie_rtol)
        STRARG("--matrix-a", matrix_a)
        STRARG("--profile-t", profile_t)
        STRARG("--profile-s", profile_s)
        STRARG("--profile-f", profile_f)
        STRARG("--labels", labels)
        STRARG("--candidates", candidates)
        STRARG("--theta-policy", theta_policy)
        DBLARG("--consistency-threshold", consistency_threshold)
        LONGARG("--threads", threads)
        STRARG("--output-json", output_json)
        else {
            fprintf(stderr, "error: flag desconocido %s\n", f);
            return 0;
        }
        #undef STRARG
        #undef LONGARG
        #undef DBLARG
    }

    const char *missing = NULL;
    if (a->N < 0) missing = "--N";
    else if (a->K < 0) missing = "--K";
    else if (!a->mode) missing = "--mode";
    else if (!a->accum) missing = "--accum";
    else if (!a->algorithm) missing = "--algorithm";
    else if (a->tie_atol < 0) missing = "--tie-atol";
    else if (a->tie_rtol < 0) missing = "--tie-rtol";
    else if (!a->matrix_a) missing = "--matrix-a";
    else if (!a->profile_t) missing = "--profile-t";
    else if (!a->profile_s) missing = "--profile-s";
    else if (!a->profile_f) missing = "--profile-f";
    else if (!a->labels) missing = "--labels";
    else if (!a->candidates) missing = "--candidates";
    else if (!a->theta_policy) missing = "--theta-policy";
    else if (a->consistency_threshold < 0) missing = "--consistency-threshold";
    if (missing) {
        fprintf(stderr, "error: falta el argumento obligatorio %s\n", missing);
        return 0;
    }
    return 1;
}

static int validate_args(const RunnerArgs *a) {
    int ref = strcmp(a->mode, "reference") == 0;
    int bench = strcmp(a->mode, "benchmark") == 0;
    if (!ref && !bench) {
        fprintf(stderr, "error: mode debe ser reference|benchmark, no '%s'\n", a->mode); return 0;
    }
    const char *expected = ref ? "float64" : "float32";
    if (strcmp(a->accum, expected) != 0) {
        fprintf(stderr, "error: accum='%s' incompatible con mode='%s' (esperado %s)\n",
                a->accum, a->mode, expected);
        return 0;
    }
    if (strcmp(a->algorithm, VALID_ALGORITHM) != 0) {
        fprintf(stderr, "error: algorithm debe ser '%s', no '%s'\n", VALID_ALGORITHM, a->algorithm); return 0;
    }
    if (strcmp(a->theta_policy, VALID_THETA_POLICY) != 0) {
        fprintf(stderr, "error: theta_policy debe ser '%s', no '%s'\n", VALID_THETA_POLICY, a->theta_policy); return 0;
    }
    if (a->K < 1) { fprintf(stderr, "error: K debe ser >= 1\n"); return 0; }
    if (a->N < 1) { fprintf(stderr, "error: N debe ser >= 1\n"); return 0; }
    if (a->K > INT64_MAX_LL / (MAIN_MAX_AUC_UNITS + 1)) {
        fprintf(stderr, "error: K=%ld excede el limite de overflow de la clave int64\n", a->K); return 0;
    }
    return 1;
}

/* --------------------------- Carga + validacion --------------------------- */

static int validate_inputs(const RunnerArgs *a,
                           const NpyArray *A, const NpyArray *T, const NpyArray *S,
                           const NpyArray *F, const NpyArray *Y, const NpyArray *C) {
    const long N = a->N, K = a->K;

    if (!(A->ndim == 2 && A->shape[0] == N_SAMPLES && (long)A->shape[1] == N)) {
        fprintf(stderr, "error: A debe ser (%d,%ld)\n", N_SAMPLES, N); return 0;
    }
    const NpyArray *prof[3] = {T, S, F};
    const char *nm[3] = {"T", "S", "F"};
    for (int p = 0; p < 3; ++p)
        if (!(prof[p]->ndim == 1 && (long)prof[p]->shape[0] == N)) {
            fprintf(stderr, "error: %s debe ser (%ld,)\n", nm[p], N); return 0;
        }
    if (!(Y->ndim == 1 && Y->shape[0] == N_SAMPLES)) {
        fprintf(stderr, "error: labels debe ser (%d,)\n", N_SAMPLES); return 0;
    }
    if (!(C->ndim == 2 && (long)C->shape[0] == K && C->shape[1] == 3)) {
        fprintf(stderr, "error: candidates_W debe ser (%ld,3)\n", K); return 0;
    }

    /* y == [0]*5 + [1]*5 */
    const int32_t *y = (const int32_t *)Y->data;
    for (int j = 0; j < N_SAMPLES; ++j) {
        int expected = (j < N_HEALTHY) ? 0 : 1;
        if (y[j] != expected) {
            fprintf(stderr, "error: y debe ser %d ceros seguidos de %d unos\n", N_HEALTHY, N_SICK); return 0;
        }
    }

    /* A: no negativo y filas suman ~1 */
    const float *Af = (const float *)A->data;
    for (int j = 0; j < N_SAMPLES; ++j) {
        double s = 0.0;
        for (long i = 0; i < N; ++i) {
            float v = Af[(size_t)j * N + i];
            if (v < 0.0f) { fprintf(stderr, "error: A tiene entradas negativas\n"); return 0; }
            s += v;
        }
        if (fabs(s - 1.0) > 1e-5) { fprintf(stderr, "error: las filas de A no suman 1\n"); return 0; }
    }

    /* candidates: no negativo y filas suman ~1 (atol 1e-4, como el loader) */
    const float *Cf = (const float *)C->data;
    for (long k = 0; k < K; ++k) {
        double s = 0.0;
        for (int t = 0; t < 3; ++t) {
            float v = Cf[(size_t)k * 3 + t];
            if (v < 0.0f) { fprintf(stderr, "error: candidates_W tiene entradas negativas\n"); return 0; }
            s += v;
        }
        if (fabs(s - 1.0) > 1e-4) { fprintf(stderr, "error: los candidatos no suman 1 (k=%ld)\n", k); return 0; }
    }

    /* T,S,F en [0,1] con tolerancia */
    for (int p = 0; p < 3; ++p) {
        const float *v = (const float *)prof[p]->data;
        for (long i = 0; i < N; ++i)
            if (v[i] < -1e-4f || v[i] > 1.0f + 1e-4f) {
                fprintf(stderr, "error: %s fuera de [0,1]\n", nm[p]); return 0;
            }
    }
    return 1;
}

/* Castea un arreglo float32 crudo a un buffer double recien asignado. */
static double *to_double(const float *src, size_t count) {
    double *dst = (double *)malloc(count * sizeof(double));
    if (!dst) return NULL;
    for (size_t i = 0; i < count; ++i) dst[i] = (double)src[i];
    return dst;
}

/* ------------------------------- Salida JSON ------------------------------ */

static void print_double_array(FILE *out, const double *v, int n) {
    fputc('[', out);
    for (int i = 0; i < n; ++i) {
        if (i) fputs(", ", out);
        fprintf(out, "%.17g", v[i]);
    }
    fputc(']', out);
}

static void build_json(FILE *out, const char *impl, const RunnerArgs *a, const ScoringData *sd,
                       const SearchOutcome *res, const double *best_w, const double *scores,
                       long auc_units, long denom, double auc, double theta, double cons) {
    fputc('{', out);
    fprintf(out, "\"implementation\": \"%s\"", impl);
    fprintf(out, ", \"mode\": \"%s\"", a->mode);
    fprintf(out, ", \"algorithm\": \"%s\"", a->algorithm);
    fprintf(out, ", \"kernel_variant\": \"%s\"", KERNEL_VARIANT);
    fprintf(out, ", \"accum_dtype\": \"%s\"", a->accum);
    fprintf(out, ", \"n_items\": %ld", a->N);
    fprintf(out, ", \"n_candidates\": %ld", a->K);
    fprintf(out, ", \"best_k\": %ld", res->best_k);
    fputs(", \"best_w\": ", out); print_double_array(out, best_w, 3);
    fprintf(out, ", \"auc_units\": %ld", auc_units);
    fprintf(out, ", \"auc_denominator\": %ld", denom);
    fprintf(out, ", \"auc\": %.17g", auc);
    fputs(", \"scores\": ", out); print_double_array(out, scores, sd->n_samples);
    fprintf(out, ", \"theta\": %.17g", theta);
    fprintf(out, ", \"consistency\": %.17g", cons);
    fprintf(out, ", \"consistency_threshold\": %.17g", a->consistency_threshold);
    fprintf(out, ", \"consistency_pass\": %s", (cons >= a->consistency_threshold) ? "true" : "false");
    fprintf(out, ", \"tie_atol\": %.17g", a->tie_atol);
    fprintf(out, ", \"tie_rtol\": %.17g", a->tie_rtol);
    fprintf(out, ", \"theta_policy\": \"%s\"", a->theta_policy);
    fprintf(out, ", \"t_core_seconds\": %.17g", res->t_core_seconds);
    fprintf(out, ", \"t_search_seconds\": %.17g", res->t_search_seconds);
    if (res->n_threads > 0)
        fprintf(out, ", \"n_threads\": %d", res->n_threads);
    fputc('}', out);
    fputc('\n', out);
}

/* ------------------------------- runner_main ------------------------------ */

int runner_main(int argc, char **argv, const char *impl_name, search_fn search) {
    RunnerArgs a;
    if (!parse_args(argc, argv, &a)) return 1;
    if (!validate_args(&a)) return 1;

    NpyArray A = {0}, T = {0}, S = {0}, F = {0}, Y = {0}, C = {0};
    int ok = npy_load(a.matrix_a,   "<f4", &A) &&
             npy_load(a.profile_t,  "<f4", &T) &&
             npy_load(a.profile_s,  "<f4", &S) &&
             npy_load(a.profile_f,  "<f4", &F) &&
             npy_load(a.labels,     "<i4", &Y) &&
             npy_load(a.candidates, "<f4", &C);
    int rc = 1;
    double *Ad = NULL, *Td = NULL, *Sd = NULL, *Fd = NULL;
    if (!ok) goto cleanup;
    if (!validate_inputs(&a, &A, &T, &S, &F, &Y, &C)) goto cleanup;

    const long N = a.N;
    const int use_double = (strcmp(a.mode, "reference") == 0);

    /* Indices por clase (y = [0]*5 + [1]*5). */
    int pos_idx[N_SICK], neg_idx[N_HEALTHY];
    int npos = 0, nneg = 0;
    const int32_t *y = (const int32_t *)Y.data;
    for (int j = 0; j < N_SAMPLES; ++j)
        if (y[j] == 1) pos_idx[npos++] = j; else neg_idx[nneg++] = j;

    ScoringData sd = {0};
    sd.n_items = (int)N;
    sd.n_cand = a.K;
    sd.n_samples = N_SAMPLES;
    sd.use_double = use_double;
    sd.cand = (const float *)C.data;
    sd.pos = pos_idx; sd.neg = neg_idx;
    sd.n_pos = npos; sd.n_neg = nneg;
    sd.atol = a.tie_atol; sd.rtol = a.tie_rtol;

    if (use_double) {
        Ad = to_double((const float *)A.data, (size_t)N_SAMPLES * N);
        Td = to_double((const float *)T.data, (size_t)N);
        Sd = to_double((const float *)S.data, (size_t)N);
        Fd = to_double((const float *)F.data, (size_t)N);
        if (!Ad || !Td || !Sd || !Fd) { fprintf(stderr, "error: sin memoria\n"); goto cleanup; }
        sd.Ad = Ad; sd.Td = Td; sd.Sd = Sd; sd.Fd = Fd;
    } else {
        sd.Af = (const float *)A.data;
        sd.Tf = (const float *)T.data;
        sd.Sf = (const float *)S.data;
        sd.Ff = (const float *)F.data;
    }

    /* Warmup fuera del cronometro. */
    (void)sd_eval(&sd, 0);

    /* Busqueda cronometrada (estrategia inyectada). */
    SearchOutcome res = {0};
    res.best_units = -1; res.best_k = -1;
    if (search(&sd, &a, &res) != 0) { fprintf(stderr, "error: fallo en la busqueda\n"); goto cleanup; }

    /* Recompute del ganador fuera del cronometro. */
    double scores[N_SAMPLES];
    sd_score_candidate(&sd, res.best_k, scores);

    long wins = 0, ties = 0;
    for (int ai = 0; ai < npos; ++ai) {
        double sa = scores[pos_idx[ai]], aa = fabs(sa);
        for (int bi = 0; bi < nneg; ++bi) {
            double sb = scores[neg_idx[bi]], ab = fabs(sb);
            double band = a.tie_atol + a.tie_rtol * (aa > ab ? aa : ab);
            double d = sa - sb;
            if (d > band) wins++; else if (d >= -band) ties++;
        }
    }
    long auc_units = 2 * wins + ties;
    long denom = 2L * npos * nneg;
    double auc = (double)auc_units / (double)denom;

    /* theta = punto medio de medias por clase. La media se acumula en el dtype de trabajo
     * para reproducir numpy: sp.mean() sobre un array float32 acumula en float32 (benchmark);
     * en reference es float64. Asi el theta coincide bit a bit con python.sequential. */
    double theta;
    if (use_double) {
        double sp = 0.0, sn = 0.0;
        for (int i = 0; i < npos; ++i) sp += scores[pos_idx[i]];
        for (int i = 0; i < nneg; ++i) sn += scores[neg_idx[i]];
        theta = 0.5 * (sp / npos + sn / nneg);
    } else {
        float sp = 0.0f, sn = 0.0f;
        for (int i = 0; i < npos; ++i) sp += (float)scores[pos_idx[i]];
        for (int i = 0; i < nneg; ++i) sn += (float)scores[neg_idx[i]];
        theta = 0.5 * ((double)(sp / (float)npos) + (double)(sn / (float)nneg));
    }

    /* consistency = balanced accuracy. Enfermo: score>theta; sano: score<=theta. */
    int tp = 0, tn = 0;
    for (int i = 0; i < npos; ++i) if (scores[pos_idx[i]] > theta) tp++;
    for (int i = 0; i < nneg; ++i) if (scores[neg_idx[i]] <= theta) tn++;
    double cons = 0.5 * ((double)tp / npos + (double)tn / nneg);

    double best_w[3];
    const float *bw = (const float *)C.data + (size_t)res.best_k * 3;
    best_w[0] = (double)bw[0]; best_w[1] = (double)bw[1]; best_w[2] = (double)bw[2];

    build_json(stdout, impl_name, &a, &sd, &res, best_w, scores, auc_units, denom, auc, theta, cons);
    if (a.output_json) {
        FILE *jf = fopen(a.output_json, "w");
        if (jf) { build_json(jf, impl_name, &a, &sd, &res, best_w, scores, auc_units, denom, auc, theta, cons); fclose(jf); }
    }
    rc = 0;

cleanup:
    free(Ad); free(Td); free(Sd); free(Fd);
    free(A.data); free(T.data); free(S.data); free(F.data); free(Y.data); free(C.data);
    return rc;
}
