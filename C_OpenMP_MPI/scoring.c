/* Implementacion del nucleo de scoring. Ver scoring.h. */
#include "scoring.h"

#include <math.h>
#include <stddef.h>

void sd_score_candidate(const ScoringData *sd, long k, double *scores_out) {
    const int    N  = sd->n_items;
    const int    M  = sd->n_samples;
    const float *cw = sd->cand + (size_t)k * 3;

    if (sd->use_double) {
        const double w0 = (double)cw[0], w1 = (double)cw[1], w2 = (double)cw[2];
        for (int j = 0; j < M; ++j) {
            const double *Arow = sd->Ad + (size_t)j * N;
            double acc = 0.0;
            for (int i = 0; i < N; ++i)
                acc += Arow[i] * (w0 * sd->Td[i] + w1 * sd->Sd[i] + w2 * sd->Fd[i]);
            scores_out[j] = acc;
        }
    } else {
        const float w0 = cw[0], w1 = cw[1], w2 = cw[2];
        for (int j = 0; j < M; ++j) {
            const float *Arow = sd->Af + (size_t)j * N;
            float acc = 0.0f;
            for (int i = 0; i < N; ++i)
                acc += Arow[i] * (w0 * sd->Tf[i] + w1 * sd->Sf[i] + w2 * sd->Ff[i]);
            scores_out[j] = (double)acc;   /* a double para la comparacion AUC */
        }
    }
}

long sd_auc_units_from_scores(const ScoringData *sd, const double *scores) {
    const double atol = sd->atol, rtol = sd->rtol;
    long wins = 0, ties = 0;
    for (int a = 0; a < sd->n_pos; ++a) {
        const double sa = scores[sd->pos[a]];
        const double aa = fabs(sa);
        for (int b = 0; b < sd->n_neg; ++b) {
            const double sb = scores[sd->neg[b]];
            const double ab = fabs(sb);
            const double band = atol + rtol * (aa > ab ? aa : ab);
            const double d = sa - sb;
            if (d > band)        wins++;
            else if (d >= -band) ties++;   /* |d| <= band */
            /* else: loss */
        }
    }
    return 2 * wins + ties;
}

long sd_eval(const ScoringData *sd, long k) {
    double scores[32];   /* n_samples = 10 << 32 */
    sd_score_candidate(sd, k, scores);
    return sd_auc_units_from_scores(sd, scores);
}

void sd_search_range(const ScoringData *sd, long k_start, long k_stop,
                     long *best_units, long *best_k) {
    long bu = -1, bk = -1;
    for (long k = k_start; k < k_stop; ++k) {
        long au = sd_eval(sd, k);
        if (au > bu) {            /* '>' estricto => ante empate conserva el menor k */
            bu = au;
            bk = k;
        }
    }
    *best_units = bu;
    *best_k = bk;
}

long long sd_pack_key(long auc_units, long k, long K) {
    return (long long)auc_units * (long long)K + (long long)(K - 1 - k);
}

void sd_unpack_key(long long key, long K, long *auc_units, long *k) {
    *auc_units = (long)(key / (long long)K);
    *k = (long)((long long)(K - 1) - (key % (long long)K));
}
