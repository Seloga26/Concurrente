/* Nucleo de scoring nativo: kernel "fused" + AUC entero + clave int64.
 *
 * Contrato (igual que la capa Python common/):
 *   - algorithm = "literal", kernel_variant = "fused": Score_j = sum_i A[j,i]*(w0*T_i+w1*S_i+w2*F_i),
 *     SIN materializar P (a diferencia del numpy "materialized_numpy"). Resultado identico,
 *     distinta realizacion.
 *   - Dos modos de precision: reference => acumulacion float64; benchmark => acumulacion float32.
 *     Los pesos del candidato (float32 en disco) se castean al dtype de trabajo. Los 10 scores
 *     resultantes SIEMPRE se comparan en double para el AUC (igual que float(scores[i]) en Python).
 *   - auc_units = 2*wins + ties; banda de empate = atol + rtol*max(|sa|,|sb|).
 *   - Argmax: mayor auc_units; empate => menor indice global k. Clave int64 = au*K + (K-1-k).
 */
#ifndef SCORING_H
#define SCORING_H

#ifdef __cplusplus
extern "C" {
#endif

/* Datos de trabajo del scoring. Solo se rellena el juego de punteros del dtype activo;
 * el otro queda en NULL (use_double selecciona cual). */
typedef struct {
    int   n_items;        /* N */
    long  n_cand;         /* K */
    int   n_samples;      /* 10 */
    int   use_double;     /* 1 = reference (float64), 0 = benchmark (float32) */

    const float  *Af, *Tf, *Sf, *Ff;   /* arreglos de trabajo float32 (benchmark) */
    const double *Ad, *Td, *Sd, *Fd;   /* arreglos de trabajo float64 (reference) */
    const float  *cand;                /* candidatos float32 crudos: n_cand x 3 */

    const int *pos, *neg;              /* indices de muestras por clase */
    int n_pos, n_neg;

    double atol, rtol;                 /* banda de empate del AUC */
} ScoringData;

/* Scores de las n_samples muestras para el candidato k, devueltos en double. */
void sd_score_candidate(const ScoringData *sd, long k, double *scores_out);

/* auc_units (entero) a partir de un vector de scores en double. */
long sd_auc_units_from_scores(const ScoringData *sd, const double *scores);

/* Evalua el candidato k y devuelve su auc_units (score fusionado + AUC, sin asignaciones). */
long sd_eval(const ScoringData *sd, long k);

/* Busqueda serial sobre el rango [k_start, k_stop): mayor auc_units, empate => menor k. */
void sd_search_range(const ScoringData *sd, long k_start, long k_stop,
                     long *best_units, long *best_k);

/* Clave int64 para reduccion paralela: key = au*K + (K-1-k). Maximizarla maximiza au y,
 * ante empate, minimiza k. */
long long sd_pack_key(long auc_units, long k, long K);
void      sd_unpack_key(long long key, long K, long *auc_units, long *k);

#ifdef __cplusplus
}
#endif

#endif /* SCORING_H */
