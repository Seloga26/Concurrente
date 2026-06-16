/* Implementacion del lector minimo de .npy. Ver npyio.h. */
#include "npyio.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const unsigned char NPY_MAGIC[6] = {0x93, 'N', 'U', 'M', 'P', 'Y'};

/* Tamano del item segun el descriptor que aceptamos. 0 = no soportado. */
static size_t descr_itemsize(const char *descr) {
    if (strcmp(descr, "<f4") == 0) return 4;
    if (strcmp(descr, "<i4") == 0) return 4;
    if (strcmp(descr, "<f8") == 0) return 8;
    return 0;
}

/* Busca "key" dentro de la cabecera ASCII y devuelve un puntero justo despues de ':'. */
static const char *find_value(const char *header, const char *key) {
    const char *p = strstr(header, key);
    if (!p) return NULL;
    p = strchr(p, ':');
    return p ? p + 1 : NULL;
}

int npy_load(const char *path, const char *expect_descr, NpyArray *out) {
    out->data = NULL;
    out->ndim = 0;
    out->count = 0;

    size_t itemsize = descr_itemsize(expect_descr);
    if (itemsize == 0) {
        fprintf(stderr, "npy_load: descriptor no soportado '%s'\n", expect_descr);
        return 0;
    }

    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "npy_load: no se pudo abrir '%s'\n", path);
        return 0;
    }

    unsigned char magic[8];
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, NPY_MAGIC, 6) != 0) {
        fprintf(stderr, "npy_load: '%s' no es un archivo .npy valido\n", path);
        fclose(f);
        return 0;
    }
    unsigned char major = magic[6];

    /* Longitud de la cabecera: uint16 LE en v1.0, uint32 LE en v2.0+. */
    size_t header_len;
    if (major >= 2) {
        unsigned char b[4];
        if (fread(b, 1, 4, f) != 4) { fclose(f); return 0; }
        header_len = (size_t)b[0] | ((size_t)b[1] << 8) | ((size_t)b[2] << 16) | ((size_t)b[3] << 24);
    } else {
        unsigned char b[2];
        if (fread(b, 1, 2, f) != 2) { fclose(f); return 0; }
        header_len = (size_t)b[0] | ((size_t)b[1] << 8);
    }

    char *header = (char *)malloc(header_len + 1);
    if (!header) { fclose(f); return 0; }
    if (fread(header, 1, header_len, f) != header_len) {
        fprintf(stderr, "npy_load: cabecera truncada en '%s'\n", path);
        free(header); fclose(f); return 0;
    }
    header[header_len] = '\0';

    /* descr */
    const char *v = find_value(header, "'descr'");
    if (!v) { fprintf(stderr, "npy_load: falta 'descr' en '%s'\n", path); free(header); fclose(f); return 0; }
    const char *q1 = strchr(v, '\'');
    const char *q2 = q1 ? strchr(q1 + 1, '\'') : NULL;
    if (!q1 || !q2) { fprintf(stderr, "npy_load: descr mal formado en '%s'\n", path); free(header); fclose(f); return 0; }
    char descr[16];
    size_t dl = (size_t)(q2 - q1 - 1);
    if (dl >= sizeof(descr)) dl = sizeof(descr) - 1;
    memcpy(descr, q1 + 1, dl);
    descr[dl] = '\0';
    if (strcmp(descr, expect_descr) != 0) {
        fprintf(stderr, "npy_load: '%s' dtype '%s' != esperado '%s'\n", path, descr, expect_descr);
        free(header); fclose(f); return 0;
    }

    /* fortran_order: debe ser False */
    v = find_value(header, "'fortran_order'");
    if (!v || strstr(v, "False") == NULL || (strstr(v, "True") != NULL && strstr(v, "True") < strstr(v, "False"))) {
        fprintf(stderr, "npy_load: '%s' requiere fortran_order=False\n", path);
        free(header); fclose(f); return 0;
    }

    /* shape: tupla de enteros entre parentesis */
    v = find_value(header, "'shape'");
    const char *lp = v ? strchr(v, '(') : NULL;
    const char *rp = lp ? strchr(lp, ')') : NULL;
    if (!lp || !rp) { fprintf(stderr, "npy_load: shape mal formada en '%s'\n", path); free(header); fclose(f); return 0; }

    int ndim = 0;
    size_t count = 1;
    const char *p = lp + 1;
    while (p < rp) {
        while (p < rp && (*p == ' ' || *p == ',')) p++;
        if (p >= rp) break;
        if (*p < '0' || *p > '9') { p++; continue; }
        size_t dim = 0;
        while (p < rp && *p >= '0' && *p <= '9') { dim = dim * 10 + (size_t)(*p - '0'); p++; }
        if (ndim >= NPY_MAX_DIMS) {
            fprintf(stderr, "npy_load: '%s' tiene demasiadas dimensiones\n", path);
            free(header); fclose(f); return 0;
        }
        out->shape[ndim++] = dim;
        count *= dim;
    }
    free(header);

    if (ndim == 0) { fprintf(stderr, "npy_load: '%s' es escalar (no soportado)\n", path); fclose(f); return 0; }

    void *data = malloc(count * itemsize);
    if (!data) { fprintf(stderr, "npy_load: sin memoria para '%s'\n", path); fclose(f); return 0; }
    if (fread(data, itemsize, count, f) != count) {
        fprintf(stderr, "npy_load: datos truncados en '%s'\n", path);
        free(data); fclose(f); return 0;
    }
    fclose(f);

    out->data = data;
    out->ndim = ndim;
    out->count = count;
    return 1;
}
