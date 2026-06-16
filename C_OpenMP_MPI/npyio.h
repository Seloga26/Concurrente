/* Lector minimo de archivos .npy (NumPy v1.0/2.0, little-endian, C-contiguo).
 *
 * Soporta solo lo que el scoring necesita: arreglos C-contiguos de un dtype fijo
 * ('<f4' = float32, '<i4' = int32), fortran_order = False. No es un lector general
 * de NumPy; valida el descriptor y la forma esperados y aborta con mensaje a stderr
 * ante cualquier discrepancia.
 */
#ifndef NPYIO_H
#define NPYIO_H

#include <stddef.h>

#define NPY_MAX_DIMS 4

/* Resultado de una carga: datos crudos en el orden original (C-contiguo) y la forma. */
typedef struct {
    void  *data;                 /* malloc()-eado; el llamante hace free() */
    int    ndim;
    size_t shape[NPY_MAX_DIMS];
    size_t count;                /* numero total de elementos */
} NpyArray;

/* Carga 'path' exigiendo el descriptor 'expect_descr' (p.ej. "<f4" o "<i4").
 * Devuelve 1 en exito, 0 en error (mensaje a stderr). 'out' queda con data=NULL en error. */
int npy_load(const char *path, const char *expect_descr, NpyArray *out);

#endif /* NPYIO_H */
