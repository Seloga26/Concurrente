"""Paquete de implementaciones Python (Nivel 1).

MÍNIMO a propósito: **no** importa numpy, **no** importa `python.sequential` ni módulos
de `common`. Con `python -m python.sequential`, este `__init__` se ejecuta primero; debe
permanecer libre de numpy para que `python.sequential` pueda fijar las variables de hilos
BLAS **antes** de cualquier importación de numpy.
"""
