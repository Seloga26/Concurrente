# Build de las variantes nativas Nivel 2 (C serial + OpenMP + MPI).
# Requiere gcc/make (build-essential) y, para MPI, mpicc (openmpi-bin libopenmpi-dev).
# En WSL Ubuntu:  make
#
#   make                 -> compila scoring_serial, scoring_openmp y scoring_mpi
#   make scoring_serial  -> solo la variante serial
#   make scoring_openmp  -> solo la variante OpenMP
#   make scoring_mpi     -> solo la variante MPI
#   make clean

CC      ?= gcc
MPICC   ?= mpicc
CFLAGS  ?= -O3 -march=native -std=c11 -Wall -Wextra
LDLIBS   = -lm
COMMON   = runner.c npyio.c scoring.c
HEADERS  = runner.h scoring.h npyio.h

.PHONY: all clean
all: scoring_serial scoring_openmp scoring_mpi

scoring_serial: serial.c $(COMMON) $(HEADERS)
	$(CC) $(CFLAGS) -o $@ serial.c $(COMMON) $(LDLIBS)

scoring_openmp: openmp.c $(COMMON) $(HEADERS)
	$(CC) $(CFLAGS) -fopenmp -o $@ openmp.c $(COMMON) $(LDLIBS)

scoring_mpi: mpi.c $(COMMON) $(HEADERS)
	$(MPICC) $(CFLAGS) -o $@ mpi.c $(COMMON) $(LDLIBS)

clean:
	rm -f scoring_serial scoring_openmp scoring_mpi
