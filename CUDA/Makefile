# Build de la variante CUDA (Nivel 3). Requiere nvcc (CUDA Toolkit) + una GPU NVIDIA.
# El host NO tiene GPU: compilar/ejecutar en Google Colab (runtime GPU). Ver README_colab.md.
#
#   make                 -> compila scoring_cuda
#   make clean
#
# Reutiliza la capa C ya probada de ../C_OpenMP_MPI (runner/npyio/scoring): los .c se compilan
# como C con gcc (-> .o) y nvcc compila el .cu y enlaza los objetos. Las guardas extern "C" de
# los headers hacen compatible el enlace C <-> C++ (.cu).

NVCC    ?= nvcc
CC      ?= gcc
NVFLAGS ?= -O3
CFLAGS  ?= -O3 -std=c11 -Wall -Wextra
# Para la GPU T4 de Colab puede fijarse la arch:  make NVFLAGS="-O3 -arch=sm_75"
CDIR     = ../C_OpenMP_MPI
COBJ     = runner.o npyio.o scoring.o
HEADERS  = scoring_device.cuh $(CDIR)/runner.h $(CDIR)/scoring.h $(CDIR)/npyio.h

.PHONY: all clean
all: scoring_cuda

scoring_cuda: scoring_kernel.cu $(COBJ) $(HEADERS)
	$(NVCC) $(NVFLAGS) -I$(CDIR) -o $@ scoring_kernel.cu $(COBJ) -lm

runner.o: $(CDIR)/runner.c $(CDIR)/runner.h $(CDIR)/scoring.h $(CDIR)/npyio.h
	$(CC) $(CFLAGS) -I$(CDIR) -c $(CDIR)/runner.c -o $@

npyio.o: $(CDIR)/npyio.c $(CDIR)/npyio.h
	$(CC) $(CFLAGS) -I$(CDIR) -c $(CDIR)/npyio.c -o $@

scoring.o: $(CDIR)/scoring.c $(CDIR)/scoring.h
	$(CC) $(CFLAGS) -I$(CDIR) -c $(CDIR)/scoring.c -o $@

clean:
	rm -f scoring_cuda $(COBJ)
