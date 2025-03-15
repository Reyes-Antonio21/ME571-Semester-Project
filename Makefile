CC = gcc
CFLAGS = -Wall -lm
MPICC = mpicc
CUDACC = nvcc

SWEP_2D: Shallow_Water_Equation_Parallelized.cu
	$(CUDACC) -o SWEP_2D Shallow_Water_Equation_Parallelized.cu

clean:
	rm -f SWEP_2D

