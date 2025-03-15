CC = gcc
CFLAGS = -Wall -lm
MPICC = mpicc
CUDACC = nvcc

swep_2d: Shallow_Water_Equation_Parallelized.cu
	$(CUDACC) -o swep_2d Shallow_Water_Equation_Parallelized.cu

swep_2d_oi: Shallow_Water_Equation_Parallelized_One_Iteration.cu
	$(CUDACC) -o swep_2d_oi Shallow_Water_Equation_Parallelized_One_Iteration.cu	

clean:
	rm -f swep_2d
	rm -f swep_2d_oi

