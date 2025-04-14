CC = gcc
CFLAGS = -Wall -lm
MPICC = mpicc
CUDACC = nvcc

swe_2d_ts: Shallow_Water_Equations_Timed_Sections.c
	$(CC) -o swe_2d_ts Shallow_Water_Equations_Timed_Sections.c

swe_2d_tt: Shallow_Water_Equations_Timed_Total.c
	$(CC) -o swe_2d_tt Shallow_Water_Equations_Timed_Total.c

swep_2d_tp: Shallow_Water_Equation_Parallelized.cu
	$(CUDACC) -o swep_2d_tp Shallow_Water_Equation_Parallelized.cu

swep_2d_tk: Shallow_Water_Equation_Timed_Kernel.cu
	$(CUDACC) -o swep_2d_tk Shallow_Water_Equation_Timed_Kernel.cu

swep_2d_tt: Shallow_Water_Equation_Timed_Total.cu
	$(CUDACC) -o swep_2d_tt Shallow_Water_Equation_Timed_Total.cu

swep_2d_an: Shallow_Water_Equation_Animate.cu
	$(CUDACC) -o swep_2d_an Shallow_Water_Equation_Animate.cu

swep_2d_ad: Shallow_Water_Equation_Animate_Drop.cu
	$(CUDACC) -o swep_2d_ad Shallow_Water_Equation_Animate_Drop.cu

swep_2d_ex: Shallow_Water_Equation_Experimental.cu
	$(CUDACC) -o swep_2d_ex Shallow_Water_Equation_Experimental.cu	

clean:
	rm -f swe_2d
	rm -f swep_2d_tp
	rm -f swep_2d_tk
	rm -f swep_2d_an
	rm -f swep_2d_tt
	rm -f swep_2d_ex