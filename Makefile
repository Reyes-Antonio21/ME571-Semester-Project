CC = gcc
CFLAGS = -Wall -lm
MPICC = mpicc
CUDACC = nvcc

swep_2d: Shallow_Water_Equation_Parallelized.cu
	$(CUDACC) -o swep_2d Shallow_Water_Equation_Parallelized.cu

swep_2d_tk: Shallow_Water_Equation_Timed_Kernel.cu
	$(CUDACC) -o swep_2d_tk Shallow_Water_Equation_Timed_Kernel.cu

swep_2d_tt: Shallow_Water_Equation_Timed_Total.cu
	$(CUDACC) -o swep_2d_tt Shallow_Water_Equation_Timed_Total.cu

tc_2d: Teacher_Code.cu
	$(CUDACC) -o tc_2d Teacher_Code.cu	

clean:
	rm -f swep_2d
	rm -f swep_2d_tk
	rm -f tc_2d
	rm -f swep_2d_tt

