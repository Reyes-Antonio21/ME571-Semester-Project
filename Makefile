CC = gcc
CFLAGS = -Wall -lm
MPICC = mpicc
CUDACC = nvcc

swe_2d_ts: Shallow_Water_Equations_Timed_Sections.c
	$(CC) -O3 -march=native -funroll-loops -ffast-math -o swe_2d_ts Shallow_Water_Equations_Timed_Sections.c -lm

swe_2d_tt: Shallow_Water_Equations_Timed_Total.c
	$(CC) -O3 -march=native -funroll-loops -ffast-math -o swe_2d_tt Shallow_Water_Equations_Timed_Total.c -lm

swem_2d_ts: Shallow_Water_Equations_MPI_Timed_Section.c
	$(MPICC) -O3 -march=native -funroll-loops -ffast-math -o swem_2d_ts Shallow_Water_Equations_MPI_Timed_Section.c -lm

swem_2d_tt: Shallow_Water_Equations_MPI_Timed_Total.c
	$(MPICC) -O3 -march=native -funroll-loops -ffast-math -o swem_2d_tt Shallow_Water_Equations_MPI_Timed_Total.c -lm

swem_2d_ex: Shallow_Water_Equations_MPI_Experimental.c
	$(MPICC) -O3 -march=native -funroll-loops -ffast-math -o swem_2d_ex Shallow_Water_Equations_MPI_Experimental.c -lm

swep_2d_tk: Shallow_Water_Equations_Timed_Kernel.cu
	$(CUDACC) -O3 --use_fast_math -arch=sm_70 -o swep_2d_tk Shallow_Water_Equations_Timed_Kernel.cu

swep_2d_tt: Shallow_Water_Equations_Timed_Total.cu
	$(CUDACC) -O3 --use_fast_math -arch=sm_70 -o swep_2d_tt Shallow_Water_Equations_Timed_Total.cu

swep_2d_st: Shallow_Water_Equations_Shared_Timed_Total.cu
	$(CUDACC) -O3 --use_fast_math -arch=sm_70 -o swep_2d_st Shallow_Water_Equations_Shared_Timed_Total.cu

swep_2d_ex: Shallow_Water_Equations_Experimental.cu
	$(CUDACC) -O3 --use_fast_math -arch=sm_70 -Xptxas -v -o swep_2d_ex Shallow_Water_Equations_Experimental.cu

swep_2d_an: Shallow_Water_Equations_Animate.cu
	$(CUDACC) -O3 --use_fast_math -arch=sm_70 -o swep_2d_an Shallow_Water_Equations_Animate.cu

swep_2d_ad: Shallow_Water_Equations_Animate_Drop.cu
	$(CUDACC) -O3 --use_fast_math -arch=sm_70 -o swep_2d_ad Shallow_Water_Equations_Animate_Drop.cu
	
het: Halo_Exchange_Test.cu
	$(CUDACC) -O3 --use_fast_math -arch=sm_70 -o het Halo_Exchange_Test.cu

clean:
	rm -f swe_2d_ts swe_2d_tt swem_2d_ts swem_2d_tt swep_2d_tk swep_2d_tt swep_2d_ex swep_2d_an swep_2d_ad
	