#!/bin/bash
#SBATCH -J swe_2d_mpi
#SBATCH -o swe_2d_mpi.%j
#SBATCH -N 1
#SBATCH -n 48
#SBATCH -p bsudfq
#SBATCH -t 00:05:00
#SBATCH --exclusive

module load gcc
module load openmpi
module load slurm
#source ~/.bashrc

mpirun -np 1 ./swe_2d_mpi
mpirun -np 2 ./swe_2d_mpi
mpirun -np 4 ./swe_2d_mpi
mpirun -np 6 ./swe_2d_mpi 
mpirun -np 8 ./swe_2d_mpi
mpirun -np 10 ./swe_2d_mpi 
mpirun -np 12 ./swe_2d_mpi
mpirun -np 14 ./swe_2d_mpi
mpirun -np 16 ./swe_2d_mpi
mpirun -np 18 ./swe_2d_mpi 
mpirun -np 20 ./swe_2d_mpi
mpirun -np 22 ./swe_2d_mpi
mpirun -np 24 ./swe_2d_mpi 
mpirun -np 26 ./swe_2d_mpi
mpirun -np 28 ./swe_2d_mpi 
mpirun -np 30 ./swe_2d_mpi
mpirun -np 32 ./swe_2d_mpi
mpirun -np 34 ./swe_2d_mpi 
mpirun -np 36 ./swe_2d_mpi
mpirun -np 38 ./swe_2d_mpi 
mpirun -np 40 ./swe_2d_mpi
mpirun -np 42 ./swe_2d_mpi
mpirun -np 44 ./swe_2d_mpi 
mpirun -np 46 ./swe_2d_mpi
mpirun -np 48 ./swe_2d_mpi 