#!/bin/bash
#SBATCH -J swe_2d_mpi
#SBATCH -o swe_2d_mpi.%j
#SBATCH -N 1
#SBATCH -n 48
#SBATCH -p bsudfq
#SBATCH -t 00:05:00
#SBATCH --exclusive

mpirun -np 36 ./swe_2d_mpi 400 400 0.002 10 10 0.5