#!/bin/bash
#SBATCH -J swe_2d_mpi
#SBATCH -o swe_2d_mpi.%j
#SBATCH -N 1
#SBATCH -n 48
#SBATCH -p bsudfq
#SBATCH -t 08:00:00
#SBATCH --exclusive

module load gcc
module load openmpi
module load slurm
#source ~/.bashrc

# Create or clear the timing log
echo "Process_Count,Time(s)" > timing_results.log

for p in 1 2 4 8 12 16 24 32 48
do 
    mpirun -np $p ./swe_2d_mpi 3600 3600 0.000225 10 10 0.5
done
