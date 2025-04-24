#!/bin/bash
#SBATCH -J swep_2d_ex
#SBATCH -o swep_2d_ex.%j
#SBATCH -c 1
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH -t 00:02:30
#SBATCH --gres=gpu:1
#SBATCH --exclusive

# Fixed simulation constants
nx=200
xlen=10
t_final=0.5
g=9.81
h_max=1.15
CFL=0.5

# Compute dt using CFL condition
dx=$(echo "$xlen / $nx" | bc -l)
c=$(echo "sqrt($g * $h_max)" | bc -l)
dt=$(echo "$CFL * $dx / $c" | bc -l)

# Run CUDA program
./swep_2d_ex $nx $dt $xlen $t_final
