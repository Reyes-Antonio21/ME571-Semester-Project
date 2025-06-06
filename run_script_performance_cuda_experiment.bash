#!/bin/bash
#SBATCH -J swep_2d_ex
#SBATCH -o swep_2d_ex.%j
#SBATCH -c 1
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH -t 00:10:00
#SBATCH --gres=gpu:1
#SBATCH --exclusive

# Fixed simulation constants
nx=8000
xlen=10
t_final=0.5
g=9.81
h_max=1.40
CFL=0.5

# Compute dt using CFL condition
dx=$(echo "$xlen / $nx" | bc -l)
c=$(echo "sqrt($g * $h_max)" | bc -l)
dt=$(echo "$CFL * $dx / $c" | bc -l)

# Run CUDA program
./het 
./swep_2d_ex $nx $dt $xlen $t_final

./swep_2d_ex 200 0.004 10 0.5


