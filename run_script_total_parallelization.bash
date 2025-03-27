#!/bin/bash
#SBATCH -J swep_2d_tp
#SBATCH -o swep_2d_tp.%j
#SBATCH -c 1
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH -t 04:30:00
#SBATCH --gres=gpu:1

./swep_2d_tp 200 0.004 10 0.5

status=$?
if [ $status -ne 0 ]; then
    exit $status
fi