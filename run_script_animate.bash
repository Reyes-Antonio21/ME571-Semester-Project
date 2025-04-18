#!/bin/bash
#SBATCH -J swep_2d_an
#SBATCH -o swep_2d_an.%j
#SBATCH -c 1
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH -t 04:30:00
#SBATCH --gres=gpu:1

./swep_2d_an 200 0.004 10 5

status=$?
if [ $status -ne 0 ]; then
    exit $status
fi