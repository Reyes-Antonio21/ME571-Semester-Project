#!/bin/bash
#SBATCH -J tc_2d
#SBATCH -o tc_2d.%j
#SBATCH -c 1
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH -t 04:30:00
#SBATCH --gres=gpu:1

./tc_2d 200 0.004 10 0.5

status=$?
if [ $status -ne 0 ]; then
    exit $status
fi