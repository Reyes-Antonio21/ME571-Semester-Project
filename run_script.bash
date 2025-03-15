#!/bin/bash
###
###
#SBATCH --time=01:00:00
#SBATCH --tasks=1
#SBATCH --job-name=SWEP_2D
#SBATCH --output=SWEP_2D.o%j
#SBATCH --partition=bsudfq

./SWEP_2D 800 0.001 10 0.5

status=$?
if [$status -ne 0]; then
    exit $status
fi
