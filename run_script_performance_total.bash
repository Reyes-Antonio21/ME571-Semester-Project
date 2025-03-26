#!/bin/bash
#SBATCH -J swep_2d_tt
#SBATCH -o swep_2d_tt.%j
#SBATCH -c 1
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH -t 04:30:00
#SBATCH --gres=gpu:1

./swep_2d_tt 200 0.004 10 0.5
./swep_2d_tt 300 0.003 10 0.5
./swep_2d_tt 400 0.002 10 0.5
./swep_2d_tt 500 0.0016 10 0.5
./swep_2d_tt 600 0.0015 10 0.5
./swep_2d_tt 700 0.00125 10 0.5
./swep_2d_tt 800 0.001 10 0.5
./swep_2d_tt 900 0.0009 10 0.5
./swep_2d_tt 1000 0.0008 10 0.5
./swep_2d_tt 1100 0.000775 10 0.5
./swep_2d_tt 1200 0.00075 10 0.5
./swep_2d_tt 1300 0.00065 10 0.5
./swep_2d_tt 1400 0.000625 10 0.5
./swep_2d_tt 1500 0.0005625 10 0.5
./swep_2d_tt 1600 0.0005 10 0.5
./swep_2d_tt 1700 0.000475 10 0.5
./swep_2d_tt 1800 0.00045 10 0.5
./swep_2d_tt 1900 0.000425 10 0.5
./swep_2d_tt 2000 0.0004 10 0.5
./swep_2d_tt 2200 0.0003875 10 0.5
./swep_2d_tt 2400 0.000375 10 0.5
./swep_2d_tt 2500 0.00035 10 0.5
./swep_2d_tt 2600 0.000325 10 0.5
./swep_2d_tt 2800 0.0003125 10 0.5
./swep_2d_tt 3000 0.0003 10 0.5
./swep_2d_tt 3200 0.00025 10 0.5
./swep_2d_tt 3400 0.0002375 10 0.5
./swep_2d_tt 3600 0.000225 10 0.5
./swep_2d_tt 3800 0.0002125 10 0.5
./swep_2d_tt 4000 0.0002 10 0.5
./swep_2d_tt 4800 0.0001875 10 0.5
./swep_2d_tt 5000 0.000175 10 0.5
./swep_2d_tt 5200 0.0001625 10 0.5
./swep_2d_tt 6000 0.00015 10 0.5
./swep_2d_tt 6400 0.000125 10 0.5
./swep_2d_tt 7200 0.0001125 10 0.5
./swep_2d_tt 8000 0.0001 10 0.5

status=$?
if [ $status -ne 0 ]; then
    exit $status
fi