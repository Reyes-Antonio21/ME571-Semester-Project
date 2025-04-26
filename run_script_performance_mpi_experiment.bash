#!/bin/bash
#SBATCH -J swem_2d_tt
#SBATCH -o swem_2d_tt.%j
#SBATCH -N 1
#SBATCH -n 48
#SBATCH -p bsudfq
#SBATCH -t 168:00:00
#SBATCH --exclusive

# Fixed values
xlen=10
ylen=10
t_final=0.5
g=9.81
h_max=1.4
CFL=0.5


for nx in 200 300 400 500 600 700 800 900 1000 
do
    dx=$(echo "$xlen / $nx" | bc -l)
    c=$(echo "sqrt($g * $h_max)" | bc -l)
    dt=$(echo "$CFL * $dx / $c" | bc -l)

    for p in 1 2 4 6 8 10 11 12 13 14 15 16 17 18 19 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48
    do
        # Run and capture all output
        mpirun -np $p ./swem_2d_tt $nx $nx $dt $xlen $ylen $t_final)

    done
done
