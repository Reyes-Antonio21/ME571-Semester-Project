#!/bin/bash
#SBATCH -J swe_2d_mpi
#SBATCH -o swe_2d_mpi.%j
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
h_max=1.0
CFL=0.5
output_file="Shallow_Water_Equations_MPI_Total_Runtime_Performance.csv"

# Write CSV header
echo "Problem size,Number of processors,Elapsed time (s)" > $output_file

# Loop over problem sizes
for size in 200 300 400 
do
    dx=$(echo "$xlen / $size" | bc -l)
    c=$(echo "sqrt($g * $h_max)" | bc -l)
    dt=$(echo "$CFL * $dx / $c" | bc -l)

    for p in 1 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48
    do
        start=$(date +%s.%N)
        mpirun -np $p ./swe_2d_mpi $size $size $dt $xlen $ylen $t_final > /dev/null
        end=$(date +%s.%N)

        elapsed=$(echo "$end - $start" | bc)
        echo "$size,$p,$elapsed" >> $output_file
    done
done
