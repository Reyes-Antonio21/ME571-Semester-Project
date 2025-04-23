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
h_max=1.4
CFL=0.5
output_file="Shallow_Water_Equations_MPI_Total_Runtime_Performance.csv"

# Header for CSV (only once)
echo "problem_size,num_processors,iterations,time_steps,elapsed_time_sec" > $output_file

for size in 200 300 400
do
    dx=$(echo "$xlen / $size" | bc -l)
    c=$(echo "sqrt($g * $h_max)" | bc -l)
    dt=$(echo "$CFL * $dx / $c" | bc -l)

    for p in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48
    do
        echo "Running with $p processes"

        # Run and capture all output
        output=$(mpirun -np $p ./swe_2d_mpi $size $size $dt $xlen $ylen $t_final)

        # Loop through each matching line
        echo "$output" | grep "Number of Processors" | while read -r line; do
            num_proc=$(echo "$line" | awk -F'[:,]' '{print $2}' | tr -d ' ')
            problem_size=$(echo "$line" | awk -F'[:,]' '{print $4}' | tr -d ' ')
            iterations=$(echo "$line" | awk -F'[:,]' '{print $6}' | tr -d ' ')
            time_steps=$(echo "$line" | awk -F'[:,]' '{print $8}' | tr -d ' ')
            elapsed_time=$(echo "$line" | awk -F'[:,]' '{print $10}' | tr -d ' s')

            echo "$problem_size,$num_proc,$iterations,$time_steps,$elapsed_time" >> $output_file
        done
    done
done
