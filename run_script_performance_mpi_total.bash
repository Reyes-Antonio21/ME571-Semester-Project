#!/bin/bash
#SBATCH -J swem_2d_tt
#SBATCH -o swem_2d_tt.%j
#SBATCH -N 1
#SBATCH -n 48
#SBATCH -p bsudfq
#SBATCH -t 72:00:00
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
echo "Problem size,Number of processors,Time steps,Iteration,Elapsed time (s)" > $output_file

for nx in 6600 6800 7000 7200 7400 7600 7800 8000
do
    dx=$(echo "$xlen / $nx" | bc -l)
    c=$(echo "sqrt($g * $h_max)" | bc -l)
    dt=$(echo "$CFL * $dx / $c" | bc -l)

    for p in 1 2 4 6 8 10 11 12 13 14 15 16 17 18 19 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48
    do
        # Run and capture all output
        output=$(mpirun -np $p ./swem_2d_tt $nx $nx $dt $xlen $ylen $t_final)

        # Loop through each matching line
        echo "$output" | grep "Problem size" | while read -r line; do
            problem_size=$(echo "$line" | awk -F'[:,]' '{print $2}' | tr -d ' ')
            num_proc=$(echo "$line" | awk -F'[:,]' '{print $4}' | tr -d ' ')
            time_steps=$(echo "$line" | awk -F'[:,]' '{print $6}' | tr -d ' ')
            iteration=$(echo "$line" | awk -F'[:,]' '{print $8}' | tr -d ' ')
            elapsed_time=$(echo "$line" | awk -F'[:,]' '{print $10}' | tr -d ' s')

            echo "$problem_size,$num_proc,$time_steps,$iteration,$elapsed_time" >> $output_file
        done
    done
done
