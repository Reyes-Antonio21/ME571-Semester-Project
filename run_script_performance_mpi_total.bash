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
output_file="Shallow_Water_Equations_MPI_Total_Runtime_Performance.csv"

# Header for CSV (only once)
echo "Problem size,Number of processors,dt,Iteration,Time steps,Elapsed time (s)" > $output_file

for nx in 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500 1600 1700 1800 1900 2000 \
          2100 2200 2300 2400 2500 2600 2700 2800 2900 3000 3200 3400 3600 3800 4000 \
          4200 4400 4600 4800 5000 5200 5400 5600 5800 6000 6200 6400 
do
    dx=$(echo "$xlen / $nx" | bc -l)
    c=$(echo "sqrt($g * $h_max)" | bc -l)
    dt=$(echo "$CFL * $dx / $c" | bc -l)

    for p in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48
    do
        # Run and capture all output
        output=$(mpirun -np $p ./swem_2d_tt $nx $nx $dt $xlen $ylen $t_final)

        # Loop through each matching line
        echo "$output" | grep "Number of Processors" | while read -r line; do
            num_proc=$(echo "$line" | awk -F'[:,]' '{print $2}' | tr -d ' ')
            problem_size=$(echo "$line" | awk -F'[:,]' '{print $4}' | tr -d ' ')
            iterations=$(echo "$line" | awk -F'[:,]' '{print $6}' | tr -d ' ')
            time_steps=$(echo "$line" | awk -F'[:,]' '{print $8}' | tr -d ' ')
            elapsed_time=$(echo "$line" | awk -F'[:,]' '{print $10}' | tr -d ' s')

            echo "$problem_size,$num_proc,$dt,$time_steps,$iterations,$elapsed_time" >> $output_file
        done
    done
done
