#!/bin/bash
#SBATCH -J swem_2d_ts
#SBATCH -o swem_2d_ts.%j
#SBATCH -N 1
#SBATCH -n 48
#SBATCH -p bsudfq
#SBATCH -t 110:00:00
#SBATCH --exclusive

# Fixed values
xlen=10
ylen=10
t_final=0.5
g=9.81
h_max=1.4
CFL=0.5
output_file="Shallow_Water_Equations_MPI_Section_Runtime_Performance.csv"

# Write CSV header
echo "Problem size,Number of processors,Time steps,Iteration,Elapsed time (s),Avg compute fluxes time (s),Avg compute variables time (s),Avg update variables time (s),Avg apply boundary conditions time (s),Avg data transfer time (s)" > $output_file

# Problem sizes
for nx in 5000
do
    # Compute dt
    dx=$(echo "$xlen / $nx" | bc -l)
    c=$(echo "sqrt($g * $h_max)" | bc -l)
    dt=$(echo "$CFL * $dx / $c" | bc -l)

    # Processor loop
    for p in 1 2 4 6 8 10 11 12 13 14 15 16 17 18 19 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48
    do
        output=$(mpirun -np $p ./swem_2d_ts $nx $nx $dt $xlen $ylen $t_final)

        # Extract data from one-liner
        echo "$output" | grep "Problem size" | while read -r line; do

            # Parse fields
            problem_size=$(echo "$line" | awk -F'[:,]' '{print $2}' | tr -d ' ')
            processors=$(echo "$line" | awk -F'[:,]' '{print $4}' | tr -d ' ')
            time_steps=$(echo "$line" | awk -F'[:,]' '{print $6}' | tr -d ' ')
            iteration=$(echo "$line" | awk -F'[:,]' '{print $8}' | tr -d ' ')
            elapsed_time=$(echo "$line" | awk -F'[:,]' '{print $10}' | tr -d ' s')
            avg_cf=$(echo "$line" | awk -F'[:,]' '{print $12}' | tr -d ' s')
            avg_cv=$(echo "$line" | awk -F'[:,]' '{print $14}' | tr -d ' s')
            avg_uv=$(echo "$line" | awk -F'[:,]' '{print $16}' | tr -d ' s')
            avg_bc=$(echo "$line" | awk -F'[:,]' '{print $18}' | tr -d ' s')
            avg_dt=$(echo "$line" | awk -F'[:,]' '{print $20}' | tr -d ' s')

            # Append to CSV
            echo "$problem_size,$processors,$time_steps,$iteration,$elapsed_time,$avg_cf,$avg_cv,$avg_uv,$avg_bc,$avg_dt" >> $output_file
        done
    done
done