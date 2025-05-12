#!/bin/bash
#SBATCH -J swe_2d_ts
#SBATCH -o swe_2d_ts.%j
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p bsudfq
#SBATCH -t 48:00:00
#SBATCH --exclusive

# Fixed simulation constants
xlen=10
t_final=0.5
g=9.81
h_max=1.4
CFL=0.5
output_file="Shallow_Water_Equations_Serial_Section_Runtime_Performance_Extra.csv"

# Write CSV header (once)
echo "Problem size,Time steps,Iteration,Elapsed time (s),Avg compute fluxes time (s),Avg compute variables time (s),Avg update variables time (s),Avg apply boundary conditions time (s)" > $output_file

# Problem size loop
for nx in 7200 7600
do
    # Compute dt using CFL condition
    dx=$(echo "$xlen / $nx" | bc -l)
    c=$(echo "sqrt($g * $h_max)" | bc -l)
    dt=$(echo "$CFL * $dx / $c" | bc -l)

    # Run CUDA program
    output=$(./swe_2d_ts $nx $dt $xlen $t_final)

    # Extract line
    echo "$output" | grep "Problem size" | while read -r line; do
        # Extract values using awk
        problem_size=$(echo "$line" | awk -F'[:,]' '{print $2}' | tr -d ' ')
        time_steps=$(echo "$line" | awk -F'[:,]' '{print $4}' | tr -d ' ')
        iteration=$(echo "$line" | awk -F'[:,]' '{print $6}' | tr -d ' ')
        elapsed_time=$(echo "$line" | awk -F'[:,]' '{print $8}' | tr -d ' s')
        avg_cf=$(echo "$line" | awk -F'[:,]' '{print $10}' | tr -d ' s')
        avg_cv=$(echo "$line" | awk -F'[:,]' '{print $12}' | tr -d ' s')
        avg_uv=$(echo "$line" | awk -F'[:,]' '{print $14}' | tr -d ' s')
        avg_bc=$(echo "$line" | awk -F'[:,]' '{print $16}' | tr -d ' s')

        # Append to CSV
        echo "$problem_size,$time_steps,$iteration,$elapsed_time,$avg_cf,$avg_cv,$avg_uv,$avg_bc" >> $output_file
    done
done