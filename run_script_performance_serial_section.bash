#!/bin/bash
#SBATCH -J swe_2d_ts
#SBATCH -o swe_2d_ts.%j
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p bsudfq
#SBATCH -t 168:00:00
#SBATCH --exclusive

# Fixed simulation constants
xlen=10
t_final=0.5
g=9.81
h_max=1.4
CFL=0.5
output_file="Shallow_Water_Equations_Serial_Section_Runtime_Performance.csv"

# Write CSV header (once)
echo "Problem size,Time steps,Iteration,Elapsed time (s),Avg compute fluxes time (s),Avg compute variables time (s),Avg update variables time (s),Avg apply boundary conditions time (s)" > $output_file

# Problem size loop
for nx in 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500 1600 1700 1800 1900 2000 \
          2100 2200 2300 2400 2500 2600 2700 2800 2900 3000 3200 3400 3600 3800 4000 \
          4200 4400 4600 4800 5000 5200 5400 5600 5800 6000 6200 6400 
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