#!/bin/bash
#SBATCH -J swe_2d_tt
#SBATCH -o swe_2d_tt.%j
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
output_file="Shallow_Water_Equations_Serial_Total_Runtime_Performance.csv"

# Write header
echo "Problem size,Time steps,Iteration,Elapsed time (s)" > $output_file

# Loop over problem sizes
for nx in 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500 1600 1700 1800 1900 2000 \
          2100 2200 2300 2400 2500 2600 2700 2800 2900 3000 3200 3400 3600 3800 4000 \
          4200 4400 4600 4800 5000 5200 5400 5600 5800 6000 6200 6400 6600 6800 7000 \
          7200 7400 7600 7800 8000
do
    # Compute dt using CFL condition
    dx=$(echo "$xlen / $nx" | bc -l)
    c=$(echo "sqrt($g * $h_max)" | bc -l)
    dt=$(echo "$CFL * $dx / $c" | bc -l)

    # Run simulation
    output=$(./swe_2d_tt $nx $dt $xlen $t_final)

    # Extract the relevant line
    echo "$output" | grep "Problem size" | while read -r line; do
        # Extract values using awk
        problem_size=$(echo "$line" | awk -F'[:,]' '{print $2}' | tr -d ' ')
        time_steps=$(echo "$line" | awk -F'[:,]' '{print $4}' | tr -d ' ')
        iteration=$(echo "$line" | awk -F'[:,]' '{print $6}' | tr -d ' ')
        elapsed_time=$(echo "$line" | awk -F'[:,]' '{print $8}' | tr -d ' s')

        # Append to CSV
        echo "$problem_size,$time_steps,$iteration,$elapsed_time" >> $output_file
    done
done