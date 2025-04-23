#!/bin/bash
#SBATCH -J swep_2d_tk
#SBATCH -o swep_2d_tk.%j
#SBATCH -c 1
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH -t 01:00:00
#SBATCH --gres=gpu:1
#SBATCH --exclusive

# Fixed simulation constants
xlen=10
t_final=0.5
g=9.81
h_max=1.4
CFL=0.5
output_file="Shallow_Water_Equations_Cuda_Kernel_Runtime_Performance.csv"

# CSV header
echo "Problem size,Time steps,Iterations,Elapsed time (s),Avg fluxes (s),Avg variables (s),Avg update (s),Avg boundary (s)" > $output_file

# Problem size loop
for nx in 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500 1600 1700 1800 1900 2000 \
          2100 2200 2300 2400 2500 2600 2700 2800 2900 3000 3200 3400 3600 3800 4000 \
          4200 4400 4600 4800 5000 5200 5400 5600 5800 6000 6200 6400 6600 6800 7000 \
          7200 7400 7600 7800 8000 9000 10000 11000 12000 13000 14000 15000 16000
do
    # Compute dt using CFL condition
    dx=$(echo "$xlen / $nx" | bc -l)
    c=$(echo "sqrt($g * $h_max)" | bc -l)
    dt=$(echo "$CFL * $dx / $c" | bc -l)

    # Run CUDA program
    output=$(./swep_2d_tk $nx $dt $xlen $t_final)

    # Extract the line
    line=$(echo "$output" | grep "Problem size")

    # Parse values from the single print line
    problem_size=$(echo "$line" | awk -F'[:,]' '{print $2}' | tr -d ' ')
    time_steps=$(echo "$line" | awk -F'[:,]' '{print $4}' | tr -d ' ')
    iterations=$(echo "$line" | awk -F'[:,]' '{print $6}' | tr -d ' ')
    elapsed_time=$(echo "$line" | awk -F'[:,]' '{print $8}' | tr -d ' s')
    avg_cf=$(echo "$line" | awk -F'[:,]' '{print $10}' | tr -d ' s')
    avg_cv=$(echo "$line" | awk -F'[:,]' '{print $12}' | tr -d ' s')
    avg_uv=$(echo "$line" | awk -F'[:,]' '{print $14}' | tr -d ' s')
    avg_bc=$(echo "$line" | awk -F'[:,]' '{print $16}' | tr -d ' s')

    # Write to CSV
    echo "$problem_size,$time_steps,$iterations,$elapsed_time,$avg_cf,$avg_cv,$avg_uv,$avg_bc" >> $output_file
done