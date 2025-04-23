#!/bin/bash
#SBATCH -J swep_2d_tt
#SBATCH -o swep_2d_tt.%j
#SBATCH -c 1
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH -t 08:00:00
#SBATCH --gres=gpu:1
#SBATCH --exclusive

# Fixed simulation constants
xlen=10
t_final=0.5
g=9.81
h_max=1.4
CFL=0.5
output_file="Shallow_Water_Equations_Cuda_Total_Runtime_Performance.csv"

# Write CSV header
echo "Problem size,dt,Iteration,Elapsed time (s),Host-Device data transfer time (s)" > $output_file

# Loop over problem sizes
for nx in 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500 1600 1700 1800 1900 2000 \
          2100 2200 2300 2400 2500 2600 2700 2800 2900 3000 3200 3400 3600 3800 4000 \
          4200 4400 4600 4800 5000 5200 5400 5600 5800 6000 6200 6400 6600 6800 7000 \
          7200 7400 7600 7800 8000 9000 10000 11000 12000 13000 14000 15000 16000
do
    # Compute dt using CFL condition
    dx=$(echo "$xlen / $nx" | bc -l)
    c=$(echo "sqrt($g * $h_max)" | bc -l)
    dt=$(echo "$CFL * $dx / $c" | bc -l)

    # Run simulation
    output=$(./swep_2d_tt $nx $dt $xlen $t_final)

    # Extract performance line(s)
    echo "$output" | grep "Problem size" | while read -r line; do
        problem_size=$(echo "$line" | awk -F'[:,]' '{print $2}' | tr -d ' ')
        iterations=$(echo "$line" | awk -F'[:,]' '{print $4}' | tr -d ' ')
        elapsed_time=$(echo "$line" | awk -F'[:,]' '{print $6}' | tr -d ' s')
        transfer_time=$(echo "$line" | awk -F'[:,]' '{print $8}' | tr -d ' s')

        echo "$problem_size,$dt,$iterations,$elapsed_time,$transfer_time" >> $output_file
    done
done
