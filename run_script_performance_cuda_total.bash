#!/bin/bash
#SBATCH -J swep_2d_tt
#SBATCH -o swep_2d_tt.%j
#SBATCH -c 1
#SBATCH -N 1
#SBATCH -p gpu-v100
#SBATCH -t 05:00:00
#SBATCH --gres=gpu:1
#SBATCH --exclusive

# Fixed simulation constants
xlen=10
t_final=0.5
g=9.81
h_max=1.4
CFL=0.5
output_file="Shallow_Water_Equations_Cuda_Total_Runtime_Performance_Extra.csv"

# CSV Header
echo "Problem size,Time steps,Iteration,Elapsed time (s),Host-device transfer time (s),Device-host transfer time (s)" > $output_file

# Loop over problem sizes
for nx in 400 500 600 700 800
do
    # Compute dt using CFL condition
    dx=$(echo "$xlen / $nx" | bc -l)
    c=$(echo "sqrt($g * $h_max)" | bc -l)
    dt=$(echo "$CFL * $dx / $c" | bc -l)

    # Run simulation
    output=$(./swep_2d_tt $nx $dt $xlen $t_final)

    # Extract line
    echo "$output" | grep "Problem size" | while read -r line; do
        # Extract values using awk
        problem_size=$(echo "$line" | awk -F'[:,]' '{print $2}' | tr -d ' ')
        time_steps=$(echo "$line" | awk -F'[:,]' '{print $4}' | tr -d ' ')
        iterations=$(echo "$line" | awk -F'[:,]' '{print $6}' | tr -d ' ')
        elapsed_time=$(echo "$line" | awk -F'[:,]' '{print $8}' | tr -d ' s')
        transfer_time_hd=$(echo "$line" | awk -F'[:,]' '{print $10}' | tr -d ' s')
        transfer_time_dh=$(echo "$line" | awk -F'[:,]' '{print $12}' | tr -d ' s')

        # Append to CSV
        echo "$problem_size,$time_steps,$iterations,$elapsed_time,$transfer_time_hd,$transfer_time_dh" >> $output_file
    done
done
