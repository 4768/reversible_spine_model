#!/bin/bash
#SBATCH --job-name=seed_trials_array_255
#SBATCH --array=1-40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your.email@domain.com  # Update with your email
#SBATCH --partition=amd
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=20:00:00
#SBATCH --output=%a_%x_%j.out
#SBATCH --error=%a_%x_%j.err

cd ${SLURM_SUBMIT_DIR}
# Activate your Python environment
# source /path/to/your/neuron-env/bin/activate
export SIM_DATA_DIR=./data  # Set data directory

echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

if [ $SLURM_ARRAY_TASK_ID -le 45 ]; then
    start_seed=$(( (SLURM_ARRAY_TASK_ID - 1) * 3 + 1 ))
    end_seed=$(( start_seed + 2 ))
else
    start_seed=$SLURM_ARRAY_TASK_ID
    end_seed=$SLURM_ARRAY_TASK_ID
fi

export START_SEED=$start_seed
export END_SEED=$end_seed

echo "Processing seeds from $start_seed to $end_seed"

# --- Configuration ---
PROGRAM_STEP1="exp2_init.py"  # simulation

# --- Main Loop - iterate through each seed in the range ---
for CURRENT_SEED in $(seq $start_seed $end_seed); do
    echo "--- Processing seed: $CURRENT_SEED ---"
    
    echo "Starting simulation with seed $CURRENT_SEED..."
    
    # Set the RANDOM_SEED environment variable for the simulation
    export RANDOM_SEED=$CURRENT_SEED
    echo "Set RANDOM_SEED environment variable to: $RANDOM_SEED"
    
    # Run the simulation with the current seed
    mpiexec nrniv -python -mpi $PROGRAM_STEP1
    
    # Check if the simulation was successful
    if [ $? -eq 0 ]; then
        echo "Simulation with seed $CURRENT_SEED completed successfully"
    else
        echo "Simulation with seed $CURRENT_SEED failed"
    fi
    
    echo ""
done

echo "All seeds processed for array task $SLURM_ARRAY_TASK_ID"
