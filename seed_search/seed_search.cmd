#!/bin/bash
#SBATCH --job-name=seed_search
#SBATCH --array=269-282
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your.email@domain.com  # Update with your email
#SBATCH --partition=cpu  # Update with your partition name
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=30:00:00
#SBATCH --output=%a_%x_%j.out
#SBATCH --error=%a_%x_%j.err

cd ${SLURM_SUBMIT_DIR}
# Activate your Python environment
# source /path/to/your/neuron-env/bin/activate
export SIM_DATA_DIR=${SIM_DATA_DIR:-./data}  # Set data directory

echo "idx:" $SLURM_ARRAY_TASK_ID

CURRENT_SEED=$SLURM_ARRAY_TASK_ID
#start_seed=$(SLURM_ARRAY_TASK_ID)
#end_seed=$(SLURM_ARRAY_TASK_ID)

export START_SEED=$start_seed
export END_SEED=$end_seed
echo "Processing seeds from $start_seed to $end_seed"

# --- Configuration ---
MAX_ITERATIONS=20  # Maximum number of iterations to try
PROGRAM_STEP1="init_270225.py"  # simulation
PROGRAM_STEP1_norm="norm_control/exp6_init.py"  # simulation
PROGRAM_STEP2="search.py"       # decide if seed is suitable
# Use configurable data directories
norm_dir="${SIM_DATA_DIR}/seed${CURRENT_SEED}_norm"


# --- Main Loop ---
#for CURRENT_SEED in $(seq $start_seed $end_seed); do

DATA_DIR="${SIM_DATA_DIR}/seed${CURRENT_SEED}"
echo "Data directory: $DATA_DIR"

echo "Using seed: $CURRENT_SEED"

# Step 1: Run simulation until baseline period completes
echo "Starting simulation with seed $CURRENT_SEED..."
export NEURON_SEED=$CURRENT_SEED
LOG_FILE="simulation_${NEURON_SEED}.log"
mpiexec nrniv -python -mpi $PROGRAM_STEP1 > $LOG_FILE 2>&1 &
SIM_PID=$!
sleep 10
# Monitor for completion message in the log file
echo "Monitoring $LOG_FILE for completion..."
SIMULATION_COMPLETE=0
while [ $SIMULATION_COMPLETE -eq 0 ]; do
    # Check if "Done; run time =" appears in the log file
    if grep -q "172.1s" $LOG_FILE; then
        echo "Completion message found in log file."
        SIMULATION_COMPLETE=1
        # Terminate the simulation process
        kill -9 $SIM_PID 2>/dev/null
    fi
    
    # Check if simulation process is still running
    if ! kill -0 $SIM_PID 2>/dev/null; then
        echo "Simulation process ended."
        SIMULATION_COMPLETE=1
    fi
    
    # Pause before checking again
    sleep 15
done

echo "Running normal control with seed $CURRENT_SEED..."
LOG_FILE_norm="simulation_${NEURON_SEED}_norm.log"
mpiexec nrniv -python -mpi $PROGRAM_STEP1_norm > $LOG_FILE_norm 2>&1 &
SIM_PID_norm=$!
echo "Monitoring $LOG_FILE_norm for completion..."
SIMULATION_COMPLETE_norm=0
while [ $SIMULATION_COMPLETE_norm -eq 0 ]; do
    if grep -q "Experiment 6 simulation completed successfully!" $LOG_FILE_norm; then
        echo "Completion message found in log file."
        SIMULATION_COMPLETE_norm=1
        kill -9 $SIM_PID_norm 2>/dev/null
    fi
    if ! kill -0 $SIM_PID_norm 2>/dev/null; then
        echo "Simulation process ended."
        SIMULATION_COMPLETE_norm=1
    fi
    sleep 15
done




# Step 2: Analyze results
echo "Analyzing branch activity..."
python $PROGRAM_STEP2 $DATA_DIR $norm_dir
STEP2_STATUS=$?

if [ $STEP2_STATUS -eq 0 ]; then
    echo "Found suitable seed: $CURRENT_SEED"
    echo "Breaking the loop and ending iterations."
    break  # Exit the loop on success
else
    echo "Seed $CURRENT_SEED was not suitable. Trying next seed."
    # ITERATION=$((ITERATION + 1))
fi

#done

echo "End of SLURM job." 
