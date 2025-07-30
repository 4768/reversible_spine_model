#!/bin/bash
#SBATCH --job-name=070525_EXP2_normal_control_255
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your.email@domain.com  # Update with your email
#SBATCH --partition=amd
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=8:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Print job information
echo "=========================================="
echo "SLURM_JOB_ID        = $SLURM_JOB_ID"
echo "SLURM_NODELIST      = $SLURM_NODELIST"
echo "SLURM_NTASKS        = $SLURM_NTASKS" 
echo "SLURM_CPUS_PER_TASK = $SLURM_CPUS_PER_TASK"
echo "This SLURM script is running on host $HOSTNAME"
echo "Working directory is $SLURM_SUBMIT_DIR"
echo "=========================================="

cd ${SLURM_SUBMIT_DIR}

# Use a stable NEURON environment that worked previously
# Activate your Python environment
# source /path/to/your/neuron-env/bin/activate
export SIM_DATA_DIR=./data  # Set data directory

# Run the program supporting MPI
mpiexec nrniv -python -mpi exp2_init.py
