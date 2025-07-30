#!/bin/bash
#SBATCH --job-name=exp1_d1_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your.email@domain.com  # Update with your email
#SBATCH --partition=cpu  # Update with your partition name
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=15:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

cd ${SLURM_SUBMIT_DIR}
export SIM_DATA_DIR=./data  # Set data directory
# Activate your Python environment
# source /path/to/your/neuron-env/bin/activate
# run the program supporting MPI with the "mpirun" command
# The -n option is not required since mpirun will automatically determine from SLURM settings
mpiexec nrniv -python -mpi run_experiment_D1.py
