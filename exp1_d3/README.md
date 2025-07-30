# Experiment 1 D3test Phase: Coherent vs. Incoherent Spine Pair Testing

This directory contains scripts for running the D3test phase of Experiment 1, which tests dendritic responses to stimulation of coherent vs. incoherent spine pairs.

## Overview

This experiment specifically uses weights from the "Before D3 test" timepoint to evaluate how coherent spine pairs (those with statistically significant positive correlation) and incoherent spine pairs respond to stimulation. This experiment is run independently from the D1test phase, which uses weights from a different timepoint.

## Key Files

- `run_experiment_D3.py`: Main script to run the D3test simulation
- `exp1_cfg_D3.py`: Configuration settings for the D3test phase
- `exp1_init_D3.py`: Setup code for spine identification, pairing, and stimulation
- `exp1_netParams.py`: Network parameters defining cell models and connectivity
- `exp1_job_D3.cmd`: SLURM job submission script

## Running the Experiment

### HPC Deployment

To submit the job to a SLURM scheduler:

```bash
sbatch exp1_job_D3.cmd
```

### MPI Usage

Manual MPI launch:

```bash
mpiexec -n 4 nrniv -python -mpi run_experiment_D3.py
```

## Experiment Parameters

- **Weight Loading Period**: "Before D3 test"
- **Activity Data Timepoint**: "D3 test 4k"
- **Trial Structure**: 2s pre-stim → 2s stim → 2s cool (repeated for both coherent and incoherent groups)


## Notes

This experiment is set up to be run independently from the D1test phase, as different weights need to be loaded for each phase. 
