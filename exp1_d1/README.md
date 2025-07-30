# Experiment 1 D1test Phase: Coherent vs. Incoherent Spine Pair Testing

This directory contains scripts for running the D1test phase of Experiment 1, which tests dendritic responses to stimulation of coherent vs. incoherent spine pairs.

## Overview

This experiment specifically uses weights from the "After D1 train" timepoint to evaluate how coherent spine pairs (those with statistically significant positive correlation) and incoherent spine pairs respond to stimulation. This experiment is run independently from the D3test phase, which uses weights from a different timepoint.

## Key Files

- `run_experiment_D1.py`: Main script to run the D1test simulation
- `exp1_cfg_D1.py`: Configuration settings for the D1test phase
- `exp1_init_D1.py`: Setup code for spine identification, pairing, and stimulation
- `exp1_netParams.py`: Network parameters defining cell models and connectivity
- `exp1_job_D1.cmd`: SLURM job submission script

## Running the Experiment

### HPC Deployment

To submit the job to a SLURM scheduler:

```bash
sbatch exp1_job_D1.cmd
```

### MPI Usage

Manual MPI launch:

```bash
mpiexec -n 4 nrniv -python -mpi run_experiment_D1.py
```

## Experiment Parameters

- **Weight Loading Period**: "After D1 train"
- **Activity Data Timepoint**: "D1 test 4k"
- **Trial Structure**: 2s pre-stim → 2s stim → 2s cool (repeated for both coherent and incoherent groups)

## Notes

This experiment is set up to be run independently from the D3test phase, as different weights need to be loaded for each phase. 
