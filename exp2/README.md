# Experiment 2: Spine Elimination Effect on D3 Test

This directory contains scripts for running Experiment 2, which investigates the effects of spine elimination on branch tuning levels during the D3 test phase. 

## Overview

The experiment:

1. **Loads spine weights** from baseline and post-D1 training periods
2. **Identifies eliminated spines** (weights below threshold after D1 training and HSP)
3. **Restores eliminated spine weights** to their pre-D1 baseline values
4. **Runs D3 test simulation** with both 4K and 12K stimulation protocols
5. **Records network responses** to evaluate the impact of spine restoration

## Key Files

- `exp2_init.py`: Main initialization script implementing the spine restoration workflow
- `exp2_cfg.py`: Configuration settings for Experiment 2, including timing parameters and spine elimination threshold
- `exp2_netParams.py`: Network parameters (identical structure to original to ensure consistency)
- `script.cmd`: SLURM job submission script for HPC deployment

## Experimental Protocol

### Timing Structure
The D3 test phase consists of two sequential stimulation periods:

1. **4K Stimulation**: 
   - Recording starts at t=0ms
   - Stimulus applied from t=5000ms to t=15000ms (10s duration)
   
2. **12K Stimulation**:
   - Recording starts at t=30000ms 
   - Stimulus applied from t=35000ms to t=45000ms (10s duration)

### Spine Restoration Algorithm

1. **Weight Loading**: Load spine weights from baseline ("Before D1 train") and after training ("Before D3 test") periods
2. **Elimination Detection**: Identify spines with weights below `cfg.spineEliminationThreshold` (default: 3)
3. **Weight Restoration**: Replace eliminated spine weights with their baseline values
4. **Network Simulation**: Run D3 test with restored weights

## Configuration Parameters

### Key Settings in `exp2_cfg.py`:

```python
cfg.experiment = 'exp2'
cfg.spineEliminationThreshold = 3  # Threshold for spine elimination
cfg.baseSaveFolder = 'data/exp2'   # Results directory
cfg.duration = 45000               # Total simulation time (45s)
```

### Stimulation Timing:
- `cfg.d3test_4k_stim_start/end`: 4K stimulation window
- `cfg.d3test_12k_stim_start/end`: 12K stimulation window

## Running the Experiment

### Prerequisites

1. **Compiled NEURON mechanisms**:
   ```bash
   cd mod/
   nrnivmodl
   ```

2. **Required data files**: Ensure spine weight files (`weight_node_X_<seed>.txt`) are available in the data directory

### HPC Deployment

Submit to SLURM scheduler:
```bash
sbatch script.cmd
```

### Manual MPI Execution

```bash
mpiexec -n <num_processes> nrniv -python -mpi exp2_init.py
```

### Local Testing

```bash
python exp2_init.py
```

## Data Dependencies

This experiment requires pre-existing spine weight data from previous experiments:
- **Baseline weights**: "Before D1 train" period from weight summary files
- **Post-training weights**: "Before D3 test" period from weight summary files

Weight files should be located in: `data/seed<SEED>/weight_node_X_<SEED>.txt`

## Output Files

Results are saved in `cfg.baseSaveFolder/seed<SEED>/`:
- Network simulation data (branch activity, etc.)

## Relationship to Other Experiments

- **exp2_norm/exp2_rand**: Control experiments for comparison with the spine restoration condition
