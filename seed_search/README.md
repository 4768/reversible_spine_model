# Seed Search Tool

This directory contains tools for automatically searching for suitable random seeds for neural network simulations. The seed search process analyzes simulation outputs to determine if a given seed produces results suitable for specific experiments.

## Overview

The seed search tool evaluates random seeds based on:

1. **Branch activity patterns**: Analyzes tuning differences between baseline and test conditions
2. **Statistical significance**: Uses t-tests to identify significantly tuned branches

## Files

- `search.py`: Main analysis script for evaluating seed suitability  
- `seed_search.cmd`: SLURM job script for automated seed search
- `cfg.py`: Configuration parameters for simulations
- `netParams.py`: Network parameters
- `init.py`: Simulation initialization script

## Usage

### Manual Seed Analysis

Test a single seed manually:

```bash
export NEURON_SEED=255
export SIM_DATA_DIR=./data
python search.py data/seed255 data/seed255_norm
```

### Batch Seed Search

For automated searching across multiple seeds using SLURM:

```bash
# Update email and partition in seed_search.cmd
sbatch seed_search.cmd
```

## Suitability Criteria

A seed is considered suitable if:

1. **Baseline difference**: `|4k_tuned - 12k_tuned| ≤ max_diff` (default: 20)
2. **4k increase**: `d3test_4k - baseline_4k > 2`  
3. **12k decrease**: `baseline_12k - d3test_12k > 2`

Where:
- `4k_tuned`/`12k_tuned`: Number of branches significantly tuned to 4kHz/12kHz stimuli
- `baseline`: Activity during baseline period
- `d3test`: Activity during D3 test period

## Configuration

### Environment Variables

- `NEURON_SEED`: Random seed to analyze
- `SIM_DATA_DIR`: Base directory for simulation data (default: `./data`)
- `DEBUG`: Enable debug output (`true`/`false`)

### Script Parameters

Modify `analyze_branch_activity()` parameters:
- `max_diff`: Maximum allowed difference between baseline 4k/12k tuning (default: 20)
- Statistical significance threshold: p < 0.05

## Data Requirements

The script expects these files in the data directories:

**Baseline data** (in `data_dir`):
- `Branch_activity_node_{0-19}_{seed}.txt`
- `spine_elim_form_node_{0-19}_{seed}.txt`

**D3 test data** (in `norm_dir`):
- `Branch_activity_node_{0-19}.txt`

## Exit Codes

- `0`: Seed is suitable
- `1`: Seed is unsuitable or error occurred

## Examples

### Basic usage:
```bash
python search.py ./data/seed255 ./data/seed255_norm
```

### Array job for multiple seeds:
```bash
# Edit seed_search.cmd to set seed range
sbatch seed_search.cmd
```
