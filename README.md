# Neural Network Simulation: Dendritic Learning and Plasticity

This repository contains neural network simulation code for studying dendritic learning and plasticity using NetPyNE and NEURON. The project investigates how dendrites process information through spine-level plasticity mechanisms.

## Overview

This simulation framework models cortical microcircuits with detailed dendritic compartments and synaptic plasticity. The main focus is on understanding how dendritic spines form functional coherence and contribute to learning through experience-dependent plasticity.

## Project Structure

```
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── exp1_d1/                 # Experiment 1: D1 test phase
├── exp1_d3/                 # Experiment 1: D3 test phase  
├── exp2/                    # Experiment 2: Experimental setup
├── exp2_norm/               # Experiment 2: Normal control
├── exp2_rand/               # Experiment 2: Random seeds
├── seed_search/             # Tools for finding suitable random seeds
└── orig/                    # Original full simulation
```

## Installation

### Prerequisites

- Python 3.8+
- NEURON 8.2+
- NetPyNE 1.0+
- MPI (for parallel simulations)

### Setup

1. Clone this repository:
```bash
git clone <repository-url>
cd simulation
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Compile NEURON mechanisms (in each experiment directory):
```bash
cd exp1_d1/mod
nrnivmodl
```

## Quick Start

Before any experiment, the original full simulation (orig/) should be run to yield base data for experiments.

### Running a Basic Experiment

1. Navigate to an experiment directory:
```bash
cd exp1_d1
```

2. For full simulation:
```bash
python run_experiment_D1.py
```

### Parallel Execution

For faster execution using MPI:
```bash
mpiexec -n 4 nrniv -python -mpi run_experiment_D1.py
```

## Experiments

### Experiment 1: Coherent vs Incoherent Spine Pairs

Tests how dendritic spines respond to coherent (correlated) vs incoherent stimulation patterns.

- **exp1_d1/**: Tests using weights after D1 training phase
- **exp1_d3/**: Tests using weights before D3 test phase

### Experiment 2: Control Studies

Baseline experiments with different weight configurations:

- **exp2/**: Basic configuration
- **exp2_norm/**: Normalized weight distributions  
- **exp2_rand/**: Random seed variations

### Original Full Simulation

- **orig/**: Complete multi-phase learning protocol with training, testing, and extinction phases

### Seed Search Tool

- **seed_search/**: Automated tool for finding suitable random seeds for experiments

The seed search tool analyzes simulation outputs to identify random seeds that produce appropriate baseline conditions and learning-induced branch tuning. See `seed_search/README.md` for detailed usage instructions.

## Configuration

Each experiment can be configured by modifying the respective configuration files:

- `*_cfg.py`: Simulation parameters (duration, recording, etc.)
- `*_netParams.py`: Network structure and connectivity
- `*_init.py`: Experimental protocol and stimulation setup

## Data Output

Simulations generate several output files:

- **Analysis results**: Various `.txt` files with spine activity data and weight data

## Analysis

The simulation includes built-in analysis for:

- Branch activity detection  
- Weight change tracking
- Correlation analysis between spine pairs

## HPC Usage

For high-performance computing environments, example SLURM job scripts are provided:

```bash
sbatch script.cmd
```

**Note**: Update the email address and resource requirements in job scripts before submission.

## Citation

If you use this code in your research, please cite:

```
[Add appropriate citation when published]
```

## License

[Add appropriate license information]
