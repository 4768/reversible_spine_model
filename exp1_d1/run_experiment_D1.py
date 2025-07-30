#!/usr/bin/env python

"""
run_experiment_D1.py

Runner script for Experiment 1 D1test phase to evaluate whether coherent spine pairs 
enhance dendritic responses to conditioned stimuli (CS) compared to incoherent pairs.

This script runs specifically the D1test phase only:
1. Analyzes existing weight and activity data to identify coherent and incoherent spine pairs for D1test
2. Creates direct stimulation sources connecting to reference spine and group members
3. Loads weights from the appropriate timepoint (After D1 test)
4. Runs simulation for D1test phase only
5. Compares dendritic responses and saves results to output files

Usage:
    python run_experiment_D1.py

MPI usage:
    mpiexec -n 4 nrniv -python -mpi run_experiment_D1.py
"""

import os
import sys
import time
import argparse
from mpi4py import MPI
import numpy as np
from neuron import h 
import random

# Add parent directory to path to import modules
currentdir = os.path.dirname(os.path.abspath(__file__))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir)

from netpyne import sim

# Import configuration and network parameters
from exp1_cfg_D1 import cfg
from exp1_netParams import netParams

# Import initialization functions (including weight loading)
import exp1_init_D1

# Get MPI rank
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Base paths using relative paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP1_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(EXP1_DIR, 'results_fixed_max', f'seed{cfg.rdseed}')



def apply_weights(weights_dict):
    """Applies weights from dictionary to network connections.
    
    Args:
        weights_dict (dict): {spine_id: weight} mapping to apply
    """
    if not weights_dict:
        print(f"Rank {rank}: No weights provided to apply.")
        return
        
    modified_count = 0
    not_found_count = 0
    skipped_count = 0
    
    # Iterate through cells and their connections
    for cell in sim.net.cells:            
        if hasattr(cell, 'conns'):
            for conn in cell.conns:
                # Skip if connection doesn't have expected attributes
                if not all(k in conn for k in ['preGid', 'sec', 'loc', 'hObj']):
                    continue
                    
                if isinstance(conn['preGid'], str) and 'NetStim' in conn['preGid']:
                    continue
                
                # Construct spine ID using cell type from network
                cell_type = cell.tags['cellType'] if 'cellType' in cell.tags else 'PT'  # Default to PT if not specified
                spine_id = f"{cell_type}_{cell.gid}_{conn['preGid']}.{conn['sec']}.{conn['loc']:.6f}" # NOTE: NEEDS TO BE CHANGED TO INCLUDE synMech
                
                # Apply weight if found in dictionary
                if spine_id in weights_dict:
                    if hasattr(conn['hObj'], 'weight'):
                        conn['hObj'].weight[0] = weights_dict[spine_id]['After D1 train']
                        modified_count += 1
                else:
                    print(f"Rank {rank}: Spine {spine_id} not found in weight data")
                    not_found_count += 1
        else:
            skipped_count += 1
    
    # Gather statistics from all ranks
    all_modified = comm.gather(modified_count, root=0)
    all_not_found = comm.gather(not_found_count, root=0)
    all_skipped = comm.gather(skipped_count, root=0)
    
    # Print per-rank information
    print(f"Rank {rank}: Applied {modified_count} weights to connections, {not_found_count} connections not found in weight data, {skipped_count} cells skipped")
    
    # Print total statistics on rank 0
    if rank == 0 and all_modified:
        total_modified = sum(all_modified)
        total_not_found = sum(all_not_found)
        total_skipped = sum(all_skipped)
        print(f"TOTAL: Applied {total_modified} weights to connections, {total_not_found} connections not found in weight data, {total_skipped} cells skipped")


def main():


    if rank == 0:
        print(f"\n=== Running Experiment 1 D1test Phase: Coherent vs. Incoherent Spine Pair Testing ===")
        print(f"Output will be saved to: {cfg.saveFolder}")
        print(f"Loading weights from: After D1 test")

    # Initialize sim, create network and cells first
    sim.initialize(
        simConfig = cfg,
        netParams = netParams,
        net = None
    )
    sim.net.createPops()
    sim.net.createCells()
    comm.Barrier() # wait for all hosts to create cells

    # Now call the main setup function from exp1_init_D1
    # This will set up only the D1test phase
    if rank == 0:
        print("Running Experiment 1 setup for D1test phase...")

    exp1_init_D1.main_setup('D1test')

    if rank == 0:
        print("Experiment 1 D1test setup complete. Proceeding with simulation...")
        print(f"Total simulation duration: {cfg.duration} ms")


    # Create connections, stimulations, and run simulation
    if rank == 0:
        print("Creating connections, stimulations, and running simulation...")

    sim.net.connectCells()
    sim.net.addStims()
    sim.setupRecording()
    
    for cell in sim.net.cells:
        for conn in cell.conns:
            STDPmech = conn.get('hSTDP')
            if STDPmech:
                setattr(STDPmech, 'NEproc', 1)
                setattr(STDPmech, 'Etau', 1e3)
                secName = conn['sec']
                loc = conn['loc']
                #h.setpointer(cell.secs[secName]['hObj'](loc)._ref_v, 'v_postsyn', STDPmech)
                h.setpointer(conn['hObj']._ref_weight[0], 'Egmax', conn['hObj'].syn())
            elif 'NetStim' in conn['preGid']:
                h.setpointer(conn['hObj']._ref_weight[0], 'Egmax', conn['hObj'].syn())

    if rank == 0:
        print("Running simulation with interval-based recording function...")

    apply_weights(sim.spine_weights)
    # Call runSimWithIntervalFunc with an interval of 20ms for recording
    sim.runSimWithIntervalFunc(20, exp1_init_D1.record_simulation_data)
    

    if rank == 0:
        print(f"=== Experiment 1 D1test finished successfully ===")
        print(f"Results saved to: {cfg.saveFolder}")
        print(f"The simulation tested D1test phase with coherent and incoherent stimulation.")

# --- Main Execution Block ---
if __name__ == '__main__':
    # Run the main simulation logic
    main() 