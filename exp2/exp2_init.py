"""
exp2_init.py

Initialization script for Experiment 2: Spine elimination effect on D3 test.
This script implements the workflow:
1. Load spine weights from files (Baseline and Before D3 test)
2. Identify eliminated spines (weight below threshold)
3. Restore weights of eliminated spines to their pre-D1 values
4. Run the D3 test simulation and record results

Usage:
    python exp2_init.py

MPI usage:
    mpiexec -n <num_processes> nrniv -python -mpi exp2_init.py
"""

import matplotlib; matplotlib.use('Agg')  # To avoid graphics error in servers
from netpyne import sim
import os
import numpy as np
from neuron import h
import sys
from mpi4py import MPI

# Get MPI rank and size
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ------------------------------------------------------------------------------
# Import configuration and network parameters
# ------------------------------------------------------------------------------
# These need to be imported after MPI setup, which is handled by sim.initialize
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)  # Ensure Experiment_6 directory is in path

from exp2_cfg import cfg
from exp2_netParams import netParams

# Define data directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Use configurable data directory
data_dir = os.environ.get('SIM_DATA_DIR', os.path.join(BASE_DIR, 'data'))
DATA_DIR = os.path.join(data_dir, f"seed{cfg.rdseed}")

# ------------------------------------------------------------------------------
# Experiment Setup
# ------------------------------------------------------------------------------
# Removed condition-based setup since we're only implementing the restored condition
cfg.simLabel = f'{cfg.experiment}'
cfg.saveFolder = os.path.join(cfg.baseSaveFolder, f'seed{cfg.rdseed}')

# Ensure save folder exists (important in MPI context)
if rank == 0:
    os.makedirs(cfg.saveFolder, exist_ok=True)
    print(f"Running Experiment 2: Restoring eliminated spine weights")
    print(f"Data will be saved in: {cfg.saveFolder}")
comm.Barrier()  # Ensure all ranks wait until directory is created

# ------------------------------------------------------------------------------
# Helper Function to Load Weights
# ------------------------------------------------------------------------------
def load_spine_weights(period_label=None):
    """
    Load spine weights from weight_summary_node_X.txt files
    
    Returns:
    --------
    spine_weights_data : dict
        Dictionary of spine weights indexed by spine ID
    """
    if rank == 0:
        print(f"Loading weights" + (f" for period '{period_label}'" if period_label else ""))
    
    spine_weights_data = {}
    
    # Load spine weights from weight files
    for node in range(40):
        weight_file = os.path.join(DATA_DIR, f"weight_node_{node}_{cfg.rdseed}.txt") #to be changed to weight_summary_node_{node}.txt
        
        if os.path.exists(weight_file):
            try:
                with open(weight_file, 'r') as f:
                    lines = f.readlines()
                
                current_period = None
                for line in lines:
                    line = line.strip()
                    
                    if line.startswith("Time:") and "Period:" in line:
                        # Extract period from line like "Time: 123; Period: D1test"
                        # Note: The period name from the file is stored directly
                        current_period = line.split("Period:")[1].strip()
                    
                    elif ":" in line:
                        # Line format: spine_id: weight
                        spine_id, rest = line.split(":")
                        spine_id = spine_id.strip()
                        
                        # Parse weight and count
                        parts = rest.split()
                        try:
                            weight = float(parts[-1].strip())
                            
                            # If specific period requested, only store for that period
                            if period_label and current_period != period_label:
                                continue

                            # Store weight data
                            spine_weights_data[spine_id] = weight
                            
                        except ValueError:
                            continue
            except Exception as e:
                if rank == 0:
                    print(f"Error loading weight file {weight_file}: {str(e)}")
    
    # Gather all weight data to rank 0
    all_weight_data = comm.gather(spine_weights_data, root=0)
    
    # Combine weight data on rank 0
    if rank == 0:
        combined_weights = {}
        for data in all_weight_data:
            combined_weights.update(data)
        spine_weights_data = combined_weights
                        
    # Broadcast combined data to all ranks
    spine_weights_data = comm.bcast(spine_weights_data, root=0)
    # Write spine weights to a single text file (always write regardless of period_label)
    debug_output_dir = os.path.join(cfg.saveFolder, 'debug_weights')
    debug_file = os.path.join(debug_output_dir, 'spine_weights_debug.txt')
    
    DEBUG = False
    if DEBUG == True:
        # Only rank 0 writes the file
        if rank == 0:
            # Create debug directory if it doesn't exist
            if not os.path.exists(debug_output_dir):
                try:
                    os.makedirs(debug_output_dir)
                    print(f"Created debug weights directory: {debug_output_dir}")
                except Exception as e:
                    print(f"Error creating debug directory: {str(e)}")
            
            # Write all weights to a single file in tuple format
            try:
                with open(debug_file, 'w') as f:
                    f.write(f"Time: 0; Period: debug\n\n")
                    f.write(f"# Debug spine weights data\n")
                    f.write(f"# Format: (spine_id, weight)\n")
                    f.write(f"# Total weights: {len(spine_weights_data)}\n\n")
                    
                    for spine_id, weight in spine_weights_data.items():
                        f.write(f"({spine_id}, {weight})\n")
                        
                print(f"Debug weights written to {debug_file}")
            except Exception as e:
                print(f"Error writing debug weights file: {str(e)}")
        if rank == 0:
            print(f"Loaded {len(spine_weights_data)} weights" + (f" for period '{period_label}'" if period_label else ""))
        
    return spine_weights_data


# ------------------------------------------------------------------------------
# Helper Function to Apply Weights to the Network
# ------------------------------------------------------------------------------
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
                    
                # Construct spine ID using cell type from network
                cell_type = cell.tags['cellType'] if 'cellType' in cell.tags else 'PT'  # Default to PT if not specified
                spine_id = f"{cell_type}_{cell.gid}_{conn['preGid']}.{conn['sec']}.{conn['loc']:.6f}" # NOTE: NEEDS TO BE CHANGED TO INCLUDE synMech
                
                # Apply weight if found in dictionary
                if spine_id in weights_dict:
                    if hasattr(conn['hObj'], 'weight'):
                        conn['hObj'].weight[0] = weights_dict[spine_id]
                        modified_count += 1
                else:
                    # print(f"Rank {rank}: {spine_id} not found in weight dictionary")
                    not_found_count += 1
        else:
            skipped_count += 1
    
    # Gather statistics from all ranks
    all_modified = comm.gather(modified_count, root=0)
    all_not_found = comm.gather(not_found_count, root=0)
    all_skipped = comm.gather(skipped_count, root=0)
    
    # Print per-rank information
    # print(f"Rank {rank}: Applied {modified_count} weights to connections, {not_found_count} connections not found in weight data, {skipped_count} cells skipped")
    
    # Print total statistics on rank 0
    if rank == 0 and all_modified:
        total_modified = sum(all_modified)
        total_not_found = sum(all_not_found)
        total_skipped = sum(all_skipped)
        print(f"TOTAL: Applied {total_modified} weights to connections, {total_not_found} connections not found in weight data, {total_skipped} cells skipped")
    
# ------------------------------------------------------------------------------
# Helper Function to Save Spine Categorization
# ------------------------------------------------------------------------------
def save_spine_categorization(eliminated_spines, restored_spines):
    """Saves information about which spines were eliminated and restored.
    
    Args:
        eliminated_spines (dict): {spine_id: (weight_before, weight_after)} for eliminated spines
        restored_spines (dict): {spine_id: (weight_before, weight_after, weight_restored)} for restored spines
    """
    if rank == 0:
        # Ensure the spines directory exists
        os.makedirs(os.path.join(cfg.saveFolder, 'debug_weights'), exist_ok=True)
        
        # Save restored spines info
        with open(os.path.join(cfg.saveFolder, 'debug_weights', 'restored_spines.txt'), 'w') as f:
            f.write("spine_id\tweight_before_D1\tweight_after_D1\tweight_restored\n")
            for spine_id, (weight_before, weight_after, weight_restored) in restored_spines.items():
                f.write(f"{spine_id}\t{weight_before}\t{weight_after}\t{weight_restored}\n")

# ------------------------------------------------------------------------------
# Helper Function to Record Simulation Data
# ------------------------------------------------------------------------------
weights_dict = {}
def record_simulation_data(t):
    global weights_dict
    """Records and saves dendritic potentials, soma spikes, and coherence data."""
    # Data is automatically recorded during simulation based on cfg.recordTraces
    # Additional post-processing and data organization specific to Exp2
    # Save dendrite potentials data

    # Record weights at consistent times across all phases
    if 0 <= t < 22:
        if 0 <= t < 22:
            period = 'debug'
        
        from collections import defaultdict

        save_path = os.path.join(cfg.saveFolder, f'weight_node_{sim.pc.id()}.txt')
        with open(save_path, 'w') as f:
            f.write(f"Time: {t}; Period: {period}\n")
            weightss = defaultdict(list)
            
            for cell in sim.net.cells:
                # Skip VecStim cells
                if 'cellType' in cell.tags and cell.tags['cellType'] == 'VecStim':
                    continue
                    
                # Get cell type, default to 'Unknown' if not specified
                cell_type = cell.tags.get('cellType', 'Unknown')
                
                for conn in cell.conns:
                    STDPmech = conn.get('hSTDP')
                    if STDPmech:
                        # Record weights for all sections, not just Adend1-3
                        spineid = f"{cell_type}_{cell.gid}_{conn['preGid']}.{conn['sec']}.{conn['loc']:6f}"
                        weight = conn['hObj'].weight[0]
                        weightss[spineid].append(weight)
                
            avgweights = {spineid: sum(weights) / len(weights) for spineid, weights in weightss.items()}
            
            # Store averages for this period
            for spineid, avg_weight in avgweights.items():
                if spineid not in weights_dict:
                    weights_dict[spineid] = {}
                weights_dict[spineid][period] = avg_weight
            
            weights_dict = {spineid: weights_dict[spineid] for spineid in sorted(weights_dict)}
        
            # Only write the current period's weight values to file
            for spineid, weights in sorted(avgweights.items()):
                f.write(f"{spineid}: {weights}\n")
            
            f.flush()

    # Extract and write trace data from simData
    if cfg.d3test_4k_record_start <= t < cfg.d3test_4k_stim_end or \
        cfg.d3test_12k_record_start <= t < cfg.d3test_12k_stim_end:
        if cfg.d3test_4k_record_start <= t < cfg.d3test_4k_stim_end: 
            timepoint = 'D3 test 4k'
        elif cfg.d3test_12k_record_start <= t < cfg.d3test_12k_stim_end: 
            timepoint = 'D3 test 12k'
        
        save_path = os.path.join(cfg.saveFolder, f'Branch_activity_node_{sim.pc.id()}.txt')
        with open(save_path, 'a') as f:
            f.write("# time_ms\tdendrite_segment_id\tpotential_mV\n")
            Bactivity = []
            for cell in sim.net.cells:
                if 'cellType' in cell.tags and cell.tags['cellType'] == 'PT':
                    for secName, sec in cell.secs.items():
                        secV = 0
                        n = 0
                        if secName in ['Adend1', 'Adend2', 'Adend3', 'Bdend']:
                            for seg in sec['hObj']:
                                n = n + 1
                                v = sec['hObj'](seg.x).v
                                secV += v
                            avgBranchV = secV / n
                            activityinfo = f"{cell.gid} {secName}: {avgBranchV};"
                            Bactivity.append(activityinfo)
            Bactivity.sort()
            f.write(f"Time: {t}; Timepoint: {timepoint}\n")
            f.write(f"\tBranch activity:\n")
            f.write(f"\t\t{' '.join(Bactivity)}\n")
            f.flush()

    # Save soma spikes data
'''    if cfg.d3test_4k_stim_end <= t < cfg.d3test_4k_stim_end+20 or \
        cfg.d3test_12k_stim_end <= t < cfg.d3test_12k_stim_end+20:
        
        if cfg.d3test_4k_stim_end <= t < cfg.d3test_4k_stim_end+20:
            timepoint = 'D3 test 4k'
            ti = cfg.d3test_4k_record_start
            te = cfg.d3test_4k_stim_end
            print(f'{t}: entering recordspike for {timepoint}')
            recordspike(ti, te, timepoint)
        elif cfg.d3test_12k_stim_end <= t < cfg.d3test_12k_stim_end+20:
            timepoint = 'D3 test 12k'
            ti = cfg.d3test_12k_record_start
            te = cfg.d3test_12k_stim_end
            print(f'{t}: entering recordspike for {timepoint}')
            recordspike(ti, te, timepoint)'''


def recordspike(ti, te, timepoint):
    spktsAll = sim.simData['spkt'] 
    spkidsAll = sim.simData['spkid'] 

    filtered_spikes = [
        (spkid, spkt)
        for spkid, spkt in zip(spkidsAll, spktsAll)
        if ti <= spkt <= te
    ]
    spikes_dict = {}
    for spkid, spkt in filtered_spikes:
        if spkid in spikes_dict:
            spikes_dict[spkid].append(spkt)
        else:
            spikes_dict[spkid] = [spkt]

    save_path = os.path.join(cfg.saveFolder, f'spkt_{sim.pc.id()}.txt')
    with open(save_path, 'a') as f:
        f.write(f"Timepoint: {timepoint}\n")
        f.write("# neuron_id: [spike_times_ms]\n")
        for neuron_id, spkt_list in spikes_dict.items():
            f.write(f'{neuron_id}: {spkt_list}\n')
# ------------------------------------------------------------------------------
# Main Simulation Logic
# ------------------------------------------------------------------------------

# Initialize the simulation infrastructure
sim.initialize(netParams=netParams, simConfig=cfg)

# Create network populations, cells, connections, and stimulation
sim.net.createPops()
sim.net.createCells()
sim.net.connectCells()


for cell in sim.net.cells:  # Each rank processes its own cells
    for conn in cell.conns:
        #print(f'{cell.gid}: {conn}')
        STDPmech = conn.get('hSTDP')
        if STDPmech:
            #nc = sim.pc.gid_connect(rnd_id, STDPmech)          
            setattr(STDPmech, 'NEproc', 1)
            setattr(STDPmech, 'Etau', 1e3)
            secName = conn['sec']
            loc = conn['loc']
            h.setpointer(cell.secs[secName]['hObj'](loc)._ref_v, 'v_postsyn', STDPmech)
            h.setpointer(conn['hObj']._ref_weight[0], 'Egmax', conn['hObj'].syn())

sim.net.addStims()
sim.setupRecording()  
# Load spine weights from files
weights_before_d1 = load_spine_weights("Baseline")
weights_after_d1 = load_spine_weights("Before D3 test")  # D1->D3 weights
# Track eliminated and restored spines for analysis
eliminated_spines = {}
restored_spines = {}


# Start with after D1 weights, then restore weights for eliminated spines
weights_to_apply = weights_after_d1.copy()

# For non-eliminated spines, keep their weights from weights_after_d1
# For eliminated spines, restore to their pre-D1 values
for spine_id in weights_before_d1:
    if spine_id in weights_after_d1:
        # Extract section name from spine_id
        section = spine_id.split('.')[1]  # Get section name from spine_id format
        

        weight_before = weights_before_d1[spine_id]
        weight_after = weights_after_d1[spine_id]
        
        if weight_before >= cfg.spineEliminationThreshold and weight_after < cfg.spineEliminationThreshold:
            # This is an eliminated spine - restore to pre-D1 weight
            weights_to_apply[spine_id] = weight_before
            eliminated_spines[spine_id] = (weight_before, weight_after)
            restored_spines[spine_id] = (weight_before, weight_after, weight_before)
        else:
            # This is a non-eliminated spine - keep the weight from weights_after_d1
            weights_to_apply[spine_id] = weight_after


if rank == 0:
    print(f"Identified {len(eliminated_spines)} eliminated spines (weight < {cfg.spineEliminationThreshold})")
    print(f"Restored weights for {len(restored_spines)} spines to their pre-D1 values")
    print(f"Kept weights from D1->D3 period for {len(weights_to_apply) - len(restored_spines)} non-eliminated spines")

if rank == 0:
    with open(os.path.join(cfg.saveFolder, 'weights_to_apply.txt'), 'w') as f:
        for spine_id, weight in weights_to_apply.items():
            f.write(f"{spine_id}: {weight}\n")

        
# Apply weights to the network connections
comm.Barrier()  # Ensure all ranks have calculated weights
apply_weights(weights_to_apply)

# Save spine categorization information
save_spine_categorization(eliminated_spines, restored_spines)

# Clear previous log files
log_files = [
    f'Branch_activity_node_{sim.pc.id()}.txt',
    #f'spkt_{sim.pc.id()}.txt',
]

for filename in log_files:
    filepath = os.path.join(cfg.saveFolder, filename)
    with open(filepath, 'w') as f:
        pass

 # run parallel Neuron simulation (calling func to record data)
sim.runSimWithIntervalFunc(20, record_simulation_data)      


if rank == 0:
    print(f"Experiment 2 simulation completed successfully!") 