#!/usr/bin/env python

"""
exp1_init_D3.py

Implementation of Experiment 1 D3test phase: Coherent vs. Incoherent Spine Pair Stimulation

This script handles data loading, candidate spine identification with pairing,
and configures NetPyNE simulation parameters (recordings, stimulations)
specifically for the D3test phase of Experiment 1.
"""

import re
import os
import numpy as np
import random
import pandas as pd
from mpi4py import MPI
from scipy import stats
import time
import math # Added for ceil
import ast  # Add this import at the top of the file

# Import NetPyNE sim object - assuming it's initialized in the calling script
from netpyne import sim

# Import cfg from the correct file
from exp1_cfg_D3 import cfg
from exp1_netParams import netParams

# Debug flag - set to True to enable debug prints
DEBUG = True  # Changed from True to False to disable debug prints

random.seed(2468)  # Master seed for reproducibility

# Global variable to hold spine weights (load once)
spine_weights = None
# Global variable to hold activity data (load once per phase potentially)
branch_activity_data = {}
# Global variable to store weight data during recording
weights_dict = {}
# Global variable to store pre-stimulus voltage values for comparison
prestim_voltage_dict = {}

phase = 'D3test'  # Set the phase for this experiment

def debug_print(*args, **kwargs):
    """
    Debug print function that only prints if DEBUG flag is True.
    Takes same arguments as print() function.
    """
    if DEBUG:
        print(*args, **kwargs)

# Get MPI rank
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Wait for directories to be created
comm.Barrier()

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Define constants - using relative paths
# Use configurable data directory
data_dir = os.environ.get('SIM_DATA_DIR', os.path.join(BASE_DIR, 'data'))
DATA_DIR = os.path.join(data_dir, f"seed{cfg.rdseed}")

COH_DIR = os.path.join(cfg.saveFolder, "temp", "Coherent")
INCOH_DIR = os.path.join(cfg.saveFolder, "temp", "Incoherent")
WEIGHT_THRESHOLD = 3.0  # Minimum weight for selecting spines
DISTANCE_THRESHOLD = 50.0  # Distance in μm for local recording/pairing (±50 μm)
P_THRESHOLD = 0.05  # p-value threshold for significant correlation
CORRELATION_THRESHOLD = 0  # Correlation coefficient threshold

# D3test specific constants
D3TEST_WEIGHT_PERIOD = 'Before D3 test'
D3TEST_ACTIVITY_TIMEPOINT = 'D3 test 4k'

def load_spine_weights():
    """
    Load spine weights from weight_summary_node_X.txt files (or new_weight/weight_node_X.txt).
    Combines data from all MPI ranks.
    
    Returns:
    --------
    spine_weights_data : dict
        Dictionary of spine weights indexed by spine ID ('gid.sec.loc'),
        with nested dicts for periods. e.g., {'100.Adend1.0.5': {'Before D3 test': 3.5}}
    """
    # Check if already loaded
    global spine_weights
    if spine_weights is not None:
        if rank == 0: print("Spine weights already loaded.")
        return spine_weights

    if rank == 0:
        print("Loading spine weights...")
    
    local_spine_weights = {}
    
    for node in range(40):
        weight_file = os.path.join(DATA_DIR, f"weight_node_{node}_{cfg.rdseed}.txt")

        if os.path.exists(weight_file):
            try:
                with open(weight_file, 'r') as f:
                    lines = f.readlines()
                current_period = None
                for line_num, line in enumerate(lines):
                    line = line.strip()
                    
                    # Detect period line (adjust format if needed)
                    if line.startswith("Time:") and "Period:" in line:
                        try:
                            current_period = line.split("Period:")[1].strip()
                        except IndexError:
                            current_period = None # Handle malformed lines
                            # if rank == 0: print(f"Warning: Malformed period line in {weight_file} line {line_num+1}: {line}")
                    elif ":" in line and current_period:
                        # Line format: spine_id: weight ... (potentially other info)
                        try:
                            spine_id, rest = line.split(":", 1) # Split only once
                            spine_id = spine_id.strip()

                            # Extract weight (assuming it's the last parsable float)
                            weight_val = float(rest.strip())
                            if weight_val is not None:
                                # Store weight data
                                if spine_id not in local_spine_weights:
                                    local_spine_weights[spine_id] = {}
                                local_spine_weights[spine_id][current_period] = weight_val
                            # else:
                                # if rank == 0: print(f"Warning: Could not parse weight for spine \'{spine_id}\' in {weight_file}, line {line_num+1}: {line}\")

                        except (ValueError, IndexError) as e:
                            if rank == 0: print(f"Warning: Skipping line due to parsing error in {weight_file} line {line_num+1}: {line} ({e})")
                            continue
                        except Exception as e:
                            if rank == 0: print(f"Warning: Error processing line in {weight_file} line {line_num+1}: {line} ({e})")

            except Exception as e:
                if rank == 0:
                    print(f"Error loading weight file {weight_file}: {str(e)}")
        
    # Gather all weight data to rank 0
    all_weight_data_list = comm.gather(local_spine_weights, root=0)

    # Combine weight data on rank 0
    if rank == 0:
        combined_weights = {}
        for data in all_weight_data_list:
            combined_weights.update(data)
        local_spine_weights = combined_weights
    
    # Broadcast combined data to all ranks
    local_spine_weights = comm.bcast(local_spine_weights, root=0)
    if rank == 0:
        spine_count = len(local_spine_weights) if local_spine_weights is not None else 0
        print(f"Loaded and combined weights for {spine_count} spines.")
    
    loc_dict = {}
    for cell in sim.net.cells:            
        for secName, sec in cell.secs.items():
            if secName == 'soma':
                continue

            for seg in sec['hObj']:
                # Construct spine ID using cell type from network
                cell_type = cell.tags['cellType'] if 'cellType' in cell.tags else 'PT'  # Default to PT if not specified
                loc_id = f"{cell.gid}.{secName}.{seg.x:.6f}"
                # Apply weight if found in dictionary
                for sid in local_spine_weights:
                    if sid.startswith(f"{cell_type}_{cell.gid}_") and sid.endswith(f".{secName}.{seg.x:.6f}"):
                        if loc_id not in loc_dict:
                            loc_dict[loc_id] = []
                        loc_dict[loc_id].append(local_spine_weights[sid][cfg.exp1['weightPeriod']])
    if rank == 0:
        print(f"Loaded weights for {len(local_spine_weights)} spines with {len(loc_dict)} locations.")
        
    return local_spine_weights, loc_dict


def load_filtered_branch_activity(phase, spine_ids_to_load):
    """
    Load branch/spine activity data from Branch_activity_node_X.txt files for a specific phase,
    but ONLY for the pre-specified spine IDs. This is a memory-optimized version that avoids 
    loading the complete dataset.
    
    Parameters:
    ----------
    phase : str
        Internal phase name ('D3test')
    spine_ids_to_load : set
        Set of spine IDs to load activity data for
    
    Returns:
    --------
    activity_lookup : dict
        Dictionary mapping spine IDs ('gid.sec.loc') to lists of (time, voltage) tuples.
        Only contains entries for the requested spine_ids.
    """
    # If no spine IDs provided, return empty dict
    if not spine_ids_to_load:
        if rank == 0: print("No spine IDs provided to load activity for.")
        return {}

    if rank == 0:
        print(f"Loading filtered branch activity for phase {phase} - {len(spine_ids_to_load)} spine IDs")

    # Store results in a local dict instead of the global one to avoid memory bloat
    filtered_activity_data = {}
    
    # Phase specific parameters
    #TODO: timepoint_label = cfg.exp1['activityTimepoint']
    timepoint_label = 'baseline 4k'
    # Each rank processes its assigned subset of activity files
    my_node_range = range(20)  # All MPI ranks scan all nodes 
    
    # Local dict to collect activity for spines on this rank
    local_activity_data = {}
    spine_id_set = set(spine_ids_to_load)  # Convert to set for faster lookups
    ##:TODO print(f"spine_id_set: {spine_id_set}")
    # Process each activity file
    # Loop over all nodes (0-19) to gather data
    activity_file = os.path.join(DATA_DIR, f"Branch_activity_node_{rank}_{cfg.rdseed}.txt")
    
    if not os.path.exists(activity_file):
        print(f"Activity file {activity_file} does not exist on rank {rank}.")
        
    try:
        with open(activity_file, 'r') as f:
            lines = f.readlines()
        
        current_time = None
        current_timepoint = None
        in_spine_section = False
        
        for line in lines:
            #if rank ==0:
            #    print(f"DEBUG: Processing line: {line.strip()}")
            line = line.strip()
            if not line:
                continue
            
            # Parse time and timepoint
            if re.search(r"^\d+\.\d+;\s*[A-Za-z0-9\s]+$", line):
                #print(line)
                try:
                    time_str, timepoint_str = line.split(';')
                    #time_str = line.split("Time:")[1].split(";")[0].strip()
                    current_time = float(time_str.strip())
                    #timepoint_str = line.split("Timepoint:")[1].strip()
                    current_timepoint = timepoint_str.strip()
                    in_spine_section = False
                except (ValueError, IndexError):
                    continue

            # Check if we're in the spine activity section
            if "Spine activity:" in line:
                in_spine_section = True
                continue
            
            # Process spine activity data - but ONLY for spines in our filter set
            elif in_spine_section and current_timepoint == timepoint_label:
                entries = line.split(';')
                for entry in entries:
                    #if rank == 0:
                        #print(f"DEBUG: Processing entry: {entry}")
                    entry = entry.strip()
                    if not entry or ':' not in entry:
                        #if rank ==0:
                        #   print(f"DEBUG: Skipping invalid entry: {entry}")
                        continue
                        
                    try:
                        spine_id, voltage_str = entry.split(':')
                        spine_id = spine_id.strip()
                        spine_id = spine_id.split('.')[:4]
                        #loc = float(spine_id_parts[2] + '.' + spine_id_parts[3])
                        #spine_id = spine_id_parts[:2] + [f'{loc:.6f}']
                        spine_id = '.'.join(spine_id)
                        #print(f"DEBUG: spine_id: {spine_id}")
                        # Only process spines in our filter set
                        if spine_id in spine_id_set:
                            voltage = float(voltage_str.strip())
                            if spine_id not in local_activity_data:
                                local_activity_data[spine_id] = []
                            local_activity_data[spine_id].append((current_time, voltage))
                    except:
                        continue
    except Exception as e:
        if rank == 0:
            print(f"Error processing file {activity_file}: {str(e)}")
    #print(f'local_activity_data: {local_activity_data}')
    return local_activity_data


def identify_spine_pairs(ref_spine_candidate, activity_lookup, all_spine_weights, phase, sec_length_map):
    """
    Identifies coherent and incoherent spine pairs for a given reference spine.
    """

    ref_spine_id = ref_spine_candidate['spine_id']
    ref_gid = ref_spine_candidate['cellid']
    ref_sec = ref_spine_candidate['secname']
    ref_loc = ref_spine_candidate['loc']

    # Get section length for distance calculation from the pre-calculated map
    ref_sec_len = sec_length_map.get(ref_gid, {}).get(ref_sec)

    # Get reference spine activity
    ref_activity = activity_lookup.get(ref_spine_id)
    if not ref_activity or len(ref_activity) < 2:
        if rank == 0: debug_print(f"DEBUG: Insufficient activity data for ref spine {ref_spine_id}")
        return {'coherent_group': [], 'incoherent_group': [], 'correlation_results': []}

    # Extract voltages for correlation - do this once
    ref_times, ref_voltages = zip(*ref_activity)
    ref_v_array = np.array(ref_voltages)

    coherent_group = []
    incoherent_group = []
    correlation_results = []

    phase_weight_period = cfg.exp1['weightPeriod']

    # Pre-filter other spines to only those on the same cell
    same_cell_spines = {
        spine_id: weights for spine_id, weights in all_spine_weights.items()
        if spine_id != ref_spine_id and  # Skip self
        phase_weight_period in weights and  # Has weight for this period
        weights[phase_weight_period] > WEIGHT_THRESHOLD and  # Meets weight threshold
        int(spine_id.split('.')[0].split('_')[1]) == ref_gid # Same cell
    }


    # Process each potential pair
    pairs_processed = 0
    total_pairs = len(same_cell_spines)
    
    for other_spine_id in same_cell_spines:
        pairs_processed += 1
        try:
            # Parse other spine info
            parts = other_spine_id.split('.')
            if len(parts) > 3:
                other_sec = parts[1]
                                    
                other_loc = float(parts[2]+'.'+parts[3])
                other_spine_id = f'{ref_gid}.{other_sec}.{other_loc}'

                # Find the true location for the other spine
                other_true_loc = None
                for cell in sim.net.cells:
                    if cell.gid == ref_gid and other_sec in cell.secs:
                        sec = cell.secs[other_sec]
                        # Find the segment with closest x value to our approximated loc
                        min_diff = float('inf')
                        for seg in sec['hObj']:
                            diff = abs(seg.x - other_loc)
                            if diff < min_diff:
                                min_diff = diff
                                other_true_loc = seg.x
                        break

                if other_true_loc is None:
                    debug_print(f"Rank {rank} (others) DEBUG: Could not find true location for spine {other_spine_id}")
                    continue

                # Calculate distance
                distance_um = float('inf')
                if ref_sec == other_sec:
                    # Same section - simple distance
                    distance_um = abs(ref_loc - other_loc) * ref_sec_len
                else:
                    # Check if sections are adjacent
                    try:
                        ref_base = ''.join([c for c in ref_sec if not c.isdigit()])
                        ref_num = int(''.join([c for c in ref_sec if c.isdigit()]) or '0')
                        other_base = ''.join([c for c in other_sec if not c.isdigit()])
                        other_num = int(''.join([c for c in other_sec if c.isdigit()]) or '0')
                        
                        if ref_base == other_base and abs(ref_num - other_num) == 1:
                            if ref_num < other_num:
                                distance_um = (1.0 - ref_loc) * ref_sec_len + other_loc * ref_sec_len
                            else:
                                distance_um = ref_loc * ref_sec_len + (1.0 - other_loc) * ref_sec_len
                    except Exception as e:
                        debug_print(f"DEBUG: Error calculating distance: {e}")
                        continue

                if distance_um > DISTANCE_THRESHOLD:
                    continue

                # Get other spine activity
                other_activity = activity_lookup.get(other_spine_id)
                if not other_activity or len(other_activity) < 2:
                    continue

                # Align activity data efficiently
                other_activity_map = dict(other_activity)
                aligned_ref_v = []
                aligned_other_v = []
                common_times = 0
                
                for t, v_ref in zip(ref_times, ref_v_array):
                    v_other = other_activity_map.get(t)
                    if v_other is not None:
                        aligned_ref_v.append(v_ref)
                        aligned_other_v.append(v_other)
                        common_times += 1

                if common_times < 5:
                    continue

                # Calculate correlation
                aligned_ref_v_np = np.array(aligned_ref_v)
                aligned_other_v_np = np.array(aligned_other_v)
                

                try:
                    beta, p_value = stats.pearsonr(aligned_ref_v_np, aligned_other_v_np)
                    if np.isnan(beta) or np.isnan(p_value):
                        continue
                except ValueError:
                    continue

                # Create pair info with true location
                pair_info = {
                    'spine_id': other_spine_id,
                    'cellid': ref_gid,
                    'secname': other_sec,
                    'loc': other_loc,
                    'true_loc': other_true_loc,  # Added true location
                }

                # Classify and store
                if p_value < P_THRESHOLD and beta > CORRELATION_THRESHOLD:
                    coherent_group.append(pair_info)
                    group_assignment = 'Coherent'
                else:
                    incoherent_group.append(pair_info)
                    group_assignment = 'Incoherent'

                correlation_results.append({
                    'ref_spine': ref_spine_id,
                    'paired_spine': other_spine_id,
                    'p_value': p_value,
                    'beta': beta,
                    'group': group_assignment,
                    'distance_um': distance_um,
                    'num_aligned_points': common_times
                })

        except Exception as e:
            if rank == 0: debug_print(f"DEBUG: Error processing pair {other_spine_id}: {e}")
            continue

    debug_print(f"DEBUG: (rank{rank}) In total {len(same_cell_spines)} spines in the same cell with {ref_spine_candidate['spine_id']}, Found {len(coherent_group)} coherent and {len(incoherent_group)} incoherent pairs")

    return {
        'coherent_group': coherent_group,
        'incoherent_group': incoherent_group,
        'correlation_results': correlation_results
    }


def get_candidate_spines_with_pairs(spine_weights_data, activity_data, phase, sec_length_map, sampled_spine_ids):
    """
    Identifies candidate reference spines and calculates their coherent/incoherent pairs.
    Limits selection to a maximum number of spines per PT cell for memory efficiency.
    Only processes spines that belong to cells on the current rank.
    """
    if rank == 0:
        print(f"Identifying candidate spines and pairs for phase: {phase}")

    # Ensure necessary data is present
    if not spine_weights_data or not activity_data:
        if rank == 0: print("Error: Missing spine weights or activity data for candidate identification.")
        return []

    # First verify that sim.net.cells is populated
    if not hasattr(sim, 'net') or not hasattr(sim.net, 'cells') or not sim.net.cells:
        if rank == 0: print("Error: sim.net.cells is not available. Make sure cells are created before calling this function.")
        return []

    candidates = []

    # Get list of cell GIDs that exist on this rank
    local_cell_gids = [cell.gid for cell in sim.net.cells]
    print(f"Local cell GIDs on rank {rank}: {local_cell_gids}")

    spines_considered = 0
    
    for spine_id in sampled_spine_ids:
        try:
            # Parse spine_id: postgid.secname.loc
            parts = spine_id.split('.')
            if len(parts) > 3:
                gid = int(parts[0])
                secname = parts[1]
                loc = float(parts[2]+'.'+parts[3])
                
                # Find the true location by matching with segment locations
                true_loc = None
                cell_found = False
                
                # Find the cell on this rank
                for cell in sim.net.cells:
                    if cell.gid == gid:
                        cell_found = True
                        if secname in cell.secs:
                            sec = cell.secs[secname]
                            # Find the segment with closest x value to our approximated loc
                            min_diff = float('inf')
                            for seg in sec['hObj']:
                                diff = abs(seg.x - loc)
                                if diff < min_diff:
                                    min_diff = diff
                                    true_loc = seg.x
                        break
                
                if not cell_found:
                    continue
                if true_loc is None:
                    print(f"Rank{rank} (ref_spine) Warning: True location not found for spine {spine_id}")
                    continue
                spines_considered += 1
                candidate_info = {
                    'spine_id': f'{gid}.{secname}.{loc}',
                    'cellid': gid,
                    'secname': secname,
                    'loc': loc,  # approximated location from the logs
                    'true_loc': true_loc,  # actual segment location from the cell
                }
                
                # Calculate pairs for this candidate using the provided section lengths
                pairing_results = identify_spine_pairs(
                    candidate_info,
                    activity_data,
                    spine_weights_data,  # Pass all weights for neighbor checks
                    phase,
                    sec_length_map
                )
                
                # Store pairs with the candidate info
                candidate_info['coherent_group'] = pairing_results.get('coherent_group', [])
                candidate_info['incoherent_group'] = pairing_results.get('incoherent_group', [])
                candidate_info['correlation_results'] = pairing_results.get('correlation_results', [])

                # Add as candidate only if it has at least one coherent or incoherent pair
                if candidate_info['coherent_group'] and candidate_info['incoherent_group']:
                    candidates.append(candidate_info)
                    #if rank == 0 and len(candidates) % 5 == 0:  # Report every 5 candidates found
                        #print(f"Found {len(candidates)} valid candidates so far")

        except Exception as e:
            if rank == 0: print(f"Error processing spine {spine_id} as candidate: {e}")

    num_coh = sum(len(c['coherent_group']) for c in candidates)
    num_incoh = sum(len(c['incoherent_group']) for c in candidates)
    print(f"Rank{rank}: Considered {spines_considered} spines. Found {len(candidates)} candidates meeting criteria with pairs within {DISTANCE_THRESHOLD}um.")
    #print(f"Rank{rank}: Total coherent pairs: {num_coh}, incoherent pairs: {num_incoh}")
        
    return candidates


# --- Concurrent Trial Grouping ---

def group_trials_by_cell(candidate_spines, max_concurrent):
    """
    Group trials (represented by candidate ref spines) that can run concurrently.
    Ensures no two trials in the same group target the same cell.
    Each group contains different cells to ensure stimuli are distributed to 
    different cells during the same time period.

    Parameters:
    ----------
    candidate_spines : list
        List of candidate spine dicts (must include 'cellid').
    max_concurrent : int
        Maximum number of concurrent trials per group.

    Returns:
    --------
    trial_groups : list
        List of lists, where each inner list is a group of candidate spine dicts
        that can be simulated concurrently.
    """
    if not candidate_spines: return []

    # Shuffle candidates to potentially improve distribution across cells in early groups
    # Make a copy before shuffling if the original order matters elsewhere
    shuffled_candidates = candidate_spines[:]
    random.shuffle(shuffled_candidates)

    # Group candidates by cellid for easier selection
    candidates_by_cell = {}
    for spine_info in shuffled_candidates:
        cell_id = spine_info['cellid']
        if cell_id not in candidates_by_cell:
            candidates_by_cell[cell_id] = []
        candidates_by_cell[cell_id].append(spine_info)

    trial_groups = []
    # Continue until we've used all available candidates
    remaining_candidates = len(shuffled_candidates)
    
    while candidates_by_cell and remaining_candidates > 0:
        current_group = []
        used_cells_in_group = set()
        
        # Try to fill this group with max_concurrent different cells
        available_cells = list(candidates_by_cell.keys())
        random.shuffle(available_cells)  # Randomize cell selection for each group
        
        # Select up to max_concurrent different cells
        selected_cells = available_cells[:max_concurrent]
        
        # Add one candidate from each selected cell to the current group
        for cell_id in selected_cells:
            if candidates_by_cell[cell_id]:  # Check if there are still candidates for this cell
                spine_info = candidates_by_cell[cell_id].pop(0)  # Take the first candidate
                current_group.append(spine_info)
                used_cells_in_group.add(cell_id)
                remaining_candidates -= 1
                
                # Remove cell from dictionary if no more candidates
                if not candidates_by_cell[cell_id]:
                    del candidates_by_cell[cell_id]
        
        # Add the group if it has any trials
        if current_group:
            trial_groups.append(current_group)
        else:
            # If we couldn't form any new group, break to avoid infinite loop
            break

    return trial_groups


# --- Simulation Setup Functions ---

def setup_stimulations():
    """
    Configure stimulation sources and targets in netParams based on candidate spines and their groups.
    
    This function implements the trial structure required by the experiment:
    - Trials are grouped into concurrent blocks
    - Each block follows a 12-second cycle:
      * 2s pre-stimulus (coherent)
      * 2s stimulus (coherent group)
      * 2s cooling
      * 2s pre-stimulus (incoherent)
      * 2s stimulus (incoherent group)
      * 2s cooling
    """
    # Read and parse the candidate spines from file
    candidate_spines_with_pairs=[]
    try:
        for node in range(20):
            with open(os.path.join(cfg.saveFolder, f'candidate_spines_with_pairs_{rank}_{cfg.rdseed}.txt'), 'r') as f:
                content = f.read()
                # Safely evaluate the string as a Python literal
                candidate_spines_with_pairs.append(ast.literal_eval(content))
                
        if not candidate_spines_with_pairs:
            print(f"Rank{rank}: No candidate spines provided for stimulation setup. Skipping.")
            return
            
        print(f"Rank{rank}: Successfully loaded {len(candidate_spines_with_pairs)} candidate spines from file")
            
    except Exception as e:
        if rank == 0:
            print(f"Error reading candidate spines file: {e}")
        return

    # Ensure empty configuration dictionaries
    if not hasattr(netParams, 'stimSourceParams'):
        if rank == 0: print("Error: sim.net.params.stimSourceParams is not available. Make sure it is defined in netParams.")
        sim.net.params.stimSourceParams = {}
    if not hasattr(netParams, 'stimTargetParams'):
        if rank == 0: print("Error: sim.net.params.stimTargetParams is not available. Make sure it is defined in netParams.")
        sim.net.params.stimTargetParams = {}


    # Group trials for concurrent execution - this ensures different cells for each trial group
    #trial_groups = group_trials_by_cell(candidate_spines_with_pairs, cfg.max_concurrent_trials)
    candidates_by_cell = {}
    for node in candidate_spines_with_pairs:
        for spine_info in node:
            #print(f"DEBUG: spine_info: {spine_info}")
            cell_id = spine_info['cellid']
            if cell_id not in candidates_by_cell:
                candidates_by_cell[cell_id] = []
            candidates_by_cell[cell_id].append(spine_info)

    trial_groups = []
    num_trials = 5 #max(len(spine_infos) for spine_infos in candidates_by_cell.values())
    print(f"Rank {rank}: Found {len(candidates_by_cell)} cells with {num_trials} trials each")
    for idx in range(num_trials):
        trial_groups.append([])
        for cell_id, spine_infos in candidates_by_cell.items():
            if idx < len(spine_infos):
                trial_groups[idx].append(spine_infos[idx])
            else:
                if rank == 0:
                    print(f"Cell {cell_id} has only {len(spine_infos)} trials")

    if rank == 0:
        print(f"Grouped {len(candidate_spines_with_pairs)} trials into {len(trial_groups)} concurrent blocks")

    comm.Barrier() # wait for all hosts to write coherence files
    # Store trial_groups in sim object for access during recording
    sim.trial_groups = trial_groups

    # Stimulation parameters
    stim_delay = 0
    stim_synmech = 'AMPA'
    stim_freq = cfg.stimRate if hasattr(cfg, 'stimRate') else 50.0  # Hz
    stim_number = int(stim_freq * cfg.trial_duration / 1000)  # Approx num of spikes (rate*duration in seconds)

    
    # Track number of sources/targets added
    sources_added_coherent = 0
    targets_added_coherent = 0
    sources_added_incoherent = 0
    targets_added_incoherent = 0

    # Iterate through phases and concurrent blocks to define stims
    global_trial_counter = 0 # For unique naming if needed across phases/blocks
    phase='D3test'

    if rank == 0: print(f"--- Setting up stims for phase: {phase} ---")

    for block_idx, trial_group in enumerate(trial_groups):
        # Calculate timing for this specific block
        # Block start time relative to the beginning of the simulation
        block_abs_start_time = cfg.tstart + block_idx * cfg.full_trial_duration
        
        # Calculate timing for coherent stimulation within this block
        coherent_prestim_start = block_abs_start_time
        coherent_stim_start = coherent_prestim_start + cfg.pre_stimuli_duration
        coherent_cooling_start = coherent_stim_start + cfg.trial_duration
        coherent_stim_number = int(stim_freq * cfg.trial_duration / 1000)  # Approx num of spikes
        
        # Calculate timing for incoherent stimulation within this block
        incoherent_prestim_start = coherent_cooling_start + cfg.reset_duration
        incoherent_stim_start = incoherent_prestim_start + cfg.pre_stimuli_duration
        incoherent_cooling_start = incoherent_stim_start + cfg.trial_duration
        incoherent_stim_number = int(stim_freq * cfg.trial_duration / 1000)  # Approx num of spikes
        
        # Define stims for each trial within this concurrent block
        # Each trial in the group is for a different cell
        for trial_idx_in_block, ref_spine_info in enumerate(trial_group):
            global_trial_counter += 1 # Increment unique ID
            trial_unique_id = f"p{phase[0]}_b{block_idx}_t{trial_idx_in_block}" # Shorter unique ID

            coherent_group = ref_spine_info.get('coherent_group', [])
            incoherent_group = ref_spine_info.get('incoherent_group', [])

            # Determine number of spines to stimulate (min of coherent/incoherent count)
            num_to_stim = min(len(coherent_group), len(incoherent_group))

            if num_to_stim <= 0:
                print(f"Rank{rank}: No spines to stimulate for trial {trial_unique_id}: coherent={len(coherent_group)}, incoherent={len(incoherent_group)}")
                continue
                
            # --- COHERENT GROUP STIMULATION ---
            k = num_to_stim
            if k > 0 and coherent_group:
                # Find correlation data for coherent spines to sort by p-value
                coherent_with_pvals = []
                for spine in coherent_group:
                    # Find the corresponding correlation data for this spine
                    pval = 1.0  # Default high p-value if not found
                    for corr_data in ref_spine_info.get('correlation_results', []):
                        if corr_data['paired_spine'] == spine['spine_id'] and corr_data['group'] == 'Coherent':
                            pval = corr_data['p_value']
                            break
                    coherent_with_pvals.append((spine, pval))
                
                # Sort by p-value (ascending order - most significant first)
                coherent_with_pvals.sort(key=lambda x: x[1])
                
                # Select the k spines with the lowest p-values
                stim_spines_coherent = [item[0] for item in coherent_with_pvals[:k]]
            else:
                stim_spines_coherent = [] # No coherent spines to stimulate
            
            # Create NetStim source and target for each selected coherent spine
            stim_source_name = f'Stim_{trial_unique_id}_Coher_{ref_spine_info.get("spine_id")}'

            sim.net.params.stimSourceParams[stim_source_name]={
                'type': 'NetStim',
                'interval': 20,
                'number': coherent_stim_number, 
                'start': coherent_stim_start, # Coherent stim window start
                'noise': 0
            }
            loc_id = f"{ref_spine_info.get('cellid')}.{ref_spine_info.get('secname')}.{ref_spine_info.get('true_loc'):.6f}"
            stim_weight = max(sim.loc_weights[loc_id])
            sim.net.params.stimTargetParams[f'Target_{trial_unique_id}_Coher_ref_{ref_spine_info.get("spine_id")}'] = {
                'source': stim_source_name,
                'sec': ref_spine_info.get('secname'),
                'loc': ref_spine_info.get('true_loc'),  # Use true location for stimulation
                'weight': stim_weight,
                #'delay': stim_delay,
                'conds': {'cellType': 'PT', 'cellList': [ref_spine_info.get('cellid')]},
                'synMech': stim_synmech
            }

            for spine_idx, spine_to_stim in enumerate(stim_spines_coherent):
                stim_gid = spine_to_stim['cellid']
                stim_sec = spine_to_stim['secname']
                stim_loc = spine_to_stim['true_loc']
                spine_id = spine_to_stim['spine_id']

                target_name = f"Target_{trial_unique_id}_Coher_{spine_id}"
                sources_added_coherent += 1
                stim_weight = max(sim.loc_weights[spine_id])
                sim.net.params.stimTargetParams[target_name] = {
                    'source': stim_source_name,
                    'sec': stim_sec,
                    'loc': stim_loc,  # Use true location for stimulation
                    'weight': stim_weight,
                    #'delay': stim_delay,
                    'conds': {'cellType': 'PT', 'cellList': [stim_gid]},
                    'synMech': stim_synmech
                }
    
                targets_added_coherent += 1

            # --- INCOHERENT GROUP STIMULATION ---
            # Randomly select 'num_to_stim' spines from the incoherent group to stimulate
            k = num_to_stim
            if k > 0 and incoherent_group:
                # Find correlation data for incoherent spines to sort by p-value
                incoherent_with_pvals = []
                for spine in incoherent_group:
                    # Find the corresponding correlation data for this spine
                    pval = 1.0  # Default high p-value if not found
                    for corr_data in ref_spine_info.get('correlation_results', []):
                        if corr_data['paired_spine'] == spine['spine_id'] and corr_data['group'] == 'Incoherent':
                            pval = corr_data['p_value']
                            break
                    incoherent_with_pvals.append((spine, pval))
                
                # Sort by p-value (ascending order - most significant first)
                incoherent_with_pvals.sort(key=lambda x: x[1])
                
                # Select the k spines with the lowest p-values
                stim_spines_incoherent = [item[0] for item in incoherent_with_pvals[:k]]
            else:
                stim_spines_incoherent = [] # No incoherent spines to stimulate

            stim_source_name = f'Stim_{trial_unique_id}_Incoh_{ref_spine_info.get("spine_id")}'

            sim.net.params.stimSourceParams[stim_source_name] = {
                'type': 'NetStim',
                'interval': 20,
                'number': incoherent_stim_number,
                'start': incoherent_stim_start, # Incoherent stim window start
                'noise': 0  
            }
            loc_id = f"{ref_spine_info.get('cellid')}.{ref_spine_info.get('secname')}.{ref_spine_info.get('true_loc'):.6f}"
            stim_weight = max(sim.loc_weights[loc_id])
            sim.net.params.stimTargetParams[f'Target_{trial_unique_id}_Incoh_ref_{ref_spine_info.get("spine_id")}'] = {
                'source': stim_source_name,
                'sec': ref_spine_info.get('secname'),
                'loc': ref_spine_info.get('true_loc'),  
                'weight': stim_weight,
                'conds': {'cellType': 'PT', 'cellList': [ref_spine_info.get('cellid')]},
                'synMech': stim_synmech
            }


            # Create VecStim source and target for each selected incoherent spine
            for spine_idx, spine_to_stim in enumerate(stim_spines_incoherent):
                stim_gid = spine_to_stim['cellid']
                stim_sec = spine_to_stim['secname']
                stim_loc = spine_to_stim['true_loc']
                spine_id = spine_to_stim['spine_id']

                target_name = f"Target_{trial_unique_id}_Incoh_{spine_id}"
                stim_weight = max(sim.loc_weights[spine_id])
                sim.net.params.stimTargetParams[target_name]={
                    'source': stim_source_name,
                    'sec': stim_sec,
                    'loc': stim_loc,  # Use true location for stimulation
                    'weight': stim_weight,
                    #'delay': stim_delay,
                    'synMech': stim_synmech,
                    'conds': {'cellType': 'PT', 'cellList': [stim_gid]}
                }
                sources_added_incoherent += 1
                targets_added_incoherent += 1


                # Define target on the specific spine
                #if rank == 0:
                #    print(f'debug spinetostim (incoherent): for {ref_spine_info.get("spine_id")}: {spine_to_stim}')
            if rank == 0: print(f'{ref_spine_info.get("spine_id")}: Added {len(stim_spines_coherent)} coherent spines and {len(stim_spines_incoherent)} incoherent spines to stim')


    if rank == 0:
        print(f"Defined {sources_added_coherent} stimulation sources in sim.net.params.stimSourceParams for coherent group.")
        print(f"Defined {targets_added_coherent} stimulation targets in sim.net.params.stimTargetParams for coherent group.")
        print(f"Defined {sources_added_incoherent} stimulation sources in sim.net.params.stimSourceParams for incoherent group.")
        print(f"Defined {targets_added_incoherent} stimulation targets in sim.net.params.stimTargetParams for incoherent group.")
        print(f"Total sources: {sources_added_coherent + sources_added_incoherent}")
        print(f"Total targets: {targets_added_coherent + targets_added_incoherent}")
# --- Main Setup Orchestration Function ---


        
def record_simulation_data(t):
    """
    Records and saves dendritic potentials, branch activities and weights(for debug) at specific time intervals.
    
    This function is called at regular intervals during simulation to record data.
    It writes weight data and branch activity to separate files.
    It captures both whole-branch activity (section average) and local activity within ±50μm of reference locations.
    Only records data for cells that exist on the current MPI node.
    Also records spike activity at the end of each trial period using NetPyNE's built-in spike recording.
    
    Parameters:
    ----------
    t : float
        Current simulation time in ms
    """
    global weights_dict, candidate_spines_with_pairs
    
    # First ensure the save directory exists
    
    # Record weights at the beginning of simulation
    if 0 <= t < 22:
        period = 'Before stimulation'  # Use appropriate period label
        
        from collections import defaultdict
        
        save_path = os.path.join(cfg.saveFolder, f'weight_node_{sim.pc.id()}.txt')
        with open(save_path, 'a') as f:
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
                        # Record weights for all sections
                        spineid = f"{cell_type}_{cell.gid}_{conn['preGid']}.{conn['sec']}.{conn['loc']}"
                        weight = conn['hObj'].weight[0]
                        weightss[spineid].append(weight)
                
            # Calculate average weights per spine
            avgweights = {spineid: sum(weights) / len(weights) for spineid, weights in weightss.items()}
            
            # Write the current period's weight values to file
            for spineid, weight in sorted(avgweights.items()):
                f.write(f"{spineid}: {weight:.6f}\n")
            
            f.flush()

    # Determine if we're in pre-stimulus or stimulus period for any trial group
    current_time_relative = t - cfg.tstart
    if current_time_relative < 0:
        return  # Before simulation starts
        
    # Calculate which block and what period within the block we're in
    block_index = int(current_time_relative / cfg.full_trial_duration)
    time_within_block = current_time_relative % cfg.full_trial_duration
    
    # Determine period type based on time within block
    if time_within_block < cfg.pre_stimuli_duration:
        period_type = "pre-coherent"
    elif time_within_block < (cfg.pre_stimuli_duration + cfg.trial_duration):
        period_type = "stim-coherent"
    elif time_within_block < (cfg.pre_stimuli_duration + cfg.trial_duration + cfg.reset_duration):
        period_type = "reset"
        return  # Skip recording during reset periods
    elif time_within_block < (cfg.pre_stimuli_duration + cfg.trial_duration + cfg.reset_duration + cfg.pre_stimuli_duration):
        period_type = "pre-incoherent"

    elif time_within_block < (cfg.pre_stimuli_duration + cfg.trial_duration + cfg.reset_duration + cfg.pre_stimuli_duration + cfg.trial_duration):
        period_type = "stim-incoherent"
    else:
        period_type = "reset"
        return  # Skip recording during reset periods
    
    # Get local cell GIDs that exist on this node
    local_cell_gids = [cell.gid for cell in sim.net.cells]
    
    #if rank == 0:
        #print(f"Recording from cells: {local_cell_gids}")
    
    # Only record during pre-stimulus or stimulus periods
    if "pre" in period_type or "stim" in period_type:
        save_path = os.path.join(cfg.saveFolder, f'Branch_activity_node_{sim.pc.id()}.txt')
        with open(save_path, 'a') as f:
            f.write(f"{t}; Block {block_index}; {period_type}\n")
            
            # Get the trial group for this block (if available)
            if not hasattr(sim, 'trial_groups') or block_index >= len(sim.trial_groups):
                f.write("\tNo trial group available for this block\n")
                return
                
            current_trial_group = sim.trial_groups[block_index]
            #f.write(f"\tTrials in this block: {len(current_trial_group)}\n")
            
            # For each reference spine in the current trial group, only process if cell exists on this node
            for trial_idx, ref_spine in enumerate(current_trial_group):
                ref_gid = ref_spine['cellid']
                
                # Skip recording for cells that don't exist on this node
                if ref_gid not in local_cell_gids:
                    continue
                    
                ref_sec = ref_spine['secname']
                ref_loc = ref_spine['true_loc']
                
                # Create unique key for this reference spine
                ref_key = f"{ref_gid}.{ref_sec}.{ref_loc:.4f}"
                
                f.write(f"\tTrial {trial_idx} - Ref Spine: {ref_key}\n")
                branch_activity = record_branch_activity(ref_gid, ref_sec)                
                local_activity = record_local_activity(ref_gid, ref_sec, ref_loc)
                f.write(f"\tBranch: {branch_activity}; Local: {local_activity}\n")
                
            f.flush()
    


def record_branch_activity(gid, secName):
    """
    Records and calculates the average voltage across an entire branch (section).
    Only records activity for cells that exist on the current node.
    
    Parameters:
    ----------
    gid : int
        Cell GID
    secName : str
        Section name
        
    Returns:
    --------
    str
        String representation of the branch activity data, or error message if not available
    """
    # Find the cell on the current node
    cell_found = False
    for cell in sim.net.cells:
        if cell.gid == gid:
            cell_found = True
            # Get the section
            if secName in cell.secs:
                sec = cell.secs[secName]
                
                # Calculate average voltage across all segments in the section
                voltages = []
                for seg in sec['hObj']:
                    v = sec['hObj'](seg.x).v
                    voltages.append(v)
                
                if not voltages:
                    return f"Error: No segments found in section"
                
                # Calculate average voltage in the section
                avg_voltage = sum(voltages) / len(voltages)
                
                # Return a formatted string with section info and average
                return f"{avg_voltage:.4f}"
            else:
                return f"Error: {secName} not found in cell {gid}"
    
    # If cell is not on this node, return appropriate message
    if not cell_found:
        return f"Cell {gid} not found on node {rank}"

def record_local_activity(gid, secName, loc):
    """
    Records and calculates the average voltage within ±50μm of a reference location.
    Handles crossing section boundaries by checking adjacent sections.
    Uses all available segments within the range instead of sampling.
    
    Parameters:
    ----------
    gid : int
        Cell GID containing the reference point
    secName : str
        Section name containing the reference point
    loc : float
        Location along section (0-1)
        
    Returns:
    --------
    str
        String representation of the local activity data, or error message if not available
    """
    # Find the cell on the current node
    cell_found = False
    for cell in sim.net.cells:
        if cell.gid == gid:
            cell_found = True
            # Get the reference section
            if secName not in cell.secs:
                return f"Error: Section {secName} not found in cell {gid}"
                
            sec = cell.secs[secName]
            sec_len = sec['geom']['L'] if ('geom' in sec and 'L' in sec['geom']) else None
            if sec_len is None:
                return f"Error: Section length not available"
                
            # Extract base name and number for reference section
            ref_base = ''.join([c for c in secName if not c.isdigit()])
            ref_num = int(''.join([c for c in secName if c.isdigit()]) or '0')
            
            # Calculate physical position of the reference point in μm
            ref_pos_um = loc * sec_len
                
            # --- Collect voltages from all segments in the range ---
            voltages = []
            sections_sampled = set()
            
            # Process main section - collect all segments
            segments = list(sec['hObj'])
            sections_sampled.add(secName)
            
            for seg in segments:
                # Calculate physical position of this segment in μm
                seg_pos_um = seg.x * sec_len
                # Calculate distance from reference point
                distance_um = abs(seg_pos_um - ref_pos_um)
                
                # Include if within 50μm
                if distance_um <= 50.0:
                    v = seg.v
                    voltages.append(v)
            
            # --- Check previous adjacent section if needed ---
            dist_needed_before = 50.0 - ref_pos_um  # How far back we need to go
            if dist_needed_before > 0:  # Need to check previous section
                prev_sec_name = f"{ref_base}{ref_num-1}"
                if prev_sec_name in cell.secs:
                    prev_sec = cell.secs[prev_sec_name]
                    prev_sec_len = prev_sec['geom']['L'] if ('geom' in prev_sec and 'L' in prev_sec['geom']) else None
                    
                    if prev_sec_len is not None:
                        sections_sampled.add(prev_sec_name)
                        
                        # Process previous section's segments
                        prev_segments = list(prev_sec['hObj'])
                        for seg in prev_segments:
                            # Position from end of previous section (in μm)
                            pos_from_end = (1.0 - seg.x) * prev_sec_len
                            
                            # Include if within our needed distance
                            if pos_from_end <= dist_needed_before:
                                v = seg.v
                                voltages.append(v)
            
            # --- Check next adjacent section if needed ---
            dist_needed_after = 50.0 - (sec_len - ref_pos_um)  # How far forward we need to go
            if dist_needed_after > 0:  # Need to check next section
                next_sec_name = f"{ref_base}{ref_num+1}"
                if next_sec_name in cell.secs:
                    next_sec = cell.secs[next_sec_name]
                    next_sec_len = next_sec['geom']['L'] if ('geom' in next_sec and 'L' in next_sec['geom']) else None
                    
                    if next_sec_len is not None:
                        sections_sampled.add(next_sec_name)
                        
                        # Process next section's segments
                        next_segments = list(next_sec['hObj'])
                        for seg in next_segments:
                            # Position from start of next section (in μm)
                            pos_from_start = seg.x * next_sec_len
                            
                            # Include if within our needed distance
                            if pos_from_start <= dist_needed_after:
                                v = seg.v
                                voltages.append(v)
            
            # Calculate average voltage across all segments in range
            if not voltages:
                return f"Error: No segments found within ±50μm range"
                
            avg_voltage = sum(voltages) / len(voltages)
            
            # Return a formatted string with details
            sections_str = ", ".join(sorted(sections_sampled))
            return f"{avg_voltage:.4f}"
            
    # If cell is not on this node, return appropriate message
    if not cell_found:
        return f"Cell {gid} not found on node {sim.pc.id()}"

def main_setup(selected_phase=None):
    """
    Main setup function: Loads data, finds candidates with pairs,
    and configures simulation parameters (recordings, stimulations).
    This function prepares `cfg` modifications needed before `sim.create()` etc.
    
    For D3test_experiment, this function only processes D3test phase data.
    
    Parameters:
    ----------
    selected_phase : str or None
        If specified ('D3test'), only processes data for that phase.
        If None, defaults to 'D3test' for this specialized script.
    
    Returns:
    --------
    list or None
        List of candidate spine dictionaries with pairing info if successful, otherwise None.
    """
    global spine_weights, branch_activity_data
    os.makedirs(cfg.saveFolder, exist_ok=True)

    # Override selected_phase to ensure we only process D3test
    selected_phase = 'D3test'

    start_time = time.time()
    if rank == 0:
        print("="*40)
        print("Starting Experiment 1 D3test Setup")
        print("="*40)

    # --- Pre-computation: Section Lengths (CRITICAL) ---
    sec_length_map = {}
    if rank == 0: print("Attempting to gather section lengths from sim.net.cells...")
    try:
        # Check if sim.net.cells is populated
        if hasattr(sim, 'net') and hasattr(sim.net, 'cells') and sim.net.cells:
            # First collect section lengths from all nodes
            local_sec_lengths = {}
            for cell in sim.net.cells:
                if 'cellType' in cell.tags:
                    gid = cell.gid
                    if gid not in local_sec_lengths:
                        local_sec_lengths[gid] = {}
                        
                    for sec_name, sec_data in cell.secs.items():
                        if 'geom' in sec_data and 'L' in sec_data['geom']:
                            local_sec_lengths[gid][sec_name] = sec_data['geom']['L']
        
        else:
            print("Warning: sim.net.cells not available or empty. Section lengths cannot be gathered.")
            print("         Distance-based pairing and local recordings will be inaccurate or skipped.")
            
        # Gather all local section lengths to rank 0
        all_sec_lengths = comm.gather(local_sec_lengths, root=0)
        
        # Combine all gathered section lengths on rank 0
        if rank == 0:
            for node_sec_lengths in all_sec_lengths:
                for gid, sec_data in node_sec_lengths.items():
                    if gid not in sec_length_map:
                        sec_length_map[gid] = {}
                    sec_length_map[gid].update(sec_data)
    except Exception as e:
        if rank == 0: print(f"Error gathering section lengths: {e}")
        if rank == 0: print(f"Error details: {str(e)}")
        import traceback
        if rank == 0: print(f"Traceback: {traceback.format_exc()}")

    # Broadcast the combined map to all ranks
    sec_length_map = comm.bcast(sec_length_map, root=0)
    if not sec_length_map and rank == 0:
        print("Warning: Section length map is empty after broadcast.")
    #elif rank == 0:
        # Print summary of gathered section lengths
        #print("\nSection Length Summary:")
        #for gid in sorted(sec_length_map.keys()):
            #print(f"GID {gid}: {len(sec_length_map[gid])} sections")
            #for sec_name, length in sorted(sec_length_map[gid].items()):
                #print(f"  {sec_name}: {length}")

    # 1. Load Weights (once)
    spine_weights, loc_weights = load_spine_weights()
    sim.spine_weights = spine_weights
    sim.loc_weights = loc_weights
    if spine_weights is None: # Check if loading failed (might be empty dict on success)
        if rank == 0: print("Aborting setup: Failed to load spine weights.")
        comm.Abort(1); return None # Abort if weights are essential
    if not spine_weights and rank == 0:
        print("Warning: Spine weights data is empty after loading.")

#######################################################################

    phase = 'D3test'
    if rank == 0: print(f"\n--- Processing D3test Phase Data ---")

    # --- MEMORY OPTIMIZATION: Pre-filter spines before loading activity data ---
    
    # Get list of cell GIDs that exist on this rank
    local_cell_gids = [cell.gid for cell in sim.net.cells]

    print(f"Local cell GIDs on rank {rank}: {local_cell_gids}")

    # First, group spines by cell and filter by weight
    phase_weight_period = cfg.exp1['weightPeriod']
    spines_by_cell = {}
    filtered_spine_ids = set()
    
    for spine_id, weights in spine_weights.items():
        if phase_weight_period in weights and weights[phase_weight_period] > WEIGHT_THRESHOLD:
            try:
                parts = spine_id.split('.')
                if len(parts) > 3:
                    celltype = parts[0].split('_')[0]
                    gid = int(parts[0].split('_')[1])
                    secname = parts[1]
                    
                    # Skip soma sections as they don't have activity records
                    if secname.startswith('soma'):
                        continue
                        
                    # Only process spines that belong to cells on this rank
                    if celltype == 'PT' and gid in local_cell_gids:
                        if gid not in spines_by_cell:
                            spines_by_cell[gid] = []
                        spines_by_cell[gid].append((spine_id, weights))
                        
                        # Add mapped spine ID format to our filtered list
                        loc = float(parts[2]+'.'+parts[3])
                        simplified_id = f'{gid}.{secname}.{loc}'
                        filtered_spine_ids.add(simplified_id)
            except Exception as e:
                if rank == 0: print(f"Error parsing spine {spine_id}: {e}")
    
    # Gather all filtered spine IDs from all ranks
    all_filtered_spine_ids = comm.gather(filtered_spine_ids, root=0)
    
    # Combine and broadcast filtered spine IDs
    if rank == 0:
        combined_spine_ids = set()
        for spine_ids in all_filtered_spine_ids:
            combined_spine_ids.update(spine_ids)
        print(f"Pre-filtered {len(combined_spine_ids)} spine IDs based on weight threshold and cell criteria")
    else:
        combined_spine_ids = None
    
    combined_spine_ids = comm.bcast(combined_spine_ids, root=0)
    
    # For more aggressive filtering, randomly sample at most N spines per cell
    max_spines_per_cell = 100
    total_spine_limit = 100   # Hard limit on total spines across all cells
    
    # Group combined spine IDs by cell for sampling
    spine_ids_by_cell = {}
    for spine_id in combined_spine_ids:
        try:
            parts = spine_id.split('.')
            gid = int(parts[0])
            if gid not in spine_ids_by_cell:
                spine_ids_by_cell[gid] = []
            spine_ids_by_cell[gid].append(spine_id)
        except:
            continue
    
    # Random sampling to reduce number of spines
    sampled_spine_ids = set()
    for gid, spines in spine_ids_by_cell.items():
        # Sample up to max_spines_per_cell from this cell
        sample_size = min(max_spines_per_cell, len(spines))
        sampled = random.sample(spines, sample_size)
        sampled_spine_ids.update(sampled)
        
    if rank == 0:
        print(f"After sampling, reduced to {len(sampled_spine_ids)} spine IDs across {len(spine_ids_by_cell)} cells")
    
    # We need to identify spines that might form pairs with our candidates
    # These are spines on the same cell within distance threshold
    
    # Organize spines by cell for easy lookup of potential neighbors
    all_spines_by_cell = {}
    for gid, spines in spine_ids_by_cell.items():
        for spine_id in spines:
            try:
                parts = spine_id.split('.')
                if len(parts) > 3:
                    secname = parts[1]
                    loc = float(parts[2]+'.'+parts[3])
                    
                    if gid not in all_spines_by_cell:
                        all_spines_by_cell[gid] = []
                    
                    all_spines_by_cell[gid].append({
                        'spine_id': spine_id,
                        'cellid': gid,
                        'secname': secname,
                        'loc': loc
                    })
            except Exception as e:
                continue
    
    # For each candidate spine, identify potential neighbors based on distance
    neighbor_spine_ids = set()
    # Use section lengths to calculate approximate distances
    for spine_id in sampled_spine_ids:
        try:
            parts = spine_id.split('.')
            ref_gid = int(parts[0])
            ref_sec = parts[1]
            ref_loc = float(parts[2]+'.'+parts[3])
            
            if ref_gid not in sec_length_map or ref_sec not in sec_length_map[ref_gid]:
                continue
                
            ref_sec_len = sec_length_map[ref_gid][ref_sec]
            
            # Examine all other spines on the same cell
            for other_spine in all_spines_by_cell.get(ref_gid, []):
                other_id = other_spine['spine_id']
                other_sec = other_spine['secname']
                other_loc = other_spine['loc']
                
                # Skip self
                if other_id == spine_id:
                    continue
                    
                # Skip if we don't have section length info
                if other_sec not in sec_length_map[ref_gid]:
                    continue
                
                other_sec_len = sec_length_map[ref_gid][other_sec]
                
                # Calculate distance
                distance_um = float('inf')
                if ref_sec == other_sec:
                    # Same section - simple distance
                    distance_um = abs(ref_loc - other_loc) * ref_sec_len
                else:
                    # Check if sections are adjacent
                    try:
                        ref_base = ''.join([c for c in ref_sec if not c.isdigit()])
                        ref_num = int(''.join([c for c in ref_sec if c.isdigit()]))
                        other_base = ''.join([c for c in other_sec if not c.isdigit()])
                        other_num = int(''.join([c for c in other_sec if c.isdigit()]))
                        
                        if ref_base == other_base and abs(ref_num - other_num) == 1:
                            if ref_num < other_num:
                                distance_um = (1.0 - ref_loc) * ref_sec_len + other_loc * other_sec_len
                            else:
                                distance_um = ref_loc * ref_sec_len + (1.0 - other_loc) * other_sec_len
                    except:
                        continue
                
                # Add to neighbor list if within distance threshold
                if distance_um <= DISTANCE_THRESHOLD:
                    neighbor_spine_ids.add(other_id)
        except Exception as e:
            continue
    
    # Combine reference and neighbor spine IDs for activity loading
    combined_spine_ids_for_activity = sampled_spine_ids.union(neighbor_spine_ids)
    
    if rank == 0:
        print(f"Added {len(neighbor_spine_ids)} potential neighbor/pair spines within {DISTANCE_THRESHOLD}um")
        print(f"Total spines for activity data loading: {len(combined_spine_ids_for_activity)}")
    
    # 2. Load ONLY pre-filtered activity data
    # We now include both candidate spines AND their potential pair spines
    activity_data = load_filtered_branch_activity(phase, combined_spine_ids_for_activity)
    if not activity_data:
        if rank == 0: print(f"Warning: Failed to load or empty activity data for pre-filtered spines. Aborting.")
        comm.Abort(1); return None
        
    if rank == 0:
        print(f"Successfully loaded activity data for {len(activity_data)} pre-filtered spines")

    # Get candidate spines WITH pre-calculated pairs for this phase
    candidates = get_candidate_spines_with_pairs(
        spine_weights,
        activity_data,
        phase,
        sec_length_map, # Provide the length map
        sampled_spine_ids
    )

    if not candidates:
        if rank == 0: print(f"Error: No valid candidate spines found for D3test phase. Aborting setup.")
        comm.Abort(1); return None

    print(f"\nRank{rank}: Total candidate reference spines identified for D3test: {len(candidates)}")

    # Store candidates in sim for access during recording
    sim.candidate_spines_with_pairs = candidates

    unique_cell_ids = set()
    # Collect unique cell IDs from candidates
    for candidate in candidates:
        unique_cell_ids.add(candidate['cellid'])

    # Print diagnostic information about cell IDs in candidates
    comm.Barrier()
    print(f"debug: rank{rank}: Unique cell IDs in candidates: {sorted(list(unique_cell_ids))}")
    print(f"debug: rank{rank}: Total candidates: {len(candidates)}")
    cell_id_counts = {}
    for candidate in candidates:
        cell_id = candidate['cellid']
        cell_id_counts[cell_id] = cell_id_counts.get(cell_id, 0) + 1
    for cell_id, count in sorted(cell_id_counts.items()):
        print(f"debug: rank{rank}:   Cell {cell_id}: {count} candidates")

    with open(os.path.join(cfg.saveFolder, f'candidate_spines_with_pairs_{rank}_{cfg.rdseed}.txt'), 'w') as f:
        f.write(f"{candidates}")
    # 3. Setup Recordings 
    # Note: Not using the old setup_recordings function anymore
    # Instead, we'll use the interval-based recording mechanism (record_simulation_data)
    # that will be called during simulation via sim.runSimWithIntervalFunc
    
    # Clear previous log files if they exist (at rank 0)
    log_files = [
        os.path.join(cfg.saveFolder, f'Branch_activity_node_{rank}.txt'),
        os.path.join(cfg.saveFolder, f'weight_node_{rank}.txt')
    ]
    
    for filepath in log_files:
        try:
            with open(filepath, 'w') as f:
                pass  # Create empty file or clear existing file
            print(f"Cleared recording file: {filepath}")
        except Exception as e:
            print(f"Warning: Could not clear file {filepath}: {e}")
    
    comm.Barrier() # wait for all hosts to write coherence files

    # 4. Setup Stimulations (based on candidates found)
    setup_stimulations()

    comm.Barrier() # wait for all hosts to write coherence files

    if rank == 0:
        elapsed_time = time.time() - start_time
        print("\n" + "="*40)
        print(f"D3test Experiment Setup Completed in {elapsed_time:.2f} seconds")
        print("NetPyNE netParams object updated with: stimSourceParams, stimTargetParams")
        print(f"Recording will be handled by interval function during simulation")
        print(f"Ensure sim.netParams includes necessary synMechs (e.g., '{cfg.exp1.get('stimSynMech', 'AMPA')}').")
        print("="*40)

    # Return the candidates list for potential use in analysis
    return candidates


# Example Call Guard (this script is meant to be imported)
if __name__ == "__main__":
    if rank == 0:
        print("*"*60)
        print(" This script (exp1_init_D3.py) is intended to be imported.")
        print(" Call 'main_setup()' from a main simulation script ")
        print(" (e.g., run_experiment_D3.py) AFTER basic sim/net initialization ")
        print(" and AFTER sim.createNetwork()/sim.createCells() are called,")
        print(" so that section lengths can be retrieved.")
        print("*"*60)


