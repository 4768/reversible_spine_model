#!/usr/bin/env python
"""
Seed Search Script for Neural Network Simulation

This script analyzes branch activity data and spine change rates to determine
if a given random seed produces suitable simulation results for experiments.
"""
import numpy as np
import os
import sys
import glob
import re
from scipy import stats

# Enable/disable debug output
DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'

def debug_print(msg):
    """Print debug messages only if DEBUG is enabled"""
    if DEBUG:
        print(f"DEBUG: {msg}")

def spine_change_rate(data_dir, seed):
    """Makes sure elimination rate is higher than formation rate"""
    files = [os.path.join(data_dir, f'spine_elim_form_node_{i}_{seed}.txt') for i in range(20)]

    debug_print(f"Found {len(files)} spine change files in {data_dir}")
    avg_elim_rate_a = []
    avg_form_rate_a = []
    avg_elim_rate_t = []
    avg_form_rate_t = []
    for file in files:
        if not os.path.exists(file):
            continue
        with open(file, 'r') as f:
            lines = f.readlines()
            if DEBUG:
                print(lines)
            for line in lines:
                if "Before D3 test" and 'Apical' in line:
                    elim_rate = float(re.search(r'elim (\d+.\d+)%;', line).group(1))
                    form_rate = float(re.search(r'form (\d+.\d+)%', line).group(1))
                    avg_elim_rate_a.append(elim_rate)
                    avg_form_rate_a.append(form_rate)
                    debug_print(f"Elimination rate: {elim_rate}, Formation rate: {form_rate}")
                elif "Before D3 test" and 'Total' in line:
                    elim_rate = float(re.search(r'elim (\d+.\d+)%;', line).group(1))
                    form_rate = float(re.search(r'form (\d+.\d+)%', line).group(1))
                    avg_elim_rate_t.append(elim_rate)
                    avg_form_rate_t.append(form_rate)
                    debug_print(f"Elimination rate: {elim_rate}, Formation rate: {form_rate}")
    if len(avg_elim_rate_a) > 0 and len(avg_form_rate_a) > 0:
        avg_elim_rate_a = np.mean(avg_elim_rate_a)
        avg_form_rate_a = np.mean(avg_form_rate_a)
        debug_print(f"Average elimination rate: {avg_elim_rate_a}, Average formation rate: {avg_form_rate_a}")
    if len(avg_elim_rate_t) > 0 and len(avg_form_rate_t) > 0:
        avg_elim_rate_t = np.mean(avg_elim_rate_t)
        avg_form_rate_t = np.mean(avg_form_rate_t)
        debug_print(f"Average elimination rate (Total): {avg_elim_rate_t}, Average formation rate (Total): {avg_form_rate_t}")
    
    # Check suitability
    apical_suitable = avg_elim_rate_a > avg_form_rate_a if len(avg_elim_rate_a) > 0 else False
    total_suitable = avg_elim_rate_t > avg_form_rate_t if len(avg_elim_rate_t) > 0 else False
    
    if apical_suitable:
        debug_print("Elimination rate (Apical) is higher than formation rate (Apical)")
    if total_suitable:
        debug_print("Elimination rate (Total) is higher than formation rate (Total)")
        
    return apical_suitable or total_suitable


def parse_branch_activity_file(filepath):
    """Parse branch activity data from file"""
    data_4k = []
    data_12k = []
    
    debug_print(f"Parsing file {filepath}")
    if 'norm' in filepath:
        timepoint = 'D3 test'
    else:
        timepoint = 'baseline'
    
    if not os.path.exists(filepath):
        debug_print(f"File not found: {filepath}")
        return [], []
        
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
        debug_print(f"File has {len(lines)} lines")
        # Uncomment below for detailed line-by-line debugging
        # for j in range(min(5, len(lines))):
        #     debug_print(f"Sample line {j}: {lines[j].strip()}")
        
        i = 0
        found_baseline_4k = False
        found_baseline_12k = False
        found_d3test_4k = False
        found_d3test_12k = False

        while i < len(lines):
            line = lines[i].strip()
            if timepoint == 'baseline':
                if 'baseline 4k' in line and i + 1 < len(lines):
                    found_baseline_4k = True
                    branch_line = lines[i + 1].strip()
                    #if 'Branch activity:' in branch_line:
                    branch_data = parse_branch_values(lines[i+1])
                    if branch_data:
                        data_4k.append(branch_data)
                
                elif 'baseline 12k' in line and i + 1 < len(lines):
                    found_baseline_12k = True
                    branch_line = lines[i + 1].strip()
                    #if 'Branch activity:' in branch_line:
                    branch_data = parse_branch_values(lines[i+1])
                    if branch_data:
                        data_12k.append(branch_data)
            elif timepoint == 'D3 test':
                if 'D3 test 4k' in line and i + 1 < len(lines):
                    found_d3test_4k = True
                    branch_line = lines[i + 1].strip()
                    #if 'Branch activity:' in branch_line:
                    branch_data = parse_branch_values(lines[i+1])
                    if branch_data:
                        data_4k.append(branch_data)
        
                elif 'D3 test 12k' in line and i + 1 < len(lines):
                    found_d3test_12k = True
                    branch_line = lines[i + 1].strip()
                    #if 'Branch activity:' in branch_line:
                    branch_data = parse_branch_values(lines[i+1])
                    if branch_data:
                        data_12k.append(branch_data)

            i += 1
    
    # Transpose data for easier analysis
    transposed_4k = [list(col) for col in zip(*data_4k)] if data_4k else []
    transposed_12k = [list(col) for col in zip(*data_12k)] if data_12k else []

    debug_print(f"Found 4k baseline: {found_baseline_4k}, records: {len(data_4k)}")
    debug_print(f"Found 12k baseline: {found_baseline_12k}, records: {len(data_12k)}")
    debug_print(f"Found 4k test: {found_d3test_4k}, records: {len(data_4k)}")
    debug_print(f"Found 12k test: {found_d3test_12k}, records: {len(data_12k)}")
    
    return transposed_4k, transposed_12k

def parse_branch_values(line):
    """Extract numerical values from a branch activity line"""
    # Parse values after colons in format like "x y: value;"
    values = []
    parts = line.split(';')
    
    for part in parts:
        part = part.strip()
        if part:
            match = re.search(r':\s*([-\d.]+)', part)
            if match:
                values.append(float(match.group(1)))
    
    return values if values else None

def analyze_branch_activity(data_dir, norm_dir, seed, max_diff=20):
    """Analyze branch activity data and determine if seed is suitable"""
    # Get all branch activity files
    branch_baseline_files = [os.path.join(data_dir, f'Branch_activity_node_{i}_{seed}.txt') for i in range(20)]
    branch_d3test_files = [os.path.join(norm_dir, f'Branch_activity_node_{i}.txt') for i in range(20)]
    
    debug_print(f"Found {len(branch_baseline_files)} branch activity files in {data_dir}")
    debug_print(f"Found {len(branch_d3test_files)} branch activity files in {norm_dir}")
    
    # Show sample files for debugging
    if DEBUG:
        for i, file in enumerate(branch_baseline_files[:3]):
            debug_print(f"Sample baseline file {i}: {file}")
        for i, file in enumerate(branch_d3test_files[:3]):
            debug_print(f"Sample d3test file {i}: {file}")
    
    if not branch_baseline_files or not branch_d3test_files:
        print(f"No branch activity files found in {data_dir} or {norm_dir}")
        return False
    
    # Collect all data
    all_4k_data = []
    all_12k_data = []
    all_d3test_4k_data = []
    all_d3test_12k_data = []
    
    for file in branch_baseline_files:
        baseline_4k_data, baseline_12k_data = parse_branch_activity_file(file)
        all_4k_data.extend(baseline_4k_data)
        all_12k_data.extend(baseline_12k_data)
    for file in branch_d3test_files:
        d3test_4k_data, d3test_12k_data = parse_branch_activity_file(file)
        all_d3test_4k_data.extend(d3test_4k_data)
        all_d3test_12k_data.extend(d3test_12k_data)
    
    debug_print(f"Total baseline 4k data records: {len(all_4k_data)}")
    debug_print(f"Total baseline 12k data records: {len(all_12k_data)}")
    debug_print(f"Total d3test 4k data records: {len(all_d3test_4k_data)}")
    debug_print(f"Total d3test 12k data records: {len(all_d3test_12k_data)}")

    if not all_4k_data or not all_12k_data or not all_d3test_4k_data or not all_d3test_12k_data:
        print("Insufficient data collected")
        return False
    

    stim_4k_arr = []
    stim_12k_arr = []
    stim_d3test_4k_arr = []
    stim_d3test_12k_arr = []
    for num in range(len(all_4k_data)):
        stim_4k_arr.append([])
        stim_12k_arr.append([])
        log_4k = False
        for i in range(len(all_4k_data[num])):
            diff = abs(all_4k_data[num][i] - (all_4k_data[num][i-1] if i > 0 else all_4k_data[num][i]))
            if diff > 20:
                log_4k = not log_4k
            if log_4k:
                stim_4k_arr[num].append(all_4k_data[num][i])
        log_12k = False
        for i in range(len(all_12k_data[num])):
            diff = abs(all_12k_data[num][i] - (all_12k_data[num][i-1] if i > 0 else all_12k_data[num][i]))
            if diff > 20:
                log_12k = not log_12k
            if log_12k:
                stim_12k_arr[num].append(all_12k_data[num][i])
        #stim_4k_arr[num] = all_4k_data[num][251:]
        #stim_12k_arr[num] = all_12k_data[num][251:]
    for num in range(min(len(all_d3test_4k_data), len(all_d3test_12k_data))):
        stim_d3test_4k_arr.append([])
        stim_d3test_12k_arr.append([])
        log_d3_4k = False
        for i in range(len(all_d3test_4k_data[num])):
            diff = abs(all_d3test_4k_data[num][i] - (all_d3test_4k_data[num][i-1] if i > 0 else all_d3test_4k_data[num][i]))
            if diff > 20:
                log_d3_4k = not log_d3_4k
            if log_d3_4k:
                stim_d3test_4k_arr[num].append(all_d3test_4k_data[num][i])
        log_d3_12k = False
        for i in range(len(all_d3test_12k_data[num])):
            diff = abs(all_d3test_12k_data[num][i] - (all_d3test_12k_data[num][i-1] if i > 0 else all_d3test_12k_data[num][i]))
            if diff > 20:
                log_d3_12k = not log_d3_12k
            if log_d3_12k:
                stim_d3test_12k_arr[num].append(all_d3test_12k_data[num][i])
        #stim_d3test_4k_arr[num] = all_d3test_4k_data[num][251:]
        #stim_d3test_12k_arr[num] = all_d3test_12k_data[num][251:]

    # Convert to numpy array
    all_4k_array = np.array(stim_4k_arr)
    all_12k_array = np.array(stim_12k_arr)
    all_4k_d3test_array = np.array(stim_d3test_4k_arr)
    all_12k_d3test_array = np.array(stim_d3test_12k_arr)
    
    debug_print(f"baseline 4k array shape: {all_4k_array.shape}")
    debug_print(f"baseline 12k array shape: {all_12k_array.shape}")
    debug_print(f"d3 4k array shape: {all_4k_d3test_array.shape}")
    debug_print(f"d3 12k array shape: {all_12k_d3test_array.shape}")

    # Calculate p-values and mean differences
    base_tuned_4k = 0
    base_tuned_12k = 0

    rowvec_length = min(all_4k_array.shape[1], all_12k_array.shape[1])
    debug_print(f"Starting from row {all_4k_array.shape[1] - rowvec_length}")

    
    for row in range(all_4k_array.shape[0]):
        # Perform t-test
        t_stat, p_val = stats.ttest_ind(all_4k_array[row, -rowvec_length:], all_12k_array[row, -rowvec_length:])
        # Calculate means from the bottom of the array up to bottom-colvec_length
        mean_4k = np.mean(all_4k_array[row, -rowvec_length:])
        mean_12k = np.mean(all_12k_array[row, -rowvec_length:])
        mean_diff = mean_4k - mean_12k
        
        # Count tuned branches
        if p_val < 0.05:
            if mean_diff > 0:  # 4k tunes
                debug_print(f"Row {row} tuned by 4k with p-value {p_val:.4f}, mean diff {mean_diff:.4f}")
                base_tuned_4k += 1
            elif mean_diff < 0:  # 12k tunes
                base_tuned_12k += 1
                debug_print(f"Row {row} tuned by 12k with p-value {p_val:.4f}, mean diff {mean_diff:.4f}")
        
    debug_print(f"baseline: Total tuned by 4k: {base_tuned_4k}, tuned by 12k: {base_tuned_12k}")
    

    # Calculate difference
    diff_baseline = abs(base_tuned_4k - base_tuned_12k)
    
    # Print results
    print(f"Branches tuned by 4k: {base_tuned_4k}")
    print(f"Branches tuned by 12k: {base_tuned_12k}")
    print(f"Difference: {diff_baseline}")
    ###---------------------------------------------------------------------------------------------------###
    d3test_tuned_4k = 0
    d3test_tuned_12k = 0
    rowvec_d3_length = min(all_4k_d3test_array.shape[1], all_12k_d3test_array.shape[1])
    debug_print(f"Starting d3test analysis from row {all_4k_d3test_array.shape[1] - rowvec_d3_length}")

    for row in range(all_4k_d3test_array.shape[0]):
        # Perform t-test
        t_stat, p_val = stats.ttest_ind(all_4k_d3test_array[row, -rowvec_length:], all_12k_d3test_array[row, -rowvec_length:])
        # Calculate means from the bottom of the array up to bottom-colvec_length
        mean_4k = np.mean(all_4k_d3test_array[row, -rowvec_length:])
        mean_12k = np.mean(all_12k_d3test_array[row, -rowvec_length:])
        mean_diff = mean_4k - mean_12k
        
        # Count tuned branches
        if p_val < 0.05:
            if mean_diff > 0:  # 4k tunes
                debug_print(f"d3: Row {row} tuned by 4k with p-value {p_val:.4f}, mean diff {mean_diff:.4f}")
                d3test_tuned_4k += 1
            elif mean_diff < 0:  # 12k tunes
                d3test_tuned_12k += 1
                debug_print(f"d3: Row {row} tuned by 12k with p-value {p_val:.4f}, mean diff {mean_diff:.4f}")
        
    debug_print(f"d3: Total tuned by 4k: {d3test_tuned_4k}, tuned by 12k: {d3test_tuned_12k}")
    

    # Calculate difference
    incre_4k = d3test_tuned_4k - base_tuned_4k
    decre_12k = base_tuned_12k - d3test_tuned_12k
    # Print results
    print(f"D3 Branches tuned by 4k: {d3test_tuned_4k}")
    print(f"D3 Branches tuned by 12k: {d3test_tuned_12k}")

    print(f"Branches tuned by 4k increased by: {incre_4k}")
    print(f"Branches tuned by 12k decreased by: {decre_12k}")
    ###---------------------------------------------------------------------------------------------------###

    # Determine if seed is suitable
    is_suitable = (diff_baseline <= max_diff) and (incre_4k > 2) and (decre_12k > 2)
    debug_print(f"Seed suitability check: {is_suitable} (max_diff={max_diff})")
    
    if is_suitable:
        print("This seed is suitable!")
        return True
    else:
        print("This seed is not suitable.")
        return False

if __name__ == "__main__":
    # Input arguments
    if len(sys.argv) < 2:
        print("Usage: python search.py <data_dir> <norm_dir>")
        print("  data_dir: Directory containing baseline simulation data")
        print("  norm_dir: Directory containing D3 test simulation data")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    norm_dir = sys.argv[2]
    seed = os.environ.get('NEURON_SEED')
    
    if not seed:
        print("Error: NEURON_SEED environment variable not set")
        sys.exit(1)
    
    print(f"Analyzing seed {seed}...")
    print(f"Data directory: {data_dir}")
    print(f"Norm directory: {norm_dir}")
    
    # Optional: Analyze spine change rate
    # spine_change_suitable = spine_change_rate(data_dir, seed)
    # if not spine_change_suitable:
    #     print("Spine change rate is not suitable")
    #     sys.exit(1)
    
    # Analyze branch activity
    is_suitable = analyze_branch_activity(data_dir, norm_dir, seed)
    
    if is_suitable:
        print(f"DEBUG: Seed {seed} is suitable")
    else:
        print(f"Seed {seed} is not suitable for experiments")
        sys.exit(1)  # Failure 