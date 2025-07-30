#!/usr/bin/env python
import numpy as np
import os
import sys
import glob
import re
from scipy import stats

def parse_branch_activity_file(filepath):
    """Parse branch activity data from file"""
    baseline_4k_data = []
    baseline_12k_data = []
    
    print(f"DEBUG: Parsing file {filepath}")
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
        print(f"DEBUG: File has {len(lines)} lines")
        # Print a sample of the first few lines to understand format
        #for j in range(min(5, len(lines))):
        #    print(f"DEBUG: Sample line {j}: {lines[j].strip()}")
        
        i = 0
        found_4k = False
        found_12k = False
        
        while i < len(lines):
            line = lines[i].strip()
            
            if 'D3 test 4k' in line and i + 1 < len(lines):
                found_4k = True
                branch_line = lines[i + 1].strip()
                if 'Branch activity:' in branch_line:
                    branch_data = parse_branch_values(lines[i+2])
                    if branch_data:
                        baseline_4k_data.append(branch_data)
            
            elif 'D3 test 12k' in line and i + 1 < len(lines):
                found_12k = True
                branch_line = lines[i + 1].strip()
                if 'Branch activity:' in branch_line:
                    branch_data = parse_branch_values(lines[i+2])
                    if branch_data:
                        baseline_12k_data.append(branch_data)
            
            i += 1
    
    transposed_4k = [list(col) for col in zip(*baseline_4k_data)]
    transposed_12k = [list(col) for col in zip(*baseline_12k_data)]
    
    print(f"DEBUG: Found 4k baseline: {found_4k}, records: {len(baseline_4k_data)}")
    print(f"DEBUG: Found 12k baseline: {found_12k}, records: {len(baseline_12k_data)}")
    
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

def analyze_branch_activity(data_dir, max_diff=5):
    """Analyze branch activity data and determine if seed is suitable"""
    # Get all branch activity files
    branch_files = [os.path.join(data_dir, f'Branch_activity_node_{i}.txt') for i in range(1,20)]

    print(f"DEBUG: Found {len(branch_files)} branch activity files in {data_dir}")
    # Print a few of the files found to verify pattern matching
    for i, file in enumerate(branch_files[:3]):
        print(f"DEBUG: Sample file {i}: {file}")
    
    if not branch_files:
        print(f"No branch activity files found in {data_dir}")
        return False
    
    # Collect all data
    all_4k_data = []
    all_12k_data = []
    
    for file in branch_files:
        baseline_4k_data, baseline_12k_data = parse_branch_activity_file(file)
        all_4k_data.extend(baseline_4k_data)
        all_12k_data.extend(baseline_12k_data)
    
    # Convert to numpy arrays
    print(f"DEBUG: Total 4k data records: {len(all_4k_data)}")
    print(f"DEBUG: Total 12k data records: {len(all_12k_data)}")
    
    if not all_4k_data or not all_12k_data:
        print("Insufficient data collected")
        return False
    
    # Make sure all arrays have the same length
    min_length = min(min(len(row) for row in all_4k_data), min(len(row) for row in all_12k_data))
    print(f"DEBUG: Minimum row length: {min_length}")
    
    #all_4k_data = [row[:min_length] for row in all_4k_data]
    #all_12k_data = [row[:min_length] for row in all_12k_data]
    test_4k_arr = []
    test_12k_arr = []
    for num in range(len(all_4k_data)):
        test_4k_arr.append([])
        test_12k_arr.append([])
        test_4k_arr[num] = all_4k_data[num][251:]
        test_12k_arr[num] = all_12k_data[num][251:]
    # Convert to numpy array
    all_4k_array = np.array(test_4k_arr)
    all_12k_array = np.array(test_12k_arr)
    
    print(f"DEBUG: 4k array shape: {all_4k_array.shape}")
    print(f"DEBUG: 12k array shape: {all_12k_array.shape}")
    
    # Calculate p-values and mean differences
    tuned_4k = 0
    tuned_12k = 0
    rowvec_length = min(all_4k_array.shape[1], all_12k_array.shape[1])
    print(f"DEBUG: Starting from {all_4k_array.shape[1] - rowvec_length} rows")
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
                print(f"DEBUG: Row {row} tuned by 4k with p-value {p_val:.4f}, mean diff {mean_diff:.4f}")
                tuned_4k += 1
            elif mean_diff < 0:  # 12k tunes
                tuned_12k += 1
                print(f"DEBUG: Row {row} tuned by 12k with p-value {p_val:.4f}, mean diff {mean_diff:.4f}")
        
    print(f"DEBUG: Total tuned by 4k: {tuned_4k}, tuned by 12k: {tuned_12k}")
    
    # Calculate difference
    diff = abs(tuned_4k - tuned_12k)
    
    # Print results
    print(f"Branches tuned by 4k: {tuned_4k}")
    print(f"Branches tuned by 12k: {tuned_12k}")
    print(f"Difference: {diff}")
    
    # Determine if seed is suitable
    '''is_suitable = (diff <= max_diff)
    if is_suitable:
        print("This seed is suitable!")
        return True
    else:
        print("This seed is not suitable.")
        return False'''

def increment_seed_file(seed_file):
    """Increment the seed value in the file"""
    try:
        with open(seed_file, 'r') as f:
            current_seed = int(f.read().strip())
        
        new_seed = current_seed + 1
        
        with open(seed_file, 'w') as f:
            f.write(str(new_seed))
        
        print(f"Incremented seed to {new_seed}")
        return new_seed
    except Exception as e:
        print(f"Error incrementing seed: {e}")
        return None

if __name__ == "__main__":
    # Input arguments
    if len(sys.argv) < 1:
        print("Usage: python search.py <data_dir>")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    #seed_file = sys.argv[2] if len(sys.argv) > 2 else "current_seed.txt"
    
    # Analyze branch activity
    is_suitable = analyze_branch_activity(data_dir)
    
    # Set exit code based on suitability
    if is_suitable:
        sys.exit(0)  # Success
    else:
        # Increment seed for next iteration
        #increment_seed_file(seed_file)
        sys.exit(1)  # Failure (need to try again) 