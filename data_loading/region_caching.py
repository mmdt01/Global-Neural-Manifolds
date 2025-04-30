"""
Module for caching and loading region-specific neural data to avoid redundant computations.
"""

import os
import json
import hashlib

def get_cache_dir(base_dir='cache'):
    """
    Get the cache directory, creating it if it doesn't exist.
    
    Parameters:
    -----------
    base_dir : str, optional
        Base directory for cache
    
    Returns:
    --------
    str
        Path to cache directory
    """
    cache_dir = os.path.join(base_dir, 'region_data')
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir

def generate_cache_key(region_labels, subject_id_list, sampling_frequency, trigger_type, 
                     tmin, tmax, bands_to_process):
    """
    Generate a unique cache key based on the analysis parameters.
    
    Parameters:
    -----------
    region_labels : list
        List of brain region names
    subject_id_list : list
        List of subject IDs
    sampling_frequency : int
        Sampling rate of the data
    trigger_type : str
        Type of trigger used ('stim' or 'emg')
    tmin : float
        Start time for epochs
    tmax : float
        End time for epochs
    bands_to_process : list or None
        List of frequency bands processed
    
    Returns:
    --------
    str
        Unique cache key as a hexadecimal string
    """
    # Create a string representation of the parameters
    param_dict = {
        'region_labels': sorted(region_labels),
        'subject_id_list': sorted(subject_id_list),
        'sampling_frequency': sampling_frequency,
        'trigger_type': trigger_type,
        'tmin': tmin,
        'tmax': tmax,
        'bands_to_process': sorted(bands_to_process) if bands_to_process else None
    }
    
    # Convert to JSON string
    param_str = json.dumps(param_dict, sort_keys=True)
    
    # Create hash of the parameter string
    hash_obj = hashlib.md5(param_str.encode())
    return hash_obj.hexdigest()


