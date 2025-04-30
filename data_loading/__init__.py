"""
Data loading module for neural data analysis.
"""

import os
import json
import mne
import pickle
import time
from .mne_converter import mne_raw, mne_epochs
from .region_caching import get_cache_dir, generate_cache_key
from .data_loader import (
    read_data, 
    read_labels, 
    get_frequency_bands, 
    filter_raw_data_and_compute_power
)

def load_subject_data(subject_id, sampling_rate, mapping_events, event_dict_gest, trigger_type, 
                     tmin, tmax, baseline=None, plot=False, bands_to_process=None, downsample_factor=1):
    """
    Load the data for a single subject, filter into frequency bands, compute band power,
    and create epochs for the stimulation triggers.
    
    Parameters:
    -----------
    subject_id : int
        ID of the subject to load
    sampling_rate : int
        Sampling rate of the data in Hz
    mapping_events : dict
        Dictionary mapping trigger values to event labels
    event_dict_gest : dict
        Dictionary mapping gesture labels to trigger values
    trigger_type : str
        Type of trigger to use ('stim' or 'emg')
    tmin : float
        Start time for epochs in seconds, relative to events
    tmax : float
        End time for epochs in seconds, relative to events
    baseline : tuple or None
        Baseline correction period (start, end) in seconds
    plot : bool
        Whether to plot the epochs
    bands_to_process : list or None
        List of frequency band names to process. If None, all bands will be processed.
    downsample_factor : int, optional
        Factor by which to downsample the band power data
    
    Returns:
    --------
    band_epochs_dict : dict
        Dictionary mapping band names to mne.Epochs objects containing the processed data
    good_channels : array
        Array of good channel indices
    names : array
        Array of electrode names
    labels : array
        Array of anatomical labels
    chn_data : array
        Array of channel indices
    """
    # Load data and labels
    data, good_channels = read_data(subject_id)
    names, labels, chn_data = read_labels(subject_id)
    
    # Create raw MNE objects
    raw_stim, raw_emg, events_stim, events_emg = mne_raw(
        sampling_rate, mapping_events, data, good_channels
    )
    
    # Select the appropriate raw data and events based on trigger type
    if trigger_type == 'stim':
        raw = raw_stim
        events = events_stim
    elif trigger_type == 'emg':
        raw = raw_emg
        events = events_emg
    else:
        raise ValueError("Invalid trigger type. Must be 'stim' or 'emg'.")
    
    # Get the frequency bands to process
    all_bands = get_frequency_bands()
    if bands_to_process is None:
        bands_to_process = list(all_bands.keys())
    elif not all(band in all_bands for band in bands_to_process):
        invalid_bands = [band for band in bands_to_process if band not in all_bands]
        raise ValueError(f"Unknown band names: {invalid_bands}. Available bands: {list(all_bands.keys())}")
    
    # Get the raw data as numpy array
    raw_data = raw.get_data()
    sfreq = raw.info['sfreq']
    
    # Initialize dictionary to store epochs for each band
    band_epochs_dict = {}
    
    # Process each frequency band
    for band_name in bands_to_process:
        # Compute band power for the entire raw signal
        band_power_data = filter_raw_data_and_compute_power(
            raw_data, sfreq, band_name, downsample_factor=1  # No downsampling yet
        )
        
        # Create a new MNE Raw object with the band power data
        band_power_raw = mne.io.RawArray(band_power_data, raw.info.copy())
        
        # If downsampling is requested, use MNE's resample method
        if downsample_factor > 1:
            # Calculate the new sampling rate
            new_sfreq = sfreq / downsample_factor
            
            # Resample the Raw object properly
            band_power_raw = band_power_raw.resample(new_sfreq)
            
            # Adjust event sample numbers for the new sampling rate
            events_resampled = events.copy()
            events_resampled[:, 0] = (events_resampled[:, 0] * new_sfreq / sfreq).astype(int)
            events_to_use = events_resampled
        else:
            events_to_use = events
        
        # Create epochs from the band power data
        band_epochs = mne_epochs(
            band_power_raw, events_to_use, event_dict_gest, 
            tmin, tmax, baseline, plot
        )
        
        # Store in the dictionary
        band_epochs_dict[band_name] = band_epochs
    
    return band_epochs_dict, good_channels, names, labels, chn_data

def load_region_data(region_labels, subject_id_list, sampling_frequency, trigger_type, 
                   tmin, tmax, bands_to_process, base_dir='cache'):
    """
    Load region-specific data from cache if available.
    
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
    base_dir : str, optional
        Base directory for cache
    
    Returns:
    --------
    tuple or None
        (region_epochs, region_channels_dict) if cache exists, None otherwise
    """
    # Get cache directory and check if cache exists
    cache_dir = get_cache_dir(base_dir)
    cache_key = generate_cache_key(region_labels, subject_id_list, sampling_frequency, 
                                 trigger_type, tmin, tmax, bands_to_process)
    
    cache_entry_dir = os.path.join(cache_dir, cache_key)
    
    if not os.path.exists(cache_entry_dir):
        return None
    
    try:
        start_time = time.time()
        print(f"\nLoading region data from cache...")
        
        # Check metadata
        metadata_file = os.path.join(cache_entry_dir, 'metadata.json')
        if not os.path.exists(metadata_file):
            print(f"Metadata file not found in cache, cannot load")
            return None
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Load region_channels_dict
        channels_file = os.path.join(cache_entry_dir, 'region_channels_dict.pkl')
        if not os.path.exists(channels_file):
            print(f"Channels dictionary file not found in cache, cannot load")
            return None
        
        with open(channels_file, 'rb') as f:
            region_channels_dict = pickle.load(f)
        
        # Initialize region_epochs
        region_epochs = {}
        
        # Load epochs for each subject and band
        for subject_id in metadata['subjects_with_data']:
            subject_dir = os.path.join(cache_entry_dir, f'subject_{subject_id}')
            
            if not os.path.exists(subject_dir):
                print(f"Subject directory not found for subject {subject_id}, skipping")
                continue
            
            # Initialize band dictionary for this subject
            region_epochs[int(subject_id)] = {}
            
            # Load each band's epochs
            for band_name in metadata['frequency_bands'].get(str(subject_id), []):
                epochs_file = os.path.join(subject_dir, f'{band_name}-epo.fif')
                
                if not os.path.exists(epochs_file):
                    print(f"Epochs file not found for subject {subject_id}, band {band_name}, skipping")
                    continue
                
                # Load epochs
                from mne import read_epochs
                epochs = read_epochs(epochs_file)
                
                # Store in region_epochs
                region_epochs[int(subject_id)][band_name] = epochs
        
        elapsed_time = time.time() - start_time
        print(f"Region data loaded from cache in {elapsed_time:.2f} seconds")
        print(f"Cache location: {cache_entry_dir}")
        
        return region_epochs, region_channels_dict
    
    except Exception as e:
        print(f"Error loading cache: {e}")
        return None

def save_region_data(region_epochs, region_channels_dict, region_labels, subject_id_list, 
                    sampling_frequency, trigger_type, tmin, tmax, bands_to_process, 
                    base_dir='cache'):
    """
    Save region-specific data to cache.
    
    Parameters:
    -----------
    region_epochs : dict
        Nested dictionary mapping subject IDs to frequency bands to Epochs objects
    region_channels_dict : dict
        Dictionary mapping subject IDs to lists of channel names
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
    base_dir : str, optional
        Base directory for cache
    
    Returns:
    --------
    str
        Path to cache file
    """
    start_time = time.time()
    print(f"\nSaving region data to cache...")
    
    # Get cache directory and create unique key
    cache_dir = get_cache_dir(base_dir)
    cache_key = generate_cache_key(region_labels, subject_id_list, sampling_frequency, 
                                 trigger_type, tmin, tmax, bands_to_process)
    
    # Create subdirectory for this cache entry
    cache_entry_dir = os.path.join(cache_dir, cache_key)
    os.makedirs(cache_entry_dir, exist_ok=True)
    
    # Save metadata
    metadata = {
        'region_labels': region_labels,
        'subject_id_list': subject_id_list,
        'sampling_frequency': sampling_frequency,
        'trigger_type': trigger_type,
        'tmin': tmin,
        'tmax': tmax,
        'bands_to_process': bands_to_process,
        'subjects_with_data': list(region_epochs.keys()),
        'cache_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'frequency_bands': {}
    }
    
    # Track frequency bands for each subject
    for subject_id, band_dict in region_epochs.items():
        if subject_id not in metadata['frequency_bands']:
            metadata['frequency_bands'][subject_id] = []
        
        metadata['frequency_bands'][subject_id] = list(band_dict.keys())
    
    # Save metadata
    with open(os.path.join(cache_entry_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save region_channels_dict
    with open(os.path.join(cache_entry_dir, 'region_channels_dict.pkl'), 'wb') as f:
        pickle.dump(region_channels_dict, f)
    
    # Save epochs for each subject and band
    for subject_id, band_dict in region_epochs.items():
        # Create subject directory
        subject_dir = os.path.join(cache_entry_dir, f'subject_{subject_id}')
        os.makedirs(subject_dir, exist_ok=True)
        
        # Save each band's epochs
        for band_name, epochs in band_dict.items():
            epochs_file = os.path.join(subject_dir, f'{band_name}-epo.fif')
            epochs.save(epochs_file, overwrite=True)
    
    elapsed_time = time.time() - start_time
    print(f"Region data saved to cache in {elapsed_time:.2f} seconds")
    print(f"Cache location: {cache_entry_dir}")
    
    return cache_entry_dir
