"""
Helper utility functions for neural data analysis.
"""

import os
import json

# Define groups for regions of interest
region_groups = {
    # precentral regions
    "precentral-rh": ["ctx-rh-precentral", "wm-rh-precentral"],
    "precentral-lh": ["ctx-lh-precentral", "wm-lh-precentral"],
    # postcentral regions
    "postcentral-rh": ["ctx-rh-postcentral", "wm-rh-postcentral"],
    "postcentral-lh": ["ctx-lh-postcentral", "wm-lh-postcentral"],
    # superior parietal regions
    "superiorparietal-rh": ["ctx-rh-superiorparietal", "wm-rh-superiorparietal"],
    "superiorparietal-lh": ["ctx-lh-superiorparietal", "wm-lh-superiorparietal"],
    # supramarginal regions
    "supramarginal-rh": ["ctx-rh-supramarginal", "wm-rh-supramarginal"],
    "supramarginal-lh": ["ctx-lh-supramarginal", "wm-lh-supramarginal"],
    # caudal middle frontal regions
    "caudalmiddlefrontal-rh": ["ctx-rh-caudalmiddlefrontal", "wm-rh-caudalmiddlefrontal"],
    "caudalmiddlefrontal-lh": ["ctx-lh-caudalmiddlefrontal", "wm-lh-caudalmiddlefrontal"],
    # rostral middle frontal regions
    "rostralmiddlefrontal-rh": ["ctx-rh-rostralmiddlefrontal", "wm-rh-rostralmiddlefrontal"],
    "rostralmiddlefrontal-lh": ["ctx-lh-rostralmiddlefrontal", "wm-lh-rostralmiddlefrontal"],
    # superior frontal regions
    "superiorfrontal-rh": ["ctx-rh-superiorfrontal", "wm-rh-superiorfrontal"],
    "superiorfrontal-lh": ["ctx-lh-superiorfrontal", "wm-lh-superiorfrontal"],
    # pars opercularis regions
    "parsopercularis-rh": ["ctx-rh-parsopercularis", "wm-rh-parsopercularis"],
    "parsopercularis-lh": ["ctx-lh-parsopercularis", "wm-lh-parsopercularis"],
    # precuneus regions
    "precuneus-rh": ["ctx-rh-precuneus", "wm-rh-precuneus"],
    "precuneus-lh": ["ctx-lh-precuneus", "wm-lh-precuneus"],
    # insula regions
    "insula-rh": ["ctx-rh-insula", "wm-rh-insula"],
    "insula-lh": ["ctx-lh-insula", "wm-lh-insula"],
}

def ensure_dir(directory):
    """
    Create directory if it doesn't exist.
    
    Parameters:
    -----------
    directory : str
        Directory path to create
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def standardize_region_name(region_name):
    """
    Standardize a brain region name by removing extra hyphens.
    
    Parameters:
    -----------
    region_name : str
        Brain region name to standardize
    
    Returns:
    --------
    str
        Standardized brain region name
    """
    return region_name.replace('--', '-')

def get_standard_gesture_colors():
    """
    Return a dictionary of standard colors for gestures.
    
    Returns:
    --------
    dict
        Dictionary mapping gesture names to color strings
    """
    return {
        "elbow": "red",
        "scissor": "blue",
        "rock": "green",
        "rotation": "purple",
        "thumb": "orange"
    }

def get_event_color_dict():
    """
    Return a dictionary of colors for event plotting.
    
    Returns:
    --------
    dict
        Dictionary mapping event names to color strings
    """
    return dict(elbow="red", scissor="blue", rock="black", rotation="green", thumb="yellow")

def get_region_lists(args):
    """Get the list of region lists to analyze based on arguments."""
    # If "all" is specified, include all region groups
    if "all" in args.regions:
        return region_groups
    
    # Otherwise, include only the specified region groups
    return {group: region_groups[group] for group in args.regions}

def load_region_subject_mapping(mapping_file, default_subjects):
    """
    Load and validate the JSON file mapping regions to specific subject IDs.
    
    Parameters:
    -----------
    mapping_file : str
        Path to JSON file containing the mapping
    default_subjects : list
        Default list of subjects to use if no mapping is provided
        
    Returns:
    --------
    dict
        Dictionary mapping region names to lists of subject IDs
    """
    if mapping_file is None or not os.path.exists(mapping_file):
        print("No region-subject mapping file provided or file doesn't exist.")
        return {}
        
    try:
        with open(mapping_file, 'r') as f:
            mapping = json.load(f)
        
        # Validate the mapping
        for region, subjects in mapping.items():
            if not isinstance(subjects, list):
                print(f"Warning: Subjects for region {region} should be a list. Using default subjects.")
                mapping[region] = default_subjects
            else:
                # Ensure all subjects are integers
                mapping[region] = [int(subject) for subject in subjects]
        
        return mapping
    except Exception as e:
        print(f"Error parsing region-subject mapping file: {e}")
        print("Using default subject list for all regions.")
        return {}
    
def combine_regions(all_region_epochs, all_region_channels_dict):
    """
    Combine epochs and channel information from different brain regions for each subject.
    
    Parameters:
    -----------
    all_region_epochs : dict
        Dictionary mapping region group names to dictionaries of subject-specific epochs
    all_region_channels_dict : dict
        Dictionary mapping region group names to dictionaries of subject-specific channel lists
    
    Returns:
    --------
    brain_wide_epochs : dict
        Dictionary mapping subject IDs to dictionaries of combined epochs for each frequency band
    brain_wide_channels_dict : dict
        Dictionary mapping subject IDs to combined channel lists
    """
    import mne
    from copy import deepcopy
    
    # Initialize dictionaries to store combined data
    brain_wide_epochs = {}
    brain_wide_channels_dict = {}
    
    # Get list of all subjects across all regions
    all_subjects = set()
    for region_epochs in all_region_epochs.values():
        all_subjects.update(region_epochs.keys())
    
    print(f"Combining data for {len(all_subjects)} subjects across {len(all_region_epochs)} region groups...")
    
    # Process each subject
    for subject_id in all_subjects:
        # Track which regions have data for this subject
        regions_with_data = []
        for region_name, region_epochs in all_region_epochs.items():
            if subject_id in region_epochs and len(region_epochs[subject_id]) > 0:
                regions_with_data.append(region_name)
        
        if len(regions_with_data) == 0:
            print(f"No valid data found for Subject {subject_id}, skipping...")
            continue
        
        print(f"\nCombining data for Subject {subject_id} from {len(regions_with_data)} regions: {', '.join(regions_with_data)}")
        
        # Initialize subject-specific dictionaries
        brain_wide_epochs[subject_id] = {}
        brain_wide_channels_dict[subject_id] = []
        
        # Get available frequency bands (assuming they're consistent across regions)
        first_region = regions_with_data[0]
        try:
            available_bands = list(all_region_epochs[first_region][subject_id].keys())
        except KeyError:
            print(f"No frequency bands found for Subject {subject_id} in region {first_region}, skipping...")
            continue
        
        print(f"Available frequency bands: {', '.join(available_bands)}")
        
        # Process each frequency band
        for band_name in available_bands:
            print(f"Processing {band_name} band...")
            
            # Collect info needed for the combined epochs
            combined_data = []
            combined_events = None
            combined_event_id = None
            combined_metadata = None
            times = None
            sfreq = None
            ch_names = []
            
            # Add channel information and data from each region
            for region_name in regions_with_data:
                if subject_id not in all_region_epochs[region_name]:
                    print(f"Subject {subject_id} not found in region {region_name}, skipping...")
                    continue
                
                if band_name not in all_region_epochs[region_name][subject_id]:
                    print(f"Band {band_name} not found for Subject {subject_id} in region {region_name}, skipping...")
                    continue
                
                # Get the epochs for this region
                region_epoch = all_region_epochs[region_name][subject_id][band_name]
                
                # If this is our first region, initialize combined events, event_id, etc.
                if combined_events is None:
                    combined_events = deepcopy(region_epoch.events)
                    combined_event_id = deepcopy(region_epoch.event_id)
                    times = region_epoch.times
                    sfreq = region_epoch.info['sfreq']
                    
                    # Initialize metadata if available
                    if hasattr(region_epoch, 'metadata') and region_epoch.metadata is not None:
                        combined_metadata = region_epoch.metadata.copy()
                
                # Get channel names from this region
                region_ch_names = all_region_channels_dict[region_name][subject_id]
                
                # Modify channel names to avoid duplicates and include region info
                region_prefix = region_name.replace('-', '_')  # Replace hyphen with underscore for valid channel names
                renamed_ch_names = [f"{ch}_{region_prefix}" for ch in region_ch_names]
                
                # Add to the combined channel list
                ch_names.extend(renamed_ch_names)
                
                # Add data from this region to the combined data
                # First, check dimensions
                region_data = region_epoch.get_data()
                print(f"  {region_name}: {len(region_ch_names)} channels, {region_data.shape[0]} epochs, {region_data.shape[2]} time points")
                
                # Append to combined data
                combined_data.append(region_data)
            
            # Ensure we have data
            if not combined_data:
                print(f"No data to combine for Subject {subject_id}, band {band_name}, skipping...")
                continue
            
            # Concatenate data along the channel dimension
            import numpy as np
            try:
                # Ensure all data arrays have same number of epochs and time points
                epoch_counts = [data.shape[0] for data in combined_data]
                time_points = [data.shape[2] for data in combined_data]
                
                if len(set(epoch_counts)) > 1 or len(set(time_points)) > 1:
                    print(f"Warning: Inconsistent data dimensions for Subject {subject_id}, band {band_name}")
                    print(f"  Epoch counts: {epoch_counts}")
                    print(f"  Time points: {time_points}")
                    print(f"  Will attempt to use the minimum common dimensions")
                    
                    # Use minimum common dimensions
                    min_epochs = min(epoch_counts)
                    min_times = min(time_points)
                    
                    # Trim each data array
                    for i in range(len(combined_data)):
                        combined_data[i] = combined_data[i][:min_epochs, :, :min_times]
                
                # Now concatenate along the channel dimension
                all_data = np.concatenate(combined_data, axis=1)
                
                print(f"  Combined data shape: {all_data.shape} (epochs, channels, time points)")
                
                # Create a new info object with combined channels
                info = mne.create_info(ch_names, sfreq, ch_types=['eeg'] * len(ch_names))
                
                # Create combined epochs object
                combined_epochs = mne.EpochsArray(
                    all_data, 
                    info,
                    events=combined_events,
                    event_id=combined_event_id,
                    tmin=times[0],
                    metadata=combined_metadata
                )
                
                # Store the combined epochs
                brain_wide_epochs[subject_id][band_name] = combined_epochs
                
                # Store the combined channel list
                brain_wide_channels_dict[subject_id] = ch_names
                
                print(f"  Successfully created combined {band_name} epochs with {len(ch_names)} channels")
            
            except Exception as e:
                print(f"Error combining data for Subject {subject_id}, band {band_name}: {str(e)}")
                continue
    
    return brain_wide_epochs, brain_wide_channels_dict
