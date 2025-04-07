"""
Functions for extracting region-specific epochs.
"""

import numpy as np
import matplotlib.pyplot as plt
from .region_extractor import get_brain_regions, create_region_mapping, brain_region_select

def get_channels_for_regions(epochs, good_channels, labels, chn_data, region_labels):
    """
    Get channel names for specific brain regions.
    
    Parameters:
    -----------
    epochs : mne.Epochs
        Epochs object containing the data
    good_channels : array
        Array of good channel indices
    labels : array
        Array of brain region labels
    chn_data : array
        Array of channel indices
    region_labels : list
        List of brain region names to find channels for
    
    Returns:
    --------
    unique_region_channels : list
        List of channel names for the specified regions
    """
    # Get brain regions for this subject
    brain_regions = get_brain_regions(good_channels, labels, chn_data)
    
    # Create mapping of standardized region names
    unique_regions = create_region_mapping(brain_regions)
    
    # Collect channels for the specified regions
    region_channels = []
    
    for region_label in region_labels:
        # First, check if the exact region name exists
        if region_label in unique_regions:
            # Find channels for all variants of this standardized region
            for variant in unique_regions[region_label]:
                region_channels.extend(brain_region_select(epochs, good_channels, labels, chn_data, variant))
        else:
            # If not, look through all region variants to find a match
            for region, variants in unique_regions.items():
                if region_label in variants:
                    # Find channels for all variants of this standardized region
                    for variant in variants:
                        region_channels.extend(brain_region_select(epochs, good_channels, labels, chn_data, variant))
                    break
    
    # Remove duplicates while preserving order
    unique_region_channels = []
    [unique_region_channels.append(ch) for ch in region_channels if ch not in unique_region_channels]

    # Sort the channels in numerical order
    unique_region_channels.sort()
    
    return unique_region_channels

def extract_region_specific_epochs(region_labels, subject_id_list, sampling_frequency, mapping_events, 
                                event_dict_gest, trigger_type, tmin, tmax):
    """
    Extract epochs data for specific brain regions across multiple subjects.
    
    Parameters:
    -----------
    region_labels : list
        List of brain region names to extract data for
    subject_id_list : list
        List of subject IDs to analyze
    sampling_frequency : int
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
    
    Returns:
    --------
    region_epochs : dict
        Dictionary mapping subject IDs to region-specific epochs objects
    region_channels_dict : dict
        Dictionary mapping subject IDs to lists of channel names
    """
    # Import here to avoid circular imports
    from data_loading import load_subject_data
    
    # Initialize results dictionaries
    region_epochs = {}
    region_channels_dict = {}
    
    # Loop through each subject
    for subject_id in subject_id_list:
        print(f"\nProcessing subject {subject_id}...")
        
        try:
            # Load subject data
            epochs, good_channels, names, labels, chn_data = load_subject_data(
                subject_id, 
                sampling_frequency, 
                mapping_events, 
                event_dict_gest, 
                trigger_type, 
                tmin, 
                tmax, 
                baseline=None, 
                plot=False
            )
            
            # Get channels for the specified regions
            region_channels = get_channels_for_regions(
                epochs, 
                good_channels, 
                labels, 
                chn_data, 
                region_labels
            )
            
            # If no channels found for the specified regions, skip this subject
            if len(region_channels) == 0:
                print(f"No channels found for the specified regions in subject {subject_id}. Skipping...")
                continue
            
            print(f"Found {len(region_channels)} channels for the specified regions in subject {subject_id}:")
            print(region_channels)
            
            # Create a new epochs object with only the channels from the specified regions
            region_epochs[subject_id] = epochs.copy().pick_channels(region_channels)
            region_channels_dict[subject_id] = region_channels
            
            print(f"Successfully extracted region-specific epochs for subject {subject_id}")
            
        except Exception as e:
            print(f"Error processing subject {subject_id}: {e}")
    
    # Check if any subjects were successfully processed
    if len(region_epochs) == 0:
        print("No subjects had channels in the specified regions.")
    else:
        print(f"\nSuccessfully extracted region-specific epochs for {len(region_epochs)} subjects:")
        for subject_id in region_epochs:
            print(f"  Subject {subject_id}: {len(region_channels_dict[subject_id])} channels")
    
    return region_epochs, region_channels_dict

def plot_region_specific_epochs(region_epochs, region_channels_dict, event_id, title=None):
    """
    Plot the region-specific epochs data for all subjects.
    
    Parameters:
    -----------
    region_epochs : dict
        Dictionary mapping subject IDs to region-specific epochs objects
    region_channels_dict : dict
        Dictionary mapping subject IDs to lists of channel names
    event_id : dict
        Dictionary mapping event labels to trigger values
    title : str, optional
        Optional title for the plots
    """
    for subject_id, epochs in region_epochs.items():
        num_channels = len(region_channels_dict[subject_id])
        
        # Create a title for the plot
        plot_title = f"Subject {subject_id} - Regional Epochs" if title is None else f"{title} - Subject {subject_id}"
        
        # Plot all channels if there are few, otherwise plot 8 at a time
        n_channels = min(8, num_channels)
        
        # Plot the epochs
        epochs.plot(
            n_channels=n_channels,
            scalings={"seeg": 5e2},
            title=plot_title,
            event_id=event_id,
            event_color=dict(elbow="red", scissor="blue", rock="black", rotation="green", thumb="yellow"),
            show=True,
            block=True
        )
