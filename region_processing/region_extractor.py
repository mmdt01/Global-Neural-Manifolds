"""
Functions for identifying and extracting brain regions.
"""

import numpy as np
from collections import defaultdict
from copy import deepcopy

def get_brain_regions(good_channels, labels, chn_data):
    """
    Return the brain regions that are present in the data.
    
    Parameters:
    -----------
    good_channels : array
        Array of good channel indices
    labels : array
        Array of anatomical labels
    chn_data : array
        Array of channel indices
    
    Returns:
    --------
    regions : array
        Array of unique brain region labels
    """
    # Find indices in labels that are contained in good_channels
    good = np.where(np.isin(chn_data, good_channels))
    labels_subset = labels[good]
    
    # Find unique labels
    regions = np.unique(labels_subset)
    return regions

def brain_region_select(epochs, good_channels, labels, chn_data, region_label):
    """
    Select the channels of a specific brain region.
    
    Parameters:
    -----------
    epochs : mne.Epochs
        Epochs object containing the data
    good_channels : array
        Array of good channel indices
    labels : array
        Array of anatomical labels
    chn_data : array
        Array of channel indices
    region_label : str
        Brain region label to select
    
    Returns:
    --------
    region_ch_names : list
        List of channel names for the specified region
    """
    # Find the indices of region_label in labels
    indices = np.where(labels == region_label)
    
    # Find indices of chn_data that correspond to good_channels
    good = np.where(np.isin(chn_data, good_channels))
    
    # Find the intersection of indices and good
    if np.all(np.isin(indices, good)):
        # All region indices are good
        data_indices = chn_data[indices]
    else:
        # Only some region indices are good
        g_indices = np.intersect1d(indices, good)
        data_indices = chn_data[g_indices]
    
    # Find indices in good_channels of data_indices
    channel_indices = np.where(np.isin(good_channels, data_indices))[0]
    
    # Get channel names
    region_ch_names = [epochs.ch_names[i] for i in channel_indices]
    return region_ch_names

def create_region_mapping(regions):
    """
    Creates a dictionary mapping standardized brain region names to their variant labels.
    
    Parameters:
    -----------
    regions : array
        Array of brain region labels
    
    Returns:
    --------
    unique_regions : dict
        Dictionary mapping standardized names to lists of variant labels
    """
    # Initialize defaultdict to collect all variants
    region_variants = defaultdict(list)
    
    for region in regions:
        # Standardize the label by removing double hyphens
        standardized = region.replace('--', '-')
        # Add the original label to the list of variants for this standardized name
        region_variants[standardized].append(region)
    
    # Convert defaultdict to regular dict and sort by keys
    unique_regions = dict(sorted(region_variants.items()))
    return unique_regions

def create_region_channel_mapping(epochs, good_channels, labels, chn_data, unique_regions):
    """
    Creates dictionaries mapping standardized brain region names to lists of channel names and counts.
    
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
    unique_regions : dict
        Dictionary mapping standardized names to lists of variant labels
    
    Returns:
    --------
    unique_region_channels : dict
        Dictionary mapping standardized names to lists of channel names
    unique_region_channel_nums : dict
        Dictionary mapping standardized names to number of channels
    """
    # Initialize defaultdict to collect all channels
    unique_region_channels = defaultdict(list)
    
    # Find the channels contained in each unique region
    for region, variants in unique_regions.items():
        region_channels = []
        # For each variant of the region label, find the channels
        for variant in variants:
            region_channels.extend(brain_region_select(epochs, good_channels, labels, chn_data, variant))
        # Add the channels to the dictionary
        unique_region_channels[region] = region_channels

    # Create new dictionary with values as the number of channels
    unique_region_channel_nums = {region: len(channels) for region, channels in unique_region_channels.items()}
    
    return unique_region_channels, unique_region_channel_nums