"""
Helper utility functions for neural data analysis.
"""

import os
import json
import numpy as np

# Define groups for regions of interest
region_groups = {
    # precentral regions
    "precentral-rh": ["ctx-rh-precentral", "wm-rh-precentral"],
    "precentral-lh": ["ctx-lh-precentral", "wm-lh-precentral"],
    # postcentral regions
    "postcentral-rh": ["ctx-rh-postcentral", "wm-rh-postcentral"],
    "postcentral-lh": ["ctx-lh-postcentral", "wm-lh-postcentral"],
    # supramarginal regions
    "supramarginal-rh": ["ctx-rh-supramarginal", "wm-rh-supramarginal"],
    "supramarginal-lh": ["ctx-lh-supramarginal", "wm-lh-supramarginal"],
    # superior parietal regions
    "superiorparietal-rh": ["ctx-rh-superiorparietal", "wm-rh-superiorparietal"],
    "superiorparietal-lh": ["ctx-lh-superiorparietal", "wm-lh-superiorparietal"],
    # superior frontal regions
    "superiorfrontal-rh": ["ctx-rh-superiorfrontal", "wm-rh-superiorfrontal"],
    "superiorfrontal-lh": ["ctx-lh-superiorfrontal", "wm-lh-superiorfrontal"],
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