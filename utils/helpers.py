"""
Helper utility functions for neural data analysis.
"""

import os
import numpy as np

# Define groups for regions of interest
region_groups = {
    "precentral-rh": ["ctx-rh-precentral", "wm-rh-precentral"],
    # "postcentral-lh": ["ctx-lh-postcentral", "wm-lh-postcentral"],
    "postcentral-rh": ["ctx-rh-postcentral", "wm-rh-postcentral"],
    # "postcentral-lh": ["ctx-lh-postcentral", "wm-lh-postcentral"],
    "supramarginal-lh": ["ctx-lh-supramarginal", "wm-lh-supramarginal"],
    "superiorfrontal-lh": ["ctx-lh-superiorfrontal", "wm-lh-superiorfrontal"],
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
    # # If custom regions are provided, use them instead of region groups
    # if args.regions is not None:
    #     return {"Custom": args.regions}
    
    # If "all" is specified, include all region groups
    if "all" in args.regions:
        return region_groups
    
    # Otherwise, include only the specified region groups
    return {group: region_groups[group] for group in args.regions}