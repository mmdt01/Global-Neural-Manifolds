"""
Helper utility functions for neural data analysis.
"""

import os
import numpy as np

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
