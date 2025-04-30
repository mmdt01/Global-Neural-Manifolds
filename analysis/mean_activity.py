"""
Functions for analyzing mean delta activity across gestures and calculating
representational distances between movement types.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy.spatial.distance import mahalanobis

def compute_gesture_mean_activity(epochs_dict, gestures=None, band_name='delta'):
    """
    Compute the time-averaged activity for each trial/epoch, and then average across trials
    to get a centroid for each gesture in channel space.
    
    Parameters:
    -----------
    epochs_dict : dict or mne.Epochs
        Either a dictionary mapping band names to mne.Epochs objects,
        or a single mne.Epochs object
    gestures : list, optional
        List of gesture names to analyze. If None, uses all available gestures
    band_name : str, optional
        Name of the frequency band to use if epochs_dict is a dictionary
    
    Returns:
    --------
    gesture_centroids : dict
        Dictionary mapping gesture names to centroid vectors (one vector per gesture)
    trial_data : dict
        Dictionary mapping gesture names to arrays of trial data (trials x channels)
    """
    # Handle either band dictionary or direct epochs object
    if isinstance(epochs_dict, dict) and band_name in epochs_dict:
        # Extract epochs for the specified band
        epochs = epochs_dict[band_name]
    else:
        # If it's already an Epochs object, use it directly
        epochs = epochs_dict
    
    # Get list of gestures if not provided
    if gestures is None:
        gestures = list(epochs.event_id.keys())
    
    # Initialize dictionaries to store results
    gesture_centroids = {}
    trial_data = {}
    
    # Process each gesture
    for gesture in gestures:
        # Extract epochs for this gesture
        gesture_epochs = epochs[gesture]
        
        # If there are no epochs for this gesture, skip it
        if len(gesture_epochs) == 0:
            print(f"No epochs found for gesture: {gesture}, skipping...")
            continue
        
        # Get the data: shape is (trials, channels, time)
        data = gesture_epochs.get_data()
        
        # Compute the time-average for each trial: shape will be (trials, channels)
        # This is what we want - each trial is represented as a point in channel space
        time_averaged_trials = data.mean(axis=2)
        
        # Store the trial data for covariance calculation
        trial_data[gesture] = time_averaged_trials
        
        # Compute the centroid (average across trials)
        # Shape will be (channels,)
        centroid = time_averaged_trials.mean(axis=0)
        
        # Store the centroid
        gesture_centroids[gesture] = centroid
    
    return gesture_centroids, trial_data

def compute_mahalanobis_distance_matrix(gesture_centroids, trial_data):
    """
    Compute the pairwise Mahalanobis distance between gesture centroids.
    
    Parameters:
    -----------
    gesture_centroids : dict
        Dictionary mapping gesture names to centroid vectors (one vector per gesture)
    trial_data : dict
        Dictionary mapping gesture names to arrays of trial data (trials x channels)
    
    Returns:
    --------
    distance_matrix : np.ndarray
        Matrix of pairwise Mahalanobis distances
    gesture_labels : list
        List of gesture names corresponding to rows/columns of the distance matrix
    """
    # Get the list of gestures
    gesture_labels = list(gesture_centroids.keys())
    n_gestures = len(gesture_labels)
    
    # Initialize distance matrix
    distance_matrix = np.zeros((n_gestures, n_gestures))
    
    # Compute pooled covariance matrix from all trials across all gestures
    # First, concatenate all trial data
    all_trials = []
    for gesture in gesture_labels:
        all_trials.append(trial_data[gesture])
    
    # Concatenate along the trials dimension
    all_trials = np.vstack(all_trials)
    
    # Compute the covariance matrix
    # Note: rowvar=False because each row is an observation (trial)
    pooled_cov = np.cov(all_trials, rowvar=False)
    
    # Ensure covariance matrix is invertible
    # Add a small amount of regularization if needed
    try:
        inv_cov = np.linalg.inv(pooled_cov)
    except np.linalg.LinAlgError:
        print("Warning: Covariance matrix is singular. Adding regularization...")
        pooled_cov += np.eye(pooled_cov.shape[0]) * 1e-6
        inv_cov = np.linalg.inv(pooled_cov)
    
    # Compute pairwise Mahalanobis distances between centroids
    for i, gesture1 in enumerate(gesture_labels):
        for j, gesture2 in enumerate(gesture_labels):
            # Compute Mahalanobis distance between centroids
            distance = mahalanobis(
                gesture_centroids[gesture1],
                gesture_centroids[gesture2],
                inv_cov
            )
            distance_matrix[i, j] = distance
    
    return distance_matrix, gesture_labels

def visualize_distance_matrix(distance_matrix, gesture_labels, subject_id=None, region_label=None, output_dir=None):
    """
    Visualize the pairwise distance matrix between gestures.
   
    Parameters:
    -----------
    distance_matrix : np.ndarray
        Matrix of pairwise Mahalanobis distances
    gesture_labels : list
        List of gesture names corresponding to rows/columns of the distance matrix
    subject_id : str or int, optional
        Subject ID for plot title
    region_label : str, optional
        Brain region label for plot title
    output_dir : str, optional
        Directory to save plots. If None, plots are displayed but not saved.
   
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object containing the visualization
    """
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create a custom colormap: black -> red -> orange -> yellow
    colors = [(0, 0, 0),      # black (distance = 0)
              (0.7, 0, 0),    # dark red
              (1, 0, 0),      # bright red
              (1, 0.5, 0),    # orange
              (1, 1, 0)]      # yellow (largest distance)
    
    custom_cmap = LinearSegmentedColormap.from_list('black_to_yellow', colors)
   
    # Create a heatmap with the custom colormap
    sns.heatmap(distance_matrix, annot=True, fmt=".2f", cmap=custom_cmap,
                xticklabels=gesture_labels, yticklabels=gesture_labels, ax=ax)
   
    # Set title
    title = "Pairwise Mahalanobis Distance Matrix Between Gestures"
    if subject_id is not None and region_label is not None:
        # For the title, we can use the full region label
        title += f"\nSubject {subject_id}, Region: {region_label}"
    elif subject_id is not None:
        title += f"\nSubject {subject_id}"
    elif region_label is not None:
        title += f"\nRegion: {region_label}"
   
    ax.set_title(title)
   
    # Set labels
    ax.set_xlabel("Gesture")
    ax.set_ylabel("Gesture")
   
    # Adjust layout
    plt.tight_layout()
   
    # Save the plot if output directory is provided
    if output_dir is not None:
        output_file = "mahalanobis_distance_matrix"
        if subject_id is not None:
            output_file += f"_sub-{subject_id}"
        
        # For the filename, handle region label differently
        if region_label is not None:
            # Check if this is a brain-wide analysis with multiple regions
            if ',' in region_label and len(region_label) > 30:
                # We have multiple regions, use a shorter identifier
                region_count = len(region_label.split(','))
                output_file += f"_brain-wide_{region_count}_regions"
            else:
                # Single region or short list, use the original approach
                region_str = region_label.replace(', ', '_').replace(' ', '_')
                output_file += f"_{region_str}"
        
        output_file += ".png"
       
        plt.savefig(os.path.join(output_dir, output_file))
   
    return fig

