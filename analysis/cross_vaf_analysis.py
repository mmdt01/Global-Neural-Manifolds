"""
Cross-projection VAF analysis for testing shared neural structure across gestures.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import os

def compute_vaf(original_data, reconstructed_data):
    """
    Compute Variance Accounted For (VAF) between original and reconstructed data.
    
    Parameters:
    -----------
    original_data : np.ndarray
        Original neural data (time_points, channels)
    reconstructed_data : np.ndarray  
        Reconstructed data from projection (time_points, channels)
        
    Returns:
    --------
    vaf : float
        Variance Accounted For as percentage (0-100)
    """
    # Compute residual variance
    residual = original_data - reconstructed_data
    
    # VAF = 1 - (residual_variance / original_variance)
    original_variance = np.var(original_data)
    residual_variance = np.var(residual)
    
    if original_variance == 0:
        return 0.0
    
    vaf = (1 - residual_variance / original_variance) * 100
    return max(0.0, min(100.0, vaf))  # Clamp to [0, 100]

def extract_gesture_data_matrix(region_epochs, subject_id, band_name, gesture_name):
    """
    Extract and format neural data for a specific gesture.
    
    Parameters:
    -----------
    region_epochs : dict
        Nested dictionary with epoch data
    subject_id : int
        Subject ID
    band_name : str
        Frequency band name
    gesture_name : str
        Gesture name
        
    Returns:
    --------
    data_matrix : np.ndarray
        Formatted data matrix (time_points, channels)
    """
    # Get epochs object for this band
    epochs_obj = region_epochs[subject_id][band_name]
    
    # Extract epochs for this specific gesture
    gesture_epochs = epochs_obj[gesture_name]
    
    # Extract data: (n_trials, n_channels, n_times)
    data = gesture_epochs.get_data()
    
    # Reshape to (time_points, channels) by concatenating trials in time
    n_trials, n_channels, n_times = data.shape
    data_matrix = data.transpose(0, 2, 1).reshape(n_trials * n_times, n_channels)
    
    return data_matrix

def compute_cross_projection_vaf(data_source, pca_target, pca_source):
    """
    Project source gesture data onto target gesture's manifold and compute VAF.
    
    Parameters:
    -----------
    data_source : np.ndarray
        Source gesture data (time_points, channels)
    pca_target : sklearn.decomposition.PCA
        Fitted PCA object from target gesture
    pca_source : sklearn.decomposition.PCA
        Fitted PCA object from source gesture (for within-gesture VAF)
        
    Returns:
    --------
    cross_vaf : float
        VAF when projecting onto target manifold
    within_vaf : float
        VAF when projecting onto own manifold (should be ~75% for 12D)
    ratio : float
        cross_vaf / within_vaf
    """
    # Project source data onto target manifold
    projected_cross = pca_target.transform(data_source)
    reconstructed_cross = pca_target.inverse_transform(projected_cross)
    cross_vaf = compute_vaf(data_source, reconstructed_cross)
    
    # Project source data onto its own manifold (reference)
    projected_within = pca_source.transform(data_source)
    reconstructed_within = pca_source.inverse_transform(projected_within)
    within_vaf = compute_vaf(data_source, reconstructed_within)
    
    # Compute ratio
    if within_vaf > 0:
        ratio = cross_vaf / within_vaf
    else:
        ratio = 0.0
    
    return cross_vaf, within_vaf, ratio

def generate_random_control_manifolds(n_channels, n_components=12, n_controls=50):
    """
    Generate random orthogonal manifolds as controls.
    
    Parameters:
    -----------
    n_channels : int
        Number of neural channels
    n_components : int
        Number of components in manifold
    n_controls : int
        Number of random controls to generate
        
    Returns:
    --------
    random_pcas : list
        List of PCA objects with random orthogonal components
    """
    random_pcas = []
    
    for i in range(n_controls):
        # Generate random orthogonal matrix
        random_matrix = np.random.randn(n_channels, n_components)
        Q, _ = np.linalg.qr(random_matrix)  # Orthogonalize
        
        # Create mock PCA object with random components
        random_pca = PCA(n_components=n_components)
        random_pca.components_ = Q.T  # PCA stores components as rows
        random_pca.mean_ = np.zeros(n_channels)  # Zero mean
        
        random_pcas.append(random_pca)
    
    return random_pcas
