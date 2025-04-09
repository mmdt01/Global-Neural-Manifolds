"""
Analysis module for neural data.
"""
import os
import numpy as np

from .time_frequency import compute_time_frequency
from .manifold import (
    compute_neural_manifold,
    compute_gesture_manifolds,
    align_subject_manifolds_with_cca
)
from .band_power_viz import (
    visualize_band_power_processing,
    visualize_multi_epoch_band_power,
    visualize_comparative_bands
)
from .high_dim_manifold import (
    analyze_high_dim_neural_manifolds,
    align_high_dim_manifolds
)

def perform_time_frequency_analysis(region_epochs, region_channels_dict, region_labels,
                                  tmin=None, tmax=None, output_dir=None):
    """
    Perform time-frequency analysis on region-specific epoch data.
    
    Parameters:
    -----------
    region_epochs : dict
        Dictionary mapping subject IDs to region-specific epochs objects
    region_channels_dict : dict
        Dictionary mapping subject IDs to lists of channel names
    region_labels : list
        List of brain region names included in the analysis
    tmin : float, optional
        Start time for analysis window
    tmax : float, optional
        End time for analysis window
    output_dir : str, optional
        Directory to save TF plots. If None, plots are displayed but not saved.
    
    Returns:
    --------
    tfr_power_dict : dict
        Dictionary mapping subject IDs to time-frequency power objects
    """
    # Define linear-spaced frequencies from 2-200 Hz
    freqs = np.linspace(2, 200, num=100)
    
    # Define number of cycles (more cycles for higher frequencies)
    n_cycles = freqs / 2  # Higher frequencies get more cycles
    
    # Set baseline period
    baseline = (-0.5, 0.0)
    
    # Crop epochs to the desired time window if specified
    if tmin is not None or tmax is not None:
        cropped_epochs = {}
        for subject_id, epochs in region_epochs.items():
            cropped_epochs[subject_id] = epochs.copy().crop(tmin=tmin, tmax=tmax)
        analysis_epochs = cropped_epochs
    else:
        analysis_epochs = region_epochs
    
    # Compute time-frequency representations
    tfr_power_dict = compute_time_frequency(
        analysis_epochs,
        region_channels_dict,
        region_labels,
        freqs=freqs,
        n_cycles=n_cycles,
        baseline=baseline,
        output_dir=output_dir
    )
    
    return tfr_power_dict

def analyze_neural_manifolds(region_epochs, region_channels_dict, region_labels, 
                           bands=None, n_components=3, downsample_factor=1, output_dir=None):
    """
    Analyze neural manifolds for region-specific epoch data across multiple frequency bands.
    
    Parameters:
    -----------
    region_epochs : dict
        Dictionary mapping subject IDs to region-specific epochs objects
    region_channels_dict : dict
        Dictionary mapping subject IDs to lists of channel names
    region_labels : list
        List of brain region names included in the analysis
    bands : list, optional
        List of frequency bands to analyze. If None, uses ['delta', 'beta', 'high_gamma']
    n_components : int, optional
        Number of PCA components to compute
    downsample_factor : int, optional
        Factor by which to downsample the data before PCA
    output_dir : str, optional
        Directory to save plots. If None, plots are displayed but not saved.
        
    Returns:
    --------
    manifold_results : dict
        Dictionary mapping band names to manifold dictionaries
    """
    # Set default bands if not provided
    if bands is None:
        bands = ['delta', 'beta', 'high_gamma']
    
    # Initialize results dictionary
    manifold_results = {}
    
    # Analyze each frequency band
    for band_name in bands:
        print(f"\n\n===== Analyzing {band_name.upper()} band neural manifolds =====")
        
        # Compute neural manifolds
        manifold_dict = compute_neural_manifold(
            region_epochs,
            region_channels_dict,
            band_name,
            n_components=n_components,
            downsample_factor=downsample_factor,
            plot=True,
            plot_title=f"Regions: {', '.join(region_labels)}",
            output_dir=output_dir
        )
        
        # Save results
        manifold_results[band_name] = manifold_dict
    
    return manifold_results

def analyze_gesture_manifolds(region_epochs, region_channels_dict, region_labels, 
                            bands=None, gestures=None, n_components=3, 
                            downsample_factor=1, output_dir=None):
    """
    Analyze neural manifolds for each gesture across multiple frequency bands.
    
    Parameters:
    -----------
    region_epochs : dict
        Dictionary mapping subject IDs to region-specific epochs objects
    region_channels_dict : dict
        Dictionary mapping subject IDs to lists of channel names
    region_labels : list
        List of brain region names included in the analysis
    bands : list, optional
        List of frequency bands to analyze. If None, uses ['delta', 'beta', 'high_gamma']
    gestures : list, optional
        List of gesture names to analyze. If None, uses all available gestures
    n_components : int, optional
        Number of PCA components to compute
    downsample_factor : int, optional
        Factor by which to downsample the data before PCA
    output_dir : str, optional
        Directory to save plots. If None, plots are displayed but not saved.
        
    Returns:
    --------
    gesture_manifold_results : dict
        Dictionary mapping band names to dictionaries of gesture manifolds
    """
    # Set default bands if not provided
    if bands is None:
        bands = ['delta', 'beta', 'high_gamma']
    
    # Get list of gestures if not provided
    if gestures is None:
        # Get gestures from the first subject's epochs
        first_subject_id = list(region_epochs.keys())[0]
        gestures = list(region_epochs[first_subject_id].event_id.keys())
    
    # Initialize results dictionary
    gesture_manifold_results = {}
    
    # Analyze each frequency band
    for band_name in bands:
        print(f"\n\n===== Analyzing {band_name.upper()} band gesture-specific neural manifolds =====")
        
        # Compute neural manifolds for each gesture
        gesture_manifolds = compute_gesture_manifolds(
            region_epochs,
            region_channels_dict,
            band_name,
            gestures,
            n_components=n_components,
            downsample_factor=downsample_factor,
            plot=True,
            output_dir=output_dir
        )
        
        # Save results
        gesture_manifold_results[band_name] = gesture_manifolds
    
    return gesture_manifold_results

def compare_subject_manifolds(manifold_results, subject_ids, bands=None, output_dir=None):
    """
    Compare manifold representations between pairs of subjects using CCA.
    
    Parameters:
    -----------
    manifold_results : dict
        Dictionary containing manifold data for each band and subject
    subject_ids : list
        List of subject IDs to compare
    bands : list, optional
        List of frequency bands to analyze. If None, uses all available bands.
    output_dir : str, optional
        Directory to save plots. If None, plots are displayed but not saved.
    
    Returns:
    --------
    dict
        Dictionary containing CCA results for each pair of subjects and each band
    """
    # If bands not specified, use all available bands
    if bands is None:
        bands = list(manifold_results.keys())
    
    # Initialize results dictionary
    cca_results = {}
    
    # Compare each pair of subjects for each band
    for band_name in bands:
        print(f"\n===== Comparing {band_name.upper()} band manifolds between subjects =====")
        
        # Skip if this band is not in the results
        if band_name not in manifold_results:
            print(f"No results found for {band_name} band, skipping...")
            continue
        
        # Initialize band-specific results
        cca_results[band_name] = {}
        
        # Compare each pair of subjects
        for i, subject_id1 in enumerate(subject_ids):
            for subject_id2 in subject_ids[i+1:]:
                # Skip if either subject is not in the results
                if subject_id1 not in manifold_results[band_name] or subject_id2 not in manifold_results[band_name]:
                    print(f"Missing data for Subject {subject_id1} or {subject_id2}, skipping...")
                    continue
                
                print(f"\nComparing Subject {subject_id1} vs Subject {subject_id2}...")
                
                # Align manifolds with CCA
                result = align_subject_manifolds_with_cca(
                    manifold_results[band_name],
                    subject_id1,
                    subject_id2,
                    band_name,
                    downsample_factor=1,
                    plot=True,
                    output_dir=output_dir
                )
                
                # Store the results
                cca_results[band_name][f"{subject_id1}_vs_{subject_id2}"] = result
    
    return cca_results

def visualize_signal_transform(epochs, subject_id, region_label, 
                             bands=None, 
                             channel_indices=None, n_channels=4,
                             epoch_indices=None, n_epochs=4,
                             show_processing_steps=True,
                             show_multi_epoch=True, 
                             show_comparative_bands=True,
                             downsample_factor=1,
                             output_dir=None):
    """
    High-level function to run all three band power visualization functions.
    
    Parameters:
    -----------
    epochs : mne.Epochs
        MNE Epochs object containing the data
    subject_id : str or int
        Subject ID for plot titles
    region_label : str
        Brain region label for plot titles
    bands : list, optional
        List of frequency bands to visualize. If None, uses ['delta', 'beta', 'high_gamma']
    channel_indices : list, optional
        List of specific channel indices to visualize. If None, first n_channels will be used.
    n_channels : int, optional
        Number of channels to use if channel_indices is None
    epoch_indices : list, optional
        List of specific epoch indices to visualize. If None, first n_epochs will be used.
    n_epochs : int, optional
        Number of epochs to use if epoch_indices is None
    show_processing_steps : bool, optional
        Whether to run visualize_band_power_processing
    show_multi_epoch : bool, optional
        Whether to run visualize_multi_epoch_band_power
    show_comparative_bands : bool, optional
        Whether to run visualize_comparative_bands
    downsample_factor : int, optional
        Factor by which to downsample the result
    output_dir : str, optional
        Directory to save plots. If None, plots are displayed but not saved.
        
    Returns:
    --------
    figures : dict
        Dictionary mapping visualization types to lists of figure objects
    """
    # Set default bands if not provided
    if bands is None:
        bands = ['delta', 'beta', 'high_gamma']
    
    # Choose channel indices
    if channel_indices is None:
        channel_indices = list(range(min(n_channels, len(epochs.ch_names))))
    
    # Choose epoch indices
    if epoch_indices is None:
        epoch_indices = list(range(min(n_epochs, len(epochs))))
    
    # Create output directory if specified
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        
    # Initialize dictionary to store figures
    figures = {
        'processing_steps': [],
        'multi_epoch': [],
        'comparative_bands': []
    }
    
    # Print information about what will be visualized
    print(f"\nGenerating signal transformation visualizations for Subject {subject_id}:")
    print(f"  Region: {region_label}")
    print(f"  Bands: {bands}")
    print(f"  Selected channels: {channel_indices}")
    print(f"  Selected epochs: {epoch_indices}")
    
    # 1. Visualize processing steps for each band
    if show_processing_steps:
        print("\nVisualizing band power processing steps...")
        for band_name in bands:
            for epoch_idx in epoch_indices[:1]:  # Just use the first epoch for processing steps
                fig = visualize_band_power_processing(
                    epochs, 
                    band_name, 
                    subject_id, 
                    region_label,
                    channels=channel_indices,
                    example_epoch=epoch_idx,
                    downsample_factor=downsample_factor,
                    output_dir=output_dir
                )
                figures['processing_steps'].append(fig)
    
    # 2. Visualize multiple epochs for each band and channel
    if show_multi_epoch:
        print("\nVisualizing multi-epoch band power...")
        for band_name in bands:
            for channel_idx in channel_indices[:1]:  # Just use the first channel for multi-epoch
                fig = visualize_multi_epoch_band_power(
                    epochs,
                    band_name,
                    subject_id,
                    region_label,
                    channel_idx=channel_idx,
                    n_epochs=len(epoch_indices),
                    downsample_factor=downsample_factor,
                    output_dir=output_dir
                )
                figures['multi_epoch'].append(fig)
    
    # 3. Visualize comparative bands for each channel and epoch
    if show_comparative_bands:
        print("\nVisualizing comparative frequency bands...")
        for channel_idx in channel_indices[:1]:  # Just use the first channel for comparative bands
            for epoch_idx in epoch_indices[:1]:  # Just use the first epoch for comparative bands
                fig = visualize_comparative_bands(
                    epochs,
                    subject_id,
                    region_label,
                    channel_idx=channel_idx,
                    epoch_idx=epoch_idx,
                    bands=bands,
                    output_dir=output_dir
                )
                figures['comparative_bands'].append(fig)
    
    print("\nVisualization complete!")
    
    return figures