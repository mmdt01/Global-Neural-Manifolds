"""
Analysis module for neural data.
"""
import os
import numpy as np
from utils.helpers import ensure_dir

from .time_frequency import compute_time_frequency

from .manifold import (
    compute_neural_manifold,
    plot_neural_vaf,
    create_gesture_comparison_summary,
    compute_gesture_manifolds,
    align_subject_manifolds_with_cca
)
from .high_dim_manifold import (
    analyze_high_dim_neural_manifolds,
    align_high_dim_manifolds
)
from .cross_region_manifold import (
    align_cross_region_manifolds,
    compute_region_similarity_matrix,
    analyze_mode_specific_correlations,
    compare_within_vs_cross_region_correlations
)
from .mean_activity import (
    compute_gesture_mean_activity,
    compute_mahalanobis_distance_matrix,
    visualize_distance_matrix
)
from .lfo_classification import (
    prepare_data_for_classification,
    run_pairwise_classification,
    run_multiclass_classification,
    visualize_pairwise_classification,
    visualize_multiclass_classification,
    analyze_cross_region_classification
)
from .cross_vaf_analysis import (
    compute_cross_gesture_vaf_analysis,
    visualize_cross_vaf_results
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
                           bands, n_components=3, output_dir=None, gesture_comparison=False):
    """
    Analyze neural manifolds for region-specific epoch data across multiple frequency bands.
    
    Parameters:
    -----------
    region_epochs : dict
        Nested dictionary mapping subject IDs to frequency bands to epochs objects
    region_channels_dict : dict
        Dictionary mapping subject IDs to lists of channel names
    region_labels : list
        List of brain region names included in the analysis
    bands : list
        List of frequency bands to analyze
    n_components : int, optional
        Number of PCA components to compute
    output_dir : str, optional
        Directory to save plots. If None, plots are displayed but not saved.
    gesture_comparison : bool, optional
        If True, applies PCA to individual gesture trials separately for comparison
        
    Returns:
    --------
    manifold_results : dict
        Dictionary mapping band names to manifold dictionaries
        If gesture_comparison=False: {band_name: {subject_id: manifold_data}}
        If gesture_comparison=True: {band_name: {gesture_name: {subject_id: manifold_data}}}
    """
    # Validate bands parameter
    if not bands:
        raise ValueError("No frequency bands specified for analysis.")

    # Initialize results dictionary
    manifold_results = {}
    
    # Create VAF output directory
    if output_dir is not None:
        vaf_output_dir = os.path.join(output_dir, "..", "vaf_plots")
        ensure_dir(vaf_output_dir)
        # Get the region/group name from output_dir path
        region_name = os.path.basename(output_dir)
        vaf_region_dir = os.path.join(vaf_output_dir, region_name)
        ensure_dir(vaf_region_dir)
    else:
        vaf_region_dir = None
    
    # Analyze each frequency band
    for band_name in bands:
        print(f"\n\n===== Analyzing {band_name.upper()} band neural manifolds =====")
        
        # Check if any subjects have this band
        subjects_with_band = [subject_id for subject_id, band_dict in region_epochs.items() 
                             if band_name in band_dict]
        
        if not subjects_with_band:
            print(f"No subjects have data for band '{band_name}', skipping...")
            continue
        
        if not gesture_comparison:
            # Original behavior: compute manifolds using all trials
            print(f"Computing manifolds using all trials combined...")
            
            manifold_dict = compute_neural_manifold(
                region_epochs,  # Pass the full nested dictionary
                region_channels_dict,
                band_name,
                n_components=n_components,
                plot=True,
                plot_title=None,
                output_dir=output_dir
            )
            
            # Save results
            manifold_results[band_name] = manifold_dict
            
            # Generate VAF plot for standard analysis
            if len(manifold_dict) > 0:
                print(f"Generating VAF plot for {band_name} band...")
                plot_neural_vaf(
                    manifold_dict,
                    region_labels,
                    band_name=band_name,
                    output_dir=vaf_region_dir,
                    max_components=n_components
                )
            
        else:
            # New behavior: compute manifolds for each gesture separately
            print(f"Running gesture comparison analysis...")
            
            # Get list of gestures from the first subject's epochs
            first_subject_id = subjects_with_band[0]
            first_epochs = region_epochs[first_subject_id][band_name]
            available_gestures = list(first_epochs.event_id.keys())
            
            print(f"Found gestures: {available_gestures}")
            
            # Initialize results for this band
            manifold_results[band_name] = {}
            
            # Process each gesture
            for gesture_name in available_gestures:
                print(f"\n--- Processing gesture: {gesture_name} ---")
                
                # Create epochs dictionary for this gesture only
                gesture_epochs = {}
                gesture_trial_counts = {}
                
                for subject_id in subjects_with_band:
                    if subject_id not in region_epochs or band_name not in region_epochs[subject_id]:
                        continue
                        
                    full_epochs = region_epochs[subject_id][band_name]
                    
                    # Extract only this gesture's trials
                    try:
                        gesture_specific_epochs = full_epochs[gesture_name]
                        
                        # Skip if no trials for this gesture
                        if len(gesture_specific_epochs) == 0:
                            print(f"  No trials for gesture '{gesture_name}' in Subject {subject_id}")
                            continue
                            
                        # Store in nested structure (maintaining original format)
                        if subject_id not in gesture_epochs:
                            gesture_epochs[subject_id] = {}
                        gesture_epochs[subject_id][band_name] = gesture_specific_epochs
                        gesture_trial_counts[subject_id] = len(gesture_specific_epochs)
                        
                        print(f"  Subject {subject_id}: {len(gesture_specific_epochs)} trials")
                        
                    except KeyError:
                        print(f"  Gesture '{gesture_name}' not found in Subject {subject_id}")
                        continue
                
                # Skip if no subjects have this gesture
                if not gesture_epochs:
                    print(f"  No subjects have trials for gesture '{gesture_name}', skipping...")
                    continue
                
                print(f"  Total subjects with {gesture_name} trials: {len(gesture_epochs)}")
                print(f"  Trial counts: {gesture_trial_counts}")
                
                # Create gesture-specific output directory
                gesture_output_dir = None
                if output_dir is not None:
                    gesture_output_dir = os.path.join(output_dir, f"gesture_{gesture_name}")
                    os.makedirs(gesture_output_dir, exist_ok=True)
                
                # Compute manifold for this gesture
                print(f"  Computing spatial manifold for gesture '{gesture_name}'...")
                gesture_manifold_dict = compute_neural_manifold(
                    gesture_epochs,
                    region_channels_dict,
                    band_name,
                    n_components=n_components,
                    plot=True,
                    plot_title=f"Gesture: {gesture_name.capitalize()}",
                    output_dir=gesture_output_dir
                )
                
                # Store results for this gesture
                manifold_results[band_name][gesture_name] = gesture_manifold_dict
                
                # Generate VAF plot for this gesture
                if len(gesture_manifold_dict) > 0:
                    print(f"  Generating VAF plot for gesture '{gesture_name}'...")
                    
                    plot_neural_vaf(
                        gesture_manifold_dict,
                        region_labels,
                        band_name=f"{band_name}_{gesture_name}",
                        output_dir=gesture_output_dir,
                        max_components=n_components
                    )
                
                print(f"  Completed analysis for gesture '{gesture_name}'")
            
            # Create gesture comparison summary for this band
            if len(manifold_results[band_name]) > 1:
                print(f"\nCreating gesture comparison summary for {band_name} band...")
                create_gesture_comparison_summary(
                    manifold_results[band_name],
                    band_name,
                    region_labels,
                    n_components,
                    output_dir
                ) 
    
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

def analyze_mean_delta_activity(region_epochs, region_channels_dict, region_labels, 
                              event_dict=None, output_dir=None, band_name='delta'):
    """
    Analyze the mean delta band activity and compute Mahalanobis distances between gestures.
    
    Parameters:
    -----------
    region_epochs : dict
        Nested dictionary mapping subject IDs to frequency bands to region-specific epochs objects
    region_channels_dict : dict
        Dictionary mapping subject IDs to lists of channel names
    region_labels : list
        List of brain region names included in the analysis
    event_dict : dict, optional
        Dictionary mapping gesture names to event IDs
    output_dir : str, optional
        Directory to save plots. If None, plots are displayed but not saved.
    band_name : str, optional
        Name of the frequency band to analyze. Default is 'delta'.
    
    Returns:
    --------
    results : dict
        Dictionary containing distance matrices and other results for each subject
    """
    # Initialize results dictionary
    results = {}
    
    # Process each subject
    for subject_id, band_epochs in region_epochs.items():
        # Skip if no epochs for this subject or if the specified band is not available
        if len(band_epochs) == 0 or band_name not in band_epochs:
            print(f"No epochs or no {band_name} band for Subject {subject_id}, skipping...")
            continue
        
        # Get the epochs for the specified band
        epochs = band_epochs[band_name]
        
        print(f"\nAnalyzing {band_name} band activity for Subject {subject_id}...")
        
        # Create subject-specific output directory if needed
        subject_output_dir = None
        if output_dir is not None:
            subject_output_dir = os.path.join(output_dir, f"subject_{subject_id}")
            os.makedirs(subject_output_dir, exist_ok=True)
        
        # 1. Compute gesture centroids and trial data
        if event_dict is None:
            gestures = list(epochs.event_id.keys())
        else:
            gestures = list(event_dict.keys())
        
        gesture_centroids, trial_data = compute_gesture_mean_activity(epochs, gestures)
        
        # 2. Compute Mahalanobis distance matrix
        distance_matrix, gesture_labels = compute_mahalanobis_distance_matrix(gesture_centroids, trial_data)
        
        # 3. Visualize distance matrix
        fig = visualize_distance_matrix(
            distance_matrix, 
            gesture_labels, 
            subject_id=subject_id, 
            region_label=', '.join(region_labels), 
            output_dir=subject_output_dir
        )
        
        # Store results for this subject
        results[subject_id] = {
            'distance_matrix': distance_matrix,
            'gesture_labels': gesture_labels,
            'gesture_centroids': gesture_centroids,
            'trial_data': trial_data
        }
    
    return results

def analyze_gesture_classification(trial_data, subject_id=None, region_label=None, output_dir=None,
                                 n_folds=5, n_permutations=100, use_pca=True, pca_components=0.95,
                                 run_pairwise=True, run_multiclass=True, n_jobs=-1, random_state=42):
    """
    Analyze gesture classification using SVM.
    
    Parameters:
    -----------
    trial_data : dict
        Dictionary mapping gesture names to arrays of trial data (trials x channels)
    subject_id : str or int, optional
        Subject ID for the plot title
    region_label : str, optional
        Brain region label for the plot title
    output_dir : str, optional
        Directory to save plots. If None, plots are displayed but not saved.
    n_folds : int, optional
        Number of cross-validation folds
    n_permutations : int, optional
        Number of permutations for statistical testing, or 0 to skip
    use_pca : bool, optional
        Whether to apply PCA before classification
    pca_components : float or int, optional
        Number of PCA components to use or variance to explain
    run_pairwise : bool, optional
        Whether to run pairwise classification
    run_multiclass : bool, optional
        Whether to run multi-class classification
    n_jobs : int, optional
        Number of parallel jobs to run
    random_state : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    results : dict
        Dictionary containing classification results
    """
    # Initialize results dictionary
    results = {
        'pairwise': None,
        'multiclass': None,
        'figures': {}
    }
    
    # Prepare data for classification
    X, y, gesture_labels = prepare_data_for_classification(trial_data)
    
    # Skip if we don't have enough data
    if len(X) < 10 or len(gesture_labels) < 2:
        print("Not enough data for classification analysis")
        return results
    
    print(f"Analyzing gesture classification for {len(gesture_labels)} gestures, " 
          f"{len(X)} trials, {X.shape[1]} channels")
    
    # Run pairwise classification if requested
    if run_pairwise:
        print("\nRunning pairwise classification...")
        pairwise_results = run_pairwise_classification(
            X, y, n_folds=n_folds, n_permutations=n_permutations,
            use_pca=use_pca, pca_components=pca_components,
            n_jobs=n_jobs, random_state=random_state
        )
        results['pairwise'] = pairwise_results
        
        # Visualize pairwise results
        pairwise_figs = visualize_pairwise_classification(
            pairwise_results, subject_id=subject_id, region_label=region_label,
            output_dir=output_dir
        )
        results['figures'].update(pairwise_figs)
    
    # Run multi-class classification if requested
    if run_multiclass:
        print("\nRunning multi-class classification...")
        multiclass_results = run_multiclass_classification(
            X, y, n_folds=n_folds, n_permutations=n_permutations,
            use_pca=use_pca, pca_components=pca_components,
            random_state=random_state
        )
        results['multiclass'] = multiclass_results
        
        # Visualize multi-class results
        multiclass_figs = visualize_multiclass_classification(
            multiclass_results, subject_id=subject_id, region_label=region_label,
            output_dir=output_dir
        )
        results['figures'].update(multiclass_figs)
    
    return results

def cross_region_lfo_classification_analysis(args, all_region_results):
    """
    Run cross-region analysis on classification results and generate box plot visualization.
   
    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments
    all_region_results : dict
        Dictionary mapping region names to classification results
   
    Returns:
    --------
    region_comparison : dict
        Dictionary containing cross-region pairwise classification results
    """
    # Create output directory
    cross_region_dir = os.path.join(args.output_dir, 'cross_region_classification')
    ensure_dir(cross_region_dir)
   
    # Run simplified cross-region analysis that focuses on the box plot
    print("\nAnalyzing pairwise classification performance across brain regions...")
    region_comparison = analyze_cross_region_classification(
        all_region_results,
        output_dir=cross_region_dir
    )
    
    print(f"\nBox plot visualization saved to {os.path.join(cross_region_dir, 'region_pairwise_boxplot.png')}")
    
    return region_comparison
