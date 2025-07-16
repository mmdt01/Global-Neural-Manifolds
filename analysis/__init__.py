"""
Analysis module for neural data.
"""
import os
import numpy as np
from utils.helpers import ensure_dir
from .time_frequency import compute_time_frequency
from scipy.stats import ttest_ind

from .manifold import (
    compute_neural_manifold,
    plot_neural_vaf,
    create_gesture_comparison_summary,
    compute_gesture_manifolds,
    align_subject_manifolds_with_cca,
    extract_region_from_channel
)
from .principal_angles import (
    compute_principal_angles,
    analyze_gesture_manifold_similarity
)
from .cross_vaf_analysis import (
    extract_gesture_data_matrix,
    compute_cross_projection_vaf,
    generate_random_control_manifolds
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
from .tme_manifold_integration import (
    verify_null_generation,
    create_enhanced_tme_visualization,
    create_aggregate_enhanced_visualization
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

# # Old files
# def analyze_gesture_manifolds(region_epochs, region_channels_dict, region_labels, 
#                             bands=None, gestures=None, n_components=3, 
#                             downsample_factor=1, output_dir=None):
#     """
#     Analyze neural manifolds for each gesture across multiple frequency bands.
    
#     Parameters:
#     -----------
#     region_epochs : dict
#         Dictionary mapping subject IDs to region-specific epochs objects
#     region_channels_dict : dict
#         Dictionary mapping subject IDs to lists of channel names
#     region_labels : list
#         List of brain region names included in the analysis
#     bands : list, optional
#         List of frequency bands to analyze. If None, uses ['delta', 'beta', 'high_gamma']
#     gestures : list, optional
#         List of gesture names to analyze. If None, uses all available gestures
#     n_components : int, optional
#         Number of PCA components to compute
#     downsample_factor : int, optional
#         Factor by which to downsample the data before PCA
#     output_dir : str, optional
#         Directory to save plots. If None, plots are displayed but not saved.
        
#     Returns:
#     --------
#     gesture_manifold_results : dict
#         Dictionary mapping band names to dictionaries of gesture manifolds
#     """
#     # Set default bands if not provided
#     if bands is None:
#         bands = ['delta', 'beta', 'high_gamma']
    
#     # Get list of gestures if not provided
#     if gestures is None:
#         # Get gestures from the first subject's epochs
#         first_subject_id = list(region_epochs.keys())[0]
#         gestures = list(region_epochs[first_subject_id].event_id.keys())
    
#     # Initialize results dictionary
#     gesture_manifold_results = {}
    
#     # Analyze each frequency band
#     for band_name in bands:
#         print(f"\n\n===== Analyzing {band_name.upper()} band gesture-specific neural manifolds =====")
        
#         # Compute neural manifolds for each gesture
#         gesture_manifolds = compute_gesture_manifolds(
#             region_epochs,
#             region_channels_dict,
#             band_name,
#             gestures,
#             n_components=n_components,
#             downsample_factor=downsample_factor,
#             plot=True,
#             output_dir=output_dir
#         )
        
#         # Save results
#         gesture_manifold_results[band_name] = gesture_manifolds
    
#     return gesture_manifold_results
# def compare_subject_manifolds(manifold_results, subject_ids, bands=None, output_dir=None):
#     """
#     Compare manifold representations between pairs of subjects using CCA.
    
#     Parameters:
#     -----------
#     manifold_results : dict
#         Dictionary containing manifold data for each band and subject
#     subject_ids : list
#         List of subject IDs to compare
#     bands : list, optional
#         List of frequency bands to analyze. If None, uses all available bands.
#     output_dir : str, optional
#         Directory to save plots. If None, plots are displayed but not saved.
    
#     Returns:
#     --------
#     dict
#         Dictionary containing CCA results for each pair of subjects and each band
#     """
#     # If bands not specified, use all available bands
#     if bands is None:
#         bands = list(manifold_results.keys())
    
#     # Initialize results dictionary
#     cca_results = {}
    
#     # Compare each pair of subjects for each band
#     for band_name in bands:
#         print(f"\n===== Comparing {band_name.upper()} band manifolds between subjects =====")
        
#         # Skip if this band is not in the results
#         if band_name not in manifold_results:
#             print(f"No results found for {band_name} band, skipping...")
#             continue
        
#         # Initialize band-specific results
#         cca_results[band_name] = {}
        
#         # Compare each pair of subjects
#         for i, subject_id1 in enumerate(subject_ids):
#             for subject_id2 in subject_ids[i+1:]:
#                 # Skip if either subject is not in the results
#                 if subject_id1 not in manifold_results[band_name] or subject_id2 not in manifold_results[band_name]:
#                     print(f"Missing data for Subject {subject_id1} or {subject_id2}, skipping...")
#                     continue
                
#                 print(f"\nComparing Subject {subject_id1} vs Subject {subject_id2}...")
                
#                 # Align manifolds with CCA
#                 result = align_subject_manifolds_with_cca(
#                     manifold_results[band_name],
#                     subject_id1,
#                     subject_id2,
#                     band_name,
#                     downsample_factor=1,
#                     plot=True,
#                     output_dir=output_dir
#                 )
                
#                 # Store the results
#                 cca_results[band_name][f"{subject_id1}_vs_{subject_id2}"] = result
    
#     return cca_results

# Time-frequency analysis
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

# Compute Mahalobis distances between gesture mean delta activity
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

# Compute manifolds for region-specific epoch data
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

# Spatial Loadings Analysis
def save_spatial_loadings(manifold_dict, region_channels_dict, band_name, 
                         n_modes=3, output_dir=None):
    """
    Save spatial loadings (PCA component weights) for each channel to text files.
    
    Parameters:
    -----------
    manifold_dict : dict
        Dictionary containing manifold results for subjects
    region_channels_dict : dict
        Dictionary mapping subject IDs to channel names
    band_name : str
        Name of the frequency band
    n_modes : int, optional
        Number of neural modes to save (default: 3)
    output_dir : str, optional
        Directory to save text files. If None, saves to current directory.
    
    Returns:
    --------
    saved_files : list
        List of saved file paths
    """
    saved_files = []
    
    print(f"\nSaving spatial loadings for first {n_modes} neural modes...")
    
    # Process each subject
    for subject_id, results in manifold_dict.items():
        print(f"Processing Subject {subject_id}...")
        
        # Get spatial patterns and channel names
        spatial_patterns = results['spatial_patterns']  # (n_channels, n_components)
        channel_names = region_channels_dict[subject_id]
        explained_variance = results['explained_variance']
        
        # Extract first n_modes
        n_channels, n_components = spatial_patterns.shape
        n_modes_to_save = min(n_modes, n_components)
        
        # Create output filename
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, f'spatial_loadings_subject_{subject_id}_{band_name}.txt')
        else:
            filename = f'spatial_loadings_subject_{subject_id}_{band_name}.txt'
        
        # Write to text file
        with open(filename, 'w') as f:
            # Write header
            f.write(f"Spatial Loadings for Subject {subject_id}, {band_name.upper()} Band\n")
            f.write(f"Neural Manifold Analysis - First {n_modes_to_save} Components\n")
            f.write("="*70 + "\n\n")
            
            # Write explained variance for each component
            f.write("Explained Variance:\n")
            for i in range(n_modes_to_save):
                f.write(f"  Component {i+1}: {explained_variance[i]*100:.2f}%\n")
            f.write(f"  Total (first {n_modes_to_save}): {sum(explained_variance[:n_modes_to_save])*100:.2f}%\n\n")
            
            # Write column headers
            header = "Channel_Name\tRegion"
            for i in range(n_modes_to_save):
                header += f"\tMode_{i+1}_Weight"
            f.write(header + "\n")
            
            # Write data for each channel
            for ch_idx, channel_name in enumerate(channel_names):
                # Extract region from channel name
                region = extract_region_from_channel(channel_name)
                
                # Start line with channel name and region
                line = f"{channel_name}\t{region}"
                
                # Add weights for each mode
                for mode_idx in range(n_modes_to_save):
                    weight = spatial_patterns[ch_idx, mode_idx]
                    line += f"\t{weight:.6f}"
                
                f.write(line + "\n")
            
            # Write summary statistics
            f.write(f"\n" + "="*70 + "\n")
            f.write("SUMMARY STATISTICS\n")
            f.write("="*70 + "\n\n")
            
            for mode_idx in range(n_modes_to_save):
                weights = spatial_patterns[:, mode_idx]
                f.write(f"Neural Mode {mode_idx + 1}:\n")
                f.write(f"  Mean absolute weight: {np.mean(np.abs(weights)):.4f}\n")
                f.write(f"  Max weight: {np.max(weights):.4f}\n")
                f.write(f"  Min weight: {np.min(weights):.4f}\n")
                f.write(f"  Standard deviation: {np.std(weights):.4f}\n")
                
                # Find channels with highest positive and negative weights
                max_idx = np.argmax(weights)
                min_idx = np.argmin(weights)
                f.write(f"  Highest positive: {channel_names[max_idx]} ({weights[max_idx]:.4f})\n")
                f.write(f"  Highest negative: {channel_names[min_idx]} ({weights[min_idx]:.4f})\n\n")
        
        saved_files.append(filename)
        print(f"  Saved: {filename}")
    
    print(f"\nSpatial loadings saved for {len(saved_files)} subjects")
    return saved_files

# Compute similarity between manifolds using principal angles
def run_manifold_similarity_analysis(manifold_results, frequency_bands, n_components, output_dir=None):
    """
    Run manifold similarity analysis for all frequency bands.
    
    Parameters:
    -----------
    manifold_results : dict
        Results from gesture comparison analysis
    frequency_bands : list
        List of frequency bands to analyze
    n_components : int
        Number of components to use
    output_dir : str, optional
        Output directory
        
    Returns:
    --------
    all_results : dict
        Results for all bands
    """
    all_results = {}
    
    for band_name in frequency_bands:
        if band_name in manifold_results:
            print(f"\n{'='*60}")
            print(f"ANALYZING BAND: {band_name.upper()}")
            print(f"{'='*60}")
            
            results = analyze_gesture_manifold_similarity(
                manifold_results[band_name],
                band_name,
                n_components=n_components,
                output_dir=output_dir
            )
            
            all_results[band_name] = results
    
    return all_results

# TME analysis that computes component-wise null principal angles for visualization
def run_enhanced_tme_analysis(region_epochs, region_channels_dict, similarity_results, 
                            args, frequency_bands, output_dir):
    """
    Enhanced TME analysis that computes component-wise null angles for visualization.
    """
    # ================================================
    print("VERIFYING NULL ANGLE GENERATION...")
    verify_null_generation(n_channels=100, n_components=20, n_surrogates=100)  # Small test first
    print("VERIFICATION COMPLETE\n")
    # ================================================

    from analysis.tme_bridge import run_tme_analysis
    
    print("Enhanced TME analysis with component-wise null computation...")
    
    # Get subject list
    subject_id_list = list(region_epochs.keys())
    
    # Store results for enhanced visualization
    enhanced_results = {}
    
    for band_name in frequency_bands:
        if band_name not in similarity_results:
            print(f"No similarity results for {band_name} band, skipping...")
            continue
            
        print(f"\n--- Enhanced TME Analysis for {band_name} band ---")
        
        band_similarity = similarity_results[band_name]
        enhanced_results[band_name] = {}
        
        for subject_id in subject_id_list:
            if (subject_id not in region_epochs or 
                band_name not in region_epochs[subject_id] or
                subject_id not in band_similarity['subject_angles']):
                print(f"Skipping Subject {subject_id} - no data or angles")
                continue
                
            print(f"\nEnhanced TME for Subject {subject_id}...")
            
            try:
                # Extract PRE-COMPUTED observed angles for this subject
                observed_angles = band_similarity['subject_angles'][subject_id]
                gesture_pairs = list(observed_angles.keys())
                
                # Determine actual number of components from observed data
                first_pair = gesture_pairs[0]
                actual_n_components = len(observed_angles[first_pair])
                
                print(f"  Observed data has {actual_n_components} components")
                print(f"  Using {actual_n_components} components for TME analysis (overriding --tme-components)")
                
                # Use actual number of components instead of args.tme_components
                n_components = actual_n_components

                print(f"  Using pre-computed observed angles for {len(gesture_pairs)} gesture pairs")
                
                # Generate TME surrogate data
                print("  Generating TME surrogate tensors...")
                tme_surrogate_results = run_tme_analysis(
                    region_epochs=region_epochs,
                    band_name=band_name,
                    subject_id=subject_id,
                    matlab_tme_path=args.matlab_tme_path,
                    n_surrogates=args.tme_surrogates,
                    preserve_dims=(2, 3),
                    cleanup=True
                )
                
                # Generate COMPONENT-WISE null distribution
                print(f"  Computing component-wise null distribution with {n_components} components...")
                
                n_channels = len(region_channels_dict[subject_id])
                
                # Store component-wise null angles for each gesture pair
                component_wise_null_angles = {pair: [] for pair in gesture_pairs}
                
                for i in range(args.tme_surrogates):
                    if (i + 1) % 200 == 0:
                        print(f"    Progress: {i + 1}/{args.tme_surrogates}")
                    
                    for pair in gesture_pairs:
                        # Generate random orthogonal subspaces for null
                        Q1, _ = np.linalg.qr(np.random.randn(n_channels, n_components))
                        Q2, _ = np.linalg.qr(np.random.randn(n_channels, n_components))
                        
                        # Compute principal angles (all components)
                        angles, _ = compute_principal_angles(Q1, Q2)
                        
                        # Store ALL component angles (not just mean)
                        component_wise_null_angles[pair].append(angles)
                
                # Convert to arrays for easier processing
                for pair in gesture_pairs:
                    component_wise_null_angles[pair] = np.array(component_wise_null_angles[pair])
                    # Shape: (n_surrogates, n_components)
                
                # Compute statistics for each component
                component_wise_stats = {}
                for pair in gesture_pairs:
                    null_data = component_wise_null_angles[pair]  # (n_surrogates, n_components)
                    
                    component_wise_stats[pair] = {
                        'null_mean_per_component': np.mean(null_data, axis=0),  # (n_components,)
                        'null_std_per_component': np.std(null_data, axis=0),   # (n_components,)
                        'null_all_data': null_data  # (n_surrogates, n_components)
                    }
                
                # Statistical testing (overall p-values)
                print("  Performing statistical tests...")
                p_values = {}
                overall_null_angles = {}
                
                for pair in gesture_pairs:
                    observed_angle = np.mean(observed_angles[pair])  # Overall mean
                    null_distribution = np.mean(component_wise_null_angles[pair], axis=1)  # Mean across components for each surrogate
                    
                    p_value = np.mean(null_distribution <= observed_angle)
                    p_values[pair] = p_value
                    overall_null_angles[pair] = null_distribution
                    
                    print(f"    {pair}: {np.degrees(observed_angle):.1f}° vs {np.degrees(np.mean(null_distribution)):.1f}°, p = {p_value:.3f}")
                
                # Compile enhanced results
                test_results = {
                    'subject_id': subject_id,
                    'band_name': band_name,
                    'n_components': n_components,
                    'n_surrogates': args.tme_surrogates,
                    'gesture_labels': tme_surrogate_results['gesture_labels'],
                    'gesture_pairs': gesture_pairs,
                    'observed_angles': observed_angles,  # Pre-computed angles
                    'null_angles': overall_null_angles,  # Overall null for backward compatibility
                    'component_wise_null_stats': component_wise_stats,  # NEW: Component-wise nulls
                    'p_values': p_values,
                    'tme_results': tme_surrogate_results
                }
                
                enhanced_results[band_name][subject_id] = test_results
                
                # Save results
                import pickle
                results_file = os.path.join(output_dir, f'enhanced_tme_results_subject_{subject_id}_{band_name}.pkl')
                with open(results_file, 'wb') as f:
                    pickle.dump(test_results, f)
                
                # Create enhanced visualization with component-wise null line
                create_enhanced_tme_visualization(test_results, output_dir=output_dir)
                
                print(f"Enhanced results saved to: {results_file}")
                
            except Exception as e:
                print(f"ERROR: Enhanced TME failed for Subject {subject_id}: {e}")
                continue
    
    # Create aggregate visualization across all subjects
    if enhanced_results:
        create_aggregate_enhanced_visualization(enhanced_results, output_dir)
    
    print("Enhanced TME analysis completed!")
    return enhanced_results

# Run cross-VAF analysis
def compute_cross_gesture_vaf_analysis(manifold_results, region_epochs, subject_id, band_name, 
                                     n_controls=50):
    """
    Complete cross-gesture VAF analysis for one subject and band.
    
    Parameters:
    -----------
    manifold_results : dict
        Results from gesture manifold analysis
    region_epochs : dict
        Neural epoch data
    subject_id : int
        Subject to analyze
    band_name : str
        Frequency band to analyze
    n_controls : int
        Number of random controls
        
    Returns:
    --------
    results : dict
        Complete analysis results
    """
    print(f"\nCross-gesture VAF analysis for Subject {subject_id}, {band_name} band...")
    
    # Get available gestures for this subject
    if subject_id not in region_epochs or band_name not in region_epochs[subject_id]:
        print(f"No data available for Subject {subject_id}, {band_name}")
        return None
    
    # Get epochs object and extract gesture names from event_id
    epochs_obj = region_epochs[subject_id][band_name]
    gesture_names = list(epochs_obj.event_id.keys())
    n_gestures = len(gesture_names)
    
    print(f"Analyzing {n_gestures} gestures: {gesture_names}")
    
    # Extract PCA objects and data for each gesture
    gesture_pcas = {}
    gesture_data = {}
    
    for gesture in gesture_names:
        # Check if manifold data exists for this gesture
        if (band_name in manifold_results and 
            gesture in manifold_results[band_name] and 
            subject_id in manifold_results[band_name][gesture]):
            
            # Get PCA object
            gesture_pcas[gesture] = manifold_results[band_name][gesture][subject_id]['pca']
            
            # Get raw data
            gesture_data[gesture] = extract_gesture_data_matrix(
                region_epochs, subject_id, band_name, gesture
            )
            
            print(f"  {gesture}: {gesture_data[gesture].shape[0]} time points, "
                  f"{gesture_data[gesture].shape[1]} channels")
        else:
            print(f"  Warning: No manifold data for gesture '{gesture}'")
    
    # Check if we have enough gestures
    if len(gesture_pcas) < 2:
        print(f"Need at least 2 gestures with manifold data, found {len(gesture_pcas)}")
        return None
    
    # Use only gestures that have both PCA and data
    available_gestures = list(gesture_pcas.keys())
    n_available = len(available_gestures)
    
    print(f"Using {n_available} gestures with complete data: {available_gestures}")
    
    # Create ratio matrix
    ratio_matrix = np.zeros((n_available, n_available))
    cross_vaf_matrix = np.zeros((n_available, n_available))
    within_vaf_matrix = np.zeros((n_available, n_available))
    
    # Compute cross-projections
    print("\nComputing cross-gesture projections...")
    
    for i, source_gesture in enumerate(available_gestures):
        for j, target_gesture in enumerate(available_gestures):
            
            # Compute cross-projection VAF
            cross_vaf, within_vaf, ratio = compute_cross_projection_vaf(
                gesture_data[source_gesture],
                gesture_pcas[target_gesture], 
                gesture_pcas[source_gesture]
            )
            
            ratio_matrix[i, j] = ratio
            cross_vaf_matrix[i, j] = cross_vaf
            within_vaf_matrix[i, j] = within_vaf
            
            if i != j:  # Only print off-diagonal
                print(f"  {source_gesture} → {target_gesture}: "
                      f"{cross_vaf:.1f}% / {within_vaf:.1f}% = {ratio:.3f}")
    
    # Generate random controls
    print(f"\nGenerating {n_controls} random control manifolds...")
    
    n_channels = gesture_data[available_gestures[0]].shape[1]
    random_pcas = generate_random_control_manifolds(n_channels, n_components=12, n_controls=n_controls)
    
    # Compute control ratios
    control_ratios = []
    
    for gesture in available_gestures:
        for random_pca in random_pcas:
            cross_vaf, within_vaf, ratio = compute_cross_projection_vaf(
                gesture_data[gesture],
                random_pca,
                gesture_pcas[gesture]
            )
            control_ratios.append(ratio)
    
    control_ratios = np.array(control_ratios)
    
    # Extract observed ratios (off-diagonal elements)
    observed_ratios = []
    for i in range(n_available):
        for j in range(n_available):
            if i != j and ratio_matrix[i, j] > 0:
                observed_ratios.append(ratio_matrix[i, j])
    
    observed_ratios = np.array(observed_ratios)
    
    # Statistical comparison
    if len(observed_ratios) > 0 and len(control_ratios) > 0:
        t_stat, p_value = ttest_ind(observed_ratios, control_ratios)
        effect_size = (np.mean(observed_ratios) - np.mean(control_ratios)) / np.std(control_ratios)
    else:
        t_stat, p_value, effect_size = 0, 1, 0
    
    # Compile results
    results = {
        'subject_id': subject_id,
        'band_name': band_name,
        'gesture_names': available_gestures,  # Use available gestures
        'ratio_matrix': ratio_matrix,
        'cross_vaf_matrix': cross_vaf_matrix,
        'within_vaf_matrix': within_vaf_matrix,
        'observed_ratios': observed_ratios,
        'control_ratios': control_ratios,
        'statistics': {
            'mean_observed': np.mean(observed_ratios),
            'mean_control': np.mean(control_ratios),
            'std_observed': np.std(observed_ratios),
            'std_control': np.std(control_ratios),
            't_statistic': t_stat,
            'p_value': p_value,
            'effect_size': effect_size
        }
    }
    
    print(f"\nResults Summary:")
    print(f"  Mean cross-gesture ratio: {np.mean(observed_ratios):.3f} ± {np.std(observed_ratios):.3f}")
    print(f"  Mean control ratio: {np.mean(control_ratios):.3f} ± {np.std(control_ratios):.3f}")
    print(f"  Effect size: {effect_size:.2f}")
    print(f"  P-value: {p_value:.3e}")
    
    return results

