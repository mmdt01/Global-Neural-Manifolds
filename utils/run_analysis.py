"""
Run full analyses on region-specific or brain-wide data.
"""

import os
import numpy as np
from scipy.stats import sem
from utils.helpers import ensure_dir
from analysis import (
    perform_time_frequency_analysis,
    analyze_neural_manifolds,
    analyze_gesture_manifolds,
    compare_subject_manifolds,
    analyze_high_dim_neural_manifolds,
    align_high_dim_manifolds,
    align_cross_region_manifolds,
    compute_region_similarity_matrix,
    analyze_mode_specific_correlations,
    compare_within_vs_cross_region_correlations,
    analyze_mean_delta_activity,
    compute_gesture_mean_activity,
    analyze_gesture_classification,
    cross_region_lfo_classification_analysis
)
from visualization import (
    plot_tf_summary,
    plot_manifold_comparison,
    visualize_canonical_correlations,
    visualize_mode_correlations,
    visualize_cross_region_correlations,
    visualize_gesture_tsne
)

def run_region_analyses(args, region_epochs, region_channels_dict, region_labels, group_name,
                        subject_id_list, event_dict_gest, mapping_events, frequency_bands,
                        trigger_type, sampling_frequency, bandpower_dir, tf_dir, manifold_dir,
                        gesture_dir, align_dir, all_region_epochs, all_region_channels_dict):
    """    
    This function encapsulates all the region-specific analyses, 
    which can be run on either individual regions or on the combined brain-wide data.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments
    region_epochs : dict
        Dictionary mapping subject IDs to epoch data
    region_channels_dict : dict
        Dictionary mapping subject IDs to channel lists
    region_labels : list
        List of region labels
    group_name : str
        Name of the region group
    subject_id_list : list
        List of subject IDs
    event_dict_gest : dict
        Dictionary mapping gesture names to event IDs
    mapping_events : dict
        Dictionary mapping event IDs to gesture names
    frequency_bands : list
        List of frequency bands
    trigger_type : str
        Type of trigger
    sampling_frequency : int
        Sampling frequency
    bandpower_dir, tf_dir, manifold_dir, gesture_dir, align_dir : str
        Output directories
    all_region_epochs : dict
        Dictionary mapping region names to dictionaries of subject-specific epochs
    all_region_channels_dict : dict
        Dictionary mapping region names to dictionaries of subject-specific channel lists
    """
    # Create group-specific output directories
    group_bandpower_dir = os.path.join(bandpower_dir, group_name)
    group_tf_dir = os.path.join(tf_dir, group_name)
    group_manifold_dir = os.path.join(manifold_dir, group_name)
    group_gesture_dir = os.path.join(gesture_dir, group_name)
    group_align_dir = os.path.join(align_dir, group_name)
    ensure_dir(group_bandpower_dir)
    ensure_dir(group_tf_dir)
    ensure_dir(group_manifold_dir)
    ensure_dir(group_gesture_dir)
    ensure_dir(group_align_dir)

    # Initialize dictionary to store classification results if it doesn't exist
    if not hasattr(run_region_analyses, 'all_region_classification_results'):
        run_region_analyses.all_region_classification_results = {}
        # Also track when we've performed cross-region analysis
        run_region_analyses.cross_region_analysis_performed = False
    
    # Run selected analyses based on command line arguments
    if args.analysis in ['none']:
        print(f"\nNo analysis method selected for {group_name}...")

    if args.analysis in ['mean-activity', 'all']:
        print(f"\nComputing static property of LFOs for {group_name}...")

        # Create output directory for mean activity analysis
        mean_activity_dir = os.path.join(args.output_dir, 'mean_activity')
        ensure_dir(mean_activity_dir)
        
        # Create group-specific output directory
        group_mean_activity_dir = os.path.join(mean_activity_dir, group_name)
        ensure_dir(group_mean_activity_dir)
        
        # Set output directory based on interactive plotting preference
        output_dir = None if args.plot else group_mean_activity_dir

        # Use the first frequency band by default
        frequency_band = frequency_bands[0]  
        print(f"Using frequency band: {frequency_band}")
        
        # Analyze mean activity (using delta band by default)
        mean_activity_results = analyze_mean_delta_activity(
            region_epochs,
            region_channels_dict,
            region_labels,
            event_dict=event_dict_gest,
            output_dir=output_dir,
            band_name=frequency_band
        )

        # Run gesture classification if requested
        if args.classify_lfo or args.classify_lfo_pairwise or args.classify_lfo_multiclass:
            print(f"\nRunning gesture classification analysis for {group_name}...")
            # Create output directory for classification analysis
            classification_dir = os.path.join(args.output_dir, 'classification')
            ensure_dir(classification_dir)
            
            # Create group-specific output directory
            group_classification_dir = os.path.join(classification_dir, group_name)
            ensure_dir(group_classification_dir)
            
            # Set output directory based on interactive plotting preference
            classification_output_dir = None if args.plot else group_classification_dir
            
            # Determine which classification types to run
            run_pairwise = args.classify_lfo or args.classify_lfo_pairwise
            run_multiclass = args.classify_lfo or args.classify_lfo_multiclass
            
            # Run classification for each subject
            classification_results = {}
            
            for subject_id, epochs_dict in region_epochs.items():
                print(f"Running classification for Subject {subject_id}...")
                
                # Prepare trial data
                _, trial_data = compute_gesture_mean_activity(
                    epochs_dict, band_name=frequency_band
                )
                
                # Create subject-specific output directory if needed
                if classification_output_dir is not None:
                    subject_classification_dir = os.path.join(classification_output_dir, f"subject_{subject_id}")
                    ensure_dir(subject_classification_dir)
                else:
                    subject_classification_dir = None
                
                # Run classification analysis
                subject_results = analyze_gesture_classification(
                    trial_data,
                    subject_id=subject_id,
                    region_label=region_labels,
                    output_dir=subject_classification_dir,
                    n_folds=args.classify_lfo_folds,
                    n_permutations=args.classify_lfo_permutations,
                    use_pca=not args.classify_lfo_no_pca,
                    pca_components=args.classify_lfo_pca_components,
                    run_pairwise=run_pairwise,
                    run_multiclass=run_multiclass,
                    n_jobs=-1  # Use all available processors
                )
                
                # Store results for this subject
                classification_results[subject_id] = subject_results

            # Store the results for this region
            run_region_analyses.all_region_classification_results[group_name] = classification_results
            
            # Print summary of classification results
            print("\nClassification analysis summary:")
            print(f"Total subjects analyzed: {len(classification_results)}")
            
            if run_multiclass:
                # Collect multiclass accuracies
                multiclass_accs = []
                multiclass_pvals = []
                
                for subject_id, results in classification_results.items():
                    if results['multiclass'] is not None and not np.isnan(results['multiclass']['accuracy']):
                        acc = results['multiclass']['accuracy']
                        pval = results['multiclass']['p_value']
                        multiclass_accs.append(acc)
                        multiclass_pvals.append(pval)
                        significance = ""
                        if not np.isnan(pval):
                            if pval < 0.05:
                                significance = "*"
                            if pval < 0.01:
                                significance = "**"
                            if pval < 0.001:
                                significance = "***"
                        print(f"  Subject {subject_id}: Multi-class accuracy = {acc:.2f} {significance}")
                
                if multiclass_accs:
                    mean_acc = np.mean(multiclass_accs)
                    std_acc = np.std(multiclass_accs)
                    sem_acc = sem(multiclass_accs)
                    significant_count = np.sum(np.array(multiclass_pvals) < 0.05)
                    
                    print(f"\nAverage multi-class accuracy: {mean_acc:.2f} ± {sem_acc:.2f} SEM")
                    print(f"Significant classification ({significant_count}/{len(multiclass_accs)} subjects)")
            
            if run_pairwise:
                # Collect pairwise accuracies for all gesture pairs
                all_pair_accuracies = {}
                
                for subject_id, results in classification_results.items():
                    if results['pairwise'] is not None and not np.all(np.isnan(results['pairwise']['accuracy_matrix'])):
                        acc_matrix = results['pairwise']['accuracy_matrix']
                        gesture_labels = results['pairwise']['gesture_labels']
                        
                        # Extract accuracies for each pair
                        for i in range(len(gesture_labels)):
                            for j in range(i+1, len(gesture_labels)):
                                g1, g2 = gesture_labels[i], gesture_labels[j]
                                pair_key = f"{g1}_vs_{g2}"
                                
                                if pair_key not in all_pair_accuracies:
                                    all_pair_accuracies[pair_key] = []
                                
                                # Add accuracy if it's valid
                                if not np.isnan(acc_matrix[i, j]):
                                    all_pair_accuracies[pair_key].append(acc_matrix[i, j])
                
                # Calculate average accuracies for each pair
                avg_pair_accuracies = {}
                
                for pair_key, accs in all_pair_accuracies.items():
                    if accs:
                        avg_acc = np.mean(accs)
                        avg_pair_accuracies[pair_key] = avg_acc
                
                # Sort pairs by average accuracy
                sorted_pairs = sorted(avg_pair_accuracies.items(), key=lambda x: x[1], reverse=True)
                
                print("\nAverage pairwise classification accuracies:")
                for pair_key, avg_acc in sorted_pairs:
                    print(f"  {pair_key}: {avg_acc:.2f}")

            print("\n===== Running Cross-Region Classification Analysis =====")
            region_comparison = cross_region_lfo_classification_analysis(
                        args, 
                        run_region_analyses.all_region_classification_results
                    )
        
        # Run t-SNE visualization for each subject
        if args.tsne:
            print(f"\nCreating t-SNE visualizations for {group_name}...")
            tsne_dir = os.path.join(args.output_dir, 'tsne_visualizations')
            ensure_dir(tsne_dir)
            
            # Create group-specific output directory
            group_tsne_dir = os.path.join(tsne_dir, group_name)
            ensure_dir(group_tsne_dir)
            
            # Set output directory based on interactive plotting preference
            tsne_output_dir = None if args.plot else group_tsne_dir
            
            # Run t-SNE for each subject
            for subject_id, epochs_dict in region_epochs.items():
                print(f"Creating t-SNE visualization for Subject {subject_id}...")
                
                # Create t-SNE visualizations
                tsne_results = visualize_gesture_tsne(
                    epochs_dict,
                    subject_id=subject_id,
                    region_label=str(region_labels),
                    band_name=frequency_band,
                    output_dir=tsne_output_dir,
                    perplexity=args.tsne_perplexity,
                    show_2d=True,
                    show_3d=True
                )
                
                if tsne_results is None:
                    print(f"Could not create t-SNE visualizations for Subject {subject_id}")
        
        # Print a summary of results
        if len(mean_activity_results) > 0:
            print("\nMean activity analysis summary:")
            print(f"Total subjects analyzed: {len(mean_activity_results)}")
            
            # Find the most consistent patterns across subjects
            gesture_pair_distances = {}
            
            # For each subject, find their most similar and dissimilar pairs
            for subject_id, result in mean_activity_results.items():
                matrix = result['distance_matrix']
                labels = result['gesture_labels']
                
                # Find most similar and dissimilar gesture pairs for this subject
                max_dist = np.max(matrix)
                # Add identity matrix with max value to avoid selecting diagonal elements (self-comparisons)
                min_dist = np.min(matrix + np.eye(len(labels)) * max_dist)
                
                # Find the gesture pairs
                for i in range(len(labels)):
                    for j in range(i+1, len(labels)):
                        pair_key = f"{labels[i]}_{labels[j]}"
                        
                        if pair_key not in gesture_pair_distances:
                            gesture_pair_distances[pair_key] = {
                                'pair': (labels[i], labels[j]),
                                'distances': [],
                                'similar_count': 0,
                                'dissimilar_count': 0
                            }
                        
                        # Store the distance for this pair
                        gesture_pair_distances[pair_key]['distances'].append(matrix[i, j])
                        
                        # Check if this is the most similar pair for this subject
                        if matrix[i, j] == min_dist:
                            gesture_pair_distances[pair_key]['similar_count'] += 1
                            print(f"  Subject {subject_id}: Most similar gestures are {labels[i]} and {labels[j]} (distance: {min_dist:.2f})")
                        
                        # Check if this is the most dissimilar pair for this subject
                        if matrix[i, j] == max_dist:
                            gesture_pair_distances[pair_key]['dissimilar_count'] += 1
                            print(f"  Subject {subject_id}: Most dissimilar gestures are {labels[i]} and {labels[j]} (distance: {max_dist:.2f})")
            
            # Calculate average distances and find most consistent patterns
            for pair_key, data in gesture_pair_distances.items():
                if len(data['distances']) > 0:
                    data['mean_distance'] = np.mean(data['distances'])
                    data['std_distance'] = np.std(data['distances'])
            
            # Find the most consistently similar and dissimilar pairs
            most_similar_pair = max(gesture_pair_distances.values(), 
                                   key=lambda x: x['similar_count'] if 'similar_count' in x else 0)
            most_dissimilar_pair = max(gesture_pair_distances.values(), 
                                      key=lambda x: x['dissimilar_count'] if 'dissimilar_count' in x else 0)
            
            # Print the most consistent patterns
            if most_similar_pair['similar_count'] > 0:
                pair = most_similar_pair['pair']
                count = most_similar_pair['similar_count']
                mean_dist = most_similar_pair['mean_distance']
                print(f"\nMost consistently similar gestures: {pair[0]} and {pair[1]}")
                print(f"  Identified as most similar in {count}/{len(mean_activity_results)} subjects")
                print(f"  Mean distance across all subjects: {mean_dist:.2f}")
            
            if most_dissimilar_pair['dissimilar_count'] > 0:
                pair = most_dissimilar_pair['pair']
                count = most_dissimilar_pair['dissimilar_count']
                mean_dist = most_dissimilar_pair['mean_distance']
                print(f"\nMost consistently dissimilar gestures: {pair[0]} and {pair[1]}")
                print(f"  Identified as most dissimilar in {count}/{len(mean_activity_results)} subjects")
                print(f"  Mean distance across all subjects: {mean_dist:.2f}")

    if args.analysis in ['tf', 'all']:
        print(f"\nRunning time-frequency analysis for {group_name}...")
        output_dir = None if args.plot else group_tf_dir
        
        # Re-extract epochs for time-frequency analysis with correct tmin/tmax
        # For brain-wide analysis, we could either:
        # 1. Combine the existing regions (faster but may cause inconsistencies)
        # 2. Extract again for the combined regions (slower but more consistent)
        # Here we choose option 1 for brain-wide analysis
        
        if group_name == "brain_wide":
            from region_processing import analyze_region_specific_data
            tf_region_epochs = {}
            tf_region_channels_dict = {}
            
            # For each subject, get the TF epochs from all regions
            for subject_id in region_epochs:
                tf_region_epochs[subject_id] = {}
                # Create an entry for each frequency band
                for band_name in next(iter(region_epochs[subject_id].values())).event_id.keys():
                    # Need to create a copy with the expanded time range
                    # This is a simplified approach - in reality TF analysis might need a full re-extract
                    tf_region_epochs[subject_id][band_name] = region_epochs[subject_id][band_name].copy()
                
                tf_region_channels_dict[subject_id] = region_channels_dict[subject_id]
        else:
            # For individual regions, extract with the expanded time range
            from region_processing import analyze_region_specific_data
            tf_region_epochs, tf_region_channels_dict = analyze_region_specific_data(
                region_labels,
                subject_id_list,
                sampling_frequency,
                mapping_events,
                event_dict_gest,
                trigger_type,
                tmin=-1.0,
                tmax=3.0,
                plot=False
            )
        
        # Perform time-frequency analysis
        tfr_power_dict = perform_time_frequency_analysis(
            tf_region_epochs,
            tf_region_channels_dict,
            region_labels,
            tmin=-1.0,
            tmax=3.0,
            output_dir=output_dir
        )
        
        # Generate summary visualization with all subjects
        plot_tf_summary(
            tfr_power_dict, 
            region_channels_dict, 
            region_labels, 
            baseline=(-0.5, 0.0),
            output_dir=group_tf_dir
        )
    
    if args.analysis in ['manifold', 'all']:
        print(f"\nAnalyzing neural manifolds for {group_name}...")
        print(f"Using frequency bands: {frequency_bands}")
        output_dir = None if args.plot else group_manifold_dir

        manifold_results = analyze_neural_manifolds(
            region_epochs,
            region_channels_dict,
            region_labels,
            bands=frequency_bands,  # Pass all frequency bands
            n_components=3,
            output_dir=output_dir
        )
        
        # Generate comparison visualizations for each band
        for band_name in manifold_results:
            if len(manifold_results[band_name]) > 0:
                # Get times from first subject's epochs for this specific band
                first_subject = next(iter(manifold_results[band_name].keys()))
                if first_subject in region_epochs and band_name in region_epochs[first_subject]:
                    times = region_epochs[first_subject][band_name].times
                    
                    # Plot comparison of subjects for this band
                    plot_manifold_comparison(
                        manifold_results[band_name], 
                        band_name, 
                        times, 
                        region_labels,
                        output_dir=group_manifold_dir
                    )
    
    if args.analysis in ['gesture-manifolds', 'all']:
        print(f"\nAnalyzing gesture-specific neural manifolds for {group_name}...")
        output_dir = None if args.plot else group_gesture_dir
        gesture_manifold_results = analyze_gesture_manifolds(
            region_epochs,
            region_channels_dict,
            region_labels,
            bands=['delta', 'beta', 'high_gamma'],
            gestures=list(event_dict_gest.keys()),
            n_components=3,
            downsample_factor=1,
            output_dir=output_dir
        )
    
    # Include cca manifold aligning as a analysis parameter
    if args.analysis in ['align-manifolds', 'all']:
        print(f"\nComputing general manifolds and aligning using CCA for {group_name}...")
        output_dir = None if args.plot else group_align_dir
        # step 1: compute overall manifold
        print("... Step 1: computing neural manifolds ...")
        manifold_results = analyze_neural_manifolds(
            region_epochs,
            region_channels_dict,
            region_labels,
            frequency_bands,
            n_components=3,
            downsample_factor=1,
            output_dir=output_dir
        )
        # step 2: align and compare manifolds between subjects
        print("... Step 2: aligning manifolds across subjects using CCA ...")
        cca_results = compare_subject_manifolds(
            manifold_results,
            subject_id_list, 
            frequency_bands,
            output_dir=output_dir
        )
        # step 3: visualize the CCA results
        print("... Step 3: visualizing the CCA results ...")
        visualize_canonical_correlations(
            cca_results, 
            frequency_bands,
            output_dir=output_dir
        )

    if args.analysis in ['high-dim-alignment', 'all']:
        print(f"\nRunning high-dimensional manifold alignment ({args.high_dim_components} components) for {group_name}...")
        high_dim_dir = os.path.join(args.output_dir, 'high_dim_manifolds')
        ensure_dir(high_dim_dir)
        
        # Create region-specific output directory
        group_high_dim_dir = os.path.join(high_dim_dir, group_name)
        ensure_dir(group_high_dim_dir)
        
        # Set output directory based on interactive plotting preference
        output_dir = None if args.plot else group_high_dim_dir
        
        # Step 1: compute high-dimensional neural manifolds
        print("... Step 1: computing high-dimensional neural manifolds ...")
        manifold_results = analyze_high_dim_neural_manifolds(
            region_epochs,
            region_channels_dict,
            region_labels,
            frequency_bands,
            n_components=args.high_dim_components,
            downsample_factor=1,
            output_dir=output_dir
        )
        
        # Step 2: align manifolds between subjects
        print("... Step 2: aligning high-dimensional manifolds across subjects using CCA ...")
        cca_results = align_high_dim_manifolds(
            manifold_results,
            subject_id_list, 
            frequency_bands,
            output_dir=output_dir
        )
        
        # Step 3: visualize the mode-specific correlations
        print("... Step 3: visualizing the mode-specific correlations ...")
        visualize_mode_correlations(
            cca_results, 
            frequency_bands,
            output_dir=output_dir
        )

    if args.analysis in ['cross-region-alignment', 'all'] and group_name != "brain_wide":
        print(f"\nRunning cross-region manifold alignment analysis...")
        
        # Check if we have at least 2 regions for comparison
        if len(all_region_epochs) < 2:
            print("ERROR: Cross-region alignment requires at least 2 regions. Please provide multiple regions using --regions.")
        else:
            # Create output directory
            cross_region_dir = os.path.join(args.output_dir, 'cross_region_manifolds')
            ensure_dir(cross_region_dir)
            
            # Set output directory based on interactive plotting preference
            output_dir = None if args.plot else cross_region_dir
            
            # Step 1: Align manifolds across regions and subjects
            print("... Step 1: aligning neural manifolds across regions and subjects ...")
            cross_region_results, region_manifold_results = align_cross_region_manifolds(
                all_region_epochs,
                all_region_channels_dict,
                all_region_epochs.keys(),  # Use all regions we have
                bands=frequency_bands,
                n_components=args.cross_region_components,
                downsample_factor=1,
                output_dir=output_dir
            )
            
            # Step 2: Compute region similarity matrices
            print("... Step 2: computing region similarity matrices ...")
            similarity_matrices = compute_region_similarity_matrix(
                cross_region_results,
                bands=frequency_bands,
                method='mean'
            )
            
            # Step 3: Analyze mode-specific correlations
            print("... Step 3: analyzing mode-specific correlations ...")
            mode_correlations = analyze_mode_specific_correlations(
                cross_region_results,
                bands=frequency_bands,
                n_modes=args.cross_region_components
            )
            
            # Step 4: Compare within-region vs cross-region correlations
            print("... Step 4: comparing within-region vs cross-region correlations ...")
            comparison_results = compare_within_vs_cross_region_correlations(
                mode_correlations,
                bands=frequency_bands,
                n_modes=args.cross_region_components
            )
            
            # Step 5: Visualize the results
            print("... Step 5: visualizing cross-region correlation results ...")
            visualize_cross_region_correlations(
                cross_region_results,
                similarity_matrices,
                mode_correlations,
                comparison_results,
                bands=frequency_bands,
                output_dir=output_dir
            )





