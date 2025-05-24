#!/usr/bin/env python
"""
Main script for neural data analysis.

This script performs highly modular brain-wide and region-specific neural data analysis 
from intracranial recordings across multiple subjects executing different hand gestures.
"""

import os
import argparse
from utils.helpers import (region_groups, ensure_dir, get_region_lists, load_region_subject_mapping, combine_regions)
from data_loading import load_region_data, save_region_data 
from region_processing import analyze_region_specific_data
from utils.run_analysis import run_region_analyses

# parse_arguments function to handle command line arguments
def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
    --------
    args : argparse.Namespace
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Neural data analysis for region-specific epochs.'
    )
    
    parser.add_argument(
        '--subjects', 
        type=int, 
        nargs='+',
        default=None,
        help='Subject IDs to use (overrides region-specific mapping when provided)'
    )

    parser.add_argument(
        '--region-subject-mapping', 
        type=str,
        default='utils/region_subject_mapping.json',
        help='Path to JSON file mapping regions to specific subject IDs (default: utils/region_subject_mapping.json)'
    )

    parser.add_argument(
        '--regions', 
        type=str, 
        nargs='+',
        choices=list(region_groups.keys()) + ["all"],
        default=["precentral-rh"],
        help='List of region groups to analyze (e.g., precentral-rh, postcentral-lh)'
    )

    parser.add_argument(
        '--bands',
        type=str,
        nargs='+',
        default=['delta', 'beta', 'high_gamma'],
        help='List of frequency bands to extract for manifold analysis'
    )
    
    parser.add_argument(
        '--trigger', 
        type=str,
        default='stim',
        choices=['stim', 'emg'],
        help='Type of trigger to use (stim or emg)'
    )
    
    parser.add_argument(
        '--tmin', 
        type=float,
        default=0.4,
        help='Start time for epochs in seconds'
    )
    
    parser.add_argument(
        '--tmax', 
        type=float,
        default=0.7,
        help='End time for epochs in seconds'
    )
    
    parser.add_argument(
        '--analysis', 
        type=str,
        default='none',
        choices=['none', 'mean-activity', 'visualize-band-power', 'tf', 'manifold', 'gesture-manifolds', 
                 'align-manifolds', 'high-dim-alignment', 'cross-region-alignment', 'all'],
        help='Type of analysis to run'
    )

    parser.add_argument(
        '--gesture-comparison',
        action='store_true',
        help='Run manifold analysis separately for each gesture class to compare spatial patterns across gestures'
    )

    parser.add_argument(
        '--similarity-components',
        type=int,
        default=20,
        help='Number of components to use for manifold similarity analysis (default: 20)'
    )

    parser.add_argument(
        '--high-dim-components',
        type=int,
        default=5,
        help='Number of components to use for high-dimensional manifold analysis'
    )

    parser.add_argument(
        '--cross-region-components',
        type=int,
        default=5,
        help='Number of components to use for cross-region manifold analysis'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str,
        default='output',
        help='Directory to save output files'
    )
    
    parser.add_argument(
        '--plot', 
        action='store_true',
        help='Plot results interactively'
    )

    parser.add_argument(
        '--n-channels',
        type=int,
        default=4,
        help='Number of channels to visualize in band-power plots'
    )
    
    parser.add_argument(
        '--n-epochs',
        type=int,
        default=4,
        help='Number of epochs to visualize in band-power plots'
    )

    parser.add_argument(
        '--cache',
        action='store_true',
        help='Enable caching of processed data'
    )

    parser.add_argument(
        '--force-recompute',
        action='store_true',
        help='Force recomputation even if cached data exists'
    )

    parser.add_argument(
        '--region-grouping',
        action='store_true',
        help='Analyze all region groups together (brain-wide analysis)'
    )

    parser.add_argument(
        '--tsne',
        action='store_true',
        help='Create t-SNE visualizations of gesture trials'
    )
    
    parser.add_argument(
        '--tsne-perplexity',
        type=float,
        default=None,
        help='Perplexity parameter for t-SNE (default: auto-calculated as sqrt(n_samples))'
    )

    parser.add_argument(
        '--classify-lfo',
        action='store_true',
        help='Run SVM classification of gestures from LFO mean activity'
    )

    parser.add_argument(
        '--classify-lfo-multiclass',
        action='store_true',
        help='Run multi-class SVM classification (default: runs both pairwise and multi-class)'
    )

    parser.add_argument(
        '--classify-lfo-pairwise',
        action='store_true',
        help='Run pairwise SVM classification (default: runs both pairwise and multi-class)'
    )

    parser.add_argument(
        '--classify-lfo-folds',
        type=int,
        default=5,
        help='Number of cross-validation folds for classification (default: 5)'
    )

    parser.add_argument(
        '--classify-lfo-permutations',
        type=int,
        default=100,
        help='Number of permutations for statistical testing (default: 100, 0 to disable)'
    )

    parser.add_argument(
        '--classify-lfo-no-pca',
        action='store_true',
        help='Disable PCA dimensionality reduction before classification'
    )

    parser.add_argument(
        '--classify-lfo-pca-components',
        type=float,
        default=0.95,
        help='Number of PCA components to use or variance to explain (default: 0.95)'
    )
    
    return parser.parse_args()

# main function to run the analysis
def main():
    """
    Main function for neural data analysis.
    """
    # Parse command line arguments
    args = parse_arguments()

    # Create cache directory
    cache_dir = os.path.join(args.output_dir, 'cache')
    ensure_dir(cache_dir)
    
    # Create output directories
    ensure_dir(args.output_dir)
    bandpower_dir = os.path.join(args.output_dir, 'band_power_plots')
    tf_dir = os.path.join(args.output_dir, 'tf_plots')
    manifold_dir = os.path.join(args.output_dir, 'manifold_plots')
    gesture_dir = os.path.join(args.output_dir, 'gesture_manifold_plots')
    align_dir = os.path.join(args.output_dir, 'aligned_manifolds_plots')
    ensure_dir(bandpower_dir)
    ensure_dir(tf_dir)
    ensure_dir(manifold_dir)
    ensure_dir(gesture_dir)
    ensure_dir(align_dir)

    # Default subject list - only used if JSON mapping fails AND no subjects are specified
    default_subject_list = [2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 
                           24, 25, 26, 29, 30, 31, 32, 34, 35, 36, 37, 39, 41, 45]

    # Load region-subject mapping if provided
    region_subject_mapping = load_region_subject_mapping(args.region_subject_mapping, default_subject_list)
    
    # Get region lists to analyze
    region_lists = get_region_lists(args)
    
    # Define default parameters
    use_default_subjects = args.subjects is not None  # Flag to indicate if default subjects should be used
    global_subject_list = args.subjects if args.subjects is not None else default_subject_list
    sampling_frequency = 1000
    trigger_type = args.trigger
    frequency_bands = args.bands
    tmin = args.tmin
    tmax = args.tmax
    
    # Define event dictionaries
    event_dict_gest = {
        "elbow": 1,
        "scissor": 2,
        "rock": 3,
        "rotation": 4,
        "thumb": 5
    }
    mapping_events = {1: "elbow", 2: "scissor", 3: "rock", 4: "rotation", 5: "thumb"}
    
    # Initialize dictionaries to store all region results
    all_region_epochs = {}
    all_region_channels_dict = {}
    
    # Extract and analyze data for each region group individually
    for group_name, region_labels in region_lists.items():
        print(f"\n===== Processing Region Group: {group_name} =====")
        
        # Determine which subjects to use for this region
        if use_default_subjects:
            subject_id_list = global_subject_list
            print(f"Using command-line specified subjects for {group_name}: {subject_id_list}")
        elif group_name in region_subject_mapping:
            subject_id_list = region_subject_mapping[group_name]
            print(f"Using region-specific subjects from mapping file for {group_name}: {subject_id_list}")
        else:
            subject_id_list = global_subject_list
            print(f"No mapping found for {group_name}, using default subjects: {subject_id_list}")
        
        print(f"Extracting data for regions: {region_labels}")
        
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
        
        # Extract and analyze region-specific data
        use_cache = args.cache and not args.force_recompute
        cached_data = None

        if use_cache:
            # Check if data is already cached
            cached_data = load_region_data(
                region_labels,
                subject_id_list, 
                sampling_frequency,
                trigger_type,
                tmin,
                tmax,
                bands_to_process=frequency_bands,
                base_dir=cache_dir
            )

        if cached_data is not None:
            # Use cached data
            region_epochs, region_channels_dict = cached_data
            print(f"Using cached data for region group: {group_name}")
        else:
            # Extract data if not cached or forced recomputation
            if args.force_recompute:
                print(f"Forced recomputation requested. Extracting region-specific data...")
            elif not args.cache:
                print(f"Caching disabled. Extracting region-specific data...")
            else:
                print(f"No cached data found. Extracting region-specific data...")
                
            region_epochs, region_channels_dict = analyze_region_specific_data(
                region_labels,
                subject_id_list,  # Now using region-specific subject list
                sampling_frequency,
                mapping_events,
                event_dict_gest,
                trigger_type,
                tmin,
                tmax,
                bands_to_process=frequency_bands,
                plot=args.plot,
                band_to_plot='delta'
            )
            
            # Save the extracted data to cache if caching is enabled
            if args.cache:
                save_region_data(
                    region_epochs,
                    region_channels_dict,
                    region_labels,
                    subject_id_list,
                    sampling_frequency,
                    trigger_type,
                    tmin,
                    tmax,
                    bands_to_process=frequency_bands,
                    base_dir=cache_dir
                )

        # # Plot all frequency bands for a specific subject - DELETE THIS LATER
        # from region_processing.epochs_extractor import plot_all_frequency_bands
        # plot_all_frequency_bands(region_epochs, region_channels_dict, event_dict_gest, subject_id=41)
        
        # Store the results for this region group
        all_region_epochs[group_name] = region_epochs
        all_region_channels_dict[group_name] = region_channels_dict
        
        # Print a summary of the results
        print(f"\nSummary of extracted data for {group_name}:")
        print(f"Total subjects with data: {len(region_epochs)}")

        for subject_id in region_epochs:
            # Get the first available frequency band
            first_band = next(iter(region_epochs[subject_id]))
            print(f"\nSubject {subject_id}:")
            print(f"  Number of channels: {len(region_channels_dict[subject_id])}")
            print(f"  Number of epochs: {len(region_epochs[subject_id][first_band])}")
            print(f"  Epoch duration: {region_epochs[subject_id][first_band].times[0]:.2f}s to {region_epochs[subject_id][first_band].times[-1]:.2f}s")
            print(f"  Number of time points: {len(region_epochs[subject_id][first_band].times)}")
            
            # Print event counts
            for event_name, event_id in event_dict_gest.items():
                event_count = len(region_epochs[subject_id][first_band][event_name])
                print(f"  {event_name} events: {event_count}")

        # Run individual region analyses here if not using region-grouping...
        if not args.region_grouping:
            # Run analysis for this specific region group
            run_region_analyses(
                args=args,
                region_epochs=region_epochs,
                region_channels_dict=region_channels_dict,
                region_labels=region_labels,
                group_name=group_name,
                subject_id_list=subject_id_list,
                event_dict_gest=event_dict_gest,
                mapping_events=mapping_events,
                frequency_bands=frequency_bands,
                trigger_type=trigger_type,
                sampling_frequency=sampling_frequency,
                bandpower_dir=bandpower_dir,
                tf_dir=tf_dir,
                manifold_dir=manifold_dir,
                gesture_dir=gesture_dir,
                align_dir=align_dir,
                all_region_epochs=all_region_epochs,
                all_region_channels_dict=all_region_channels_dict
            )

    # Extract and analyze data for all region groups together (brain-wide analysis)
    if args.region_grouping and len(all_region_epochs) > 0:
        print("\n===== Processing All Region Groups Together (Brain-Wide Analysis) =====")
        
        # Combine regions into a brain-wide analysis
        brain_wide_epochs, brain_wide_channels_dict = combine_regions(
            all_region_epochs, 
            all_region_channels_dict
        )
        
        # Create output directories for brain-wide analysis
        brain_wide_dir = "brain_wide"
        brain_wide_bandpower_dir = os.path.join(bandpower_dir, brain_wide_dir)
        brain_wide_tf_dir = os.path.join(tf_dir, brain_wide_dir)
        brain_wide_manifold_dir = os.path.join(manifold_dir, brain_wide_dir)
        brain_wide_gesture_dir = os.path.join(gesture_dir, brain_wide_dir)
        brain_wide_align_dir = os.path.join(align_dir, brain_wide_dir)
        ensure_dir(brain_wide_bandpower_dir)
        ensure_dir(brain_wide_tf_dir)
        ensure_dir(brain_wide_manifold_dir)
        ensure_dir(brain_wide_gesture_dir)
        ensure_dir(brain_wide_align_dir)
        
        # Get a list of all region labels for the brain-wide analysis
        all_region_labels = []
        for regions in region_lists.values():
            all_region_labels.extend(regions)
        
        # Print a summary of the combined data
        print(f"\nSummary of brain-wide analysis data:")
        print(f"Total subjects with data: {len(brain_wide_epochs)}")
        print(f"Regions included: {', '.join(all_region_labels)}")
        
        for subject_id in brain_wide_epochs:
            # Get the first available frequency band
            first_band = next(iter(brain_wide_epochs[subject_id]))
            print(f"\nSubject {subject_id}:")
            print(f"  Number of channels: {len(brain_wide_channels_dict[subject_id])}")
            print(f"  Number of epochs: {len(brain_wide_epochs[subject_id][first_band])}")
            
            # Print event counts
            for event_name, event_id in event_dict_gest.items():
                try:
                    event_count = len(brain_wide_epochs[subject_id][first_band][event_name])
                    print(f"  {event_name} events: {event_count}")
                except KeyError:
                    print(f"  {event_name} events: not available")
        
        # Run the analyses with the combined data
        run_region_analyses(
            args=args,
            region_epochs=brain_wide_epochs,
            region_channels_dict=brain_wide_channels_dict,
            region_labels=all_region_labels,
            group_name=brain_wide_dir,
            subject_id_list=global_subject_list,  # Use all subjects for brain-wide analysis
            event_dict_gest=event_dict_gest,
            mapping_events=mapping_events,
            frequency_bands=frequency_bands,
            trigger_type=trigger_type,
            sampling_frequency=sampling_frequency,
            bandpower_dir=bandpower_dir,
            tf_dir=tf_dir,
            manifold_dir=manifold_dir,
            gesture_dir=gesture_dir,
            align_dir=align_dir,
            all_region_epochs=all_region_epochs,
            all_region_channels_dict=all_region_channels_dict
        )

    print("\nAnalysis completed successfully!")

if __name__ == "__main__":
    main()
