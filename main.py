#!/usr/bin/env python
"""
Main script for neural data analysis.

This script performs region-specific neural data extraction and analysis
across multiple subjects, with options for time-frequency analysis and
neural manifold visualization.
"""

import os
import argparse
from utils.helpers import (
    ensure_dir, 
    get_region_lists, 
    load_region_subject_mapping,
    region_groups
)
from region_processing import analyze_region_specific_data
from analysis import (
    perform_time_frequency_analysis,
    analyze_neural_manifolds,
    analyze_gesture_manifolds,
    compare_subject_manifolds,
    visualize_signal_transform,
    analyze_high_dim_neural_manifolds,
    align_high_dim_manifolds
)
from visualization import (
    plot_tf_summary,
    plot_manifold_comparison,
    visualize_canonical_correlations,
    visualize_mode_correlations
)

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
        default=0.2,
        help='Start time for epochs in seconds'
    )
    
    parser.add_argument(
        '--tmax', 
        type=float,
        default=0.6,
        help='End time for epochs in seconds'
    )
    
    parser.add_argument(
        '--analysis', 
        type=str,
        default='none',
        choices=['none', 'visualize-band-power', 'tf', 'manifold', 'gesture', 'align-manifolds', 'high-dim-alignment', 'all'],
        help='Type of analysis to run'
    )

    parser.add_argument(
        '--high-dim-components',
        type=int,
        default=5,
        help='Number of components to use for high-dimensional manifold analysis'
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
    
    return parser.parse_args()

def main():
    """
    Main function for neural data analysis.
    """
    # Parse command line arguments
    args = parse_arguments()
    
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
    
    # Extract and analyze data for each region group
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
        region_epochs, region_channels_dict = analyze_region_specific_data(
            region_labels,
            subject_id_list,  # Now using region-specific subject list
            sampling_frequency,
            mapping_events,
            event_dict_gest,
            trigger_type,
            tmin,
            tmax,
            plot=args.plot
        )
        
        # Store the results for this region group
        all_region_epochs[group_name] = region_epochs
        all_region_channels_dict[group_name] = region_channels_dict
        
        # Print a summary of the results
        print(f"\nSummary of extracted data for {group_name}:")
        print(f"Total subjects with data: {len(region_epochs)}")

        for subject_id in region_epochs:
            print(f"\nSubject {subject_id}:")
            print(f"  Number of channels: {len(region_channels_dict[subject_id])}")
            print(f"  Number of epochs: {len(region_epochs[subject_id])}")
            print(f"  Epoch duration: {region_epochs[subject_id].times[0]:.2f}s to {region_epochs[subject_id].times[-1]:.2f}s")
            print(f"  Number of time points: {len(region_epochs[subject_id].times)}")
            
            # Print event counts
            for event_name, event_id in event_dict_gest.items():
                event_count = len(region_epochs[subject_id][event_name])
                print(f"  {event_name} events: {event_count}")

    
        ###################################### analysis ######################################

        # Run selected analyses based on command line arguments
        if args.analysis in ['none']:
            print(f"\nNo analysis method selected for {group_name}...")

        if args.analysis in ['visualize-band-power', 'all']:
            print(f"\nVisualizing signal transformation for {group_name}...")
            # Generate visualizations for each subject
            for subject_id, epochs in region_epochs.items():
                # Skip if no epochs for this subject
                if len(epochs) == 0:
                    print(f"No epochs for Subject {subject_id}, skipping...")
                    continue

                # Create a subject-specific output directory
                subject_dir = os.path.join(bandpower_dir, f"subject_{subject_id}")
                ensure_dir(subject_dir)
                
                # Region label for titles
                region_label = group_name
                
                # Run all visualizations with the high-level function
                figures = visualize_signal_transform(
                    epochs,
                    subject_id,
                    region_label,
                    bands=frequency_bands,
                    n_channels=args.n_channels,
                    n_epochs=args.n_epochs,
                    show_processing_steps=True,
                    show_multi_epoch=True,
                    show_comparative_bands=True,
                    output_dir=subject_dir
                )

        if args.analysis in ['tf', 'all']:
            print(f"\nRunning time-frequency analysis for {group_name}...")
            output_dir = None if args.plot else group_tf_dir
            # re-extract epochs for time-frequency analysis with correct tmin/tmax
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
            print(f"\nAnalyzing overall neural manifolds for {group_name}...")
            output_dir = None if args.plot else group_manifold_dir
            manifold_results = analyze_neural_manifolds(
                region_epochs,
                region_channels_dict,
                region_labels,
                bands=['delta', 'beta', 'high_gamma'],
                n_components=3,
                downsample_factor=1,
                output_dir=output_dir
            )
            
            # Generate comparison visualizations for each band
            for band_name in manifold_results:
                if len(manifold_results[band_name]) > 0:
                    # Get times from first subject's epochs (with downsampling)
                    first_subject = list(region_epochs.keys())[0]
                    times = region_epochs[first_subject].times[::1]  # No downsampling in this case
                    
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
        
        # include cca manifold aligning as a analysis parameter
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

    print("\nAnalysis completed successfully!")

if __name__ == "__main__":
    main()