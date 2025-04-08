#!/usr/bin/env python
"""
Main script for neural data analysis.

This script performs region-specific neural data extraction and analysis
across multiple subjects, with options for time-frequency analysis and
neural manifold visualization.
"""

import os
import argparse
from utils.helpers import ensure_dir

from region_processing import analyze_region_specific_data
from analysis import (
    perform_time_frequency_analysis,
    analyze_neural_manifolds,
    analyze_gesture_manifolds,
    compare_subject_manifolds
)
from visualization import (
    plot_tf_summary,
    plot_manifold_comparison,
    visualize_canonical_correlations
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
        default=[2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 29, 30, 31, 32, 34, 35, 36, 37, 39, 41, 45], # all subjects
        help='List of subject IDs to analyze'
    )
    
    parser.add_argument(
        '--regions', 
        type=str, 
        nargs='+',
        default=["ctx-rh-precentral", "wm-rh-precentral"],
        help='List of brain region labels to extract data for'
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
        default='all',
        choices=['tf', 'manifold', 'gesture', 'align-manifolds', 'all'],
        help='Type of analysis to run'
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
    
    return parser.parse_args()

def main():
    """
    Main function for neural data analysis.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Create output directories
    ensure_dir(args.output_dir)
    tf_dir = os.path.join(args.output_dir, 'tf_plots')
    manifold_dir = os.path.join(args.output_dir, 'manifold_plots')
    gesture_dir = os.path.join(args.output_dir, 'gesture_manifold_plots')
    align_dir = os.path.join(args.output_dir, 'aligned_manifolds_plots')
    ensure_dir(tf_dir)
    ensure_dir(manifold_dir)
    ensure_dir(gesture_dir)
    ensure_dir(align_dir)
    
    # Define parameters
    region_labels = args.regions
    subject_id_list = args.subjects
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
    
    # Extract and analyze region-specific data
    print(f"Extracting data for regions: {region_labels}")
    print(f"Using subjects: {subject_id_list}")
    
    region_epochs, region_channels_dict = analyze_region_specific_data(
        region_labels,
        subject_id_list,
        sampling_frequency,
        mapping_events,
        event_dict_gest,
        trigger_type,
        tmin,
        tmax,
        plot=args.plot
    )
    
    # Print a summary of the results
    print("\nSummary of extracted region-specific data:")
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
    
    ################### analysis ###################

    # Run selected analyses based on command line arguments
    if args.analysis in ['tf', 'all']:
        print("\nRunning time-frequency analysis...")
        output_dir = None if args.plot else tf_dir
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
            output_dir=tf_dir
        )
    
    if args.analysis in ['manifold', 'all']:
        print("\nAnalyzing overall neural manifolds...")
        output_dir = None if args.plot else manifold_dir
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
                    output_dir=manifold_dir
                )
    
    if args.analysis in ['gesture-manifolds', 'all']:
        print("\nAnalyzing gesture-specific neural manifolds...")
        output_dir = None if args.plot else gesture_dir
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
        print("\nComputing general manifolds and aligning using CCA ...")
        output_dir = None if args.plot else align_dir
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


    print("\nAnalysis completed successfully!")

if __name__ == "__main__":
    main()
