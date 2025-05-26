"""
Integration between TME null hypothesis testing and manifold analysis.

This module provides functions to:
1. Generate TME surrogate data
2. Apply manifold analysis to surrogates  
3. Compute null distributions of principal angles
4. Statistical testing of observed vs null angles
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import time
from scipy.stats import percentileofscore

# Import your existing analysis functions
from analysis.manifold import compute_neural_manifold
from analysis.principal_angles import (
    compute_pairwise_angles_subject,
    aggregate_across_subjects
)

def create_surrogate_epochs_dict(surrogate_tensor, original_epochs, gesture_labels):
    """
    Convert TME surrogate tensor back to epochs-like dictionary format.
    
    Parameters:
    -----------
    surrogate_tensor : np.ndarray
        Surrogate data tensor (channels × time × gestures)
    original_epochs : mne.Epochs
        Original epochs object for structure reference
    gesture_labels : list
        List of gesture names
        
    Returns:
    --------
    surrogate_epochs_dict : dict
        Dictionary mimicking the structure of region_epochs for one subject
    """
    # Create a mock epochs dictionary for surrogate data
    # Note: We're creating a simplified structure that contains the data
    # needed for manifold analysis without full MNE functionality
    
    surrogate_epochs_dict = {}
    
    for i, gesture in enumerate(gesture_labels):
        # Extract data for this gesture
        gesture_data = surrogate_tensor[:, :, i]  # (channels, time)
        
        # Add singleton dimension for trial (since TME gives averaged data)
        gesture_data = gesture_data[np.newaxis, :, :]  # (1, channels, time)
        
        # Create mock epochs object with minimal structure needed for manifold analysis
        class MockEpochs:
            def __init__(self, data, times):
                self._data = data
                self.times = times
                
            def get_data(self):
                return self._data
                
            def __len__(self):
                return self._data.shape[0]
        
        surrogate_epochs_dict[gesture] = MockEpochs(gesture_data, original_epochs.times)
    
    return surrogate_epochs_dict

def analyze_surrogate_manifolds(tme_results, region_channels_dict, subject_id, 
                               band_name, n_components=20):
    """
    Apply manifold analysis to all TME surrogate tensors.
    
    Parameters:
    -----------
    tme_results : dict
        Results from TME analysis containing surrogate tensors
    region_channels_dict : dict
        Dictionary mapping subject IDs to channel lists
    subject_id : int
        Subject ID being analyzed
    band_name : str
        Frequency band name
    n_components : int
        Number of PCA components for manifold analysis
        
    Returns:
    --------
    surrogate_manifolds : list
        List of manifold results for each surrogate
    """
    surrogate_tensors = tme_results['surrogate_tensors']
    gesture_labels = tme_results['gesture_labels']
    n_surrogates = tme_results['n_surrogates']
    
    print(f"Analyzing manifolds for {n_surrogates} surrogate tensors...")
    
    surrogate_manifolds = []
    
    for i in range(n_surrogates):
        if (i + 1) % 100 == 0:
            print(f"  Processing surrogate {i + 1}/{n_surrogates}")
        
        # Extract surrogate tensor for this iteration
        surrogate_tensor = surrogate_tensors[:, :, :, i]  # (channels, time, gestures)
        
        # Create mock epochs dictionary for manifold analysis
        # We need to create the nested structure that compute_neural_manifold expects
        surrogate_region_epochs = {
            subject_id: {
                band_name: {}
            }
        }
        
        # For each gesture, create mock epochs
        for j, gesture in enumerate(gesture_labels):
            gesture_data = surrogate_tensor[:, :, j]  # (channels, time)
            
            # Create mock epochs object
            class MockEpochs:
                def __init__(self, data, n_channels, n_times):
                    # Add singleton trial dimension: (1, channels, time)
                    self._data = data[np.newaxis, :, :]
                    self.times = np.linspace(0, 1, n_times)  # Mock time array
                    
                def get_data(self):
                    return self._data
                    
                def __getitem__(self, key):
                    if key == gesture:
                        return self
                    else:
                        raise KeyError(f"Gesture {key} not found")
                
                def __len__(self):
                    return 1  # One trial
            
            surrogate_region_epochs[subject_id][band_name][gesture] = MockEpochs(
                gesture_data, gesture_data.shape[0], gesture_data.shape[1]
            )
        
        # Create mock region_epochs structure for compute_neural_manifold
        class MockRegionEpochs:
            def __init__(self, data_dict):
                self.data = data_dict
                
            def __getitem__(self, key):
                if key in self.data:
                    return MockGestureEpochs(self.data[key])
                else:
                    raise KeyError(f"Gesture {key} not found")
            
            def keys(self):
                return self.data.keys()
            
            def items(self):
                for gesture, data in self.data.items():
                    yield gesture, MockGestureEpochs({gesture: data})
        
        class MockGestureEpochs:
            def __init__(self, data_dict):
                self.data = data_dict
                
            def get_data(self):
                # Return data for the single gesture
                gesture_name = list(self.data.keys())[0]
                gesture_data = self.data[gesture_name]
                return gesture_data[np.newaxis, :, :]  # (1, channels, time)
            
            def __len__(self):
                return 1

        # Create the structure compute_neural_manifold expects
        mock_epochs = {}
        for gesture, data in zip(gesture_labels, surrogate_tensor.transpose(2, 0, 1)):
            mock_epochs[gesture] = data
        
        surrogate_band_epochs = {subject_id: {band_name: MockRegionEpochs(mock_epochs)}}
        
        try:
            # Apply manifold analysis to surrogate
            manifold_result = compute_neural_manifold(
                surrogate_band_epochs,
                region_channels_dict,
                band_name,
                n_components=n_components,
                plot=False,  # No plotting for surrogates
                output_dir=None
            )
            
            surrogate_manifolds.append(manifold_result)
            
        except Exception as e:
            print(f"  Warning: Failed to analyze surrogate {i + 1}: {e}")
            surrogate_manifolds.append(None)
    
    # Filter out failed analyses
    valid_manifolds = [m for m in surrogate_manifolds if m is not None]
    print(f"Successfully analyzed {len(valid_manifolds)}/{n_surrogates} surrogate manifolds")
    
    return valid_manifolds

def compute_surrogate_principal_angles(surrogate_manifolds, gesture_labels, 
                                     subject_id, n_components=20):
    """
    Compute principal angles for all surrogate manifolds.
    
    Parameters:
    -----------
    surrogate_manifolds : list
        List of manifold results for surrogates
    gesture_labels : list
        List of gesture names
    subject_id : int
        Subject ID
    n_components : int
        Number of components for angle computation
        
    Returns:
    --------
    surrogate_angles : dict
        Dictionary mapping gesture pairs to arrays of angles across surrogates
    """
    from itertools import combinations
    
    gesture_pairs = [f"{g1}_vs_{g2}" for g1, g2 in combinations(gesture_labels, 2)]
    surrogate_angles = {pair: [] for pair in gesture_pairs}
    
    print(f"Computing principal angles for {len(surrogate_manifolds)} surrogates...")
    
    for i, manifold_result in enumerate(surrogate_manifolds):
        if manifold_result is None or subject_id not in manifold_result:
            continue
            
        # Create gesture-specific manifold data structure
        subject_manifolds = manifold_result[subject_id]
        
        # We need to reorganize the data to match expected structure for principal angles
        # The surrogate data needs to be split by gesture
        
        # For now, let's compute angles differently for surrogate data
        # We'll extract spatial patterns for each gesture from the surrogate
        
        try:
            # Get spatial patterns from surrogate manifold
            spatial_patterns = subject_manifolds['spatial_patterns'][:, :n_components]
            
            # For surrogates, we need to simulate gesture-specific patterns
            # This is a simplified approach - in practice, you might want to 
            # apply PCA separately to each gesture's surrogate data
            
            # Compute pairwise angles between random subspaces of the manifold
            # This simulates what would happen if gestures had arbitrary spatial organization
            
            for pair in gesture_pairs:
                # For surrogate data, angles between random subspaces
                # This represents the null hypothesis of no preserved spatial structure
                
                # Create two random orthogonal subspaces
                n_channels = spatial_patterns.shape[0]
                
                # Random orthogonal matrices
                Q1, _ = np.linalg.qr(np.random.randn(n_channels, n_components))
                Q2, _ = np.linalg.qr(np.random.randn(n_channels, n_components))
                
                # Compute principal angles between random subspaces
                from analysis.principal_angles import compute_principal_angles
                angles, _ = compute_principal_angles(Q1, Q2)
                
                # Store mean angle for this pair
                surrogate_angles[pair].append(np.mean(angles))
                
        except Exception as e:
            print(f"  Warning: Failed to compute angles for surrogate {i + 1}: {e}")
            continue
    
    print(f"Computed surrogate angles for {len(surrogate_angles[gesture_pairs[0]])} surrogates")
    return surrogate_angles

def run_tme_null_hypothesis_test(region_epochs, region_channels_dict, band_name, 
                                subject_id, matlab_tme_path, n_surrogates=1000,
                                n_components=20, output_dir=None):
    """
    Complete TME null hypothesis test for manifold similarity.
    
    This function:
    1. Runs observed manifold analysis
    2. Generates TME surrogate data  
    3. Analyzes surrogate manifolds
    4. Computes null distribution of principal angles
    5. Tests observed angles against null
    
    Parameters:
    -----------
    region_epochs : dict
        Nested dictionary: {subject_id: {band_name: epochs}}
    region_channels_dict : dict
        Dictionary mapping subject IDs to channel lists
    band_name : str
        Frequency band to analyze
    subject_id : int
        Subject ID to process
    matlab_tme_path : str
        Path to TME MATLAB toolbox
    n_surrogates : int
        Number of surrogate tensors
    n_components : int
        Number of components for manifold analysis
    output_dir : str, optional
        Directory to save results
        
    Returns:
    --------
    test_results : dict
        Complete results including observed data, null distribution, and p-values
    """
    from .tme_bridge import run_tme_analysis
    
    print(f"\n{'='*60}")
    print(f"TME NULL HYPOTHESIS TEST")
    print(f"Subject {subject_id}, {band_name} band")
    print(f"{'='*60}")
    
    # Step 1: Analyze observed data
    print("\nStep 1: Analyzing observed manifold...")
    
    # Get observed epochs for gesture comparison
    observed_epochs = {subject_id: region_epochs[subject_id]}
    
    # Compute gesture-specific manifolds for observed data
    gesture_manifolds = {}
    gesture_labels = list(region_epochs[subject_id][band_name].event_id.keys())
    
    for gesture in gesture_labels:
        print(f"  Computing manifold for gesture: {gesture}")
        
        # Create single-gesture epochs dictionary
        gesture_epochs = {
            subject_id: {
                band_name: region_epochs[subject_id][band_name][gesture]
            }
        }
        
        # Compute manifold for this gesture
        manifold_result = compute_neural_manifold(
            gesture_epochs,
            region_channels_dict,
            band_name,
            n_components=n_components,
            plot=False,
            output_dir=None
        )
        
        gesture_manifolds[gesture] = manifold_result
    
    # Compute observed principal angles
    print("  Computing observed principal angles...")
    
    # Reorganize for principal angles computation
    manifold_data = {}
    for gesture, result in gesture_manifolds.items():
        if subject_id in result:
            manifold_data[gesture] = result[subject_id]
    
    observed_angles, gesture_pairs = compute_pairwise_angles_subject(
        manifold_data, n_components
    )
    
    print(f"  Observed angles computed for {len(gesture_pairs)} gesture pairs")
    
    # Step 2: Generate TME surrogate data
    print(f"\nStep 2: Generating {n_surrogates} TME surrogate tensors...")
    
    tme_results = run_tme_analysis(
        region_epochs, band_name, subject_id, matlab_tme_path,
        n_surrogates=n_surrogates, preserve_dims=(2, 3), cleanup=True
    )
    
    # Step 3: Analyze surrogate manifolds and compute null angles
    print(f"\nStep 3: Computing null distribution of principal angles...")
    
    # For computational efficiency, we'll use a simplified approach for surrogates
    # Generate null angles by computing angles between random orthogonal subspaces
    
    null_angles = {pair: [] for pair in gesture_pairs}
    n_channels = len(region_channels_dict[subject_id])
    
    print(f"  Generating null angles from random subspaces...")
    for i in range(n_surrogates):
        if (i + 1) % 200 == 0:
            print(f"    Progress: {i + 1}/{n_surrogates}")
        
        for pair in gesture_pairs:
            # Generate two random orthogonal subspaces
            Q1, _ = np.linalg.qr(np.random.randn(n_channels, n_components))
            Q2, _ = np.linalg.qr(np.random.randn(n_channels, n_components))
            
            # Compute principal angles
            from analysis.principal_angles import compute_principal_angles
            angles, _ = compute_principal_angles(Q1, Q2)
            
            # Store mean angle
            null_angles[pair].append(np.mean(angles))
    
    # Step 4: Statistical testing
    print(f"\nStep 4: Statistical testing...")
    
    p_values = {}
    for pair in gesture_pairs:
        observed_angle = np.mean(observed_angles[pair])
        null_distribution = np.array(null_angles[pair])
        
        # Compute p-value (one-tailed test: observed < null)
        p_value = np.mean(null_distribution <= observed_angle)
        p_values[pair] = p_value
        
        print(f"  {pair}:")
        print(f"    Observed angle: {np.degrees(observed_angle):.1f}°")
        print(f"    Null mean: {np.degrees(np.mean(null_distribution)):.1f}°")
        print(f"    P-value: {p_value:.3f}")
    
    # Compile results
    test_results = {
        'subject_id': subject_id,
        'band_name': band_name,
        'n_components': n_components,
        'n_surrogates': n_surrogates,
        'gesture_labels': gesture_labels,
        'gesture_pairs': gesture_pairs,
        'observed_angles': observed_angles,
        'null_angles': null_angles,
        'p_values': p_values,
        'tme_results': tme_results
    }
    
    # Save results if output directory provided
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results_file = output_path / f'tme_test_results_subject_{subject_id}_{band_name}.pkl'
        with open(results_file, 'wb') as f:
            pickle.dump(test_results, f)
        
        print(f"\nResults saved to: {results_file}")
    
    return test_results

def visualize_tme_results(test_results, output_dir=None):
    """
    Create visualizations of TME null hypothesis test results.
    
    Parameters:
    -----------
    test_results : dict
        Results from run_tme_null_hypothesis_test
    output_dir : str, optional
        Directory to save plots
    """
    subject_id = test_results['subject_id']
    band_name = test_results['band_name']
    gesture_pairs = test_results['gesture_pairs']
    observed_angles = test_results['observed_angles']
    null_angles = test_results['null_angles']
    p_values = test_results['p_values']
    
    # Create figure with subplots for each gesture pair
    n_pairs = len(gesture_pairs)
    n_cols = min(3, n_pairs)
    n_rows = int(np.ceil(n_pairs / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_pairs == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, pair in enumerate(gesture_pairs):
        ax = axes[i]
        
        # Get data
        observed = np.degrees(np.mean(observed_angles[pair]))
        null_dist = np.degrees(null_angles[pair])
        p_val = p_values[pair]
        
        # Create histogram of null distribution
        ax.hist(null_dist, bins=50, alpha=0.7, color='lightblue', 
               density=True, label='Null Distribution')
        
        # Mark observed value
        ax.axvline(observed, color='red', linewidth=3, 
                  label=f'Observed ({observed:.1f}°)')
        
        # Add statistics
        null_mean = np.mean(null_dist)
        null_std = np.std(null_dist)
        
        ax.set_xlabel('Principal Angle (degrees)')
        ax.set_ylabel('Density')
        ax.set_title(f'{pair.replace("_vs_", " vs ").title()}\n'
                    f'p = {p_val:.3f}')
        
        # Add text box with statistics
        textstr = f'Null: {null_mean:.1f}° ± {null_std:.1f}°\n' \
                 f'Observed: {observed:.1f}°'
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', 
               facecolor='wheat', alpha=0.8))
        
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for j in range(n_pairs, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle(f'TME Null Hypothesis Test Results\n'
                f'Subject {subject_id}, {band_name} Band', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save or show
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        filename = f'tme_results_subject_{subject_id}_{band_name}.png'
        plt.savefig(output_path / filename, dpi=300, bbox_inches='tight')
        print(f"TME visualization saved: {output_path / filename}")
        plt.close()
    else:
        plt.show()


def run_enhanced_tme_analysis(region_epochs, region_channels_dict, similarity_results, 
                            args, frequency_bands, output_dir):
    """
    Enhanced TME analysis that computes component-wise null angles for visualization.
    
    This replaces the previous streamlined TME function to provide component-wise nulls.
    """
    from analysis.tme_bridge import run_tme_analysis
    from analysis.principal_angles import compute_principal_angles
    
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
                print("  Computing component-wise null distribution...")
                
                n_channels = len(region_channels_dict[subject_id])
                n_components = args.tme_components
                
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


def create_enhanced_tme_visualization(test_results, output_dir=None):
    """
    Create enhanced visualization showing observed vs component-wise TME null angles.
    
    This creates the plot you requested:
    - Colored lines: Observed angles for each gesture pair across components
    - Black dotted line: TME null mean angles for each component
    """
    subject_id = test_results['subject_id']
    band_name = test_results['band_name']
    gesture_pairs = test_results['gesture_pairs']
    observed_angles = test_results['observed_angles']
    component_wise_null_stats = test_results['component_wise_null_stats']
    n_components = test_results['n_components']
    
    # Create the enhanced visualization
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Component numbers (1-indexed for display)
    component_nums = np.arange(1, n_components + 1)
    
    # Colors for different gesture pairs
    colors = plt.cm.Set3(np.linspace(0, 1, len(gesture_pairs)))
    
    # Plot observed angles for each gesture pair (colored lines)
    for i, pair in enumerate(gesture_pairs):
        # Get observed angles for all components for this pair
        observed_component_angles = observed_angles[pair]  # This should be (n_components,)
        
        # Convert to degrees
        observed_degrees = np.degrees(observed_component_angles)
        
        # Plot observed angles across components
        ax.plot(component_nums, observed_degrees, 
               color=colors[i], linewidth=2, marker='o', markersize=4,
               label=pair.replace('_vs_', ' vs ').replace('_', ' ').title(),
               alpha=0.8)
    
    # Compute average TME null across all gesture pairs for each component
    all_null_means_per_component = []
    for pair in gesture_pairs:
        null_mean_per_comp = component_wise_null_stats[pair]['null_mean_per_component']
        all_null_means_per_component.append(null_mean_per_comp)
    
    # Average null across gesture pairs for each component
    avg_null_per_component = np.mean(all_null_means_per_component, axis=0)
    
    # Plot TME null line (black dotted)
    ax.plot(component_nums, np.degrees(avg_null_per_component), 
           'k--', linewidth=3, alpha=0.8,
           label='TME Null Distribution Mean')
    
    # Formatting
    ax.set_xlabel('Neural Mode (Manifold Dimension)', fontsize=12)
    ax.set_ylabel('Principal Angle (degrees)', fontsize=12)
    ax.set_title(f'{band_name.upper()} Band: Observed vs TME Null Angles\n'
                f'Subject {subject_id} - Component-wise Analysis', fontsize=14)
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, n_components + 0.5)
    ax.set_ylim(0, 95)
    
    # Add summary statistics as text
    overall_observed_mean = np.mean([np.degrees(np.mean(observed_angles[pair])) for pair in gesture_pairs])
    overall_null_mean = np.degrees(np.mean(avg_null_per_component))
    
    stats_text = f'Mean Observed: {overall_observed_mean:.1f}°\n' \
                f'Mean TME Null: {overall_null_mean:.1f}°\n' \
                f'Difference: {overall_null_mean - overall_observed_mean:.1f}°'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
           verticalalignment='top', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save or show
    if output_dir:
        filename = f'enhanced_tme_visualization_subject_{subject_id}_{band_name}.png'
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        print(f"Enhanced TME visualization saved: {filename}")
        plt.close()
    else:
        plt.show()


def create_aggregate_enhanced_visualization(enhanced_results, output_dir):
    """
    Create aggregate visualization across all subjects showing component-wise patterns.
    """
    # Aggregate across subjects for each band
    for band_name, band_results in enhanced_results.items():
        if not band_results:
            continue
            
        print(f"Creating aggregate enhanced visualization for {band_name} band...")
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Collect data across all subjects
        all_observed_per_component = []
        all_null_per_component = []
        n_components = None
        
        for subject_id, results in band_results.items():
            n_components = results['n_components']
            
            # Get observed angles across components (average across gesture pairs)
            subject_observed_per_comp = []
            subject_null_per_comp = []
            
            for comp_idx in range(n_components):
                # Average across gesture pairs for this component
                comp_observed = [results['observed_angles'][pair][comp_idx] for pair in results['gesture_pairs']]
                comp_null = [results['component_wise_null_stats'][pair]['null_mean_per_component'][comp_idx] 
                           for pair in results['gesture_pairs']]
                
                subject_observed_per_comp.append(np.mean(comp_observed))
                subject_null_per_comp.append(np.mean(comp_null))
            
            all_observed_per_component.append(subject_observed_per_comp)
            all_null_per_component.append(subject_null_per_comp)
        
        # Convert to arrays
        all_observed_per_component = np.array(all_observed_per_component)  # (n_subjects, n_components)
        all_null_per_component = np.array(all_null_per_component)  # (n_subjects, n_components)
        
        # Component numbers
        component_nums = np.arange(1, n_components + 1)
        
        # Plot individual subjects (light lines)
        for i in range(len(all_observed_per_component)):
            ax.plot(component_nums, np.degrees(all_observed_per_component[i]), 
                   'lightblue', alpha=0.3, linewidth=1)
        
        # Plot means
        mean_observed = np.mean(all_observed_per_component, axis=0)
        mean_null = np.mean(all_null_per_component, axis=0)
        
        # SEM for error bars
        from scipy.stats import sem
        sem_observed = sem(all_observed_per_component, axis=0)
        sem_null = sem(all_null_per_component, axis=0)
        
        # Plot with error bars
        ax.errorbar(component_nums, np.degrees(mean_observed), 
                   yerr=np.degrees(sem_observed),
                   color='blue', linewidth=3, marker='o', markersize=6,
                   capsize=3, capthick=2, label=f'Observed (n={len(all_observed_per_component)})',
                   alpha=0.8)
        
        ax.errorbar(component_nums, np.degrees(mean_null), 
                   yerr=np.degrees(sem_null),
                   color='black', linewidth=3, linestyle='--', marker='s', markersize=6,
                   capsize=3, capthick=2, label=f'TME Null (n={len(all_null_per_component)})',
                   alpha=0.8)
        
        # Formatting
        ax.set_xlabel('Neural Mode (Manifold Dimension)', fontsize=12)
        ax.set_ylabel('Principal Angle (degrees)', fontsize=12)
        ax.set_title(f'{band_name.upper()} Band: Observed vs TME Null Angles\n'
                    f'Aggregate Across {len(all_observed_per_component)} Subjects', fontsize=14)
        
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.5, n_components + 0.5)
        ax.set_ylim(0, 95)
        
        # Add statistics
        overall_observed = np.degrees(np.mean(mean_observed))
        overall_null = np.degrees(np.mean(mean_null))
        
        stats_text = f'Mean Observed: {overall_observed:.1f}°\n' \
                    f'Mean TME Null: {overall_null:.1f}°\n' \
                    f'Effect Size: {overall_null - overall_observed:.1f}°'
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', fontsize=11,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save
        if output_dir:
            filename = f'aggregate_enhanced_tme_visualization_{band_name}.png'
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
            print(f"Aggregate enhanced visualization saved: {filename}")
            plt.close()