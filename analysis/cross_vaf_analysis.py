"""
Cross-projection VAF analysis for testing shared neural structure across gestures.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
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

def visualize_cross_vaf_results(results, output_dir=None):
    """
    Create visualizations of cross-gesture VAF analysis.
    
    Parameters:
    -----------
    results : dict
        Results from compute_cross_gesture_vaf_analysis
    output_dir : str, optional
        Directory to save plots
    """
    if results is None:
        return
    
    subject_id = results['subject_id']
    band_name = results['band_name']
    gesture_names = results['gesture_names']
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Ratio Matrix Heatmap
    ax1 = axes[0]
    
    im = ax1.imshow(results['ratio_matrix'], cmap='viridis', vmin=0, vmax=1)
    
    # Add text annotations
    for i in range(len(gesture_names)):
        for j in range(len(gesture_names)):
            ratio = results['ratio_matrix'][i, j]
            color = 'white' if ratio < 0.5 else 'black'
            ax1.text(j, i, f'{ratio:.2f}', ha='center', va='center', color=color, fontweight='bold')
    
    ax1.set_xticks(range(len(gesture_names)))
    ax1.set_yticks(range(len(gesture_names)))
    ax1.set_xticklabels([g.capitalize() for g in gesture_names], rotation=45)
    ax1.set_yticklabels([g.capitalize() for g in gesture_names])
    ax1.set_xlabel('Project ONTO (Target Manifold)')
    ax1.set_ylabel('Project FROM (Source Data)')
    ax1.set_title(f'Cross-Gesture VAF Ratios\nSubject {subject_id}, {band_name} Band')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('VAF Ratio (Cross/Within)')
    
    # Plot 2: Distribution Comparison
    ax2 = axes[1]
    
    # Plot histograms
    ax2.hist(results['control_ratios'], bins=30, alpha=0.7, color='gray', 
            label=f"Random Control\n(n={len(results['control_ratios'])})", density=True)
    ax2.hist(results['observed_ratios'], bins=15, alpha=0.8, color='blue',
            label=f"Cross-Gesture\n(n={len(results['observed_ratios'])})", density=True)
    
    # Add vertical lines for means
    ax2.axvline(results['statistics']['mean_control'], color='gray', linestyle='--', linewidth=2,
               label=f"Control Mean: {results['statistics']['mean_control']:.3f}")
    ax2.axvline(results['statistics']['mean_observed'], color='blue', linestyle='--', linewidth=2,
               label=f"Observed Mean: {results['statistics']['mean_observed']:.3f}")
    
    ax2.set_xlabel('VAF Ratio (Cross/Within)')
    ax2.set_ylabel('Density')
    ax2.set_title(f'Cross-Gesture vs Control Ratios\np = {results["statistics"]["p_value"]:.2e}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save or show
    if output_dir is not None:
        filename = f'cross_vaf_analysis_subject_{subject_id}_{band_name}.png'
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        print(f"Cross-VAF visualization saved: {filename}")
        plt.close()
    else:
        plt.show()
    
    return fig
    """
    Create visualizations of cross-gesture VAF analysis.
    
    Parameters:
    -----------
    results : dict
        Results from compute_cross_gesture_vaf_analysis
    output_dir : str, optional
        Directory to save plots
    """
    if results is None:
        return
    
    subject_id = results['subject_id']
    band_name = results['band_name']
    gesture_names = results['gesture_names']
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Ratio Matrix Heatmap
    ax1 = axes[0]
    
    im = ax1.imshow(results['ratio_matrix'], cmap='viridis', vmin=0, vmax=1)
    
    # Add text annotations
    for i in range(len(gesture_names)):
        for j in range(len(gesture_names)):
            ratio = results['ratio_matrix'][i, j]
            color = 'white' if ratio < 0.5 else 'black'
            ax1.text(j, i, f'{ratio:.2f}', ha='center', va='center', color=color, fontweight='bold')
    
    ax1.set_xticks(range(len(gesture_names)))
    ax1.set_yticks(range(len(gesture_names)))
    ax1.set_xticklabels([g.capitalize() for g in gesture_names], rotation=45)
    ax1.set_yticklabels([g.capitalize() for g in gesture_names])
    ax1.set_xlabel('Project ONTO (Target Manifold)')
    ax1.set_ylabel('Project FROM (Source Data)')
    ax1.set_title(f'Cross-Gesture VAF Ratios\nSubject {subject_id}, {band_name} Band')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('VAF Ratio (Cross/Within)')
    
    # Plot 2: Distribution Comparison
    ax2 = axes[1]
    
    # Plot histograms
    ax2.hist(results['control_ratios'], bins=30, alpha=0.7, color='gray', 
            label=f"Random Control\n(n={len(results['control_ratios'])})", density=True)
    ax2.hist(results['observed_ratios'], bins=15, alpha=0.8, color='blue',
            label=f"Cross-Gesture\n(n={len(results['observed_ratios'])})", density=True)
    
    # Add vertical lines for means
    ax2.axvline(results['statistics']['mean_control'], color='gray', linestyle='--', linewidth=2,
               label=f"Control Mean: {results['statistics']['mean_control']:.3f}")
    ax2.axvline(results['statistics']['mean_observed'], color='blue', linestyle='--', linewidth=2,
               label=f"Observed Mean: {results['statistics']['mean_observed']:.3f}")
    
    ax2.set_xlabel('VAF Ratio (Cross/Within)')
    ax2.set_ylabel('Density')
    ax2.set_title(f'Cross-Gesture vs Control Ratios\np = {results["statistics"]["p_value"]:.2e}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save or show
    if output_dir is not None:
        filename = f'cross_vaf_analysis_subject_{subject_id}_{band_name}.png'
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        print(f"Cross-VAF visualization saved: {filename}")
        plt.close()
    else:
        plt.show()
    
    return fig