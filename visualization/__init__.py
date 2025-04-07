"""
Visualization module for neural data analysis.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from .manifold_stats import (
    plot_correlation_heatmap,
    plot_correlation_radar
)

def plot_tf_summary(power_dict, region_channels_dict, region_labels, baseline, output_dir=None):
    """
    Create summary plots of time-frequency results for all subjects.
    
    Parameters:
    -----------
    power_dict : dict
        Dictionary mapping subject IDs to time-frequency power objects
    region_channels_dict : dict
        Dictionary mapping subject IDs to lists of channel names
    region_labels : list
        List of brain region names included in the analysis
    baseline : tuple
        Baseline period (start, end) in seconds
    output_dir : str, optional
        Directory to save plots. If None, plots are displayed but not saved.
    """
    region_str = ', '.join(region_labels)
    
    # Create a summary figure with all subjects
    num_subjects = len(power_dict)
    
    if num_subjects == 0:
        print("No time-frequency results to plot.")
        return
    
    # Calculate grid dimensions for subjects
    n_rows = int(np.ceil(np.sqrt(num_subjects)))
    n_cols = int(np.ceil(num_subjects / n_rows))
    
    # Create figure
    fig = plt.figure(figsize=(4*n_cols, 3*n_rows))
    plt.suptitle(f"Time-Frequency Summary: {region_str}", fontsize=16)
    
    # Plot each subject
    for i, (subject_id, power) in enumerate(power_dict.items()):
        ax = plt.subplot(n_rows, n_cols, i+1)
        
        # Number of channels for this subject
        num_channels = len(region_channels_dict[subject_id])
        
        # Calculate median across channels
        avg_power_data = np.median(power.data, axis=0)
        
        # Extract data for plotting
        times = power.times
        freqs = power.freqs
        extent = [times[0], times[-1], 0, len(freqs)-1]
        
        # Plot the data
        im = ax.imshow(avg_power_data, extent=extent, aspect='auto', origin='lower', 
                      cmap='RdBu_r', vmin=-1.5, vmax=1.5)
        
        # Set frequency ticks
        n_yticks = 4
        ytick_indices = np.round(np.linspace(0, len(freqs)-1, n_yticks)).astype(int)
        ytick_values = freqs[ytick_indices]
        ytick_labels = [f"{freq:.1f}" for freq in ytick_values]
        
        ax.set_yticks(ytick_indices)
        ax.set_yticklabels(ytick_labels)
        
        # Set labels only for outer plots
        if i % n_cols == 0:
            ax.set_ylabel('Frequency (Hz)')
        if i >= num_subjects - n_cols:
            ax.set_xlabel('Time (s)')
        
        # Set title
        ax.set_title(f"Subject {subject_id} ({num_channels} ch)")
        
        # Mark baseline period
        if baseline[0] is not None:
            ax.axvline(x=baseline[0], color='black', linestyle='--', alpha=0.5)
        if baseline[1] is not None:
            ax.axvline(x=baseline[1], color='black', linestyle='--', alpha=0.5)
    
    # Add colorbar for the entire figure
    plt.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('Power change (%)')
    
    # Save or show the figure
    if output_dir is not None:
        plt.savefig(f"{output_dir}/tfr_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout(rect=[0, 0, 0.9, 0.98])
        plt.show()

def plot_manifold_comparison(manifold_dict, band_name, times, region_labels, output_dir=None):
    """
    Create a comparison plot of manifolds for multiple subjects.
    
    Parameters:
    -----------
    manifold_dict : dict
        Dictionary mapping subject IDs to manifold data
    band_name : str
        Name of the frequency band
    times : array
        Array of time points
    region_labels : list
        List of brain region names
    output_dir : str, optional
        Directory to save plot. If None, plot is displayed but not saved.
    """
    # Number of subjects
    num_subjects = len(manifold_dict)
    
    if num_subjects == 0:
        print(f"No manifold data available for {band_name} band.")
        return
    
    # Calculate grid layout
    n_rows = int(np.ceil(np.sqrt(num_subjects)))
    n_cols = int(np.ceil(num_subjects / n_rows))
    
    # Create figure
    fig = plt.figure(figsize=(5*n_cols, 4*n_rows))
    plt.suptitle(f"{band_name.capitalize()} Band Neural Manifolds - Regions: {', '.join(region_labels)}", 
                fontsize=16)
    
    # Create a shared colormap for time
    norm = plt.Normalize(times.min(), times.max())
    cmap = sns.color_palette("crest", as_cmap=True)
    
    # Plot each subject's manifold
    for i, (subject_id, data) in enumerate(manifold_dict.items()):
        # Create 3D subplot
        ax = fig.add_subplot(n_rows, n_cols, i+1, projection='3d')
        
        # Get manifold and explained variance
        manifold = data['manifold']
        var_explained = data['explained_variance'] * 100
        
        # Plot trajectory colored by time
        colors = cmap(norm(times))
        
        # Plot 3D trajectory
        scatter = ax.scatter(
            manifold[:, 0], 
            manifold[:, 1], 
            manifold[:, 2], 
            c=colors, 
            s=10, 
            alpha=0.8,
            marker='o'
        )
        
        # Mark specific time points (start, middle, end)
        time_markers = [0, len(times)//2, len(times)-1]
        for idx in time_markers:
            t = times[idx]
            x, y, z = manifold[idx, 0], manifold[idx, 1], manifold[idx, 2]
            ax.scatter([x], [y], [z], c='red', s=30, edgecolors='black')
            ax.text(x, y, z, f"{t:.2f}s", fontsize=6)
        
        # Set labels with explained variance
        ax.set_xlabel(f"PC1 ({var_explained[0]:.1f}%)")
        ax.set_ylabel(f"PC2 ({var_explained[1]:.1f}%)")
        ax.set_zlabel(f"PC3 ({var_explained[2]:.1f}%)")
        
        # Set title for this subplot
        ax.set_title(f"Subject {subject_id}")
        
        # Optimize viewing angle
        ax.view_init(elev=30, azim=45)
    
    # Add a shared colorbar for time
    plt.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Time (s)')
    
    # Save or show the figure
    if output_dir is not None:
        plt.savefig(f"{output_dir}/manifold_comparison_{band_name}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])
        plt.show()

def visualize_canonical_correlations(cca_results, bands=None, output_dir=None):
    """
    Create multiple visualizations of canonical correlations.
    
    Parameters:
    -----------
    cca_results : dict
        Dictionary containing CCA results for each band and subject pair
    bands : list, optional
        List of frequency bands to analyze. If None, uses all available bands.
    output_dir : str, optional
        Directory to save plots. If None, plots are displayed but not saved.
    """
    # If bands not specified, use all available bands
    if bands is None:
        bands = list(cca_results.keys())
    
    # Create output directory if needed
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    print("\n===== Creating canonical correlation visualizations =====")
    
    # For each band, create the individual visualizations
    for band in bands:
        if band not in cca_results or not cca_results[band]:
            print(f"No CCA results found for {band} band, skipping...")
            continue
        
        print(f"\nVisualizing {band} band correlations...")
        
        # Create heatmap
        plot_correlation_heatmap(cca_results, band, output_dir)
        
        # Create radar plot
        plot_correlation_radar(cca_results, band, output_dir)
    
    print("\nVisualization complete!")
