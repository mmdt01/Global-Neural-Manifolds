"""
Neural manifold analysis functions for neural data.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import logging
from scipy.linalg import qr, svd, inv
from utils.helpers import ensure_dir
from .band_power import compute_band_power, get_frequency_bands

def compute_neural_manifold_time(region_epochs, region_channels_dict, band_name, n_components=3,
                                 plot=True, plot_title=None, output_dir=None):
    """
    This function computes the neural manifold for each subject's region by applying PCA to the band power data.
    It uses the pre-computed band power data directly from the epochs objects and mean-centers each channel
    to focus on covariations rather than absolute activity levels.
    
    Parameters:
    -----------
    region_epochs : dict
        Nested dictionary mapping subject IDs to frequency bands to epochs objects with band power
    region_channels_dict : dict
        Dictionary mapping subject IDs to lists of channel names
    band_name : str
        Name of the frequency band to analyze
    n_components : int, optional
        Number of PCA components to compute
    downsample_factor : int, optional
        Factor by which to downsample the data before PCA (usually 1 as no downsampling needed)
    plot : bool, optional
        Whether to plot the low-dimensional representation
    plot_title : str, optional
        Title prefix for the plots
    output_dir : str, optional
        Directory to save plots. If None, plots are displayed but not saved.
    
    Returns:
    --------
    manifold_dict : dict
        Dictionary mapping subject IDs to dictionaries containing:
            'manifold': array of shape (n_times, n_components) - the neural manifold
            'explained_variance': array of explained variance ratios
            'pca': fitted PCA object
    """
    # Initialize results dictionary
    manifold_dict = {}
    
    # Process each subject
    for subject_id, band_epochs in region_epochs.items():
        # Skip if this subject doesn't have data for the specified band
        if band_name not in band_epochs:
            print(f"No {band_name} band data for Subject {subject_id}, skipping...")
            continue
        
        print(f"\nComputing neural manifold for Subject {subject_id}, {band_name} band...")
        
        try:
            # Number of channels for this subject
            num_channels = len(region_channels_dict[subject_id])
            print(f"Total number of channels: {num_channels}")
            
            # Get the specific band's epochs object
            epochs = band_epochs[band_name]
            
            # Extract the data directly - it's already band power
            band_power = epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)
            
            # Get dimensions
            n_epochs, n_channels, n_times = band_power.shape

            # Mean-center each channel across epochs and time: this removes baseline differences between channels and focuses on covariations
            print(f"Mean-centering {n_channels} channels...")
            band_power_centered = np.zeros_like(band_power)

            for ch_idx in range(n_channels):
                # Compute mean activity for this channel across all epochs and time points
                channel_mean = np.mean(band_power[:, ch_idx, :])
                
                # Subtract the mean from all data points for this channel
                band_power_centered[:, ch_idx, :] = band_power[:, ch_idx, :] - channel_mean
                
                print(f"  Channel {ch_idx}: mean activity = {channel_mean:.4f}")
            
            # Reshape to 2D for PCA: (n_epochs*n_channels, n_times)
            X = band_power_centered.reshape(n_epochs * n_channels, n_times)
            
            # Apply PCA
            pca = PCA(n_components=n_components)
            components = pca.fit_transform(X.T)  # Transpose to have time points as samples
            
            # Save results
            manifold_dict[subject_id] = {
                'manifold': components,
                'explained_variance': pca.explained_variance_ratio_,
                'pca': pca,
                'mean_centered_data': band_power_centered 
            }
            
            # Print explained variance
            print(f"Explained variance: {pca.explained_variance_ratio_}")
            print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.2f}")
            
            # Plot if requested and if 3 components
            if plot and n_components == 3:
                # Create a title
                if plot_title is None:
                    title = f"Subject {subject_id}: {band_name} Neural Manifold\n({num_channels} channels, mean-centered)"
                else:
                    title = f"{plot_title} - Subject {subject_id}: {band_name} band (mean-centered)"
                
                # Create 3D plot
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                
                # Get time points
                times = epochs.times
                
                # Create a colormap for time
                norm = plt.Normalize(times.min(), times.max())
                cmap = sns.color_palette("crest", as_cmap=True)
                colors = cmap(norm(times))
                
                # Plot 3D trajectory
                ax.scatter(
                    components[:, 0], 
                    components[:, 1], 
                    components[:, 2], 
                    c=colors, 
                    s=15, 
                    alpha=0.8,
                    marker='o'
                )
                
                # Add a colorbar for time
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax, pad=0.1)
                cbar.set_label('Time (s)')
                
                # Mark specific time points with markers and annotations
                # Find indices of evenly spaced time points for annotation
                time_markers = np.linspace(0, len(times)-1, 5).astype(int)
                
                for idx in time_markers:
                    t = times[idx]
                    x, y, z = components[idx, 0], components[idx, 1], components[idx, 2]
                    ax.scatter([x], [y], [z], c='red', s=50, edgecolors='black', linewidths=1)
                    ax.text(x, y, z, f"{t:.2f}s", fontsize=8)
                
                # Set labels and title
                var_explained = pca.explained_variance_ratio_ * 100
                ax.set_xlabel(f"PC1 ({var_explained[0]:.1f}%)")
                ax.set_ylabel(f"PC2 ({var_explained[1]:.1f}%)")
                ax.set_zlabel(f"PC3 ({var_explained[2]:.1f}%)")
                
                plt.title(title)
                plt.tight_layout()
                
                # Save or show the figure
                if output_dir is not None:
                    plt.savefig(f"{output_dir}/manifold_{band_name}_subject_{subject_id}.png", 
                               dpi=300, bbox_inches='tight')
                    plt.close()
                else:
                    plt.show()
            
            print(f"Neural manifold computation completed for Subject {subject_id}")
            
        except Exception as e:
            print(f"Error computing neural manifold for Subject {subject_id}: {e}")
    
    # Return the manifold dictionary
    return manifold_dict

def compute_neural_manifold(region_epochs, region_channels_dict, band_name, n_components=3,
                            plot=True, plot_title=None, output_dir=None):
    """
    This function computes the neural manifold for each subject's region by applying SPATIAL PCA 
    to the band power data. This approach finds spatial patterns of co-activation across brain regions.
    
    Spatial PCA concatenates trials in time and applies PCA across channels to find population-level
    activity patterns that represent how different brain regions coordinate during the task.
    
    Parameters:
    -----------
    region_epochs : dict
        Nested dictionary mapping subject IDs to frequency bands to epochs objects with band power
    region_channels_dict : dict
        Dictionary mapping subject IDs to lists of channel names
    band_name : str
        Name of the frequency band to analyze
    n_components : int, optional
        Number of PCA components to compute
    plot : bool, optional
        Whether to plot the low-dimensional representation
    plot_title : str, optional
        Title prefix for the plots
    output_dir : str, optional
        Directory to save plots. If None, plots are displayed but not saved.
    
    Returns:
    --------
    manifold_dict : dict
        Dictionary mapping subject IDs to dictionaries containing:
            'manifold': array of shape (n_epochs × n_times, n_components) - the neural manifold trajectories
            'explained_variance': array of explained variance ratios
            'pca': fitted PCA object
            'spatial_patterns': array of shape (n_channels, n_components) - the spatial patterns (loadings)
    """
    # Initialize results dictionary
    manifold_dict = {}
    
    # Process each subject
    for subject_id, band_epochs in region_epochs.items():
        # Skip if this subject doesn't have data for the specified band
        if band_name not in band_epochs:
            print(f"No {band_name} band data for Subject {subject_id}, skipping...")
            continue
        
        print(f"\nComputing SPATIAL neural manifold for Subject {subject_id}, {band_name} band...")
        
        try:
            # Number of channels for this subject
            num_channels = len(region_channels_dict[subject_id])
            print(f"Total number of channels: {num_channels}")
            
            # Get the specific band's epochs object
            epochs = band_epochs[band_name]
            
            # Extract the data directly - it's already band power
            band_power = epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)
            
            # Get dimensions
            n_epochs, n_channels, n_times = band_power.shape
            print(f"Data shape: {n_epochs} epochs × {n_channels} channels × {n_times} time points")
            
            # SPATIAL PCA APPROACH: Concatenate trials in time dimension
            # Reshape to (n_channels, n_epochs × n_times) - each row is a channel, columns are time points across all trials
            X = band_power.transpose(1, 0, 2)  # (n_channels, n_epochs, n_times)
            X = X.reshape(n_channels, n_epochs * n_times)  # (n_channels, n_epochs × n_times)
            
            print(f"Reshaped for spatial PCA: {X.shape[0]} channels × {X.shape[1]} time points (across all trials)")
            
            # Mean-center each channel across all time points and trials
            print(f"Mean-centering {n_channels} channels across all trials and time points...")
            X_centered = np.zeros_like(X)
            channel_means = np.zeros(n_channels)
            
            for ch_idx in range(n_channels):
                # Compute mean activity for this channel across all trials and time points
                channel_mean = np.mean(X[ch_idx, :])
                channel_means[ch_idx] = channel_mean
                
                # Subtract the mean from all data points for this channel
                X_centered[ch_idx, :] = X[ch_idx, :] - channel_mean
                
                print(f"  Channel {ch_idx} ({region_channels_dict[subject_id][ch_idx]}): mean = {channel_mean:.4f}")
            
            # Apply PCA to find spatial patterns
            # Transpose so that samples are time points and features are channels
            # PCA input: (n_epochs × n_times, n_channels)
            print(f"\nApplying PCA to find spatial co-activation patterns...")
            pca = PCA(n_components=n_components)
            manifold_trajectories = pca.fit_transform(X_centered.T)  # Shape: (n_epochs × n_times, n_components)
            
            # Get the spatial patterns (how each channel contributes to each component)
            spatial_patterns = pca.components_.T  # Shape: (n_channels, n_components)
            
            # Reshape manifold trajectories back to (n_epochs, n_times, n_components) for easier interpretation
            manifold_reshaped = manifold_trajectories.reshape(n_epochs, n_times, n_components)
            
            print(f"PCA completed. Output shapes:")
            print(f"  Manifold trajectories: {manifold_trajectories.shape}")
            print(f"  Spatial patterns: {spatial_patterns.shape}")
            print(f"  Manifold reshaped: {manifold_reshaped.shape}")

            #################################################################
            # Print spatial patterns (channel loadings) for each component
            print(f"\nSpatial patterns - Channel loadings for each component:")
            print("=" * 80)

            # Get channel names for this subject
            channel_names = region_channels_dict[subject_id]

            # For each principal component, show the spatial pattern
            for pc_idx in range(n_components):
                print(f"\nPrincipal Component {pc_idx + 1} (explains {pca.explained_variance_ratio_[pc_idx]*100:.1f}% variance):")

            # Print overall summary
            print(f"\nSPATIAL PCA SUMMARY:")
            print("=" * 80)
            print(f"• Found {n_components} spatial modes of neural co-activation")
            print(f"• Total variance explained: {sum(pca.explained_variance_ratio_)*100:.1f}%")
            print(f"• Each mode represents a pattern of brain regions that activate together")
            print(f"• Positive loadings = regions that increase together")
            print(f"• Negative loadings = regions that decrease when others increase")
            
            # Find most important channels overall
            overall_importance = np.mean(np.abs(spatial_patterns), axis=1)
            most_important_channels = np.argsort(overall_importance)[::-1]
            
            print(f"\nMost important brain regions across all {n_components} spatial modes:")
            for rank, ch_idx in enumerate(most_important_channels[:5]):
                channel_name = channel_names[ch_idx]
                importance = overall_importance[ch_idx]
                print(f"  {rank+1}. {channel_name:<15}: {importance:.3f}")

            #######################################################################
            
            # Save results
            manifold_dict[subject_id] = {
                'manifold': manifold_trajectories,  # (n_epochs × n_times, n_components)
                'manifold_reshaped': manifold_reshaped,  # (n_epochs, n_times, n_components)
                'explained_variance': pca.explained_variance_ratio_,
                'pca': pca,
                'spatial_patterns': spatial_patterns,  # (n_channels, n_components)
                'channel_means': channel_means,
                'mean_centered_data': X_centered.T  # (n_epochs × n_times, n_channels)
            }
            
            # Print explained variance
            print(f"\nExplained variance per component: {pca.explained_variance_ratio_}")
            print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.3f}")
            
            # Plot if requested and if 3 components equal to or less than 6
            if plot and n_components <= 6:
                plot_spatial_manifold(
                    n_components,
                    manifold_reshaped, 
                    spatial_patterns,
                    channel_names,
                    epochs.times,
                    pca.explained_variance_ratio_,
                    subject_id,
                    band_name,
                    plot_title,
                    output_dir
                )
            
            print(f"Spatial neural manifold computation completed for Subject {subject_id}")
            
        except Exception as e:
            print(f"Error computing spatial neural manifold for Subject {subject_id}: {e}")
            import traceback
            traceback.print_exc()
    
    # Return the manifold dictionary
    return manifold_dict

def plot_neural_vaf(manifold_results, region_labels, band_name, output_dir=None, max_components=None):
    """
    Plot the cumulative neural variance accounted for (VAF) by principal components.
    
    Parameters:
    -----------
    manifold_results : dict
        Dictionary containing manifold results for a single band (subject_id -> results)
    region_labels : list
        List of brain region names included in the analysis
    band_name : str
        Name of the frequency band
    output_dir : str, optional
        Directory to save plots. If None, plots are displayed but not saved.
    max_components : int, optional
        Maximum number of components to display
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    """
    # Check if we have data
    if len(manifold_results) == 0:
        print(f"No data available for {band_name} band")
        return None

    # Set up the plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Collect cumulative variance data from all subjects
    all_cumulative_variance = []
    subject_ids = []
    
    for subject_id, results in manifold_results.items():
        explained_var = results['explained_variance'] * 100  # Convert to percentage
        cumulative_var = np.cumsum(explained_var)
        
        all_cumulative_variance.append(cumulative_var)
        subject_ids.append(subject_id)
    
    # Convert to numpy array
    all_cumulative_variance = np.array(all_cumulative_variance)
    
    # Determine number of components to plot
    n_components = all_cumulative_variance.shape[1]
    if max_components is not None:
        n_components = min(n_components, max_components)
    
    component_numbers = np.arange(1, n_components + 1)
    
    # Plot individual subjects (lighter color)
    for i, subject_id in enumerate(subject_ids):
        ax.plot(component_numbers, all_cumulative_variance[i, :n_components], 
               'o-', alpha=0.6, color='#e74c3c', markersize=4, linewidth=1)
    
    # Calculate mean and SEM across subjects
    mean_cumulative = np.mean(all_cumulative_variance[:, :n_components], axis=0)
    # sem_cumulative = sem(all_cumulative_variance[:, :n_components], axis=0)
    
    # # Plot mean with error bars
    # ax.errorbar(component_numbers, mean_cumulative, yerr=sem_cumulative,
    #            color='#c0392b', linewidth=2, markersize=6,
    #            capsize=3, capthick=2, label=f'Mean ± SEM (n={len(subject_ids)})')
    
    # Formatting
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Cumulative Neural VAF (%)')
    ax.set_title(f'{band_name.upper()} Band - Cumulative VAF')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Set x-axis ticks to integers
    ax.set_xticks(component_numbers)
    
    # Set y-axis limits
    ax.set_ylim(0, 105)
    
    # Add text annotations for key statistics
    total_var_first_3 = mean_cumulative[min(2, n_components-1)]
    ax.text(0.02, 0.98, f'First 3 PCs: {total_var_first_3:.1f}% VAF', 
           transform=ax.transAxes, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save or show the figure
    if output_dir is not None:
        filename = f'neural_vaf_{band_name}_band.png'
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        print(f"Neural VAF plot saved: {os.path.join(output_dir, filename)}")
        plt.close()
    else:
        plt.show()
    
    return fig

def create_gesture_comparison_summary(gesture_results, band_name, region_labels, n_components, output_dir=None):
    """
    Create summary visualizations comparing manifolds across gestures (SIMPLIFIED VERSION).
    This version only creates VAF plots, not principal angles (to avoid redundancy).
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import sem
    
    # Create comparison output directory
    if output_dir is not None:
        comparison_dir = os.path.join(output_dir, "gesture_comparison")
        ensure_dir(comparison_dir)
    else:
        comparison_dir = None
    
    gestures = list(gesture_results.keys())
    print(f"Creating VAF comparison plots for gestures: {gestures}")
    
    # VAF Comparison Plot ONLY (no principal angles - those are computed elsewhere)
    plt.figure(figsize=(12, 6))
    
    # Colors for different gestures
    colors = plt.cm.Set1(np.linspace(0, 1, len(gestures)))
    
    for i, (gesture_name, manifold_data) in enumerate(gesture_results.items()):
        if not manifold_data:  # Skip if no data
            continue
            
        # Collect variance explained data
        all_explained_variance = []
        for subject_id, results in manifold_data.items():
            explained_var = results['explained_variance'] * 100
            all_explained_variance.append(np.cumsum(explained_var))
        
        if not all_explained_variance:
            continue
            
        all_explained_variance = np.array(all_explained_variance)
        component_numbers = np.arange(1, min(n_components, all_explained_variance.shape[1]) + 1)
        
        # Calculate mean and SEM
        mean_cumulative = np.mean(all_explained_variance[:, :len(component_numbers)], axis=0)
        sem_cumulative = sem(all_explained_variance[:, :len(component_numbers)], axis=0)
        
        # Plot with error bars
        plt.errorbar(component_numbers, mean_cumulative, yerr=sem_cumulative,
                    label=f'{gesture_name.capitalize()} (n={len(all_explained_variance)})',
                    color=colors[i], linewidth=2, marker='o', capsize=3)
    
    plt.xlabel('Principal Component')
    plt.ylabel('Cumulative VAF (%)')
    plt.title(f'{band_name.upper()} Band: Gesture Comparison - Cumulative VAF')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 105)
    
    if comparison_dir is not None:
        plt.savefig(f"{comparison_dir}/gesture_vaf_comparison_{band_name}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    print(f"VAF comparison completed for {band_name} band (principal angles computed separately)")

def plot_spatial_manifold(n_components, manifold_reshaped, spatial_patterns, channel_names, times, 
                         explained_variance, subject_id, band_name, plot_title=None, output_dir=None):
    """
    Plot spatial manifold results in two separate figures:
    1. 3D trajectory through spatial modes (if n_components >= 3)
    2. Spatial patterns (loadings) for ALL components
    
    Parameters:
    -----------
    n_components : int
        Number of components computed
    manifold_reshaped : array
        Manifold trajectories reshaped to (n_epochs, n_times, n_components)
    spatial_patterns : array  
        Spatial patterns (n_channels, n_components)
    channel_names : list
        List of channel names
    times : array
        Time points
    explained_variance : array
        Explained variance ratios
    subject_id : int
        Subject ID
    band_name : str
        Frequency band name
    plot_title : str, optional
        Title prefix
    output_dir : str, optional
        Output directory for saving plots
    """
    var_explained = explained_variance * 100
    
    # Check if we have enough components for 3D plot
    if n_components >= 3:

        # ===== FIGURE 1: 3D Trajectory Plot =====

        fig1 = plt.figure(figsize=(10, 8))
        ax1 = fig1.add_subplot(111, projection='3d')
        
        # Average across epochs to get mean trajectory
        mean_trajectory = np.mean(manifold_reshaped, axis=0)  # (n_times, n_components)
        
        # Create a colormap for time
        norm = plt.Normalize(times.min(), times.max())
        cmap = sns.color_palette("crest", as_cmap=True)
        colors = cmap(norm(times))
        
        # Plot 3D trajectory
        ax1.scatter(mean_trajectory[:, 0], mean_trajectory[:, 1], mean_trajectory[:, 2], 
                c=colors, s=25, alpha=0.8, marker='o')
        
        # Mark time points
        time_markers = np.linspace(0, len(times)-1, 5).astype(int)
        for idx in time_markers:
            t = times[idx]
            x, y, z = mean_trajectory[idx, 0], mean_trajectory[idx, 1], mean_trajectory[idx, 2]
            ax1.scatter([x], [y], [z], c='red', s=60, edgecolors='black', linewidths=1.5)
            ax1.text(x, y, z, f"{t:.2f}s", fontsize=9, fontweight='bold')
        
        # Set labels
        ax1.set_xlabel(f"Spatial Mode 1 ({var_explained[0]:.1f}%)", fontsize=12)
        ax1.set_ylabel(f"Spatial Mode 2 ({var_explained[1]:.1f}%)", fontsize=12)
        ax1.set_zlabel(f"Spatial Mode 3 ({var_explained[2]:.1f}%)", fontsize=12)
        
        if plot_title is None:
            title = f"Subject {subject_id}: {band_name} Spatial Neural Manifold Trajectory"
        else:
            title = f"{plot_title} - Subject {subject_id}: {band_name} Spatial Trajectory"
        ax1.set_title(title, fontsize=14, pad=20)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax1, pad=0.1, shrink=0.8)
        cbar.set_label('Time (s)', fontsize=11)
        
        plt.tight_layout()
        
        # Save or show trajectory figure
        if output_dir is not None:
            plt.savefig(f"{output_dir}/spatial_trajectory_{band_name}_subject_{subject_id}.png", 
                    dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    # ===== FIGURE 2: Spatial Patterns (Bar Plots) for ALL Components =====
    
    # Extract brain regions from channel names
    def extract_region_from_channel(channel_name):
        """Extract region from channel name like 'seeg-117_caudalmiddlefrontal_rh'"""
        try:
            # Split by underscores and get the region part (second element after 'seeg-X')
            parts = channel_name.split('_')
            if len(parts) >= 2:
                # Remove the 'seeg-X' part and get the region
                region = parts[1]
                return region
            else:
                return 'unknown'
        except:
            return 'unknown'
    
    # Get regions for all channels
    channel_regions = [extract_region_from_channel(ch) for ch in channel_names]
    unique_regions = list(set(channel_regions))
    unique_regions.sort()  # Sort for consistent coloring
    
    print(f"Found {len(unique_regions)} unique brain regions: {unique_regions}")
    
    # Create color mapping for regions
    region_colors = {}
    # Use a colormap with enough distinct colors
    cmap_regions = plt.cm.get_cmap('tab20')  # Up to 20 distinct colors
    if len(unique_regions) > 20:
        cmap_regions = plt.cm.get_cmap('hsv')  # For more than 20 regions
    
    for i, region in enumerate(unique_regions):
        region_colors[region] = cmap_regions(i / max(len(unique_regions) - 1, 1))
    
    # Determine subplot layout based on number of components
    n_components_to_plot = spatial_patterns.shape[1]
    
    # Calculate optimal subplot arrangement
    if n_components_to_plot <= 3:
        n_rows, n_cols = 1, n_components_to_plot
        fig_width = 5 * n_components_to_plot
        fig_height = 5
    elif n_components_to_plot <= 6:
        n_rows, n_cols = 2, 3
        fig_width = 15
        fig_height = 8
    elif n_components_to_plot <= 9:
        n_rows, n_cols = 3, 3
        fig_width = 15
        fig_height = 12
    elif n_components_to_plot <= 12:
        n_rows, n_cols = 3, 4
        fig_width = 20
        fig_height = 12
    else:
        # For more than 12 components, use 4 columns
        n_cols = 4
        n_rows = int(np.ceil(n_components_to_plot / n_cols))
        fig_width = 20
        fig_height = 4 * n_rows
    
    print(f"Creating subplot layout: {n_rows} rows × {n_cols} columns for {n_components_to_plot} components")
    
    fig2, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    
    # Handle case where there's only one subplot
    if n_components_to_plot == 1:
        axes = [axes]
    elif n_rows == 1:
        # If only one row, axes is 1D
        pass
    else:
        # If multiple rows, flatten the axes array for easier indexing
        axes = axes.flatten()
    
    # Plot each component
    for comp_idx in range(n_components_to_plot):
        # Get the correct axis
        if n_components_to_plot == 1:
            ax = axes[0]
        else:
            ax = axes[comp_idx]
        
        loadings = spatial_patterns[:, comp_idx]
        
        # Get colors for each bar based on the region
        bar_colors = [region_colors[region] for region in channel_regions]
        
        # Create bar plot with region-based colors
        bars = ax.bar(range(len(channel_names)), loadings, 
                     color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Channels (grouped by brain region)', fontsize=9)
        ax.set_ylabel('Spatial Loading', fontsize=9)
        ax.set_title(f'Spatial Mode {comp_idx+1}\n({var_explained[comp_idx]:.1f}% variance)', 
                    fontsize=10, fontweight='bold')
        
        # Remove x-axis labels as requested
        ax.set_xticks([])
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    # Hide any unused subplots
    total_subplots = n_rows * n_cols
    for idx in range(n_components_to_plot, total_subplots):
        if n_components_to_plot == 1:
            break  # No unused subplots
        else:
            axes[idx].set_visible(False)
    
    # Create legend for brain regions
    legend_elements = []
    for region in unique_regions:
        legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=region_colors[region], 
                                           alpha=0.8, edgecolor='black', linewidth=0.5,
                                           label=region))
    
    # Adjust legend position based on number of components
    legend_y_pos = -0.05 if n_rows <= 2 else -0.02
    legend_ncol = min(len(unique_regions), 6)  # Max 6 columns for legend
    
    # Add legend below the plots
    fig2.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, legend_y_pos), 
               ncol=legend_ncol, fontsize=8, title='Brain Regions',
               title_fontsize=9, columnspacing=1.2)
    
    if plot_title is None:
        suptitle = "" # f"Subject {subject_id}: {band_name} Spatial Patterns (All {n_components_to_plot} Components)"
    else:
        suptitle = f"{plot_title} - Subject {subject_id}: {band_name} Spatial Patterns (All {n_components_to_plot} Components)"
    fig2.suptitle(suptitle, fontsize=12, fontweight='bold', y=0.95)
    
    # Adjust layout based on number of rows
    top_margin = 0.88 if n_rows <= 2 else 0.92
    bottom_margin = 0.15 if n_rows <= 2 else 0.08
    
    plt.tight_layout()
    plt.subplots_adjust(top=top_margin, bottom=bottom_margin)
    
    # Save or show spatial patterns figure
    if output_dir is not None:
        plt.savefig(f"{output_dir}/spatial_patterns_{band_name}_subject_{subject_id}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# functions for analysing manifolds of different gestures independently

def compute_gesture_manifolds(region_epochs, region_channels_dict, band_name, gestures, 
                            n_components=3, downsample_factor=1, plot=True, output_dir=None):
    """
    Compute low-dimensional neural manifold representations for each gesture type.
    
    Parameters:
    -----------
    region_epochs : dict
        Dictionary mapping subject IDs to region-specific epochs objects
    region_channels_dict : dict
        Dictionary mapping subject IDs to lists of channel names
    band_name : str
        Name of the frequency band to analyze
    gestures : list
        List of gesture names to analyze
    n_components : int, optional
        Number of PCA components to compute
    downsample_factor : int, optional
        Factor by which to downsample the data
    plot : bool, optional
        Whether to plot the manifolds
    output_dir : str, optional
        Directory to save plots. If None, plots are displayed but not saved.
    
    Returns:
    --------
    gesture_manifolds : dict
        Dictionary mapping subject IDs to dictionaries of gesture-specific manifolds
    """
    # Get frequency bands dictionary
    bands = get_frequency_bands()
    
    # Check if the requested band exists
    if band_name not in bands:
        raise ValueError(f"Unknown band name: {band_name}. Available bands: {list(bands.keys())}")
    
    # Initialize results dictionary
    gesture_manifolds = {}
    
    # Process each subject
    for subject_id, epochs in region_epochs.items():
        print(f"\nComputing gesture-specific manifolds for Subject {subject_id}, {band_name} band...")
        
        try:
            # Number of channels for this subject
            num_channels = len(region_channels_dict[subject_id])
            
            # Initialize dictionary for this subject's gesture manifolds
            gesture_manifolds[subject_id] = {}
            
            # Process each gesture
            for gesture in gestures:
                print(f"  Processing gesture: {gesture}")
                
                # Get epochs for this gesture
                gesture_epochs = epochs[gesture]
                
                # If no epochs for this gesture, skip
                if len(gesture_epochs) == 0:
                    print(f"  No epochs found for gesture {gesture}, skipping...")
                    continue
                
                # Compute band power for this gesture's epochs
                band_power = compute_band_power(gesture_epochs, band_name, downsample_factor)
                
                # Get dimensions
                n_epochs, n_channels, n_times = band_power.shape
                
                # Reshape to 2D for PCA: (n_epochs*n_channels, n_times)
                X = band_power.reshape(n_epochs * n_channels, n_times)
                
                # Apply PCA
                pca = PCA(n_components=n_components)
                components = pca.fit_transform(X.T)  # Transpose to have time points as samples
                
                # Store the results
                gesture_manifolds[subject_id][gesture] = {
                    'manifold': components,
                    'explained_variance': pca.explained_variance_ratio_,
                    'pca': pca
                }
                
                print(f"  Explained variance for {gesture}: {pca.explained_variance_ratio_}")
                print(f"  Total explained variance: {sum(pca.explained_variance_ratio_):.2f}")
            
            # Plot all gestures for this subject if requested
            if plot and n_components == 3 and len(gesture_manifolds[subject_id]) > 0:
                plot_gesture_manifolds(
                    subject_id, 
                    gesture_manifolds[subject_id], 
                    epochs.times[::downsample_factor], 
                    band_name, 
                    output_dir
                )
            
            print(f"Gesture-specific manifold computation completed for Subject {subject_id}")
            
        except Exception as e:
            print(f"Error computing gesture manifolds for Subject {subject_id}: {e}")
    
    return gesture_manifolds

def plot_gesture_manifolds(subject_id, gesture_data, times, band_name, output_dir=None):
    """
    Plot gesture-specific manifolds for a single subject.
    
    Parameters:
    -----------
    subject_id : int
        Subject ID
    gesture_data : dict
        Dictionary mapping gesture names to manifold data
    times : array
        Array of time points
    band_name : str
        Name of the frequency band
    output_dir : str, optional
        Directory to save plots. If None, plots are displayed but not saved.
    """
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Define colors for each gesture (using distinct colors)
    colors = {
        "elbow": "red",
        "scissor": "blue",
        "rock": "green",
        "rotation": "purple",
        "thumb": "orange"
    }
    
    # Plot each gesture's manifold
    for gesture, data in gesture_data.items():
        manifold = data['manifold']
        var_explained = data['explained_variance'] * 100
        
        # Plot 3D trajectory
        ax.plot(manifold[:, 0], manifold[:, 1], manifold[:, 2], 
               color=colors.get(gesture, "gray"), linewidth=2, label=gesture)
        
        # Mark specific time points (start, middle, end)
        time_markers = [0, len(times)//2, len(times)-1]
        for idx in time_markers:
            t = times[idx]
            x, y, z = manifold[idx, 0], manifold[idx, 1], manifold[idx, 2]
            ax.scatter([x], [y], [z], color=colors.get(gesture, "gray"), s=50, edgecolors='black')
            ax.text(x, y, z, f"{t:.2f}s", fontsize=8)
    
    # Set labels with explained variance
    first_gesture = list(gesture_data.keys())[0]
    var_explained = gesture_data[first_gesture]['explained_variance'] * 100
    ax.set_xlabel(f"PC1 ({var_explained[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1f}%)")
    ax.set_zlabel(f"PC3 ({var_explained[2]:.1f}%)")
    
    # Add title and legend
    ax.set_title(f"Subject {subject_id}: {band_name} Neural Manifolds by Gesture", fontsize=14)
    ax.legend(title="Gestures", loc="upper right")
    
    # Adjust view angle for better visualization
    ax.view_init(elev=30, azim=45)
    
    # Save or show the figure
    if output_dir is not None:
        plt.savefig(f"{output_dir}/gesture_manifolds_{band_name}_subject_{subject_id}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

# functions for aligning manifolds using CCA

def canoncorr(X:np.array, Y: np.array, fullReturn: bool = False) -> np.array:
    """
    Canonical Correlation Analysis (CCA)
    line-by-line port from Matlab implementation of `canoncorr`
    X,Y: (samples/observations) x (features) matrix, for both: X.shape[0] >> X.shape[1]
    fullReturn: whether all outputs should be returned or just `r` be returned (not in Matlab)
    
    returns: A,B,r,U,V 
    A,B: Canonical coefficients for X and Y
    U,V: Canonical scores for the variables X and Y
    r:   Canonical correlations
    
    Signature:
    A,B,r,U,V = canoncorr(X, Y)
    """
    n, p1 = X.shape
    p2 = Y.shape[1]
    if p1 >= n or p2 >= n:
        logging.warning('Not enough samples, might cause problems')

    # Center the variables
    X = X - np.mean(X,0)
    Y = Y - np.mean(Y,0)

    # Factor the inputs, and find a full rank set of columns if necessary
    Q1,T11,perm1 = qr(X, mode='economic', pivoting=True, check_finite=True)

    rankX = sum(np.abs(np.diagonal(T11)) > np.finfo(type((np.abs(T11[0,0])))).eps*max([n,p1]))

    if rankX == 0:
        logging.error(f'stats:canoncorr:BadData = X')
    elif rankX < p1:
        logging.warning('stats:canoncorr:NotFullRank = X')
        Q1 = Q1[:,:rankX]
        T11 = T11[:rankX,:rankX]

    Q2,T22,perm2 = qr(Y, mode='economic', pivoting=True, check_finite=True)
    rankY = sum(np.abs(np.diagonal(T22)) > np.finfo(type((np.abs(T22[0,0])))).eps*max([n,p2]))

    if rankY == 0:
        logging.error(f'stats:canoncorr:BadData = Y')
    elif rankY < p2:
        logging.warning('stats:canoncorr:NotFullRank = Y')
        Q2 = Q2[:,:rankY]
        T22 = T22[:rankY,:rankY]

    # Compute canonical coefficients and canonical correlations.  For rankX >
    # rankY, the economy-size version ignores the extra columns in L and rows
    # in D. For rankX < rankY, need to ignore extra columns in M and D
    # explicitly. Normalize A and B to give U and V unit variance.
    d = min(rankX,rankY)
    L,D,M = svd(Q1.T @ Q2, full_matrices=True, check_finite=True, lapack_driver='gesdd')
    M = M.T

    A = inv(T11) @ L[:,:d] * np.sqrt(n-1)
    B = inv(T22) @ M[:,:d] * np.sqrt(n-1)
    r = D[:d]
    # remove roundoff errs
    r[r>=1] = 1
    r[r<=0] = 0

    if not fullReturn:
        return r

    # Put coefficients back to their full size and their correct order
    A[perm1,:] = np.vstack((A, np.zeros((p1-rankX,d))))
    B[perm2,:] = np.vstack((B, np.zeros((p2-rankY,d))))
    
    # Compute the canonical variates
    U = X @ A
    V = Y @ B

    return A, B, r, U, V

def align_subject_manifolds_with_cca(manifold_dict, subject_id1, subject_id2, band_name, downsample_factor=1, plot=True, output_dir=None):
    """
    Align manifold representations of two subjects using CCA.
    
    Parameters:
    -----------
    manifold_dict : dict
        Dictionary containing manifold data for each subject (from compute_neural_manifold)
    subject_id1, subject_id2 : int
        IDs of the subjects to align
    band_name : str
        Name of the frequency band being analyzed
    plot : bool, optional
        Whether to plot the aligned manifolds
    output_dir : str, optional
        Directory to save plots. If None, plots are displayed but not saved.
    
    Returns:
    --------
    dict
        Dictionary containing CCA results and aligned manifolds
    """
    # Get the manifold data for both subjects
    X = manifold_dict[subject_id1]['manifold']  # Shape: (n_times, n_components)
    Y = manifold_dict[subject_id2]['manifold']  # Shape: (n_times, n_components)
    
    # Ensure both manifolds have the same number of time points
    min_times = min(X.shape[0], Y.shape[0])
    X = X[:min_times, :]
    Y = Y[:min_times, :]
    
    # Perform CCA
    A, B, r, U, V = canoncorr(X, Y, fullReturn=True)
    
    print(f"Canonical correlations between Subject {subject_id1} and Subject {subject_id2}:")
    for i, corr in enumerate(r):
        print(f"  Component {i+1}: {corr:.4f}")
    
    # Create a result dictionary
    result = {
        'A': A,  # Canonical coefficients for subject 1
        'B': B,  # Canonical coefficients for subject 2
        'r': r,  # Canonical correlations
        'U': U,  # Aligned manifold for subject 1 (canonical variates)
        'V': V,  # Aligned manifold for subject 2 (canonical variates)
        'X': X,  # Original manifold for subject 1
        'Y': Y   # Original manifold for subject 2
    }
    
    # Plot if requested
    if plot and X.shape[1] >= 3 and Y.shape[1] >= 3:
        # Plot original manifolds
        fig = plt.figure(figsize=(16, 12))
        
        # Get time points
        times = np.arange(min_times) / (1000 / downsample_factor)  # Convert to seconds based on sample rate
        
        # Create a colormap for time - using "crest" for original and "flare" for aligned
        norm = plt.Normalize(times.min(), times.max())
        cmap_orig = sns.color_palette("crest", as_cmap=True)
        cmap_aligned = sns.color_palette("flare", as_cmap=True)
        colors_orig = cmap_orig(norm(times))
        colors_aligned = cmap_aligned(norm(times))
        
        # Plot original manifolds
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=colors_orig, s=15, alpha=0.8)
        ax1.set_title(f"Subject {subject_id1} Original Manifold", fontsize=12)
        ax1.set_xlabel("PC1")
        ax1.set_ylabel("PC2")
        ax1.set_zlabel("PC3")
        
        ax2 = fig.add_subplot(2, 2, 2, projection='3d')
        ax2.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c=colors_orig, s=15, alpha=0.8)
        ax2.set_title(f"Subject {subject_id2} Original Manifold", fontsize=12)
        ax2.set_xlabel("PC1")
        ax2.set_ylabel("PC2")
        ax2.set_zlabel("PC3")
        
        # Determine the common axis limits for aligned manifolds
        # Combine U and V data to find overall min and max for each dimension
        combined_aligned = np.vstack([U[:, :3], V[:, :3]])
        min_vals = np.min(combined_aligned, axis=0)
        max_vals = np.max(combined_aligned, axis=0)
        
        # Add a small margin (10%) to the ranges
        ranges = max_vals - min_vals
        min_vals = min_vals - 0.1 * ranges
        max_vals = max_vals + 0.1 * ranges
        
        # Plot aligned manifolds (using first 3 canonical components)
        ax3 = fig.add_subplot(2, 2, 3, projection='3d')
        ax3.scatter(U[:, 0], U[:, 1], U[:, 2], c=colors_aligned, s=15, alpha=0.8)
        ax3.set_title(f"Subject {subject_id1} Aligned Manifold", fontsize=12)
        ax3.set_xlabel(f"CC1 (r={r[0]:.3f})")
        ax3.set_ylabel(f"CC2 (r={r[1]:.3f})")
        ax3.set_zlabel(f"CC3 (r={r[2]:.3f})")
        
        # Set common limits for the aligned manifolds
        ax3.set_xlim(min_vals[0], max_vals[0])
        ax3.set_ylim(min_vals[1], max_vals[1])
        ax3.set_zlim(min_vals[2], max_vals[2])
        
        ax4 = fig.add_subplot(2, 2, 4, projection='3d')
        ax4.scatter(V[:, 0], V[:, 1], V[:, 2], c=colors_aligned, s=15, alpha=0.8)
        ax4.set_title(f"Subject {subject_id2} Aligned Manifold", fontsize=12)
        ax4.set_xlabel(f"CC1 (r={r[0]:.3f})")
        ax4.set_ylabel(f"CC2 (r={r[1]:.3f})")
        ax4.set_zlabel(f"CC3 (r={r[2]:.3f})")
        
        # Set common limits for the aligned manifolds
        ax4.set_xlim(min_vals[0], max_vals[0])
        ax4.set_ylim(min_vals[1], max_vals[1])
        ax4.set_zlim(min_vals[2], max_vals[2])
        
        # Add a colorbar for time - original manifolds
        fig.subplots_adjust(right=0.85)  # Make room for colorbar
        cbar_ax1 = fig.add_axes([0.88, 0.55, 0.02, 0.3])  # Position for top colorbar
        sm1 = plt.cm.ScalarMappable(cmap=cmap_orig, norm=norm)
        sm1.set_array([])
        cbar1 = fig.colorbar(sm1, cax=cbar_ax1)
        cbar1.set_label('Time (s) - Original')
        
        # Add a colorbar for time - aligned manifolds
        cbar_ax2 = fig.add_axes([0.88, 0.15, 0.02, 0.3])  # Position for bottom colorbar
        sm2 = plt.cm.ScalarMappable(cmap=cmap_aligned, norm=norm)
        sm2.set_array([])
        cbar2 = fig.colorbar(sm2, cax=cbar_ax2)
        cbar2.set_label('Time (s) - Aligned')
        
        # Adjust the layout
        plt.suptitle(f"Band Neural Manifold Alignment ({band_name})\nSubject {subject_id1} vs Subject {subject_id2}", 
                    fontsize=16)
        plt.tight_layout(rect=[0, 0, 0.85, 0.96])  # Adjust for colorbar space
        
        # Save or show the figure
        if output_dir is not None:
            plt.savefig(f"{output_dir}/cca_aligned_{band_name}_sub{subject_id1}_sub{subject_id2}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
            
    return result

