"""
Time-frequency visualization functions.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def plot_tf_median(power, region_str, subject_id, baseline, output_dir=None):
    """
    Plot median time-frequency representation across channels.
    
    Parameters:
    -----------
    power : mne.time_frequency.EpochsTFR
        Time-frequency power object
    region_str : str
        String describing the brain regions
    subject_id : int
        Subject ID
    baseline : tuple
        Baseline period (start, end) in seconds
    output_dir : str, optional
        Directory to save plot. If None, plot is displayed but not saved.
    """
    # Number of channels
    num_channels = power.data.shape[0]
    
    # Create title with relevant information
    title = f"Subject {subject_id}: {region_str}\n({num_channels} channels)"
    
    # Create figure
    fig = plt.figure(figsize=(12, 8))
    
    # Calculate median across channels       
    avg_power_data = np.median(power.data, axis=0)
    
    # Plot with proper logarithmic frequency scale
    ax = fig.add_subplot(111)
    
    # Extract data for plotting
    times = power.times
    freqs = power.freqs
    extent = [times[0], times[-1], 0, len(freqs)-1]
    
    # Plot the data with a logarithmic y-axis
    im = ax.imshow(
        avg_power_data, 
        extent=extent, 
        aspect='auto', 
        origin='lower', 
        cmap='RdBu_r', 
        vmin=-1.5, 
        vmax=1.5
    )
    
    # Set frequency ticks
    n_yticks = 8
    ytick_indices = np.round(np.linspace(0, len(freqs)-1, n_yticks)).astype(int)
    ytick_values = freqs[ytick_indices]
    ytick_labels = [f"{freq:.1f}" for freq in ytick_values]
    
    ax.set_yticks(ytick_indices)
    ax.set_yticklabels(ytick_labels)
    
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Power change (%)')

    # Mark baseline period
    if baseline[0] is not None:
        plt.axvline(x=baseline[0], color='black', linestyle='--', alpha=0.5)
        plt.text(baseline[0] + 0.03, 5, 'Baseline start', rotation=90, va='bottom')
    
    if baseline[1] is not None:
        plt.axvline(x=baseline[1], color='black', linestyle='--', alpha=0.5)
        plt.text(baseline[1] + 0.03, 5, 'Baseline end', rotation=90, va='bottom')
    
    # Save or show the figure
    if output_dir is not None:
        plt.savefig(f"{output_dir}/tfr_subject_{subject_id}.png", dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

def plot_tf_channel_grid(power, channels, subject_id, region_str, baseline, fig_idx, num_figures, output_dir=None):
    """
    Plot time-frequency representations for a grid of channels.
    
    Parameters:
    -----------
    power : mne.time_frequency.EpochsTFR
        Time-frequency power object
    channels : list
        List of channel names
    subject_id : int
        Subject ID
    region_str : str
        String describing the brain regions
    baseline : tuple
        Baseline period (start, end) in seconds
    fig_idx : int
        Index of the current figure
    num_figures : int
        Total number of figures
    output_dir : str, optional
        Directory to save plot. If None, plot is displayed but not saved.
    """
    # Determine which channels to plot in this figure (max 12 per figure)
    start_idx = fig_idx * 12
    end_idx = min(start_idx + 12, len(channels))
    channels_to_plot = channels[start_idx:end_idx]
    num_plots = len(channels_to_plot)
    
    # Calculate grid dimensions
    if num_plots <= 3:
        n_rows, n_cols = 1, num_plots
    elif num_plots <= 6:
        n_rows, n_cols = 2, (num_plots + 1) // 2
    elif num_plots <= 9:
        n_rows, n_cols = 3, (num_plots + 2) // 3
    else:  # 10, 11, or 12 channels
        n_rows, n_cols = 4, 3
    
    # Create figure and axes with extra top margin for title
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows+0.5), 
                           sharex=True, sharey=True)
    main_title = f"Subject {subject_id}: {region_str} ({len(channels)} channels)"
    fig.suptitle(f"{main_title} - Set {fig_idx+1}/{num_figures}", fontsize=16, y=0.99)
    
    # Flatten axes array for easier indexing if it's multi-dimensional
    if num_plots > 1:
        axes = axes.flatten()
    else:
        axes = [axes]  # Make it a list for consistent indexing
    
    # Plot each channel
    for i, ch_idx in enumerate(range(start_idx, end_idx)):
        ax = axes[i]
        
        # Extract data for this channel
        ch_data = power.data[ch_idx]
        
        # Extract plotting parameters
        times = power.times
        freqs = power.freqs
        extent = [times[0], times[-1], 0, len(freqs)-1]
        
        # Plot the data
        im = ax.imshow(ch_data, extent=extent, aspect='auto', origin='lower', 
                     cmap='RdBu_r', vmin=-1.5, vmax=1.5)
        
        # Set channel title
        ax.set_title(f"Channel: {channels[ch_idx]}")
        
        # Only set y-label and y-ticks for leftmost plots
        if i % n_cols == 0:
            # Set frequency ticks
            n_yticks = 5
            ytick_indices = np.round(np.linspace(0, len(freqs)-1, n_yticks)).astype(int)
            ytick_values = freqs[ytick_indices]
            ytick_labels = [f"{freq:.1f}" for freq in ytick_values]
            
            ax.set_yticks(ytick_indices)
            ax.set_yticklabels(ytick_labels)
            ax.set_ylabel('Frequency (Hz)')
        else:
            ax.set_yticks([])
        
        # Only set x-label for bottom plots
        if i >= num_plots - n_cols:
            ax.set_xlabel('Time (s)')
        
        # Mark baseline period
        if baseline[0] is not None:
            ax.axvline(x=baseline[0], color='black', linestyle='--', alpha=0.5)
        if baseline[1] is not None:
            ax.axvline(x=baseline[1], color='black', linestyle='--', alpha=0.5)
    
    # Hide unused subplots if any
    for j in range(num_plots, len(axes)):
        axes[j].set_visible(False)
    
    # Add colorbar (single colorbar for the whole figure)
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Power change (%)')
    
    # Save or show the figure
    if output_dir is not None:
        plt.savefig(f"{output_dir}/tfr_subject_{subject_id}_channels_set{fig_idx+1}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout(rect=[0, 0, 0.9, 0.99])  # Leave more room for suptitle
        plt.show()

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
