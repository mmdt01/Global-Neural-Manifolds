"""
Time-frequency analysis functions for neural data.
"""

import numpy as np
import matplotlib.pyplot as plt
from mne.time_frequency import tfr_morlet

def compute_time_frequency(region_epochs, region_channels_dict, region_labels, freqs, n_cycles, 
                         baseline, output_dir=None):
    """
    Compute time-frequency representations for region-specific epoch data with per-channel plots.
    
    Parameters:
    -----------
    region_epochs : dict
        Dictionary mapping subject IDs to region-specific epochs objects
    region_channels_dict : dict
        Dictionary mapping subject IDs to lists of channel names
    region_labels : list
        List of brain region names included in the analysis
    freqs : array
        Array of frequencies of interest
    n_cycles : array or float
        Number of cycles for Morlet wavelets
    baseline : tuple
        Baseline period to apply correction (start, end) in seconds
    output_dir : str, optional
        Directory to save TF plots. If None, plots are displayed but not saved.
    
    Returns:
    --------
    tfr_power_dict : dict
        Dictionary mapping subject IDs to time-frequency power objects
    """    
    # Dictionary to store power objects
    tfr_power_dict = {}
    
    # Process each subject
    for subject_id, epochs in region_epochs.items():
        print(f"\nComputing time-frequency for Subject {subject_id}...")
        
        try:
            # Get channel names for this subject
            channels = region_channels_dict[subject_id]
            num_channels = len(channels)
            
            # Compute time-frequency representation
            power = tfr_morlet(
                epochs, 
                freqs=freqs, 
                n_cycles=n_cycles, 
                use_fft=True, 
                return_itc=False, 
                decim=3, 
                n_jobs=1, 
                average=True
            )
            
            # Apply baseline correction
            power.apply_baseline(baseline=baseline, mode='percent')
            
            # Store the power object
            tfr_power_dict[subject_id] = power
            
            # Create title with relevant information
            region_str = ', '.join(region_labels)
            main_title = f"Subject {subject_id}: {region_str} ({num_channels} channels)"
            
            # Plot the time-frequency representations for each channel
            # Calculate how many figures we need (maximum 12 channels per figure)
            num_figures = (num_channels + 11) // 12  # Ceiling division
            
            for fig_idx in range(num_figures):
                # Determine which channels to plot in this figure
                start_idx = fig_idx * 12
                end_idx = min(start_idx + 12, num_channels)
                channels_to_plot = channels[start_idx:end_idx]
                
                # Calculate grid dimensions (try to make it as square as possible)
                num_plots = len(channels_to_plot)
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
                    extent = [times[0], times[-1], 0, len(freqs)-1]
                    
                    # Plot the data
                    im = ax.imshow(ch_data, extent=extent, aspect='auto', origin='lower', 
                                 cmap='RdBu_r', vmin=-1.5, vmax=1.5)
                    
                    # Set channel title
                    ax.set_title(f"Channel: {channels[ch_idx]}")
                    
                    # Only set y-label and y-ticks for leftmost plots
                    if i % n_cols == 0:
                        # Set logarithmic frequency ticks
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
            
            # Also create a figure with the median across channels for comparison
            fig = plt.figure(figsize=(12, 8.5))
            
            # Calculate median across channels
            avg_power_data = np.median(power.data, axis=0)
            
            # Plot with proper logarithmic frequency scale
            ax = fig.add_subplot(111)
            
            # Extract data for plotting
            times = power.times
            extent = [times[0], times[-1], 0, len(freqs)-1]
            
            # Plot the data with a logarithmic y-axis
            im = ax.imshow(avg_power_data, extent=extent, aspect='auto', origin='lower', 
                         cmap='RdBu_r', vmin=-1.5, vmax=1.5)
            
            # Set logarithmic frequency ticks
            n_yticks = 8
            ytick_indices = np.round(np.linspace(0, len(freqs)-1, n_yticks)).astype(int)
            ytick_values = freqs[ytick_indices]
            ytick_labels = [f"{freq:.1f}" for freq in ytick_values]
            
            ax.set_yticks(ytick_indices)
            ax.set_yticklabels(ytick_labels)
            
            # Add title with more padding for small channel sets
            plt.title(f"{main_title} - Median across all channels", pad=20)
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
                plt.savefig(f"{output_dir}/tfr_subject_{subject_id}_median.png", dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.tight_layout()
                plt.show()
            
            print(f"Time-frequency analysis completed for Subject {subject_id}")
            
        except Exception as e:
            print(f"Error computing time-frequency for Subject {subject_id}: {e}")
    
    # Return the power dictionary
    return tfr_power_dict

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