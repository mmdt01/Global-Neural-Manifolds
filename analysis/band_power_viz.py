"""
Visualization utilities for band power analysis of neural data.
"""

import numpy as np
import matplotlib.pyplot as plt
from mne.filter import filter_data, resample
from scipy.signal import hilbert, savgol_filter
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import os

from .band_power import get_frequency_bands

def visualize_band_power_processing(epochs, band_name, subject_id, region_label, 
                                  channels=None, n_channels=4, example_epoch=0, 
                                  downsample_factor=1, output_dir=None):
    """
    Visualize the steps of band power computation for neural data.
    
    Parameters:
    -----------
    epochs : mne.Epochs
        MNE Epochs object containing the data
    band_name : str
        Name of the frequency band (delta, theta, alpha, beta, low_gamma, high_gamma, broad)
    subject_id : str or int
        Subject ID for plot titles
    region_label : str
        Brain region label for plot titles
    channels : list, optional
        List of specific channel indices to plot. If None, first n_channels will be used.
    n_channels : int, optional
        Number of channels to plot if channels not specified
    example_epoch : int, optional
        Index of the epoch to use for visualization
    downsample_factor : int, optional
        Factor by which to downsample the result
    output_dir : str, optional
        Directory to save plots. If None, plots are displayed but not saved.
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure object
    """
    # Get frequency bands dictionary
    bands = get_frequency_bands()
    
    # Check if the requested band exists
    if band_name not in bands:
        raise ValueError(f"Unknown band name: {band_name}. Available bands: {list(bands.keys())}")
    
    # Get the band limits
    band = bands[band_name]
    
    # Get sampling frequency
    sfreq = epochs.info['sfreq']
    
    # Get data for the selected epoch
    data = epochs.get_data()[example_epoch]  # Shape: (n_channels, n_times)
    
    # Select channels to visualize
    if channels is None:
        channels = list(range(min(n_channels, data.shape[0])))
    else:
        # Ensure we're not asking for channels that don't exist
        channels = [ch for ch in channels if ch < data.shape[0]]
        if len(channels) == 0:
            raise ValueError(f"No valid channels to plot. Data has {data.shape[0]} channels.")
    
    # Get channel names for titles
    ch_names = [epochs.ch_names[ch] for ch in channels]
    
    # Get time array for plotting
    times = epochs.times
    
    # Create a figure with subplots for each channel and signal type
    n_rows = len(channels)
    fig = plt.figure(figsize=(14, 4 * n_rows))
    gs = gridspec.GridSpec(n_rows, 4, figure=fig, wspace=0.3, hspace=0.4)
    
    # Process each selected channel
    for i, (ch_idx, ch_name) in enumerate(zip(channels, ch_names)):
        # Get the raw data for this channel
        raw_signal = data[ch_idx]
        
        # Step 1: Filter the data in the specified band
        filtered_signal = filter_data(
            raw_signal.reshape(1, -1), 
            sfreq, 
            band[0], 
            band[1], 
            method='iir', 
            verbose=False
        )[0]
        
        # Step 2: Apply Hilbert transform to get analytic signal
        analytic_signal = hilbert(filtered_signal)
        
        # Step 3: Get instantaneous power (squared magnitude of analytic signal)
        inst_power = np.abs(analytic_signal) ** 2
        
        # Step 4: Filter instantaneous power to remove high frequency noise
        window_size = min(101, len(raw_signal)//10)
        # Ensure window_size is odd
        if window_size % 2 == 0:
            window_size += 1
        inst_power_smooth = savgol_filter(inst_power, window_size, 3)
        
        # Plot original signal
        ax1 = fig.add_subplot(gs[i, 0])
        ax1.plot(times, raw_signal)
        ax1.set_title(f"Original Signal\nChannel: {ch_name}")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Amplitude")
        ax1.grid(True)
        ax1.xaxis.set_major_locator(MaxNLocator(5))
        
        # Plot filtered signal
        ax2 = fig.add_subplot(gs[i, 1])
        ax2.plot(times, filtered_signal)
        ax2.set_title(f"Filtered Signal ({band_name}: {band[0]}-{band[1]} Hz)\nChannel: {ch_name}")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Amplitude")
        ax2.grid(True)
        ax2.xaxis.set_major_locator(MaxNLocator(5))
        
        # Plot instantaneous power
        ax3 = fig.add_subplot(gs[i, 2])
        ax3.plot(times, inst_power)
        ax3.set_title(f"Instantaneous Power\nChannel: {ch_name}")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Power")
        ax3.grid(True)
        ax3.xaxis.set_major_locator(MaxNLocator(5))
        
        # Plot smoothed instantaneous power
        ax4 = fig.add_subplot(gs[i, 3])
        ax4.plot(times, inst_power_smooth)
        ax4.set_title(f"Smoothed Instantaneous Power\nChannel: {ch_name}")
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Power")
        ax4.grid(True)
        ax4.xaxis.set_major_locator(MaxNLocator(5))
    
    # Set overall title
    fig.suptitle(f"Band Power Processing Steps\nSubject: {subject_id}, Region: {region_label}, Band: {band_name}, Epoch: {example_epoch}", 
                fontsize=16, y=0.98)
    
    # Save or show the figure
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, 
                              f"band_power_viz_subj_{subject_id}_{region_label}_{band_name}_epoch{example_epoch}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
        plt.close(fig)
    else:
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
        plt.show()
    
    return fig

def visualize_multi_epoch_band_power(epochs, band_name, subject_id, region_label, 
                                  channel_idx=0, n_epochs=4, 
                                  downsample_factor=1, output_dir=None):
    """
    Visualize the band power for multiple epochs on the same channel.
    
    Parameters:
    -----------
    epochs : mne.Epochs
        MNE Epochs object containing the data
    band_name : str
        Name of the frequency band (delta, theta, alpha, beta, low_gamma, high_gamma, broad)
    subject_id : str or int
        Subject ID for plot titles
    region_label : str
        Brain region label for plot titles
    channel_idx : int, optional
        Index of the channel to visualize
    n_epochs : int, optional
        Number of epochs to visualize
    downsample_factor : int, optional
        Factor by which to downsample the result
    output_dir : str, optional
        Directory to save plots. If None, plots are displayed but not saved.
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure object
    """
    # Get frequency bands dictionary
    bands = get_frequency_bands()
    
    # Check if the requested band exists
    if band_name not in bands:
        raise ValueError(f"Unknown band name: {band_name}. Available bands: {list(bands.keys())}")
    
    # Get the band limits
    band = bands[band_name]
    
    # Get sampling frequency
    sfreq = epochs.info['sfreq']
    
    # Get channel name
    ch_name = epochs.ch_names[channel_idx] if channel_idx < len(epochs.ch_names) else f"Channel {channel_idx}"
    
    # Number of epochs to process
    n_epochs = min(n_epochs, len(epochs))
    epoch_indices = list(range(n_epochs))
    
    # Get time array for plotting
    times = epochs.times
    
    # Create figure
    fig, axs = plt.subplots(n_epochs, 2, figsize=(16, 4 * n_epochs), sharex=True)
    
    # Process each epoch
    for i, epoch_idx in enumerate(epoch_indices):
        # Get the raw data for this epoch and channel
        data = epochs.get_data()[epoch_idx]
        raw_signal = data[channel_idx]
        
        # Filter the data in the specified band
        filtered_signal = filter_data(
            raw_signal.reshape(1, -1), 
            sfreq, 
            band[0], 
            band[1], 
            method='iir', 
            verbose=False
        )[0]
        
        # Apply Hilbert transform to get analytic signal
        analytic_signal = hilbert(filtered_signal)
        
        # Get instantaneous power (squared magnitude of analytic signal)
        inst_power = np.abs(analytic_signal) ** 2
        
        # Filter instantaneous power to remove high frequency noise
        window_size = min(101, len(raw_signal)//10)
        if window_size % 2 == 0:
            window_size += 1
        inst_power_smooth = savgol_filter(inst_power, window_size, 3)
        
        # Plot original signal
        axs[i, 0].plot(times, raw_signal, 'b-', label='Original')
        axs[i, 0].plot(times, filtered_signal, 'g-', label='Filtered')
        axs[i, 0].set_title(f"Epoch {epoch_idx}: Original & Filtered Signal")
        axs[i, 0].set_ylabel("Amplitude")
        axs[i, 0].grid(True)
        axs[i, 0].legend()
        
        # Plot power signals
        axs[i, 1].plot(times, inst_power, 'r-', alpha=0.5, label='Instantaneous Power')
        axs[i, 1].plot(times, inst_power_smooth, 'k-', label='Smoothed Power')
        axs[i, 1].set_title(f"Epoch {epoch_idx}: Instantaneous & Smoothed Power")
        axs[i, 1].set_ylabel("Power")
        axs[i, 1].grid(True)
        axs[i, 1].legend()
    
    # Set labels for the bottom row
    axs[-1, 0].set_xlabel("Time (s)")
    axs[-1, 1].set_xlabel("Time (s)")
    
    # Set overall title
    fig.suptitle(f"Band Power Analysis Across Epochs\nSubject: {subject_id}, Region: {region_label}, Band: {band_name}, Channel: {ch_name}", 
                fontsize=16, y=0.99)
    
    # Save or show the figure
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, 
                              f"multi_epoch_band_power_subj_{subject_id}_{region_label}_{band_name}_ch{channel_idx}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
        plt.close(fig)
    else:
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for the suptitle
        plt.show()
    
    return fig

def visualize_comparative_bands(epochs, subject_id, region_label, 
                              channel_idx=0, epoch_idx=0,
                              bands=None, output_dir=None):
    """
    Compare band power across different frequency bands for a single channel and epoch.
    
    Parameters:
    -----------
    epochs : mne.Epochs
        MNE Epochs object containing the data
    subject_id : str or int
        Subject ID for plot titles
    region_label : str
        Brain region label for plot titles
    channel_idx : int, optional
        Index of the channel to visualize
    epoch_idx : int, optional
        Index of the epoch to visualize
    bands : list, optional
        List of band names to compare. If None, uses ['delta', 'theta', 'alpha', 'beta', 'low_gamma', 'high_gamma']
    output_dir : str, optional
        Directory to save plots. If None, plots are displayed but not saved.
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure object
    """
    # Set default bands if not provided
    if bands is None:
        bands = ['delta', 'theta', 'alpha', 'beta', 'low_gamma', 'high_gamma']
    
    # Get frequency bands dictionary
    freq_bands = get_frequency_bands()
    
    # Check if all requested bands exist
    for band_name in bands:
        if band_name not in freq_bands:
            raise ValueError(f"Unknown band name: {band_name}. Available bands: {list(freq_bands.keys())}")
    
    # Get channel name
    ch_name = epochs.ch_names[channel_idx] if channel_idx < len(epochs.ch_names) else f"Channel {channel_idx}"
    
    # Get data for this epoch and channel
    data = epochs.get_data()[epoch_idx]
    raw_signal = data[channel_idx]
    
    # Get sampling frequency
    sfreq = epochs.info['sfreq']
    
    # Get time array for plotting
    times = epochs.times
    
    # Create figure
    fig, axs = plt.subplots(len(bands) + 1, 1, figsize=(14, 3 * (len(bands) + 1)), sharex=True)
    
    # Plot original signal
    axs[0].plot(times, raw_signal, 'k-')
    axs[0].set_title(f"Original Signal - Channel: {ch_name}, Epoch: {epoch_idx}")
    axs[0].set_ylabel("Amplitude")
    axs[0].grid(True)
    
    # Process each band
    for i, band_name in enumerate(bands):
        # Get band limits
        band = freq_bands[band_name]
        
        # Filter the data in the specified band
        filtered_signal = filter_data(
            raw_signal.reshape(1, -1), 
            sfreq, 
            band[0], 
            band[1], 
            method='iir', 
            verbose=False
        )[0]
        
        # Apply Hilbert transform to get analytic signal
        analytic_signal = hilbert(filtered_signal)
        
        # Get instantaneous power (squared magnitude of analytic signal)
        inst_power = np.abs(analytic_signal) ** 2
        
        # Filter instantaneous power to remove high frequency noise
        window_size = min(101, len(raw_signal)//10)
        if window_size % 2 == 0:
            window_size += 1
        inst_power_smooth = savgol_filter(inst_power, window_size, 3)
        
        # Normalize for better visualization
        normalized_power = inst_power_smooth / np.max(inst_power_smooth) if np.max(inst_power_smooth) > 0 else inst_power_smooth
        
        # Plot band power
        axs[i+1].plot(times, normalized_power)
        axs[i+1].set_title(f"{band_name.capitalize()} Band Power ({band[0]}-{band[1]} Hz)")
        axs[i+1].set_ylabel("Normalized Power")
        axs[i+1].grid(True)
    
    # Set label for the bottom subplot
    axs[-1].set_xlabel("Time (s)")
    
    # Set overall title
    fig.suptitle(f"Comparative Band Power Analysis\nSubject: {subject_id}, Region: {region_label}, Channel: {ch_name}, Epoch: {epoch_idx}", 
                fontsize=16, y=0.99)
    
    # Save or show the figure
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, 
                              f"comparative_bands_subj_{subject_id}_{region_label}_ch{channel_idx}_epoch{epoch_idx}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
        plt.close(fig)
    else:
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for the suptitle
        plt.show()
    
    return fig
