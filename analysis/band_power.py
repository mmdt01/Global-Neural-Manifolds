"""
Band power analysis functions for neural data.
"""

import numpy as np
from mne.filter import filter_data, resample
from scipy.signal import hilbert, savgol_filter

def get_frequency_bands():
    """
    Return dictionary of standard frequency bands.
    
    Returns:
    --------
    bands : dict
        Dictionary mapping band names to (low_freq, high_freq) tuples
    """
    return {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 12),
        'beta': (12, 30),
        'low_gamma': (30, 60),
        'high_gamma': (60, 100),
        'broad': (0.5, 100)
    }

def calculate_instantaneous_band_power(data, sfreq, band, downsample_factor=1):
    """
    Calculate instantaneous power for a frequency band using the Hilbert transform.
    
    Parameters:
    -----------
    data : array, shape (n_channels, n_times)
        The input time series data
    sfreq : float
        Sampling frequency of the data
    band : tuple
        Frequency band as (low_freq, high_freq)
    downsample_factor : int, optional
        Factor by which to downsample the result
    
    Returns:
    --------
    inst_power_smooth : array, shape (n_channels, n_times//downsample_factor)
        Smoothed instantaneous power in the specified frequency band
    """
    # Filter the data in the specified band
    filtered_data = filter_data(
        data, 
        sfreq, 
        band[0], 
        band[1], 
        method='iir', 
        verbose=False
    )
    
    # Downsample if requested
    if downsample_factor > 1:
        filtered_data = resample(filtered_data, down=downsample_factor, npad='auto')
    
    # Apply Hilbert transform to get analytic signal
    analytic_signal = hilbert(filtered_data, axis=-1)
    
    # Get instantaneous power (squared magnitude of analytic signal)
    inst_power = np.abs(analytic_signal) ** 2

    # Filter instantaneous power to remove high frequency noise
    inst_power_smooth = savgol_filter(inst_power, min(100, data.shape[1]//10), 3, axis=-1)
    
    return inst_power_smooth

def compute_band_power(epochs, band_name, downsample_factor=1):
    """
    Compute band power for all epochs in the specified frequency band.
    
    Parameters:
    -----------
    epochs : mne.Epochs
        MNE Epochs object containing the data
    band_name : str
        Name of the frequency band (delta, theta, alpha, beta, low_gamma, high_gamma, broad)
    downsample_factor : int, optional
        Factor by which to downsample the result
    
    Returns:
    --------
    band_power : array, shape (n_epochs, n_channels, n_times//downsample_factor)
        Instantaneous power in the specified frequency band for all epochs
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
    
    # Initialize array to store band power for all epochs
    n_epochs = len(epochs)
    n_channels = len(epochs.ch_names)
    
    # Calculate resulting time points after downsampling
    if downsample_factor > 1:
        n_times = len(epochs.times) // downsample_factor
    else:
        n_times = len(epochs.times)
    
    # Initialize array for band power
    band_power = np.zeros((n_epochs, n_channels, n_times))
    
    # Calculate band power for each epoch
    for i, epoch in enumerate(epochs):
        band_power[i] = calculate_instantaneous_band_power(
            epoch, sfreq, band, downsample_factor)
    
    return band_power
