"""
Data loading and band power functions for neural data analysis.
"""

import numpy as np
import hdf5storage
from mne.filter import filter_data, resample
from scipy.signal import hilbert, savgol_filter

def read_data(subject_id):
    """
    Load preprocessed neural data from .mat file.
    
    Parameters:
    -----------
    subject_id : int
        ID of the subject to load
    
    Returns:
    --------
    data : array
        Array of neural data
    good_channels : array
        Array of good channel indices (0-indexed)
    """
    data_path = f"preprocessed/P{subject_id}/preprocessed2.mat"
    mat = hdf5storage.loadmat(data_path)
    data = mat['Datacell']
    good_channels = mat['good_channels']
    del mat

    # Concatenate the two data arrays
    data = np.concatenate((data[0, 0], data[0, 1]), 0)
    data = data.astype(np.float32)

    # Create integer list of good data channels (0-indexed)
    good_channels = good_channels.flatten().astype(int) - 1

    return data, good_channels

def read_labels(subject_id):
    """
    Load electrode labels and positions from EleCTX files.
    
    Parameters:
    -----------
    subject_id : int
        ID of the subject to load
    
    Returns:
    --------
    names : array
        Array of electrode names
    labels : array
        Array of anatomical labels
    chn_data : array
        Array of channel indices (0-indexed)
    """
    # Load the electrode labels
    electrode_labels = hdf5storage.loadmat(f'EleCTX_Files/P{subject_id}/electrodes_Final_Norm.mat')
    elec_info = electrode_labels['elec_Info_Final_wm']

    # Handle different data structures
    if elec_info.shape == (1, 1):
        # Subject 41 type structure (1,1)
        elec_struct = elec_info[0, 0]
    elif elec_info.shape == (1,):
        # Subject 32 type structure (1,)
        elec_struct = elec_info[0]
    
    # Extract electrode names and labels
    names = np.concatenate(np.concatenate(elec_struct['name'])).flatten()
    labels = np.concatenate(np.concatenate(elec_struct['ana_label_name'])).flatten()

    # Load CHN mapping array
    electrode_registrations = hdf5storage.loadmat(
        f'EleCTX_Files/P{subject_id}/SignalChanel_Electrode_Registration.mat'
    )
    chn_data = electrode_registrations['CHN'].flatten().astype(int) - 1

    return names, labels, chn_data

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
    
    # Apply Hilbert transform to get analytic signal
    analytic_signal = hilbert(filtered_data, axis=-1)
    
    # Get instantaneous power (squared magnitude of analytic signal)
    inst_power = np.abs(analytic_signal) ** 2
    
    # Filter instantaneous power to remove high frequency noise
    inst_power_smooth = savgol_filter(inst_power, min(100, data.shape[1]//10), 3, axis=-1)
    
    # Downsample if requested
    if downsample_factor > 1:
        inst_power_smooth = resample(inst_power_smooth, down=downsample_factor, npad='auto')
    
    return inst_power_smooth

def filter_raw_data_and_compute_power(raw_data, sfreq, band_name, downsample_factor=1):
    """
    Filter raw data in a specific frequency band and compute band power.
    
    Parameters:
    -----------
    raw_data : array, shape (n_channels, n_times)
        Raw neural data
    sfreq : float
        Sampling frequency of the data
    band_name : str
        Name of the frequency band
    downsample_factor : int, optional
        Factor by which to downsample the result
    
    Returns:
    --------
    band_power : array, shape (n_channels, n_times//downsample_factor)
        Band power data for the specified frequency band
    """
    # Get frequency bands dictionary
    bands = get_frequency_bands()
    
    # Check if the requested band exists
    if band_name not in bands:
        raise ValueError(f"Unknown band name: {band_name}. Available bands: {list(bands.keys())}")
    
    # Get the band limits
    band = bands[band_name]
    
    # Calculate instantaneous band power
    band_power = calculate_instantaneous_band_power(raw_data, sfreq, band, downsample_factor)
    
    return band_power
