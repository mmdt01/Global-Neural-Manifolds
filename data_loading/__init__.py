"""
Data loading module for neural data analysis.
"""

from .data_loader import read_data, read_labels
from .mne_converter import mne_raw, mne_epochs

def load_subject_data(subject_id, sampling_rate, mapping_events, event_dict_gest, trigger_type, 
                     tmin, tmax, baseline=None, plot=False):
    """
    Load the data for a single subject and create epochs for the stimulation triggers.
    
    Parameters:
    -----------
    subject_id : int
        ID of the subject to load
    sampling_rate : int
        Sampling rate of the data in Hz
    mapping_events : dict
        Dictionary mapping trigger values to event labels
    event_dict_gest : dict
        Dictionary mapping gesture labels to trigger values
    trigger_type : str
        Type of trigger to use ('stim' or 'emg')
    tmin : float
        Start time for epochs in seconds, relative to events
    tmax : float
        End time for epochs in seconds, relative to events
    baseline : tuple or None
        Baseline correction period (start, end) in seconds
    plot : bool
        Whether to plot the epochs
    
    Returns:
    --------
    epochs : mne.Epochs
        Epochs object containing the segmented data
    good_channels : array
        Array of good channel indices
    names : array
        Array of electrode names
    labels : array
        Array of anatomical labels
    chn_data : array
        Array of channel indices
    """
    # Load data and labels
    data, good_channels = read_data(subject_id)
    names, labels, chn_data = read_labels(subject_id)
    
    # Create raw MNE objects
    raw_stim, raw_emg, events_stim, events_emg = mne_raw(
        sampling_rate, mapping_events, data, good_channels
    )
    
    # Select the appropriate raw data and events based on trigger type
    if trigger_type == 'stim':
        epochs = mne_epochs(raw_stim, events_stim, event_dict_gest, tmin, tmax, baseline, plot)
    elif trigger_type == 'emg':
        epochs = mne_epochs(raw_emg, events_emg, event_dict_gest, tmin, tmax, baseline, plot)
    else:
        raise ValueError("Invalid trigger type. Must be 'stim' or 'emg'.")
    
    return epochs, good_channels, names, labels, chn_data
