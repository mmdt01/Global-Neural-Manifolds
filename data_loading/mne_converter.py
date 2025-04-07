"""
Functions for converting data to MNE objects.
"""

import mne
import numpy as np

def mne_raw(sampling_rate, mapping_events, data, good_channels):
    """
    Create MNE raw object from the data and events array.
    
    Parameters:
    -----------
    sampling_rate : float
        Sampling rate of the data in Hz
    mapping_events : dict
        Dictionary mapping trigger values to event labels
    data : array
        Array of neural data
    good_channels : array
        Array of good channel indices
    
    Returns:
    --------
    raw_trig : mne.io.RawArray
        Raw data object with trigger events as annotations
    raw_emg : mne.io.RawArray
        Raw data object with EMG events as annotations
    events_trig : array
        Array of trigger events
    events_emg : array
        Array of EMG events
    """
    # Create channel names and types
    chn_names = np.append([f'seeg-{ch}' for ch in good_channels], 
                         ["emg0", "emg1", "stim_trigger", "stim_emg"])
    chn_types = np.append(["seeg"] * len(good_channels), 
                         ["emg", "emg", "stim", "stim"])
    
    # Create MNE info object
    info = mne.create_info(ch_names=list(chn_names), 
                          ch_types=list(chn_types), 
                          sfreq=sampling_rate)
    
    # Create raw array
    raw = mne.io.RawArray(data.transpose(), info)
    
    # Find events
    events_trig = mne.find_events(raw, stim_channel='stim_trigger')
    events_emg = mne.find_events(raw, stim_channel='stim_emg')
    
    # Remove the EMG and event channels
    raw.drop_channels(["emg0", "emg1", "stim_trigger", "stim_emg"])
    
    # Create annotations from events
    annot_from_events_trig = mne.annotations_from_events(
        events=events_trig, event_desc=mapping_events, sfreq=sampling_rate
    )
    annot_from_events_emg = mne.annotations_from_events(
        events=events_emg, event_desc=mapping_events, sfreq=sampling_rate
    )
    
    # Set annotations
    raw_trig = raw.copy().set_annotations(annot_from_events_trig)
    raw_emg = raw.copy().set_annotations(annot_from_events_emg)
    
    return raw_trig, raw_emg, events_trig, events_emg

def mne_epochs(raw, events, event_id, tmin, tmax, baseline, plot=True):
    """
    Create epochs from the raw data.
    
    Parameters:
    -----------
    raw : mne.io.RawArray
        Raw data object
    events : array
        Array of events
    event_id : dict
        Dictionary mapping event names to event IDs
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
    """
    # Create epochs
    epochs = mne.Epochs(
        raw, 
        events, 
        event_id,
        tmin=tmin, 
        tmax=tmax, 
        baseline=baseline, 
        preload=True
    )
    
    # Plot epochs if requested
    if plot:
        epochs.plot(
            n_channels=8, 
            scalings={"seeg": 5e2}, 
            title="Epochs", 
            events=events,
            event_id=event_id,
            event_color=dict(elbow="red", scissor="blue", rock="black", 
                            rotation="green", thumb="yellow"),
            show=True,
            block=True
        )
    
    return epochs
