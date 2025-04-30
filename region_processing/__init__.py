"""
Region processing module for neural data analysis.
"""

from .epochs_extractor import (
    extract_region_specific_epochs, 
    plot_region_specific_epochs
)

def analyze_region_specific_data(region_labels, subject_id_list, sampling_frequency, mapping_events, 
                              event_dict_gest, trigger_type, tmin, tmax, bands_to_process=None, 
                              plot=False, band_to_plot=None):
    """
    Extract and analyze region-specific epochs data across multiple subjects.
    
    Parameters:
    -----------
    region_labels : list
        List of brain region names to extract data for
    subject_id_list : list
        List of subject IDs to analyze
    sampling_frequency : int
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
    bands_to_process : list or None
        List of frequency band names to process. If None, all bands will be processed.
    plot : bool
        Whether to plot the region-specific epochs
    band_to_plot : str, optional
        Specific frequency band to plot. If None, plots the first available band.
    
    Returns:
    --------
    region_epochs : dict
        Nested dictionary mapping subject IDs to frequency bands to region-specific epochs objects
    region_channels_dict : dict
        Dictionary mapping subject IDs to lists of channel names
    """
    if bands_to_process:
        print(f"Processing frequency bands: {bands_to_process}")
    
    # Extract region-specific epochs
    region_epochs, region_channels_dict = extract_region_specific_epochs(
        region_labels,
        subject_id_list,
        sampling_frequency,
        mapping_events,
        event_dict_gest,
        trigger_type,
        tmin,
        tmax,
        bands_to_process
    )
    
    # Plot the region-specific epochs if requested
    if plot and len(region_epochs) > 0:
        plot_title = f"Regions: {', '.join(region_labels)}"
        plot_region_specific_epochs(region_epochs, region_channels_dict, event_dict_gest, plot_title, band_to_plot)
    
    return region_epochs, region_channels_dict
