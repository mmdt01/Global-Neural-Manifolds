# Python script that extracts region-specific channel data across subjects & analyses region-specific epochs data
# 05.03.2025

import mne
import numpy as np
import hdf5storage
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from copy import deepcopy
from collections import defaultdict
from mne.time_frequency import tfr_morlet
from mne.filter import filter_data
from scipy.signal import hilbert
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

def read_data(subject_id):
    """
    Load the data from the .mat file
    """
    data_path = f"preprocessed/P{subject_id}/preprocessed2.mat"
    mat = hdf5storage.loadmat(data_path)
    data = mat['Datacell']
    good_channels = mat['good_channels']
    del mat

    # concatenate the two data arrays
    data = np.concatenate((data[0, 0], data[0, 1]), 0)
    data = data.astype(np.float32)

    # create integer list of good data channels (0-indexed)
    good_channels = good_channels.flatten()
    good_channels = good_channels.astype(int)
    good_channels = good_channels - 1

    # return the data, the number of channels and the good channels
    return data, good_channels

def read_labels(subject_id):
    """
    Load the electrode labels and positions from the EleCTX files
    """
    # load the electrode labels
    electrode_labels = hdf5storage.loadmat(f'EleCTX_Files/P{subject_id}/electrodes_Final_Norm.mat')
    elec_info = electrode_labels['elec_Info_Final_wm']

    # Check the shape and structure of the loaded data
    if elec_info.shape == (1, 1):
        # Handle Subject 41 type structure (1,1)
        elec_struct = elec_info[0, 0]
    
    elif elec_info.shape == (1,):
        # Handle Subject 32 type structure (1,)
        elec_struct = elec_info[0]
    
    # extract the electrode names
    names = elec_struct['name']
    names_flat1 = np.concatenate(names).flatten()
    names_flat2 = np.concatenate(names_flat1).flatten()

    # extract the electrode anatomical labels
    labels = elec_struct['ana_label_name']
    labels_flat1 = np.concatenate(labels).flatten()
    labels_flat2 = np.concatenate(labels_flat1).flatten()

    # load CHN mapping array
    electrode_registrations = hdf5storage.loadmat(f'EleCTX_Files/P{subject_id}/SignalChanel_Electrode_Registration.mat')
    chn_data = electrode_registrations['CHN'].flatten()
    chn_data = chn_data.astype(int)
    chn_data = chn_data - 1

    # rename the variables
    names = names_flat2
    labels = labels_flat2

    # return the electrode labels and the CHN mapping array
    return names, labels, chn_data

def mne_raw(sampling_rate, mapping_events, data, good_channels):
    """
    Create mne raw object from the data and the events array
    """
    chn_names = np.append([f'seeg-{ch}' for ch in good_channels], ["emg0","emg1","stim_trigger","stim_emg"])
    chn_types = np.append(["seeg"]*len(good_channels), ["emg","emg","stim","stim"])
    info = mne.create_info(ch_names=list(chn_names), ch_types=list(chn_types), sfreq=sampling_rate)
    raw = mne.io.RawArray(data.transpose(), info)

    # create events array for the raw data
    events_trig = mne.find_events(raw, stim_channel='stim_trigger')
    events_emg = mne.find_events(raw, stim_channel='stim_emg')
    # remove the emg and event channels from the raw data
    raw.drop_channels(["emg0", "emg1", "stim_trigger", "stim_emg"])

    # set annotations from events for the raw data: this is needed for epoching
    annot_from_events = mne.annotations_from_events(events=events_trig, event_desc=mapping_events, sfreq=sampling_rate)
    raw_trig = raw.set_annotations(annot_from_events)
    annot_from_events = mne.annotations_from_events(events=events_emg, event_desc=mapping_events, sfreq=sampling_rate)
    raw_emg = raw.set_annotations(annot_from_events)

    # return the raw data and the events array
    return raw_trig, raw_emg, events_trig, events_emg

def mne_epochs(raw, events, event_id, tmin, tmax, baseline, plot=True):
    """
    Create epochs from the raw data
    """
    epochs = mne.Epochs(
        raw, 
        events, 
        event_id,
        tmin=tmin, 
        tmax=tmax, 
        baseline=baseline, 
        preload=True
    )
    if plot:
        epochs.plot(
            n_channels=8, 
            scalings={"seeg": 5e2}, 
            title="Epochs", 
            events=events,
            event_id=event_id,
            event_color=dict(elbow="red", scissor="blue", rock="black", rotation="green", thumb="yellow"),
            show=True,
            block=True
        )
        plt.show()
    return epochs

def load_subject_data(subject_id, sampling_rate, mapping_events, event_dict_gest, trigger_type, tmin, tmax, baseline=None, plot=False):
    """
    Load the data for a single subject and create epochs for the stimulation triggers.
    
    Parameters:
    subject_id (int): ID of the subject to load
    sampling_rate (int): Sampling rate of the data
    mapping_events (dict): Dictionary mapping trigger values to event labels
    event_dict_gest (dict): Dictionary mapping gesture labels to trigger values
    trigger_type (str): Type of trigger to use ('stim' or 'emg')
    tmin (float): Start time for epochs relative to events
    tmax (float): End time for epochs relative to events
    baseline (tuple or None): Baseline correction period
    plot (bool): Whether to plot the epochs
    
    Returns:
    mne.Epochs: Epochs object containing the stimulation triggers
    numpy.ndarray: Array of good channel indices
    numpy.ndarray: Array of channel names
    numpy.ndarray: Array of brain region labels
    numpy.ndarray: Array of channel indices
    """
    data, good_channels = read_data(subject_id)
    names, labels, chn_data = read_labels(subject_id)
    raw_stim, raw_emg, events_stim, events_emg = mne_raw(sampling_rate, mapping_events, data, good_channels)
    if trigger_type == 'stim':
        epochs = mne_epochs(raw_stim, events_stim, event_dict_gest, tmin, tmax, baseline, plot)
    elif trigger_type == 'emg':
        epochs = mne_epochs(raw_emg, events_emg, event_dict_gest, tmin, tmax, baseline, plot)
    else:
        raise ValueError("Invalid trigger type. Must be 'stim' or 'emg'.")
    return epochs, good_channels, names, labels, chn_data

def get_brain_regions(good_channels, labels, chn_data):
    """
    Return the brain regions that are present in the data
    """
    # find indices in labels that are contained in good_channels
    good = np.where(np.isin(chn_data, good_channels))
    labels = labels[good]
    # find unique labels 
    regions = np.unique(labels)
    return regions

def brain_region_select(epochs, good_channels, labels, chn_data, region_label):
    """
    Select the channels of a specific brain region
    """
    # find the indices of region_label in elec_info
    indices = np.where(labels == region_label)

    # find indices of chn_data that are good_channels
    good = np.where(np.isin(chn_data, good_channels))

    # check if region_label index is in good_channels
    if np.all(np.isin(indices, good)):
        # find the seeg data indices
        data_indices = chn_data[indices]
    else:
        # only return the indices that are in good
        g_indices = np.intersect1d(indices, good)
        data_indices = chn_data[g_indices]

    # find index in good_channels of data_indices
    channel_indices = np.where(np.isin(good_channels, data_indices))[0]
    # return the channel names
    region_ch_names = [epochs.ch_names[i] for i in channel_indices]
    return region_ch_names

def create_region_mapping(regions):
    """
    Creates a dictionary mapping standardized brain region names to their variant labels.

    Parameters:
    regions (numpy.ndarray): Array of brain region labels
    
    Returns:
    dict: Dictionary mapping standardized names to lists of variant labels
    """
    # Initialize defaultdict to collect all variants
    region_variants = defaultdict(list)
    
    for region in regions:
        # Standardize the label by removing double hyphens
        standardized = region.replace('--', '-')
        # Add the original label to the list of variants for this standardized name
        region_variants[standardized].append(region)
    # Convert defaultdict to regular dict and sort by keys
    unique_regions = dict(sorted(region_variants.items()))
    return unique_regions

def create_region_channel_mapping(epochs, good_channels, labels, chn_data, unique_regions):
    """
    Creates two dictionaries mapping standardized brain region names to lists of channel names and number.
    
    Parameters:
    epochs (mne.Epochs): Epochs object containing the data
    good_channels (numpy.ndarray): Array of good channel indices
    labels (numpy.ndarray): Array of brain region labels
    chn_data (numpy.ndarray): Array of channel indices
    unique_regions (dict): Dictionary mapping standardized names to lists of variant labels
    
    Returns:
    dict1: Dictionary mapping standardized names to lists of channel names
    dict2: Dictionary mapping standardized names to number of channels
    """
    # Initialize defaultdict to collect all channels
    unique_region_channels = defaultdict(list)
    
    # find the channels contained in each unique region
    for region, variants in unique_regions.items():
        region_channels = []
        # for each variant of the region label, find the channels
        for variant in variants:
            region_channels.extend(brain_region_select(epochs, good_channels, labels, chn_data, variant))
        # add the channels to the dictionary
        unique_region_channels[region] = region_channels

    # create new dictionary with values as the number of channels
    unique_region_channel_nums = deepcopy(unique_region_channels)

    for region, channels in unique_region_channel_nums.items():
        unique_region_channel_nums[region] = len(channels)
    
    return unique_region_channels, unique_region_channel_nums

def get_channels_for_regions(epochs, good_channels, labels, chn_data, region_labels):
    """
    Get channel names for specific brain regions.
    
    Parameters:
    epochs (mne.Epochs): Epochs object containing the data
    good_channels (numpy.ndarray): Array of good channel indices
    labels (numpy.ndarray): Array of brain region labels
    chn_data (numpy.ndarray): Array of channel indices
    region_labels (list): List of brain region names to find channels for
    
    Returns:
    list: List of channel names for the specified regions
    """
    # Get brain regions for this subject
    brain_regions = get_brain_regions(good_channels, labels, chn_data)
    
    # Create mapping of standardized region names
    unique_regions = create_region_mapping(brain_regions)
    
    # Get channel mappings for all regions
    unique_region_channels, _ = create_region_channel_mapping(
        epochs, 
        good_channels, 
        labels, 
        chn_data, 
        unique_regions
    )
    
    # Collect channels for the specified regions
    region_channels = []
    
    for region_label in region_labels:
        # First, check if the exact region name exists
        if region_label in unique_region_channels:
            region_channels.extend(unique_region_channels[region_label])
        else:
            # If not, look through all region variants to find a match
            for region, variants in unique_regions.items():
                if region_label in variants:
                    region_channels.extend(unique_region_channels[region])
                    break
    
    # Remove duplicates while preserving order
    unique_region_channels = []
    [unique_region_channels.append(ch) for ch in region_channels if ch not in unique_region_channels]

    # sort the channels in numerical order
    unique_region_channels.sort()
    
    return unique_region_channels

def extract_region_specific_epochs(region_labels, subject_id_list, sampling_frequency, mapping_events, event_dict_gest, trigger_type, tmin, tmax):
    """
    Extract epochs data for specific brain regions across multiple subjects.
    
    Parameters:
    region_labels (list): List of brain region names to extract data for
    subject_id_list (list): List of subject IDs to analyze
    sampling_frequency (int): Sampling rate of the data
    mapping_events (dict): Dictionary mapping trigger values to event labels
    event_dict_gest (dict): Dictionary mapping gesture labels to trigger values
    trigger_type (str): Type of trigger to use ('stim' or 'emg')
    tmin (float): Start time for epochs relative to events
    tmax (float): End time for epochs relative to events
    
    Returns:
    dict: Dictionary mapping subject IDs to region-specific epochs objects
    dict: Dictionary mapping subject IDs to lists of channel names
    """
    # Initialize results dictionary
    region_epochs = {}
    region_channels_dict = {}
    
    # Loop through each subject
    for subject_id in subject_id_list:
        print(f"\nProcessing subject {subject_id}...")
        
        try:
            # Load subject data
            epochs, good_channels, names, labels, chn_data = load_subject_data(
                subject_id, 
                sampling_frequency, 
                mapping_events, 
                event_dict_gest, 
                trigger_type, 
                tmin, 
                tmax, 
                baseline=None, 
                plot=False
            )
            # Get channels for the specified regions
            region_channels = get_channels_for_regions(
                epochs, 
                good_channels, 
                labels, 
                chn_data, 
                region_labels
            )
            
            # If no channels found for the specified regions, skip this subject
            if len(region_channels) == 0:
                print(f"No channels found for the specified regions in subject {subject_id}. Skipping...")
                continue
            
            print(f"Found {len(region_channels)} channels for the specified regions in subject {subject_id}:")
            print(region_channels)
            
            # Create a new epochs object with only the channels from the specified regions
            region_epochs[subject_id] = epochs.copy().pick_channels(region_channels)
            region_channels_dict[subject_id] = region_channels
            
            print(f"Successfully extracted region-specific epochs for subject {subject_id}")
            
        except Exception as e:
            print(f"Error processing subject {subject_id}: {e}")
    
    # Check if any subjects were successfully processed
    if len(region_epochs) == 0:
        print("No subjects had channels in the specified regions.")
    else:
        print(f"\nSuccessfully extracted region-specific epochs for {len(region_epochs)} subjects:")
        for subject_id in region_epochs:
            print(f"  Subject {subject_id}: {len(region_channels_dict[subject_id])} channels")
    
    return region_epochs, region_channels_dict

def plot_region_specific_epochs(region_epochs, region_channels_dict, event_id, title=None):
    """
    Plot the region-specific epochs data for all subjects.
    
    Parameters:
    region_epochs (dict): Dictionary mapping subject IDs to region-specific epochs objects
    region_channels_dict (dict): Dictionary mapping subject IDs to lists of channel names
    event_id (dict): Dictionary mapping event labels to trigger values
    title (str): Optional title for the plots
    """
    for subject_id, epochs in region_epochs.items():
        num_channels = len(region_channels_dict[subject_id])
        
        # Create a title for the plot
        plot_title = f"Subject {subject_id} - Regional Epochs" if title is None else f"{title} - Subject {subject_id}"
        
        # Plot all channels if there are few, otherwise plot 8 at a time
        n_channels = min(8, num_channels)
        
        # Plot the epochs
        epochs.plot(
            n_channels=n_channels,
            scalings={"seeg": 5e2},
            title=plot_title,
            event_id=event_id,
            event_color=dict(elbow="red", scissor="blue", rock="black", rotation="green", thumb="yellow"),
            show=True,
            block=True
        )

def analyze_region_specific_data(region_labels, subject_id_list, sampling_frequency, mapping_events, event_dict_gest, trigger_type, tmin, tmax, plot=False):
    """
    Extract and analyze region-specific epochs data across multiple subjects.
    
    Parameters:
    region_labels (list): List of brain region names to extract data for
    subject_id_list (list): List of subject IDs to analyze
    sampling_frequency (int): Sampling rate of the data
    mapping_events (dict): Dictionary mapping trigger values to event labels
    event_dict_gest (dict): Dictionary mapping gesture labels to trigger values
    trigger_type (str): Type of trigger to use ('stim' or 'emg')
    tmin (float): Start time for epochs relative to events
    tmax (float): End time for epochs relative to events
    plot (bool): Whether to plot the region-specific epochs
    
    Returns:
    dict: Dictionary mapping subject IDs to region-specific epochs objects
    dict: Dictionary mapping subject IDs to lists of channel names
    """
    print(f"Extracting data for regions: {region_labels}")
    
    # Extract region-specific epochs
    region_epochs, region_channels_dict = extract_region_specific_epochs(
        region_labels,
        subject_id_list,
        sampling_frequency,
        mapping_events,
        event_dict_gest,
        trigger_type,
        tmin,
        tmax
    )
    
    # Plot the region-specific epochs if requested
    if plot and len(region_epochs) > 0:
        plot_title = f"Regions: {', '.join(region_labels)}"
        plot_region_specific_epochs(region_epochs, region_channels_dict, event_dict_gest, plot_title)
    
    return region_epochs, region_channels_dict

# time-frequency and dimensionality reduction functions

def compute_time_frequency_median(region_epochs, region_channels_dict, region_labels, freqs, n_cycles, 
                          baseline, output_dir=None):
    """
    Compute time-frequency representations for region-specific epoch data.
    
    Parameters:
    -----------
    region_epochs : dict
        Dictionary mapping subject IDs to region-specific epochs objects
    region_channels_dict : dict
        Dictionary mapping subject IDs to lists of channel names
    region_labels : list
        List of brain region names included in the analysis
    freqs : array, optional
        Array of frequencies of interest
    n_cycles : array or float, optional
        Number of cycles for Morlet wavelets
    baseline : tuple, optional
        Baseline period to apply correction (start, end) in seconds
    output_dir : str, optional
        Directory to save TF plots. If None, plots are displayed but not saved.
    
    Returns:
    --------
    dict
        Dictionary mapping subject IDs to time-frequency power objects
    """    
    # Dictionary to store power objects
    tfr_power_dict = {}
    
    # Process each subject
    for subject_id, epochs in region_epochs.items():
        print(f"\nComputing time-frequency for Subject {subject_id}...")
        
        try:
            # Number of channels for this subject
            num_channels = len(region_channels_dict[subject_id])
            
            # Compute time-frequency representation
            power = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, 
                              use_fft=True, return_itc=False, decim=3, 
                              n_jobs=1, average=True)
            
            # Apply baseline correction
            power.apply_baseline(baseline=baseline, mode='percent')
            
            # Store the power object
            tfr_power_dict[subject_id] = power
            
            # Create title with relevant information
            region_str = ', '.join(region_labels)
            title = f"Subject {subject_id}: {region_str}\n({num_channels} channels)"
            
            # Plot the time-frequency representation
            fig = plt.figure(figsize=(12, 8))
            
            # Average across channels to get one TF plot per subject using median         
            avg_power_data = np.median(power.data, axis=0)

            # # Pick one channel to plot
            # avg_power_data = power.data[0]
            
            # Plot with proper logarithmic frequency scale
            ax = fig.add_subplot(111)
            
            # Extract data for plotting
            times = power.times
            extent = [times[0], times[-1], 0, len(freqs)-1]
            
            # Plot the data with a logarithmic y-axis
            im = ax.imshow(avg_power_data, extent=extent, aspect='auto', origin='lower', 
                         cmap='RdBu_r', vmin=-1.5, vmax=1.5)
            
            # Set logarithmic frequency ticks
            # Choose a subset of frequencies for tick labels to avoid overcrowding
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

            # Mark baseline period start
            if baseline[0] is not None:
                plt.axvline(x=baseline[0], color='black', linestyle='--', alpha=0.5)
                plt.text(baseline[0] + 0.03, 5, 'Baseline start', rotation=90, va='bottom')
            
            # Mark baseline period end
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
            
            print(f"Time-frequency analysis completed for Subject {subject_id}")
            
        except Exception as e:
            print(f"Error computing time-frequency for Subject {subject_id}: {e}")
    
    # Return the power dictionary
    return tfr_power_dict

def compute_time_frequency(region_epochs, region_channels_dict, region_labels, freqs, n_cycles, 
                          baseline, output_dir=None):
    """
    Compute time-frequency representations for region-specific epoch data.
    
    Parameters:
    -----------
    region_epochs : dict
        Dictionary mapping subject IDs to region-specific epochs objects
    region_channels_dict : dict
        Dictionary mapping subject IDs to lists of channel names
    region_labels : list
        List of brain region names included in the analysis
    freqs : array, optional
        Array of frequencies of interest
    n_cycles : array or float, optional
        Number of cycles for Morlet wavelets
    baseline : tuple, optional
        Baseline period to apply correction (start, end) in seconds
    output_dir : str, optional
        Directory to save TF plots. If None, plots are displayed but not saved.
    
    Returns:
    --------
    dict
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
            power = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, 
                              use_fft=True, return_itc=False, decim=3, 
                              n_jobs=1, average=True)
            
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
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows+0.5), sharex=True, sharey=True)
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
            # Increase figure height slightly to ensure title doesn't overlap
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
    band_power : array, shape (n_channels, n_times//downsample_factor)
        Instantaneous power in the specified frequency band
    """
    # Filter the data in the specified band
    filtered_data = filter_data(data, sfreq, band[0], band[1], 
                              method='iir', verbose=False)
    
    # Downsample if requested
    if downsample_factor > 1:
        filtered_data = mne.filter.resample(filtered_data, down=downsample_factor, npad='auto')
    
    # Apply Hilbert transform to get analytic signal
    analytic_signal = hilbert(filtered_data, axis=-1)
    
    # Get instantaneous power (squared magnitude of analytic signal)
    inst_power = np.abs(analytic_signal) ** 2

    # filter intantaneous power to remove high frequency noise
    inst_power_smooth = savgol_filter(inst_power, 100, 3, axis=-1)
    
    return inst_power_smooth

def get_frequency_bands():
    """
    Return dictionary of standard frequency bands.
    
    Returns:
    --------
    dict
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

def compute_neural_manifold(region_epochs, region_channels_dict, band_name, n_components=3, 
                          downsample_factor=1, plot=True, plot_title=None, output_dir=None):
    """
    Compute low-dimensional neural manifold representation of region-specific epoch data.
    
    Parameters:
    -----------
    region_epochs : dict
        Dictionary mapping subject IDs to region-specific epochs objects
    region_channels_dict : dict
        Dictionary mapping subject IDs to lists of channel names
    band_name : str
        Name of the frequency band to analyze (delta, theta, alpha, beta, low_gamma, high_gamma, broad)
    n_components : int, optional
        Number of PCA components to compute
    downsample_factor : int, optional
        Factor by which to downsample the data before PCA
    plot : bool, optional
        Whether to plot the low-dimensional representation
    plot_title : str, optional
        Title prefix for the plots
    output_dir : str, optional
        Directory to save plots. If None, plots are displayed but not saved.
    
    Returns:
    --------
    dict
        Dictionary mapping subject IDs to dictionaries containing:
            'manifold': array of shape (n_times, n_components) - the neural manifold
            'explained_variance': array of explained variance ratios
            'pca': fitted PCA object
    """
    # Get frequency bands dictionary
    bands = get_frequency_bands()
    
    # Check if the requested band exists
    if band_name not in bands:
        raise ValueError(f"Unknown band name: {band_name}. Available bands: {list(bands.keys())}")
    
    # Initialize results dictionary
    manifold_dict = {}
    
    # Process each subject
    for subject_id, epochs in region_epochs.items():
        print(f"\nComputing neural manifold for Subject {subject_id}, {band_name} band...")
        
        try:
            # Number of channels for this subject
            num_channels = len(region_channels_dict[subject_id])
            
            # Compute band power
            band_power = compute_band_power(epochs, band_name, downsample_factor)
            
            # Get dimensions
            n_epochs, n_channels, n_times = band_power.shape
            
            # Reshape to 2D for PCA: (n_epochs*n_channels, n_times)
            X = band_power.reshape(n_epochs * n_channels, n_times)
            
            # Apply PCA
            pca = PCA(n_components=n_components)
            components = pca.fit_transform(X.T)  # Transpose to have time points as samples
            
            # Calculate mean across epochs and channels
            # Reshape components back to (n_times, n_components)
            manifold = components  # Already in shape (n_times, n_components)
            
            # Save results
            manifold_dict[subject_id] = {
                'manifold': manifold,
                'explained_variance': pca.explained_variance_ratio_,
                'pca': pca
            }
            
            # Print explained variance
            print(f"Explained variance: {pca.explained_variance_ratio_}")
            print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.2f}")
            
            # Plot if requested and if 3 components
            if plot and n_components == 3:
                # Create a title
                if plot_title is None:
                    title = f"Subject {subject_id}: {band_name} Neural Manifold\n({num_channels} channels)"
                else:
                    title = f"{plot_title} - Subject {subject_id}: {band_name} band"
                
                # Create 3D plot
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                
                # Get downsampled time points
                times = epochs.times[::downsample_factor]
                
                # Create a colormap for time
                norm = plt.Normalize(times.min(), times.max())
                cmap = sns.color_palette("crest", as_cmap=True)
                colors = cmap(norm(times))
                
                # Plot 3D trajectory
                ax.scatter(
                    manifold[:, 0], 
                    manifold[:, 1], 
                    manifold[:, 2], 
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
                    x, y, z = manifold[idx, 0], manifold[idx, 1], manifold[idx, 2]
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
    dict
        Dictionary mapping subject IDs to time-frequency power objects
    """
    # Define log-spaced frequencies from 2-200 Hz
    # freqs = np.logspace(np.log10(2), np.log10(200), num=100)

    # Define linear-spaced frequencies from 2-200 Hz
    freqs = np.linspace(2, 200, num=100)
    
    # Define number of cycles (more cycles for higher frequencies)
    n_cycles = freqs / 2  # Higher frequencies get more cycles
    
    # Set baseline period (first 0.2 seconds)
    # baseline = (None, 0.2)
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

def analyze_neural_manifolds(region_epochs, region_channels_dict, region_labels, 
                           bands=None, n_components=3, downsample_factor=1, output_dir=None):
    """
    Analyze neural manifolds for region-specific epoch data across multiple frequency bands.
    
    Parameters:
    -----------
    region_epochs : dict
        Dictionary mapping subject IDs to region-specific epochs objects
    region_channels_dict : dict
        Dictionary mapping subject IDs to lists of channel names
    region_labels : list
        List of brain region names included in the analysis
    bands : list, optional
        List of frequency bands to analyze. If None, uses ['delta', 'beta', 'high_gamma']
    n_components : int, optional
        Number of PCA components to compute
    downsample_factor : int, optional
        Factor by which to downsample the data before PCA
    output_dir : str, optional
        Directory to save plots. If None, plots are displayed but not saved.
        
    Returns:
    --------
    dict
        Dictionary mapping band names to manifold dictionaries
    """
    # Set default bands if not provided
    if bands is None:
        bands = ['delta', 'beta', 'high_gamma']
    
    # Initialize results dictionary
    manifold_results = {}
    
    # Analyze each frequency band
    for band_name in bands:
        print(f"\n\n===== Analyzing {band_name.upper()} band neural manifolds =====")
        
        # Compute neural manifolds
        manifold_dict = compute_neural_manifold(
            region_epochs,
            region_channels_dict,
            band_name,
            n_components=n_components,
            downsample_factor=downsample_factor,
            plot=True,
            plot_title=f"Regions: {', '.join(region_labels)}",
            output_dir=output_dir
        )
        
        # Save results
        manifold_results[band_name] = manifold_dict
    
    return manifold_results

# gesture-specific manifold analysis

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
    dict
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
            if plot and n_components == 3:
                plot_gesture_manifolds(subject_id, gesture_manifolds[subject_id], 
                                     epochs.times[::downsample_factor], 
                                     band_name, output_dir)
            
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

def analyze_gesture_manifolds(region_epochs, region_channels_dict, region_labels, 
                            bands=None, gestures=None, n_components=3, 
                            downsample_factor=1, output_dir=None):
    """
    Analyze neural manifolds for each gesture across multiple frequency bands.
    
    Parameters:
    -----------
    region_epochs : dict
        Dictionary mapping subject IDs to region-specific epochs objects
    region_channels_dict : dict
        Dictionary mapping subject IDs to lists of channel names
    region_labels : list
        List of brain region names included in the analysis
    bands : list, optional
        List of frequency bands to analyze. If None, uses ['delta', 'beta', 'high_gamma']
    gestures : list, optional
        List of gesture names to analyze. If None, uses all available gestures
    n_components : int, optional
        Number of PCA components to compute
    downsample_factor : int, optional
        Factor by which to downsample the data before PCA
    output_dir : str, optional
        Directory to save plots. If None, plots are displayed but not saved.
        
    Returns:
    --------
    dict
        Dictionary mapping band names to dictionaries of gesture manifolds
    """
    # Set default bands if not provided
    if bands is None:
        bands = ['delta', 'beta', 'high_gamma']
    
    # Get list of gestures if not provided
    if gestures is None:
        # Get gestures from the first subject's epochs
        first_subject_id = list(region_epochs.keys())[0]
        gestures = list(region_epochs[first_subject_id].event_id.keys())
    
    # Initialize results dictionary
    gesture_manifold_results = {}
    
    # Analyze each frequency band
    for band_name in bands:
        print(f"\n\n===== Analyzing {band_name.upper()} band gesture-specific neural manifolds =====")
        
        # Compute neural manifolds for each gesture
        gesture_manifolds = compute_gesture_manifolds(
            region_epochs,
            region_channels_dict,
            band_name,
            gestures,
            n_components=n_components,
            downsample_factor=downsample_factor,
            plot=True,
            output_dir=output_dir
        )
        
        # Save results
        gesture_manifold_results[band_name] = gesture_manifolds
    
    return gesture_manifold_results


# Main program:
if __name__ == "__main__":
    # Define parameters
    region_labels = ["ctx-rh-precentral", "wm-rh-precentral"]
    subject_id_list = [34, 41]
    sampling_frequency = 1000
    trigger_type = 'stim'
    tmin = 0.2  
    tmax = 0.6  
    
    # Define event dictionaries
    event_dict_gest = {
        "elbow": 1,
        "scissor": 2,
        "rock": 3,
        "rotation": 4,
        "thumb": 5
    }
    mapping_events = {1: "elbow", 2: "scissor", 3: "rock", 4: "rotation", 5: "thumb"}
    
    # Extract and analyze region-specific data
    region_epochs, region_channels_dict = analyze_region_specific_data(
        region_labels,
        subject_id_list,
        sampling_frequency,
        mapping_events,
        event_dict_gest,
        trigger_type,
        tmin,
        tmax,
        plot=False
    )
    
    # Print a summary of the results
    print("\nSummary of extracted region-specific data:")
    print(f"Total subjects with data: {len(region_epochs)}")
    
    for subject_id in region_epochs:
        print(f"\nSubject {subject_id}:")
        print(f"  Number of channels: {len(region_channels_dict[subject_id])}")
        print(f"  Number of epochs: {len(region_epochs[subject_id])}")
        print(f"  Epoch duration: {region_epochs[subject_id].times[0]:.2f}s to {region_epochs[subject_id].times[-1]:.2f}s")
        print(f"  Number of time points: {len(region_epochs[subject_id].times)}")
        
        # Print event counts
        for event_name, event_id in event_dict_gest.items():
            event_count = len(region_epochs[subject_id][event_name])
            print(f"  {event_name} events: {event_count}")

    # Ask user which analysis to run
    print("\nChoose analysis to run:")
    print("1. Time-frequency analysis")
    print("2. Overall neural manifolds")
    print("3. Gesture-specific neural manifolds")
    print("4. All analyses")
    
    choice = input("Enter your choice (1-4): ")
    
    # Set output directories
    import os
    os.makedirs("tf_plots", exist_ok=True)
    os.makedirs("manifold_plots", exist_ok=True)
    os.makedirs("gesture_manifold_plots", exist_ok=True)
    
    # Initialize result variables
    manifold_results = None
    gesture_manifold_results = None
    
    # Run selected analyses
    if choice in ["1", "4",]:
        print("\nRunning time-frequency analysis...")
        tfr_power_dict = perform_time_frequency_analysis(
            region_epochs,
            region_channels_dict,
            region_labels,
            tmin=-1.0,  # Start time for analysis
            tmax=3.0,  # End time for analysis
            output_dir="tf_plots"  # Set to a path to save plots, or None to display
        )
    
    if choice in ["2", "4"]:
        print("\nAnalyzing overall neural manifolds...")
        manifold_results = analyze_neural_manifolds(
            region_epochs,
            region_channels_dict,
            region_labels,
            bands=['delta', 'beta', 'high_gamma'],  # Specific bands to analyze
            n_components=3,                         # For 3D visualization
            downsample_factor=1,                    # Reduce data size for faster processing
            output_dir="manifold_plots"             # Save visualizations
        )
    
    if choice in ["3", "4"]:
        print("\nAnalyzing gesture-specific neural manifolds...")
        gesture_manifold_results = analyze_gesture_manifolds(
            region_epochs,
            region_channels_dict,
            region_labels,
            bands=['delta', 'beta', 'high_gamma'],  # Specific bands to analyze
            gestures=list(event_dict_gest.keys()),  # All gestures
            n_components=3,                         # For 3D visualization
            downsample_factor=1,                    # Reduce data size for faster processing
            output_dir="gesture_manifold_plots"     # Save visualizations
        )
    
    print("\nAnalysis completed successfully!")