"""
Data loading functions for neural data analysis.
"""

import numpy as np
import hdf5storage

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
