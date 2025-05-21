# Python script for EMG processing

# import required libraries
import pandas as pd
import numpy as np
import scipy.signal
import itertools

import hdf5storage
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def read_data(subject_id):
    """
    load the data from the .mat file
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

    # STILL NEEDED??
    channel_num = len(good_channels)

    # return the data, the number of channels and the good channels
    return data, good_channels, channel_num

def emg_process(emg_signal, sampling_rate=1000, is_clean=True, duration_min=0.0):
    """
    Automatically processes an EMG signal

    Parameters:
        emg_signal : np.array, the raw electromyography channel
        sampling_rate : int, the sampling frequency of ``emg_signal`` (in Hz)
        is_clean : bool, whether to clean the signal or not
        duration_min : to be fixed, do not perform as expected
            float, the minimum duration of a period of activity or non-activity in seconds

    Returns:
        signals : DataFrame, containing the following columns:
            EMG_Raw|The raw EMG signal
            EMG_Clean|The cleaned EMG signal
            EMG_Amplitude|The signal amplitude, or the activation of the signal
            EMG_Activity|The activity of the signal for which amplitude exceeds the threshold 
                specified, marked as "1" in a list of zeros
            EMG_Onsets|The onsets of the amplitude, marked as "1" in a list of zeros
            EMG_Offsets|The offsets of the amplitude, marked as "1" in a list of zeros
        info : dict, containing the information of each amplitude onset, offset, peak activity 
            and the signals' sampling rate
    """
    # Clean signal
    if is_clean:
        emg_cleaned = emg_clean(emg_signal, sampling_rate=sampling_rate)
    else:
        emg_cleaned = emg_signal

    # Get amplitude
    amplitude = emg_amplitude(emg_cleaned)

    # Get onsets, offsets, and periods of activity
    activity_signal, info = emg_activation(
        emg_amplitude=amplitude,
        sampling_rate=sampling_rate,
        threshold="default",
        duration_min=int(duration_min * sampling_rate),
    )
    info["sampling_rate"] = sampling_rate  # Add sampling rate in dict info

    # Prepare output
    signals = pd.DataFrame(
        {"EMG_Raw": emg_signal, "EMG_Clean": emg_cleaned, "EMG_Amplitude": amplitude}
    )

    signals = pd.concat([signals, activity_signal], axis=1)

    return signals, info

def emg_clean(emg_signal, sampling_rate=1000):
    """
    Clean an EMG signal using a fourth order 100 Hz highpass Butterworth filter followed by a 
    constant detrending

    Parameters:
        emg_signal : np.array

    Returns:
        array: Vector containing the cleaned EMG signal
    """

    # Missing data
    n_missing = np.sum(np.isnan(emg_signal))
    if n_missing > 0:
        emg_signal = pd.DataFrame.pad(pd.Series(emg_signal))

    # Filtering
    order = 4
    frequency = 100
    frequency = (2 * np.array(frequency) / sampling_rate)
    sos = scipy.signal.butter(N=order, Wn=frequency, btype="highpass", output="sos")
    filtered = scipy.signal.sosfiltfilt(sos, emg_signal)

    # Baseline detrending
    X = np.linspace(0, 100, len(filtered))
    coefs = np.polyfit(X, filtered, deg=0)
    y_predicted = np.polyval(coefs, X)
    clean = filtered - y_predicted
    
    return clean

def emg_amplitude(emg_cleaned, sampling_rate=1000):
    """
    Compute electromyography amplitude given the cleaned respiration signal, done by calculating the
    linear envelope of the signal

    Parameters:
        emg_cleaned : np.array

    Returns:
        array: a vector containing the electromyography amplitude
    """

    # Calculates the Teager–Kaiser Energy operator to improve onset detection
    tkeo = emg_cleaned.copy()
    tkeo[1:-1] = emg_cleaned[1:-1] * emg_cleaned[1:-1] - emg_cleaned[:-2] * emg_cleaned[2:]
    tkeo[0], tkeo[-1] = tkeo[1], tkeo[-2]

    # Calculate the linear envelope 
    order = 2
    frequency = [10, 400]
    frequency = (2 * np.array(frequency) / sampling_rate)
    sos = scipy.signal.butter(N=order, Wn=frequency, btype="bandpass", output="sos")
    filtered = scipy.signal.sosfiltfilt(sos, tkeo)

    envelope = np.abs(filtered)

    order = 2
    frequency = 8
    frequency = (2 * np.array(frequency) / sampling_rate)
    sos = scipy.signal.butter(N=order, Wn=frequency, btype="lowpass", output="sos")
    amplitude = scipy.signal.sosfiltfilt(sos, envelope)

    return amplitude

def emg_activation(
    emg_amplitude=None,
    sampling_rate=1000,
    threshold="default",
    duration_min="default",
):
    """
    Detects onset in EMG signal based on the amplitude threshold

    Parameters:
        emg_amplitude : array
        threshold : str， float
            it corresponds to the minimum amplitude to detect as onset, 
            defaults to one tenth of the standard deviation of ``emg_amplitude``
        duration_min : float
            The minimum duration of a period of activity or non-activity in seconds
            If ``default``, will be set to 0.05s

    Returns:
        activity_signal : DataFrame
            A DataFrame of same length as the input signal in which occurences of onsets, offsets, and
            activity (above the threshold) of the EMG signal are marked as "1" in lists of zeros with
            the same length as ``emg_amplitude``
        info : dict
            A dictionary containing additional information, in this case the samples at which the
            onsets, offsets, and periods of activations of the EMG signal occur
        
    """
    def _signal_binarize(signal, threshold="auto"):
        if threshold == "auto":
            threshold = np.mean([np.nanmax(signal), np.nanmin(signal)])
        if threshold == "mean":
            threshold = np.nanmean(signal)
        if threshold == "median":
            threshold = np.nanmedian(signal)

        binary = np.zeros(len(signal))
        binary[signal > threshold] = 1
        return binary
    
    def _events_find(event_channel, threshold="auto", threshold_keep="above", duration_min=0):
        """
        Find and select events in a continuous signal

        Parameters:
            event_channel : array or list or DataFrame
                The channel containing the events.
            threshold : str or float
                The threshold value by which to select the events. If ``"auto"``, takes the value between
                the max and the min.
            threshold_keep : str
                ``"above"`` or ``"below"``, define the events as above or under the threshold.
            duration_min : int
                The minimum duration of an event to be considered as such (in time points).

        Returns:
            dict: containing 2 arrays, ``"onset"`` for event onsets, ``"duration"`` for event durations
        """

        binary = _signal_binarize(event_channel, threshold=threshold)
        if threshold_keep != "above":
            binary = np.abs(binary - 1)  # Reverse if events are below

        # Initialize data
        events = {"onset": [], "duration": []}
        index = 0
        for event, group in itertools.groupby(binary):
            duration = len(list(group))
            if event == 1:
                events["onset"].append(index)
                events["duration"].append(duration)
            index += duration
        events["onset"] = np.array(events["onset"])
        events["duration"] = np.array(events["duration"])

        # Remove based on duration
        to_keep = np.full(len(events["onset"]), True)
        to_keep[events["duration"] < duration_min] = False
        events["onset"] = events["onset"][to_keep]
        events["duration"] = events["duration"][to_keep]

        return events

    def _emg_activation_activations(activity, duration_min=0.05):
        activations = _events_find(activity, threshold=0.5, duration_min=duration_min)
        activations["offset"] = activations["onset"] + activations["duration"]

        baseline = _events_find(activity == 0, threshold=0.5, duration_min=duration_min)
        baseline["offset"] = baseline["onset"] + baseline["duration"]

        # Cross-comparison
        valid = np.isin(activations["onset"], baseline["offset"])
        onsets = activations["onset"][valid]
        offsets = activations["offset"][valid]

        # make sure offset indices are within length of signal
        offsets = offsets[offsets < len(activity)]

        new_activity = np.array([])
        for x, y in zip(onsets, offsets):
            activated = np.arange(x, y)
            new_activity = np.append(new_activity, activated)

        # Prepare Output.
        info = {"EMG_Onsets": onsets, "EMG_Offsets": offsets, "EMG_Activity": new_activity}

        return info

    def _signal_formatpeaks(info, desired_length, peak_indices=None):
        """
        Transforms a peak-info dict to a signal of given length
        """

        def _signal_sanitize_indices(indices, values):
            # Check if nan in indices
            if np.sum(np.isnan(indices)) > 0:
                to_drop = np.argwhere(np.isnan(indices))[0]
                for i in to_drop:
                    indices = np.delete(indices, i)
                    values = np.delete(values, i)

            return indices, values


        def _signal_from_indices(indices, desired_length=None, value=1):

            signal = pd.Series(np.zeros(desired_length, dtype=float))

            if isinstance(indices, list) and (not indices):  # skip empty lists
                return signal
            if isinstance(indices, np.ndarray) and (indices.size == 0):  # skip empty arrays
                return signal

            # Force indices as int
            if isinstance(indices[0], float):
                indices = indices[~np.isnan(indices)].astype(int)

            # Appending single value
            if isinstance(value, (int, float)):
                signal[indices] = value
            # Appending multiple values
            elif isinstance(value, (np.ndarray, list)):
                for index, val in zip(indices, value):
                    signal.iloc[index] = val
            else:
                if len(value) != len(indices):
                    raise ValueError(
                        "error: _signal_from_indices(): The number of values is different from the number of indices."
                    )
                signal[indices] = value

            return signal

        signals = {}
        for feature, values in info.items():
            # Get indices of features
            if any(x in str(feature) for x in ["Onset", "Offset"]):
                signals[feature] = _signal_from_indices(values, desired_length, 1)
                signals[feature] = signals[feature].astype("int64")  # indexing of feature using 1 and 0

            else:
                # Sanitize indices and values
                peak_indices, values = _signal_sanitize_indices(peak_indices, values)
                # Append peak values to signal
                signals[feature] = _signal_from_indices(peak_indices, desired_length, values)

        signals = pd.DataFrame(signals)
        return signals
    

    if duration_min == "default":
        duration_min = int(0.05 * sampling_rate)

    # Find offsets and onsets.
    if threshold == "default":
        threshold = 0.1 * np.std(emg_amplitude)
    if threshold > np.max(emg_amplitude):
        raise ValueError(
            "error: emg_activation(): the threshold specified exceeds the maximum of the signal amplitude."
        )
    activity = _signal_binarize(emg_amplitude, threshold=threshold)

    # Sanitize activity.
    info = _emg_activation_activations(activity, duration_min=duration_min)

    # Prepare Output.
    df_activity = _signal_formatpeaks(
        {"EMG_Activity": info["EMG_Activity"]},
        desired_length=len(emg_amplitude),
        peak_indices=info["EMG_Activity"],
    )
    df_onsets = _signal_formatpeaks(
        {"EMG_Onsets": info["EMG_Onsets"]},
        desired_length=len(emg_amplitude),
        peak_indices=info["EMG_Onsets"],
    )
    df_offsets = _signal_formatpeaks(
        {"EMG_Offsets": info["EMG_Offsets"]},
        desired_length=len(emg_amplitude),
        peak_indices=info["EMG_Offsets"],
    )

    # Modify output produced by signal_formatpeaks.
    for x in range(len(emg_amplitude)):
        if df_activity.loc[x, "EMG_Activity"] != 0:
            if df_activity.index[x] == df_activity.index.get_loc(x):
                df_activity.loc[x, "EMG_Activity"] = 1
            else:
                df_activity.loc[x, "EMG_Activity"] = 0
        if df_offsets.loc[x, "EMG_Offsets"] != 0:
            if df_offsets.index[x] == df_offsets.index.get_loc(x):
                df_offsets.loc[x, "EMG_Offsets"] = 1
            else:
                df_offsets.loc[x, "EMG_Offsets"] = 0

    activity_signal = pd.concat([df_activity, df_onsets, df_offsets], axis=1)

    return activity_signal, info

def emg_plot(emg_signals, info=None):
    """
    Visualize EMG data.

    Parameters:
        emg_signals : DataFrame
        info : dict
    """
    def _emg_plot_interactive(emg_signals, x_axis, onsets, offsets, sampling_rate):

        # Prepare figure.
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
        fig.update_layout(title="EMG", font=dict(size=18), height=600)

        # Plot cleaned and raw EMG.
        fig.add_trace(
            go.Scatter(
                x=x_axis, y=emg_signals["EMG_Raw"],
                mode="lines", name="Raw", line=dict(color="#B0BEC5")
                ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x_axis, y=emg_signals["EMG_Clean"], 
                mode="lines", name="Cleaned", line=dict(color="#FFC107")
                ),
            row=1,
            col=1,
        )

        # Plot Amplitude.
        fig.add_trace(
            go.Scatter(
                x=x_axis, y=emg_signals["EMG_Amplitude"], 
                mode="lines", name="Amplitude", line=dict(color="#FF9800")
                ),
            row=2,
            col=1,
        )

        # Mark onsets and offsets.
        fig.add_trace(
            go.Scatter(
                x=x_axis[onsets], y=emg_signals["EMG_Amplitude"][onsets], 
                mode="markers", name="Onsets", marker=dict(color="#f03e65", size=10)
                ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x_axis[offsets], y=emg_signals["EMG_Amplitude"][offsets], 
                mode="markers", name="Offsets", marker=dict(color="#f03e65", size=10)
                ),
            row=2,
            col=1,
        )

        onsets = onsets / sampling_rate
        offsets = offsets / sampling_rate
        fig.update_xaxes(title_text="Time (seconds)", row=2, col=1)

        for i, j in zip(list(onsets), list(offsets)):
            fig.add_shape(
                type="line", x0=i, y0=0, x1=i, y1=1e6,
                line=dict(color="#4a4a4a", width=2, dash="dash"),
                row=2, col=1,
            )
            fig.add_shape(
                type="line", x0=j, y0=0, x1=j, y1=1e6,
                line=dict(color="#4a4a4a", width=2, dash="dash"),
                row=2, col=1,
            )

        fig.update_yaxes(title_text="Amplitude", row=2, col=1)
        return fig

    # Mark onsets, offsets, activity
    onsets = np.where(emg_signals["EMG_Onsets"] == 1)[0]
    offsets = np.where(emg_signals["EMG_Offsets"] == 1)[0]

    # Determine what to display on the x-axis, mark activity.
    x_axis = np.linspace(
        0, emg_signals.shape[0] / info["sampling_rate"], emg_signals.shape[0]
    )

   
    return _emg_plot_interactive(emg_signals, x_axis, onsets, offsets, info["sampling_rate"])

if __name__ == "__main__":
    
    data, _, _ = read_data(subject_id=39)
    emg0 = data[:, -4]
    emg1 = data[:, -3]
    emg_diff = emg0 - emg1

    # could take 50s to process EMG
    signals, info = emg_process(emg_diff, sampling_rate=1000, is_clean=False)
    # could take 7min to plot EMG
    emg_plot(signals, info)



