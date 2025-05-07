import numpy as np
import scipy
from einops import rearrange
from typing import Optional

import constants


# TODO: Test this function.
def preprocess_neural_data(
    signal: np.array,
    fs: int,
    new_fs: int,
    sample_secs: int,
    bands: Optional[list[list[int]]] = None,
    env: Optional[bool] = False,
    pad_before_sample: bool = False,
    dtype=np.float32,
) -> np.array:
    """Preprocess and reshape neural data for VideoMAE model.

    Args:
        signal (np.array): Of shape [num_electrodes, num_samples]. Should already be cropped to the desired number of samples.
        fs (int): The sampling rate of the signal.
        new_fs (int): The sampling rate to resample the data to.
        sample_secs (int): The number of seconds in a sample.
        bands (Optional[list[list[int]]], optional): Should be a list of the form [[4, 8], [10, 50]] where
            each set of two numbers represents a band of frequencies to filter from the provided signal.
            If not set then signal is assumed to represent one band and is used as a lone band signal.
            Defaults to None.
        norm (Optional[str], optional): If "hour" then will use the passed means and stds to normalize the signal. Defaults to None.
        means (Optional[np.array], optional): Of shape [num_electrodes]. Means for each electrode. Defaults to None.
        stds (Optional[np.array], optional): Of shape [num_electrodes]. Standard deviations for each electrode. Defaults to None.
        env (Optional[bool]): If true then apply power envelope to signal after filtering. Else just return filtered signal.
        pad_before_sample (bool): If true then samples which are not the desired length will be padded with 0's before the actual extracted signal. Useful if sample is taken from the very start of the signal.

    Returns:
        np.array:
            shape t*h*w*c, where
            t = number of datapoints within a sample
            h = height of grid (currently 8)
            w = width of grid (currently 8)
            c = freq bands
    """

    # Extract frequency bands if provided.
    if bands:
        filtered_signal = np.zeros((len(bands), 64, signal.shape[1]))

        for i, freqs in enumerate(bands):
            sos = scipy.signal.butter(
                4, freqs, btype="bandpass", analog=False, output="sos", fs=fs
            )
            filtered_signal[i] = scipy.signal.sosfiltfilt(sos, signal)
            if env:
                filtered_signal[i] = np.abs(scipy.signal.hilbert(filtered_signal[i]))
    else:
        # Add band axis of size 1 for non-filtered data.
        filtered_signal = np.expand_dims(signal, axis=0)

    if fs != new_fs:
        window_width = fs // new_fs
        resampled = filtered_signal.reshape(
            filtered_signal.shape[0], filtered_signal.shape[1], -1, window_width
        ).mean(-1)
    else:
        resampled = filtered_signal
    # rearrange into shape c*t*d*h*w, where
    # c = freq bands
    # t = number of datapoints within a sample
    # h = height of grid (currently 8)
    # w = width of grid (currently 8)
    preprocessed_signal = rearrange(
        np.array(resampled, dtype=np.float32),
        "c (h w) t -> c t h w",
        h=constants.GRID_SIZE,
        w=constants.GRID_SIZE,
    )

    # Zero-pad if sample is too short.
    expected_sample_length = int(sample_secs * new_fs / 1000)
    if preprocessed_signal.shape[1] < expected_sample_length:
        padding = (
            np.ones(
                (
                    preprocessed_signal.shape[0],
                    expected_sample_length - preprocessed_signal.shape[1],
                    1,
                    8,
                    8,
                ),
                dtype=dtype,
            )
            * np.nan
        )
        if pad_before_sample:
            preprocessed_signal = np.concatenate((padding, preprocessed_signal), axis=1)
        else:
            preprocessed_signal = np.concatenate((preprocessed_signal, padding), axis=1)

    return preprocessed_signal
