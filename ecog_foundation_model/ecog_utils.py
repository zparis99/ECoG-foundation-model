import numpy as np
import scipy.signal
from einops import rearrange  # Assuming this is from `from einops import rearrange`
import logging
from typing import Optional

from ecog_foundation_model import constants


logger = logging.getLogger(__name__)


def preprocess_and_normalize_neural_data(
    signal: np.ndarray,
    original_fs: int,
    target_fs: int,
    bands: Optional[list[list[int]]] = None,
    apply_envelope: bool = False,
    num_electrodes: int = 64,  # Expected number of electrodes for reshaping
    dtype=np.float32,
) -> np.ndarray:
    """
    Preprocesses and normalizes neural data from a given signal array.

    This function performs:
    1. Bandpass filtering for specified frequency bands.
    2. Optional power envelope extraction (Hilbert transform magnitude).
    3. Resampling (downsampling) to a target sampling frequency.
    4. Z-score normalization per-electrode, per-band across the entire time series.
    5. Reshaping the data into (bands, timepoints, grid_height, grid_width) format.

    Args:
        signal (np.ndarray): Input neural signal array of shape [num_electrodes, num_timepoints].
                             Assumes electrodes are ordered such that they can be reshaped
                             into a (GRID_SIZE, GRID_SIZE) grid if num_electrodes == GRID_SIZE*GRID_SIZE.
        original_fs (int): The original sampling frequency of the `signal`.
        target_fs (int): The desired target sampling frequency after resampling.
        bands (Optional[list[list[int]]], optional): A list of frequency bands, e.g., [[4, 8], [8, 12]].
                                                     If None, no bandpass filtering is applied, and the
                                                     signal is treated as a single band.
        apply_envelope (bool): If True, compute the power envelope using the Hilbert transform
                               after filtering. Defaults to False.
        num_electrodes (int): The expected number of electrodes in the input signal. This is used
                              for validation and reshaping. Defaults to 64.
        dtype: Data type for the output array. Defaults to np.float32.

    Returns:
        np.ndarray: Preprocessed and normalized neural data of shape
                    [num_bands, num_resampled_timepoints, GRID_SIZE, GRID_SIZE].
                    If `bands` is None, `num_bands` will be 1.
    """
    if signal.shape[0] != num_electrodes:
        raise ValueError(
            f"Input signal must have {num_electrodes} electrodes (rows), "
            f"but got {signal.shape[0]}."
        )
    if num_electrodes != constants.GRID_SIZE * constants.GRID_SIZE:
        logger.warning(
            f"Number of electrodes ({num_electrodes}) does not match "
            f"GRID_SIZE ({constants.GRID_SIZE}) squared. Reshaping might be problematic."
        )

    processed_bands_signals = []

    if bands:
        for freqs in bands:
            # Filtering
            sos = scipy.signal.butter(
                4, freqs, btype="bandpass", analog=False, output="sos", fs=original_fs
            )
            filtered_band = scipy.signal.sosfiltfilt(
                sos, signal, axis=-1
            )  # Filter along time axis
            if apply_envelope:
                filtered_band = np.abs(scipy.signal.hilbert(filtered_band, axis=-1))
            processed_bands_signals.append(filtered_band)
        combined_filtered_signal = np.array(processed_bands_signals, dtype=dtype)
        # combined_filtered_signal shape: [num_bands, num_electrodes, original_timepoints]
    else:
        # If no bands specified, treat the original signal as a single band
        combined_filtered_signal = np.expand_dims(signal, axis=0).astype(dtype)
        # combined_filtered_signal shape: [1, num_electrodes, original_timepoints]

    # --- Resampling (Downsampling by averaging) ---
    if original_fs != target_fs:
        if original_fs % target_fs != 0:
            logger.warning(
                f"Original FS ({original_fs}) is not perfectly divisible by Target FS ({target_fs}). "
                f"Resampling might truncate data or produce slightly uneven last window."
            )
        window_width = original_fs // target_fs
        # Ensure that the time dimension is perfectly divisible by window_width
        # If not, truncate the end of the signal
        num_original_timepoints = combined_filtered_signal.shape[-1]
        num_resampled_timepoints = num_original_timepoints // window_width
        truncated_signal = combined_filtered_signal[
            ..., : num_resampled_timepoints * window_width
        ]

        resampled_signal = truncated_signal.reshape(
            truncated_signal.shape[0],  # bands
            truncated_signal.shape[1],  # electrodes
            num_resampled_timepoints,
            window_width,
        ).mean(axis=-1)
    else:
        resampled_signal = combined_filtered_signal
    # resampled_signal shape: [num_bands, num_electrodes, resampled_timepoints]

    # --- Z-score Normalization (Per-electrode, Per-band) ---
    normalized_signal = np.zeros_like(resampled_signal, dtype=dtype)
    for band_idx in range(resampled_signal.shape[0]):
        for electrode_idx in range(resampled_signal.shape[1]):
            data_to_normalize = resampled_signal[band_idx, electrode_idx]
            # Handle NaN values: compute mean/std on non-NaNs, then fill NaNs
            non_nan_data = data_to_normalize[~np.isnan(data_to_normalize)]
            if non_nan_data.size > 0:
                mean = np.mean(non_nan_data)
                std = np.std(non_nan_data)
                if std != 0:
                    normalized_signal[band_idx, electrode_idx] = (
                        data_to_normalize - mean
                    ) / std
                else:
                    # If std is zero, all values are the same. Normalize to 0.
                    normalized_signal[band_idx, electrode_idx] = 0.0
            else:
                # If entire electrode/band is NaN, keep it NaN
                normalized_signal[band_idx, electrode_idx] = np.nan

    # --- Reshape for model input (c t h w) ---
    # This assumes 'num_electrodes' can be perfectly reshaped into 'constants.GRID_SIZE' x 'constants.GRID_SIZE'
    if num_electrodes != constants.GRID_SIZE * constants.GRID_SIZE:
        raise ValueError(
            f"Cannot reshape {num_electrodes} electrodes into a "
            f"{constants.GRID_SIZE}x{constants.GRID_SIZE} grid. "
            "Please ensure num_electrodes matches GRID_SIZE*GRID_SIZE."
        )

    final_preprocessed_signal = rearrange(
        normalized_signal,
        "c (h w) t -> c t h w",
        h=constants.GRID_SIZE,
        w=constants.GRID_SIZE,
    )
    # final_preprocessed_signal shape: [num_bands, num_resampled_timepoints, GRID_SIZE, GRID_SIZE]

    return final_preprocessed_signal
