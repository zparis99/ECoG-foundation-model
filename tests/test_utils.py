import numpy as np

from utils import resample_mean_signals


def test_resampling_correctly_averages_windows():
    # Signal of shape [bands, electrodes, samples]
    input_signal = np.array(
        [
            [[1, 1, 1, 1, 2, 2, 2, 2], [3, 3, 3, 3, 4, 4, 4, 4]],
            [[5, 5, 5, 5, 6, 6, 6, 6], [7, 7, 7, 7, 8, 8, 8, 8]],
        ]
    )

    old_fs = 4
    new_fs = 1

    assert np.allclose(
        resample_mean_signals(input_signal, old_fs, new_fs),
        np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
    )
   
    
def test_resampling_can_handle_not_perfectly_divisible_durations():
    # Signal of shape [bands, electrodes, samples]
    input_signal = np.array(
        [
            [[1, 1, 1, 1, 2, 2, 2, 2, 10], [3, 3, 3, 3, 4, 4, 4, 4, 11]],
            [[5, 5, 5, 5, 6, 6, 6, 6, 12], [7, 7, 7, 7, 8, 8, 8, 8, 13]],
        ]
    )

    old_fs = 4
    new_fs = 1

    assert np.allclose(
        resample_mean_signals(input_signal, old_fs, new_fs),
        np.array([[[1, 2, 10], [3, 4, 11]], [[5, 6, 12], [7, 8, 13]]]),
    )
    
def test_resampling_can_handle_non_divisible_sampling_rates():
    # Signal of shape [bands, electrodes, samples]
    input_signal = np.array(
        [
            # This is not an  optimal test setup because it assumes the implementation of the function
            # but suffices for now. This could be made better in the future.
            [[1, 2, 3, 3], [4, 5, 6, 6]],
            [[7, 8, 9, 9], [10, 11, 12, 12]],
        ]
    )

    old_fs = 4
    new_fs = 3

    assert np.allclose(
        resample_mean_signals(input_signal, old_fs, new_fs),
        np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]),
    )
