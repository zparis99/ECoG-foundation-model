import pytest

from einops import rearrange
from einops.layers.torch import Rearrange
import numpy as np
import torch

from config import ViTConfig
from mask import get_tube_mask, get_decoder_mask
from utils import resample_mean_signals, rearrange_signals, get_signal_correlations

FRAME_PATCH_SIZE = 4
NUM_BANDS = 5
GRID_HEIGHT = 8
GRID_WIDTH = 8
NUM_FRAMES = 40

@pytest.fixture
def fake_model():
    class FakeModel():
        def __init__(self):
            self.patchify = Rearrange(
                "b c (f pf) (d pd) (h ph) (w pw) -> b f d h w (pd ph pw pf c)",
                pd=1,
                ph=1,
                pw=1,
                pf=FRAME_PATCH_SIZE,
            )

    return FakeModel()

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
    

def test_get_signal_correlations():
    signal_a = torch.ones(16, 64, 5, 40)
    
    for i in range(signal_a.shape[-1]):
        signal_a[:, :, :, i] *= -1**i * i
    
    signal_b = signal_a * -1
    
    corr_matrix = get_signal_correlations(signal_a, signal_b)
    
    assert corr_matrix.detach().numpy().shape == (64, 5)
    assert torch.isclose(corr_matrix, -torch.ones_like(corr_matrix)).all()
    