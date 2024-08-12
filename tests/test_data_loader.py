import os

import matplotlib.pyplot as plt
import numpy as np
import mne
import pytest

from config import ECoGDataConfig
from loader import ECoGDataset

NUM_CHANNELS = 64
FILE_SAMPLING_FREQUENCY = 512


@pytest.fixture
def fake_mne_file(tmp_path):
    info = mne.create_info(
        ch_names=["G" + str(i + 1) for i in range(NUM_CHANNELS + 1)],
        sfreq=FILE_SAMPLING_FREQUENCY,
    )
    # Add sine wave scaled by electrode index to all channels.
    times = np.linspace(0, 100, 100 * FILE_SAMPLING_FREQUENCY)
    data = [(i + 1) * np.sin(np.pi * times) for i in range(NUM_CHANNELS + 1)]

    simulated_raw = mne.io.RawArray(data, info)

    data_path = os.path.join(tmp_path, "simulated_data.edf")
    mne.export.export_raw(data_path, simulated_raw)

    return data_path


@pytest.fixture
def data_loader_creation_fn(fake_mne_file):
    def get_data_loader(config: ECoGDataConfig) -> ECoGDataset:
        return ECoGDataset(fake_mne_file, FILE_SAMPLING_FREQUENCY, config)

    return get_data_loader


def test_data_loader_can_handle_configurable_bands(data_loader_creation_fn):
    config = ECoGDataConfig(
        batch_size=32, bands=[[4, 8], [8, 13], [13, 30], [30, 55]], new_fs=20
    )
    data_loader = data_loader_creation_fn(config)

    for data in data_loader:
        assert data.shape == (len(config.bands), config.sample_length * config.new_fs, 1, 8, 8)
