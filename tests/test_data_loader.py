import os

import matplotlib.pyplot as plt
import numpy as np
import mne
import pytest

from config import ECoGDataConfig
from loader import ECoGDataset

NUM_CHANNELS = 64
FILE_SAMPLING_FREQUENCY = 512


def create_fake_sin_data():
    # Create fake data to be used in tests. Sin wave scaled by index of channel
    # Add sine wave scaled by electrode index to all channels.
    times = np.linspace(0, 100, 100 * FILE_SAMPLING_FREQUENCY)
    data = np.array([(i + 1) * np.sin(np.pi * times) for i in range(NUM_CHANNELS + 1)])
    return data


@pytest.fixture
def create_fake_mne_file_fn(tmp_path):
    def create_fake_mne_file(ch_names: list[str], data: np.array):
        """Creates a fake mne file in tmp_dir with ch_names channels and data.

        Args:
            ch_names (np.array): List of channel names. Must have length data.shape[0]
            data (np.array): Data to write to file. Must have data.shape[0] == len(ch_names)

        Returns:
            str: path to fake file
        """
        info = mne.create_info(
            ch_names=ch_names,
            sfreq=FILE_SAMPLING_FREQUENCY,
            ch_types="misc",
        )

        simulated_raw = mne.io.RawArray(data, info)

        data_path = os.path.join(tmp_path, "simulated_data_raw.fif")
        # mne.export.export_raw(data_path, simulated_raw, add_ch_type=True)
        simulated_raw.save(data_path)

        return data_path

    return create_fake_mne_file


@pytest.fixture
def data_loader_creation_fn(create_fake_mne_file_fn):
    def get_data_loader(
        config: ECoGDataConfig,
        ch_names: list[str] = ["G" + str(i + 1) for i in range(NUM_CHANNELS + 1)],
        data: np.array = create_fake_sin_data(),
    ) -> ECoGDataset:
        fake_mne_file = create_fake_mne_file_fn(ch_names, data)
        return ECoGDataset(fake_mne_file, FILE_SAMPLING_FREQUENCY, config)

    return get_data_loader


def test_data_loader_can_handle_configurable_bands(data_loader_creation_fn):
    config = ECoGDataConfig(
        batch_size=32, bands=[[4, 8], [8, 13], [13, 30], [30, 55]], new_fs=20, max_samples=10
    )
    data_loader = data_loader_creation_fn(config)
        
    for data in data_loader:
        assert data.shape == (
            len(config.bands),
            config.sample_length * config.new_fs,
            1,
            8,
            8,
        )


def test_data_loader_grid_creation_returns_input_data(data_loader_creation_fn):
    config = ECoGDataConfig(
        batch_size=32, bands=[[4, 8], [8, 13], [13, 30], [30, 55]], new_fs=20, max_samples=10
    )
    data_loader = data_loader_creation_fn(config)

    # Iterate through all possible data.
    index = 0
    fake_data = create_fake_sin_data()

    while index < data_loader.max_samples:
        assert np.allclose(
            data_loader._load_grid_data(index),
            fake_data[
                :64,
                (config.sample_length * index * FILE_SAMPLING_FREQUENCY) : (
                    (index + 1) * config.sample_length * FILE_SAMPLING_FREQUENCY
                ),
            ],
        )

        index += 1


def test_data_loader_can_handle_missing_channel(data_loader_creation_fn):
    config = ECoGDataConfig(
        batch_size=32, bands=[[4, 8], [8, 13], [13, 30], [30, 55]], new_fs=20, max_samples=10
    )
    # Omit "G1" from list of channels so loader will fill that channels with zeros.
    ch_names = ["G" + str(i + 1) for i in range(1, 65)]
    # To make checking easier just make data all ones.
    fake_data = np.ones((len(ch_names), 100 * FILE_SAMPLING_FREQUENCY))
    data_loader = data_loader_creation_fn(config, ch_names, fake_data)

    # Iterate through all possible data.
    index = 0

    while index < data_loader.max_samples:
        actual_data = data_loader._load_grid_data(index)
        assert np.allclose(actual_data[0], np.zeros(config.sample_length * FILE_SAMPLING_FREQUENCY))
        assert np.allclose(
            actual_data[1:],
            np.ones((63, config.sample_length * FILE_SAMPLING_FREQUENCY)),
        )

        index += 1
        

# def test_data_loader_can_handle_durations_not_divisible_by_sample_length(data_loader_creation_fn):
#     config = ECoGDataConfig(
#         batch_size=32, bands=[[4, 8], [8, 13], [13, 30], [30, 55]], new_fs=20, max_samples=10.498, sample_length=1
#     )
#     # Create data for 10.5 seconds.
#     fake_data = np.ones((65, int(10.5 * FILE_SAMPLING_FREQUENCY)))
#     data_loader = data_loader_creation_fn(config, data=fake_data)

#     index = 0

#     while index < data_loader.max_samples:
#         # Data should be ones for all samples except for the last one which should have 0's padded in.
#         actual_data = data_loader._load_grid_data(index)
#         if index == 10:
#             last_signal = np.ones((64, int(0.5 * FILE_SAMPLING_FREQUENCY)))
#             padding = np.zeros((64, int(0.5 * FILE_SAMPLING_FREQUENCY)))
#             expected_data = np.hstack([last_signal, padding])
#             assert np.allclose(
#                 actual_data,
#                 expected_data,
#             )
#         else:
#             assert np.allclose(
#                 actual_data,
#                 np.ones((64, config.sample_length * FILE_SAMPLING_FREQUENCY)),
#             )

#         index += 1
