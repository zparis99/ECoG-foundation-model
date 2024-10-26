import os

import numpy as np
import mne
import pytest

from config import ECoGDataConfig
from loader import ECoGDataset

@pytest.fixture
def create_fake_mne_file_fn(tmp_path):
    def create_fake_mne_file(ch_names: list[str], data: np.array, file_sampling_frequency: int):
        """Creates a fake mne file in tmp_dir with ch_names channels and data.

        Args:
            ch_names (np.array): List of channel names. Must have length data.shape[0]
            data (np.array): Data to write to file. Must have data.shape[0] == len(ch_names)

        Returns:
            str: path to fake file
        """
        info = mne.create_info(
            ch_names=ch_names,
            sfreq=file_sampling_frequency,
            ch_types="misc",
        )

        simulated_raw = mne.io.RawArray(data, info)

        data_path = os.path.join(tmp_path, "simulated_data_raw.fif")
        simulated_raw.save(data_path)

        return data_path

    return create_fake_mne_file


@pytest.fixture
def data_loader_creation_fn(create_fake_mne_file_fn):
    def get_data_loader(
        config: ECoGDataConfig,
        data: np.array,
        ch_names: list[str] = ["G" + str(i + 1) for i in range(64 + 1)],
        file_sampling_frequency: int = 512,
    ) -> ECoGDataset:
        config.original_fs = file_sampling_frequency
        fake_mne_file = create_fake_mne_file_fn(ch_names, data, file_sampling_frequency)
        return ECoGDataset(fake_mne_file, config)

    return get_data_loader
