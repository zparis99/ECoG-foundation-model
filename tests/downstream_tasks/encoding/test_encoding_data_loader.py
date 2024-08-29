import os
import pytest

import numpy as np
import scipy

from downstream_tasks.encoding.config import EncodingDataConfig
from downstream_tasks.encoding.load_signal import EncodingDataset

ELECTRODE_FILE_FORMAT_STR = "electrode_{elec_id}.mat"

@pytest.fixture
def create_fake_electrode_files_fn(tmp_path):
    def create_fake_electrode_files(signals: np.array):
        """Writes fake .mat files with given signals.

        Args:
            signals (np.array): Shape [num_electrodes, num_signals]. Writes one file for every electrode.
        """
        for i, signal in enumerate(signals):
            scipy.io.savemat(os.path.join(tmp_path, ELECTRODE_FILE_FORMAT_STR.format(elec_id = i + 1)), {"p1st": signal})

        return tmp_path

    return create_fake_electrode_files


def test_correctly_locates_electrodes(create_fake_electrode_files_fn):
    fake_signal = np.ones((64, 1000))
    dataset_path = create_fake_electrode_files_fn(fake_signal)
    
    config = EncodingDataConfig(dataset_path=dataset_path, electrode_glob_path=ELECTRODE_FILE_FORMAT_STR)
    
    data_loader = EncodingDataset(512, config)
    
    actual_electode_data = data_loader._load_grid_data()
    
    assert np.all(actual_electode_data == fake_signal)

