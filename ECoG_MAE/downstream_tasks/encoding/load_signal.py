# Here we want to load the signal for the grid electrodes and organize it in an 8*8 matrix. We also want to perform filtering here - we might need to think about how to do it
# such that it is the same as the filtering we perform for model training - since there we do it using mne, which we can't here, since the preprocessed signal here are mat files.

import os
import glob
import numpy as np
import pandas as pd
import scipy

from utils import get_signal_stats, preprocess_neural_data
from downstream_tasks.encoding.config import EncodingDataConfig


class EncodingDataset:
    def __init__(self, fs: int, config: EncodingDataConfig):
        self.config = config
        self.neural_dataset_path = config.dataset_path
        self.electrode_glob_path = config.electrode_glob_path
        self.fs = fs
        self.lag = config.lag
        self.new_fs = config.new_fs
        self.sample_secs = config.sample_length

        self.signal = self._load_grid_data()
        conversation_data = pd.read_csv("word-embeddings/gpt2-layer-8-emb.pkl", index_col=0)
        # Limit only to words with onset time data.
        self.timed_word_data = conversation_data.loc[
            conversation_data["onset"].dropna().index
        ]

        # since we take sample_length sec samples, the number of samples we can stream from our dataset is determined by the duration of the chunk in sec divided by sample_length.
        # Optionally can configure max_samples directly as well.
        self.max_samples = self.signal.shape[1] / self.fs / config.sample_length
        if config.norm == "hour":
            self.means, self.stds = get_signal_stats(self.signal)
        else:
            self.means = None
            self.stds = None

        self.index = 0


    def _load_grid_data(self):
        grid_data = []
        for i in range(64):
            curr_electrode_glob_path = self.electrode_glob_path.format(elec_id=str(i + 1))
            final_glob_path = os.path.join(
                self.neural_dataset_path, curr_electrode_glob_path
            )
            electrode_file = glob.glob(final_glob_path)

            if len(electrode_file) != 1:
                raise ValueError(
                    "There can only be one matching file associated with electrode {}. Got {} files matching {}.".format(
                        i + 1, len(electrode_file), final_glob_path
                    )
                )

            grid_data.append(scipy.io.loadmat(electrode_file[0])["p1st"])

        return np.array(grid_data)

    def __iter__(self):
        """Iterate through dataset for encoding task.

        Yields:
            tuple[np.array, np.array]: (word embedding, neural embedding)
        """
        if self.index >= self.timed_words_data.shape[0]:
            self.index = 0
        else:
            word_data = self.timed_words_data.iloc[self.index]

            lag_start_time = word_data.loc["onset"] + self.lag
            lag_start_sample = int(lag_start_time / 1000 * self.fs)
            curr_sample = self.signal[
                :, lag_start_sample : lag_start_sample + self.fs * self.sample_secs
            ]
            preprocessed_signal = preprocess_neural_data(curr_sample)

            yield word_data.loc["embeddings"], preprocessed_signal

            self.index += 1
