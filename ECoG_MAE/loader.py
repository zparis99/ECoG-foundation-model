# TODO check filtering, envelope and resampling with Arnab, implement code such that we can flexibly load data from different patients

import numpy as np
import pandas as pd
import time as t
import os
import mne
from mne_bids import BIDSPath
import pyedflib
from pyedflib import highlevel
import scipy.signal
from einops import rearrange
import torch

from config import ECoGDataConfig, VideoMAEExperimentConfig
from utils import resample_mean_signals

class ECoGDataset(torch.utils.data.IterableDataset):

    def __init__(self, path: str, fs: int, config: ECoGDataConfig):
        self.config = config
        self.path = path
        self.bands = config.bands
        self.fs = fs
        self.new_fs = config.new_fs
        self.sample_secs = config.sample_length
        # since we take sample_length sec samples, the number of samples we can stream from our dataset is determined by the duration of the chunk in sec divided by sample_length.
        # Optionally can configure max_samples directly as well.
        self.max_samples = config.max_samples if config.max_samples else (
            highlevel.read_edf_header(edf_file=self.path)["Duration"]
            / config.sample_length
        )
        if config.norm == "hour":
            self.means, self.stds = get_signal_stats(self.path)
        self.index = 0
        self.signal = self._load_grid_data()

    def __iter__(self):
        print("Iter")
        # this is to make sure we stop streaming from our dataset after the max number of samples is reached
        while self.index < self.max_samples:
            yield self.sample_data()
            self.index += 1
            print(self.index)
        # this is to reset the counter after we looped through the dataset so that streaming starts at 0 in the next epoch, since the dataset is not initialized again
        if self.index >= self.max_samples:
            print("Reset!")
            self.index = 0
            print(self.index)

    def sample_data(self) -> np.array:

        start = t.time()
        
        start_sample = self.index * self.sample_secs * self.fs
        end_sample = (self.index + 1) * self.sample_secs * self.fs
        current_sample = self.signal[:, start_sample:end_sample]
        
        def norm(input, ch_idx):
            output = input - self.means[ch_idx] / self.stds[ch_idx]

            return output

        if self.config.norm == "hour":

            # z-score signal for each channel separately
            for i in range(0, 64):
                current_sample[i, :] = norm(current_sample[i], i)

        # Extract frequency bands
        filtered_signal = np.zeros((len(self.bands), 64, current_sample.shape[1]))
        
        for i, freqs in enumerate(self.bands):
            sos = scipy.signal.butter(4, freqs, btype="bandpass", analog=False, output="sos", fs=self.fs)
            filtered_signal[i] = scipy.signal.sosfiltfilt(sos, current_sample)
            filtered_signal[i] = np.abs(scipy.signal.hilbert(filtered_signal[i]))
        
        resampled = resample_mean_signals(filtered_signal, self.fs, self.new_fs)
        # rearrange into shape c*t*d*h*w, where
        # c = freq bands,
        # t = number of datapoints within a sample
        # d = depth (currently 1)
        # h = height of grid (currently 8)
        # w = width of grid (currently 8)
        preprocessed_signal = rearrange(
            np.array(resampled, dtype=np.float32), "c (h w) t -> c t () h w", h=8, w=8
        )
        
        # Zero-pad if sample is too short.
        expected_sample_length = self.sample_secs * self.new_fs
        if preprocessed_signal.shape[1] < expected_sample_length:
            padding = np.zeros((len(self.bands), expected_sample_length - preprocessed_signal.shape[1], 1, 8, 8))
            preprocessed_signal = np.concatenate((preprocessed_signal, padding), axis=1)

        end = t.time()

        # print('Time elapsed: ' + str(end-start))

        return preprocessed_signal

    
    def _load_grid_data(self):
        """Overridable function to load data from an mne file and return it in an unprocessed grid.
        
        Can be overridden to support different data types. Data will be preprocessed in the same way and returned via iteration over the dataset.
        
        Returns:
            numpy array of shape [number of electrodes, num_samples].
        """
        
        # load edf and extract signal
        raw = read_raw(self.path)
        
        # here we define the grid - since for patient 798 grid electrodes are G1 - G64
        grid_ch_names = []
        for i in range(64):
            channel = "G" + str(i + 1)
            if np.isin(channel, raw.info.ch_names):
                grid_ch_names.append(channel)
        
        sig = raw.get_data(
                picks=grid_ch_names,
            )
        n_samples = sig.shape[1]
        
        # zero pad if channel is not included in grid #TODO a bit clunky right now, implement in a better and more flexible way
        # since we will load by index position of channel (so if a channel is not included it will load channel n+1 at position 1),
        # we correct that by inserting 0 at position n and shift value one upwards
        for i in range(64):
            channel = "G" + str(i + 1)
            # first we check whether the channel is included
            if not np.isin(channel, raw.info.ch_names):
                # if not we insert 0 padding and shift upwards
                sig = np.insert(
                    sig, i, np.zeros((n_samples)), axis=0
                )
                
        # delete items that were shifted upwards
        sig = sig[:64, :]
                
        return sig


def get_signal_stats(path):

    reader = pyedflib.EdfReader(path)

    means = []
    stds = []

    for i in range(0, 64):

        signal = reader.readSignal(i, 0)
        means.append(np.mean(signal))
        stds.append(np.std(signal))

    return means, stds


def split_dataframe(shuffle: bool, df: pd.DataFrame, ratio: float):
    """
    Shuffles a pandas dataframe and splits it into two dataframes with the specified ratio

    Args:
        shuffle: If true then shuffle the dataframe before splitting.
        df: The dataframe to split
        ratio: The proportion of data for the first dataframe (default: 0.9)

    Returns:
        df1: train split dataframe containing a proportion of ratio of full dataframe
        df2: test split dataframe containing a proportion of 1-ratio of the full dataframe
    """

    # # Shuffle the dataframe
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)

    # Calculate the split index based on the ratios
    split_index = int(ratio * len(df))

    # Create the two dataframes
    df1 = df.iloc[:split_index, :]
    df2 = df.iloc[split_index:, :]

    return df1, df2


def read_raw(filename):
    """
    Reads and loads an edf file into a mne raw object: https://mne.tools/stable/auto_tutorials/raw/10_raw_overview.html

    Args:
        filename: Path to edf file

    Returns:
        raw: a mne raw instance
    """

    raw = mne.io.read_raw(filename, verbose=False)

    return raw


def get_dataset_path_info(
    sample_length: int, root: str, data_split: pd.DataFrame
) -> tuple[list[str], int, pd.DataFrame]:
    """Generates information about the data referenced in data_split.

    Args:
        sample_length (int): number of seconds for each sample
        root (str): Filepath to root of BIDS dataset.
        data_split (pd.DataFrame): Dataframe storing references to the files to be used in this data split. Should have columns subject, task, and chunk.

    Returns: (List of filepaths to be used for data_split, Number of  samples for the data split, Dataframe with columns {'name': <filepath>, 'num_samples': <number of samples in file>})
    """
    split_filepaths = []

    num_samples = 0

    sample_desc = []

    for i, row in data_split.iterrows():
        path = BIDSPath(
            root=root,
            datatype="car",
            subject=f"{row.subject:02d}",
            task=f"part{row.task:03d}chunk{row.chunk:02d}",
            suffix="desc-preproc_ieeg",
            extension=".edf",
            check=False,
        )

        data_path = str(path.fpath)
        sample_desc.append(
            {
                "name": data_path,
                "num_samples": int(
                    highlevel.read_edf_header(edf_file=data_path)["Duration"] / 2
                ),
            }
        )

        num_samples = num_samples + int(
            highlevel.read_edf_header(edf_file=data_path)["Duration"] / 2
        )

        split_filepaths.append(data_path)

    return split_filepaths, num_samples, pd.DataFrame(sample_desc)


def dl_setup(
    config: VideoMAEExperimentConfig,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, int]:
    """
    Sets up dataloaders for train and test split. Here, we use a chain dataset implementation, meaning we concatenate 1 hour chunks of our data as iterable datasets into a larger
    dataset from which we can stream - https://discuss.pytorch.org/t/using-chaindataset-to-combine-iterabledataset/85236

    Args:
        config: command line arguments

    Returns:
        train_dl: dataloader instance for train split
        test_dl: dataloader instance for test split
    """

    dataset_path = os.path.join(os.getcwd(), config.ecog_data_config.dataset_path)
    root = os.path.join(dataset_path, "derivatives/preprocessed")
    data = pd.read_csv(os.path.join(dataset_path, "dataset.csv"))

    # only look at subset of data
    data = data.iloc[: int(len(data) * config.ecog_data_config.data_size), :]
    # data = data.iloc[int(len(data) * (1 - config.ecog_data_config.data_size)) :, :]
    train_data, test_data = split_dataframe(
        config.ecog_data_config.shuffle,
        data,
        config.ecog_data_config.train_data_proportion,
    )

    fs = 512

    # load and concatenate data for train split
    train_filepaths, num_train_samples, train_samples = get_dataset_path_info(
        config.ecog_data_config.sample_length, root, train_data
    )
    train_datasets = [
        ECoGDataset(train_path, fs, config.ecog_data_config)
        for train_path in train_filepaths
    ]
    train_dataset_combined = torch.utils.data.ChainDataset(train_datasets)
    train_dl = torch.utils.data.DataLoader(
        train_dataset_combined, batch_size=config.ecog_data_config.batch_size
    )

    # load and concatenate data for test split
    test_filepaths, _, test_samples = get_dataset_path_info(
        config.ecog_data_config.sample_length, root, test_data
    )
    test_datasets = [
        ECoGDataset(test_path, fs, config.ecog_data_config)
        for test_path in test_filepaths
    ]
    test_dataset_combined = torch.utils.data.ChainDataset(test_datasets)
    test_dl = torch.utils.data.DataLoader(
        test_dataset_combined, batch_size=config.ecog_data_config.batch_size
    )

    dir = os.getcwd() + f"/results/samples/"
    if not os.path.exists(dir):
        os.makedirs(dir)

    train_samples.to_csv(
        dir + f"{config.job_name}_train_samples.csv",
        index=False,
    )

    test_samples.to_csv(
        dir + f"{config.job_name}_test_samples.csv",
        index=False,
    )

    return train_dl, test_dl, num_train_samples
