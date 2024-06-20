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


class ECoGDataset(torch.utils.data.IterableDataset):

    def __init__(self, path: str, fs: int, config: ECoGDataConfig):
        self.config = config
        self.path = path
        self.bands = config.bands
        self.fs = fs
        self.new_fs = config.new_fs
        self.sample_secs = config.sample_length
        # since we take sample_length sec samples, the number of samples we can stream from our dataset is determined by the duration of the chunk in sec divided by sample_length
        self.max_samples = highlevel.read_edf_header(edf_file=self.path)["Duration"] / config.sample_length
        if config.norm == "hour":
            self.means, self.stds = get_signal_stats(self.path)
        self.index = 0

    def __iter__(self):
        # this is to make sure we stop streaming from our dataset after the max number of samples is reached
        while self.index < self.max_samples:
            yield self.sample_data()
            self.index += 1
        # this is to reset the counter after we looped through the dataset so that streaming starts at 0 in the next epoch, since the dataset is not initialized again
        if self.index == self.max_samples:
            self.index = 0

    def sample_data(self) -> np.array:

        start = t.time()

        # here we define the grid - since for patient 798 grid electrodes are G1 - G64
        grid = np.linspace(1, 64, 64).astype(int)

        n_samples = int(self.sample_secs * self.fs)
        n_new_samples = int(self.sample_secs * self.new_fs)

        # load edf and extract signal
        raw = read_raw(self.path)

        # crop 2 sec sample segment
        sample = raw.crop(self.index * 2, (self.index + 1) * 2, include_tmax=False)
        sample.load_data(verbose=False)

        def func(input, ch_idx):

            output = input - self.means[ch_idx] / self.stds[ch_idx]

            return output

        if self.args.norm == "hour":

            for i in range(0, 64):

                raw.apply_function(func, picks=[i], channel_wise=True, ch_idx=i)

        # Extract frequency bands
        band_raws = []

        bands = {
            "alpha": (8, 13),
            "beta": (13, 30),
            "theta": (4, 8),
            "gamma": (30, 55),
            "highgamma": (70, 200),
        }

        iir_params = dict(order=4, ftype="butter")
        for band, freqs in bands.items():
            band_raw = sample.copy()
            band_raw = band_raw.filter(
                *freqs, picks="data", method="iir", iir_params=iir_params, verbose=False
            )
            band_raw = band_raw.apply_hilbert(envelope=True, verbose=False)
            band_raws.append(band_raw)

        sig = np.zeros((len(self.bands), 64, n_samples))

        for i in range(0, 5):

            if len(band_raws[i].get_data(picks=grid)[1]) < n_samples:
                padding = np.zeros((64, n_samples - len(sig[0])))
                sig = np.concatenate((sig, padding), axis=1)
            else:
                sig[i, :, :] = band_raws[i].get_data(
                    picks=grid,
                )

        # zero pad if channel is not included in grid #TODO a bit clunky right now, implement in a better and more flexible way
        # since we will load by index position of channel (so if a channel is not included it will load channel n+1 at position 1),
        # we correct that by inserting 0 at position n and shift value one upwards
        for i in range(0, 64):
            chn = "G" + str(i + 1)

            # first we check whether the channel is included
            if np.isin(chn, raw.info.ch_names) == False:
                # if not we insert 0 padding and shift upwards
                sig = np.insert(sig, i, np.zeros((5, 1, n_samples)), axis=1)

        # delete items that were shifted upwards
        sig = sig[:, :64, :]

        resampled = scipy.signal.resample(sig, n_new_samples, axis=2)

        # rearrange into shape c*t*d*h*w, where
        # c = freq bands,
        # t = number of datapoints within a sample
        # d = depth (currently 1)
        # h = height of grid (currently 8)
        # w = width of grid (currently 8)
        out = rearrange(
            np.array(resampled, dtype=np.float32), "c (h w) t -> c t () h w", h=8, w=8
        )

        end = t.time()

        # print('Time elapsed: ' + str(end-start))

        return out


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
    test_filepaths, _, test_samples = get_dataset_path_info(config.ecog_data_config.sample_length, root, test_data)
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
