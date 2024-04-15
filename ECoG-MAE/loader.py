import numpy as np
import pandas as pd
import time as t
import mne
from pyedflib import highlevel
import scipy.signal
from einops import rearrange
import torch


class ECoGDataset(torch.utils.data.IterableDataset):
    def __init__(self, root, path, bands, fs, new_fs):
        self.root = root
        self.path = path
        self.bands = bands
        self.fs = fs
        self.new_fs = new_fs
        self.max_samples = highlevel.read_edf_header(edf_file=self.path)["Duration"] / 2
        self.index = 0

    def __iter__(self):
        while self.index < self.max_samples:
            yield self.sample_data()
            self.index += 1
        if self.index == self.max_samples:
            self.index = 0

    def sample_data(self):
        start = t.time()

        grid = np.linspace(1, 64, 64).astype(int)
        n_samples = int(2 * self.fs)
        n_new_samples = int(2 * self.new_fs)

        # load edf and extract signal
        raw = read_raw(self.path)
        sig = raw.get_data(
            picks=grid,
            start=(n_samples * self.index),
            stop=(n_samples * (self.index + 1)),
        )

        norm_sig = sig.copy()

        # normalize signal within each sec chunk
        for ch in range(0, len(sig)):
            norm_sig[ch] = sig[ch] - np.mean(sig[ch]) / np.std(sig[ch])

        # zero pad if chunk is shorter than 2 sec
        if len(norm_sig[0]) < n_samples:  
            padding = np.zeros((64, n_samples - len(norm_sig[0])))
            norm_sig = np.concatenate((norm_sig, padding), axis=1)

        # zero pad if channel is not included in grid
        for i in range(0, 64):
            chn = "G" + str(i + 1)

            if np.isin(chn, raw.info.ch_names) == False:
                # shift upwards
                norm_sig = np.insert(norm_sig, i, np.zeros((1, n_samples)), axis=0)

        # delete items that were shifted upwards
        norm_sig = norm_sig[:64, :]

        # extract frequency bands
        nyq = 0.5 * self.fs
        filtered = []

        for i in range(0, len(self.bands)):
            lowcut = self.bands[i][0]
            highcut = self.bands[i][1]
            low = lowcut / nyq
            high = highcut / nyq

            sos = scipy.signal.butter(N=4, Wn=[low, high], btype="band", output="sos")
            filtered.append(scipy.signal.sosfilt(sos, norm_sig))

        filtered = np.array(filtered)

        # compute power envelope
        envelope = np.abs(scipy.signal.hilbert(filtered, axis=2))

        # resample
        resampled = scipy.signal.resample(envelope, n_new_samples, axis=2)

        # rearrange into shape c*t*d*h*w, where d is currently 1
        out = rearrange(
            np.array(resampled, dtype=np.float32), "c (h w) t -> c t () h w", h=8, w=8
        )

        end = t.time()

        # print('Time elapsed: ' + str(end-start))

        return out
    

def split_dataframe(df, ratio):

    """
    Shuffles a pandas dataframe and splits it into two dataframes with the specified ratio

    Args:
        df: The dataframe to split
        ratio: The proportion of data for the first dataframe (default: 0.9)
        
    Returns:
        A tuple of two dataframes, the first containing ratio proportion of the data and the second containing 1-ratio proportion
    """

    # Shuffle the dataframe
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
        Path to edf file
        
    Returns:
        A mne raw object
    """

    raw = mne.io.read_raw(filename, verbose=False)
    raw.set_channel_types('ecog')

    return raw

    
def dl_setup(args):

    root = "/scratch/gpfs/ln1144/fm-preproc/dataset/derivatives/preprocessed"
    data = pd.read_csv("/scratch/gpfs/ln1144/fm-preproc/dataset/dataset.csv")
    # only look at subset of data
    data = data.iloc[: int(len(data) * args.data_size), :]
    train_data, test_data = split_dataframe(data, 0.9)
    bands = args.bands
    fs = 512
    new_fs = args.new_fs
    batch_size = args.batch_size

    train_datasets = []

    for i, row in train_data.iterrows():
        path = mne.BIDSPath(
            root=root,
            datatype="car",
            subject=f"{row.subject:02d}",
            task=f"part{row.task:03d}chunk{row.chunk:02d}",
            suffix="desc-preproc_ieeg",
            extension=".edf",
            check=False,
        )

        train_path = str(path.fpath)

        train_datasets.append(ECoGDataset(root, train_path, bands, fs, new_fs))

    train_dataset_combined = torch.utils.data.ChainDataset(train_datasets)
    train_dl = torch.utils.data.DataLoader(
        train_dataset_combined, batch_size=batch_size
    )

    test_datasets = []

    for i, row in test_data.iterrows():
        path = mne.BIDSPath(
            root=root,
            datatype="car",
            subject=f"{row.subject:02d}",
            task=f"part{row.task:03d}chunk{row.chunk:02d}",
            suffix="desc-preproc_ieeg",
            extension=".edf",
            check=False,
        )

        test_path = str(path.fpath)

        test_datasets.append(ECoGDataset(root, test_path, bands, fs, new_fs))

    test_dataset_combined = torch.utils.data.ChainDataset(test_datasets)
    test_dl = torch.utils.data.DataLoader(test_dataset_combined, batch_size=batch_size)

    return train_dl, test_dl