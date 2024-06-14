import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time as t
import os
import torch
import torch.nn as nn
from einops import rearrange
from tqdm import tqdm


def get_signal_stats(args, signal, signal_stats, epoch, dl_i):
    """
    Get mean and standard deviation of the raw signal that is streamed from the dataloader.

    Args:
        args
        signal: original raw signal
        signal_stats:
        epoch: current epoch
        dl_i: current dataloader iteration

    Returns:
        new_signal_stats: dataframe containing mean, standard deviation for each epoch and dataloader iteration
    """

    res_list = []
    i = 1
    bands = ["theta", "alpha", "beta", "gamma", "highgamma"]

    res = {}
    res["epoch"] = epoch
    res["dl_i"] = dl_i

    x = signal[:, :, :, :, :, :].flatten()
    res["mean"] = np.mean(x)
    res["std"] = np.std(x)

    new_signal_stats = pd.DataFrame(res)

    return new_signal_stats


def get_correlation(bands: list[list[int]], signal, recon_signal, epoch, dl_i):
    """
    Get pearson correlation between original and reconstructed signal.

    Args:
        bands: the bands of frequencies which we're filtering over
        signal: original normalized signal
        recon_signal: reconstructed signal
        epoch: current epoch
        dl_i: current dataloader iteration

    Returns:
        new_corr: dataframe containing correlation between normalized original and reconstructed signal
        for each epoch and dataloader iteration
    """

    signal = np.array(signal.detach().detach().cpu())
    recon_signal = np.array(recon_signal.detach().cpu())

    # calculate correlation
    res_list = []
    i = 1
    band_names = ["theta", "alpha", "beta", "gamma", "highgamma"]

    for h in range(0, 8):
        for w in range(0, 8):
            res = {}
            res["epoch"] = epoch
            res["dl_i"] = dl_i
            res["elec"] = i
            i += 1
            for c in range(0, len(bands)):
                # correlate across samples in batch
                x = signal[:, c, :, :, h, w].flatten()
                y = recon_signal[:, c, :, :, h, w].flatten()
                # add check to make sure x and y are the same length #TODO
                n = len(x)
                if np.isnan(x).any() or np.isnan(y).any():
                    r = 0
                else:
                    r = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (
                        np.sqrt(
                            (n * np.sum(x**2) - (np.sum(x)) ** 2)
                            * (n * np.sum(y**2) - (np.sum(y)) ** 2)
                        )
                    )

                res["band"] = band_names[c]
                res["corr"] = r
                res_list.append(res.copy())

    new_corr = pd.DataFrame(res_list)

    return new_corr


def get_correlation_across_elecs(bands, signal, recon_signal, epoch, dl_i):
    """
    Get pearson correlation between original and reconstructed signal averaged across all electrodes.

    Args:
        bands: the bands of frequencies which we're filtering over
        signal: original normalized signal
        recon_signal: reconstructed signal
        epoch: current epoch
        dl_i: current dataloader iteration

    Returns:
        new_corr: dataframe containing correlation between normalized original and reconstructed signal
        averaged across all electrodes for each epoch and dataloader iteration
    """

    signal = np.array(signal.detach().detach().cpu())
    recon_signal = np.array(recon_signal.detach().cpu())

    # calculate correlation
    res_list = []
    i = 1
    band_names = ["theta", "alpha", "beta", "gamma", "highgamma"]

    for c in range(0, len(bands)):

        res = {}
        res["epoch"] = epoch
        res["train_i"] = dl_i
        res["elec"] = 0

        # average across electrodes
        corrs = []
        for s in range(0, len(signal[0, 0, 0, :])):

            corrs = []

            x = signal[:, c, :, s].flatten()
            y = recon_signal[:, c, :, s].flatten()
            # add check to make sure x and y are the same length #TODO
            n = len(x)
            # this is for the channels that we zero padded, to avoid division by 0
            # we could also just exclude those channels
            if np.sum(x) == 0 or np.sum(y) == 0:
                r = 0
            else:
                r = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (
                    np.sqrt(
                        (n * np.sum(x**2) - (np.sum(x)) ** 2)
                        * (n * np.sum(y**2) - (np.sum(y)) ** 2)
                    )
                )
            corrs.append(r)

        res["band"] = band_names[c]
        res["corr"] = np.mean(corrs)
        res_list.append(res.copy())

    new_test_corr = pd.DataFrame(res_list)

    return new_test_corr


def get_model_recon(bands, signal, recon_signal, epoch):
    """
    Get original and reconstructed sample signal.

    Args:
        bands: frequency bands used for filtering
        signal: original normalized signal
        recon_signal: reconstructed signal
        epoch: current epoch

    Returns:
        new_model_recon: dataframe containing timnecourse of original and reconstructed sample signal for each epoch.
    """

    signal = np.array(signal.detach().detach().cpu())
    recon_signal = np.array(recon_signal.detach().cpu())

    res_list = []
    i = 1

    for h in range(0, 8):
        for w in range(0, 8):

            res = {}
            res["epoch"] = epoch
            res["elec"] = i
            i += 1

            x = signal[0, (len(bands) - 1), :, 0, h, w].flatten()
            y = recon_signal[0, (len(bands) - 1), :, 0, h, w].flatten()

            if np.isnan(x).any() or np.isnan(y).any():
                continue

            res["x"] = [x]
            res["y"] = [y]

            res_list.append(res.copy())

    new_model_recon = pd.DataFrame(res_list)

    return new_model_recon
