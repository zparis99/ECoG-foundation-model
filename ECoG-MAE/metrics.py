import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time as t
import os
import torch
import torch.nn as nn
from einops import rearrange
from tqdm import tqdm


def get_signal_stats(args, signal, signal_stats, epoch, test_i):

    res_list = []
    i = 1
    bands = ["theta", "alpha", "beta", "gamma", "highgamma"]

    res = {}
    res["epoch"] = epoch
    res["test_i"] = test_i

    x = signal[:, :, :, :, :, :].flatten()
    res["mean"] = np.mean(x)
    res["std"] = np.std(x)

    new_signal_stats = pd.DataFrame(res)

    return new_signal_stats


def get_correlation(args, signal, recon_signal, epoch, test_i):

    # calculate correlation
    res_list = []
    i = 1
    bands = ["theta", "alpha", "beta", "gamma", "highgamma"]

    for h in range(0, 8):
        for w in range(0, 8):
            res = {}
            res["epoch"] = epoch
            res["test_i"] = test_i
            res["elec"] = i
            i += 1
            for c in range(0, len(args.bands)):
                # correlate across samples in batch
                x = signal[:, c, :, :, h, w].flatten()
                y = recon_signal[:, c, :, :, h, w].flatten()
                # add check to make sure x and y are the same length #TODO
                n = len(x)
                # this is for the channels that we zero padded, to avoid division by 0
                # we could also just exclude those channels
                if np.sum(x) == 0:
                    r = 0
                else:
                    r = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (
                        np.sqrt(
                            (n * np.sum(x**2) - (np.sum(x)) ** 2)
                            * (n * np.sum(y**2) - (np.sum(y)) ** 2)
                        )
                    )

                res["band"] = bands[c]
                res["corr"] = r
                res_list.append(res.copy())

    new_test_corr = pd.DataFrame(res_list)

    return new_test_corr


def get_correlation_across_elecs(args, signal, recon_signal, epoch, test_i):

    # calculate correlation
    res_list = []
    i = 1
    bands = ["theta", "alpha", "beta", "gamma", "highgamma"]

    for c in range(0, len(args.bands)):

        res = {}
        res["epoch"] = epoch
        res["train_i"] = test_i
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

        res["band"] = bands[c]
        res["corr"] = np.mean(corrs)
        res_list.append(res.copy())

    new_test_corr = pd.DataFrame(res_list)

    return new_test_corr


def get_model_recon(signal, recon_signal, epoch):

    res_list = []
    i = 1

    for h in range(0, 8):
        for w in range(0, 8):

            res = {}
            res["epoch"] = epoch
            res["elec"] = i
            i += 1

            x = signal[8, 4, :, 0, h, w].flatten()
            y = recon_signal[8, 4, :, 0, h, w].flatten()

            res["x"] = [x]
            res["y"] = [y]

            res_list.append(res.copy())

    new_model_recon = pd.DataFrame(res_list)

    return new_model_recon
