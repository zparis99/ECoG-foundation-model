import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time as t
import os
import torch
import torch.nn as nn
from einops import rearrange
from tqdm import tqdm


def get_correlation(args, test_corr, signal, output, epoch, test_i):

    # calculate correlation
    res_list = []
    i = 1
    bands = ["theta", "alpha", "beta", "gamma", "highgamma"]

    for h in range(0, 8):
        for w in range(0, 8):
            res = {}
            res["epoch"] = epoch
            res["train_i"] = test_i
            res["elec"] = "G" + str(i)
            i += 1
            for c in range(0, len(args.bands)):
                # correlate across samples in batch
                x = signal[:, c, :, :, h, w].flatten()
                y = output[:, c, :, :, h, w].flatten()
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


def get_correlation_across_elecs(args, signal, output, epoch, test_i):

    # calculate correlation
    res_list = []
    i = 1
    bands = ["theta", "alpha", "beta", "gamma", "highgamma"]

    for c in range(0, len(args.bands)):

        res = {}
        res["epoch"] = epoch
        res["train_i"] = test_i

        # average across electrodes
        corrs = []
        for s in range(0, len(signal[0, 0, 0, :])):

            corrs = []

            x = signal[:, c, :, s].flatten()
            y = output[:, c, :, s].flatten()
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
            corrs.append(r)

        res["band"] = bands[c]
        res["corr"] = np.mean(corrs)
        res_list.append(res.copy())

    new_test_corr = pd.DataFrame(res_list)

    return new_test_corr


def get_recon_signals(signal, output, epoch):

    res_list = []
    i = 1

    for h in range(0, 8):
        for w in range(0, 8):

            res = {}
            res["epoch"] = epoch
            res["elec"] = "G" + str(i)
            i += 1

            x = signal[0, 4, :, 0, h, w].flatten()
            y = output[0, 4, :, 0, h, w].flatten()

            res["x"] = [x]
            res["y"] = [y]

            res_list.append(res.copy())

    new_recon_signals = pd.DataFrame(res_list)

    return new_recon_signals
