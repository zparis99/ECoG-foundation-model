#!/usr/bin/env python
# coding: utf-8

import argparse
import ast
import os
import random
import sys
import time
import time as t

import matplotlib.pyplot as plt
import mne
import nibabel as nib
import numpy as np
import pandas as pd
import scipy.signal
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchio as tio
import webdataset as wds
from accelerate import Accelerator, DeepSpeedPlugin
from einops import rearrange
from einops.layers.torch import Rearrange
from mne_bids import BIDSPath, read_raw_bids
from models import *
from pyedflib import highlevel
from scipy.signal import resample
from torchvision import transforms
from tqdm import tqdm

import utils

# tf32 data type is faster than standard float32
torch.backends.cuda.matmul.allow_tf32 = True

### Specify parameters for model training and get system params ###


# Custom type for parsing list of integers
def parse_list_of_ints(arg):
    try:
        # Split the input string by commas and convert each element to an integer
        ints = [int(x) for x in arg.split(",")]
        return ints
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Invalid format. Please provide a list of integers separated by commas."
        )


# Custom type for parsing list of lists
def parse_list_of_lists(arg):
    try:
        list_of_lists = [list(map(int, sublist.split())) for sublist in sys.argv[1:]]
        return list_of_lists
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Invalid format. Please provide a list of lists."
        )


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-name", type=str)
    parser.add_argument("--debug", type=bool)
    parser.add_argument("--data-size", type=float, default=1)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--new-fs", type=int)
    parser.add_argument("--sample-length", type=int)
    parser.add_argument("--patch-size", nargs="+", type=int)
    parser.add_argument("--frame-patch-size", type=int)
    parser.add_argument("--tube-mask-ratio", type=float)
    parser.add_argument("--decoder-mask-ratio", type=float)
    parser.add_argument("--bands", type=str)
    parser.add_argument("--num-epochs", type=int)
    parser.add_argument("--use-contrastive-loss", type=bool, default=False)
    parser.add_argument("--use-cls-token", type=bool, default=False)
    args = parser.parse_args()

    # parse string input to list of lists
    args.bands = ast.literal_eval(args.bands)

    return args


def system_setup():
    accelerator = Accelerator(split_batches=False, mixed_precision="fp16")
    device = "cuda:0"

    # set data_type to match your mixed precision
    if accelerator.mixed_precision == "bf16":
        data_type = torch.bfloat16
    elif accelerator.mixed_precision == "fp16":
        data_type = torch.float16
    else:
        data_type = torch.float32

    local_rank = os.getenv("RANK")
    if local_rank is None:
        local_rank = 0
    else:
        local_rank = int(local_rank)

    return accelerator, device, data_type, local_rank


def split_dataframe(df, ratio1, ratio2):
    """
    Shuffles a pandas dataframe and splits it into two dataframes with the specified ratios.

    Args:
        df: The dataframe to split.
        ratio1: The proportion of data for the first dataframe (default: 0.9).
        ratio2: The proportion of data for the second dataframe (default: 0.1).

    Returns:
        A tuple of two dataframes, the first containing ratio1 proportion of the data and the second containing ratio2 proportion.
    """

    # Shuffle the dataframe
    df = df.sample(frac=1).reset_index(drop=True)

    # Calculate the split index based on the ratios
    split_index = int(ratio1 * len(df))

    # Create the two dataframes
    df1 = df.iloc[:split_index, :]
    df2 = df.iloc[split_index:, :]

    return df1, df2


def read_raw(filename, bids=False):
    # TODO check why we do not bids here
    if not bids:
        raw = mne.io.read_raw(filename, verbose=False)
    else:
        basename, sub, datatype, filename = filename.split("/")[-4:]
        path = BIDSPath(
            subject=sub.split("-")[1],
            task="conversation",  # TODO: fix
            suffix=datatype,
            extension=".edf",
            root=basename,
        )
        raw = read_raw_bids(path)

    # TODO check whether DC are ALWAYS stim
    types = {}
    for name in raw.ch_names:
        if "DC" in name:
            types[name] = "stim"
        else:
            types[name] = "ecog"

    raw.set_channel_types(types)

    return raw


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

        # padding
        if len(norm_sig[0]) < n_samples:  # apply padding if chunk is shorter than 2 sec
            padding = np.zeros((64, n_samples - len(norm_sig[0])))
            norm_sig = np.concatenate((norm_sig, padding), axis=1)

        # check if channel is included and if not pad to zero
        for i in range(0, 64):
            chn = "G" + str(i + 1)

            if np.isin(chn, raw.info.ch_names) == False:
                # shift upwards
                norm_sig = np.insert(norm_sig, i, np.zeros((1, n_samples)), axis=0)

        # delete items that were shifted upwards
        norm_sig = norm_sig[:64, :]

        # filter
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


def dl_setup(args):
    root = "/scratch/gpfs/ln1144/fm-preproc/dataset/derivatives/preprocessed"
    data = pd.read_csv("/scratch/gpfs/ln1144/fm-preproc/dataset/dataset.csv")
    # only look at subset of data
    data = data.iloc[: int(len(data) * args.data_size), :]
    train_data, test_data = split_dataframe(data, 0.9, 0.1)
    bands = args.bands
    fs = 512
    new_fs = args.new_fs
    batch_size = args.batch_size

    train_datasets = []

    for i, row in train_data.iterrows():
        path = BIDSPath(
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
        path = BIDSPath(
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

    # # Test dataloader
    # if args.debug:

    #     start = t.time()

    #     train_samples = 0
    #     print('test train_dl')
    #     for train_i,signal in enumerate(train_dl):

    #         if train_i == 1:
    #             break

    #         print('train batch ' + str(train_i))
    #         print('signal ' + str(signal.shape))
    #         train_samples += len(signal)

    #     end = t.time()

    #     print('Dataloader tested with batch size ' + str(batch_size) + '. Time elapsed: ' + str(end-start))

    #     test_samples = 0
    #     print('\ntest test_dl')
    #     for test_i,signal in enumerate(test_dl):

    #         if test_i == 1:
    #             break

    #         test_samples += len(signal)
    #         print('test batch ' + str(test_i))
    #         print('signal ' + str(signal.shape))

    return train_dl, test_dl, 1000


def model_setup(args, device, num_train_samples):
    ### class token config ###
    use_cls_token = args.use_cls_token

    ### Loss Config ###
    # Are we planning on using contrastive loss?
    use_contrastive_loss = args.use_contrastive_loss
    constrastive_loss_weight = 1.0
    use_cls_token = (
        True if use_contrastive_loss else use_cls_token
    )  # if using contrastive loss, we need to add a class token

    input_size = [1, 8, 8]
    print("input_size", input_size)
    seed = 42
    num_frames = args.sample_length * args.new_fs
    tubelet_size = 1  # What is this?

    img_size = (1, 8, 8)
    patch_size = tuple(args.patch_size)
    frame_patch_size = args.frame_patch_size
    num_patches = int(  # Defining the number of patches
        (img_size[0] / patch_size[0])
        * (img_size[1] / patch_size[1])
        * (img_size[2] / patch_size[2])
        * num_frames
        / frame_patch_size
    )

    num_encoder_patches = int(num_patches * (1 - args.tube_mask_ratio))
    num_decoder_patches = int(num_patches * (1 - args.decoder_mask_ratio))
    print("num_patches", num_patches)
    print("num_encoder_patches", num_encoder_patches)
    print("num_decoder_patches", num_decoder_patches)

    max_lr = 3e-5  # 3e-5 seems to be working best? original videomae used 1.5e-4

    model = SimpleViT(
        image_size=img_size,  # depth, height, width
        image_patch_size=patch_size,  # depth, height, width patch size - change width from patch_size to 1
        frames=num_frames,
        frame_patch_size=frame_patch_size,
        depth=12,
        heads=12,
        dim=512,
        mlp_dim=512,  # TODO: right now dim needs to equal mlp_dim, and both need to be 512
        num_encoder_patches=num_encoder_patches,
        num_decoder_patches=num_decoder_patches,
        channels=len(args.bands),
        use_rope_emb=False,
        use_cls_token=False,
    )
    utils.count_params(model)

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    opt_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 1e-2,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(opt_grouped_parameters, lr=max_lr)

    num_iterations_per_epoch = (
        1000  # num_train_samples/args.batch_size # TODO change to actual duration
    )
    total_steps = args.num_epochs * num_iterations_per_epoch
    print("total_steps", total_steps)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        final_div_factor=1000,
        last_epoch=-1,
        pct_start=2 / args.num_epochs,
    )

    print("\nDone with model preparations!")

    # if args.debug:

    #     print('Testing model')

    #     # test that the model works without error
    #     model = model.to(device).eval()
    #     encoder_mask = torch.zeros(num_patches).to(device).to(torch.bool)
    #     encoder_mask[:num_encoder_patches] = True
    #     decoder_mask = torch.zeros(num_patches).to(device).to(torch.bool)
    #     decoder_mask[-num_decoder_patches:] = True

    #     with torch.no_grad():
    #         print("\nencoder")
    #         encoder_out = model(
    #             torch.randn(2, len(args.bands), 40, 1, 8, 8).to(device),
    #             encoder_mask=encoder_mask,
    #             verbose=True,
    #         )

    #         print("\ndecoder")
    #         decoder_out = model(
    #             encoder_out, encoder_mask=encoder_mask, decoder_mask=decoder_mask, verbose=True
    #         )

    #         if use_cls_token:
    #             enc_cls_token = encoder_out[:, :1, :]
    #             encoder_patches = encoder_out[:, 1:, :]
    #             dec_cls_token = decoder_out[:, :1, :]
    #             decoder_patches = decoder_out[:, 1:, :]
    #             print("enc_cls_token", enc_cls_token.shape)
    #             print("encoder_patches", encoder_patches.shape)
    #             print("dec_cls_token", dec_cls_token.shape)
    #             print("decoder_patches", decoder_patches.shape)

    return model, lr_scheduler, optimizer, num_patches


def train_model(
    args,
    model,
    train_dl,
    test_dl,
    num_patches,
    lr_scheduler,
    optimizer,
    accelerator,
    device,
    data_type,
    local_rank,
):
    ### class token config ###
    use_cls_token = True

    ### Loss Config ###
    # Are we planning on using contrastive loss?
    use_contrastive_loss = False
    constrastive_loss_weight = 1.0
    use_cls_token = (
        True if use_contrastive_loss else use_cls_token
    )  # if using contrastive loss, we need to add a class token

    torch.cuda.empty_cache()
    model.to(device)

    num_encoder_patches = int(num_patches * (1 - args.tube_mask_ratio))
    num_decoder_patches = int(num_patches * (1 - args.decoder_mask_ratio))

    num_frames = args.sample_length * args.new_fs

    epoch = 0
    losses, test_losses, lrs = [], [], []
    best_test_loss = 1e9
    torch.cuda.empty_cache()
    # model, optimizer, train_dl, lr_scheduler = accelerator.prepare(
    #     model, optimizer, train_dl, lr_scheduler)
    model, optimizer, train_dl = accelerator.prepare(model, optimizer, train_dl)

    mse = nn.MSELoss()
    if use_contrastive_loss:
        logit_scale = nn.Parameter(
            torch.ones([]) * np.log(1 / 0.07)
        )  # learned logit scale
    lrs, recon_losses, contrastive_losses, test_losses = [], [], [], []
    train_corr = pd.DataFrame()
    test_corr = pd.DataFrame()
    recon_image_list = []
    progress_bar = tqdm(
        range(epoch, args.num_epochs), ncols=1200, disable=(local_rank != 0)
    )
    for epoch in progress_bar:
        start = t.time()
        with torch.cuda.amp.autocast(dtype=data_type):
            model.train()
            for train_i, batch in enumerate(
                train_dl
            ):  # total samples in 1 epoch = train_dl.nsamples
                optimizer.zero_grad()

                signal = batch.to(device)

                tube_mask = (
                    torch.zeros(num_patches // num_frames).to(device).to(torch.bool)
                )
                mask_idx_candidates = torch.randperm(len(tube_mask))
                tube_idx = mask_idx_candidates[
                    : int(num_patches / num_frames * (1 - args.tube_mask_ratio))
                ]
                tube_mask[tube_idx] = True
                tube_mask = tube_mask.tile(num_frames)

                # create decoder mask similar to tube mask, but ensure no overlap
                decoder_mask = torch.zeros(num_patches).to(device).to(torch.bool)
                remaining_mask_idx = (~tube_mask).nonzero()

                # implement running cell masking here

                decoder_mask_idx = remaining_mask_idx[
                    : int(num_patches * (1 - args.decoder_mask_ratio))
                ]
                decoder_mask[decoder_mask_idx] = True

                # encode the tube patches
                encoder_out = model(signal, encoder_mask=tube_mask)
                if use_cls_token:
                    enc_cls_token = encoder_out[:, :1, :]

                # decode both the encoder_out patches and masked decoder patches
                decoder_out = model(
                    encoder_out, encoder_mask=tube_mask, decoder_mask=decoder_mask
                )

                if args.decoder_mask_ratio != 0:
                    # subset only the reconstructed decoder patches
                    output = decoder_out[:, -num_decoder_patches:]
                elif args.decoder_mask_ratio == 0:
                    output = decoder_out

                # compare to ground truth and calculate loss
                target_patches = model.patchify(signal)
                target_patches_vit = rearrange(target_patches, "b ... d -> b (...) d")
                target = target_patches_vit[:, decoder_mask]
                rec_output = output[:, decoder_mask]
                loss = mse(rec_output, target)

                # calculate correlation
                # signal = np.array(rearrange(signal,
                #         "b c f d h w -> c (b f) d h w").cpu().detach())
                # output = np.array(rearrange(model.unpatchify(output),
                #         "b c f d h w -> c (b f) d h w").cpu().detach())

                signal = np.array(signal.cpu().detach())
                output = np.array(model.unpatchify(output).cpu().detach())

                res = {}
                res_list = []
                i = 1
                bands = ["theta", "alpha", "beta", "gamma", "highgamma"]

                for h in range(0, 8):
                    for w in range(0, 8):
                        res["epoch"] = epoch
                        res["test_i"] = train_i
                        res["elec"] = "G" + str(i)
                        i += 1
                        for c in range(0, len(args.bands)):
                            # average across samples in batch
                            corrs = []
                            for b in range(0, len(signal[:, 0, 0, 0, 0, 0])):
                                corr = np.correlate(
                                    signal[b, c, :, :, h, w].flatten(),
                                    output[b, c, :, :, h, w].flatten(),
                                )[0]
                                corrs.append(corr)
                            res["band"] = bands[c]
                            res["corr"] = np.mean(corr)
                            res_list.append(res.copy())

                new_train_corr = pd.DataFrame(res_list)
                train_corr = pd.concat([train_corr, new_train_corr])

                train_corr.to_csv(
                    f"/scratch/gpfs/ln1144/ECoG-foundation-model/results/{args.job_name}_train_corr.csv",
                    index=False,
                )

                accelerator.backward(loss)
                optimizer.step()
                recon_losses.append(loss.item())
                lrs.append(optimizer.param_groups[0]["lr"])

            # print('\nmemory usage train after epoch ' + str(epoch) + ': ' + str(torch.cuda.memory_allocated(device)*1e-09))

            model.eval()
            for test_i, batch in enumerate(test_dl):
                signal = batch.to(device)

                tube_mask = (
                    torch.zeros(num_patches // num_frames).to(device).to(torch.bool)
                )
                mask_idx_candidates = torch.randperm(len(tube_mask))
                tube_idx = mask_idx_candidates[
                    : int(num_patches / num_frames * (1 - args.tube_mask_ratio))
                ]
                tube_mask[tube_idx] = True
                tube_mask = tube_mask.tile(num_frames)

                # create decoder mask similar to tube mask, but ensure no overlap
                decoder_mask = torch.zeros(num_patches).to(device).to(torch.bool)
                remaining_mask_idx = (~tube_mask).nonzero()

                # implement running cell masking here #TODO

                decoder_mask_idx = remaining_mask_idx[
                    : int(num_patches * (1 - args.decoder_mask_ratio))
                ]
                decoder_mask[decoder_mask_idx] = True

                # encode the tube patches
                encoder_out = model(signal, encoder_mask=tube_mask)
                if use_cls_token:
                    enc_cls_token = encoder_out[:, :1, :]

                # decode both the encoder_out patches and masked decoder patches
                decoder_out = model(
                    encoder_out, encoder_mask=tube_mask, decoder_mask=decoder_mask
                )

                if args.decoder_mask_ratio != 0:
                    # subset only the reconstructed decoder patches
                    output = decoder_out[:, -num_decoder_patches:]
                elif args.decoder_mask_ratio == 0:
                    output = decoder_out

                # compare to ground truth and calculate loss
                target_patches = model.patchify(signal)
                target_patches_vit = rearrange(target_patches, "b ... d -> b (...) d")
                target = target_patches_vit[:, decoder_mask]
                rec_output = output[:, decoder_mask]
                loss = mse(rec_output, target)
                test_losses.append(loss.item())

                # calculate correlation
                # signal = np.array(rearrange(signal,
                #         "b c f d h w -> c (b f) d h w").cpu().detach())
                # output = np.array(rearrange(model.unpatchify(output),
                #         "b c f d h w -> c (b f) d h w").cpu().detach())

                signal = np.array(signal.cpu().detach())
                output = np.array(model.unpatchify(output).cpu().detach())

                res = {}
                res_list = []
                i = 1
                bands = ["theta", "alpha", "beta", "gamma", "highgamma"]

                for h in range(0, 8):
                    for w in range(0, 8):
                        res["epoch"] = epoch
                        res["test_i"] = test_i
                        res["elec"] = "G" + str(i)
                        i += 1
                        for c in range(0, len(args.bands)):
                            # average across samples in batch
                            corrs = []
                            for b in range(0, len(signal[:, 0, 0, 0, 0, 0])):
                                corr = np.correlate(
                                    signal[b, c, :, :, h, w].flatten(),
                                    output[b, c, :, :, h, w].flatten(),
                                )[0]
                                corrs.append(corr)
                            res["band"] = bands[c]
                            res["corr"] = np.mean(corr)
                            res_list.append(res.copy())

                new_test_corr = pd.DataFrame(res_list)
                test_corr = pd.concat([test_corr, new_test_corr])

                test_corr.to_csv(
                    f"/scratch/gpfs/ln1144/ECoG-foundation-model/results/{args.job_name}_test_corr.csv",
                    index=False,
                )

            # print('\nmemory usage test after epoch ' + str(epoch) + ': ' + str(torch.cuda.memory_allocated(device)*1e-09))

            end = t.time()

            print("Epoch " + str(epoch) + " done. Time elapsed: " + str(end - start))

            logs = {
                "train/loss": np.mean(recon_losses[-(train_i + 1) :]),
                "test/loss": np.mean(test_losses[-(test_i + 1) :]),
            }
            progress_bar.set_postfix(**logs)

        plt.figure(figsize=(8, 3))
        plt.plot(recon_losses)
        plt.title("Training re-construction losses")
        # plt.show()
        plt.savefig(
            f"/scratch/gpfs/ln1144/ECoG-foundation-model/results/{args.job_name}_training_loss.png"
        )

        if use_contrastive_loss:
            plt.figure(figsize=(8, 3))
            plt.plot(contrastive_losses)
            plt.title("Training contrastive losses")
            # plt.show()

        plt.figure(figsize=(8, 3))
        plt.plot(test_losses)
        plt.title("Test losses")
        # plt.show()

        plt.savefig(
            f"/scratch/gpfs/ln1144/ECoG-foundation-model/results/{args.job_name}_test_loss.png"
        )

    return model


def main(args):
    accelerator, device, data_type, local_rank = system_setup()
    train_dl, test_dl, num_train_samples = dl_setup(args)
    model, lr_scheduler, optimizer, num_patches = model_setup(
        args, device, num_train_samples
    )
    model = train_model(
        args,
        model,
        train_dl,
        test_dl,
        num_patches,
        lr_scheduler,
        optimizer,
        accelerator,
        device,
        data_type,
        local_rank,
    )

    # save model ckpt
    torch.save({"model_state_dict": model.state_dict()}, "last.ckpt")


if __name__ == "__main__":
    args = arg_parser()
    main(args)
