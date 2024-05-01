# TODO implement loss, correlation (and other metrics) in a separate 'metrics' module, possibly also a separate 'plotting' module or notebook

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time as t
import os
import torch
import torch.nn as nn
from einops import rearrange
from tqdm import tqdm


def train_model(
    args,
    device,
    model,
    train_dl,
    test_dl,
    num_patches,
    optimizer,
    lr_scheduler,
    accelerator,
    data_type,
    local_rank,
):
    """
    Runs model training

    Args:
        args: input arguments
        device: the gpu to be used for model training
        model: an untrained model instance with randomly initialized parameters
        train_dl: dataloader instance for train split
        test_dl: dataloader instance for test split
        num_patches: number of patches in which the input data is segmented
        optimizer: Adam optimizer instance - https://www.analyticsvidhya.com/blog/2023/12/adam-optimizer/
        lr_scheduler: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
        accelerator: an accelerator instance - https://huggingface.co/docs/accelerate/en/index
        data_type: the data type to be used, we use "fp16" mixed precision - https://towardsdatascience.com/understanding-mixed-precision-training-4b246679c7c4
        local_rank: the local rank environment variable (only needed for multi-gpu training)

    Returns:
        model: model instance with updated parameters after training
    """

    ### class token config ###
    use_cls_token = args.use_cls_token

    ### Loss Config ###
    use_contrastive_loss = args.use_contrastive_loss
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

    model, optimizer, train_dl, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dl, lr_scheduler
    )

    mse = nn.MSELoss()
    if use_contrastive_loss:
        logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    lrs, recon_losses, contrastive_losses, test_losses = [], [], [], []
    train_corr = pd.DataFrame()
    test_corr = pd.DataFrame()
    recon_signals = pd.DataFrame()

    progress_bar = tqdm(
        range(epoch, args.num_epochs), ncols=1200, disable=(local_rank != 0)
    )
    for epoch in progress_bar:
        start = t.time()
        with torch.cuda.amp.autocast(dtype=data_type):
            model.train()
            for train_i, batch in enumerate(train_dl):
                optimizer.zero_grad()

                raw_signal = batch.to(device)

                # z-score normalization across batches
                mean = torch.mean(raw_signal, dim=(0, 2), keepdim=True)
                std = torch.std(raw_signal, dim=(0, 2), keepdim=True)
                signal = (raw_signal - mean) / std

                # replace all NaN's with zeros #HACK
                signal[torch.isnan(signal)] = 0

                tube_mask = (
                    torch.zeros(num_patches // num_frames).to(device).to(torch.bool)
                )
                mask_idx_candidates = torch.randperm(len(tube_mask))
                tube_idx = mask_idx_candidates[
                    : int(num_patches / num_frames * (1 - args.tube_mask_ratio))
                ]
                tube_mask[tube_idx] = True
                tube_mask = tube_mask.tile(num_frames)

                if args.decoder_mask_ratio == 0:
                    # create decoder mask similar to tube mask, but ensure no overlap
                    decoder_mask = torch.zeros(num_patches).to(device).to(torch.bool)
                    remaining_mask_idx = (~tube_mask).nonzero()
                    decoder_mask_idx = remaining_mask_idx[
                        : int(num_patches * (1 - args.decoder_mask_ratio))
                    ]
                    decoder_mask[decoder_mask_idx] = True
                else:
                    if args.running_cell_masking:
                        num_patch_per_cell = 4
                        num_mask_per_cell = int(
                            args.decoder_mask_ratio * num_patch_per_cell
                        )
                        stride = int(num_patch_per_cell / num_mask_per_cell)
                        num_patch_per_frame = 16  # change to be flexible #TODO
                        cell = torch.ones(
                            num_frames // args.frame_patch_size, num_patch_per_cell
                        )
                        # mask out patches in cell so that it spatially progresses across frames
                        # Quing et al., 2023, MAR: Masked Autoencoder for Efficient Action Recognition
                        for i in range(num_frames // args.frame_patch_size):
                            for j in range(num_mask_per_cell):
                                cell[i, (int(j * stride) + i) % num_patch_per_cell] = 0

                        decoder_mask = (
                            cell.repeat(
                                1, int(num_patch_per_frame / num_patch_per_cell)
                            )
                            .flatten(0)
                            .to(device)
                            .to(torch.bool)
                        )

                        # filter out patches that were seen by the encoder
                        decoder_mask[~tube_mask] == False

                # encode the tube patches
                encoder_out = model(signal, encoder_mask=tube_mask)
                if use_cls_token:
                    enc_cls_token = encoder_out[:, :1, :]

                # decode both the encoder_out patches and masked decoder patches
                decoder_out = model(
                    encoder_out, encoder_mask=tube_mask, decoder_mask=decoder_mask
                )

                output = decoder_out
                recon_output = decoder_out[:, num_encoder_patches:]

                # compare to ground truth and calculate loss
                target_patches = model.patchify(signal)
                target_patches_vit = rearrange(target_patches, "b ... d -> b (...) d")
                target = target_patches_vit[:, decoder_mask]
                loss = mse(recon_output, target)

                # implement contrastive loss #TODO

                if args.decoder_mask_ratio == 0:

                    signal = np.array(signal.cpu().detach())
                    output = np.array(model.unpatchify(output).cpu().detach())

                    # calculate correlation
                    res_list = []
                    i = 1
                    bands = ["theta", "alpha", "beta", "gamma", "highgamma"]

                    for h in range(0, 8):
                        for w in range(0, 8):
                            res = {}
                            res["epoch"] = epoch
                            res["train_i"] = train_i
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

                    new_train_corr = pd.DataFrame(res_list)
                    train_corr = pd.concat([train_corr, new_train_corr])

                    train_corr.to_csv(
                        os.getcwd() + f"/results/{args.job_name}_train_corr.csv",
                        index=False,
                    )

                    # save original and reconstructed signal for plotting (highgamma for one sample for now)
                    if train_i == 8:

                        res_list = []
                        i = 1

                        for h in range(0, 8):
                            for w in range(0, 8):

                                res = {}
                                res["epoch"] = epoch
                                res["elec"] = "G" + str(i)
                                i += 1

                                x = signal[50, 4, :, 0, h, w].flatten()
                                y = output[50, 4, :, 0, h, w].flatten()

                                res["x"] = [x]
                                res["y"] = [y]

                                res_list.append(res.copy())

                        new_recon_signals = pd.DataFrame(res_list)
                        recon_signals = pd.concat([recon_signals, new_recon_signals])

                        recon_signals.to_pickle(
                            os.getcwd() + f"/results/{args.job_name}_recon_signals.txt"
                        )

                else:

                    # make more flexible #TODO

                    signal = np.array(
                        rearrange(
                            target,
                            "b (f d s) (pd ps pf c) -> b c (f pf) (d pd s ps)",
                            c=len(args.bands),
                            d=1,
                            s=4,
                            f=num_frames,
                            pd=1,
                            ps=4,
                            pf=1,
                        )
                        .cpu()
                        .detach()
                    )

                    output = np.array(
                        rearrange(
                            recon_output,
                            "b (f d s) (pd ps pf c) -> b c (f pf) (d pd s ps)",
                            c=len(args.bands),
                            d=1,
                            s=4,
                            f=num_frames,
                            pd=1,
                            ps=4,
                            pf=1,
                        )
                        .cpu()
                        .detach()
                    )

                    # calculate correlation
                    res_list = []
                    i = 1
                    bands = ["theta", "alpha", "beta", "gamma", "highgamma"]

                    for c in range(0, len(args.bands)):

                        res = {}
                        res["epoch"] = epoch
                        res["train_i"] = train_i

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

                    new_train_corr = pd.DataFrame(res_list)
                    train_corr = pd.concat([train_corr, new_train_corr])

                    train_corr.to_csv(
                        os.getcwd() + f"/results/{args.job_name}_train_corr.csv",
                        index=False,
                    )

                accelerator.backward(loss)
                optimizer.step()
                recon_losses.append(loss.item())
                lrs.append(optimizer.param_groups[0]["lr"])

            model.eval()
            for test_i, batch in enumerate(test_dl):

                raw_signal = batch.to(device)

                # z-score normalization across batches
                mean = torch.mean(raw_signal, dim=(0, 2), keepdim=True)
                std = torch.std(raw_signal, dim=(0, 2), keepdim=True)
                signal = (raw_signal - mean) / std

                # replace all NaN's with zeros #HACK
                signal[torch.isnan(signal)] = 0
                tube_mask = (
                    torch.zeros(num_patches // num_frames).to(device).to(torch.bool)
                )
                mask_idx_candidates = torch.randperm(len(tube_mask))
                tube_idx = mask_idx_candidates[
                    : int(num_patches / num_frames * (1 - args.tube_mask_ratio))
                ]
                tube_mask[tube_idx] = True
                tube_mask = tube_mask.tile(num_frames)

                if args.decoder_mask_ratio == 0:
                    # create decoder mask similar to tube mask, but ensure no overlap
                    decoder_mask = torch.zeros(num_patches).to(device).to(torch.bool)
                    remaining_mask_idx = (~tube_mask).nonzero()
                    decoder_mask_idx = remaining_mask_idx[
                        : int(num_patches * (1 - args.decoder_mask_ratio))
                    ]
                    decoder_mask[decoder_mask_idx] = True
                else:
                    if args.running_cell_masking:
                        num_patch_per_cell = 4
                        num_mask_per_cell = int(
                            args.decoder_mask_ratio * num_patch_per_cell
                        )
                        stride = int(num_patch_per_cell / num_mask_per_cell)
                        num_patch_per_frame = 16  # change to be flexible #TODO
                        cell = torch.ones(
                            num_frames // args.frame_patch_size, num_patch_per_cell
                        )
                        # mask out patches in cell so that it spatially progresses across frames
                        # Quing et al., 2023, MAR: Masked Autoencoder for Efficient Action Recognition
                        for i in range(num_frames // args.frame_patch_size):
                            for j in range(num_mask_per_cell):
                                cell[i, (int(j * stride) + i) % num_patch_per_cell] = 0

                        decoder_mask = (
                            cell.repeat(
                                1, int(num_patch_per_frame / num_patch_per_cell)
                            )
                            .flatten(0)
                            .to(device)
                            .to(torch.bool)
                        )

                        # filter out patches that were seen by the encoder
                        decoder_mask[~tube_mask] == False

                # encode the tube patches
                encoder_out = model(signal, encoder_mask=tube_mask)
                if use_cls_token:
                    enc_cls_token = encoder_out[:, :1, :]

                # decode both the encoder_out patches and masked decoder patches
                decoder_out = model(
                    encoder_out, encoder_mask=tube_mask, decoder_mask=decoder_mask
                )

                output = decoder_out
                recon_output = decoder_out[:, num_encoder_patches:]

                # compare to ground truth and calculate loss
                target_patches = model.patchify(signal)
                target_patches_vit = rearrange(target_patches, "b ... d -> b (...) d")
                target = target_patches_vit[:, decoder_mask]
                loss = mse(recon_output, target)
                test_losses.append(loss.item())

                # implement contrastive loss #TODO

                # signal = np.array(signal.cpu().detach())
                # output = np.array(model.unpatchify(output).cpu().detach())

                # # calculate correlation
                # res = {}
                # res_list = []
                # i = 1
                # bands = ["theta", "alpha", "beta", "gamma", "highgamma"]

                # for h in range(0, 8):
                #     for w in range(0, 8):
                #         res["epoch"] = epoch
                #         res["test_i"] = test_i
                #         res["elec"] = "G" + str(i)
                #         i += 1
                #         for c in range(0, len(args.bands)):
                #             # average across samples in batch
                #             corrs = []
                #             for b in range(0, len(signal[:, 0, 0, 0, 0, 0])):
                #                 x = signal[b, c, :, :, h, w].flatten()
                #                 y = output[b, c, :, :, h, w].flatten()
                #                 # add check to make sure x and y are the same length #TODO
                #                 n = len(x)
                #                 # this is for the channels that we zero padded, to avoid division by 0
                #                 # we could also just exclude those channels
                #                 if np.sum(x) == 0:
                #                     r = 0
                #                 else:
                #                     r = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (
                #                         np.sqrt(
                #                             (n * np.sum(x**2) - (np.sum(x)) ** 2)
                #                             * (n * np.sum(y**2) - (np.sum(y)) ** 2)
                #                         )
                #                     )
                #                 corrs.append(r)
                #             res["band"] = bands[c]
                #             res["corr"] = np.mean(corrs)
                #             res_list.append(res.copy())

                # new_test_corr = pd.DataFrame(res_list)
                # test_corr = pd.concat([test_corr, new_test_corr])

                # test_corr.to_csv(
                #     os.getcwd() + f"/results/{args.job_name}_test_corr.csv",
                #     index=False,
                # )

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
        plt.savefig(os.getcwd() + f"/results/{args.job_name}_training_loss.png")

        if use_contrastive_loss:
            plt.figure(figsize=(8, 3))
            plt.plot(contrastive_losses)
            plt.title("Training contrastive losses")
            # plt.show()

        plt.figure(figsize=(8, 3))
        plt.plot(test_losses)
        plt.title("Test losses")
        # plt.show()

        plt.savefig(os.getcwd() + f"/results/{args.job_name}_test_loss.png")

    return model
