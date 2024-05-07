import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time as t
import os
import torch
import torch.nn as nn
from einops import rearrange
from tqdm import tqdm

from mask import *
from metrics import *
from plot import *


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
    best_test_loss = 1e9

    model, optimizer, train_dl, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dl, lr_scheduler
    )

    mse = nn.MSELoss()
    if use_contrastive_loss:
        logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    lrs, recon_losses, test_losses, contrastive_losses = [], [], [], []
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

                tube_mask = get_tube_mask(args, num_patches, num_frames, device)

                if args.decoder_mask_ratio == 0:

                    decoder_mask = get_decoder_mask(
                        args, num_patches, tube_mask, device
                    )
                else:
                    if args.running_cell_masking:
                        decoder_mask = get_running_cell_mask(
                            args, num_frames, tube_mask, device
                        )

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
                # implement correlation loss #TODO

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

                tube_mask = get_tube_mask(args, num_patches, num_frames, device)

                if args.decoder_mask_ratio == 0:

                    decoder_mask = get_decoder_mask(
                        args, num_patches, tube_mask, device
                    )
                else:
                    if args.running_cell_masking:
                        decoder_mask = get_running_cell_mask(
                            args, num_patches, tube_mask, device
                        )

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
                # implement correlation loss #TODO

                signal = np.array(signal.cpu().detach())
                output = np.array(model.unpatchify(output).cpu().detach())

                if args.decoder_mask_ratio == 0:

                    new_test_corr = get_correlation(
                        args, test_corr, signal, output, epoch, test_i
                    )
                    test_corr = pd.concat([test_corr, new_test_corr])

                    dir = os.getcwd() + f"/results/correlation/"
                    if not os.path.exists(dir):
                        os.makedirs(dir)

                    test_corr.to_csv(
                        dir + f"{args.job_name}_test_corr.csv",
                        index=False,
                    )

                    # save original and reconstructed signal for plotting (highgamma for one sample for now)
                    if test_i == 8:

                        new_recon_signals = get_recon_signals(signal, output, epoch)
                        recon_signals = pd.concat([recon_signals, new_recon_signals])

                        dir = os.getcwd() + f"/results/recon_signals/"
                        if not os.path.exists(dir):
                            os.makedirs(dir)

                        recon_signals.to_pickle(
                            dir + f"{args.job_name}_recon_signals.txt"
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

                    new_test_corr = get_correlation_across_elecs(
                        args, signal, output, epoch, test_i
                    )
                    test_corr = pd.concat([test_corr, new_test_corr])

                    dir = os.getcwd() + f"/results/correlation/"
                    if not os.path.exists(dir):
                        os.makedirs(dir)

                    test_corr.to_csv(
                        dir + f"{args.job_name}_test_corr.csv",
                        index=False,
                    )

            end = t.time()

            print("Epoch " + str(epoch) + " done. Time elapsed: " + str(end - start))

            logs = {
                "train/loss": np.mean(recon_losses[-(train_i + 1) :]),
                "test/loss": np.mean(test_losses[-(test_i + 1) :]),
            }
            progress_bar.set_postfix(**logs)

        plot_losses(args, recon_losses, test_losses)
        if use_contrastive_loss:
            plot_contrastive_loss(args, contrastive_losses)

        if args.decoder_mask_ratio == 0:

            plot_correlation(args, test_corr)
            plot_recon_signals(args, recon_signals)
        else:
            plot_correlation_across_electrodes(args, test_corr)

    return model
