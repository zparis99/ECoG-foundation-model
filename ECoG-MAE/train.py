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

    if args.learning_rate == None:
        model, optimizer, train_dl, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dl, lr_scheduler
        )

    mse = nn.MSELoss()
    if use_contrastive_loss:
        logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    lrs, recon_losses, test_losses, contrastive_losses = [], [], [], []
    test_corr = pd.DataFrame()
    seen_corr = pd.DataFrame()
    unseen_corr = pd.DataFrame()
    model_recon = pd.DataFrame()

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

                if args.norm == "batch":
                    # z-score normalization across batches
                    mean = torch.mean(raw_signal, dim=(0, 2), keepdim=True)
                    std = torch.std(raw_signal, dim=(0, 2), keepdim=True)
                    signal = (raw_signal - mean) / std
                else:
                    signal = raw_signal

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

                if args.norm:
                    # z-score normalization across batches
                    mean = torch.mean(raw_signal, dim=(0, 2), keepdim=True)
                    std = torch.std(raw_signal, dim=(0, 2), keepdim=True)
                    signal = (raw_signal - mean) / std
                else:
                    signal = raw_signal

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

                recon_output = decoder_out[:, num_encoder_patches:]
                seen_output = decoder_out[:, :num_encoder_patches]

                # compare to ground truth and calculate loss
                target_patches = model.patchify(signal)
                target_patches_vit = rearrange(target_patches, "b ... d -> b (...) d")
                recon_target = target_patches_vit[:, decoder_mask]
                seen_target = target_patches_vit[:, ~decoder_mask]
                loss = mse(recon_output, recon_target)
                test_losses.append(loss.item())

                # implement contrastive loss #TODO
                # implement correlation loss #TODO

                # reorganize into full signal
                recon_patches = torch.zeros(target_patches_vit.shape).to(device)

                tube_idx = torch.nonzero(tube_mask).squeeze()
                decoder_idx = torch.nonzero(decoder_mask).squeeze()

                breakpoint()

                recon_patches[:, tube_idx, :] = seen_output
                recon_patches[:, decoder_idx, :] = recon_output

                recon_signal = np.array(
                    model.unpatchify(recon_patches).detach().to("cpu")
                )
                signal = np.array(signal.detach().to("cpu"))

                new_test_corr = get_correlation(
                    args, signal, recon_signal, epoch, test_i
                )
                test_corr = pd.concat([test_corr, new_test_corr])

                dir = os.getcwd() + f"/results/correlation/"
                if not os.path.exists(dir):
                    os.makedirs(dir)

                test_corr.to_csv(
                    dir + f"{args.job_name}_test_corr.csv",
                    index=False,
                )

                # compare parts of the signal that were seen by the encoder
                seen_output_signal = np.array(
                    rearrange(
                        seen_output,
                        "b (f d s) (pd ps pf c) -> b c (f pf) (d pd s ps)",
                        c=len(args.bands),
                        d=1,
                        s=4,
                        f=num_frames // args.frame_patch_size,
                        pd=1,
                        ps=4,
                        pf=args.frame_patch_size,
                    )
                    .cpu()
                    .detach()
                )

                seen_target_signal = np.array(
                    rearrange(
                        seen_target,
                        "b (f d s) (pd ps pf c) -> b c (f pf) (d pd s ps)",
                        c=len(args.bands),
                        d=1,
                        s=4,
                        f=num_frames // args.frame_patch_size,
                        pd=1,
                        ps=4,
                        pf=args.frame_patch_size,
                    )
                    .cpu()
                    .detach()
                )

                new_seen_corr = get_correlation_across_elecs(
                    args, seen_output_signal, seen_target_signal, epoch, test_i
                )

                seen_corr = pd.concat([seen_corr, new_seen_corr])

                dir = os.getcwd() + f"/results/correlation/"
                if not os.path.exists(dir):
                    os.makedirs(dir)

                seen_corr.to_csv(
                    dir + f"{args.job_name}_seen_corr.csv",
                    index=False,
                )

                # compare parts of the signal that were not seen
                recon_output_signal = np.array(
                    rearrange(
                        recon_output,
                        "b (f d s) (pd ps pf c) -> b c (f pf) (d pd s ps)",
                        c=len(args.bands),
                        d=1,
                        s=12,
                        f=num_frames // args.frame_patch_size,
                        pd=1,
                        ps=4,
                        pf=args.frame_patch_size,
                    )
                    .cpu()
                    .detach()
                )

                recon_target_signal = np.array(
                    rearrange(
                        recon_target,
                        "b (f d s) (pd ps pf c) -> b c (f pf) (d pd s ps)",
                        c=len(args.bands),
                        d=1,
                        s=12,
                        f=num_frames // args.frame_patch_size,
                        pd=1,
                        ps=4,
                        pf=args.frame_patch_size,
                    )
                    .cpu()
                    .detach()
                )

                new_unseen_corr = get_correlation_across_elecs(
                    args, recon_output_signal, recon_target_signal, epoch, test_i
                )

                unseen_corr = pd.concat([unseen_corr, new_unseen_corr])

                dir = os.getcwd() + f"/results/correlation/"
                if not os.path.exists(dir):
                    os.makedirs(dir)

                unseen_corr.to_csv(
                    dir + f"{args.job_name}_unseen_corr.csv",
                    index=False,
                )

                # save original and reconstructed signal for plotting (highgamma for one sample for now)
                if test_i == 0:

                    new_model_recon = get_model_recon(signal, recon_signal, epoch)
                    model_recon = pd.concat([model_recon, new_model_recon])

                    dir = os.getcwd() + f"/results/recon_signals/"
                    if not os.path.exists(dir):
                        os.makedirs(dir)

                    model_recon.to_pickle(dir + f"{args.job_name}_recon_signal.txt")

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

        plot_correlation(args, test_corr, "correlation")
        plot_correlation(args, seen_corr, "seen_correlation")
        plot_correlation(args, unseen_corr, "unseen_correlation")
        plot_recon_signals(args, model_recon)

    return model
