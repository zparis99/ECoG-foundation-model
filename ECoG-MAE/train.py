import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
import time as t
import os
import torch
import torch.nn as nn
from einops import rearrange
from tqdm import tqdm

from mask import *
from utils import *
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

    (
        lrs,
        train_losses,
        test_losses,
        seen_train_losses,
        seen_test_losses,
        contrastive_losses,
    ) = ([], [], [], [], [], [])

    signal_means, signal_stds = [], []
    train_corr = pd.DataFrame()
    test_corr = pd.DataFrame()
    seen_corr = pd.DataFrame()
    unseen_corr = pd.DataFrame()
    model_recon = pd.DataFrame()
    trains = []
    tests = []

    progress_bar = tqdm(
        range(epoch, args.num_epochs), ncols=1200, disable=(local_rank != 0)
    )
    for epoch in progress_bar:
        start = t.time()
        with torch.cuda.amp.autocast(dtype=data_type):
            model.train()
            for train_i, batch in enumerate(train_dl):
                optimizer.zero_grad()

                signal = batch.to(device)

                trains.append(
                    {
                        "train batch ": str(train_i),
                        "num_samples": str(signal.shape),
                    }
                )

                signal_means.append(torch.mean(signal).detach().to("cpu").item())
                signal_stds.append(torch.std(signal).detach().to("cpu").item())

                if args.norm == "batch":
                    signal = normalize(signal)
                else:
                    signal = torch.where(
                        signal == 0, torch.tensor(float("nan")), signal
                    )

                # mask indicating positions of channels that were rejected during preprocessing
                padding_mask = get_padding_mask(signal, model, device)

                # convert nans to 0
                signal = torch.nan_to_num(signal)

                # masking out parts of the input to the encoder (same mask across frames)
                tube_mask = get_tube_mask(
                    args, num_patches, num_frames, padding_mask, device
                )

                # selecting parts of the signal for the decoder to reconstruct
                if args.decoder_mask_ratio == 0:
                    decoder_mask = get_decoder_mask(
                        args, num_patches, tube_mask, device
                    )
                else:
                    if args.running_cell_masking:
                        decoder_mask = get_running_cell_mask(
                            args, num_frames, tube_mask, device
                        )

                # make sure the decoder only reconstructs channels that were not discarded during preprocessing
                decoder_padding_mask = padding_mask[decoder_mask]

                # encode the tube patches
                encoder_out = model(
                    signal, encoder_mask=tube_mask, tube_padding_mask=padding_mask
                )

                # decode both the encoder_out patches and masked decoder patches
                decoder_out = model(
                    encoder_out,
                    encoder_mask=tube_mask,
                    decoder_mask=decoder_mask,
                    decoder_padding_mask=decoder_padding_mask,
                )

                # rearrange reconstructed and original patches (seen and not seen by encoder) into signal
                (
                    recon_signal,
                    recon_output,
                    recon_target,
                    seen_output,
                    seen_target,
                    seen_target_signal,
                    seen_output_signal,
                    recon_target_signal,
                    recon_output_signal,
                ) = rearrange_signals(
                    args,
                    model,
                    device,
                    signal,
                    num_frames,
                    decoder_out,
                    padding_mask,
                    tube_mask,
                    decoder_mask,
                    decoder_padding_mask,
                )

                # get correlation between original and reconstructed signal
                new_train_corr = get_correlation(
                    args, signal, recon_signal, epoch, train_i
                )
                train_corr = pd.concat([train_corr, new_train_corr])

                # get correlation between original and reconstructed signal seen by the encoder
                new_seen_corr = get_correlation_across_elecs(
                    args, seen_target_signal, seen_output_signal, epoch, train_i
                )

                seen_corr = pd.concat([seen_corr, new_seen_corr])

                # get correlation between original and reconstructed signal not seen by the encoder
                new_unseen_corr = get_correlation_across_elecs(
                    args, recon_target_signal, recon_output_signal, epoch, train_i
                )

                unseen_corr = pd.concat([unseen_corr, new_unseen_corr])

                # calculate loss
                if args.loss == "patch":
                    loss = mse(recon_output, recon_target)
                    seen_loss = mse(seen_output, seen_target)
                elif args.loss == "signal":
                    loss = mse(recon_output_signal, recon_target_signal)
                    seen_loss = mse(seen_output_signal, seen_target_signal)
                elif args.loss == "both":
                    loss = mse(recon_output, recon_target) + mse(
                        recon_output_signal, recon_target_signal
                    )
                    seen_loss = mse(seen_output, seen_target) + mse(
                        seen_output_signal, seen_target_signal
                    )

                accelerator.backward(loss)
                optimizer.step()

                train_losses.append(loss.item())
                seen_train_losses.append(seen_loss.item())
                lrs.append(optimizer.param_groups[0]["lr"])

            model.eval()
            for test_i, batch in enumerate(test_dl):

                signal = batch.to(device)

                tests.append(
                    {
                        "test batch ": str(test_i),
                        "num_samples": str(signal.shape),
                    }
                )

                signal_means.append(torch.mean(signal).detach().to("cpu").item())
                signal_stds.append(torch.std(signal).detach().to("cpu").item())

                if args.norm == "batch":
                    signal = normalize(signal)

                # mask indicating positions of channels that were rejected during preprocessing
                padding_mask = get_padding_mask(signal, model, device)

                # convert nans to 0
                signal = torch.nan_to_num(signal)

                # masking out parts of the input to the encoder (same mask across frames)
                tube_mask = get_tube_mask(
                    args, num_patches, num_frames, padding_mask, device
                )

                # selecting parts of the signal for the decoder to reconstruct
                if args.decoder_mask_ratio == 0:
                    decoder_mask = get_decoder_mask(
                        args, num_patches, tube_mask, device
                    )
                else:
                    if args.running_cell_masking:
                        decoder_mask = get_running_cell_mask(
                            args, num_frames, tube_mask, device
                        )

                # make sure the decoder only reconstructs channels that were not discarded during preprocessing
                decoder_padding_mask = padding_mask[decoder_mask]

                # encode the tube patches
                encoder_out = model(
                    signal, encoder_mask=tube_mask, tube_padding_mask=padding_mask
                )

                # decode both the encoder_out patches and masked decoder patches
                decoder_out = model(
                    encoder_out,
                    encoder_mask=tube_mask,
                    decoder_mask=decoder_mask,
                    decoder_padding_mask=decoder_padding_mask,
                )

                # rearrange reconstructed and original patches (seen and not seen by encoder) into signal
                (
                    recon_signal,
                    recon_output,
                    recon_target,
                    seen_output,
                    seen_target,
                    seen_target_signal,
                    seen_output_signal,
                    recon_target_signal,
                    recon_output_signal,
                ) = rearrange_signals(
                    args,
                    model,
                    device,
                    signal,
                    num_frames,
                    decoder_out,
                    padding_mask,
                    tube_mask,
                    decoder_mask,
                    decoder_padding_mask,
                )

                new_test_corr = get_correlation(
                    args, signal, recon_signal, epoch, test_i
                )
                test_corr = pd.concat([test_corr, new_test_corr])

                # calculate loss
                if args.loss == "patch":
                    loss = mse(recon_output, recon_target)
                    seen_loss = mse(seen_output, seen_target)
                elif args.loss == "signal":
                    loss = mse(recon_output_signal, recon_target_signal)
                    seen_loss = mse(seen_output_signal, seen_target_signal)
                elif args.loss == "both":
                    loss = mse(recon_output, recon_target) + mse(
                        recon_output_signal, recon_target_signal
                    )
                    seen_loss = mse(seen_output, seen_target) + mse(
                        seen_output_signal, seen_target_signal
                    )

                test_losses.append(loss.item())
                seen_test_losses.append(seen_loss.item())

                # save original and reconstructed signal for plotting (highgamma for one sample for now)
                if test_i == 0:

                    if args.norm == "batch":
                        recon_signal = normalize(recon_signal)
                    else:
                        recon_signal = recon_signal

                    new_model_recon = get_model_recon(args, signal, recon_signal, epoch)
                    model_recon = pd.concat([model_recon, new_model_recon])

                    dir = os.getcwd() + f"/results/recon_signals/"
                    if not os.path.exists(dir):
                        os.makedirs(dir)

                    model_recon.to_pickle(dir + f"{args.job_name}_recon_signal.txt")

            end = t.time()

            print("Epoch " + str(epoch) + " done. Time elapsed: " + str(end - start))

            logs = {
                "train/loss": np.mean(train_losses[-(train_i + 1) :]),
                "test/loss": np.mean(test_losses[-(test_i + 1) :]),
            }
            progress_bar.set_postfix(**logs)

        # save model checkpoints
        checkpoint = {
            "epoch": epoch,
            "model": model,
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "train_losses": train_losses,
            "seen_train_losses": seen_train_losses,
            "test_losses": test_losses,
            "seen_test_losses": seen_test_losses,
        }

        dir = os.getcwd() + f"/checkpoints/"
        if not os.path.exists(dir):
            os.makedirs(dir)

        torch.save(checkpoint, dir + f"{args.job_name}_checkpoint.pth")

        # save correlations
        dir = os.getcwd() + f"/results/correlation/"
        if not os.path.exists(dir):
            os.makedirs(dir)

        train_corr.to_csv(
            dir + f"{args.job_name}_train_corr.csv",
            index=False,
        )

        test_corr.to_csv(
            dir + f"{args.job_name}_test_corr.csv",
            index=False,
        )

        seen_corr.to_csv(
            dir + f"{args.job_name}_seen_corr.csv",
            index=False,
        )

        unseen_corr.to_csv(
            dir + f"{args.job_name}_unseen_corr.csv",
            index=False,
        )

        # save train and test sample information
        train_samples = pd.DataFrame(trains)
        test_samples = pd.DataFrame(tests)

        dir = os.getcwd() + f"/results/test_loader/"
        if not os.path.exists(dir):
            os.makedirs(dir)

        train_samples.to_csv(
            dir + f"{args.job_name}_train_samples.csv",
            index=False,
        )

        test_samples.to_csv(
            dir + f"{args.job_name}_test_samples.csv",
            index=False,
        )

        # plot results
        # Just closing plots isn't enough to free up the RAM used by matplotlib. Instead what we need to do to return that memory
        # is to plot in a subprocess which will be killed at completion. When that process is killed
        # we actually get the memory back.
        # See https://stackoverflow.com/questions/28516828/memory-leaks-using-matplotlib?rq=4 for more details.
        def _plot_summary(
            args,
            train_losses,
            test_losses,
            use_contrastive_loss,
            contrastive_losses,
            signal_means,
            signal_stds,
            test_corr,
            train_corr,
            model_recon,
        ):
            plot_losses(
                args, train_losses, seen_train_losses, test_losses, seen_test_losses
            )
            if use_contrastive_loss:
                plot_contrastive_loss(args, contrastive_losses)

            plot_signal_stats(args, signal_means, signal_stds)

            plot_correlation(args, train_corr, "train_correlation")
            plot_correlation(args, test_corr, "test_correlation")
            plot_correlation(args, seen_corr, "seen_correlation")
            plot_correlation(args, unseen_corr, "unseen_correlation")
            plot_recon_signals(args, model_recon)

            plt.close("all")

        proc = mp.Process(
            target=_plot_summary,
            args=(
                args,
                train_losses,
                test_losses,
                use_contrastive_loss,
                contrastive_losses,
                signal_means,
                signal_stds,
                test_corr,
                train_corr,
                model_recon,
            ),
        )
        proc.start()
        # wait until proc terminates.
        proc.join()

    return model
