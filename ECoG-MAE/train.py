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
    signal_means, signal_stds = [], []
    train_corr = pd.DataFrame()
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

                raw_signal = batch["signal"].to(device)

                if args.norm == "batch":
                    # z-score normalization across batches
                    mean = torch.mean(raw_signal, dim=(0, 2), keepdim=True)
                    std = torch.std(raw_signal, dim=(0, 2), keepdim=True)
                    signal = (raw_signal - mean) / std
                else:
                    signal = raw_signal

                padding_mask = ~torch.isnan(signal).to(device)
                padding_mask = rearrange(
                    model.patchify(padding_mask), "b ... d -> b (...) d"
                )

                padding_mask = torch.all(padding_mask, dim=0)
                padding_mask = torch.all(padding_mask, dim=1)

                # plot_signal(
                #     args,
                #     np.array(raw_signal[0, 4, :, 0, 0, 0].detach().to("cpu")),
                #     "raw",
                # )
                # plot_signal(
                #     args, np.array(signal[0, 4, :, 0, 0, 0].detach().to("cpu")), "norm"
                # )

                signal_means.append(torch.mean(signal).detach().to("cpu").item())
                signal_stds.append(torch.std(signal).detach().to("cpu").item())

                tube_mask = get_tube_mask(args, num_patches, num_frames, device)

                tube_padding_mask = padding_mask[tube_mask]

                if args.decoder_mask_ratio == 0:

                    decoder_mask = get_decoder_mask(
                        args, num_patches, tube_mask, device
                    )
                else:
                    if args.running_cell_masking:
                        decoder_mask = get_running_cell_mask(
                            args, num_frames, tube_mask, device
                        )

                decoder_padding_mask = padding_mask[decoder_mask]

                # encode the tube patches
                encoder_out = model(
                    signal, encoder_mask=tube_mask, tube_padding_mask=tube_padding_mask
                )
                if use_cls_token:
                    enc_cls_token = encoder_out[:, :1, :]

                # decode both the encoder_out patches and masked decoder patches
                decoder_out = model(
                    encoder_out,
                    encoder_mask=tube_mask,
                    decoder_mask=decoder_mask,
                    tube_padding_mask=tube_padding_mask,
                    decoder_padding_mask=decoder_padding_mask,
                )

                # recon_output = decoder_out[:, num_encoder_patches:]
                recon_output = decoder_out[
                    :, len(tube_padding_mask[tube_padding_mask == True]) :
                ]

                seen_output = decoder_out[
                    :, : len(tube_padding_mask[tube_padding_mask] == True)
                ]

                # compare to ground truth and calculate loss
                target_patches = model.patchify(signal)
                target_patches_vit = rearrange(target_patches, "b ... d -> b (...) d")
                recon_target = target_patches_vit[:, decoder_mask][
                    :, decoder_padding_mask
                ]
                seen_target = target_patches_vit[:, ~decoder_mask][:, tube_padding_mask]
                loss = mse(recon_output, recon_target)

                # implement contrastive loss #TODO
                # implement correlation loss #TODO

                accelerator.backward(loss)
                optimizer.step()
                recon_losses.append(loss.item())
                lrs.append(optimizer.param_groups[0]["lr"])

                # reorganize into full signal
                recon_patches = (
                    torch.zeros(target_patches_vit.shape).fill_(float("nan")).to(device)
                )

                tube_idx = torch.nonzero(tube_mask & padding_mask).squeeze()
                decoder_idx = torch.nonzero(decoder_mask & padding_mask).squeeze()

                recon_patches[:, tube_idx, :] = seen_output
                recon_patches[:, decoder_idx, :] = recon_output

                recon_signal = np.array(
                    model.unpatchify(recon_patches).detach().to("cpu")
                )
                signal = np.array(signal.detach().to("cpu"))

                new_train_corr = get_correlation(
                    args, signal, recon_signal, epoch, train_i
                )
                train_corr = pd.concat([train_corr, new_train_corr])

            model.eval()
            for test_i, batch in enumerate(test_dl):

                raw_signal = batch["signal"].to(device)

                if args.norm:
                    # z-score normalization across batches
                    mean = torch.mean(raw_signal, dim=(0, 2), keepdim=True)
                    std = torch.std(raw_signal, dim=(0, 2), keepdim=True)
                    signal = (raw_signal - mean) / std
                else:
                    signal = raw_signal

                padding_mask = ~torch.isnan(signal).to(device)
                padding_mask = rearrange(
                    model.patchify(padding_mask), "b ... d -> b (...) d"
                )

                padding_mask = torch.all(padding_mask, dim=0)
                padding_mask = torch.all(padding_mask, dim=1)

                signal_means.append(torch.mean(signal).detach().to("cpu").item())
                signal_stds.append(torch.std(signal).detach().to("cpu").item())

                tube_mask = get_tube_mask(args, num_patches, num_frames, device)

                tube_padding_mask = padding_mask[tube_mask]

                if args.decoder_mask_ratio == 0:

                    decoder_mask = get_decoder_mask(
                        args, num_patches, tube_mask, device
                    )
                else:
                    if args.running_cell_masking:
                        decoder_mask = get_running_cell_mask(
                            args, num_frames, tube_mask, device
                        )

                decoder_padding_mask = padding_mask[decoder_mask]

                # encode the tube patches
                encoder_out = model(
                    signal, encoder_mask=tube_mask, tube_padding_mask=tube_padding_mask
                )
                if use_cls_token:
                    enc_cls_token = encoder_out[:, :1, :]

                # decode both the encoder_out patches and masked decoder patches
                decoder_out = model(
                    encoder_out,
                    encoder_mask=tube_mask,
                    decoder_mask=decoder_mask,
                    tube_padding_mask=tube_padding_mask,
                    decoder_padding_mask=decoder_padding_mask,
                )

                # recon_output = decoder_out[:, num_encoder_patches:]
                recon_output = decoder_out[
                    :, len(tube_padding_mask[tube_padding_mask == True]) :
                ]

                seen_output = decoder_out[
                    :, : len(tube_padding_mask[tube_padding_mask] == True)
                ]

                # compare to ground truth and calculate loss
                target_patches = model.patchify(signal)
                target_patches_vit = rearrange(target_patches, "b ... d -> b (...) d")
                recon_target = target_patches_vit[:, decoder_mask][
                    :, decoder_padding_mask
                ]
                seen_target = target_patches_vit[:, ~decoder_mask][:, tube_padding_mask]
                loss = mse(recon_output, recon_target)

                test_losses.append(loss.item())

                # implement contrastive loss #TODO
                # implement correlation loss #TODO

                # reorganize into full signal
                recon_patches = (
                    torch.zeros(target_patches_vit.shape).fill_(float("nan")).to(device)
                )

                tube_idx = torch.nonzero(tube_mask & padding_mask).squeeze()
                decoder_idx = torch.nonzero(decoder_mask & padding_mask).squeeze()

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

                # # write as separate function in different file #TODO
                # # compare parts of the signal that were seen by the encoder
                # seen_output_signal = np.array(
                #     rearrange(
                #         seen_output,
                #         "b (f d s) (pd ps pf c) -> b c (f pf) (d pd s ps)",
                #         c=len(args.bands),
                #         d=1,
                #         s=4,
                #         f=num_frames // args.frame_patch_size,
                #         pd=1,
                #         ps=4,
                #         pf=args.frame_patch_size,
                #     )
                #     .cpu()
                #     .detach()
                # )

                # seen_target_signal = np.array(
                #     rearrange(
                #         seen_target,
                #         "b (f d s) (pd ps pf c) -> b c (f pf) (d pd s ps)",
                #         c=len(args.bands),
                #         d=1,
                #         s=4,
                #         f=num_frames // args.frame_patch_size,
                #         pd=1,
                #         ps=4,
                #         pf=args.frame_patch_size,
                #     )
                #     .cpu()
                #     .detach()
                # )

                # new_seen_corr = get_correlation_across_elecs(
                #     args, seen_output_signal, seen_target_signal, epoch, test_i
                # )

                # seen_corr = pd.concat([seen_corr, new_seen_corr])

                # # write as separate function in different file #TODO
                # # compare parts of the signal that were not seen
                # recon_output_signal = np.array(
                #     rearrange(
                #         recon_output,
                #         "b (f d s) (pd ps pf c) -> b c (f pf) (d pd s ps)",
                #         c=len(args.bands),
                #         d=1,
                #         s=12,
                #         f=num_frames // args.frame_patch_size,
                #         pd=1,
                #         ps=4,
                #         pf=args.frame_patch_size,
                #     )
                #     .cpu()
                #     .detach()
                # )

                # recon_target_signal = np.array(
                #     rearrange(
                #         recon_target,
                #         "b (f d s) (pd ps pf c) -> b c (f pf) (d pd s ps)",
                #         c=len(args.bands),
                #         d=1,
                #         s=12,
                #         f=num_frames // args.frame_patch_size,
                #         pd=1,
                #         ps=4,
                #         pf=args.frame_patch_size,
                #     )
                #     .cpu()
                #     .detach()
                # )

                # new_unseen_corr = get_correlation_across_elecs(
                #     args, recon_output_signal, recon_target_signal, epoch, test_i
                # )

                # unseen_corr = pd.concat([unseen_corr, new_unseen_corr])

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

        # save model checkpoints
        checkpoint = {
            "epoch": epoch,
            "model": model,
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
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

        # seen_corr.to_csv(
        #     dir + f"{args.job_name}_seen_corr.csv",
        #     index=False,
        # )

        # unseen_corr.to_csv(
        #     dir + f"{args.job_name}_unseen_corr.csv",
        #     index=False,
        # )

        # plot results
        # Just closing plots isn't enough to free up the RAM used by matplotlib. Instead what we need to do to return that memory 
        # is to plot in a subprocess which will be killed at completion. When that process is killed
        # we actually get the memory back.
        # See https://stackoverflow.com/questions/28516828/memory-leaks-using-matplotlib?rq=4 for more details.
        def _plot_summary(args, recon_losses, test_losses, use_contrastive_loss, contrastive_losses, signal_means, signal_stds, test_corr, train_corr, model_recon):
            plot_losses(args, recon_losses, test_losses)
            if use_contrastive_loss:
                plot_contrastive_loss(args, contrastive_losses)

            plot_signal_stats(args, signal_means, signal_stds)

            plot_correlation(args, test_corr, "correlation")
            plot_correlation(args, train_corr, "correlation")
            # plot_correlation(args, seen_corr, "seen_correlation")
            # plot_correlation(args, unseen_corr, "unseen_correlation")
            plot_recon_signals(args, model_recon)
            
            plt.close('all')
        
        proc = mp.Process(target=_plot_summary, args=(args, recon_losses, test_losses, use_contrastive_loss, contrastive_losses, signal_means, signal_stds, test_corr, train_corr, model_recon))
        proc.start()
        # wait until proc terminates.
        proc.join()

    return model
