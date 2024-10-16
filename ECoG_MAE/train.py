import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
import time as t
import logging
import os
import torch
import torch.nn as nn
from einops import rearrange
from tqdm import tqdm

from config import VideoMAEExperimentConfig
from mask import *
from utils import *
from metrics import *
from plot import *


logger = logging.getLogger(__name__)


def train_model(
    config: VideoMAEExperimentConfig,
    device: str,
    model,
    train_dl: torch.utils.data.DataLoader,
    test_dl: torch.utils.data.DataLoader,
    num_patches: int,
    optimizer,
    lr_scheduler,
    accelerator,
    data_type,
    local_rank,
):
    """
    Runs model training

    Args:
        config: experiment config for this run
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
    model_config = config.video_mae_task_config.vit_config

    torch.cuda.empty_cache()
    model.to(device)

    num_frames = config.ecog_data_config.sample_length * config.ecog_data_config.new_fs

    if config.trainer_config.learning_rate == None:
        model, optimizer, train_dl, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dl, lr_scheduler
        )

    mse = nn.MSELoss(reduction='none')

    progress_bar = tqdm(
        range(epoch, config.trainer_config.num_epochs),
        ncols=1200,
        disable=(local_rank != 0),
    )
    for epoch in progress_bar:
        start = t.time()
        with torch.cuda.amp.autocast(dtype=data_type):
            model.train()
            for train_i, batch in enumerate(train_dl):
                optimizer.zero_grad()

                signal = batch.to(device)

                if config.ecog_data_config.norm == "batch":
                    signal = normalize(signal)
                else:
                    signal = torch.where(
                        signal == 0, torch.tensor(float("nan")), signal
                    )

                # mask indicating positions of channels that were rejected during preprocessing
                padding_mask = get_padding_mask(signal, model, device)

                # convert nans to 0
                signal = torch.nan_to_num(signal)

                # tube mask ratio as given by args for model train
                tube_mask_ratio = config.video_mae_task_config.tube_mask_ratio

                # masking out parts of the input to the encoder (same mask across frames)
                tube_mask = get_tube_mask(
                    model_config.frame_patch_size,
                    config.video_mae_task_config.tube_mask_ratio,
                    num_patches,
                    num_frames,
                    padding_mask,
                    device,
                )

                # selecting parts of the signal for the decoder to reconstruct
                if config.video_mae_task_config.decoder_mask_ratio == 0:
                    decoder_mask = get_decoder_mask(
                        config.video_mae_task_config.decoder_mask_ratio,
                        num_patches,
                        tube_mask,
                        device,
                    )
                else:
                    if config.video_mae_task_config.running_cell_masking:
                        decoder_mask = get_running_cell_mask(
                            config.video_mae_task_config.decoder_mask_ratio,
                            model_config.frame_patch_size,
                            num_frames,
                            tube_mask,
                            device,
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
                    full_recon_signal,
                    recon_output,
                    recon_target,
                    seen_output,
                    seen_target,
                    seen_target_signal,
                    seen_recon_signal,
                    unseen_target_signal,
                    unseen_recon_signal,
                ) = rearrange_signals(
                    config.ecog_data_config,
                    model_config,
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

                # calculate loss
                if config.trainer_config.loss == "patch":
                    loss = mse(recon_output, recon_target).nanmean()
                    seen_loss = mse(seen_output, seen_target).nanmean()
                elif config.trainer_config.loss == "signal":
                    loss = mse(unseen_recon_signal, unseen_target_signal).nanmean()
                    seen_loss = mse(seen_recon_signal, seen_target_signal).nanmean()
                elif config.trainer_config.loss == "both":
                    loss = mse(recon_output, recon_target).nanmean() + mse(
                        unseen_recon_signal, unseen_target_signal
                    ).nanmean()
                    seen_loss = mse(seen_output, seen_target).nanmean() + mse(
                        seen_recon_signal, seen_target_signal
                    ).nanmean()
                elif config.trainer_config.loss == "full":
                    loss = mse(full_recon_signal, signal).nanmean()
                    seen_loss = mse(seen_recon_signal, seen_target_signal).nanmean()
                elif config.trainer_config.loss == "highgamma":
                    loss = loss = mse(
                        unseen_recon_signal[:, 4, :, :],
                        unseen_target_signal[:, 4, :, :],
                    ).nanmean()
                    seen_loss = mse(
                        seen_recon_signal[:, 4, :, :],
                        seen_target_signal[:, 4, :, :],
                    ).nanmean()
                if torch.isnan(loss):
                    logger.error(f"Got nan loss for index {train_i}. Ignoring and continuing...")
                    continue

                accelerator.backward(loss)
                optimizer.step()

            model.eval()
            for test_i, batch in enumerate(test_dl):

                signal = batch.to(device)

                if config.ecog_data_config.norm == "batch":
                    signal = normalize(signal)
                else:
                    signal = torch.where(
                        signal == 0, torch.tensor(float("nan")), signal
                    )

                # mask indicating positions of channels that were rejected during preprocessing
                padding_mask = get_padding_mask(signal, model, device)

                # convert nans to 0
                signal = torch.nan_to_num(signal)

                # fixed tube mask ratio for model eval
                tube_mask_ratio = 0.5

                # masking out parts of the input to the encoder (same mask across frames)
                tube_mask = get_tube_mask(
                    model_config.frame_patch_size,
                    config.video_mae_task_config.tube_mask_ratio,
                    num_patches,
                    num_frames,
                    padding_mask,
                    device,
                )

                # selecting parts of the signal for the decoder to reconstruct
                if config.video_mae_task_config.decoder_mask_ratio == 0:
                    decoder_mask = get_decoder_mask(
                        config.video_mae_task_config.decoder_mask_ratio,
                        num_patches,
                        tube_mask,
                        device,
                    )
                elif config.video_mae_task_config.running_cell_masking:
                    decoder_mask = get_running_cell_mask(
                        config.video_mae_task_config.decoder_mask_ratio,
                        model_config.frame_patch_size,
                        num_frames,
                        tube_mask,
                        device,
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
                    full_recon_signal,
                    recon_output,
                    recon_target,
                    seen_output,
                    seen_target,
                    seen_target_signal,
                    seen_recon_signal,
                    unseen_target_signal,
                    unseen_recon_signal,
                ) = rearrange_signals(
                    config.ecog_data_config,
                    model_config,
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


                # calculate loss
                if config.trainer_config.loss == "patch":
                    loss = mse(recon_output, recon_target).nanmean()
                    seen_loss = mse(seen_output, seen_target).nanmean()
                elif config.trainer_config.loss == "signal":
                    loss = mse(unseen_recon_signal, unseen_target_signal).nanmean()
                    seen_loss = mse(seen_recon_signal, seen_target_signal).nanmean()
                elif config.trainer_config.loss == "both":
                    loss = mse(recon_output, recon_target).nanmean() + mse(
                        unseen_recon_signal, unseen_target_signal
                    ).nanmean()
                    seen_loss = mse(seen_output, seen_target).nanmean() + mse(
                        seen_recon_signal, seen_target_signal
                    ).nanmean()
                elif config.trainer_config.loss == "full":
                    loss = mse(full_recon_signal, signal).nanmean()
                    seen_loss = mse(seen_recon_signal, seen_target_signal).nanmean()
                elif config.trainer_config.loss == "highgamma":
                    loss = loss = mse(
                        unseen_recon_signal[:, 4, :, :],
                        unseen_target_signal[:, 4, :, :],
                    ).nanmean()
                    seen_loss = mse(
                        seen_recon_signal[:, 4, :, :],
                        seen_target_signal[:, 4, :, :],
                    ).nanmean()

                # save original and reconstructed signal for plotting (for highgamma)
                if test_i in [0, 5, 10, 15, 20]:

                    if config.ecog_data_config.norm == "batch":
                        recon_signal = normalize(recon_signal)
                    else:
                        full_recon_signal = full_recon_signal

                    new_model_recon = get_model_recon(
                        config.ecog_data_config.bands,
                        signal,
                        full_recon_signal,
                        epoch,
                        test_i,
                    )
                    model_recon = pd.concat([model_recon, new_model_recon])

                    dir = os.getcwd() + f"/results/full_recon_signals/"
                    if not os.path.exists(dir):
                        os.makedirs(dir)

                    model_recon.to_pickle(dir + f"{config.job_name}_recon_signal.txt")

            end = t.time()

            print("Epoch " + str(epoch) + " done. Time elapsed: " + str(end - start))

        # save model checkpoints
        checkpoint = {
            "epoch": epoch,
            "model": model,
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "experiment_config": config,
        }

    return model
