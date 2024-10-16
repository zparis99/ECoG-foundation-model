import torch
from mask import *
from utils import *
from metrics import *
from plot import *


def forward_model(model, device, config, num_patches, num_frames, mse):
    # TODO: Actually move this code into model.
    model_config = config.video_mae_task_config.vit_config
    padding_mask = get_padding_mask(signal, model, device)

    # convert nans to 0
    signal = torch.nan_to_num(signal)

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
        loss = mse(recon_output, recon_target)
        seen_loss = mse(seen_output, seen_target)
    elif config.trainer_config.loss == "signal":
        loss = mse(unseen_recon_signal, unseen_target_signal)
        seen_loss = mse(seen_recon_signal, seen_target_signal)
    elif config.trainer_config.loss == "both":
        loss = mse(recon_output, recon_target) + mse(
            unseen_recon_signal, unseen_target_signal
        )
        seen_loss = mse(seen_output, seen_target) + mse(
            seen_recon_signal, seen_target_signal
        )
    elif config.trainer_config.loss == "full":
        loss = mse(full_recon_signal, signal)
        seen_loss = mse(seen_recon_signal, seen_target_signal)
    elif config.trainer_config.loss == "highgamma":
        loss = loss = mse(
            unseen_recon_signal[:, 4, :, :],
            unseen_target_signal[:, 4, :, :],
        )
        seen_loss = mse(
            seen_recon_signal[:, 4, :, :],
            seen_target_signal[:, 4, :, :],
        )
        
    return loss, seen_loss


def train_single_epoch(train_dl, accelerator, optimizer, device, model, config, num_patches, num_frames, logger, mse):
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
        loss, _ = forward_model(model, device, config, num_patches, num_frames, mse)
        if torch.isnan(loss):
            logger.error(f"Got nan loss for index {train_i}. Ignoring and continuing...")
            continue

        accelerator.backward(loss)
        optimizer.step()
        

def test_single_epoch(test_dl, device, model, config, num_patches, num_frames, logger, mse):
    model.eval()
    with torch.no_grad():
        for test_i, batch in enumerate(test_dl):

            signal = batch.to(device)

            if config.ecog_data_config.norm == "batch":
                signal = normalize(signal)
            else:
                signal = torch.where(
                    signal == 0, torch.tensor(float("nan")), signal
                )

            loss, _ = forward_model(model, device, config, num_patches, num_frames, mse)
            if torch.isnan(loss):
                logger.error(f"Got nan loss for index {train_i}. Ignoring and continuing...")
                continue