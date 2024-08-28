import os
import torch
import math
import numpy as np
import random
from einops import rearrange

from config import ECoGDataConfig, ViTConfig


def seed_everything(seed=0, cudnn_deterministic=True):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def resample_mean_signals(signal: np.array, old_fs: int, new_fs: int) -> np.array:
    """Resample signal with sampling rate of old_fs Hz to new_fs Hz by taking means over windows of data.

    Args:
        signal (np.array): shape [bands, num_electrodes, samples]
        old_fs (int): Old sample rate in Hz.
        new_fs (int): Sample rate to resample to in Hz.
        
    Returns:
        np.array: Resampled signal with the new sample rate.
    """
    window_width = old_fs / new_fs
    num_samples = signal.shape[2]
    # TODO: revisit using ceil here. By using ceil our final entry in our new array may be averaged
    # over fewer samples then the previous entries.
    num_new_samples = int(np.ceil(num_samples * new_fs / old_fs))
    
    resampled_signal = np.zeros((signal.shape[0], signal.shape[1], num_new_samples))
    
    # Current index needs to be a float to handle cases where old_fs and new_fs are not divisible.
    # Flooring results only after summing window widths ensures all of the data is included in the
    # averages.
    current_idx = 0.0
    for i in range(num_new_samples):
        start_idx = int(np.floor(current_idx))
        end_idx = int(np.floor(current_idx + window_width))
        
        if end_idx > start_idx:  # Ensure there is a range to average over
            resampled_signal[:, :, i] = np.mean(signal[:, :, start_idx:end_idx], axis=2)
        else:  # If the window size rounds to 0, just take the value at start_idx
            resampled_signal[:, :, i] = signal[:, :, start_idx]
        
        current_idx += window_width
    
    return resampled_signal
        

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("param counts:\n{:,} total\n{:,} trainable".format(total, trainable))
    return trainable


def normalize(raw_signal):

    mean = torch.mean(raw_signal, dim=(0, 2), keepdim=True)
    std = torch.std(raw_signal, dim=(0, 2), keepdim=True)
    signal = (raw_signal - mean) / std

    return signal


def rearrange_signals(
    data_config: ECoGDataConfig,
    model_config: ViTConfig,
    model,
    device,
    signal,
    num_frames,
    decoder_out,
    padding_mask,
    tube_mask,
    decoder_mask,
    decoder_padding_mask,
):

    # parts of the reconstructed signal that were not seen by the encoder
    unseen_output = decoder_out[:, len(tube_mask.nonzero()) :]

    # parts of the reconstructed signal that were seen by the encoder
    seen_output = decoder_out[:, : len(tube_mask.nonzero())]

    # rearrange original signal into patches
    target_patches = model.patchify(signal)
    target_patches_vit = rearrange(target_patches, "b ... d -> b (...) d")

    # parts of the original signal not seen by the encoder
    unseen_target = target_patches_vit[:, decoder_mask][:, decoder_padding_mask]

    # parts of the original signal seen by the encoder
    seen_target = target_patches_vit[:, ~decoder_mask]

    # rearranging seen and unseen parts of the reconstructed signal into original position
    full_recon_patches = (
        torch.zeros(target_patches_vit.shape).fill_(float("nan")).to(device)
    )

    tube_idx = torch.nonzero(tube_mask).squeeze()
    decoder_idx = torch.nonzero(decoder_mask & padding_mask).squeeze()

    full_recon_patches[:, tube_idx, :] = seen_output
    full_recon_patches[:, decoder_idx, :] = unseen_output

    full_recon_signal = model.unpatchify(full_recon_patches)

    # rearrange seen patches into signal
    seen_recon_signal = rearrange(
        seen_output,
        "b (f d s) (pd ps pf c) -> b c (f pf) (d pd s ps)",
        c=len(data_config.bands),
        d=1,
        f=num_frames // model_config.frame_patch_size,
        pd=1,
        ps=model_config.patch_size,
        pf=model_config.frame_patch_size,
    )

    seen_target_signal = rearrange(
        seen_target,
        "b (f d s) (pd ps pf c) -> b c (f pf) (d pd s ps)",
        c=len(data_config.bands),
        d=1,
        f=num_frames // model_config.frame_patch_size,
        pd=1,
        ps=model_config.patch_size,
        pf=model_config.frame_patch_size,
    )

    # rearrange unseen patches into signal
    unseen_recon_signal = rearrange(
        unseen_output,
        "b (f d s) (pd ps pf c) -> b c (f pf) (d pd s ps)",
        c=len(data_config.bands),
        d=1,
        f=num_frames // model_config.frame_patch_size,
        pd=1,
        ps=model_config.patch_size,
        pf=model_config.frame_patch_size,
    )

    unseen_target_signal = rearrange(
        unseen_target,
        "b (f d s) (pd ps pf c) -> b c (f pf) (d pd s ps)",
        c=len(data_config.bands),
        d=1,
        f=num_frames // model_config.frame_patch_size,
        pd=1,
        ps=model_config.patch_size,
        pf=model_config.frame_patch_size,
    )

    return (
        full_recon_signal,
        unseen_output,
        unseen_target,
        seen_output,
        seen_target,
        seen_target_signal,
        seen_recon_signal,
        unseen_target_signal,
        unseen_recon_signal,
    )


def contrastive_loss(
    cls_token1: torch.Tensor, cls_token2: torch.Tensor, temperature: torch.Tensor
):
    feat1 = cls_token1 / cls_token1.norm(dim=1, keepdim=True)
    feat2 = cls_token2 / cls_token2.norm(dim=1, keepdim=True)

    cosine_sim = feat1 @ feat2.T
    logit_scale = temperature.exp()  # log scale, learned during training
    feat1 = cosine_sim * logit_scale
    feat2 = feat1.T

    labels = torch.arange(feat1.shape[0]).to(feat1.device)
    loss = (
        torch.nn.functional.cross_entropy(feat1, labels)
        + torch.nn.functional.cross_entropy(feat2, labels)
    ) / 2
    return loss
