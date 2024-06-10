import os
import torch
import numpy as np
import random
from einops import rearrange


def seed_everything(seed=0, cudnn_deterministic=True):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
        c=len(args.bands),
        d=1,
        f=num_frames // args.frame_patch_size,
        pd=1,
        ps=args.patch_size,
        pf=args.frame_patch_size,
    )

    seen_target_signal = rearrange(
        seen_target,
        "b (f d s) (pd ps pf c) -> b c (f pf) (d pd s ps)",
        c=len(args.bands),
        d=1,
        f=num_frames // args.frame_patch_size,
        pd=1,
        ps=args.patch_size,
        pf=args.frame_patch_size,
    )

    # rearrange unseen patches into signal
    unseen_recon_signal = rearrange(
        unseen_output,
        "b (f d s) (pd ps pf c) -> b c (f pf) (d pd s ps)",
        c=len(args.bands),
        d=1,
        f=num_frames // args.frame_patch_size,
        pd=1,
        ps=args.patch_size,
        pf=args.frame_patch_size,
    )

    unseen_target_signal = rearrange(
        unseen_target,
        "b (f d s) (pd ps pf c) -> b c (f pf) (d pd s ps)",
        c=len(args.bands),
        d=1,
        f=num_frames // args.frame_patch_size,
        pd=1,
        ps=args.patch_size,
        pf=args.frame_patch_size,
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
