import torch


def get_tube_mask(args, num_patches, num_frames, device):

    tube_mask = torch.zeros(num_patches // num_frames).to(device).to(torch.bool)
    mask_idx_candidates = torch.randperm(len(tube_mask))
    tube_idx = mask_idx_candidates[
        : int(num_patches / num_frames * (1 - args.tube_mask_ratio))
    ]
    tube_mask[tube_idx] = True
    tube_mask = tube_mask.tile(num_frames)

    return tube_mask


def get_decoder_mask(args, num_patches, tube_mask, device):

    decoder_mask = torch.zeros(num_patches).to(device).to(torch.bool)
    remaining_mask_idx = (~tube_mask).nonzero()
    decoder_mask_idx = remaining_mask_idx[
        : int(num_patches * (1 - args.decoder_mask_ratio))
    ]
    decoder_mask[decoder_mask_idx] = True

    return decoder_mask


def get_running_cell_mask(args, num_frames, tube_mask, device):

    num_patch_per_cell = 4
    num_mask_per_cell = int(args.decoder_mask_ratio * num_patch_per_cell)
    stride = int(num_patch_per_cell / num_mask_per_cell)
    num_patch_per_frame = 16  # change to be flexible #TODO
    cell = torch.ones(num_frames // args.frame_patch_size, num_patch_per_cell)
    # mask out patches in cell so that the mask spatially progresses across frames
    # Quing et al., 2023, MAR: Masked Autoencoder for Efficient Action Recognition
    for i in range(num_frames // args.frame_patch_size):
        for j in range(num_mask_per_cell):
            cell[i, (int(j * stride) + i) % num_patch_per_cell] = 0

    running_cell_mask = (
        cell.repeat(1, int(num_patch_per_frame / num_patch_per_cell))
        .flatten(0)
        .to(device)
        .to(torch.bool)
    )

    # filter out patches that were seen by the encoder
    running_cell_mask[~tube_mask] == False

    return running_cell_mask
