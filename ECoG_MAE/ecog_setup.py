# TODO implement learning rate scheduler for better performance - initially Paul had implemented one, but we are using a fixed LR for now ...

import os
import math
import torch
import numpy as np
from accelerate import Accelerator, DeepSpeedPlugin

import utils
from models import *
from config import VideoMAEExperimentConfig
from mae_st_util.models_mae import MaskedAutoencoderViT


def system_setup():
    """
    Sets up accelerator, device, datatype precision and local rank

    Args:

    Returns:
        accelerator: an accelerator instance - https://huggingface.co/docs/accelerate/en/index
        device: the gpu to be used for model training
        data_type: the data type to be used, we use "fp16" mixed precision - https://towardsdatascience.com/understanding-mixed-precision-training-4b246679c7c4
        local_rank: the local rank environment variable (only needed for multi-gpu training)
    """

    # tf32 data type is faster than standard float32
    torch.backends.cuda.matmul.allow_tf32 = True

    # seed all random functions
    seed = 42
    utils.seed_everything(seed)

    # accelerator = Accelerator(split_batches=False, mixed_precision="fp16")
    accelerator = Accelerator(split_batches=False)

    device = "cuda:0"

    # set data_type to match your mixed precision
    if accelerator.mixed_precision == "bf16":
        data_type = torch.bfloat16
    elif accelerator.mixed_precision == "fp16":
        data_type = torch.float16
    else:
        data_type = torch.float32

    # only need this if we want to set up multi GPU training
    local_rank = os.getenv("RANK")
    if local_rank is None:
        local_rank = 0
    else:
        local_rank = int(local_rank)

    return accelerator, device, data_type, local_rank


def model_setup(config: VideoMAEExperimentConfig, device, num_train_samples):
    """
    Sets up model config

    Args:
        config: experiment config
        device: cuda device

    Returns:
        model: an untrained model instance with randomly initialized parameters
        optimizer: an Adam optimizer instance - https://www.analyticsvidhya.com/blog/2023/12/adam-optimizer/
        lr_scheduler: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
        num_patches: the number of patches in which the input data is segmented
    """
    model_config = config.video_mae_task_config.vit_config

    num_frames = config.ecog_data_config.sample_length * config.ecog_data_config.new_fs

    frame_patch_size = model_config.frame_patch_size
    num_patches = int(  # Defining the number of patches
        constants.GRID_SIZE ** 2 * num_frames // model_config.patch_size // frame_patch_size
    )

    num_encoder_patches = int(num_patches * (1 - config.video_mae_task_config.encoder_mask_ratio))
    num_decoder_patches = int(num_patches * (1 - config.video_mae_task_config.decoder_mask_ratio))
    print("num_patches", num_patches)
    print("num_encoder_patches", num_encoder_patches)
    print("num_decoder_patches", num_decoder_patches)
        
    model = MaskedAutoencoderViT(
        img_size=constants.GRID_SIZE,
        patch_size=model_config.patch_size,
        in_chans=len(config.ecog_data_config.bands),
        embed_dim=model_config.dim,
        depth=model_config.depth,
        num_heads=model_config.num_heads,
        decoder_embed_dim=model_config.decoder_embed_dim,
        decoder_depth=model_config.decoder_depth,
        decoder_num_heads=model_config.decoder_num_heads,
        mlp_ratio=model_config.mlp_ratio,
        norm_pix_loss=config.video_mae_task_config.norm_pix_loss,
        num_frames=num_frames,
        t_patch_size=model_config.frame_patch_size,
        no_qkv_bias=model_config.no_qkv_bias,
        sep_pos_embed=model_config.sep_pos_embed,
        trunc_init=model_config.trunc_init,
        cls_embed=model_config.use_cls_token,
        pred_t_dim=num_frames // model_config.frame_patch_size,
        img_mask=None,
        pct_masks_to_decode=config.video_mae_task_config.pct_masks_to_decode,
    )
    utils.count_params(model)

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    opt_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 1e-2,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(opt_grouped_parameters, lr=config.trainer_config.max_learning_rate)

    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.trainer_config.max_learning_rate,
        epochs=config.trainer_config.num_epochs,
        steps_per_epoch=math.ceil(num_train_samples / config.ecog_data_config.batch_size),
    )

    print("\nDone with model preparations!")

    return model, optimizer, lr_scheduler, num_patches
