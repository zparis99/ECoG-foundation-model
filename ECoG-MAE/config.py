# TODO implement learning rate scheduler for better performance - initially Paul had implemented one, but we are using a fixed LR for now ...

import os
import math
import torch
import numpy as np
from accelerate import Accelerator, DeepSpeedPlugin
from dataclasses import dataclass, field

import utils
from models import *


# Config classes here are very roughly following the format of Tensorflow Model Garden: https://www.tensorflow.org/guide/model_garden#training_framework
# to try and make expanding to new models and tasks slightly easier by logically breaking up the parameters to training into distinct pieces and directly
# documenting the fields which can be configured.
@dataclass
class ECoGDataConfig:
    # If 'batch' then will normalize data within a batch.
    norm: str = None
    # Percentage of data to include in training/testing.
    data_size: float = 1.0
    # Batch size to train with.
    batch_size: int = 32
    # If true then convert data to power envelope by taking magnitude of Hilbert
    # transform.
    env: bool = False
    # Frequency bands for filtering raw iEEG data.
    bands: list[list[int]] = ([[4, 8], [8, 13], [13, 30], [30, 55], [70, 200]],)
    # Frequency to resample data to.
    new_fs: int = 20
    # Relative path to the dataset root directory.
    dataset_path: str = None
    # Proportion of data to have in training set. The rest will go to test set.
    train_data_proportion: float = 0.9
    # Number of seconds of data to use for a training example.
    sample_length: int = 2
    # TODO: Figure out what this does.
    shuffle: bool = False
    # If True then uses a mock data loader.
    test_loader: bool = False


@dataclass
class TrainerConfig:
    # Learning rate for training. If 0 then uses Adam scheduler.
    learning_rate: float = 0.0
    # TODO: Add max learning rate
    # Number of epochs to train over data.
    num_epochs: int = 10


@dataclass
class ViTConfig:
    # Dimensionality of token embeddings.
    dim: int = 512
    # Dimensionality of feedforward network after attention layer.
    mlp_dim: int = 512
    # TODO
    patch_size: list[int] = None
    # TODO
    frame_patch_size: int = 0
    # TODO
    patch_dims: list[int] = field(default_factory=lambda: [1, 1, 1])
    # Prepend classification token to input if True. Always True if
    # use_contrastive_loss is True.
    use_cls_token: bool = False


@dataclass
class VideoMAETaskConfig:
    # Config for model.
    vit_config: ViTConfig = field(default_factory=ViTConfig)
    # Proportion of tubes to mask out. See VideoMAE paper for details.
    tube_mask_ratio: float = 0.5
    # Proportion of
    decoder_mask_ratio: float = 0
    # If true then use contrastive loss to train model. Currently not supported.
    use_contrastive_loss: bool = False


@dataclass
class VideoMAEExperimentConfig:
    video_mae_task_config: VideoMAETaskConfig = field(
        default_factory=VideoMAETaskConfig
    )
    ecog_data_config: ECoGDataConfig = field(default_factory=ECoGDataConfig)
    trainer_config: TrainerConfig = field(default_factory=TrainerConfig)
    # Name of training job. Will be used to save metrics.
    job_name: str = None


def create_video_mae_experiment_config(args):
    """Convert command line arguments to an experiment config for VideoMAE."""
    return VideoMAEExperimentConfig(
        video_mae_task_config=VideoMAETaskConfig(
            vit_config=ViTConfig(
                dim=args.dim,
                mlp_dim=args.mlp_dim,
                patch_size=args.patch_size,
                patch_dims=args.patch_dims,
                frame_patch_size=args.frame_patch_size,
                use_cls_token=args.use_cls_token,
            ),
            tube_mask_ratio=args.tube_mask_ratio,
            decoder_mask_ration=args.decoder_mask_ratio,
            use_contrastive_loss=args.use_contrastive_loss,
        ),
        trainer_config=TrainerConfig(
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
        ),
        ecog_data_config=ECoGDataConfig(
            norm=args.norm,
            data_size=args.data_size,
            env=args.env,
            bands=args.bands,
            new_fs=args.new_fs,
            dataset_path=args.dataset_path,
            train_data_proportion=args.train_data_proportion,
            sample_length=args.sample_length,
            shuffle=args.shuffle,
            test_loader=args.test_loader,
        ),
        job_name=args.job_name,
    )


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
        args: input arguments
        device: cuda device

    Returns:
        model: an untrained model instance with randomly initialized parameters
        optimizer: an Adam optimizer instance - https://www.analyticsvidhya.com/blog/2023/12/adam-optimizer/
        lr_scheduler: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
        num_patches: the number of patches in which the input data is segmented
    """
    model_config = config.video_mae_task_config.vit_config

    ### class token config ###
    use_cls_token = model_config.use_cls_token

    ### Loss Config ###
    use_contrastive_loss = args.use_contrastive_loss
    constrastive_loss_weight = 1.0
    use_cls_token = (
        True if use_contrastive_loss else use_cls_token
    )  # if using contrastive loss, we need to add a class token

    input_size = [1, 8, 8]
    print("input_size", input_size)
    num_frames = args.sample_length * args.new_fs

    img_size = (1, 8, 8)
    patch_dims = tuple(model_config.patch_dims)
    frame_patch_size = model_config.frame_patch_size
    num_patches = int(  # Defining the number of patches
        (img_size[0] / patch_dims[0])
        * (img_size[1] / patch_dims[1])
        * (img_size[2] / patch_dims[2])
        * num_frames
        / frame_patch_size
    )

    num_encoder_patches = int(num_patches * (1 - args.tube_mask_ratio))
    num_decoder_patches = int(num_patches * (1 - args.decoder_mask_ratio))
    print("num_patches", num_patches)
    print("num_encoder_patches", num_encoder_patches)
    print("num_decoder_patches", num_decoder_patches)

    if model_config.dim == 0:
        dim = (
            patch_dims[0]
            * patch_dims[1]
            * patch_dims[2]
            * frame_patch_size
            * len(args.bands)
        )
    else:
        dim = args.dim

    if model_config.mlp_dim == 0:
        mlp_dim = (
            patch_dims[0]
            * patch_dims[1]
            * patch_dims[2]
            * frame_patch_size
            * len(args.bands)
        )
    else:
        mlp_dim = model_config.mlp_dim

    model = SimpleViT(
        image_size=img_size,  # depth, height, width
        image_patch_size=patch_dims,  # depth, height, width patch size - change width from patch_dims to 1
        frames=num_frames,
        frame_patch_size=frame_patch_size,
        depth=12,
        heads=12,
        dim=dim,
        mlp_dim=mlp_dim,
        channels=len(args.bands),
        use_rope_emb=False,
        use_cls_token=model_config.use_cls_token,
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

    max_lr = 3e-5  # 3e-5 seems to be working best? original videomae used 1.5e-4

    if args.learning_rate == 0:
        optimizer = torch.optim.AdamW(opt_grouped_parameters, lr=max_lr)
    else:
        optimizer = torch.optim.AdamW(opt_grouped_parameters, lr=args.learning_rate)

    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        epochs=args.num_epochs,
        steps_per_epoch=math.ceil(num_train_samples / args.batch_size),
    )

    print("\nDone with model preparations!")

    return model, optimizer, lr_scheduler, num_patches
