from dataclasses import dataclass, fields, field, is_dataclass, asdict
import yaml
from typing import Optional

# TODO: reduce this file to just the necessary pieces for all use-cases. (likely just part of data config and ViTConfig)


# Config classes here are very roughly following the format of Tensorflow Model Garden: https://www.tensorflow.org/guide/model_garden#training_framework
# to try and make expanding to new models and tasks slightly easier by logically breaking up the parameters to training into distinct pieces and directly
# documenting the fields which can be configured.
@dataclass
class ECoGDataConfig:
    # Percentage of data to include in training/testing.
    data_size: float = 1.0
    # Batch size to train with.
    batch_size: int = 32
    # If true then convert data to power envelope by taking magnitude of Hilbert
    # transform.
    env: bool = False
    # Frequency bands for filtering raw iEEG data.
    bands: list[list[int]] = field(
        default_factory=lambda: [[4, 8], [8, 13], [13, 30], [30, 55], [70, 200]]
    )
    # Original sanmpling frequency of data.
    original_fs: int = 512
    # Frequency to resample data to.
    new_fs: int = 20
    # Relative path to the dataset root directory.
    dataset_path: str = None
    # Proportion of data to have in training set. The rest will go to test set.
    train_data_proportion: float = 0.9
    # Number of seconds of data to use for a training example.
    sample_length: int = 2
    # If true then shuffle the data before splitting to train and eval.
    shuffle: bool = False
    # The maximum number of files to sample from at once. Limited by RAM.
    max_open_files: int = 10


@dataclass
class TrainerConfig:
    # Max learning rate for scheduler.
    max_learning_rate: float = 3e-5
    # Number of epochs to train over data.
    num_epochs: int = 10
    # Weight decay for optimizer.
    weight_decay: float = 0.0
    # Mixed precision to use in training. See accelerate.Accelerator for details.
    mixed_precision: str = "no"
    # Number of gradient accumulation steps.
    gradient_accumulation_steps: int = 1


@dataclass
class ViTConfig:
    # Dimensionality of token embeddings.
    dim: int = 1024
    # Dimensionality to transform encoder embeddings into when passing into the decoder.
    decoder_embed_dim: int = 512
    # Ratio of input dimensionality to use as a hidden layer in Transformer Block MLP's
    mlp_ratio: float = 4.0
    # Depth of encoder.
    depth: int = 24
    # Depth of decoder.
    decoder_depth: int = 8
    # Number of heads in encoder.
    num_heads: int = 16
    # Number of heads in decoder.
    decoder_num_heads: int = 16
    # The number of electrodes in a patch.
    patch_size: int = 0
    # The number of frames to include in a tube per video mae.
    frame_patch_size: int = 1
    # Prepend classification token to input if True.
    use_cls_token: bool = False
    # If true then use a separate position embedding for the decoder.
    sep_pos_embed: bool = True
    # Use truncated normal initialization if True.
    trunc_init: bool = False
    # If True then don't use a bias for query, key, and values in attention blocks.
    no_qkv_bias: bool = False
    # Attention projection layer dropout.
    proj_drop: float = 0.1
    # Stochastic depth for residual connections.
    drop_path: float = 0.05


@dataclass
class LoggingConfig:
    # Directory to write logs to (i.e. tensorboard events, etc).
    event_log_dir: str = "event_logs/"
    # Directory to write plots to.
    plot_dir: str = "plots/"
    # Number of steps to print training progress after.
    print_freq: int = 20


@dataclass
class VideoMAETaskConfig:
    # Config for model.
    vit_config: ViTConfig = field(default_factory=ViTConfig)
    # Proportion of tubes to mask out. See VideoMAE paper for details.
    encoder_mask_ratio: float = 0.5
    # Percentage of masks tokens to pass into decoder for reconstruction.
    pct_masks_to_decode: float = 0
    # Weight factor for loss computation. Final loss is determined by
    # loss = alpha * -(pearson correlation) + (1- alpha) * mean squared error. Alpha=1 is -correlation loss,
    # alpha = 0 is mse loss.
    alpha: float = 0.5
    # Name of model in registry to use.
    model_name: Optional[str] = None


@dataclass
class VideoMAEExperimentConfig:
    video_mae_task_config: VideoMAETaskConfig = field(
        default_factory=VideoMAETaskConfig
    )
    ecog_data_config: ECoGDataConfig = field(default_factory=ECoGDataConfig)
    trainer_config: TrainerConfig = field(default_factory=TrainerConfig)
    logging_config: LoggingConfig = field(default_factory=LoggingConfig)
    # Name of training job. Will be used to save metrics.
    job_name: str = None
    format_fields: list[str] = None


# Utility function to recursively convert dicts to dataclass instances
def dict_to_config(d: dict, config_class):
    """Recursively convert a dict d to an instance of config_class."""
    init_kwargs = {}
    for field_info in fields(config_class):
        field_name = field_info.name
        field_type = field_info.type
        if field_name not in d:
            continue
        field_value = d[field_name]
        if is_dataclass(field_type) and isinstance(field_value, dict):
            init_kwargs[field_name] = dict_to_config(field_value, field_type)
        else:
            init_kwargs[field_name] = field_value
    return config_class(**init_kwargs)


# Load YAML config into nested dataclass


def create_video_mae_experiment_config_from_yaml(
    yaml_file_path: str,
) -> VideoMAEExperimentConfig:
    with open(yaml_file_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return dict_to_config(config_dict, VideoMAEExperimentConfig)


# Write the config to YAML
def write_config_file_to_yaml(path: str, experiment_config):
    config_dict = asdict(experiment_config)

    with open(path, "w") as f:
        yaml.safe_dump(config_dict, f, sort_keys=False)
