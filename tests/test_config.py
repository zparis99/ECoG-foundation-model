import configparser
from dataclasses import asdict
import os
import pytest

from config import (
    create_video_mae_experiment_config,
    create_video_mae_experiment_config_from_file,
    VideoMAEExperimentConfig,
    VideoMAETaskConfig,
    ViTConfig,
    ECoGDataConfig,
    TrainerConfig,
    LoggingConfig,
    write_config_file,
)
from parser import arg_parser

CONFIG_VALUES = {
    "VideoMAETaskConfig.ViTConfig": {
        "dim": 128,
        "decoder_embed_dim": 64,
        "mlp_ratio": 4.0,
        "depth": 12,
        "decoder_depth": 4,
        "num_heads": 8,
        "decoder_num_heads": 4,
        "patch_size": 1,
        "frame_patch_size": 4,
        "use_cls_token": False,
        "sep_pos_embed": False,
        "trunc_init": False,
        "no_qkv_bias": False,
    },
    "VideoMAETaskConfig": {
        "encoder_mask_ratio": 0.75,
        "pct_masks_to_decode": 1.0,
        "alpha": 0.5,
    },
    "ECoGDataConfig": {
        "data_size": 1.0,
        "batch_size": 64,
        "env": False,
        "bands": [[4, 8], [8, 13], [13, 30], [30, 55], [70, 200]],
        "original_fs": 512,
        "new_fs": 20,
        "dataset_path": "dataset_full",
        "train_data_proportion": 0.9,
        "sample_length": 2,
        "shuffle": False,
        "test_loader": False,
    },
    "LoggingConfig": {
        "event_log_dir": "test_logging_dir/",
        "print_freq": 30,
        "plot_dir": "test_plot_dir/",
    },
    "TrainerConfig": {
        "max_learning_rate": 0.0,
        "num_epochs": 10,
    },
}


@pytest.fixture
def fake_config_path(tmp_path):
    config = configparser.ConfigParser()
    config.read_dict(CONFIG_VALUES)
    path = os.path.join(tmp_path, "config.ini")
    with open(path, "w") as fp:
        config.write(fp)

    return path


def test_create_video_mae_experiment_config_from_file(fake_config_path):
    experiment_config = create_video_mae_experiment_config_from_file(fake_config_path)

    # Make sure all experiment config values match config file
    for section in CONFIG_VALUES.keys():
        for field, value in CONFIG_VALUES[section].items():
            if section == "VideoMAETaskConfig.ViTConfig":
                actual_config_dict = asdict(
                    experiment_config.video_mae_task_config.vit_config
                )
            if section == "VideoMAETaskConfig":
                actual_config_dict = asdict(experiment_config.video_mae_task_config)
            if section == "ECoGDataConfig":
                actual_config_dict = asdict(experiment_config.ecog_data_config)
            if section == "TrainerConfig":
                actual_config_dict = asdict(experiment_config.trainer_config)
            if section == "LoggingConfig":
                actual_config_dict = asdict(experiment_config.logging_config)

            assert actual_config_dict[field] == value


def test_config_file_loads_unchanged_without_command_line_args(
    mocker, fake_config_path
):
    mocker.patch("sys.argv", ["main.py", "--config-file", fake_config_path])

    args = arg_parser()
    experiment_config = create_video_mae_experiment_config(args)

    # Make sure all experiment config values match config file
    for section in CONFIG_VALUES.keys():
        for field, value in CONFIG_VALUES[section].items():
            if section == "VideoMAETaskConfig.ViTConfig":
                actual_config_dict = asdict(
                    experiment_config.video_mae_task_config.vit_config
                )
            if section == "VideoMAETaskConfig":
                actual_config_dict = asdict(experiment_config.video_mae_task_config)
            if section == "ECoGDataConfig":
                actual_config_dict = asdict(experiment_config.ecog_data_config)
            if section == "TrainerConfig":
                actual_config_dict = asdict(experiment_config.trainer_config)
            if section == "LoggingConfig":
                actual_config_dict = asdict(experiment_config.logging_config)

            assert actual_config_dict[field] == value

    # Clean up the temporary file
    os.unlink(fake_config_path)


def test_command_line_args_overwrite_config(mocker, fake_config_path):
    command_line_args = {
        # ViTConfig parameters
        "dim": 130,
        "decoder_embed_dim": 96,
        "mlp_ratio": 3.0,
        "depth": 16,
        "decoder_depth": 6,
        "num_heads": 10,
        "decoder_num_heads": 6,
        "patch_size": 2,
        "frame_patch_size": 5,
        "use_cls_token": True,
        "sep_pos_embed": True,
        "trunc_init": True,
        "no_qkv_bias": True,
        # VideoMAETaskConfig parameters
        "encoder_mask_ratio": 0.73,
        "pct_masks_to_decode": 0.02,
        "alpha": 0.75,
        # ECoGDataConfig parameters
        "data_size": 0.98,
        "batch_size": 63,
        "env": True,
        "bands": [[3, 7], [7, 12], [12, 29], [29, 54], [69, 199]],
        "original_fs": 510,
        "new_fs": 21,
        "dataset_path": "dataset_modified",
        "train_data_proportion": 0.89,
        "sample_length": 3,
        "shuffle": True,
        "test_loader": True,
        # TrainerConfig parameters
        "max_learning_rate": 0.001,
        "num_epochs": 11,
        # LoggingConfig parameters
        "event_log_dir": "new_dir/",
        "print_freq": 100,
        "plot_dir": "new_plot_dir/",
    }

    # Convert to list which can be passed into sys.argv
    cli_argv = []
    for arg_name, arg_value in command_line_args.items():
        # Prepend -- and replace _ with -
        arg_name = arg_name.replace("_", "-")
        arg_name = "--" + arg_name

        # Only add arg name if bool value.
        if isinstance(arg_value, bool) and arg_value:
            cli_argv.append(arg_name)
        elif not isinstance(arg_value, bool):
            cli_argv.append(arg_name)
            cli_argv.append(str(arg_value))
    print(cli_argv)
    mocker.patch(
        "sys.argv",
        [
            "main.py",
            "--config-file",
            fake_config_path,
        ]
        + cli_argv,
    )

    args = arg_parser()
    experiment_config = create_video_mae_experiment_config(args)

    config_file = configparser.ConfigParser()
    config_file.read("tests/testdata/video_mae_config_file.ini")
    # Make sure all experiment config values match config file
    for section in CONFIG_VALUES.keys():
        for field, value in CONFIG_VALUES[section].items():
            if section == "VideoMAETaskConfig.ViTConfig":
                actual_config_dict = asdict(
                    experiment_config.video_mae_task_config.vit_config
                )
            if section == "VideoMAETaskConfig":
                actual_config_dict = asdict(experiment_config.video_mae_task_config)
            if section == "ECoGDataConfig":
                actual_config_dict = asdict(experiment_config.ecog_data_config)
            if section == "TrainerConfig":
                actual_config_dict = asdict(experiment_config.trainer_config)
            if section == "LoggingConfig":
                actual_config_dict = asdict(experiment_config.logging_config)

            assert (
                actual_config_dict[field] == command_line_args[field]
            ), f"For field {field} expected: {command_line_args[field]} actual: {actual_config_dict[field]}"

    # Clean up the temporary file
    os.unlink(fake_config_path)


def test_write_config_file(mocker, tmp_path):
    test_config = VideoMAEExperimentConfig(
        video_mae_task_config=VideoMAETaskConfig(
            vit_config=ViTConfig(
                dim=768,
                decoder_embed_dim=384,
                mlp_ratio=3.0,
                depth=12,
                decoder_depth=4,
                num_heads=8,
                decoder_num_heads=8,
                patch_size=2,
                frame_patch_size=4,
                use_cls_token=True,
                sep_pos_embed=False,
                trunc_init=True,
                no_qkv_bias=True,
            ),
            encoder_mask_ratio=0.75,
            pct_masks_to_decode=0.25,
            alpha=0.5,
        ),
        ecog_data_config=ECoGDataConfig(
            data_size=0.8,
            batch_size=128,
            env=True,
            bands=[[1, 4], [4, 8], [8, 13], [13, 30], [30, 70]],
            original_fs=1000,
            new_fs=100,
            dataset_path="custom_dataset",
            train_data_proportion=0.8,
            sample_length=4,
            shuffle=True,
            test_loader=True,
        ),
        trainer_config=TrainerConfig(max_learning_rate=1e-4, num_epochs=50),
        logging_config=LoggingConfig(
            event_log_dir="custom_logs/", print_freq=50, plot_dir="custom_plots/"
        ),
        job_name="custom_training_job",
    )

    tmp_file_path = os.path.join(tmp_path, "write_config.ini")

    write_config_file(tmp_file_path, test_config)

    # Read the written file
    config = configparser.ConfigParser()
    config.read(tmp_file_path)

    # Check if all expected sections are present
    assert set(config.sections()) == {
        "VideoMAETaskConfig.ViTConfig",
        "VideoMAETaskConfig",
        "ECoGDataConfig",
        "LoggingConfig",
        "TrainerConfig",
        "JobDetails",
    }

    # Lastly make sure it can be read correctly by our reading function.
    mocker.patch("sys.argv", ["main.py", "--config-file", tmp_file_path])

    args = arg_parser()
    experiment_config = create_video_mae_experiment_config(args)

    # Make sure all experiment config values match config file
    assert experiment_config == test_config

    # Clean up the temporary file
    os.unlink(tmp_file_path)
