import configparser
from dataclasses import asdict
import os
import pytest

from config import create_video_mae_experiment_config
from parser import arg_parser

CONFIG_VALUES = {
    "VideoMAETaskConfig.ViTConfig": {
        "dim": 128,
        "mlp_dim": 128,
        "patch_size": 1,
        "patch_dims": [1, 1, 1],
        "frame_patch_size": 4,
        "use_cls_token": False,
    },
    "VideoMAETaskConfig": {
        "tube_mask_ratio": 0.75,
        "decoder_mask_ratio": 0.0,
        "use_contrastive_loss": False,
        "running_cell_masking": False,
    },
    "ECoGDataConfig": {
        "norm": "hour",
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
    },
    "TrainerConfig": {
        "learning_rate": 0.0,
        "num_epochs": 10,
        "loss": "patch",
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

def test_config_file_loads_unchanged_without_command_line_args(mocker, fake_config_path):
    mocker.patch(
        "sys.argv",
        [
            "main.py",
            "--config-file",
            fake_config_path
        ]
    )
    
    args = arg_parser()
    experiment_config = create_video_mae_experiment_config(args)
    
    # Make sure all experiment config values match config file
    for section in CONFIG_VALUES.keys():
        for field, value in CONFIG_VALUES[section].items():
            if section == "VideoMAETaskConfig.ViTConfig":
                actual_config_dict = asdict(experiment_config.video_mae_task_config.vit_config)
            if section == "VideoMAETaskConfig":
                actual_config_dict = asdict(experiment_config.video_mae_task_config)
            if section == "ECoGDataConfig":
                actual_config_dict = asdict(experiment_config.ecog_data_config)
            if section == "TrainerConfig":
                actual_config_dict = asdict(experiment_config.trainer_config)
            if section == "LoggingConfig":
                actual_config_dict = asdict(experiment_config.logging_config)
                
            assert actual_config_dict[field] == value
            

def test_command_line_args_overwrite_config(mocker, fake_config_path):
    command_line_args = {
        "dim": 130,
        "mlp_dim": 130,
        "patch_size": 2,
        "patch_dims": [1, 2, 1],
        "frame_patch_size": 5,
        "use_cls_token": True,
        "tube_mask_ratio": 0.73,
        "decoder_mask_ratio": 0.02,
        "use_contrastive_loss": True,
        "running_cell_masking": True,
        "norm": "minute",
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
        "learning_rate": 0.001,
        "num_epochs": 11,
        "loss": "segment",
        "event_log_dir": "new_dir/",
        "print_freq": 100,
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
        
            
        

    mocker.patch(
        "sys.argv",
        [
            "main.py",
            "--config-file",
            fake_config_path,
        ] + cli_argv
    )
    
    args = arg_parser()
    experiment_config = create_video_mae_experiment_config(args)
    
    config_file = configparser.ConfigParser()
    config_file.read("tests/testdata/video_mae_config_file.ini")
    # Make sure all experiment config values match config file
    for section in CONFIG_VALUES.keys():
        for field, value in CONFIG_VALUES[section].items():
            if section == "VideoMAETaskConfig.ViTConfig":
                actual_config_dict = asdict(experiment_config.video_mae_task_config.vit_config)
            if section == "VideoMAETaskConfig":
                actual_config_dict = asdict(experiment_config.video_mae_task_config)
            if section == "ECoGDataConfig":
                actual_config_dict = asdict(experiment_config.ecog_data_config)
            if section == "TrainerConfig":
                actual_config_dict = asdict(experiment_config.trainer_config)
            if section == "LoggingConfig":
                actual_config_dict = asdict(experiment_config.logging_config)
                
            assert actual_config_dict[field] == command_line_args[field], f"For field {field} expected: {command_line_args[field]} actual: {actual_config_dict[field]}"