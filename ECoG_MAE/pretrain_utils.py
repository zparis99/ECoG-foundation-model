import torch
from torch.utils.data import DataLoader
from mask import *
from utils import *
from metrics import *
from plot import *
from mae_st_util.models_mae import MaskedAutoencoderViT
from config import VideoMAEExperimentConfig

import mae_st_util.misc as misc
from mae_st_util.logging import master_print as print


def model_forward(model, signal, mask_ratio):
    """Pass signal through model after converting nan's to 0."""
    signal = torch.nan_to_num(signal)
    return model(signal, mask_ratio=mask_ratio)


def train_single_epoch(
    train_dl: DataLoader,
    epoch: int,
    accelerator,
    optimizer,
    lr_scheduler,
    device: str,
    model: MaskedAutoencoderViT,
    config: VideoMAEExperimentConfig,
    logger,
    log_writer=None,
):
    model.train()

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter(
        "cpu_mem", misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    metric_logger.add_meter(
        "cpu_mem_all", misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    metric_logger.add_meter(
        "gpu_mem", misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    header = "Epoch: [{}]".format(epoch)

    for train_i, batch in enumerate(
        metric_logger.log_every(train_dl, config.logging_config.print_freq, header)
    ):
        optimizer.zero_grad()

        signal = batch.to(device)

        padding_mask = get_padding_mask(signal, device)
        # TODO: We don't necessarily need to call this so often but for now this is easier.
        # We could be more clever with this though.
        model.initialize_mask(padding_mask)

        # TODO: Add more metrics using the other outputs.
        loss, _, _, _ = model_forward(
            model, signal, config.video_mae_task_config.encoder_mask_ratio
        )
        if torch.isnan(loss):
            logger.error(
                f"Got nan loss for index {train_i}. Ignoring and continuing..."
            )
            continue

        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()

        loss_value = loss.item()

        metric_logger.update(loss=loss_value)
        metric_logger.update(cpu_mem=misc.cpu_mem_usage()[0])
        metric_logger.update(cpu_mem_all=misc.cpu_mem_usage()[1])
        metric_logger.update(gpu_mem=misc.gpu_mem_usage())

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        if log_writer is not None:
            """We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((train_i / len(train_dl) + epoch) * 1000)
            log_writer.add_scalar("loss/train", loss_value, epoch_1000x)
            log_writer.add_scalar("lr", lr, epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def test_single_epoch(
    test_dl: DataLoader,
    epoch: int,
    device: str,
    model: MaskedAutoencoderViT,
    config: VideoMAEExperimentConfig,
    logger,
    log_writer=None,
):
    model.eval()
    with torch.no_grad():
        running_loss = 0.0
        for test_i, batch in enumerate(test_dl):
            signal = batch.to(device)

            padding_mask = get_padding_mask(signal, device)
            # TODO: We don't necessarily need to call this so often but for now this is easier.
            # We could be more clever with this though.
            model.initialize_mask(padding_mask)

            # TODO: Add more metrics using the other outputs.
            loss, _, _, _ = model_forward(
                model, signal, config.video_mae_task_config.encoder_mask_ratio
            )
            if torch.isnan(loss):
                logger.error(
                    f"Got nan loss for index {test_i}. Ignoring and continuing..."
                )
                continue

            running_loss += loss.item()

        loss_mean = running_loss / (test_i + 1)

        # Write averages for test data.
        if log_writer is not None:
            """We use epoch_1000x as the x-axis in tensorboard.
            This aligns with the training loop.
            """
            epoch_1000x = int((test_i / len(test_dl) + epoch) * 1000)
            log_writer.add_scalar("loss/test", loss_mean, epoch_1000x)
