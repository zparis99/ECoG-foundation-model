import torch
from torch.utils.data import DataLoader
from mask import *
from utils import *
from plot import save_reconstruction_plot
from mae_st_util.models_mae import MaskedAutoencoderViT
from config import VideoMAEExperimentConfig
import constants

import mae_st_util.misc as misc
from mae_st_util.logging import master_print as print


def model_forward(model, signal, mask_ratio):
    """Pass signal through model after converting nan's to 0."""
    signal = torch.nan_to_num(signal)
    return model(signal, mask_ratio=mask_ratio)


def write_correlation_metrics(
    log_writer, prefix_str, band_correlations, channel_correlations, epoch_1000x
):
    for i, band_correlation in enumerate(band_correlations):
        log_writer.add_scalar(
            f"bands/{prefix_str}_correlations_{i}", band_correlation, epoch_1000x
        )

    for i, channel_correlation in enumerate(channel_correlations):
        log_writer.add_scalar(
            f"channels/{prefix_str}_correlations_{i}", channel_correlation, epoch_1000x
        )


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
    metric_logger.add_meter(
        "correlation", misc.SmoothedValue(window_size=1, fmt="{value: 6f}")
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
        loss, _, _, _, correlations = model_forward(
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
        correlation_value = correlations.mean().item()
        band_correlations = correlations.view(
            len(config.ecog_data_config.bands), -1
        ).mean(dim=1)
        channel_correlations = correlations.mean(dim=0).flatten()

        metric_logger.update(loss=loss_value)
        metric_logger.update(cpu_mem=misc.cpu_mem_usage()[0])
        metric_logger.update(cpu_mem_all=misc.cpu_mem_usage()[1])
        metric_logger.update(gpu_mem=misc.gpu_mem_usage())
        metric_logger.update(correlation=correlation_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        if log_writer is not None:
            """We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((train_i / len(train_dl) + epoch) * 1000)
            log_writer.add_scalar("loss/train", loss_value, epoch_1000x)
            log_writer.add_scalar("lr", lr, epoch_1000x)

            # Write overall correlation, individual channels, and bands.
            log_writer.add_scalar("correlation/train", correlation_value, epoch_1000x)
            write_correlation_metrics(
                log_writer,
                "train",
                band_correlations,
                channel_correlations,
                epoch_1000x,
            )

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
        running_correlation = 0.0
        running_band_correlations = torch.zeros(len(config.ecog_data_config.bands))
        running_channel_correlations = torch.zeros(
            constants.GRID_SIZE * constants.GRID_SIZE
        )
        for test_i, batch in enumerate(test_dl):
            signal = batch.to(device)

            padding_mask = get_padding_mask(signal, device)
            # TODO: We don't necessarily need to call this so often but for now this is easier.
            # We could be more clever with this though.
            model.initialize_mask(padding_mask)

            # TODO: Add more metrics using the other outputs.
            loss, pred, _, _, correlations = model_forward(
                model, signal, config.video_mae_task_config.encoder_mask_ratio
            )
            if torch.isnan(loss):
                logger.error(
                    f"Got nan loss for index {test_i}. Ignoring and continuing..."
                )
                continue

            # Draw a plot of the first electrode in the first batch so we can see the reconstruction.
            if test_i == 0:
                signal_np = signal.detach().cpu().numpy()

                pred_signal = model.unpatchify(pred)
                pred_signal_np = pred_signal.detach().cpu().numpy()
                save_reconstruction_plot(
                    signal_np,
                    pred_signal_np,
                    epoch,
                    config.logging_config.plot_dir,
                    log_writer=log_writer,
                    t_patch_size=config.video_mae_task_config.vit_config.frame_patch_size,
                )

                # Save a reconstruction plot for scaled signal as well.
                save_reconstruction_plot(
                    signal_np,
                    pred_signal_np,
                    epoch,
                    config.logging_config.plot_dir,
                    log_writer=log_writer,
                    t_patch_size=config.video_mae_task_config.vit_config.frame_patch_size,
                    tag="signal_reconstruction_scaled",
                    scale_output=True,
                )

            running_loss += loss.item()
            correlations = correlations.detach().cpu()
            running_correlation += correlations.mean().item()
            running_band_correlations += correlations.view(
                len(config.ecog_data_config.bands), -1
            ).mean(dim=1)
            running_channel_correlations += correlations.mean(dim=0).flatten()

        loss_mean = running_loss / (test_i + 1)
        correlation_mean = running_correlation / (test_i + 1)
        band_correlation_mean = running_band_correlations / (test_i + 1)
        channel_correlation_mean = running_channel_correlations / (test_i + 1)

        # Write averages for test data.
        if log_writer is not None:
            """We use epoch_1000x as the x-axis in tensorboard.
            This aligns with the training loop.
            """
            epoch_1000x = int((test_i / len(test_dl) + epoch) * 1000)
            log_writer.add_scalar("loss/test", loss_mean, epoch_1000x)
            # Write overall correlation, individual channels, and bands.
            log_writer.add_scalar("correlation/test", correlation_mean, epoch_1000x)
            write_correlation_metrics(
                log_writer,
                "test",
                band_correlation_mean,
                channel_correlation_mean,
                epoch_1000x,
            )
