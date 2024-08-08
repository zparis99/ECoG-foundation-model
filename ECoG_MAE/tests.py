# TODO implement precise error messages

import pandas as pd
import time as t
import torch
import os
from config import VideoMAEExperimentConfig


def test_loader(config: VideoMAEExperimentConfig, train_dl: torch.utils.data.DataLoader, test_dl: torch.utils.data.DataLoader):
    """
    Tests if dataloader works as intended

    Args:
        config: experiment config
        train_dl: dataloader object
        tets_dl dataloader object

    Returns:
    """

    # Test dataloader

    start = t.time()

    trains = []

    print("test train_dl")
    for train_i, signal in enumerate(train_dl):

        trains.append(
            {
                "train batch ": str(train_i),
                "num_samples": str(signal.shape),
            }
        )

    end = t.time()

    train_samples = pd.DataFrame(trains)

    print(
        "Dataloader tested with batch size "
        + str(config.ecog_data_config.batch_size)
        + ". Time elapsed: "
        + str(end - start)
    )

    tests = []

    print("\ntest test_dl")
    for test_i, signal in enumerate(test_dl):

        tests.append(
            {
                "train batch ": str(test_i),
                "num_samples": str(signal.shape),
            }
        )

    test_samples = pd.DataFrame(tests)

    dir = os.getcwd() + f"/results/test_loader/"
    if not os.path.exists(dir):
        os.makedirs(dir)

    train_samples.to_csv(
        dir + f"{config.job_name}_train_samples.csv",
        index=False,
    )

    test_samples.to_csv(
        dir + f"{config.job_name}_test_samples.csv",
        index=False,
    )


def test_model(args, device, model, num_patches):
    """
    Tests if model works as intended

    Args:
        args: input arguments
        device:
        model:
        num_patches: number of patches in which the input data is segmented

    Returns:
    """

    print("Testing model")

    num_encoder_patches = int(num_patches * (1 - args.tube_mask_ratio))
    num_decoder_patches = int(num_patches * (1 - args.decoder_mask_ratio))

    # test that the model works without error
    model = model.to(device).eval()
    encoder_mask = torch.zeros(num_patches).to(device).to(torch.bool)
    encoder_mask[:num_encoder_patches] = True
    decoder_mask = torch.zeros(num_patches).to(device).to(torch.bool)
    decoder_mask[-num_decoder_patches:] = True

    with torch.no_grad():
        print("\nencoder")
        encoder_out = model(
            torch.randn(2, len(args.bands), 40, 1, 8, 8).to(device),
            encoder_mask=encoder_mask,
            verbose=True,
        )

        print("\ndecoder")
        decoder_out = model(
            encoder_out,
            encoder_mask=encoder_mask,
            decoder_mask=decoder_mask,
            verbose=True,
        )

        if args.use_cls_token:
            enc_cls_token = encoder_out[:, :1, :]
            encoder_patches = encoder_out[:, 1:, :]
            dec_cls_token = decoder_out[:, :1, :]
            decoder_patches = decoder_out[:, 1:, :]
            print("enc_cls_token", enc_cls_token.shape)
            print("encoder_patches", encoder_patches.shape)
            print("dec_cls_token", dec_cls_token.shape)
            print("decoder_patches", decoder_patches.shape)
