import sys
from parser import arg_parser
from ecog_setup import system_setup, model_setup
from config import create_video_mae_experiment_config
from loader import dl_setup
from mae_st_util.logging import setup_logging
from tests import test_loader
from train import train_model


def main(args):

    setup_logging()

    experiment_config = create_video_mae_experiment_config(args)

    accelerator, device, data_type, local_rank = system_setup()
    train_dl, test_dl, num_train_samples = dl_setup(experiment_config)
    model, optimizer, lr_scheduler, _ = model_setup(
        experiment_config, device, num_train_samples
    )

    if args.test_loader:
        # TODO: Can migrate test loader to use new configs as well but for now
        test_loader(experiment_config, train_dl, test_dl)
        # test_model(args, device, model, num_patches)
        sys.exit("Stopping the script")

    model = train_model(
        experiment_config,
        device,
        model,
        train_dl,
        test_dl,
        optimizer,
        lr_scheduler,
        accelerator,
        data_type,
        local_rank,
    )


if __name__ == "__main__":

    args = arg_parser()
    main(args)
