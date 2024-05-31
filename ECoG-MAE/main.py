import torch
import sys
from parser import arg_parser
from config import system_setup, model_setup
from loader import dl_setup
from models import *
from tests import test_loader, test_model
from train import train_model


def main(args):

    accelerator, device, data_type, local_rank = system_setup()
    train_dl, test_dl, num_train_samples = dl_setup(args)
    model, optimizer, lr_scheduler, num_patches = model_setup(
        args, device, num_train_samples
    )

    if args.test_loader:

        test_loader(args, train_dl, test_dl)
        # test_model(args, device, model, num_patches)
        sys.exit("Stopping the script")

    model = train_model(
        args,
        device,
        model,
        train_dl,
        test_dl,
        num_patches,
        optimizer,
        lr_scheduler,
        accelerator,
        data_type,
        local_rank,
    )


if __name__ == "__main__":

    args = arg_parser()
    main(args)
