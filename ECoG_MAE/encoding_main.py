# Load model,
# create dataloader
# iterate through dataloader and generate embeddings
# train linear layer

import numpy as np
import torch

from models import SimpleViT
from downstream_tasks.encoding.load_signal import EncodingDataset
from downstream_tasks.encoding.parser import arg_parser
from downstream_tasks.encoding.config import create_encoding_experiment_config
from downstream_tasks.encoding.utils import (
    pearson_correlation,
    run_regression,
    generate_embedding_dataset,
)


def main(args):
    # Setup config
    experiment_config = create_encoding_experiment_config(args)
    inference_device_name = experiment_config.encoding_task_config.embedding_device

    # Load model
    model = torch.load(
        experiment_config.encoding_task_config.model_path,
        map_location=torch.device(inference_device_name),
    )
    model.device = inference_device_name
    model.image_size = [1, 8, 8]
    model.patch_dims = [1, 1, 1]
    model.frame_patch_size = 4

    dataset = EncodingDataset(experiment_config.encoding_data_config)

    word_embeddings, neural_embeddings = generate_embedding_dataset(
        dataset,
        model,
        experiment_config.encoding_task_config.embedding_batch_size,
        inference_device_name,
    )

    predictions = run_regression(
        word_embeddings,
        neural_embeddings,
        experiment_config.encoding_task_config.num_folds,
    )

    rp, _, _ = pearson_correlation(neural_embeddings, predictions)
    mspe = np.square(neural_embeddings - predictions).mean()

    # TODO: Improve metrics used to measure performance of encoding task.
    print("Pearson correlations:", rp)
    print("MSPE:", mspe)


if __name__ == "__main__":
    args = arg_parser()
    main(args)
