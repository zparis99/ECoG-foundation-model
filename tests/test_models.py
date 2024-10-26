import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

import constants
from config import ECoGDataConfig
from mae_st_util.models_mae import MaskedAutoencoderViT
from pretrain_utils import model_forward

EMBEDDING_DIM = 64
FRAMES_PER_SAMPLE = 40
NUM_BANDS = 5
FRAME_PATCH_SIZE = 4


@pytest.fixture
def model():
    return MaskedAutoencoderViT(
        img_size=constants.GRID_SIZE,
        patch_size=1,
        in_chans=NUM_BANDS,
        norm_pix_loss=False,
        num_frames=FRAMES_PER_SAMPLE,
        t_patch_size=FRAME_PATCH_SIZE,
        cls_embed=False,
        pred_t_dim=FRAMES_PER_SAMPLE // FRAME_PATCH_SIZE,
        embed_dim=EMBEDDING_DIM,
        depth=2,
        num_heads=2,
        decoder_embed_dim=32,
        decoder_depth=1,
        decoder_num_heads=1,
        mlp_ratio=2.0,
    )


def test_model_forward_without_mask_succeeds(model):
    fake_batch = torch.randn(
        16, NUM_BANDS, FRAMES_PER_SAMPLE, constants.GRID_SIZE, constants.GRID_SIZE
    )
    loss, pred, mask, latent = model_forward(model, fake_batch, mask_ratio=0.8)

    num_patches = (
        FRAMES_PER_SAMPLE
        * constants.GRID_SIZE
        * constants.GRID_SIZE
        // FRAME_PATCH_SIZE
    )
    assert loss.detach().numpy().shape == ()
    assert not torch.isnan(loss)
    assert pred.detach().numpy().shape == (16, num_patches, NUM_BANDS)
    assert mask.detach().numpy().shape == (16, num_patches)
    assert latent.detach().numpy().shape == (
        16,
        int(num_patches * (1 - 0.8)),
        EMBEDDING_DIM,
    )


def test_model_forward_with_mask_succeeds(model):
    fake_batch = torch.randn(
        16, NUM_BANDS, FRAMES_PER_SAMPLE, constants.GRID_SIZE, constants.GRID_SIZE
    )
    
    fake_batch[:, :, :, 0, 0] *= torch.nan
    fake_batch[:, :, :, 0, 1] *= torch.nan

    mask = torch.ones(constants.GRID_SIZE, constants.GRID_SIZE, dtype=torch.bool)
    mask[0][0] = False
    mask[0][1] = False

    model.initialize_mask(mask)
    loss, pred, mask, latent = model_forward(model, fake_batch, mask_ratio=0.8)

    num_patches = (
        FRAMES_PER_SAMPLE
        * constants.GRID_SIZE
        * constants.GRID_SIZE
        // FRAME_PATCH_SIZE
    )
    num_patches_excluding_padding = (
        num_patches - 2 * FRAMES_PER_SAMPLE // FRAME_PATCH_SIZE
    )
    assert loss.detach().numpy().shape == ()
    assert not torch.isnan(loss)
    assert pred.detach().numpy().shape == (16, num_patches, NUM_BANDS)
    assert mask.detach().numpy().shape == (16, num_patches)
    assert latent.detach().numpy().shape == (
        16,
        int(num_patches_excluding_padding * (1 - 0.8)),
        EMBEDDING_DIM,
    )


def test_model_with_data_loader_input_succeeds(model, data_loader_creation_fn):
    data_config = ECoGDataConfig(
        batch_size=16,
        bands=[[i + 1, i + 2] for i in range(NUM_BANDS)],
        new_fs=FRAMES_PER_SAMPLE / 2,
        sample_length=2,
    )
    # Two batches of data.
    fake_data = np.ones((65, int(32 * 2 * 512)))
    data_loader = DataLoader(
        data_loader_creation_fn(
            data_config, data=fake_data, file_sampling_frequency=512
        ),
        batch_size=16,
    )

    for data in data_loader:
        loss, pred, mask, latent = model_forward(model, data, mask_ratio=0.8)

        num_patches = (
            FRAMES_PER_SAMPLE
            * constants.GRID_SIZE
            * constants.GRID_SIZE
            // FRAME_PATCH_SIZE
        )
        assert loss.detach().numpy().shape == ()
        assert not torch.isnan(loss)
        assert pred.detach().numpy().shape == (16, num_patches, NUM_BANDS)
        assert mask.detach().numpy().shape == (16, num_patches)
        assert latent.detach().numpy().shape == (
            16,
            int(num_patches * (1 - 0.8)),
            EMBEDDING_DIM,
        )
