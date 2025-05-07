import numpy as np
import pytest
import torch

import constants
from mae_st_util.models_mae import MaskedAutoencoderViT

EMBEDDING_DIM = 64
FRAMES_PER_SAMPLE = 40
NUM_BANDS = 5
FRAME_PATCH_SIZE = 4


def model_forward(model, signal, mask_ratio, alpha):
    """Pass signal through model after converting nan's to 0."""
    signal = torch.nan_to_num(signal)
    return model(signal, mask_ratio=mask_ratio, alpha=alpha)


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
    loss, mse, pred, mask, latent, correlations = model_forward(
        model, fake_batch, mask_ratio=0.8, alpha=0.5
    )

    num_patches = (
        FRAMES_PER_SAMPLE
        * constants.GRID_SIZE
        * constants.GRID_SIZE
        // FRAME_PATCH_SIZE
    )
    assert loss.detach().numpy().shape == ()
    assert mse.detach().numpy().shape == ()
    assert not torch.isnan(loss)
    assert pred.detach().numpy().shape == (16, num_patches, NUM_BANDS)
    assert mask.detach().numpy().shape == (16, num_patches)
    assert latent.detach().numpy().shape == (
        16,
        int(num_patches * (1 - 0.8)),
        EMBEDDING_DIM,
    )
    assert correlations.detach().numpy().shape == ()

    # Check that loss is set as expected
    assert torch.isclose((1 - correlations) / 2 * 0.5 + mse * 0.5, loss)


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
    loss, mse, pred, mask, latent, correlations = model_forward(
        model, fake_batch, mask_ratio=0.8, alpha=0.75
    )

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
    assert mse.detach().numpy().shape == ()
    assert not torch.isnan(loss)
    assert pred.detach().numpy().shape == (16, num_patches, NUM_BANDS)
    assert mask.detach().numpy().shape == (16, num_patches)
    assert latent.detach().numpy().shape == (
        16,
        int(num_patches_excluding_padding * (1 - 0.8)),
        EMBEDDING_DIM,
    )
    assert correlations.detach().numpy().shape == ()

    # Check that loss is set as expected
    assert torch.isclose((1 - correlations) / 2 * 0.75 + mse * 0.25, loss)
