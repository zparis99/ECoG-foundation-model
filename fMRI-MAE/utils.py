from io import BytesIO

import numpy as np
import torch
from einops import rearrange
from nilearn import plotting
from PIL import Image
from skimage import filters
from torchvision import transforms
import nibabel as nib


def is_interactive():
    import __main__ as main

    return not hasattr(main, "__file__")


def grayscale_decoder(image_data):
    return np.array(Image.open(BytesIO(image_data))).astype(np.float32) / 65535


def numpy_decoder(npy_data):
    return np.load(BytesIO(npy_data))


def reshape_to_2d(tensor):
    if tensor.ndim == 5:
        tensor = tensor[0]
    assert tensor.ndim == 4
    return rearrange(tensor, "b h w c -> (b h) (c w)")


def reshape_to_original(tensor_2d, h=64, w=64, c=48):
    # print(tensor_2d.shape) # torch.Size([1, 256, 3072])
    return rearrange(tensor_2d, "(tr h) (c w) -> tr h w c", h=h, w=w, c=c)


def plot_numpy_nii(image):
    while image.ndim > 3:
        image = image[0]
    nii = nib.Nifti1Image(image.astype(np.float32), np.eye(4))  # noqa
    plotting.plot_epi(nii, cmap="gray")


def threshold_based_masking(org_images):
    org_images[org_images == 1] = 0  # ignore the padding
    thresholds = filters.threshold_multiotsu(org_images.numpy(), classes=3)
    brain_segmentation = org_images > thresholds.min()
    return brain_segmentation


def get_brain_pos_patches(
    brain_segmentation,
    patch_depth=8,
    patch_height=8,
    patch_width=8,
    frame_patch_size=1,
    masking_strategy="conservative",
):
    reshaped_mask = reshape_to_original(brain_segmentation)
    frames, _, _, depth = reshaped_mask.shape
    if masking_strategy == "conservative":
        # plt.imshow(reshaped_mask.sum(axis=(0, -1))) # [64, 64]
        reshaped_mask = reshaped_mask.sum(axis=(0, -1), keepdim=True).repeat(
            frames, 1, 1, depth
        )  # [4, 64, 64, 48]

    patched_mask = rearrange(
        reshaped_mask,
        "(f pf) (d pd) (h ph) (w pw) -> f d h w (pd ph pw pf)",
        pd=patch_depth,
        ph=patch_height,
        pw=patch_width,
        pf=frame_patch_size,
    )
    return (patched_mask.sum(-1) > 0).int().flatten()


class DataPrepper:
    def __init__(
        self,
        masking_strategy="conservative",
        patch_depth=8,
        patch_height=8,
        patch_width=8,
        frame_patch_size=1,
    ):
        self.masking_strategy = masking_strategy
        self.patch_depth = 8
        self.patch_height = 8
        self.patch_width = 8
        self.frame_patch_size = 1

    def __call__(self, sample):
        func, minmax, meansd = sample
        min_, max_, min_meansd, max_meansd = minmax
        reshaped_func = reshape_to_original(func)

        if len(reshaped_func) == 4:
            timepoints = np.arange(4)
        else:
            start_timepoint = np.random.choice(np.arange(len(reshaped_func) - 4))
            timepoints = np.arange(start_timepoint, start_timepoint + 4)

        func = torch.Tensor(reshaped_func[timepoints])
        meansd = torch.Tensor(reshape_to_original(meansd))

        # Keep track of the empty patches
        mean, sd = meansd
        org_images = reshape_to_2d(func * mean + sd)
        brain_segmentation = threshold_based_masking(org_images)
        pos_patches = get_brain_pos_patches(
            brain_segmentation,
            patch_depth=self.patch_depth,
            patch_height=self.patch_height,
            patch_width=self.patch_width,
            frame_patch_size=self.frame_patch_size,
            masking_strategy=self.masking_strategy,
        )
        return func, meansd, pos_patches


def plot_slices(unpatches):
    if unpatches.ndim == 5:
        unpatches = unpatches[0]
    return transforms.ToPILImage()(reshape_to_2d(unpatches))


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("param counts:\n{:,} total\n{:,} trainable".format(total, trainable))
    return trainable


def contrastive_loss(
    cls_token1: torch.Tensor, cls_token2: torch.Tensor, temperature: torch.Tensor
):
    feat1 = cls_token1 / cls_token1.norm(dim=1, keepdim=True)
    feat2 = cls_token2 / cls_token2.norm(dim=1, keepdim=True)

    cosine_sim = feat1 @ feat2.T
    logit_scale = temperature.exp()  # log scale, learned during training
    feat1 = cosine_sim * logit_scale
    feat2 = feat1.T

    labels = torch.arange(feat1.shape[0]).to(feat1.device)
    loss = (
        torch.nn.functional.cross_entropy(feat1, labels)
        + torch.nn.functional.cross_entropy(feat2, labels)
    ) / 2
    return loss
