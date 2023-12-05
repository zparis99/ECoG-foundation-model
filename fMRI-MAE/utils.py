import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from io import BytesIO
from nilearn import plotting
from einops import rearrange
import matplotlib.pyplot as plt
import torchio as tio

def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')

def grayscale_decoder(image_data):
    return np.array(Image.open(BytesIO(image_data))).astype(np.float32) / 65535

def numpy_decoder(npy_data):
    return np.load(BytesIO(npy_data))

def reshape_to_2d(tensor):
    if tensor.ndim == 5:
        tensor = tensor[0]
    assert tensor.ndim == 4
    return rearrange(tensor, 'b h w c -> (b h) (c w)')

def reshape_to_original(tensor_2d, h=64, w=64, c=48):
    # print(tensor_2d.shape) # torch.Size([1, 256, 3072])
    return rearrange(tensor_2d, '(tr h) (c w) -> tr h w c', h=h, w=w, c=c)


def plot_numpy_nii(image):
    while image.ndim > 3:
        image = image[0]
    nii = nib.Nifti1Image(image.astype(np.float32), np.eye(4))
    plotting.plot_epi(nii,cmap='gray')
    
    
class DataPrepper:
    def __init__(self):
        pass
    def __call__(self, sample):
        func, minmax, meansd = sample
        min_, max_, min_meansd, max_meansd = minmax
        reshaped_func = reshape_to_original(func)
        
        if len(reshaped_func) == 4:
            timepoints = np.arange(4)
        else:
            start_timepoint = np.random.choice(np.arange(len(reshaped_func)-4))
            timepoints = np.arange(start_timepoint,start_timepoint+4)
        
        func = torch.Tensor(reshaped_func[timepoints])
        meansd = torch.Tensor(reshape_to_original(meansd))
        return func, meansd
    
    
def plot_slices(unpatches): 
    if unpatches.ndim == 5:
        unpatches = unpatches[0]
    return transforms.ToPILImage()(reshape_to_2d(unpatches))

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('param counts:\n{:,} total\n{:,} trainable'.format(total, trainable))
    return trainable