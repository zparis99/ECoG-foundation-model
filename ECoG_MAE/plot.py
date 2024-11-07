import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import math
import os
from torch.utils.tensorboard import SummaryWriter
from matplotlib.figure import Figure


def interpolate_signal(signal, target_length):
    """
    Interpolate a signal to a target length.
    
    Parameters:
    -----------
    signal : np.ndarray
        Signal to interpolate
    target_length : int
        Desired length after interpolation
        
    Returns:
    --------
    np.ndarray
        Interpolated signal of length target_length
    """
    original_steps = np.arange(len(signal))
    target_steps = np.linspace(0, len(signal) - 1, target_length)
    interpolator = interp1d(original_steps, signal, kind='cubic')
    return interpolator(target_steps)


def plot_multi_band_reconstruction(original_signal, reconstructed_signal, t_patch_size, 
                                 batch_idx=0, height_idx=0, width_idx=0, 
                                 epoch=0):
    """
    Plot original and reconstructed signals for all bands in a subplot grid.
    Returns the figure object instead of saving/showing.
    
    Parameters:
    -----------
    original_signal : np.ndarray
        Original signal of shape [batch_size, num_bands, time_steps, height, width]
    reconstructed_signal : np.ndarray
        Reconstructed signal of shape [batch_size, num_bands, time_steps/t_patch_size, height, width]
    t_patch_size : int
        Number of frames in each temporal patch
    batch_idx : int
        Index of the batch to plot
    height_idx : int
        Height position to plot
    width_idx : int
        Width position to plot
    epoch : int
        Current training epoch (for title)
        
    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure object
    """
    num_bands = original_signal.shape[1]
    
    # Calculate subplot grid dimensions
    num_cols = min(3, num_bands)
    num_rows = math.ceil(num_bands / num_cols)
    
    # Create figure with subplots
    fig = Figure(figsize=(6 * num_cols, 4 * num_rows))
    fig.suptitle(f'Multi-band Signal Reconstruction (Epoch {epoch})\n'
                 f'Electrode ({height_idx}, {width_idx})', fontsize=16, y=1.02)
    
    # Create subplots for each band
    for band_idx in range(num_bands):
        # Extract signals for this band
        original = original_signal[batch_idx, band_idx, :, height_idx, width_idx]
        reconstructed_downsampled = reconstructed_signal[batch_idx, band_idx, :, height_idx, width_idx]
        
        # Interpolate reconstructed signal
        reconstructed = interpolate_signal(reconstructed_downsampled, len(original))
        
        # Create subplot
        ax = fig.add_subplot(num_rows, num_cols, band_idx + 1)
        
        # Plot signals
        time_steps = np.arange(len(original))
        ax.plot(time_steps, original, label='Original', color='blue', alpha=0.7)
        ax.plot(time_steps, reconstructed, label='Reconstructed', color='red', alpha=0.7, linestyle='--')
        
        # Plot patch boundaries
        for i in range(0, len(original), t_patch_size):
            ax.axvline(x=i, color='gray', alpha=0.2, linestyle=':')
        
        # Calculate metrics
        mse = np.mean((original - reconstructed) ** 2)
        mae = np.mean(np.abs(original - reconstructed))
        corr = np.corrcoef(original, reconstructed)[0, 1]
        
        # Add metrics text
        metrics_text = (f'MSE: {mse:.4f}\n'
                       f'MAE: {mae:.4f}\n'
                       f'Corr: {corr:.4f}')
        ax.text(0.02, 0.98, metrics_text, 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=8)
        
        # Customize subplot
        ax.set_title(f'Band {band_idx}')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
        
        # Only show legend for first subplot
        if band_idx == 0:
            ax.legend()
    
    # Adjust layout
    fig.tight_layout()
    return fig


def save_reconstruction_plot(original_signal, reconstructed_signal, epoch, 
                           output_dir, log_writer=None, t_patch_size=4,
                           batch_idx=0, height_idx=0, width_idx=0, 
                           tag='signal_reconstruction'):
    """
    Generate, save, and optionally log to TensorBoard a multi-band signal reconstruction plot.
    
    Parameters:
    -----------
    original_signal : np.ndarray
        Original signal of shape [batch_size, num_bands, time_steps, height, width]
    reconstructed_signal : np.ndarray
        Reconstructed signal of shape [batch_size, num_bands, time_steps/t_patch_size, height, width]
    epoch : int
        Current epoch number
    output_dir : str
        Directory to save plot files
    writer : torch.utils.tensorboard.SummaryWriter, optional
        TensorBoard writer for logging plots
    t_patch_size : int
        Number of frames in each temporal patch
    batch_idx : int
        Index of the batch to visualize
    height_idx : int
        Height position to visualize
    width_idx : int
        Width position to visualize
    tag : str
        Tag for TensorBoard logging
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate the figure
    fig = plot_multi_band_reconstruction(
        original_signal, reconstructed_signal,
        t_patch_size=t_patch_size,
        batch_idx=batch_idx, 
        height_idx=height_idx, 
        width_idx=width_idx,
        epoch=epoch
    )
    
    # Save to file
    save_path = os.path.join(output_dir, f'reconstruction_epoch_{epoch:04d}.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Log to TensorBoard if writer is provided
    if log_writer is not None:
        log_writer.add_figure(tag, fig, global_step=epoch)
    
    plt.close(fig)
