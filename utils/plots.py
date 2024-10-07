import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def visualize_batch(batch_tensor, nrow=8, title='', figsize=None):
    # Calculate the number of rows needed based on the batch size and desired number of columns (nrow)
    batch_size = batch_tensor.size(0)
    nrows = math.ceil(batch_size / nrow)

    # Automatically adjust the figsize based on the number of images if not provided
    if figsize is None:
        figsize = (nrow, nrows)

    # Create a grid of images
    grid_img = make_grid(batch_tensor, nrow=nrow)

    # Plotting
    plt.figure(figsize=figsize)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.title(title)
    plt.axis('off')
    plt.show()


def save_image(batch_images, workdir, n=64, padding=2, pos="horizontal", nrow=3, w=5.5, file_format="jpeg", name="data_samples", scale=4, show=False):
    """
    Plot a grid of images for a given size.

    Args:
        batch_images (tensor): Tensor of size NxCxHxW.
        workdir (str): Path where the image grid will be saved.
        n (int): Number of images to display.
        padding (int): Padding size of the grid.
        pos (str): Position of the grid. Options: "horizontal", "square", or "vertical".
        nrow (int): Number of rows in the grid (only applicable when pos="vertical").
        w (float): Width size.
        file_format (str): File format for saving the image (e.g., "png").
        name (str): Name of the saved image file.
        scale (int): Scaling factor for the saved image.
        show (bool): Whether to display the image using plt.show().

    """
    if pos == "horizontal":
        sample_grid = make_grid(batch_images[:n], nrow=n, padding=padding)
    elif pos == "square":
        n = batch_images.shape[0] if batch_images.shape[0] < n else n
        sample_grid = make_grid(batch_images[:n], nrow=int(np.sqrt(n)), padding=padding)
    elif pos == "vertical":
        sample_grid = make_grid(batch_images[:n], nrow=nrow, padding=padding)
    else:
        raise ValueError("Invalid 'pos' value. Use 'horizontal', 'square', or 'vertical'.")

    fig = plt.figure(figsize=(n * w / scale, w))
    plt.axis('off')
    plt.imshow(sample_grid.permute(1, 2, 0).cpu())
    fig.savefig(os.path.join(workdir, "{}.{}".format(name, file_format)), bbox_inches='tight', pad_inches=0)
    if show:
        plt.show()
    plt.close(fig)


def rescaling(x):
    """Rescale data to [-1, 1]. Assumes data in [0,1]."""

    return x * 2. - 1.

def rescaling_inv(x):
    """Rescale data back from [-1, 1] to [0,1]."""
    return .5 * x + .5

def make_equidistant(x, y, num_bins):
    # Find the min and max of x to define the bin edges
    min_x, max_x = torch.min(x), torch.max(x)
    bin_edges = torch.linspace(min_x, max_x, steps=num_bins + 1)
    
    # Digitize x to find out which bin each value belongs to
    bin_indices = torch.bucketize(x, bin_edges, right=True) - 1
    bin_indices[bin_indices == num_bins] = num_bins - 1  # Adjust last index to fall within range
    
    # Create a tensor to store the new y-values
    new_x = torch.zeros(num_bins)
    new_y, new_y_std = torch.zeros(num_bins), torch.zeros(num_bins)
    
    # Aggregate y-values by averaging within each bin
    for i in range(num_bins):
        mask = (bin_indices == i)
        if mask.sum() > 0:
            new_x[i] = x[mask].mean()
            new_y[i] = y[mask].mean()
            new_y_std[i] = y[mask].std()
        else:
            # Handle empty bins if necessary
            new_x[i] = (bin_edges[i] + bin_edges[i + 1]) / 2
            new_y[i] = float('nan')  # or some other placeholder
            new_y_std[i] = float('nan')
    
    return new_x, new_y, new_y_std, bin_indices, bin_edges