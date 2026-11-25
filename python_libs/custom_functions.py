# custom functions

import torch
from tqdm import tqdm
import math
from typing import Sequence, Union


def last_n_average(metrics: Union[Sequence[float], list], n: int) -> float:
    """
    Calculate the average of the last `n` values in `metrics`.
    If there are fewer than `n` values, averages all available values.
    Returns `NaN` if `metrics` is empty.
    
    Useful during model training to monitor smoothed recent performance
    (loss, accuracy, etc.).
    """
    if not metrics:  # empty list/tuple/etc.
        return math.nan
    
    k = min(n, len(metrics))
    last_k = metrics[-k:]  # slicing handles the "last k" efficiently
    return sum(last_k) / k


def compute_class_frequencies(dataset, num_classes=19, ignore_index=255):
    """
    Compute raw class pixel counts across the entire dataset.
    Similar to your Julia `compute_class_frequencies`.
    """
    print("Computing class frequencies...")
    class_counts = torch.zeros(num_classes, dtype=torch.long, device='cpu')
    total_valid_pixels = 0

    for _, mask in tqdm(dataset, desc="Scanning dataset"):
        mask = mask.view(-1)                    # flatten [H*W]
        valid_mask = (mask != ignore_index)
        valid_pixels = mask[valid_mask]

        for c in range(num_classes):
            class_counts[c] += (valid_pixels == c).sum().item()

        total_valid_pixels += valid_mask.sum().item()

    print(f"Total valid pixels: {total_valid_pixels:,}")
    return class_counts.float(), total_valid_pixels


def compute_class_weights(class_counts, total_pixels=None, num_classes=19, eps=1e-6):
    """
    Compute multiple class weighting strategies (inspired by your Julia function).
    
    Returns a dictionary with several strategies.
    """
    if total_pixels is None:
        total_pixels = class_counts.sum().item()

    vf = class_counts.float() + eps
    frequencies = vf / total_pixels
    frequencies = torch.clamp(frequencies, min=eps, max=1.0)

    C = float(num_classes)

    # 1. Median Frequency Balancing
    median_freq = torch.median(frequencies)
    median_weights = median_freq / frequencies
    median_weights = torch.clamp(median_weights, min=0.1, max=20.0)

    # 2. Inverse Frequency
    inverse_weights = 1.0 / frequencies
    inverse_weights = torch.clamp(inverse_weights, min=eps)

    # 3. Inverse Squared Frequency (very effective for segmentation)
    inverse_squared = 1.0 / (frequencies ** 2)
    inverse_squared = inverse_squared / inverse_squared.mean()   # normalize mean ≈ 1
    inverse_squared = torch.clamp(inverse_squared, min=eps)

    # 4. Normalized Inverse (sum to C, common in practice)
    inverse_class_weights = inverse_weights * (C / inverse_weights.sum())
    inverse_class_weights = torch.clamp(inverse_class_weights, min=eps)

    print("\nClass Weights Summary:")
    print("Class | Freq(%) | Median |    Inv |   Inv² | NormInv")
    for i in range(num_classes):
        print(f"{i:2d}    |  {frequencies[i]*100:6.2f} | {median_weights[i]:6.3f} | "
              f"{inverse_weights[i]:6.2f} | {inverse_squared[i]:6.2f} | {inverse_class_weights[i]:6.3f}")

    return {
        'median': median_weights,
        'inverse': inverse_weights,
        'inverse_squared': inverse_squared,
        'normalized_inverse': inverse_class_weights,
        'frequencies': frequencies
    }


import os
import random
import numpy as np
import torch
# import tensorflow as tf

def set_determinism(seed=42):
    # 1. Built-in Python hash and random modules
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    
    # 2. NumPy library
    np.random.seed(seed)
    # Modern NumPy alternative: rng = np.random.default_rng(seed)
    
    # 3. PyTorch (CPU and CUDA)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Enforce deterministic algorithms (can slow down performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    # # 4. TensorFlow
    # tf.random.set_seed(seed)
    # # Enforce deterministic operations where supported
    # tf.config.experimental.enable_op_determinism()

# Call the function at the very entry point of your script
# set_determinism(42)
