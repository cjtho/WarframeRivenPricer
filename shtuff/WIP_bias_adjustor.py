"""
IN-PROGRESS

The method by which we adjust bias needs to both be efficient and have some
statistical science as to why it is a valid adjustor.
"""
import collections
from typing import Dict, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tqdm


def diagnose_transformation(original_values: np.ndarray,
                            transformed_values: np.ndarray,
                            target_summary_statistics: Dict[str, Union[float, int]]):
    """
    Visualizes the original and transformed distributions and prints summary statistics.

    Parameters:
    - original_values: The original predicted listing prices.
    - transformed_values: The transformed values after optimization.
    - summary_statistics: The target summary statistics for the traded prices.
    """
    # Compute original and new summary statistics
    original_stats = calculate_summary_statistics(original_values)
    new_stats = calculate_summary_statistics(transformed_values)

    # Print the summary statistics
    print("Original Summary Statistics:")
    for key, value in original_stats.items():
        print(f"  {key}: {value:.4f}")

    print("\nTarget Summary Statistics (Given):")
    for key, value in target_summary_statistics.items():
        print(f"  {key}: {value:.4f}")

    print("\nNew Summary Statistics (After Transformation):")
    for key, value in new_stats.items():
        print(f"  {key}: {value:.4f}")

    # Create a plot to visualize the distributions
    plt.figure(figsize=(10, 6))

    sns.kdeplot(original_values, color="blue", label="Original Distribution", linewidth=2, bw_adjust=0.1)
    sns.kdeplot(transformed_values, color="red", label="Transformed Distribution", linewidth=2, bw_adjust=0.1)

    plt.title("Comparison of Original and Transformed Distributions")
    plt.xlabel("Values")
    plt.ylabel("Density")
    plt.legend()

    plt.show()


def calculate_summary_statistics(values: np.ndarray) -> Dict[str, float]:
    return {
        "avg": values.mean(),
        "stddev": values.std(),
        "median": np.median(values),
        "min": values.min(),
        "max": values.max()
    }


def min_max_scale(values: np.ndarray, min_: float = None, max_: float = None) -> np.ndarray:
    values_min = values.min()
    values_max = values.max()
    values_std = (values - values_min) / (values_max - values_min)
    values_scaled = values_std * (max_ - min_) + min_
    return values_scaled


def gaussian_kernel(size: int, sigma: float = 1) -> np.ndarray:
    """Returns a 1D Gaussian kernel of length 'size' with standard deviation 'sigma'."""
    if size % 2 == 0:
        raise ValueError("Length of the Gaussian kernel must be odd.")
    center = size // 2
    x = np.arange(0, size) - center
    kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel /= np.sum(kernel)  # Normalize the kernel so that the sum is 1
    return kernel


def calculate_loss(values: np.ndarray, target_summary_statistics: Dict[str, float]) -> float:
    summ_stats = calculate_summary_statistics(values)
    loss = 0
    for metric in ["avg", "stddev"]:
        loss += (summ_stats[metric] - target_summary_statistics[metric]) ** 2
    return loss


def adjust_values(values: pd.Series, target_summary_statistics: Dict[str, float]) -> np.ndarray:
    """Adjust values using gradient descent and smooth updates with a Gaussian kernel."""

    values = np.array(values)

    # Scale the initial values to match the target min/max range
    new_values = min_max_scale(values, target_summary_statistics["min"], target_summary_statistics["max"])

    # Create Gaussian kernel
    kernel = gaussian_kernel(11, sigma=1)

    # Hyperparameters
    max_iter = 1000
    learning_rate = 1

    pbar = tqdm.tqdm(range(max_iter))
    for _ in pbar:
        orig_loss = calculate_loss(new_values, target_summary_statistics)

        # For each value, compute its impact on the loss
        gradient = np.zeros_like(new_values)

        for i in range(len(new_values)):
            # Small step forward
            new_values[i] += learning_rate
            loss_increase = calculate_loss(new_values, target_summary_statistics)

            # Compute gradient (difference in loss)
            gradient[i] = (loss_increase - orig_loss) / learning_rate

            # Revert the change
            new_values[i] -= learning_rate

        # Update all values using the gradient and smooth it with Gaussian kernel
        smoothed_gradient = np.convolve(gradient, kernel, mode='same')
        new_values -= learning_rate * smoothed_gradient
        new_values = min_max_scale(new_values, target_summary_statistics["min"], target_summary_statistics["max"])

        # Check for convergence (optional stopping criteria)
        loss = np.abs(orig_loss - calculate_loss(new_values, target_summary_statistics))
        pbar.set_postfix(loss=orig_loss)
        if loss < 1e-6:
            break

    print("Final Loss:", calculate_loss(new_values, target_summary_statistics))
    d = collections.defaultdict(list)
    for v1, v2 in zip(values, new_values):
        d[v1].append(v2)
    for v1, v2l in d.items():
        print(v1, v2l)
    diagnose_transformation(values, new_values, target_summary_statistics)
    quit()

    return new_values
