import collections
import gc

import numpy as np
import tensorflow as tf
import tqdm
from sklearn.cluster import MiniBatchKMeans

from warframe_marketplace_predictor.filepaths import *
from warframe_marketplace_predictor.shtuff.data_handler import DataHandler
from warframe_marketplace_predictor.shtuff.make_prediction import PricePredictor
from warframe_marketplace_predictor.shtuff.riven_funcs import get_possible_rivens
from warframe_marketplace_predictor.shtuff.storage_handling import save_json

os.environ["LOKY_MAX_CPU_COUNT"] = "4"

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)


def compute_prices_frequency_distribution(riven_attribute_combo_types):
    """Calculate the normalized price distribution (prices_pdf) for the rivens."""

    attribute_combo_frequencies = collections.Counter(riven_attribute_combo_types)
    target_count = max(attribute_combo_frequencies.values())
    prices_frequencies = np.array([
        target_count / attribute_combo_frequencies[trait_combo_type]
        for trait_combo_type in riven_attribute_combo_types
    ])
    total_riven_amount = np.sum(prices_frequencies)
    prices_pdf = prices_frequencies / total_riven_amount
    return prices_pdf, prices_frequencies


def compute_attribute_importance(
        rivens, prices, price_frequencies, combo_types
):
    """
    Compute the impact of each trait on the price and visualize it.

    Parameters:
    - rivens: List of dictionaries, each with 'positives' and 'negatives' traits.
    - prices: List of prices corresponding to each riven.
    - price_frequencies: List indicating the frequency or weight of each price.
    - combo_types: List indicating the combo type for each riven.

    Returns:
    - attribute_importance: Dictionary with normalized importance scores for positives and negatives.
    """
    attribute_importance = {
        "positives": collections.defaultdict(float),
        "negatives": collections.defaultdict(float)
    }

    # Initialize aggregation dictionary for each combo type
    combo_trait_impacts = {
        combo_type: {
            "positives": collections.defaultdict(float),
            "negatives": collections.defaultdict(float)
        }
        for combo_type in ["p2n0", "p2n1", "p3n0", "p3n1"]
    }

    # Aggregate the sum of prices within each trait category per combo type
    for riven, price, frequency, combo_type in zip(rivens, prices, price_frequencies, combo_types):
        for trait_type in ["positives", "negatives"]:
            for trait in riven[trait_type]:
                combo_trait_impacts[combo_type][trait_type][trait] += price * frequency

    # Normalize category by the maximum
    for trait_impacts in combo_trait_impacts.values():
        for trait_type in ["positives", "negatives"]:
            traits = trait_impacts[trait_type]
            if traits:
                min_value = min(traits.values())
                max_value = max(traits.values())
                if max_value > min_value:  # Avoid division by zero
                    trait_impacts[trait_type] = {
                        trait: value / max_value
                        for trait, value in traits.items()
                    }
                else:
                    trait_impacts[trait_type] = {trait: 0 for trait in traits}  # All values are the same

    # Combine each category by an equal weighting
    num_combo_types = len(combo_trait_impacts)
    weight = 1.0 / num_combo_types if num_combo_types > 0 else 0
    for trait_impacts in combo_trait_impacts.values():
        for trait_type in ["positives", "negatives"]:
            for trait, normalized_value in trait_impacts[trait_type].items():
                attribute_importance[trait_type][trait] += normalized_value * weight

    # Final normalization of combined traits to the region (1, 0)
    for trait_type in ["positives", "negatives"]:
        if not attribute_importance[trait_type]:
            continue

        min_combined = min(attribute_importance[trait_type].values())
        max_combined = max(attribute_importance[trait_type].values())
        if max_combined > min_combined:  # Avoid division by zero
            attribute_importance[trait_type] = {
                trait: value / max_combined
                for trait, value in attribute_importance[trait_type].items()
            }
        else:
            attribute_importance[trait_type] = {trait: 0 for trait in
                                                attribute_importance[trait_type]}  # All values are the same

        attribute_importance[trait_type] = {trait: attribute_importance[trait_type][trait]
                                            for trait in sorted(attribute_importance[trait_type].keys(),
                                                                key=attribute_importance[trait_type].get,
                                                                reverse=True)}

    return attribute_importance


def compute_sparse_prices_pdf_kmeans(prices, prices_frequencies, num_bins):
    """
    Compute a sparse representation of the price distribution using K-Means clustering.

    Parameters:
    - prices (np.ndarray): Array of price values.
    - prices_frequencies (np.ndarray): Corresponding frequencies of each price.
    - num_bins (int): Desired number of bins.

    Returns:
    - sparse_prices_distribution (dict): Mapping of expected price per bin to total frequency.
    - bin_edges (np.ndarray): Edges of the bins.
    """
    # Reshape prices for clustering
    prices_reshaped = prices.reshape(-1, 1)

    # Initialize KMeans with the desired number of bins
    kmeans = MiniBatchKMeans(n_clusters=num_bins, batch_size=5120, random_state=42)

    # Fit KMeans with sample weights as frequencies
    kmeans.fit(prices_reshaped, sample_weight=prices_frequencies)

    # Get cluster centers and sort them
    cluster_centers = np.sort(kmeans.cluster_centers_.flatten())

    # Define bin edges as midpoints between cluster centers
    bin_edges = np.concatenate([
        [prices.min() - 1],  # Extend the first bin to include the minimum price
        (cluster_centers[:-1] + cluster_centers[1:]) / 2,
        [prices.max() + 1]  # Extend the last bin to include the maximum price
    ])

    # Assign each price to a bin
    bin_indices = np.digitize(prices, bin_edges) - 1  # bin_indices start from 0

    # Aggregate frequencies and compute expected price per bin
    sparse_prices_distribution = {}
    for bin_idx in range(num_bins):
        in_bin = bin_indices == bin_idx
        if not np.any(in_bin):
            continue
        total_frequency = np.sum(prices_frequencies[in_bin])
        expected_price = np.average(prices[in_bin], weights=prices_frequencies[in_bin])
        sparse_prices_distribution[expected_price] = total_frequency

    del kmeans

    return sparse_prices_distribution


def compute_sparse_prices_pdf(prices, prices_frequencies, granularity):
    """
    Compute sparse representations of the price distribution using percentile-based and K-Means-based binning.
    Plot both sparse distributions alongside the dense distribution for comparison.

    Parameters:
    - prices (np.ndarray): Array of price values.
    - prices_pdf (np.ndarray): Probability density function values for each price.
    - prices_frequencies (np.ndarray): Corresponding frequencies of each price.
    - granularity (int): Number of bins for percentile-based binning.
    """

    # Sort the prices and corresponding data
    sorted_indices = np.argsort(prices)
    prices_sorted = prices[sorted_indices]
    prices_frequencies_sorted = prices_frequencies[sorted_indices]

    sparse_prices_distribution_kmeans = compute_sparse_prices_pdf_kmeans(
        prices_sorted, prices_frequencies_sorted, num_bins=granularity)

    return sparse_prices_distribution_kmeans


def create_weapon_information():
    price_predictor = PricePredictor(model_predict_batch_size=4096)
    data_handler = DataHandler()
    weapon_names = data_handler.get_url_names()
    weapon_ranking_information = []
    all_prices = []
    all_prices_frequencies = []

    pbar = tqdm.tqdm(weapon_names, desc="Determining characteristics", unit="weapon category")
    for weapon_name in pbar:
        pbar.set_postfix(weapon=weapon_name)

        possible_rivens = get_possible_rivens(weapon_name, re_rolled=True)
        if len(possible_rivens) == 0:
            # Handle weapons with no possible rivens
            print(f"No rivens found for weapon: {weapon_name}. Skipping.")
            continue

        prices = price_predictor.predict(possible_rivens, verbose=False, skip_validation=True)

        # Get attribute combo types
        riven_attribute_combo_types = [
            f"p{len(riven['positives'])}n{len(riven['negatives'])}" for riven in possible_rivens
        ]

        # Calculate the price distribution using the weighted data
        prices_pdf, prices_frequencies = compute_prices_frequency_distribution(riven_attribute_combo_types)

        # Calculate the expected value (EV)
        expected_price = np.dot(prices_pdf, prices)

        # Compute attribute importance
        attribute_importance = compute_attribute_importance(possible_rivens, prices, prices_frequencies,
                                                            riven_attribute_combo_types)

        # Collect all prices and prices_pdf for global processing
        all_prices.extend(prices.tolist())
        all_prices_frequencies.extend(prices_frequencies.tolist())

        # Compute sparse prices_pdf for the weapon
        weapon_prices_pdf_sparse = compute_sparse_prices_pdf(prices, prices_frequencies, granularity=2000)

        weapon_ranking_information.append((weapon_name, expected_price,
                                           attribute_importance, weapon_prices_pdf_sparse))

        # Clear memory
        del possible_rivens, prices
        gc.collect()  # Godly

        # Clear the Keras/TensorFlow session
        tf.keras.backend.clear_session()

    # Sort the rankings by expected value
    weapon_ranking_information.sort(key=lambda x: x[1], reverse=True)
    weapon_ranking_information = {
        weapon_name: {
            "rank": i,
            "expected_value": expected_value,
            "attribute_importance": attribute_importance,
            "price_distribution": price_distribution
        }
        for i, (weapon_name, expected_value, attribute_importance, price_distribution) in
        enumerate(weapon_ranking_information, start=1)
    }

    # Save the results to JSON files
    save_json(weapon_ranking_information_file_path, weapon_ranking_information)

    # After processing all weapons, compute global price bins using sparse representation
    all_prices = np.array(all_prices)
    all_prices_frequencies = np.array(all_prices_frequencies)
    global_price_freq = compute_sparse_prices_pdf(all_prices, all_prices_frequencies, granularity=1000)

    # Save the results to JSON files
    save_json(global_price_freq_file_path, global_price_freq)

    print("Finished evaluating weapons.")


def main():
    create_weapon_information()  # ~20 min


if __name__ == "__main__":
    main()
