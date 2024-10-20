from typing import Dict, Any, Union, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
from warframe_marketplace_predictor.filepaths import *
from warframe_marketplace_predictor.shtuff.data_handler import DataHandler
from warframe_marketplace_predictor.training.preprocessors.liquidity_model_preprocessor import Preprocessor


class PricePredictor:
    def __init__(self, model_predict_batch_size: int = 256):
        """
        Initializes the PricePredictor class, loading the preprocessor, model,
        attribute name shortcuts, and item name to URL mapping only once.
        """
        self.model_predict_batch_size = model_predict_batch_size
        self.data_handler = DataHandler()

        # Load the preprocessor and model once
        self.preprocessor = Preprocessor().load(liquidity_model_preprocessor_file_path)
        self.model: tf.keras.Model = tf.keras.models.load_model(liquidity_model_model_file_path)
        self._mask_token = "<NONE>"

        self.price_range_ = np.linspace(0, 10_000, 1, endpoint=True, dtype=np.float32)

    def is_valid(self, item: Dict[str, Any]) -> bool:
        """
        Checks if the provided attribute names (positives + negatives) are valid by comparing them against the
        shortcuts.

        Args:
            item (Dict[str, Any]): A dictionary containing the item's "positives" and "negatives" attributes.

        Returns:
            bool: True if all attribute names are valid, False otherwise.
        """
        if not self.data_handler.weapon_exists(item["name"]):
            print(f"{item['name']} is not a valid weapon name")
            print("Name suggestions:")
            print([k for k in sorted(self.data_handler.get_item_names())
                   if k and item["name"] and (k[0]).lower() == (item["name"][0]).lower()])
            return False

        if "re_rolls" in item:
            if not isinstance(item["re_rolls"], int):
                print("'re_rolls' must be an integer.")
                return False
            item["re_rolled"] = item["re_rolls"] > 0

        if "re_rolled" not in item or not isinstance(item["re_rolled"], bool):
            print("'re_rolled' is missing or incorrectly formatted.")
            return False

        # Combine the positives and negatives from the item to validate
        attribute_names = item["positives"] + item["negatives"]
        for attribute_name in attribute_names:
            if not self.data_handler.is_valid_attribute_shortcut(attribute_name):
                print(f"{attribute_name} is not a valid attribute.")
                print("Did you mean:")
                print([k for k in sorted(self.data_handler.get_attribute_shortcuts())
                       if k and attribute_name and k[0] == attribute_name[0]])
                return False

        return True

    def prepare(self, item: Dict[str, Any]) -> Dict[str, Any]:
        res = {
            "weapon_url_name": self.data_handler.get_url_name(item["name"]),
            "positive1": item["positives"][0] if len(item["positives"]) >= 1 else self._mask_token,
            "positive2": item["positives"][1] if len(item["positives"]) >= 2 else self._mask_token,
            "positive3": item["positives"][2] if len(item["positives"]) >= 3 else self._mask_token,
            "negative": item["negatives"][0] if len(item["negatives"]) >= 1 else self._mask_token,
            "re_rolled": item["re_rolled"],
            # "group": self.data_handler.get_weapon_group(self.data_handler.get_url_name(item["name"])),
            # "has_incarnon": self.data_handler.weapon_has_incarnon(self.data_handler.get_url_name(item["name"])),
            # "avg_trade_price": self.data_handler.get_average_trade_price(
            #     self.data_handler.get_url_name(item["name"]), rolled_status="rerolled"),
            # "disposition": self.data_handler.get_disposition(self.data_handler.get_url_name(item["name"])),
        }
        return res

    def get_prepared_data(self, data: Iterable[Dict[str, Any]], skip_validation: bool, verbose: bool) -> pd.DataFrame:
        prepared_data = []

        iterator = tqdm(data, desc="Preparing data", unit="riven") if verbose else data
        for item in iterator:

            # Check for invalid names or shortcuts
            if not skip_validation and not self.is_valid(item):
                return pd.DataFrame()

            row = self.prepare(item)

            if not skip_validation:
                row["positive1"] = self.data_handler.get_proper_attribute_name(row["positive1"])
                row["positive2"] = self.data_handler.get_proper_attribute_name(row["positive2"])
                row["positive3"] = self.data_handler.get_proper_attribute_name(row["positive3"])
                row["negative"] = self.data_handler.get_proper_attribute_name(row["negative"])

            for val in self.price_range_:
                row_c = row.copy()
                row_c["listing_price"] = val
                prepared_data.append(row_c)

        prepared_data = pd.DataFrame(prepared_data)
        return prepared_data

    def predict(self,
                data: Union[Iterable[Dict[str, Any]], Dict[str, Any], pd.DataFrame],
                price_range: Iterable[float] = None,
                verbose: bool = True,
                skip_validation: bool = False, raw: bool = False) -> np.ndarray:
        """
        Predicts outcomes based on the provided input data using a pre-trained model.
        """

        if isinstance(data, dict):
            data = [data]

        if price_range is not None:
            self.price_range_ = np.array(price_range, dtype=np.float32)

        if not raw:
            data = self.get_prepared_data(data, skip_validation, verbose)

        model_ready_data = self.preprocessor.transform(data)
        predictions = self.model.predict(model_ready_data,
                                         batch_size=self.model_predict_batch_size,
                                         verbose=verbose)

        if raw:
            return predictions
        else:
            return predictions.reshape(-1, len(self.price_range_))


def main():
    # Example data
    rivens = [
        {
            "name": "Latron",
            "positives": ["ms", "sc", "cd"],
            "negatives": ["zoom"],
            "re_rolled": True
        },
        # {
        #     "name": "Praedos",
        #     "positives": ["cc", "dmg", "slash"],
        #     "negatives": [""],
        #     "re_rolled": True
        # },
        # {
        #     "name": "Zenith",
        #     "positives": ["ms", "cd", "cc"],
        #     "negatives": [""],
        #     "re_rolled": True
        # },
        # {
        #     "name": "Cyngas",
        #     "positives": ["corpus", "zoom", ""],
        #     "negatives": ["cc"],
        #     "re_rolled": True
        # },
        # {
        #     "name": "Sydon",
        #     "positives": ["cc", "cd", "speed"],
        #     "negatives": ["punc"],
        #     "re_rolled": True
        # },
        # {
        #     "name": "Acceltra",
        #     "positives": ["speed", "dmg", "cc"],
        #     "negatives": [""],
        #     "re_rolled": True
        # },
    ]

    predictor = PricePredictor()
    predictions = predictor.predict(rivens, price_range=list(range(0, 10_001, 1)))
    predictions = gaussian_filter1d(predictions, sigma=1)

    plt.figure(figsize=(16, 8))

    # Generate colors using colormap for better distinction
    colors = plt.cm.tab10(np.linspace(0, 1, len(rivens), endpoint=True))

    lines = []

    # Plot the predictions with improved labels and line styles
    for p, r, color in zip(predictions, rivens, colors):
        label = f"{r['name']} | Positives: {', '.join(filter(None, r['positives']))}"
        if r['negatives'][0] != "":
            label += f" | Negative: {r['negatives'][0]}"
        line, = plt.plot(predictor.price_range_, p, label=label, linewidth=2.5, color=color)
        lines.append(line)

    # Format y-axis as percentages and adjust ticks
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y * 100:.0f}%'))
    plt.yticks(np.arange(0, 1.1, 0.1), fontweight="bold")
    plt.ylim([-0.05, 1.05])

    # Set axis labels with increased font size and weight
    plt.xlabel("Platinum Price", fontsize=14, weight='bold')
    plt.ylabel("Confidence (%)", fontsize=14, weight='bold')

    # Define function to find the index where confidence drops below 50%
    def find_rightmost_index(arr, threshold=0.5):
        indices = np.where(arr >= threshold)[0]
        return indices[-1] if len(indices) > 0 else 0

    # Adjust x-axis limits based on data
    x_limit = max(predictor.price_range_[find_rightmost_index(p)] for p in predictions)
    x_limit_padding = 0.2 * x_limit
    plt.xlim([-x_limit_padding / 3, x_limit + x_limit_padding])
    plt.xticks(np.arange(0, x_limit + x_limit_padding, 100), fontweight="bold")

    # Add horizontal line at 50% confidence and shaded regions
    plt.axhline(0.5, color="black", linestyle="--", linewidth=1)
    plt.fill_between([-20_000, 20_000], 0.5, 1.05, color="green", alpha=0.1)
    plt.fill_between([-20_000, 20_000], -0.05, 0.5, color="red", alpha=0.1)

    # Enhance grid and legend
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=16, loc='upper right', fancybox=True, framealpha=1, edgecolor='black', facecolor='white')

    # Set title with increased font size and weight
    plt.title("Model Confidence in Possible Sale Prices (Item Liquidity Proof of Concept)", fontsize=20, weight='bold')
    plt.tight_layout()

    # Offsets for annotations to avoid overlap
    offsets = [(-50, -0.05), (50, 0.05), (-50, 0.05), (50, -0.05)]

    for i, (p, line) in enumerate(zip(predictions, lines)):
        index_of_interest = find_rightmost_index(p)

        # Draw vertical line where confidence dips below 50%
        plt.axvline(predictor.price_range_[index_of_interest], 0, p[index_of_interest], color="gray", linestyle="--",
                    linewidth=1)

        x_value = predictor.price_range_[index_of_interest]
        y_value = 0
        plt.annotate(
            f"50% at {x_value:.0f}p",
            xy=(x_value, y_value),
            xytext=(x_value - 30, y_value),
            fontsize=11,
            fontweight='bold',
            color="white",
            bbox=dict(boxstyle="round,pad=0.3", edgecolor='white', facecolor='black', alpha=0.7)
        )

        # Calculate expected price and corresponding confidence
        above_50s_indices = np.where(p >= np.float32(0.5))[0]
        filtered_prices = predictor.price_range_[above_50s_indices]
        filtered_probs = p[above_50s_indices]

        expected_price = np.dot(filtered_prices, filtered_probs) / np.sum(filtered_probs)
        confidence_at_expected_price = np.interp(expected_price, predictor.price_range_, p)

        # Plot marker at expected price
        plt.scatter(expected_price, confidence_at_expected_price, color=line.get_color(), marker='o', s=100, zorder=5)

        # Annotate expected price with adjusted positions
        offset = offsets[i % len(offsets)]
        # print(rivens[i]["name"], expected_price)
        plt.annotate(
            f"Expected Price: {expected_price:.0f}p",
            xy=(expected_price, confidence_at_expected_price),
            xytext=(expected_price + offset[0], confidence_at_expected_price + offset[1]),
            arrowprops=dict(arrowstyle="->", color='gray'),
            fontsize=11,
            fontweight='bold',
            color="white",
            bbox=dict(boxstyle="round,pad=0.3", edgecolor='white', facecolor='black', alpha=0.7)
        )

        b = sum(p[:index_of_interest]) / len(p[:index_of_interest]) * expected_price
        c = np.interp(b, predictor.price_range_[:index_of_interest], p[:index_of_interest])
        plt.annotate(
            f"Weighted Price: {b:.0f}p",
            xy=(np.average(p[:index_of_interest]) * expected_price, c),
            xytext=(np.average(p[:index_of_interest]) * expected_price + offset[0], c + offset[1]),
            arrowprops=dict(arrowstyle="->", color='gray'),
            fontsize=11,
            fontweight='bold',
            color="white",
            bbox=dict(boxstyle="round,pad=0.3", edgecolor='white', facecolor='black', alpha=0.7)
        )

    # plt.savefig(fname="img.png")
    plt.show()


if __name__ == "__main__":
    main()
