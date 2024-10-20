import pickle
import warnings
from typing import List

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers, Model

from warframe_marketplace_predictor.shtuff.data_handler import DataHandler

# Suppress specific FutureWarning related to .fillna downcasting
warnings.filterwarnings("ignore", message=".*Downcasting object dtype arrays.*")


class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Scalers for continuous variables
        # self.disposition_scaler = MinMaxScaler()
        # self.avg_trade_price_scaler = MinMaxScaler()
        # self.listing_scaler = MinMaxScaler()
        #
        # # Mean value for disposition (used for imputation)
        # self.disposition_most_common = None
        #
        # # Regression model for imputing avg_trade_price
        # self.avg_trade_price_model = None
        #
        # # Mapping of weapon_url_name to known avg_trade_price
        # self.known_avg_trade_prices = {}
        pass

    def fit(self, X: pd.DataFrame, y=None) -> 'Preprocessor':
        # Store mean value for disposition
        # self.disposition_most_common = X["disposition"].mode()[0]
        #
        # # Prepare data for regression model
        # known_avg_trade = X.dropna(subset=["avg_trade_price"])
        # self._store_known_avg_trade_prices(known_avg_trade)
        # model_data = self._prepare_regression_data(known_avg_trade)
        # self._train_regression_model(model_data)
        #
        # # Fit scalers
        # self._fit_scalers(X, known_avg_trade)

        return self

    def transform(self, X: pd.DataFrame) -> List[pd.DataFrame]:
        X_copy = X.copy()

        # # Convert boolean columns to integers
        # self._convert_boolean_columns(X_copy)
        #
        # # Handle disposition
        # self._transform_disposition(X_copy)
        #
        # # Transform listing
        # self._transform_listing(X_copy)
        #
        # # Impute and transform avg_trade_price
        # self._impute_and_transform_avg_trade_price(X_copy)

        X_copy["re_rolled"] = X_copy["re_rolled"].astype(np.float32)
        X_copy["listing_price"] = np.log1p(X_copy["listing_price"])

        # Fill remaining missing values with "<NONE>"
        X_copy = X_copy.fillna("<NONE>")

        return self.split_X(X_copy)

    # def _store_known_avg_trade_prices(self, known_avg_trade: pd.DataFrame):
    #     """Store known avg_trade_price values for each weapon_url_name."""
    #     self.known_avg_trade_prices = known_avg_trade.groupby("weapon_url_name")["avg_trade_price"].first().to_dict()
    #
    # def _prepare_regression_data(self, known_avg_trade: pd.DataFrame) -> pd.DataFrame:
    #     """Prepare the dataset for training the regression model."""
    #     # Aggregate listing features
    #     listing_agg = known_avg_trade.groupby("weapon_url_name").agg({
    #         "listing": ["mean", "median", "std", "min", "max"]
    #     })
    #     listing_agg.columns = ['listing_' + stat for stat in ['mean', 'median', 'std', 'min', 'max']]
    #
    #     # Target variable: avg_trade_price per weapon
    #     avg_trade_price_per_weapon = known_avg_trade.groupby("weapon_url_name")["avg_trade_price"].first()
    #
    #     # Merge features and target
    #     model_data = listing_agg.join(avg_trade_price_per_weapon)
    #
    #     return model_data
    #
    # def _train_regression_model(self, model_data: pd.DataFrame):
    #     """Train the regression model for imputing avg_trade_price."""
    #     X_model = model_data.drop(columns=["avg_trade_price"])
    #     y_model = model_data["avg_trade_price"]
    #
    #     self.avg_trade_price_model = LinearRegression()
    #     self.avg_trade_price_model.fit(X_model, y_model)
    #
    # def _fit_scalers(self, X: pd.DataFrame, known_avg_trade: pd.DataFrame):
    #     """Fit scalers for disposition, avg_trade_price, and listing."""
    #     # Disposition scaler
    #     X_disposition = X["disposition"].fillna(self.disposition_most_common).to_frame()
    #     self.disposition_scaler.fit(X_disposition)
    #
    #     # avg_trade_price scaler (after log transformation)
    #     avg_trade_price_log = np.log1p(known_avg_trade["avg_trade_price"]).to_frame()
    #     self.avg_trade_price_scaler.fit(avg_trade_price_log)
    #
    #     # listing scaler (after log transformation)
    #     listing_log = np.log1p(X["listing"]).to_frame()
    #     self.listing_scaler.fit(listing_log)
    #
    # def _convert_boolean_columns(self, X: pd.DataFrame):
    #     """Convert boolean columns to integers."""
    #     X["has_incarnon"] = X["has_incarnon"].astype(int)
    #     X["re_rolled"] = X["re_rolled"].astype(int)
    #
    # def _transform_disposition(self, X: pd.DataFrame):
    #     """Impute and scale disposition values."""
    #     X["disposition"] = X["disposition"].fillna(self.disposition_most_common)
    #     X["disposition"] = self.disposition_scaler.transform(X[["disposition"]])
    #
    # def _transform_listing(self, X: pd.DataFrame):
    #     """Log-transform and scale listing."""
    #     X["listing"] = np.log1p(X["listing"])
    #     X["listing"] = self.listing_scaler.transform(X[["listing"]])
    #
    # def _impute_and_transform_avg_trade_price(self, X: pd.DataFrame):
    #     """Impute missing avg_trade_price and apply transformations."""
    #     self._impute_avg_trade_price(X)
    #     X["avg_trade_price"] = np.log1p(X["avg_trade_price"])
    #     X["avg_trade_price"] = self.avg_trade_price_scaler.transform(X[["avg_trade_price"]])
    #
    # def _impute_avg_trade_price(self, X: pd.DataFrame):
    #     """Impute missing avg_trade_price values using the regression model."""
    #     # Map known avg_trade_price values
    #     X["avg_trade_price"] = X["weapon_url_name"].map(self.known_avg_trade_prices)
    #
    #     # Identify weapons with missing avg_trade_price
    #     missing_weapons = X[X["avg_trade_price"].isna()]["weapon_url_name"].unique()
    #
    #     for weapon in missing_weapons:
    #         weapon_mask = X["weapon_url_name"] == weapon
    #         weapon_data = X.loc[weapon_mask]
    #
    #         # Aggregate listing features for this weapon
    #         features_df = self._aggregate_listing_features(weapon_data)
    #
    #         # Predict avg_trade_price
    #         predicted_avg_trade_price = self.avg_trade_price_model.predict(features_df)[0]
    #
    #         # Assign predicted value
    #         X.loc[weapon_mask, "avg_trade_price"] = predicted_avg_trade_price
    #
    # def _aggregate_listing_features(self, weapon_data: pd.DataFrame) -> pd.DataFrame:
    #     """Aggregate listing features for a weapon."""
    #     listings = weapon_data["listing"].values
    #     features = {
    #         'listing_mean': listings.mean(),
    #         'listing_median': np.median(listings),
    #         'listing_std': listings.std(ddof=0),
    #         'listing_min': listings.min(),
    #         'listing_max': listings.max()
    #     }
    #     features_df = pd.DataFrame([features])
    #
    #     # Handle any missing values in features
    #     features_df = features_df.fillna(0)
    #
    #     return features_df

    @staticmethod
    def split_X(X: pd.DataFrame) -> List[pd.DataFrame]:
        """Split the DataFrame into components."""
        return [
            X[["weapon_url_name"]],
            # X[["group"]],
            # X[["has_incarnon"]],
            # X[["avg_trade_price"]],
            X[["re_rolled"]],
            # X[["disposition"]],
            X[["positive1", "positive2", "positive3", "negative"]],
            X[["listing_price"]]
        ]

    def save(self, filepath: str):
        """Save the preprocessor instance to a pickle file."""
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath: str) -> 'Preprocessor':
        """Load the preprocessor instance from a pickle file."""
        with open(filepath, "rb") as f:
            return pickle.load(f)


def get_model_architecture():
    weapon_name_embedding_size = 32
    attributes_embedding_size = 32

    # Load vocabularies
    data_handler = DataHandler()
    weapon_url_names = data_handler.get_url_names()
    # group_names = data_handler.get_groups()
    attributes = data_handler.get_attribute_names()

    # Inputs
    weapon_url_name_input = layers.Input(shape=(1,), dtype=tf.string, name="weapon_url_name_input")
    # group_input = layers.Input(shape=(1,), dtype=tf.string, name="group_input")
    # has_incarnon_input = layers.Input(shape=(1,), dtype=tf.float32, name="has_incarnon_input")
    # avg_trade_price_input = layers.Input(shape=(1,), dtype=tf.float32, name="avg_trade_price_input")
    re_rolled_input = layers.Input(shape=(1,), dtype=tf.float32, name="re_rolled_input")
    # disposition_input = layers.Input(shape=(1,), dtype=tf.float32, name="disposition_input")
    attributes_input = layers.Input(shape=(4,), dtype=tf.string, name="attributes_input")
    listing_input = layers.Input(shape=(1,), dtype=tf.float32, name="listing_input")

    # Lookups
    weapon_url_name_lookup = layers.StringLookup(
        vocabulary=weapon_url_names,
        mask_token="<NONE>",
        name="weapon_url_name_lookup"
    )
    # group_lookup = layers.StringLookup(
    #     vocabulary=group_names,
    #     mask_token="<NONE>",
    #     name="group_lookup"
    # )
    attributes_lookup = layers.StringLookup(
        vocabulary=attributes,
        mask_token="<NONE>",
        name="attributes_lookup"
    )

    weapon_url_name_indices = weapon_url_name_lookup(weapon_url_name_input)
    # group_indices = group_lookup(group_input)
    attributes_indices = attributes_lookup(attributes_input)

    # Embeddings
    weapon_url_name_embedding_layer = layers.Embedding(
        input_dim=len(weapon_url_names) + 1,
        output_dim=weapon_name_embedding_size,
        name="weapon_url_name_embedding"
    )
    attributes_embedding_layer = layers.Embedding(
        input_dim=len(attributes) + 1,
        output_dim=attributes_embedding_size,
        name="attributes_embedding"
    )

    weapon_url_name_embedding_output = layers.Flatten()(weapon_url_name_embedding_layer(weapon_url_name_indices))
    attributes_embedding_output = layers.Flatten()(attributes_embedding_layer(attributes_indices))

    # Combine Weapon and Attribute Embeddings
    combined_embedding = layers.Concatenate(name="combined_embedding")(
        [weapon_url_name_embedding_output, attributes_embedding_output, listing_input, ]
    )

    # -- Dense Layers for Final Prediction --
    x = combined_embedding
    x = layers.Dense(units=128, activation="relu")(x)
    x = layers.Dense(units=32, activation="relu")(x)

    output = layers.Dense(units=1, activation="sigmoid", name="output")(x)

    # Define the model with all inputs and the output
    model = Model(inputs=[
        weapon_url_name_input, re_rolled_input, attributes_input, listing_input
    ], outputs=output, name="riven_model")

    return model
